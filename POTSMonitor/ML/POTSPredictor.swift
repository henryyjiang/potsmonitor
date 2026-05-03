import Foundation
import CoreML
import CreateML
import Combine

@MainActor
class POTSPredictor: ObservableObject {
    
    @Published var modelLoaded = false
    @Published var isTraining = false
    @Published var lastTrainedDate: Date?
    @Published var lastPrediction: Double = 0
    @Published var trainingStatus: String = ""
    @Published var modelAccuracy: Double?
    
    private var model: MLModel?
    private let featureEngine = FeatureEngine()
    private let fm = FileManager.default
    
    private var modelURL: URL {
        fm.urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("POTSFlareupModel.mlmodelc")
    }
    private var metaURL: URL {
        fm.urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("model_meta.json")
    }
    
    init() { loadModel() }
    
    // MARK: - Load / Reset
    
    func loadModel() {
        guard fm.fileExists(atPath: modelURL.path) else { modelLoaded = false; return }
        do {
            let cfg = MLModelConfiguration(); cfg.computeUnits = .cpuOnly
            model = try MLModel(contentsOf: modelURL, configuration: cfg)
            modelLoaded = true
            loadMeta()
        } catch {
            print("[Predictor] Load failed: \(error)")
            modelLoaded = false
        }
    }
    
    func resetModel() {
        try? fm.removeItem(at: modelURL)
        try? fm.removeItem(at: metaURL)
        model = nil
        modelLoaded = false
        lastTrainedDate = nil
        modelAccuracy = nil
        lastPrediction = 0
        trainingStatus = "Model reset"
    }
    
    // MARK: - Real-time prediction
    
    func ingestAndPredict(hr: HRSample?, acc: AccSample?, temp: Double?, baselineHR: Double = 0, baselineRMSSD: Double = 0) {
        featureEngine.currentBaseline = baselineHR
        featureEngine.baselineRMSSD = baselineRMSSD
        if let hr = hr { featureEngine.ingestHR(hr) }
        if let acc = acc { featureEngine.ingestAcc(acc) }
        if let temp = temp { featureEngine.ingestTemp(temp) }
        
        guard modelLoaded, let model = model,
              let w = featureEngine.tryComputeWindow() else { return }
        
        do {
            let input = try featureDict(w)
            let pred = try model.prediction(from: input)
            if let prob = pred.featureValue(for: "labelProbability")?.dictionaryValue,
               let p = prob[1 as NSNumber]?.doubleValue {
                DispatchQueue.main.async { self.lastPrediction = p }
            } else if let lbl = pred.featureValue(for: "label")?.int64Value {
                DispatchQueue.main.async { self.lastPrediction = lbl == 1 ? 1.0 : 0.0 }
            }
        } catch {
            print("[Predictor] Predict error: \(error)")
        }
    }
    
    // MARK: - Train on last 30 days
    
    func trainOnRecentData(dataStore: DataStore) async {
        await MainActor.run { isTraining = true; trainingStatus = "Loading data..." }
        
        do {
            let cutoff = Calendar.current.date(byAdding: .day, value: -30, to: Date()) ?? Date()
            let (hrAll, accAll) = try loadCSVs(from: dataStore.dataDir)
            
            // Filter to last 30 days
            let hr = hrAll.filter { $0.0 >= cutoff }
            let acc = accAll.filter { $0.0 >= cutoff }
            let flareups = dataStore.detectedFlareups.filter { $0.start >= cutoff }
            
            guard !flareups.isEmpty else {
                await MainActor.run { trainingStatus = "No flareups in last 30 days"; isTraining = false }
                return
            }
            
            await MainActor.run { trainingStatus = "Computing features..." }
            
            let windows = FeatureEngine.generateTrainingData(
                hrSamples: hr, accSamples: acc, flareups: flareups
            )
            
            let pos = windows.filter { $0.label == 1 }.count
            let neg = windows.filter { $0.label == 0 }.count
            
            guard windows.count >= 20 else {
                await MainActor.run {
                    trainingStatus = "Need more data (\(windows.count) windows, need 20+)"
                    isTraining = false
                }
                return
            }
            
            await MainActor.run { trainingStatus = "Training (\(pos) flareup, \(neg) normal windows)..." }
            
            var dict: [String: MLDataValueConvertible] = [:]
            dict["meanHR"] = windows.map { $0.meanHR }
            dict["maxHR"] = windows.map { $0.maxHR }
            dict["minHR"] = windows.map { $0.minHR }
            dict["hrDelta"] = windows.map { $0.hrDelta }
            dict["rmssd"] = windows.map { $0.rmssd }
            dict["sdnn"] = windows.map { $0.sdnn }
            dict["meanRR"] = windows.map { $0.meanRR }
            dict["accMagMean"] = windows.map { $0.accMagnitudeMean }
            dict["accMagStd"] = windows.map { $0.accMagnitudeStd }
            dict["accVertDelta"] = windows.map { $0.accVerticalDelta }
            dict["hrRiseFromBaseline"] = windows.map { $0.hrRiseFromBaseline ?? 0.0 }
            dict["rmssdPctChange"] = windows.map { $0.rmssdPercentChange ?? 0.0 }
            dict["label"] = windows.map { $0.label ?? 0 }
            
            let table = try MLDataTable(dictionary: dict)
            let (train, test) = table.randomSplit(by: 0.8, seed: 42)
            
            let classifier = try MLBoostedTreeClassifier(
                trainingData: train, targetColumn: "label",
                parameters: MLBoostedTreeClassifier.ModelParameters(
                    maxDepth: 6, maxIterations: 100, minLossReduction: 0.0,
                    minChildWeight: 1.0, stepSize: 0.3
                )
            )
            
            let accuracy = 1.0 - classifier.evaluation(on: test).classificationError
            
            await MainActor.run { trainingStatus = "Saving (\(String(format: "%.1f%%", accuracy*100)))..." }
            
            let meta = MLModelMetadata(author: "POTS Monitor", shortDescription: "Flareup predictor", version: "1.0")
            try classifier.write(to: modelURL, metadata: meta)
            
            let m = ModelMeta(trainedAt: Date(), windows: windows.count, positive: pos, negative: neg, accuracy: accuracy)
            try JSONEncoder().encode(m).write(to: metaURL)
            
            loadModel()
            
            await MainActor.run {
                lastTrainedDate = Date(); modelAccuracy = accuracy
                trainingStatus = "Done — \(String(format: "%.1f%%", accuracy*100)) accuracy"
                isTraining = false
            }
        } catch {
            await MainActor.run { trainingStatus = "Failed: \(error.localizedDescription)"; isTraining = false }
        }
    }
    
    // MARK: - CSV Parsing
    
    private func loadCSVs(from dir: URL) throws -> ([(Date, Int, [Int])], [(Date, Int32, Int32, Int32)]) {
        let tsF = DateFormatter(); tsF.dateFormat = "yyyy-MM-dd'T'HH:mm:ss.SSS"; tsF.timeZone = .current
        var hr: [(Date, Int, [Int])] = []
        var acc: [(Date, Int32, Int32, Int32)] = []

        let files = try fm.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil)

        for f in files where f.lastPathComponent.hasPrefix("hr_") {
            guard let content = readCSVFile(f) else { continue }
            for line in content.split(separator: "\n").dropFirst() {
                let c = parseLine(String(line))
                guard c.count >= 3, let ts = tsF.date(from: c[0]), let h = Int(c[1]) else { continue }
                let rrs = c[2].trimmingCharacters(in: .init(charactersIn: "\"")).split(separator: ";").compactMap { Int($0) }
                hr.append((ts, h, rrs))
            }
        }
        for f in files where f.lastPathComponent.hasPrefix("acc_") {
            guard let content = readCSVFile(f) else { continue }
            for line in content.split(separator: "\n").dropFirst() {
                let c = parseLine(String(line))
                guard c.count >= 4, let ts = tsF.date(from: c[0]),
                      let x = Int32(c[1]), let y = Int32(c[2]), let z = Int32(c[3]) else { continue }
                acc.append((ts, x, y, z))
            }
        }
        hr.sort { $0.0 < $1.0 }; acc.sort { $0.0 < $1.0 }
        return (hr, acc)
    }

    private func readCSVFile(_ url: URL) -> String? {
        if url.pathExtension == "zlib" {
            guard let compressed = try? Data(contentsOf: url),
                  let decompressed = try? (compressed as NSData).decompressed(using: .zlib) as Data else { return nil }
            return String(data: decompressed, encoding: .utf8)
        }
        return try? String(contentsOf: url, encoding: .utf8)
    }
    
    private func parseLine(_ line: String) -> [String] {
        var result: [String] = []; var cur = ""; var inQ = false
        for ch in line {
            if ch == "\"" { inQ.toggle() }
            else if ch == "," && !inQ { result.append(cur); cur = "" }
            else { cur.append(ch) }
        }
        result.append(cur); return result
    }
    
    private func featureDict(_ w: FeatureWindow) throws -> MLDictionaryFeatureProvider {
        try MLDictionaryFeatureProvider(dictionary: [
            "meanHR": MLFeatureValue(double: w.meanHR),
            "maxHR": MLFeatureValue(double: w.maxHR),
            "minHR": MLFeatureValue(double: w.minHR),
            "hrDelta": MLFeatureValue(double: w.hrDelta),
            "rmssd": MLFeatureValue(double: w.rmssd),
            "sdnn": MLFeatureValue(double: w.sdnn),
            "meanRR": MLFeatureValue(double: w.meanRR),
            "accMagMean": MLFeatureValue(double: w.accMagnitudeMean),
            "accMagStd": MLFeatureValue(double: w.accMagnitudeStd),
            "accVertDelta": MLFeatureValue(double: w.accVerticalDelta),
            "hrRiseFromBaseline": MLFeatureValue(double: w.hrRiseFromBaseline ?? 0.0),
            "rmssdPctChange": MLFeatureValue(double: w.rmssdPercentChange ?? 0.0),
        ])
    }
    
    private struct ModelMeta: Codable {
        let trainedAt: Date; let windows: Int; let positive: Int; let negative: Int; let accuracy: Double
    }
    private func loadMeta() {
        guard let d = try? Data(contentsOf: metaURL),
              let m = try? JSONDecoder().decode(ModelMeta.self, from: d) else { return }
        lastTrainedDate = m.trainedAt; modelAccuracy = m.accuracy
    }
}
