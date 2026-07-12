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
    @Published var modelF1: Double?

    private var model: MLModel?
    private let featureEngine = FeatureEngine()
    private let fm = FileManager.default

    /// The 14 features the app supplies at inference, in order. Any model we
    /// load must declare exactly these inputs — see `validateFeatures`.
    static let featureNames = ["meanHR","maxHR","minHR","hrDelta","rmssd","sdnn","meanRR",
                               "accMagMean","accMagStd","accVertDelta","postureJerkPeak",
                               "ecgRAmpDev","ecgRAmpStdDev","ecgTAmpDev","ecgTAmpStdDev","ecgRTDev",
                               "hrRiseFromBaseline","rmssdPctChange","hrSlope","pNN50"]

    // ACC streams at ~200 Hz; feed every 4th sample to match the 4× subsampling
    // used when building training data (loadCSVs / compute_features.py).
    private var accSubsampleCounter = 0

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
        let cfg = MLModelConfiguration(); cfg.computeUnits = .cpuOnly
        // Prefer user-trained model in Documents; fall back to bundled model.
        let candidates: [URL] = [
            modelURL,
            Bundle.main.url(forResource: "POTSFlareupModel", withExtension: "mlmodelc"),
        ].compactMap { $0 }
        for url in candidates {
            guard fm.fileExists(atPath: url.path) else { continue }
            do {
                let candidate = try MLModel(contentsOf: url, configuration: cfg)
                if let missing = Self.validateFeatures(candidate) {
                    // A model whose inputs don't match featureDict would make every
                    // prediction throw silently (this exact bug shipped once). Refuse it.
                    print("[Predictor] Rejecting \(url.lastPathComponent): input mismatch (\(missing))")
                    continue
                }
                model = candidate
                modelLoaded = true
                loadMeta()
                return
            } catch {
                print("[Predictor] Load failed for \(url.lastPathComponent): \(error)")
            }
        }
        modelLoaded = false
    }

    /// Returns a human-readable description of any mismatch between the model's
    /// declared inputs and the features the app computes, or nil if they align.
    private static func validateFeatures(_ model: MLModel) -> String? {
        let declared = Set(model.modelDescription.inputDescriptionsByName.keys)
        let supplied = Set(featureNames)
        let missing = declared.subtracting(supplied)   // model wants inputs we don't send
        let extra   = supplied.subtracting(declared)   // we send inputs the model ignores
        guard missing.isEmpty && extra.isEmpty else {
            return "model needs \(missing.sorted()), app also has \(extra.sorted())"
        }
        return nil
    }

    func resetModel() {
        try? fm.removeItem(at: modelURL)
        try? fm.removeItem(at: metaURL)
        model = nil
        modelLoaded = false
        lastTrainedDate = nil
        modelF1 = nil
        lastPrediction = 0
        loadModel()
    }

    // MARK: - Real-time prediction

    /// Feed one accelerometer sample (subsampled 4× for train/serve parity).
    /// Called at the raw ~200 Hz stream rate; cheap buffering only.
    func ingestAcc(_ acc: AccSample) {
        accSubsampleCounter += 1
        if accSubsampleCounter % 4 != 0 { return }
        featureEngine.ingestAcc(acc)
    }

    /// Feed one ECG batch (kept at full 130 Hz — the morphology DSP needs the
    /// whole waveform). Buffering only; features are computed at window time.
    func ingestECG(_ ecg: ECGSample) {
        featureEngine.ingestECG(ecg)
    }

    /// Ingest an HR sample and, once a full window is available, run a prediction.
    /// Returns true when a new prediction was produced (so callers can act on it).
    @discardableResult
    func ingestAndPredict(hr: HRSample, temp: Double? = nil, baselineHR: Double = 0, baselineRMSSD: Double = 0) -> Bool {
        featureEngine.currentBaseline = baselineHR
        featureEngine.baselineRMSSD = baselineRMSSD
        featureEngine.ingestHR(hr)
        if let temp = temp { featureEngine.ingestTemp(temp) }

        guard modelLoaded, let model = model,
              let w = featureEngine.tryComputeWindow() else { return false }

        do {
            let input = try featureDict(w)
            let pred = try model.prediction(from: input)
            if let prob = pred.featureValue(for: "labelProbability")?.dictionaryValue,
               let p = prob[1 as NSNumber]?.doubleValue {
                lastPrediction = p
            } else if let lbl = pred.featureValue(for: "label")?.int64Value {
                lastPrediction = lbl == 1 ? 1.0 : 0.0
            }
            return true
        } catch {
            print("[Predictor] Predict error: \(error)")
            return false
        }
    }

    // MARK: - Train on last 30 days

    func trainOnRecentData(dataStore: DataStore) async {
        isTraining = true; trainingStatus = "Loading data..."

        let cutoff = Calendar.current.date(byAdding: .day, value: -30, to: Date()) ?? Date()
        let dataDir = dataStore.dataDir
        let flareups = dataStore.detectedFlareups.filter { $0.start >= cutoff }
        let localFM = fm
        let localModelURL = modelURL
        let localMetaURL = metaURL

        guard !flareups.isEmpty else {
            trainingStatus = "No flareups in last 30 days"; isTraining = false; return
        }

        trainingStatus = "Computing features..."

        do {
            let result = try await Task.detached(priority: .userInitiated) { () -> TrainResult in
                let (hr, acc) = try POTSPredictor.loadCSVs(from: dataDir, since: cutoff, fm: localFM)
                let windows = FeatureEngine.generateTrainingData(hrSamples: hr, accSamples: acc, flareups: flareups)
                let pos = windows.filter { $0.label == 1 }.count
                let neg = windows.filter { $0.label == 0 }.count

                guard windows.count >= 20, pos > 0 else {
                    throw NSError(domain: "POTSTraining", code: 1,
                        userInfo: [NSLocalizedDescriptionKey: "Need more data (\(windows.count) windows, \(pos) flareup)"])
                }

                // Split by DAY so overlapping sliding windows never straddle the
                // train/test boundary. Hold out ~30% of the days that contain a
                // flareup so the test set contains real pre-flareup episodes.
                let dayF = DateFormatter(); dayF.dateFormat = "yyyy-MM-dd"; dayF.timeZone = .current
                let byDay = Dictionary(grouping: windows) { dayF.string(from: $0.windowEnd) }
                let positiveDays = byDay.filter { $0.value.contains { $0.label == 1 } }.keys.sorted()
                let testDayCount = max(1, positiveDays.count * 3 / 10)
                // Spread held-out days across the timeline rather than taking a contiguous block.
                let stride = max(1, positiveDays.count / testDayCount)
                let testDays = Set(Swift.stride(from: 0, to: positiveDays.count, by: stride).prefix(testDayCount).map { positiveDays[$0] })

                var trainWindows = windows.filter { !testDays.contains(dayF.string(from: $0.windowEnd)) }
                let testWindows  = windows.filter {  testDays.contains(dayF.string(from: $0.windowEnd)) }

                // Oversample the minority class on the TRAIN split only.
                let trPos = trainWindows.filter { $0.label == 1 }.count
                let trNeg = trainWindows.filter { $0.label == 0 }.count
                if trPos > 0 && trNeg / trPos > 2 {
                    let target = trNeg / 2
                    let posWindows = trainWindows.filter { $0.label == 1 }
                    while trainWindows.filter({ $0.label == 1 }).count < target {
                        trainWindows.append(contentsOf: posWindows)
                    }
                    trainWindows.shuffle()
                }

                let classifier = try MLBoostedTreeClassifier(
                    trainingData: try POTSPredictor.dataTable(trainWindows), targetColumn: "label",
                    parameters: MLBoostedTreeClassifier.ModelParameters(
                        // Shallow: only a handful of independent flareup-days exist,
                        // so deep/many-iteration boosting just memorises them.
                        maxDepth: 4, maxIterations: 150, minLossReduction: 0.0,
                        minChildWeight: 10.0, stepSize: 0.1
                    )
                )

                // Honest F1 on the held-out days (accuracy is meaningless at ~3% positive).
                var f1 = 0.0, precision = 0.0, recall = 0.0
                if !testWindows.isEmpty {
                    let preds = try classifier.predictions(from: try POTSPredictor.dataTable(testWindows))
                    var tp = 0, fp = 0, fn = 0
                    for (i, w) in testWindows.enumerated() {
                        let p = preds[i].intValue ?? 0
                        let t = w.label ?? 0
                        if t == 1 && p == 1 { tp += 1 }
                        else if t == 0 && p == 1 { fp += 1 }
                        else if t == 1 && p == 0 { fn += 1 }
                    }
                    precision = tp + fp > 0 ? Double(tp) / Double(tp + fp) : 0
                    recall    = tp + fn > 0 ? Double(tp) / Double(tp + fn) : 0
                    f1        = precision + recall > 0 ? 2 * precision * recall / (precision + recall) : 0
                }

                // Ship a model fit on ALL days (train + held-out) for maximum data use.
                let shipModel = try MLBoostedTreeClassifier(
                    trainingData: try POTSPredictor.dataTable(trainWindows + testWindows), targetColumn: "label",
                    parameters: MLBoostedTreeClassifier.ModelParameters(
                        maxDepth: 4, maxIterations: 150, minLossReduction: 0.0,
                        minChildWeight: 10.0, stepSize: 0.1
                    )
                )
                let meta = MLModelMetadata(author: "POTS Monitor", shortDescription: "Flareup predictor", version: "2.0")
                try shipModel.write(to: localModelURL, metadata: meta)
                let m = ModelMeta(trainedAt: Date(), windows: windows.count, positive: pos, negative: neg, f1: f1)
                try JSONEncoder().encode(m).write(to: localMetaURL)
                return TrainResult(f1: f1, precision: precision, recall: recall, pos: pos, neg: neg)
            }.value

            trainingStatus = String(format: "F1 %.2f (P %.2f · R %.2f) · %d flareup, %d normal windows",
                                    result.f1, result.precision, result.recall, result.pos, result.neg)
            loadModel()
            lastTrainedDate = Date()
            modelF1 = result.f1
            isTraining = false
        } catch {
            trainingStatus = "Failed: \(error.localizedDescription)"; isTraining = false
        }
    }

    private struct TrainResult { let f1, precision, recall: Double; let pos, neg: Int }

    /// Builds an MLDataTable of exactly the 14 model features + label.
    nonisolated private static func dataTable(_ windows: [FeatureWindow]) throws -> MLDataTable {
        var dict: [String: MLDataValueConvertible] = [:]
        dict["meanHR"]             = windows.map { $0.meanHR }
        dict["maxHR"]              = windows.map { $0.maxHR }
        dict["minHR"]              = windows.map { $0.minHR }
        dict["hrDelta"]            = windows.map { $0.hrDelta }
        dict["rmssd"]              = windows.map { $0.rmssd }
        dict["sdnn"]               = windows.map { $0.sdnn }
        dict["meanRR"]             = windows.map { $0.meanRR }
        dict["accMagMean"]         = windows.map { $0.accMagnitudeMean }
        dict["accMagStd"]          = windows.map { $0.accMagnitudeStd }
        dict["accVertDelta"]       = windows.map { $0.accVerticalDelta }
        dict["postureJerkPeak"]    = windows.map { $0.postureJerkPeak }
        dict["ecgRAmpDev"]         = windows.map { $0.ecgRAmpDev }
        dict["ecgRAmpStdDev"]      = windows.map { $0.ecgRAmpStdDev }
        dict["ecgTAmpDev"]         = windows.map { $0.ecgTAmpDev }
        dict["ecgTAmpStdDev"]      = windows.map { $0.ecgTAmpStdDev }
        dict["ecgRTDev"]           = windows.map { $0.ecgRTDev }
        dict["hrRiseFromBaseline"] = windows.map { $0.hrRiseFromBaseline ?? 0.0 }
        dict["rmssdPctChange"]     = windows.map { $0.rmssdPercentChange ?? 0.0 }
        dict["hrSlope"]            = windows.map { $0.hrSlope }
        dict["pNN50"]              = windows.map { $0.pNN50 }
        dict["label"]              = windows.map { $0.label ?? 0 }
        return try MLDataTable(dictionary: dict)
    }

    // MARK: - CSV Parsing (static so Task.detached can call without actor hop)

    nonisolated private static func loadCSVs(from dir: URL, since cutoff: Date, fm: FileManager) throws -> ([(Date, Int, [Int])], [(Date, Int32, Int32, Int32)]) {
        let tsF = DateFormatter(); tsF.dateFormat = "yyyy-MM-dd'T'HH:mm:ss.SSS"; tsF.timeZone = .current
        let dayF = DateFormatter(); dayF.dateFormat = "yyyy-MM-dd"
        let cutoffStr = dayF.string(from: cutoff)
        var hr: [(Date, Int, [Int])] = []
        var acc: [(Date, Int32, Int32, Int32)] = []

        let files = try fm.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil)

        for f in files where f.lastPathComponent.hasPrefix("hr_") {
            guard let dateStr = fileDate(from: f), dateStr >= cutoffStr else { continue }
            autoreleasepool {
                guard let content = readCSVFile(f) else { return }
                for line in content.split(separator: "\n").dropFirst() {
                    let c = parseLine(String(line))
                    guard c.count >= 3, let ts = tsF.date(from: c[0]), ts >= cutoff, let h = Int(c[1]) else { continue }
                    let rrs = c[2].trimmingCharacters(in: .init(charactersIn: "\"")).split(separator: ";").compactMap { Int($0) }
                    hr.append((ts, h, rrs))
                }
            }
        }

        // Subsample ACC 4×: reduces a 200 Hz stream from ~43 M tuples to ~11 M
        // while still giving enough signal for magnitude/vertical-delta features.
        var accLineIdx = 0
        for f in files where f.lastPathComponent.hasPrefix("acc_") {
            guard let dateStr = fileDate(from: f), dateStr >= cutoffStr else { continue }
            autoreleasepool {
                guard let content = readCSVFile(f) else { return }
                for line in content.split(separator: "\n").dropFirst() {
                    accLineIdx += 1
                    guard accLineIdx % 4 == 0 else { continue }
                    let c = parseLine(String(line))
                    guard c.count >= 4, let ts = tsF.date(from: c[0]), ts >= cutoff,
                          let x = Int32(c[1]), let y = Int32(c[2]), let z = Int32(c[3]) else { continue }
                    acc.append((ts, x, y, z))
                }
            }
        }

        hr.sort { $0.0 < $1.0 }; acc.sort { $0.0 < $1.0 }
        return (hr, acc)
    }

    nonisolated private static func fileDate(from url: URL) -> String? {
        var name = url.lastPathComponent
        if name.hasSuffix(".csv.zlib") { name = String(name.dropLast(9)) }
        else if name.hasSuffix(".csv") { name = String(name.dropLast(4)) }
        guard let underscore = name.firstIndex(of: "_") else { return nil }
        let dateStr = String(name[name.index(after: underscore)...])
        return dateStr.count == 10 ? dateStr : nil
    }

    nonisolated private static func readCSVFile(_ url: URL) -> String? {
        if url.pathExtension == "zlib" {
            guard let compressed = try? Data(contentsOf: url),
                  let decompressed = try? (compressed as NSData).decompressed(using: .zlib) as Data else { return nil }
            return String(data: decompressed, encoding: .utf8)
        }
        return try? String(contentsOf: url, encoding: .utf8)
    }

    nonisolated private static func parseLine(_ line: String) -> [String] {
        var result: [String] = []; var cur = ""; var inQ = false
        for ch in line {
            if ch == "\"" { inQ.toggle() }
            else if ch == "," && !inQ { result.append(cur); cur = "" }
            else { cur.append(ch) }
        }
        result.append(cur); return result
    }

    // MARK: - Private helpers

    private func featureDict(_ w: FeatureWindow) throws -> MLDictionaryFeatureProvider {
        try MLDictionaryFeatureProvider(dictionary: [
            "meanHR":              MLFeatureValue(double: w.meanHR),
            "maxHR":               MLFeatureValue(double: w.maxHR),
            "minHR":               MLFeatureValue(double: w.minHR),
            "hrDelta":             MLFeatureValue(double: w.hrDelta),
            "rmssd":               MLFeatureValue(double: w.rmssd),
            "sdnn":                MLFeatureValue(double: w.sdnn),
            "meanRR":              MLFeatureValue(double: w.meanRR),
            "accMagMean":          MLFeatureValue(double: w.accMagnitudeMean),
            "accMagStd":           MLFeatureValue(double: w.accMagnitudeStd),
            "accVertDelta":        MLFeatureValue(double: w.accVerticalDelta),
            "postureJerkPeak":     MLFeatureValue(double: w.postureJerkPeak),
            "ecgRAmpDev":          MLFeatureValue(double: w.ecgRAmpDev),
            "ecgRAmpStdDev":       MLFeatureValue(double: w.ecgRAmpStdDev),
            "ecgTAmpDev":          MLFeatureValue(double: w.ecgTAmpDev),
            "ecgTAmpStdDev":       MLFeatureValue(double: w.ecgTAmpStdDev),
            "ecgRTDev":            MLFeatureValue(double: w.ecgRTDev),
            "hrRiseFromBaseline":  MLFeatureValue(double: w.hrRiseFromBaseline ?? 0.0),
            "rmssdPctChange":      MLFeatureValue(double: w.rmssdPercentChange ?? 0.0),
            "hrSlope":             MLFeatureValue(double: w.hrSlope),
            "pNN50":               MLFeatureValue(double: w.pNN50),
        ])
    }

    private struct ModelMeta: Codable {
        let trainedAt: Date; let windows: Int; let positive: Int; let negative: Int; let f1: Double
    }
    private func loadMeta() {
        guard let d = try? Data(contentsOf: metaURL),
              let m = try? JSONDecoder().decode(ModelMeta.self, from: d) else { return }
        lastTrainedDate = m.trainedAt; modelF1 = m.f1
    }
}
