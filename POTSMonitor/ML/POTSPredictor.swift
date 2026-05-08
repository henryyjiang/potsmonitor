import Foundation
import CoreML

@MainActor
class POTSPredictor: ObservableObject {

    @Published var modelLoaded = false
    @Published var lastPrediction: Double = 0

    private var model: MLModel?
    private let featureEngine = FeatureEngine()
    private let fm = FileManager.default

    private var modelURL: URL {
        fm.urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("POTSFlareupModel.mlmodelc")
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
                model = try MLModel(contentsOf: url, configuration: cfg)
                modelLoaded = true
                return
            } catch {
                print("[Predictor] Load failed for \(url.lastPathComponent): \(error)")
            }
        }
        modelLoaded = false
    }

    func resetModel() {
        try? fm.removeItem(at: modelURL)
        model = nil
        modelLoaded = false
        lastPrediction = 0
        loadModel()
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
                lastPrediction = p
            } else if let lbl = pred.featureValue(for: "label")?.int64Value {
                lastPrediction = lbl == 1 ? 1.0 : 0.0
            }
        } catch {
            print("[Predictor] Predict error: \(error)")
        }
    }

    // MARK: - Private

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
            "hrRiseFromBaseline":  MLFeatureValue(double: w.hrRiseFromBaseline ?? 0.0),
            "rmssdPctChange":      MLFeatureValue(double: w.rmssdPercentChange ?? 0.0),
        ])
    }
}
