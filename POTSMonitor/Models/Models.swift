import Foundation

// MARK: - Sensor Samples

struct HRSample: Codable {
    let timestamp: Date
    let hr: Int
    let rrIntervals: [Int]     // ms
    let contactStatus: Bool
    let contactStatusSupported: Bool
}

struct AccSample: Codable {
    let timestamp: Date
    let x: Int32   // mG
    let y: Int32
    let z: Int32
}

struct TemperatureSample: Codable {
    let timestamp: Date
    let celsius: Double
}

struct ECGSample: Codable {
    let timestamp: Date
    let microVolts: [Int32]  // batch of ECG values at 130Hz
    let sampleRate: Int      // Hz (130 for H10)
}

// MARK: - Auto-Detected Flareup

struct DetectedFlareup: Identifiable, Codable {
    let id: UUID
    let start: Date
    var end: Date?
    let peakHR: Int
    let durationSeconds: Int
    var baselineHR: Double?
    var thresholdUsed: Double?
    var hrvConfirmed: Bool?
}

// MARK: - Prediction Tracking

struct PredictionRecord: Identifiable, Codable {
    let id: UUID
    let timestamp: Date
    let probability: Double
    var outcome: Outcome

    enum Outcome: String, Codable {
        case pending, truePositive, falsePositive
    }
}

// MARK: - ML Feature Window

struct FeatureWindow: Codable {
    let windowEnd: Date
    let meanHR: Double
    let maxHR: Double
    let minHR: Double
    let hrDelta: Double
    let rmssd: Double
    let sdnn: Double
    let meanRR: Double
    let accMagnitudeMean: Double
    let accMagnitudeStd: Double
    let accVerticalDelta: Double
    let skinTemp: Double?
    let baselineHR: Double?
    let hrRiseFromBaseline: Double?
    let rmssdPercentChange: Double?
    let label: Int?
}
