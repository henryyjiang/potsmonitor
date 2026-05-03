import Foundation
import Combine

@MainActor
class FlareupDetector: ObservableObject {

    // MARK: - Config

    static let hrRiseThreshold: Double = 30.0
    static let sustainedDuration: TimeInterval = 60.0
    static let baselineWindow: TimeInterval = 30 * 60
    static let minimumBaselineSamples = 30
    static let rmssdCollapseRatio: Double = 0.5
    static let recentHRVWindow: TimeInterval = 120

    // MARK: - Published

    @Published var isInFlareup = false
    @Published var currentFlareupStart: Date?
    @Published var currentPeakHR: Int = 0
    @Published var activeFlareupDuration: TimeInterval = 0
    @Published var currentBaseline: Double = 0
    @Published var adaptiveThreshold: Int = 0
    @Published var hrvConfirmed = false
    @Published var hrvCapable = false

    // MARK: - Baseline buffers

    private var hrHistory: [(Date, Int)] = []
    private var rrHistory: [(Date, Double)] = []

    // MARK: - Flareup state

    private var aboveThresholdSince: Date?
    private var peakDuringCurrent: Int = 0
    private var confirmed = false
    private var hrvCollapseDuringCurrent = false

    var onFlareupDetected: ((DetectedFlareup) -> Void)?
    var onFlareupConfirmed: ((Int, Double) -> Void)?

    /// Current baseline RMSSD over the 30-min window, for use by FeatureEngine.
    var currentBaselineRMSSD: Double {
        let allRRs = rrHistory.map { $0.1 }
        guard allRRs.count >= 20 else { return 0 }
        return Self.computeRMSSD(allRRs)
    }

    // MARK: - Ingest

    func processHR(_ sample: HRSample) {
        let hr = sample.hr
        let timestamp = sample.timestamp

        hrHistory.append((timestamp, hr))
        for rr in sample.rrIntervals {
            rrHistory.append((timestamp, Double(rr)))
        }
        trimHistory(before: timestamp.addingTimeInterval(-Self.baselineWindow))

        guard let baseline = computeBaseline() else { return }
        let threshold = baseline + Self.hrRiseThreshold

        currentBaseline = baseline
        adaptiveThreshold = Int(threshold)

        let deviceHasHRV = isHRVCapable(at: timestamp)
        hrvCapable = deviceHasHRV
        let hrvCollapse = checkHRVCollapse(at: timestamp)

        if Double(hr) > threshold {
            if aboveThresholdSince == nil {
                aboveThresholdSince = timestamp
                peakDuringCurrent = hr
                hrvCollapseDuringCurrent = hrvCollapse
            } else {
                peakDuringCurrent = max(peakDuringCurrent, hr)
                if hrvCollapse { hrvCollapseDuringCurrent = true }
            }

            if let start = aboveThresholdSince {
                let elapsed = timestamp.timeIntervalSince(start)
                let durationMet = elapsed >= Self.sustainedDuration

                if !confirmed {
                    let shouldConfirm = deviceHasHRV
                        ? durationMet && hrvCollapseDuringCurrent
                        : durationMet

                    if shouldConfirm {
                        confirmed = true
                        isInFlareup = true
                        currentFlareupStart = start
                        currentPeakHR = peakDuringCurrent
                        hrvConfirmed = hrvCollapseDuringCurrent
                        onFlareupConfirmed?(peakDuringCurrent, baseline)
                    }
                }

                if confirmed {
                    activeFlareupDuration = elapsed
                    currentPeakHR = peakDuringCurrent
                    if hrvCollapse { hrvCollapseDuringCurrent = true }
                    hrvConfirmed = hrvCollapseDuringCurrent
                }
            }
        } else {
            if confirmed, let start = aboveThresholdSince {
                let duration = Int(timestamp.timeIntervalSince(start))
                let flareup = DetectedFlareup(
                    id: UUID(),
                    start: start,
                    end: timestamp,
                    peakHR: peakDuringCurrent,
                    durationSeconds: duration,
                    baselineHR: baseline,
                    thresholdUsed: threshold,
                    hrvConfirmed: hrvCollapseDuringCurrent
                )
                onFlareupDetected?(flareup)
            }
            resetState()
        }
    }

    func flush() {
        if confirmed, let start = aboveThresholdSince {
            let now = Date()
            let baseline = computeBaseline() ?? 0
            let flareup = DetectedFlareup(
                id: UUID(),
                start: start,
                end: now,
                peakHR: peakDuringCurrent,
                durationSeconds: Int(now.timeIntervalSince(start)),
                baselineHR: baseline,
                thresholdUsed: baseline + Self.hrRiseThreshold,
                hrvConfirmed: hrvCollapseDuringCurrent
            )
            onFlareupDetected?(flareup)
        }
        resetState()
    }

    // MARK: - Private

    private func isHRVCapable(at timestamp: Date) -> Bool {
        let recentCutoff = timestamp.addingTimeInterval(-Self.recentHRVWindow)
        return rrHistory.filter { $0.0 >= recentCutoff }.count >= 10
    }

    private func computeBaseline() -> Double? {
        guard hrHistory.count >= Self.minimumBaselineSamples else { return nil }
        let sum = hrHistory.reduce(0.0) { $0 + Double($1.1) }
        return sum / Double(hrHistory.count)
    }

    private func checkHRVCollapse(at timestamp: Date) -> Bool {
        let recentCutoff = timestamp.addingTimeInterval(-Self.recentHRVWindow)
        let recentRRs = rrHistory.filter { $0.0 >= recentCutoff }.map { $0.1 }
        let allRRs = rrHistory.map { $0.1 }

        guard recentRRs.count >= 4, allRRs.count >= 20 else { return false }

        let recentRMSSD = Self.computeRMSSD(recentRRs)
        let baselineRMSSD = Self.computeRMSSD(allRRs)

        guard baselineRMSSD > 0 else { return false }
        return recentRMSSD < baselineRMSSD * Self.rmssdCollapseRatio
    }

    static func computeRMSSD(_ rrs: [Double]) -> Double {
        guard rrs.count >= 2 else { return 0 }
        var s: Double = 0
        for i in 1..<rrs.count { let d = rrs[i] - rrs[i-1]; s += d * d }
        return sqrt(s / Double(rrs.count - 1))
    }

    private func trimHistory(before cutoff: Date) {
        hrHistory.removeAll { $0.0 < cutoff }
        rrHistory.removeAll { $0.0 < cutoff }
    }

    private func resetState() {
        aboveThresholdSince = nil
        peakDuringCurrent = 0
        confirmed = false
        hrvCollapseDuringCurrent = false
        isInFlareup = false
        currentFlareupStart = nil
        currentPeakHR = 0
        activeFlareupDuration = 0
        hrvConfirmed = false
    }
}
