import Foundation

class FeatureEngine {

    static let windowDuration: TimeInterval = 60.0
    static let slideStep: TimeInterval = 10.0

    private var hrBuf: [(Date, Int)] = []
    private var rrBuf: [(Date, Int)] = []
    private var accBuf: [(Date, Int32, Int32, Int32)] = []
    private var lastTemp: Double?
    private var lastWindowEnd: Date = .distantPast

    var currentBaseline: Double = 0
    var baselineRMSSD: Double = 0

    func ingestHR(_ s: HRSample) {
        hrBuf.append((s.timestamp, s.hr))
        for rr in s.rrIntervals { rrBuf.append((s.timestamp, rr)) }
        trim()
    }

    func ingestAcc(_ s: AccSample) {
        accBuf.append((s.timestamp, s.x, s.y, s.z))
        trim()
    }

    func ingestTemp(_ c: Double) { lastTemp = c }

    private func trim() {
        let cut = Date().addingTimeInterval(-120)
        hrBuf.removeAll { $0.0 < cut }
        rrBuf.removeAll { $0.0 < cut }
        accBuf.removeAll { $0.0 < cut }
    }

    func tryComputeWindow() -> FeatureWindow? {
        let now = Date()
        guard now.timeIntervalSince(lastWindowEnd) >= Self.slideStep else { return nil }
        let wStart = now.addingTimeInterval(-Self.windowDuration)

        let hrs = hrBuf.filter { $0.0 >= wStart }.map { Double($0.1) }
        let rrs = rrBuf.filter { $0.0 >= wStart }.map { Double($0.1) }
        let accs = accBuf.filter { $0.0 >= wStart }

        guard hrs.count >= 5 else { return nil }
        lastWindowEnd = now

        let mags = accs.map { sqrt(Double($0.1*$0.1) + Double($0.2*$0.2) + Double($0.3*$0.3)) }
        let zVals = accs.map { Double($0.3) }
        let vDelta: Double = zVals.count >= 4
            ? abs(Array(zVals.suffix(zVals.count/4)).mean - Array(zVals.prefix(zVals.count/4)).mean)
            : 0
        let postureJerk = Self.peakAngularVelocity(accs)

        let windowRMSSD = Self.rmssd(rrs)
        var rmssdPctChange: Double? = nil
        if baselineRMSSD > 0 && rrs.count >= 4 {
            rmssdPctChange = (windowRMSSD - baselineRMSSD) / baselineRMSSD
        }

        let hrTimes = hrBuf.filter { $0.0 >= wStart }.map { $0.0.timeIntervalSinceReferenceDate }
        let hrSlope = Self.linSlope(hrs, times: hrTimes)
        let winPNN50 = Self.pnn50(rrs)

        return FeatureWindow(
            windowEnd: now,
            meanHR: hrs.mean, maxHR: hrs.max() ?? 0, minHR: hrs.min() ?? 0,
            hrDelta: (hrs.max() ?? 0) - (hrs.min() ?? 0),
            rmssd: windowRMSSD, sdnn: rrs.stdDev, meanRR: rrs.mean,
            accMagnitudeMean: mags.isEmpty ? 0 : mags.mean,
            accMagnitudeStd: mags.isEmpty ? 0 : mags.stdDev,
            accVerticalDelta: vDelta,
            postureJerkPeak: postureJerk,
            skinTemp: lastTemp,
            baselineHR: currentBaseline > 0 ? currentBaseline : nil,
            hrRiseFromBaseline: currentBaseline > 0 ? hrs.mean - currentBaseline : nil,
            rmssdPercentChange: rmssdPctChange,
            hrSlope: hrSlope,
            pNN50: winPNN50,
            label: nil
        )
    }

    // MARK: - Batch (for training)

    static func generateTrainingData(
        hrSamples: [(Date, Int, [Int])],
        accSamples: [(Date, Int32, Int32, Int32)],
        flareups: [DetectedFlareup],
        windowDuration: TimeInterval = 60.0,
        slideStep: TimeInterval = 10.0
    ) -> [FeatureWindow] {
        guard let earliest = hrSamples.first?.0,
              let latest = hrSamples.last?.0 else { return [] }

        // Pre-extract timestamps as TimeInterval so binary search avoids
        // repeated Date boxing. O(n) once here instead of O(n) per window.
        let hrTS  = hrSamples.map  { $0.0.timeIntervalSinceReferenceDate }
        let accTS = accSamples.map { $0.0.timeIntervalSinceReferenceDate }

        // Pre-compute flareup intervals once (avoids repeated closure captures).
        let flareupIntervals: [(start: TimeInterval, end: TimeInterval)] = flareups.map { f in
            let s = f.start.timeIntervalSinceReferenceDate
            let e = (f.end ?? f.start.addingTimeInterval(Double(f.durationSeconds))).timeIntervalSinceReferenceDate
            return (s, e)
        }

        var windows: [FeatureWindow] = []
        var t = earliest.addingTimeInterval(windowDuration)
        let baselineSpan: TimeInterval = 30 * 60
        let predictHorizon: TimeInterval = 10 * 60

        while t <= latest {
            let tRef       = t.timeIntervalSinceReferenceDate
            let wStartRef  = tRef - windowDuration
            let blStartRef = tRef - baselineSpan

            // Binary search — O(log n) per window instead of O(n) linear scan.
            let hrLo = lowerBound(hrTS, wStartRef)
            let hrHi = upperBound(hrTS, tRef)
            guard hrHi - hrLo >= 5 else { t = t.addingTimeInterval(slideStep); continue }

            let accLo = lowerBound(accTS, wStartRef)
            let accHi = upperBound(accTS, tRef)

            // Window slices (no copying — ArraySlice shares storage).
            let hrW  = hrSamples[hrLo..<hrHi]
            let accW = accSamples[accLo..<accHi]

            let hrs  = hrW.map { Double($0.1) }
            let hrTs = hrW.map { $0.0.timeIntervalSinceReferenceDate }
            let rrs  = hrW.flatMap { $0.2 }.map { Double($0) }
            let mags = accW.map { sqrt(Double($0.1*$0.1) + Double($0.2*$0.2) + Double($0.3*$0.3)) }
            let zV   = accW.map { Double($0.3) }
            let vD: Double = zV.count >= 4
                ? abs(Array(zV.suffix(zV.count/4)).mean - Array(zV.prefix(zV.count/4)).mean)
                : 0
            let postureJerk = peakAngularVelocity(Array(accW))

            // 30-min baseline — one binary search, reuse hrHi upper bound.
            let blLo = lowerBound(hrTS, blStartRef)
            let blW  = hrSamples[blLo..<hrHi]

            var baselineHR: Double? = nil
            var hrRise: Double? = nil
            let windowRMSSD = rmssd(rrs)
            var rmssdPctChange: Double? = nil
            let windowHRSlope = linSlope(hrs, times: hrTs)
            let windowPNN50   = pnn50(rrs)

            if blW.count >= 30 {
                let bhr = blW.map { Double($0.1) }.mean
                baselineHR = bhr
                hrRise = hrs.mean - bhr

                let blRRs = blW.flatMap { $0.2 }.map { Double($0) }
                let blRMSSD = rmssd(blRRs)
                if blRMSSD > 0 && rrs.count >= 4 {
                    rmssdPctChange = (windowRMSSD - blRMSSD) / blRMSSD
                }
            }

            // Label: skip during-flareup windows; positive if flareup starts within the horizon.
            let duringFlareup = flareupIntervals.contains { tRef >= $0.start && tRef <= $0.end }
            if duringFlareup { t = t.addingTimeInterval(slideStep); continue }

            let preFlareup = flareupIntervals.contains { tRef < $0.start && tRef >= $0.start - predictHorizon }

            windows.append(FeatureWindow(
                windowEnd: t,
                meanHR: hrs.mean, maxHR: hrs.max() ?? 0, minHR: hrs.min() ?? 0,
                hrDelta: (hrs.max() ?? 0) - (hrs.min() ?? 0),
                rmssd: windowRMSSD, sdnn: rrs.stdDev, meanRR: rrs.mean,
                accMagnitudeMean: mags.isEmpty ? 0 : mags.mean,
                accMagnitudeStd: mags.isEmpty ? 0 : mags.stdDev,
                accVerticalDelta: vD,
                postureJerkPeak: postureJerk,
                skinTemp: nil,
                baselineHR: baselineHR,
                hrRiseFromBaseline: hrRise,
                rmssdPercentChange: rmssdPctChange,
                hrSlope: windowHRSlope,
                pNN50: windowPNN50,
                label: preFlareup ? 1 : 0
            ))
            t = t.addingTimeInterval(slideStep)
        }
        return windows
    }

    // MARK: - Binary search helpers (mirrors np.searchsorted left/right)

    // First index where timestamps[i] >= value  (searchsorted left)
    private static func lowerBound(_ timestamps: [TimeInterval], _ value: TimeInterval) -> Int {
        var lo = 0, hi = timestamps.count
        while lo < hi {
            let mid = (lo + hi) >> 1
            if timestamps[mid] < value { lo = mid + 1 } else { hi = mid }
        }
        return lo
    }

    // First index where timestamps[i] > value  (searchsorted right)
    private static func upperBound(_ timestamps: [TimeInterval], _ value: TimeInterval) -> Int {
        var lo = 0, hi = timestamps.count
        while lo < hi {
            let mid = (lo + hi) >> 1
            if timestamps[mid] <= value { lo = mid + 1 } else { hi = mid }
        }
        return lo
    }

    static func rmssd(_ rrs: [Double]) -> Double {
        guard rrs.count >= 2 else { return 0 }
        var s: Double = 0
        for i in 1..<rrs.count { let d = rrs[i]-rrs[i-1]; s += d*d }
        return sqrt(s / Double(rrs.count-1))
    }

    /// Largest gravity-vector rotation (deg) between consecutive 1-second bins in
    /// the window. Averaging to 1 s bins strips motion jitter and leaves torso
    /// orientation; the max captures the brief sit/lie→stand transient (the
    /// orthostatic trigger) that whole-window averaging washes out. Must stay in
    /// parity with `peak_angular_velocity` in compute_features.py.
    static func peakAngularVelocity(_ samples: [(Date, Int32, Int32, Int32)]) -> Double {
        guard samples.count >= 2 else { return 0 }
        var means: [(Double, Double, Double)] = []
        var curSec = samples[0].0.timeIntervalSinceReferenceDate.rounded(.down)
        var sx = 0.0, sy = 0.0, sz = 0.0, c = 0.0
        for s in samples {
            let sec = s.0.timeIntervalSinceReferenceDate.rounded(.down)
            if sec != curSec {
                if c > 0 { means.append((sx/c, sy/c, sz/c)) }
                curSec = sec; sx = 0; sy = 0; sz = 0; c = 0
            }
            sx += Double(s.1); sy += Double(s.2); sz += Double(s.3); c += 1
        }
        if c > 0 { means.append((sx/c, sy/c, sz/c)) }
        guard means.count >= 2 else { return 0 }

        var maxAngle = 0.0
        for i in 1..<means.count {
            let a = means[i-1], b = means[i]
            let na = sqrt(a.0*a.0 + a.1*a.1 + a.2*a.2)
            let nb = sqrt(b.0*b.0 + b.1*b.1 + b.2*b.2)
            guard na > 0, nb > 0 else { continue }
            let cosA = max(-1.0, min(1.0, (a.0*b.0 + a.1*b.1 + a.2*b.2) / (na * nb)))
            let ang = acos(cosA) * 180 / .pi
            if ang > maxAngle { maxAngle = ang }
        }
        return maxAngle
    }

    static func pnn50(_ rrs: [Double]) -> Double {
        guard rrs.count >= 2 else { return 0 }
        var count = 0
        for i in 1..<rrs.count { if abs(rrs[i] - rrs[i-1]) > 50 { count += 1 } }
        return Double(count) / Double(rrs.count - 1)
    }

    static func linSlope(_ values: [Double], times: [Double]) -> Double {
        guard values.count >= 3, values.count == times.count else { return 0 }
        let n = Double(values.count)
        let mx = times.mean; let my = values.mean
        var num = 0.0; var den = 0.0
        for i in 0..<values.count {
            let dx = times[i] - mx
            num += dx * (values[i] - my)
            den += dx * dx
        }
        return den > 0 ? num / den : 0
    }
}

extension Array where Element == Double {
    var mean: Double { isEmpty ? 0 : reduce(0,+)/Double(count) }
    var stdDev: Double {
        guard count >= 2 else { return 0 }
        let m = mean; return sqrt(map { ($0-m)*($0-m) }.reduce(0,+)/Double(count-1))
    }
}
