import Foundation

/// ECG morphology features derived from the H10's 130 Hz waveform. These expose
/// signals HR + RR intervals cannot: R-wave amplitude (electrical-axis / posture /
/// sympathetic shifts), T-wave amplitude (sympathetic tone), and the R→T-peak
/// interval (a QT proxy that shortens with sympathetic drive).
///
/// The DSP here is a direct port of the scipy pipeline in build_ecg_features.py
/// (Butterworth filtfilt + find_peaks + T-wave search) and is verified to match it
/// to <0.5% on real data. Keeping this parity is what lets the offline-trained model
/// use these features live without train/serve skew.
enum ECGFeatures {

    static let fs = 130.0
    static let minBeats = 5

    struct Morphology {
        let rAmpMean, rAmpStd, tAmpMean, tAmpStd, rtMean: Double
    }

    // Butterworth coefficients (scipy butter(2, ...) at 130 Hz), hardcoded for parity.
    private static let bpB = [0.08560470548729655, 0.0, -0.1712094109745931, 0.0, 0.08560470548729655]
    private static let bpA = [1.0, -2.657395687639593, 2.863527458909936, -1.5335175290358514, 0.36156443847608793]
    private static let bpZI = [-0.08560470548729454, -0.08560470548729991, 0.08560470548729893, 0.08560470548729582]
    private static let lpB = [0.02914205405784711, 0.05828410811569422, 0.02914205405784711]
    private static let lpA = [1.0, -1.462404515841618, 0.5789727320730064]
    private static let lpZI = [0.9708579459421529, -0.5498306780151593]

    /// Compute the 5 morphology features over one window's raw µV samples
    /// (time-ordered). Returns nil if there aren't enough clean beats.
    static func morphology(_ signal: [Double]) -> Morphology? {
        guard signal.count >= Int(fs) else { return nil }
        let filt = filtfilt(bpB, bpA, bpZI, signal)
        let peaks = findPeaks(filt, distance: Int(0.3 * fs), height: std(filt) * 1.5)
        guard peaks.count >= minBeats else { return nil }
        let low = filtfilt(lpB, lpA, lpZI, signal)

        let rAmp = peaks.map { signal[$0] }
        var tAmp: [Double] = []
        var rt: [Double] = []
        let w0 = Int(0.15 * fs), w1 = Int(0.40 * fs), bwin = Int(0.05 * fs)
        for p in peaks {
            let lo = p + w0, hi = p + w1
            guard hi < low.count else { continue }
            let base = p >= bwin ? mean(Array(low[(p - bwin)..<p])) : low[p]
            let seg = Array(low[lo..<hi])
            var bestJ = 0, bestV = -1.0
            for (j, v) in seg.enumerated() {
                let d = abs(v - base)
                if d > bestV { bestV = d; bestJ = j }
            }
            tAmp.append(seg[bestJ] - base)
            rt.append(Double(bestJ + w0) / fs * 1000)
        }
        guard !tAmp.isEmpty else { return nil }
        return Morphology(rAmpMean: mean(rAmp), rAmpStd: std(rAmp),
                          tAmpMean: mean(tAmp), tAmpStd: std(tAmp), rtMean: mean(rt))
    }

    // MARK: - DSP primitives (parity with scipy)

    /// Direct-form II transposed IIR filter with initial state `zi`.
    private static func lfilter(_ b: [Double], _ a: [Double], _ x: [Double], _ zi: [Double]) -> [Double] {
        let n = max(b.count, a.count)
        let bb = b + Array(repeating: 0, count: n - b.count)
        let aa = a + Array(repeating: 0, count: n - a.count)
        var z = zi
        var y = [Double](repeating: 0, count: x.count)
        for i in 0..<x.count {
            let xi = x[i]
            let yi = bb[0] * xi + z[0]
            if n > 2 { for j in 1..<(n - 1) { z[j - 1] = bb[j] * xi + z[j] - aa[j] * yi } }
            z[n - 2] = bb[n - 1] * xi - aa[n - 1] * yi
            y[i] = yi
        }
        return y
    }

    /// Odd signal extension (scipy `odd_ext`) used by filtfilt padding.
    private static func oddExt(_ x: [Double], _ n: Int) -> [Double] {
        let L = x.count
        let left = (0..<n).map { 2 * x[0] - x[n - $0] }
        let right = (0..<n).map { 2 * x[L - 1] - x[L - 2 - $0] }
        return left + x + right
    }

    /// Zero-phase forward-backward filtering (scipy `filtfilt`, method='pad').
    private static func filtfilt(_ b: [Double], _ a: [Double], _ zi: [Double], _ x: [Double]) -> [Double] {
        let ntaps = max(b.count, a.count)
        let padlen = 3 * (ntaps - 1)
        guard x.count > padlen else { return x }
        let ext = oddExt(x, padlen)
        var y = lfilter(b, a, ext, zi.map { $0 * ext[0] })
        y.reverse()
        y = lfilter(b, a, y, zi.map { $0 * y[0] })
        y.reverse()
        return Array(y[padlen..<(y.count - padlen)])
    }

    /// Local maxima above `height`, greedily thinned to a minimum `distance`
    /// (highest-first) — mirrors scipy `find_peaks(..., distance=, height=)`.
    private static func findPeaks(_ x: [Double], distance: Int, height: Double) -> [Int] {
        guard x.count > 2 else { return [] }
        var cand: [Int] = []
        for i in 1..<(x.count - 1) where x[i] > x[i - 1] && x[i] > x[i + 1] && x[i] >= height {
            cand.append(i)
        }
        let ordered = cand.sorted { x[$0] > x[$1] }
        var removed = Set<Int>()
        var kept: [Int] = []
        for p in ordered where !removed.contains(p) {
            kept.append(p)
            for q in cand where q != p && abs(q - p) < distance { removed.insert(q) }
        }
        return kept.sorted()
    }

    private static func mean(_ v: [Double]) -> Double { v.isEmpty ? 0 : v.reduce(0, +) / Double(v.count) }
    private static func std(_ v: [Double]) -> Double {
        guard !v.isEmpty else { return 0 }
        let m = mean(v)
        return (v.map { ($0 - m) * ($0 - m) }.reduce(0, +) / Double(v.count)).squareRoot()
    }
}
