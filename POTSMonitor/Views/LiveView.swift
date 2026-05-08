import SwiftUI

struct LiveView: View {
    @EnvironmentObject var polar: PolarManager
    @EnvironmentObject var dataStore: DataStore
    @EnvironmentObject var detector: FlareupDetector
    @EnvironmentObject var notifications: NotificationManager
    @EnvironmentObject var tracker: PredictionTracker
    @StateObject private var predictor = POTSPredictor()
    
    @State private var pulse = false
    
    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 18) {
                    statusBar
                    hrCircle
                    
                    if detector.isInFlareup {
                        flareupBanner
                    }
                    
                    if predictor.modelLoaded {
                        predictionBar
                    }
                    
                    metricsRow
                    trackingButton
                    recordingInfo
                }
                .padding()
            }
            .background(Color(.systemGroupedBackground))
            .navigationTitle("POTS Monitor")
        }
        .onChange(of: polar.currentHR) { _ in
            predictor.ingestAndPredict(
                hr: HRSample(timestamp: Date(), hr: polar.currentHR, rrIntervals: polar.currentRR,
                             contactStatus: true, contactStatusSupported: true),
                acc: nil, temp: nil,
                baselineHR: detector.currentBaseline,
                baselineRMSSD: detector.currentBaselineRMSSD
            )
            tracker.resolvePending()
        }
        .onChange(of: predictor.lastPrediction) { newValue in
            let threshold = NotificationManager.predictionThreshold
            if newValue >= threshold {
                notifications.sendFlareupPredictedNotification(probability: newValue)
                tracker.recordPrediction(probability: newValue)
            }
        }
    }
    
    // MARK: - Status
    
    private var statusBar: some View {
        HStack(spacing: 6) {
            Circle().fill(polar.isConnected ? Color.green : Color.red).frame(width: 8, height: 8)
            Text(polar.connectionStatus)
                .font(.system(.caption, design: .monospaced)).foregroundColor(.secondary)
            Spacer()
            if polar.isConnected {
                Image(systemName: "battery.\(polar.batteryLevel > 50 ? "100" : "25")")
                    .foregroundColor(polar.batteryLevel > 20 ? .green : .red).font(.caption)
                Text("\(polar.batteryLevel)%")
                    .font(.system(.caption, design: .monospaced)).foregroundColor(.secondary)
            }
        }
        .padding(.horizontal, 12).padding(.vertical, 7)
        .background(Color(.secondarySystemGroupedBackground)).cornerRadius(9)
    }
    
    // MARK: - HR Circle
    
    private var hrCircle: some View {
        ZStack {
            Circle().stroke(Color.red.opacity(0.12), lineWidth: 3).frame(width: 170, height: 170)
            
            if polar.isStreaming && polar.currentHR > 0 {
                Circle().stroke(Color.red.opacity(0.35), lineWidth: 3)
                    .frame(width: 170, height: 170)
                    .scaleEffect(pulse ? 1.1 : 1.0)
                    .opacity(pulse ? 0 : 0.4)
                    .animation(.easeOut(duration: 60.0/Double(max(polar.currentHR,40)))
                        .repeatForever(autoreverses: false), value: pulse)
            }
            
            VStack(spacing: 2) {
                Image(systemName: "heart.fill").font(.title3).foregroundColor(.red)
                Text(polar.currentHR > 0 ? "\(polar.currentHR)" : "--")
                    .font(.system(size: 52, weight: .bold, design: .rounded))
                    .foregroundColor(polar.currentHR > 120 ? .red : polar.currentHR > 100 ? .orange : .primary)
                Text("BPM").font(.system(.caption2, design: .monospaced)).foregroundColor(.secondary)
            }
        }
        .padding(.vertical, 2)
        .onAppear { pulse = true }
    }
    
    // MARK: - Auto Flareup Banner
    
    private var flareupBanner: some View {
        HStack(spacing: 10) {
            Image(systemName: "exclamationmark.triangle.fill")
                .foregroundColor(.white).font(.title3)
            VStack(alignment: .leading, spacing: 2) {
                Text("FLAREUP DETECTED")
                    .font(.system(.subheadline, design: .rounded, weight: .bold))
                    .foregroundColor(.white)
                Text("+\(Int(Double(detector.currentPeakHR) - detector.currentBaseline)) BPM over baseline for \(Int(detector.activeFlareupDuration))s\(detector.hrvConfirmed ? " · HRV ↓" : "")")
                    .font(.system(.caption, design: .monospaced))
                    .foregroundColor(.white.opacity(0.85))
                Text("Baseline \(Int(detector.currentBaseline)) · Threshold \(detector.adaptiveThreshold) · Peak \(detector.currentPeakHR)")
                    .font(.system(.caption2, design: .monospaced))
                    .foregroundColor(.white.opacity(0.7))
            }
            Spacer()
        }
        .padding(12)
        .background(Color.red).cornerRadius(12)
    }
    
    // MARK: - Prediction
    
    private var predictionBar: some View {
        HStack {
            Image(systemName: predictor.lastPrediction > 0.6 ? "exclamationmark.triangle.fill" : "checkmark.shield.fill")
                .foregroundColor(predColor)
            VStack(alignment: .leading, spacing: 2) {
                Text(predLabel)
                    .font(.system(.subheadline, design: .rounded, weight: .medium))
                    .foregroundColor(predColor)
                ProgressView(value: predictor.lastPrediction).tint(predColor)
            }
            Spacer()
            Text("\(Int(predictor.lastPrediction * 100))%")
                .font(.system(.title3, design: .rounded, weight: .bold))
                .foregroundColor(predColor)
        }
        .padding(12)
        .background(predColor.opacity(0.08)).cornerRadius(11)
    }
    
    private var predColor: Color {
        predictor.lastPrediction > 0.7 ? .red : predictor.lastPrediction > 0.4 ? .orange : .green
    }
    private var predLabel: String {
        predictor.lastPrediction > 0.7 ? "Flareup Likely" : predictor.lastPrediction > 0.4 ? "Elevated Risk" : "Stable"
    }
    
    // MARK: - Metrics
    
    private var metricsRow: some View {
        HStack(spacing: 8) {
            metric("HRV", value: rmssdStr, unit: "ms", icon: "waveform.path.ecg")
            metric("ECG", value: polar.isStreaming ? "Live" : "--", unit: "", icon: "waveform.path.ecg.rectangle")
            metric("Flareups", value: "\(todayFlareupCount)", unit: "", icon: "flame.fill")
        }
    }
    
    private var todayFlareupCount: Int {
        let startOfDay = Calendar.current.startOfDay(for: Date())
        return dataStore.detectedFlareups.filter { $0.start >= startOfDay }.count
    }

    private var rmssdStr: String {
        let rr = polar.currentRR; guard rr.count >= 2 else { return "--" }
        var s: Double = 0
        for i in 1..<rr.count { let d = Double(rr[i]-rr[i-1]); s += d*d }
        return String(format: "%.0f", sqrt(s/Double(rr.count-1)))
    }
    
    private func metric(_ title: String, value: String, unit: String, icon: String) -> some View {
        VStack(spacing: 3) {
            Image(systemName: icon).font(.system(size: 13)).foregroundColor(.secondary)
            HStack(alignment: .lastTextBaseline, spacing: 1) {
                Text(value).font(.system(size: 18, weight: .semibold, design: .rounded))
                if !unit.isEmpty { Text(unit).font(.system(.caption2, design: .monospaced)).foregroundColor(.secondary) }
            }
            Text(title).font(.system(.caption2, design: .monospaced)).foregroundColor(.secondary)
        }
        .frame(maxWidth: .infinity).padding(.vertical, 9)
        .background(Color(.secondarySystemGroupedBackground)).cornerRadius(9)
    }
    
    // MARK: - Pause/Resume
    
    private var trackingButton: some View {
        Button {
            if dataStore.isTracking {
                detector.flush()
                dataStore.pauseTracking()
            } else {
                dataStore.startTracking()
            }
        } label: {
            HStack(spacing: 8) {
                Image(systemName: dataStore.isTracking ? "pause.fill" : "play.fill")
                    .font(.title3)
                Text(dataStore.isTracking ? "Pause Tracking" : "Resume Tracking")
                    .font(.system(.title3, design: .rounded, weight: .bold))
            }
            .foregroundColor(.white)
            .frame(maxWidth: .infinity).padding(.vertical, 14)
            .background(RoundedRectangle(cornerRadius: 13).fill(dataStore.isTracking ? Color.orange : Color.green))
        }
        .buttonStyle(.plain)
        .disabled(!polar.isConnected)
    }
    
    // MARK: - Recording Info
    
    private var recordingInfo: some View {
        HStack(spacing: 5) {
            if dataStore.isTracking {
                Circle().fill(Color.red).frame(width: 6, height: 6)
                Text("REC").font(.system(.caption2, design: .monospaced, weight: .bold)).foregroundColor(.red)
                Text("\(dataStore.sampleCount) samples")
                    .font(.system(.caption2, design: .monospaced)).foregroundColor(.secondary)
                Spacer()
                Text(dataStore.totalDataSize())
                    .font(.system(.caption2, design: .monospaced)).foregroundColor(.secondary)
            } else {
                Text(polar.isConnected ? "Tracking paused" : "Connect device to start")
                    .font(.system(.caption2, design: .monospaced)).foregroundColor(.secondary)
            }
        }
        .padding(.horizontal, 10).padding(.vertical, 6)
        .background(Color(.secondarySystemGroupedBackground)).cornerRadius(7)
    }
}
