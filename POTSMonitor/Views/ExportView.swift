import SwiftUI

struct ExportView: View {
    @EnvironmentObject var dataStore: DataStore
    @StateObject private var predictor = POTSPredictor()
    
    @State private var showShare = false
    @State private var showClearConfirm = false
    @State private var showResetConfirm = false
    @State private var aggregatedFiles: [URL] = []
    
    var body: some View {
        NavigationStack {
            List {
                dataSummarySection
                recentFlareupSection
                csvFilesSection
                exportSection
                mlModelSection
                clearDataSection
            }
            .navigationTitle("Export & Train")
            .sheet(isPresented: $showShare) {
                ShareSheet(items: aggregatedFiles)
            }
            .alert("Clear All Data?", isPresented: $showClearConfirm) {
                Button("Clear", role: .destructive) { dataStore.clearAllData() }
                Button("Cancel", role: .cancel) {}
            } message: {
                Text("This deletes all CSV files and flareup records. The trained model is kept.")
            }
            .alert("Reset Model?", isPresented: $showResetConfirm) {
                Button("Reset", role: .destructive) { predictor.resetModel() }
                Button("Cancel", role: .cancel) {}
            } message: {
                Text("Removes the trained model. You'll need to retrain or load a new one.")
            }
        }
    }
    
    // MARK: - Sections
    
    private var dataSummarySection: some View {
        Section("Collected Data") {
            let samples = String(dataStore.sampleCount)
            let flareupCount = String(dataStore.detectedFlareups.count)
            let size = dataStore.totalDataSize()
            labelRow("HR Samples", samples)
            labelRow("Auto Flareups", flareupCount)
            labelRow("Storage", size)
        }
    }
    
    @ViewBuilder
    private var recentFlareupSection: some View {
        if !dataStore.detectedFlareups.isEmpty {
            Section("Recent Flareups (auto-detected)") {
                let recent = Array(dataStore.detectedFlareups.suffix(5).reversed())
                ForEach(recent) { f in
                    flareupRow(f)
                }
            }
        }
    }
    
    private var csvFilesSection: some View {
        Section("CSV Files") {
            let files = dataStore.exportFiles()
            if files.isEmpty {
                Text("No data yet")
                    .font(.caption)
                    .foregroundColor(.secondary)
            } else {
                ForEach(files, id: \.self) { file in
                    fileRow(file)
                }
            }
        }
    }
    
    private var exportSection: some View {
        Section {
            Button {
                aggregatedFiles = dataStore.exportAggregatedFiles()
                showShare = true
            } label: {
                Label("Export All Data", systemImage: "square.and.arrow.up")
            }
            .disabled(dataStore.exportFiles().isEmpty)
        } footer: {
            Text("Exports aggregated CSVs (hr.csv, acc.csv, ecg.csv, flareups.csv). AirDrop to Mac, save to Files, or email for external model training.")
        }
    }
    
    private var mlModelSection: some View {
            Section {
                modelStatusRows
                trainButton
                resetButton
                trainingStatusRow
            } header: {
                Text("ML Model")
            } footer: {
                Text("Trains a boosted tree on HR, HRV, and accelerometer features from the last 30 days. Uses auto-detected flareups (HR >120 for 30s+) as labels.")
            }
        }
        
        @ViewBuilder
        private var modelStatusRows: some View {
            if predictor.modelLoaded {
                labelRow("Status", "Trained")
                if let d = predictor.lastTrainedDate {
                    let dateStr = d.formatted(date: .abbreviated, time: .shortened)
                    labelRow("Trained", dateStr)
                }
                if let a = predictor.modelAccuracy {
                    let accStr = String(format: "%.1f%%", a * 100)
                    labelRow("Accuracy", accStr)
                }
            } else {
                Text("No model. Collect data with flareups, then train.")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
        }
        
        @ViewBuilder
        private var trainingStatusRow: some View {
            if !predictor.trainingStatus.isEmpty {
                Text(predictor.trainingStatus)
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
        }
    
    private var trainButton: some View {
        Button {
            Task { await predictor.trainOnRecentData(dataStore: dataStore) }
        } label: {
            HStack {
                if predictor.isTraining {
                    ProgressView()
                        .padding(.trailing, 3)
                }
                Image(systemName: "cpu")
                let title = predictor.modelLoaded ? "Retrain (last 30 days)" : "Train Model"
                Text(title)
            }
        }
        .disabled(predictor.isTraining || dataStore.detectedFlareups.isEmpty)
    }
    
    @ViewBuilder
    private var resetButton: some View {
        if predictor.modelLoaded {
            Button(role: .destructive) {
                showResetConfirm = true
            } label: {
                Label("Reset to Base Model", systemImage: "arrow.counterclockwise")
            }
        }
    }
    
    private var clearDataSection: some View {
        Section {
            Button(role: .destructive) {
                showClearConfirm = true
            } label: {
                Label("Clear All Stored Data", systemImage: "trash")
            }
        } footer: {
            Text("Deletes all CSV files and flareup history. Does not delete the trained model.")
        }
    }
    
    // MARK: - Helper Views
    
    private func labelRow(_ title: String, _ value: String) -> some View {
        HStack {
            Text(title)
            Spacer()
            Text(value)
                .foregroundColor(.secondary)
        }
    }
    
    private func flareupRow(_ f: DetectedFlareup) -> some View {
        let startText = f.start.formatted(date: .abbreviated, time: .shortened)
        let peakStr = String(f.peakHR)
        let durStr = String(f.durationSeconds)
        let detail = "Peak " + peakStr + " BPM · " + durStr + "s"
        
        return HStack {
            Image(systemName: "flame.fill")
                .foregroundColor(.red)
                .font(.caption)
            VStack(alignment: .leading, spacing: 1) {
                Text(startText)
                    .font(.body)
                Text(detail)
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
        }
    }
    
    private func fileRow(_ file: URL) -> some View {
        let name = file.lastPathComponent
        return HStack(spacing: 6) {
            Image(systemName: "doc.text")
                .foregroundColor(.blue)
                .font(.caption)
            Text(name)
                .font(.caption)
        }
    }
}

struct ShareSheet: UIViewControllerRepresentable {
    let items: [Any]
    func makeUIViewController(context: Context) -> UIActivityViewController {
        UIActivityViewController(activityItems: items, applicationActivities: nil)
    }
    func updateUIViewController(_ vc: UIActivityViewController, context: Context) {}
}
