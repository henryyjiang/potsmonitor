import SwiftUI

struct ExportView: View {
    @EnvironmentObject var dataStore: DataStore
    @StateObject private var predictor = POTSPredictor()

    @State private var showShare = false
    @State private var showClearConfirm = false
    @State private var showResetConfirm = false
    @State private var aggregatedFiles: [URL] = []
    @State private var isExporting = false

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
                Text("Deletes all CSV files and flareup records. The trained model is kept.")
            }
            .alert("Reset Model?", isPresented: $showResetConfirm) {
                Button("Reset", role: .destructive) { predictor.resetModel() }
                Button("Cancel", role: .cancel) {}
            } message: {
                Text("Removes any user-trained model and reverts to the bundled model.")
            }
        }
    }

    // MARK: - Sections

    private var dataSummarySection: some View {
        Section("Collected Data") {
            labelRow("HR Samples", String(dataStore.sampleCount))
            labelRow("Auto Flareups", String(dataStore.detectedFlareups.count))
            labelRow("Storage", dataStore.totalDataSize())
        }
    }

    @ViewBuilder
    private var recentFlareupSection: some View {
        if !dataStore.detectedFlareups.isEmpty {
            Section("Recent Flareups (auto-detected)") {
                ForEach(Array(dataStore.detectedFlareups.suffix(5).reversed())) { f in
                    flareupRow(f)
                }
            }
        }
    }

    private var csvFilesSection: some View {
        Section("CSV Files") {
            let files = dataStore.exportFiles()
            if files.isEmpty {
                Text("No data yet").font(.caption).foregroundColor(.secondary)
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
                Task {
                    isExporting = true
                    aggregatedFiles = await dataStore.exportAggregatedFiles()
                    isExporting = false
                    showShare = true
                }
            } label: {
                if isExporting {
                    HStack { ProgressView().padding(.trailing, 3); Text("Preparing Export…") }
                } else {
                    Label("Export All Data", systemImage: "square.and.arrow.up")
                }
            }
            .disabled(dataStore.exportFiles().isEmpty || isExporting)
        } footer: {
            Text("Aggregates all daily CSVs into hr.csv, acc.csv, ecg.csv, flareups.csv. AirDrop to Mac or save to Files for external model training.")
        }
    }

    private var mlModelSection: some View {
        Section {
            modelStatusRows
            trainButton
            if predictor.modelLoaded { resetButton }
            trainingStatusRow
        } header: {
            Text("ML Model")
        } footer: {
            Text("Trains a boosted tree on HR, HRV, and accelerometer features from the last 30 days, labelled using auto-detected flareups. Predictions fire 15 minutes before a flareup.")
        }
    }

    @ViewBuilder
    private var modelStatusRows: some View {
        if predictor.modelLoaded {
            labelRow("Status", "Loaded")
            if let d = predictor.lastTrainedDate {
                labelRow("Trained", d.formatted(date: .abbreviated, time: .shortened))
            }
            if let a = predictor.modelAccuracy {
                labelRow("Accuracy", String(format: "%.1f%%", a * 100))
            }
        } else {
            Text("No model loaded.").font(.caption).foregroundColor(.secondary)
        }
    }

    private var trainButton: some View {
        Button {
            Task { await predictor.trainOnRecentData(dataStore: dataStore) }
        } label: {
            HStack {
                if predictor.isTraining { ProgressView().padding(.trailing, 3) }
                Image(systemName: "cpu")
                Text(predictor.modelLoaded ? "Retrain (last 30 days)" : "Train Model")
            }
        }
        .disabled(predictor.isTraining || dataStore.detectedFlareups.isEmpty)
    }

    private var resetButton: some View {
        Button(role: .destructive) { showResetConfirm = true } label: {
            Label("Reset to Bundled Model", systemImage: "arrow.counterclockwise")
        }
    }

    @ViewBuilder
    private var trainingStatusRow: some View {
        if !predictor.trainingStatus.isEmpty {
            Text(predictor.trainingStatus).font(.caption).foregroundColor(.secondary)
        }
    }

    private var clearDataSection: some View {
        Section {
            Button(role: .destructive) { showClearConfirm = true } label: {
                Label("Clear All Stored Data", systemImage: "trash")
            }
        } footer: {
            Text("Deletes all CSV files and flareup history. The trained ML model is not affected.")
        }
    }

    // MARK: - Helper Views

    private func labelRow(_ title: String, _ value: String) -> some View {
        HStack {
            Text(title); Spacer()
            Text(value).foregroundColor(.secondary)
        }
    }

    private func flareupRow(_ f: DetectedFlareup) -> some View {
        HStack {
            Image(systemName: "flame.fill").foregroundColor(.red).font(.caption)
            VStack(alignment: .leading, spacing: 1) {
                Text(f.start.formatted(date: .abbreviated, time: .shortened))
                Text("Peak \(f.peakHR) BPM · \(f.durationSeconds)s")
                    .font(.caption).foregroundColor(.secondary)
            }
        }
    }

    private func fileRow(_ file: URL) -> some View {
        HStack(spacing: 6) {
            Image(systemName: "doc.text").foregroundColor(.blue).font(.caption)
            Text(file.lastPathComponent).font(.caption)
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
