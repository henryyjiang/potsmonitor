import SwiftUI

struct SettingsView: View {
    @EnvironmentObject var polar: PolarManager
    @EnvironmentObject var dataStore: DataStore

    @AppStorage("storeDailyData") private var storeDailyData = true
    @State private var showClearConfirm = false

    var body: some View {
        NavigationStack {
            List {
                polarSection
                flareupDetectionSection
                dataStorageSection
                if let err = polar.errorMessage {
                    Section("Status") {
                        Text(err).font(.caption).foregroundColor(.red)
                    }
                }
                footerSection
            }
            .navigationTitle("Settings")
            .alert("Clear All Data?", isPresented: $showClearConfirm) {
                Button("Clear", role: .destructive) { dataStore.clearAllData() }
                Button("Cancel", role: .cancel) {}
            } message: {
                Text("Deletes all CSV sensor files and flareup records. The trained ML model is not affected.")
            }
        }
    }

    // MARK: - Sections

    private var polarSection: some View {
        Section("Polar Loop") {
            if polar.isConnected {
                LabeledContent("Device", value: polar.deviceName)
                LabeledContent("ID", value: polar.deviceId)
                LabeledContent("Battery", value: "\(polar.batteryLevel)%")
                LabeledContent("Streaming", value: polar.isStreaming ? "Active" : "Idle")

                Button("Disconnect", role: .destructive) {
                    dataStore.pauseTracking()
                    polar.disconnect()
                }
            } else {
                Button {
                    polar.searchForDevices()
                } label: {
                    HStack {
                        Image(systemName: "antenna.radiowaves.left.and.right")
                        Text("Search for Polar Loop")
                    }
                }

                if polar.isSearching {
                    HStack(spacing: 6) {
                        ProgressView()
                        Text("Scanning...").font(.caption).foregroundColor(.secondary)
                    }
                }

                ForEach(polar.availableDevices, id: \.deviceId) { dev in
                    Button {
                        polar.stopSearch()
                        polar.connectToDevice(dev.deviceId)
                        dataStore.startTracking()
                    } label: {
                        HStack(spacing: 8) {
                            Image(systemName: "wave.3.right.circle.fill").foregroundColor(.blue)
                            VStack(alignment: .leading, spacing: 1) {
                                Text(dev.name).font(.system(.body, weight: .medium))
                                Text(dev.deviceId)
                                    .font(.system(.caption, design: .monospaced))
                                    .foregroundColor(.secondary)
                            }
                        }
                    }
                }
            }
        }
    }

    private var flareupDetectionSection: some View {
        Section("Flareup Detection") {
            LabeledContent("Threshold", value: "+30 BPM over baseline")
            LabeledContent("Duration", value: "60 seconds")
            Text("Automatically records a flareup when heart rate rises 30+ BPM above the rolling 30-minute baseline for 60+ consecutive seconds.")
                .font(.caption).foregroundColor(.secondary)
        }
    }

    private var dataStorageSection: some View {
        Section {
            Toggle("Save Daily Data", isOn: $storeDailyData)
                .onChange(of: storeDailyData) { _ in
                    dataStore.maintainStorage()
                }
            Text(storeDailyData
                 ? "Sensor data is compressed and kept for 30 days. Recommended if you plan to retrain the model."
                 : "Yesterday's sensor data is deleted at midnight. Only today's data is kept. The ML model is unaffected.")
                .font(.caption).foregroundColor(.secondary)

            LabeledContent("Storage Used", value: dataStore.totalDataSize())

            Button("Clear Collected Data", role: .destructive) {
                showClearConfirm = true
            }
        } header: {
            Text("Data Storage")
        } footer: {
            Text("The ML model is bundled in the app — you can safely clear collected data without affecting predictions.")
        }
    }

    @ViewBuilder
    private var footerSection: some View {
        Section {} footer: {
            VStack(alignment: .leading, spacing: 2) {
                Text("POTS Monitor v1.0")
                Text("Streams HR, RR, accelerometer from Polar Loop via BLE.")
                Text("ML model predicts flareups 15 minutes in advance.")
                Text("Sensor data: On My iPhone › POTSMonitor › POTSData")
            }.font(.caption2)
        }
    }
}
