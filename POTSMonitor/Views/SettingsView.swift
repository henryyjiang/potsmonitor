import SwiftUI

struct SettingsView: View {
    @EnvironmentObject var polar: PolarManager
    @EnvironmentObject var dataStore: DataStore
    
    var body: some View {
        NavigationStack {
            List {
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
                
                Section("Flareup Detection") {
                    LabeledContent("Threshold", value: "+30 BPM over baseline")
                    LabeledContent("Duration", value: "60 seconds")
                    Text("Automatically records a flareup when heart rate rises 30+ BPM above the rolling 30-minute baseline for 60+ consecutive seconds. If HRV data is available, an HRV collapse must also be confirmed.")
                        .font(.caption).foregroundColor(.secondary)
                }
                
                if let err = polar.errorMessage {
                    Section("Status") {
                        Text(err).font(.caption).foregroundColor(.red)
                    }
                }
                
                Section {} footer: {
                    VStack(alignment: .leading, spacing: 2) {
                        Text("POTS Monitor v1.0")
                        Text("Streams HR, RR, accelerometer, temperature from Polar Loop via BLE.")
                        Text("Auto-detects flareups. Trains on-device ML model.")
                        Text("Data: Documents/POTSData (accessible via Files app)")
                    }.font(.caption2)
                }
            }
            .navigationTitle("Settings")
        }
    }
}
