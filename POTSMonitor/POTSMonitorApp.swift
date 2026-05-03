import SwiftUI

@main
struct POTSMonitorApp: App {
    @StateObject private var polar = PolarManager()
    @StateObject private var dataStore = DataStore()
    @StateObject private var detector = FlareupDetector()
    @StateObject private var notifications = NotificationManager()
    
    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(polar)
                .environmentObject(dataStore)
                .environmentObject(detector)
                .environmentObject(notifications)
                .onAppear { wireUp() }
        }
    }
    
    private func wireUp() {
        notifications.requestAuthorization()

        // Sensor data → DataStore logging + FlareupDetector
        polar.onHRSample = { sample in
            dataStore.logHR(sample)
            detector.processHR(sample)
        }
        polar.onAccSample = { sample in
            dataStore.logAcc(sample)
        }
        polar.onECGSample = { sample in
            dataStore.logECG(sample)
        }

        // FlareupDetector → DataStore recording + notification
        detector.onFlareupDetected = { flareup in
            dataStore.recordFlareup(flareup)
        }
        detector.onFlareupConfirmed = { peakHR, baseline in
            notifications.sendFlareupDetectedNotification(peakHR: peakHR, baseline: baseline)
        }
    }
}
