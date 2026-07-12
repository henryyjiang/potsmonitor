import SwiftUI

@main
struct POTSMonitorApp: App {
    @StateObject private var polar = PolarManager()
    @StateObject private var dataStore = DataStore()
    @StateObject private var detector = FlareupDetector()
    @StateObject private var notifications = NotificationManager()
    @StateObject private var tracker = PredictionTracker()
    @StateObject private var predictor = POTSPredictor()

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(polar)
                .environmentObject(dataStore)
                .environmentObject(detector)
                .environmentObject(notifications)
                .environmentObject(tracker)
                .environmentObject(predictor)
                .onAppear { wireUp() }
        }
    }

    private func wireUp() {
        notifications.requestAuthorization()

        polar.onHRSample = { sample in
            dataStore.logHR(sample)
            detector.processHR(sample)

            // Feed the predictor the same live stream and act on new predictions
            // here (not in a view) so warnings fire regardless of the active tab.
            let produced = predictor.ingestAndPredict(
                hr: sample,
                baselineHR: detector.currentBaseline,
                baselineRMSSD: detector.currentBaselineRMSSD)
            if produced {
                let p = predictor.lastPrediction
                if p >= NotificationManager.predictionThreshold {
                    notifications.sendFlareupPredictedNotification(probability: p)
                    tracker.recordPrediction(probability: p)
                }
                tracker.resolvePending()
            }
        }
        polar.onAccSample = { sample in
            dataStore.logAcc(sample)
            predictor.ingestAcc(sample)   // live ACC now reaches the model (was missing)
        }
        polar.onECGSample = { sample in
            dataStore.logECG(sample)
        }

        detector.onFlareupDetected = { flareup in
            dataStore.recordFlareup(flareup)
            tracker.processFlareup(detectedAt: flareup.start)
        }
        detector.onFlareupConfirmed = { peakHR, baseline in
            notifications.sendFlareupDetectedNotification(peakHR: peakHR, baseline: baseline)
        }
    }
}
