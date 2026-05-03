import Foundation
import Combine
import UserNotifications

@MainActor
class NotificationManager: NSObject, ObservableObject, UNUserNotificationCenterDelegate {

    private var lastDetectedNotification: Date?
    private var lastPredictedNotification: Date?

    static let detectedCooldown: TimeInterval = 5 * 60
    static let predictedCooldown: TimeInterval = 10 * 60
    static let predictionThreshold: Double = 0.7

    override init() {
        super.init()
        UNUserNotificationCenter.current().delegate = self
    }

    func requestAuthorization() {
        UNUserNotificationCenter.current().requestAuthorization(options: [.alert, .sound, .badge]) { granted, error in
            if let error = error {
                print("[Notifications] Authorization error: \(error)")
            }
        }
    }

    // MARK: - Detected Flareup

    func sendFlareupDetectedNotification(peakHR: Int, baseline: Double) {
        guard shouldSend(last: lastDetectedNotification, cooldown: Self.detectedCooldown) else { return }
        lastDetectedNotification = Date()

        let content = UNMutableNotificationContent()
        content.title = "Flareup Detected"
        content.body = "HR peaked at \(peakHR) BPM (+\(Int(Double(peakHR) - baseline)) over baseline)"
        content.sound = .default
        content.interruptionLevel = .timeSensitive

        let request = UNNotificationRequest(
            identifier: "flareup-detected-\(UUID().uuidString)",
            content: content,
            trigger: nil
        )
        UNUserNotificationCenter.current().add(request)
    }

    // MARK: - Predicted Flareup

    func sendFlareupPredictedNotification(probability: Double) {
        guard probability >= Self.predictionThreshold else { return }
        guard shouldSend(last: lastPredictedNotification, cooldown: Self.predictedCooldown) else { return }
        lastPredictedNotification = Date()

        let content = UNMutableNotificationContent()
        content.title = "Flareup Warning"
        content.body = "ML model predicts \(Int(probability * 100))% chance of flareup soon"
        content.sound = .default
        content.interruptionLevel = .active

        let request = UNNotificationRequest(
            identifier: "flareup-predicted-\(UUID().uuidString)",
            content: content,
            trigger: nil
        )
        UNUserNotificationCenter.current().add(request)
    }

    // MARK: - Foreground Delivery

    nonisolated func userNotificationCenter(
        _ center: UNUserNotificationCenter,
        willPresent notification: UNNotification,
        withCompletionHandler completionHandler: @escaping (UNNotificationPresentationOptions) -> Void
    ) {
        completionHandler([.banner, .sound])
    }

    // MARK: - Private

    private func shouldSend(last: Date?, cooldown: TimeInterval) -> Bool {
        guard let last = last else { return true }
        return Date().timeIntervalSince(last) >= cooldown
    }
}
