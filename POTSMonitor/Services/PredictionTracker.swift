import Foundation

@MainActor
class PredictionTracker: ObservableObject {

    static let resolutionWindow: TimeInterval = 15 * 60

    @Published private(set) var records: [PredictionRecord] = []
    @Published private(set) var missedFlareupDates: [Date] = []
    @Published private(set) var trackedSince: Date?

    // MARK: - Computed Stats

    var tp: Int { records.filter { $0.outcome == .truePositive  }.count }
    var fp: Int { records.filter { $0.outcome == .falsePositive }.count }
    var fn: Int { missedFlareupDates.count }

    var tn: Int {
        guard let since = trackedSince else { return 0 }
        let totalDays = max(0, Int(Date().timeIntervalSince(since) / 86400))
        let calendar = Calendar.current
        var eventDays = Set<Int>()
        for r in records where r.outcome != .pending {
            eventDays.insert(calendar.ordinality(of: .day, in: .era, for: r.timestamp) ?? 0)
        }
        for d in missedFlareupDates {
            eventDays.insert(calendar.ordinality(of: .day, in: .era, for: d) ?? 0)
        }
        return max(0, totalDays - eventDays.count)
    }

    var precision: Double? {
        let d = tp + fp; return d > 0 ? Double(tp) / Double(d) : nil
    }
    var recall: Double? {
        let d = tp + fn; return d > 0 ? Double(tp) / Double(d) : nil
    }
    var f1: Double? {
        guard let p = precision, let r = recall, p + r > 0 else { return nil }
        return 2 * p * r / (p + r)
    }

    init() { load() }

    // MARK: - Events

    func recordPrediction(probability: Double) {
        // Don't stack predictions while one is still pending resolution
        guard !records.contains(where: { $0.outcome == .pending }) else { return }
        if trackedSince == nil { trackedSince = Date() }
        records.append(PredictionRecord(id: UUID(), timestamp: Date(), probability: probability, outcome: .pending))
        save()
    }

    func processFlareup(detectedAt date: Date) {
        if trackedSince == nil { trackedSince = date }
        // Find the most recent pending prediction that covers this flareup (within 15 min prior)
        let idx = records.indices.last {
            records[$0].outcome == .pending
            && date.timeIntervalSince(records[$0].timestamp) >= 0
            && date.timeIntervalSince(records[$0].timestamp) <= Self.resolutionWindow
        }
        if let i = idx {
            records[i].outcome = .truePositive
        } else {
            missedFlareupDates.append(date)
        }
        resolvePending()
        save()
    }

    func resolvePending() {
        let now = Date()
        for i in records.indices where records[i].outcome == .pending {
            if now.timeIntervalSince(records[i].timestamp) >= Self.resolutionWindow {
                records[i].outcome = .falsePositive
            }
        }
        save()
    }

    func clearStats() {
        records = []
        missedFlareupDates = []
        trackedSince = nil
        UserDefaults.standard.removeObject(forKey: "prediction_records")
        UserDefaults.standard.removeObject(forKey: "missed_flareup_dates")
        UserDefaults.standard.removeObject(forKey: "prediction_tracked_since")
    }

    // MARK: - Persistence

    private func save() {
        if let d = try? JSONEncoder().encode(records) {
            UserDefaults.standard.set(d, forKey: "prediction_records")
        }
        if let d = try? JSONEncoder().encode(missedFlareupDates) {
            UserDefaults.standard.set(d, forKey: "missed_flareup_dates")
        }
        if let since = trackedSince {
            UserDefaults.standard.set(since.timeIntervalSince1970, forKey: "prediction_tracked_since")
        }
    }

    private func load() {
        if let d = UserDefaults.standard.data(forKey: "prediction_records"),
           let v = try? JSONDecoder().decode([PredictionRecord].self, from: d) {
            records = v
        }
        if let d = UserDefaults.standard.data(forKey: "missed_flareup_dates"),
           let v = try? JSONDecoder().decode([Date].self, from: d) {
            missedFlareupDates = v
        }
        let t = UserDefaults.standard.double(forKey: "prediction_tracked_since")
        if t > 0 { trackedSince = Date(timeIntervalSince1970: t) }
        resolvePending()
    }
}
