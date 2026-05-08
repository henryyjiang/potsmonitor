import Foundation
import Combine

@MainActor
class DataStore: ObservableObject {
    
    @Published var isTracking = false
    @Published var sampleCount: Int = 0
    @Published var detectedFlareups: [DetectedFlareup] = []
    @Published var trackingStartTime: Date?
    
    private var hrHandle: FileHandle?
    private var accHandle: FileHandle?
    private var ecgHandle: FileHandle?
    private var currentDateStr: String = ""
    private let fm = FileManager.default
    
    private let tsF: DateFormatter = {
        let f = DateFormatter(); f.dateFormat = "yyyy-MM-dd'T'HH:mm:ss.SSS"; f.timeZone = .current; return f
    }()
    private let dayF: DateFormatter = {
        let f = DateFormatter(); f.dateFormat = "yyyy-MM-dd"; return f
    }()
    
    var dataDir: URL {
        let d = fm.urls(for: .documentDirectory, in: .userDomainMask)[0].appendingPathComponent("POTSData")
        try? fm.createDirectory(at: d, withIntermediateDirectories: true)
        return d
    }
    
    static let retentionDays = 30

    init() {
        loadFlareups()
        maintainStorage()
    }
    
    // MARK: - Tracking (pause/resume)
    
    func startTracking() {
        guard !isTracking else { return }
        isTracking = true
        trackingStartTime = Date()
        sampleCount = 0
        openFiles()
    }
    
    func pauseTracking() {
        isTracking = false
        closeFiles()
    }
    
    // MARK: - Logging
    
    func logHR(_ s: HRSample) {
        guard isTracking else { return }
        rollIfNeeded()
        let ts = tsF.string(from: s.timestamp)
        let rr = s.rrIntervals.map(String.init).joined(separator: ";")
        let line = "\(ts),\(s.hr),\"\(rr)\",\(s.contactStatus ? 1 : 0)\n"
        if let d = line.data(using: .utf8) { hrHandle?.write(d); DispatchQueue.main.async { self.sampleCount += 1 } }
    }
    
    func logAcc(_ s: AccSample) {
        guard isTracking else { return }
        rollIfNeeded()
        let line = "\(tsF.string(from: s.timestamp)),\(s.x),\(s.y),\(s.z)\n"
        if let d = line.data(using: .utf8) { accHandle?.write(d) }
    }
    
    func logECG(_ s: ECGSample) {
        guard isTracking else { return }
        rollIfNeeded()
        // Each batch has multiple microVolt samples at 130Hz
        // Write one row per batch with timestamp and semicolon-separated values
        let ts = tsF.string(from: s.timestamp)
        let values = s.microVolts.map(String.init).joined(separator: ";")
        let line = "\(ts),\(s.sampleRate),\"\(values)\"\n"
        if let d = line.data(using: .utf8) { ecgHandle?.write(d) }
    }
    
    func logTemp(_ s: TemperatureSample) {
        guard isTracking else { return }
        let path = dataDir.appendingPathComponent("temp_\(today()).csv")
        ensureFile(path, header: "timestamp,skin_temp_c\n")
        let line = "\(tsF.string(from: s.timestamp)),\(String(format: "%.2f", s.celsius))\n"
        if let h = try? FileHandle(forWritingTo: path) { h.seekToEndOfFile(); h.write(line.data(using: .utf8)!); h.closeFile() }
    }
    
    // MARK: - Auto-detected flareups
    
    func recordFlareup(_ f: DetectedFlareup) {
        DispatchQueue.main.async { self.detectedFlareups.append(f) }
        
        // Write to flareups CSV
        let path = dataDir.appendingPathComponent("flareups.csv")
        ensureFile(path, header: "start,end,peak_hr,duration_s,baseline_hr,threshold_used,hrv_confirmed\n")
        let startStr = tsF.string(from: f.start)
        let endStr = f.end.map { tsF.string(from: $0) } ?? ""
        let baseStr = f.baselineHR.map { String(format: "%.1f", $0) } ?? ""
        let threshStr = f.thresholdUsed.map { String(format: "%.1f", $0) } ?? ""
        let hrvStr = f.hrvConfirmed.map { $0 ? "1" : "0" } ?? ""
        let line = "\(startStr),\(endStr),\(f.peakHR),\(f.durationSeconds),\(baseStr),\(threshStr),\(hrvStr)\n"
        if let h = try? FileHandle(forWritingTo: path) { h.seekToEndOfFile(); h.write(line.data(using: .utf8)!); h.closeFile() }
        
        saveFlareups()
    }
    
    // MARK: - Clear Data
    
    func clearAllData() {
        closeFiles()
        if let files = try? fm.contentsOfDirectory(at: dataDir, includingPropertiesForKeys: nil) {
            for f in files { try? fm.removeItem(at: f) }
        }
        detectedFlareups = []
        sampleCount = 0
        UserDefaults.standard.removeObject(forKey: "detected_flareups")
        if isTracking { openFiles() }
    }
    
    // MARK: - Export
    
    func exportFiles() -> [URL] {
        (try? fm.contentsOfDirectory(at: dataDir, includingPropertiesForKeys: nil))?
            .filter { $0.pathExtension == "csv" || $0.pathExtension == "zlib" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent } ?? []
    }

    func readCSVContent(_ url: URL) -> String? {
        if url.pathExtension == "zlib" {
            guard let compressed = try? Data(contentsOf: url),
                  let decompressed = try? (compressed as NSData).decompressed(using: .zlib) as Data else { return nil }
            return String(data: decompressed, encoding: .utf8)
        }
        return try? String(contentsOf: url, encoding: .utf8)
    }
    
    func totalDataSize() -> String {
        var total: UInt64 = 0
        if let files = try? fm.contentsOfDirectory(at: dataDir, includingPropertiesForKeys: [.fileSizeKey]) {
            for f in files { total += UInt64((try? f.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0) }
        }
        return total > 1_000_000
            ? String(format: "%.1f MB", Double(total) / 1_000_000)
            : String(format: "%.0f KB", Double(total) / 1_000)
    }
    
    // MARK: - File Ops
    
    private func today() -> String { dayF.string(from: Date()) }
    
    private func openFiles() {
        currentDateStr = today()
        let hrPath = dataDir.appendingPathComponent("hr_\(currentDateStr).csv")
        ensureFile(hrPath, header: "timestamp,hr_bpm,rr_intervals_ms,contact\n")
        hrHandle = try? FileHandle(forWritingTo: hrPath); hrHandle?.seekToEndOfFile()
        
        let accPath = dataDir.appendingPathComponent("acc_\(currentDateStr).csv")
        ensureFile(accPath, header: "timestamp,x_mG,y_mG,z_mG\n")
        accHandle = try? FileHandle(forWritingTo: accPath); accHandle?.seekToEndOfFile()
        
        let ecgPath = dataDir.appendingPathComponent("ecg_\(currentDateStr).csv")
        ensureFile(ecgPath, header: "timestamp,sample_rate_hz,micro_volts\n")
        ecgHandle = try? FileHandle(forWritingTo: ecgPath); ecgHandle?.seekToEndOfFile()
    }
    
    private func closeFiles() {
        hrHandle?.closeFile(); accHandle?.closeFile(); ecgHandle?.closeFile()
        hrHandle = nil; accHandle = nil; ecgHandle = nil
    }
    
    private func rollIfNeeded() {
        let t = today()
        if t != currentDateStr {
            closeFiles()
            openFiles()
            maintainStorage()
        }
    }
    
    private func ensureFile(_ path: URL, header: String) {
        if !fm.fileExists(atPath: path.path) {
            fm.createFile(atPath: path.path, contents: header.data(using: .utf8))
        }
    }
    
    // MARK: - Storage Maintenance

    func maintainStorage() {
        let dir = dataDir
        let todayStr = today()
        let cutoffDate = Calendar.current.date(byAdding: .day, value: -Self.retentionDays, to: Date()) ?? Date()
        let cutoffStr = dayF.string(from: cutoffDate)
        let shouldStore = UserDefaults.standard.object(forKey: "storeDailyData") as? Bool ?? true

        DispatchQueue.global(qos: .utility).async { [fm] in
            if shouldStore {
                Self.compressOldFiles(in: dir, today: todayStr, fm: fm)
                Self.purgeOldFiles(in: dir, cutoff: cutoffStr, fm: fm)
            } else {
                Self.deleteOldFiles(in: dir, today: todayStr, fm: fm)
            }
        }
    }

    private static func compressOldFiles(in dir: URL, today: String, fm: FileManager) {
        guard let files = try? fm.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil) else { return }

        let dailyCSVs = files.filter { url in
            let name = url.lastPathComponent
            return url.pathExtension == "csv" && name != "flareups.csv" && !name.contains(today)
        }

        for file in dailyCSVs {
            guard let data = try? Data(contentsOf: file),
                  let compressed = try? (data as NSData).compressed(using: .zlib) as Data else { continue }
            let dest = file.appendingPathExtension("zlib")
            do {
                try compressed.write(to: dest)
                try fm.removeItem(at: file)
            } catch {
                try? fm.removeItem(at: dest)
            }
        }
    }

    private static func purgeOldFiles(in dir: URL, cutoff: String, fm: FileManager) {
        guard let files = try? fm.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil) else { return }

        for file in files {
            let name = file.lastPathComponent
            guard name != "flareups.csv" else { continue }
            guard let dateStr = extractDate(from: name) else { continue }
            if dateStr < cutoff {
                try? fm.removeItem(at: file)
            }
        }
    }

    private static func deleteOldFiles(in dir: URL, today: String, fm: FileManager) {
        guard let files = try? fm.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil) else { return }
        for file in files {
            let name = file.lastPathComponent
            guard name != "flareups.csv" else { continue }
            if let dateStr = extractDate(from: name), dateStr < today {
                try? fm.removeItem(at: file)
            }
        }
    }

    private static func extractDate(from filename: String) -> String? {
        let parts = filename.split(separator: "_", maxSplits: 1)
        guard parts.count == 2 else { return nil }
        let datePart = parts[1].prefix(10)
        guard datePart.count == 10 else { return nil }
        return String(datePart)
    }

    // MARK: - Persistence
    
    private func saveFlareups() {
        if let d = try? JSONEncoder().encode(detectedFlareups) {
            UserDefaults.standard.set(d, forKey: "detected_flareups")
        }
    }
    private func loadFlareups() {
        if let d = UserDefaults.standard.data(forKey: "detected_flareups"),
           let loaded = try? JSONDecoder().decode([DetectedFlareup].self, from: d) {
            detectedFlareups = loaded
        }
    }
}
