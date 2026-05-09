import Foundation
import Combine
import Compression

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

    func exportAggregatedFiles() async -> [URL] {
        let allFiles = exportFiles()
        let tempDir = fm.temporaryDirectory.appendingPathComponent("POTSExport-\(UUID().uuidString)")
        try? fm.createDirectory(at: tempDir, withIntermediateDirectories: true)
        let localFM = fm

        return await Task.detached(priority: .userInitiated) {
            let groups: [(prefix: String, output: String)] = [
                ("hr_",   "hr.csv"),
                ("acc_",  "acc.csv"),
                ("ecg_",  "ecg.csv"),
                ("temp_", "temp.csv"),
            ]
            var result: [URL] = []
            for group in groups {
                let matching = allFiles
                    .filter { $0.lastPathComponent.hasPrefix(group.prefix) }
                    .sorted { $0.lastPathComponent < $1.lastPathComponent }
                if let url = DataStore.streamAggregateCSVs(matching, to: tempDir.appendingPathComponent(group.output)) {
                    result.append(url)
                }
            }
            if let flareupFile = allFiles.first(where: { $0.lastPathComponent == "flareups.csv" }) {
                let dest = tempDir.appendingPathComponent("flareups.csv")
                try? localFM.copyItem(at: flareupFile, to: dest)
                result.append(dest)
            }
            return result
        }.value
    }

    private static func streamAggregateCSVs(_ files: [URL], to destination: URL) -> URL? {
        guard !files.isEmpty else { return nil }
        FileManager.default.createFile(atPath: destination.path, contents: nil)
        guard let outHandle = try? FileHandle(forWritingTo: destination) else { return nil }
        defer { outHandle.closeFile() }
        for (i, file) in files.enumerated() {
            DataStore.streamCSVLines(from: file, to: outHandle, skipHeader: i > 0)
        }
        let size = (try? destination.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0
        return size > 0 ? destination : nil
    }

    // Streams a CSV (plain or .zlib) line-by-line into outHandle using the Compression
    // framework. Peak RAM is ~350 KB regardless of file size — no full decompression into memory.
    private static func streamCSVLines(from url: URL, to outHandle: FileHandle, skipHeader: Bool) {
        let nlByte = UInt8(ascii: "\n")
        var carry = Data()
        var headerSeen = false

        let emit: (Data) -> Void = { raw in
            var buf = carry + raw
            carry = Data()
            while let idx = buf.firstIndex(of: nlByte) {
                let end = buf.index(after: idx)
                if skipHeader && !headerSeen { headerSeen = true; buf.removeSubrange(..<end); continue }
                headerSeen = true
                outHandle.write(Data(buf[..<end]))
                buf.removeSubrange(..<end)
            }
            carry = Data(buf)
        }

        guard let inHandle = try? FileHandle(forReadingFrom: url) else { return }
        defer { inHandle.closeFile() }

        if url.pathExtension != "zlib" {
            while case let c = inHandle.readData(ofLength: 65_536), !c.isEmpty { emit(c) }
            if !carry.isEmpty { outHandle.write(carry) }
            return
        }

        // Streaming zlib decompression — C-allocated buffers have a fixed address,
        // which is required for compression_stream's src_ptr/dst_ptr.
        let inN = 65_536, outN = 262_144
        let srcBuf = UnsafeMutablePointer<UInt8>.allocate(capacity: inN)
        let dstBuf = UnsafeMutablePointer<UInt8>.allocate(capacity: outN)
        defer { srcBuf.deallocate(); dstBuf.deallocate() }

        var stream = compression_stream()
        guard compression_stream_init(&stream, COMPRESSION_STREAM_DECODE, COMPRESSION_ZLIB) == COMPRESSION_STATUS_OK else { return }
        defer { compression_stream_destroy(&stream) }

        var done = false
        while !done {
            let chunk = inHandle.readData(ofLength: inN)
            let isLast = chunk.count < inN
            let flags: Int32 = isLast ? Int32(COMPRESSION_STREAM_FINALIZE.rawValue) : 0
            chunk.copyBytes(to: srcBuf, count: chunk.count)
            stream.src_ptr = srcBuf
            stream.src_size = chunk.count

            var innerDone = false
            while !innerDone {
                stream.dst_ptr = dstBuf
                stream.dst_size = outN
                let status = compression_stream_process(&stream, flags)
                let produced = outN - stream.dst_size
                if produced > 0 { emit(Data(bytes: dstBuf, count: produced)) }
                switch status {
                case COMPRESSION_STATUS_END, COMPRESSION_STATUS_ERROR:
                    done = true; innerDone = true
                default:
                    innerDone = stream.src_size == 0 && stream.dst_size > 0
                }
            }
            if isLast { done = true }
        }
        if !carry.isEmpty { outHandle.write(carry) }
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
