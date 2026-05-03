import Foundation
import PolarBleSdk
import RxSwift
import CoreBluetooth
import Combine

@MainActor
class PolarManager: ObservableObject {
    
    @Published var isConnected = false
    @Published var isSearching = false
    @Published var isStreaming = false
    @Published var deviceId: String = ""
    @Published var deviceName: String = ""
    @Published var batteryLevel: UInt = 0
    @Published var currentHR: Int = 0
    @Published var currentRR: [Int] = []
    @Published var connectionStatus: String = "Disconnected"
    @Published var errorMessage: String?
    @Published var availableDevices: [PolarDeviceInfo] = []
    
    var onHRSample: ((HRSample) -> Void)?
    var onAccSample: ((AccSample) -> Void)?
    var onECGSample: ((ECGSample) -> Void)?
    
    private var api: PolarBleApi!
    private let disposeBag = DisposeBag()
    private var searchDisposable: Disposable?
    private var hrDisposable: Disposable?
    private var ppiDisposable: Disposable?
    private var ecgDisposable: Disposable?
    
    init() {
        // SDK 5.x feature set — no temperature in this version
        api = PolarBleApiDefaultImpl.polarImplementation(
            DispatchQueue.main,
            features: [
                .feature_hr,
                .feature_battery_info,
                .feature_device_info,
                .feature_polar_sdk_mode,
                .feature_polar_online_streaming
            ]
        )
        api.observer = self
        api.deviceInfoObserver = self
        api.powerStateObserver = self
        
        if let saved = UserDefaults.standard.string(forKey: "polar_device_id"), !saved.isEmpty {
            deviceId = saved
            DispatchQueue.main.asyncAfter(deadline: .now() + 1.0) {
                self.connectToDevice(saved)
            }
        }
    }
    
    // MARK: - Scanning
    
    func searchForDevices() {
        isSearching = true
        availableDevices = []
        connectionStatus = "Scanning..."
        
        searchDisposable = api.searchForDevice()
            .observe(on: MainScheduler.instance)
            .subscribe(
                onNext: { [weak self] info in
                    guard let self = self else { return }
                    if !self.availableDevices.contains(where: { $0.deviceId == info.deviceId }) {
                        self.availableDevices.append(info)
                    }
                },
                onError: { [weak self] error in
                    self?.errorMessage = "Scan: \(error.localizedDescription)"
                    self?.isSearching = false
                },
                onCompleted: { [weak self] in self?.isSearching = false }
            )
    }
    
    func stopSearch() {
        searchDisposable?.dispose()
        searchDisposable = nil
        isSearching = false
    }
    
    // MARK: - Connection
    
    func connectToDevice(_ id: String) {
        deviceId = id
        connectionStatus = "Connecting..."
        UserDefaults.standard.set(id, forKey: "polar_device_id")
        do { try api.connectToDevice(id) }
        catch {
            errorMessage = "Connect: \(error.localizedDescription)"
            connectionStatus = "Failed"
        }
    }
    
    func disconnect() {
        stopAllStreams()
        guard !deviceId.isEmpty else { return }
        do { try api.disconnectFromDevice(deviceId) } catch {}
    }
    
    // MARK: - Streaming
    
    func startAllStreams() {
        guard isConnected, !deviceId.isEmpty else { return }
        startHR()
        startPPI()
        startAcc()
        startECG()
        isStreaming = true
    }
    
    func stopAllStreams() {
        hrDisposable?.dispose()
        ppiDisposable?.dispose()
        ecgDisposable?.dispose()
        hrDisposable = nil
        ppiDisposable = nil
        ecgDisposable = nil
        isStreaming = false
    }
    
    private func startHR() {
        hrDisposable = api.startHrStreaming(deviceId)
            .observe(on: MainScheduler.instance)
            .subscribe(
                onNext: { [weak self] data in
                    guard let self = self else { return }
                    for s in data {
                        self.currentHR = Int(s.hr)
                        self.currentRR = s.rrsMs.map { Int($0) }
                        self.onHRSample?(HRSample(
                            timestamp: Date(), hr: Int(s.hr),
                            rrIntervals: s.rrsMs.map { Int($0) },
                            contactStatus: s.contactStatus,
                            contactStatusSupported: s.contactStatusSupported
                        ))
                    }
                },
                onError: { [weak self] e in self?.errorMessage = "HR: \(e.localizedDescription)" }
            )
    }
    
    private func startPPI() {
        // SDK 5.x: PPI samples use ppInMs (UInt16), skinContactStatus/Supported are Int (0/1)
        ppiDisposable = api.startPpiStreaming(deviceId)
            .observe(on: MainScheduler.instance)
            .subscribe(
                onNext: { [weak self] data in
                    for s in data.samples {
                        self?.onHRSample?(HRSample(
                            timestamp: Date(), hr: s.hr,
                            rrIntervals: [Int(s.ppInMs)],
                            contactStatus: s.skinContactStatus != 0,
                            contactStatusSupported: s.skinContactSupported != 0
                        ))
                    }
                },
                onError: { e in print("[Polar] PPI unavailable: \(e.localizedDescription)") }
            )
    }
    
    private func startAcc() {
        api.requestStreamSettings(deviceId, feature: .acc)
            .asObservable()
            .flatMap { [weak self] settings -> Observable<PolarAccData> in
                guard let self = self else { return Observable.empty() }
                return self.api.startAccStreaming(self.deviceId, settings: settings.maxSettings())
            }
            .observe(on: MainScheduler.instance)
            .subscribe(
                onNext: { [weak self] data in
                    for s in data.samples {
                        self?.onAccSample?(AccSample(timestamp: Date(), x: s.x, y: s.y, z: s.z))
                    }
                },
                onError: { e in print("[Polar] ACC: \(e.localizedDescription)") }
            )
            .disposed(by: disposeBag)
    }
    
    private func startECG() {
        // H10 ECG at 130Hz
        api.requestStreamSettings(deviceId, feature: .ecg)
            .asObservable()
            .flatMap { [weak self] settings -> Observable<PolarEcgData> in
                guard let self = self else { return Observable.empty() }
                return self.api.startEcgStreaming(self.deviceId, settings: settings.maxSettings())
            }
            .observe(on: MainScheduler.instance)
            .subscribe(
                onNext: { [weak self] data in
                    let ts = Date()
                    let microVolts = data.samples.map { $0.voltage }
                    self?.onECGSample?(ECGSample(
                        timestamp: ts,
                        microVolts: microVolts,
                        sampleRate: 130
                    ))
                },
                onError: { e in print("[Polar] ECG: \(e.localizedDescription)") }
            )
            .disposed(by: disposeBag)
    }
}

// MARK: - PolarBleApiObserver

extension PolarManager: PolarBleApiObserver {
    func deviceConnecting(_ info: PolarDeviceInfo) {
        DispatchQueue.main.async { self.connectionStatus = "Connecting..."; self.deviceName = info.name }
    }
    func deviceConnected(_ info: PolarDeviceInfo) {
        DispatchQueue.main.async {
            self.isConnected = true; self.connectionStatus = "Connected"
            self.deviceId = info.deviceId; self.deviceName = info.name; self.errorMessage = nil
            DispatchQueue.main.asyncAfter(deadline: .now() + 2.0) { self.startAllStreams() }
        }
    }
    func deviceDisconnected(_ info: PolarDeviceInfo, pairingError: Bool) {
        DispatchQueue.main.async {
            self.isConnected = false; self.isStreaming = false
            self.currentHR = 0; self.currentRR = []
            self.connectionStatus = pairingError ? "Pairing Error" : "Disconnected"
            if !pairingError && !self.deviceId.isEmpty {
                DispatchQueue.main.asyncAfter(deadline: .now() + 3.0) {
                    guard !self.isConnected else { return }
                    self.connectToDevice(self.deviceId)
                }
            }
        }
    }
}

// MARK: - PolarBleApiDeviceInfoObserver

extension PolarManager: PolarBleApiDeviceInfoObserver {
    func batteryLevelReceived(_ id: String, batteryLevel: UInt) {
        DispatchQueue.main.async { self.batteryLevel = batteryLevel }
    }
    func disInformationReceived(_ id: String, uuid: CBUUID, value: String) {}
}

// MARK: - PolarBleApiPowerStateObserver
// SDK 5.x uses blePowerOn/blePowerOff instead of blePowerStateChanged

extension PolarManager: PolarBleApiPowerStateObserver {
    func blePowerOn() {
        DispatchQueue.main.async {
            if self.connectionStatus == "Bluetooth Off" {
                self.connectionStatus = "Disconnected"
                self.errorMessage = nil
            }
        }
    }
    
    func blePowerOff() {
        DispatchQueue.main.async {
            self.connectionStatus = "Bluetooth Off"
            self.errorMessage = "Enable Bluetooth"
        }
    }
}
