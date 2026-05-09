# POTS Monitor

> **⚠️ Medical Disclaimer**
> POTS Monitor is a personal research tool and is **not a medical device**. It is not intended to, and cannot, replace the role of a trained cardiac alert service dog or any other professionally trained animal. It does not provide medical advice, diagnosis, or treatment. Do not use this app to make clinical decisions. Always consult a qualified healthcare provider regarding your condition and treatment plan.

---

POTS Monitor is an iOS app that streams heart rate, heart rate variability, and accelerometer data from a **Polar H10** chest strap over Bluetooth Low Energy, automatically detects tachycardia flareups in real time, and uses an on-device Core ML model to predict flareups up to **15 minutes in advance**. A companion Python/Swift toolchain lets you export your collected data and retrain the model on your own Mac.

---

## Features

- **Live dashboard** — real-time HR, HRV (RMSSD), and accelerometer data with a pulsing circle visualisation
- **Automatic flareup detection** — fires when HR rises 30+ BPM above a rolling 30-minute baseline for 60+ consecutive seconds, optionally confirmed by HRV collapse
- **15-minute predictive warning** — Core ML boosted tree model predicts an upcoming flareup before it starts and sends a local notification
- **Prediction accuracy stats** — confusion matrix (TP / FP / FN / TN), precision, recall, and F1 tracked live in the Stats tab
- **Export tab** — aggregates daily CSV files into a single download; on-device model retraining from the last 30 days of data
- **Data storage toggle** — choose between keeping 30 days of compressed sensor data or deleting each day's files at midnight to save space

---

## Requirements

| Component | Minimum |
|---|---|
| iPhone | iOS 16.0+ |
| Hardware | Polar H10 heart rate monitor |
| Xcode | 15.0+ |
| macOS | 13 Ventura+ (for local model training) |
| Python | 3.9+ (for local model training) |
| CocoaPods | 1.12+ |

---

## Xcode Setup

### 1. Clone the repository

```bash
git clone https://github.com/henryyjiang/potsmonitor.git
cd potsmonitor
```

### 2. Install CocoaPods dependencies

```bash
sudo gem install cocoapods   # skip if already installed
pod install
```

This installs **PolarBleSdk 5.5.0** (and its dependencies RxSwift and SwiftProtobuf).

### 3. Open the workspace — not the project

```bash
open POTSMonitor.xcworkspace
```

> Always open `.xcworkspace`, never `.xcodeproj` directly, or the Polar SDK won't link.

### 4. Configure signing

1. Select the **POTSMonitor** target → **Signing & Capabilities**
2. Set **Team** to your Apple developer account
3. Change the **Bundle Identifier** to something unique (e.g. `com.yourname.potsmonitor`)

### 5. Required capabilities

The following are already configured in `Info.plist` and the entitlements file but verify they are present:

- **Bluetooth Always Usage** (`NSBluetoothAlwaysUsageDescription`)
- **Background Modes** → Bluetooth central
- **Notifications** (`UNUserNotificationCenter`)

### 6. Build and run

Connect your iPhone, select it as the run destination, and press **⌘R**. Accept the Bluetooth and notification permission prompts on first launch.

---

## Usage

1. Open the app and go to **Settings**
2. Tap **Search for Polar Loop** and select your H10 from the list — tracking starts automatically
3. Wear the H10 and the app begins logging HR, HRV, accelerometer, and ECG data to `On My iPhone › POTSMonitor › POTSData`
4. The **Live** tab shows real-time data. Once the bundled ML model loads, a prediction probability bar appears — a "Flareup Warning" notification fires when probability exceeds 70%
5. The **Stats** tab tracks prediction outcomes over time (TP/FP/FN/TN, precision, recall)
6. The **Export** tab lets you share your data and retrain the model on-device

### Data storage

Toggle **Save Daily Data** in Settings:
- **On (default):** previous days are compressed (`.csv.zlib`) and retained for 30 days
- **Off:** previous days are deleted at midnight — only today's raw data is kept

---

## Retraining the Model

The bundled model (`POTSFlareupModel.mlmodel`) was trained on one person's data. For best results, retrain on your own data after accumulating at least a few weeks of wear time that includes flareup events.

### Option A — On-device retraining (simple, slower)

Go to **Export › Train Model**. The app trains a new `MLBoostedTreeClassifier` on the last 30 days of your data directly on the phone. This takes several minutes and keeps the screen active during training.

### Option B — Local Mac retraining (recommended, fast)

This replicates the original training pipeline and completes in seconds.

#### 1. Export your app data

In the **Export** tab, tap **Export All Data** and AirDrop or save the files to your Mac. You'll receive:

```
hr.csv        heart rate + RR intervals
acc.csv       accelerometer (x/y/z in mG)
ecg.csv       raw ECG (130 Hz)
flareups.csv  auto-detected flareup events
```

Place them in a working directory, e.g. `~/potsmonitor-data/`.

Alternatively, connect your iPhone to your Mac and use **Xcode › Window › Devices and Simulators** to download the app container directly:
1. Select your device → POTSMonitor → ⋯ → **Download Container**
2. Right-click the `.xcappdata` file → **Show Package Contents**
3. Navigate to `AppData › Documents › POTSData`

#### 2. Install Python dependencies

```bash
pip install numpy pandas
```

#### 3. Edit and run `compute_features.py`

Open `compute_features.py` at the root of the repo and update the two paths at the top:

```python
DATA_DIR = Path("/path/to/your/POTSData")
OUT_CSV  = Path("/path/to/your/features.csv")
```

Then run:

```bash
python3 compute_features.py
```

This decompresses the `.csv.zlib` files, computes 60-second sliding windows (10-second slide) with a 30-minute rolling baseline, and labels windows that fall 0–15 minutes before a flareup start as positive. Output is a `features.csv` with columns:

```
meanHR, maxHR, minHR, hrDelta, rmssd, sdnn, meanRR,
accMagMean, accMagStd, accVertDelta, hrRiseFromBaseline, rmssdPctChange, label
```

#### 4. Edit and run `train_model.swift`

Open `train_model.swift` and update the two paths:

```swift
let featuresURL = URL(fileURLWithPath: "/path/to/your/features.csv")
let modelOut    = URL(fileURLWithPath: "/path/to/your/POTSFlareupModel.mlmodel")
```

Then run:

```bash
swift train_model.swift
```

This trains an `MLBoostedTreeClassifier` (max depth 6, 100 iterations) and saves a `.mlmodel` file. Training takes under a minute.

#### 5. Replace the bundled model

1. In Xcode, select `POTSMonitor › ML › POTSFlareupModel.mlmodel` in the file navigator
2. Right-click → **Show in Finder**
3. Replace the file with your newly trained `POTSFlareupModel.mlmodel`
4. Rebuild and deploy to your phone — the new model is now bundled in the app

> **User-trained model priority:** if a model has also been trained on-device (via the Export tab), it lives in the app's Documents folder and takes priority over the bundled model. To force the bundled model, tap **Reset to Bundled Model** in the Export tab.

---

## Project Structure

```
POTSMonitor/
├── ML/
│   ├── FeatureEngine.swift       sliding-window feature computation
│   ├── POTSFlareupModel.mlmodel  bundled pre-trained Core ML model
│   └── POTSPredictor.swift       real-time prediction + on-device training
├── Models/
│   └── Models.swift              data types (HRSample, DetectedFlareup, …)
├── Services/
│   ├── DataStore.swift           CSV logging, compression, streaming export
│   ├── FlareupDetector.swift     rule-based real-time flareup detection
│   ├── NotificationManager.swift local notification delivery
│   ├── PolarManager.swift        Polar H10 BLE connection via PolarBleSdk
│   └── PredictionTracker.swift   TP/FP/FN/TN outcome tracking + persistence
└── Views/
    ├── ExportView.swift           export + on-device training UI
    ├── LiveView.swift             real-time dashboard
    ├── SettingsView.swift         device connection + data storage settings
    └── StatsView.swift            confusion matrix + accuracy metrics
compute_features.py               Mac-side feature extraction pipeline
train_model.swift                 Mac-side CreateML training script
```

---

## How the ML Model Works

**Features (per 60-second window, computed every 10 seconds):**

| Feature | Description |
|---|---|
| `meanHR` / `maxHR` / `minHR` / `hrDelta` | Heart rate statistics |
| `rmssd` / `sdnn` / `meanRR` | HRV metrics from RR intervals |
| `accMagMean` / `accMagStd` | Accelerometer magnitude (movement) |
| `accVertDelta` | Vertical axis change (posture shift) |
| `hrRiseFromBaseline` | HR relative to 30-minute rolling mean |
| `rmssdPctChange` | HRV change relative to 30-minute baseline |

**Labelling:** A window is labelled positive (pre-flareup) if its end timestamp falls 0–15 minutes before a detected flareup start. Windows during a flareup are excluded from training.

**Model:** `MLBoostedTreeClassifier` — max depth 6, 100 boosting iterations, step size 0.3. A notification fires when predicted flareup probability ≥ 70%, with a 10-minute cooldown.

---

## Dependencies

| Library | Version | Purpose |
|---|---|---|
| [PolarBleSdk](https://github.com/polarofficial/polar-ble-sdk) | 5.5.0 | Polar H10 BLE streaming |
| RxSwift | 6.5.0 | Required by PolarBleSdk |
| SwiftProtobuf | 1.37.0 | Required by PolarBleSdk |

---

## License

MIT License

Copyright (c) 2026 Henry Jiang

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
