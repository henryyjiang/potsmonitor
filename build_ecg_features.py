#!/usr/bin/env python3
"""
Derive ECG morphology features per labeled window → features_ecg.csv.

Detection is done PER 60 s WINDOW (not per day) so it matches exactly what the app
computes live in FeatureEngine/ECGFeatures.swift — same filtfilt, same per-window
std threshold, same T-wave search. That parity is what keeps train and serve
consistent (verified against the Swift port to <0.5%).

Per-window ABSOLUTE morphology (R/T amplitude, R->T interval) turned out NOT to help
the honest metric — only the *deviation from a 30-min rolling baseline* does (the
slow posture/sympathetic drift, analogous to hrRiseFromBaseline). So the shipped
features are those deviations:
  ecgRAmpDev, ecgRAmpStdDev  — R-wave amplitude vs baseline (electrical axis / posture)
  ecgTAmpDev, ecgTAmpStdDev  — T-wave amplitude vs baseline (sympathetic tone)
  ecgRTDev                   — R->T-peak interval vs baseline (QT proxy)
The 30-min baseline (rolling median of the last 180 windows, min 20) is exactly what
FeatureEngine.swift maintains live. Windows with no ECG get deviation 0.
"""
import warnings; warnings.filterwarnings("ignore")
import zlib, io, numpy as np, pandas as pd
from pathlib import Path
from scipy.signal import butter, filtfilt, find_peaks

D = Path("/Users/paff/Desktop/Projects/POTSMonitor/POTSData")
FEATURES = Path("/Users/paff/Desktop/Projects/POTSMonitor/features.csv")
OUT = Path("/Users/paff/Desktop/Projects/POTSMonitor/features_ecg.csv")
FS = 130
BP_B, BP_A = butter(2, [5/(FS/2), 20/(FS/2)], btype='band')
LP_B, LP_A = butter(2, 8/(FS/2), btype='low')
ABS_COLS = ["rAmpMean", "rAmpStd", "tAmpMean", "tAmpStd", "rtMean"]
# Shipped deviation features and their absolute source + app-side name.
DEV_MAP = {"rAmpMean": "ecgRAmpDev", "rAmpStd": "ecgRAmpStdDev",
           "tAmpMean": "ecgTAmpDev", "tAmpStd": "ecgTAmpStdDev", "rtMean": "ecgRTDev"}
BASELINE_WINDOW = 180   # 30 min at 10 s slide
BASELINE_MINP = 20

def dz(p):
    with open(p, 'rb') as f: return zlib.decompress(f.read(), wbits=-15).decode()

def load_signal(day):
    p = D / f"ecg_{day}.csv.zlib"
    if not p.exists(): return None, None
    df = pd.read_csv(io.StringIO(dz(p)))
    raw = df['micro_volts'].astype(str).str.strip('"')
    counts = raw.str.count(';').values + 1
    sig = np.fromstring(';'.join(raw.values), sep=';')
    t0 = pd.to_datetime(df['timestamp']).astype('int64').values / 1e9
    times = np.concatenate([t0[i] + np.arange(counts[i]) / FS for i in range(len(t0))])
    n = min(len(times), len(sig))
    return sig[:n], times[:n]

def morph5(seg):
    """5 morphology features on one window's raw signal (per-window detection).
    Mirrors ECGFeatures.swift exactly."""
    if len(seg) < 130:
        return None
    filt = filtfilt(BP_B, BP_A, seg)
    peaks, _ = find_peaks(filt, distance=int(0.3*FS), height=np.std(filt) * 1.5)
    if len(peaks) < 5:
        return None
    low = filtfilt(LP_B, LP_A, seg)
    r_amp = seg[peaks]
    t_amp, rt = [], []
    for p in peaks:
        lo, hi = p + int(0.15*FS), p + int(0.40*FS)
        if hi < len(low):
            base = low[max(0, p-int(0.05*FS)):p].mean() if p >= int(0.05*FS) else low[p]
            sg = low[lo:hi]; j = int(np.argmax(np.abs(sg - base)))
            t_amp.append(sg[j] - base); rt.append((j + int(0.15*FS)) / FS * 1000)
    if not t_amp:
        return None
    return dict(rAmpMean=float(np.mean(r_amp)), rAmpStd=float(np.std(r_amp)),
                tAmpMean=float(np.mean(t_amp)), tAmpStd=float(np.std(t_amp)),
                rtMean=float(np.mean(rt)))

def main():
    feat = pd.read_csv(FEATURES).sort_values(["date", "ts"]).reset_index(drop=True)
    for c in ABS_COLS:
        feat[c] = 0.0                       # absolute per-window morphology (temp)
    for day, grp in feat.groupby("date"):
        sig, times = load_signal(day)
        if sig is None:
            print(f"  {day}: no ECG"); continue
        n_ok = 0
        for idx, ts in zip(grp.index, grp["ts"].values):
            lo = np.searchsorted(times, ts - 60, "left")
            hi = np.searchsorted(times, ts, "right")
            m = morph5(sig[lo:hi]) if hi - lo >= 130 else None
            if m:
                for c in ABS_COLS: feat.at[idx, c] = m[c]
                n_ok += 1
        print(f"  {day}: {n_ok}/{len(grp)} windows have ECG")

    # Deviation from a per-day 30-min rolling-median baseline (matches FeatureEngine).
    # Only ECG-present windows (nonzero) feed the baseline; absent windows get 0.
    for src, dev in DEV_MAP.items():
        base = feat.groupby("date")[src].transform(
            lambda s: s.replace(0, np.nan).rolling(BASELINE_WINDOW, min_periods=BASELINE_MINP).median())
        feat[dev] = np.where(feat[src] != 0, feat[src] - base, 0.0)
        feat[dev] = feat[dev].fillna(0.0)
    feat = feat.drop(columns=ABS_COLS)      # ship only the deviations
    feat.to_csv(OUT, index=False)
    cov = (feat["ecgRAmpDev"] != 0).mean()
    print(f"\nSaved {OUT}  ({cov*100:.0f}% of windows have ECG deviation features)")

if __name__ == "__main__":
    main()
