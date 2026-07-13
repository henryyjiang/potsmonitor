#!/usr/bin/env python3
"""Decompress app data and compute training features matching FeatureEngine.swift."""

import io
import zlib
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path("/Users/paff/Desktop/Projects/POTSMonitor/POTSData")
OUT_CSV  = Path("/Users/paff/Desktop/Projects/POTSMonitor/features.csv")

WINDOW_SEC      = 60.0
SLIDE_SEC       = 10.0
BASELINE_SEC    = 30 * 60.0
PREDICT_HORIZON = 10 * 60.0   # label positive up to 10 min before flareup start
MIN_HR_ROWS     = 5
ACC_SUBSAMPLE   = 4   # take every Nth acc row


def decompress(path: Path) -> str:
    with open(path, "rb") as f:
        return zlib.decompress(f.read(), wbits=-15).decode()


def load_hr(data_dir: Path) -> pd.DataFrame:
    dfs = []
    for p in sorted(data_dir.glob("hr_*.csv.zlib")):
        dfs.append(pd.read_csv(io.StringIO(decompress(p))))
    for p in sorted(data_dir.glob("hr_*.csv")):
        dfs.append(pd.read_csv(p))
    if not dfs:
        return pd.DataFrame()
    df = pd.concat(dfs, ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    def parse_rr(s):
        s = str(s).strip('"').strip()
        return [int(x) for x in s.split(";") if x.isdigit()] if s else []
    df["rr_list"] = df["rr_intervals_ms"].apply(parse_rr)
    df["ts"] = df["timestamp"].astype("int64") / 1e9
    return df


def load_acc(data_dir: Path) -> pd.DataFrame:
    dfs = []
    for p in sorted(data_dir.glob("acc_*.csv.zlib")):
        dfs.append(pd.read_csv(io.StringIO(decompress(p))).iloc[::ACC_SUBSAMPLE])
    for p in sorted(data_dir.glob("acc_*.csv")):
        dfs.append(pd.read_csv(p).iloc[::ACC_SUBSAMPLE])
    if not dfs:
        return pd.DataFrame()
    df = pd.concat(dfs, ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["ts"] = df["timestamp"].astype("int64") / 1e9
    return df


def load_flareups(data_dir: Path) -> pd.DataFrame:
    """Auto-detected + user-logged flareups, both used as positive labels.

    Manual flareups (manual_flareups.csv) are episodes the HR-threshold detector
    missed — the independent ground truth that lets the model learn beyond the
    rule. `source` is kept for analysis but both contribute equally to labels.
    """
    frames = []
    for name, src in [("flareups.csv", "auto"), ("manual_flareups.csv", "manual")]:
        p = data_dir / name
        if p.exists():
            d = pd.read_csv(p)
            if len(d):
                d = d[["start", "end"]].copy()
                d["source"] = src
                frames.append(d)
    if not frames:
        return pd.DataFrame(columns=["start", "end", "source"])
    df = pd.concat(frames, ignore_index=True)
    df["start"] = pd.to_datetime(df["start"]).astype("int64") / 1e9
    df["end"]   = pd.to_datetime(df["end"]).astype("int64") / 1e9
    n_manual = int((df["source"] == "manual").sum())
    print(f"  ({len(df) - n_manual} auto + {n_manual} manual flareups)")
    return df


def pnn50(rrs: np.ndarray) -> float:
    if len(rrs) < 2:
        return 0.0
    diffs = np.abs(np.diff(rrs.astype(float)))
    return float((diffs > 50).sum() / len(diffs))


def rmssd(rrs: np.ndarray) -> float:
    if len(rrs) < 2:
        return 0.0
    return float(np.sqrt(np.mean(np.diff(rrs.astype(float)) ** 2)))


def peak_angular_velocity(ts: np.ndarray, ax, ay, az) -> float:
    """
    Largest rotation (deg) of the gravity vector between consecutive 1-second bins
    within the window. Averaging to 1 s bins removes motion jitter and leaves the
    torso-orientation change; the max over the window captures the brief sit/lie→
    stand transient (the orthostatic trigger) without it being washed out by
    whole-window averaging. `ts` must be ascending (acc_ts is globally sorted).
    """
    if len(ts) < 2:
        return 0.0
    sec = np.floor(ts).astype(np.int64)                 # non-decreasing
    starts = np.concatenate([[0], np.flatnonzero(np.diff(sec)) + 1])
    ends = np.concatenate([starts[1:], [len(sec)]])
    counts = (ends - starts).astype(float)
    mx = np.add.reduceat(ax, starts) / counts
    my = np.add.reduceat(ay, starts) / counts
    mz = np.add.reduceat(az, starts) / counts
    v = np.stack([mx, my, mz], axis=1)
    n = np.linalg.norm(v, axis=1, keepdims=True); n[n == 0] = 1
    u = v / n
    if len(u) < 2:
        return 0.0
    dots = np.clip((u[1:] * u[:-1]).sum(1), -1.0, 1.0)
    return float(np.degrees(np.arccos(dots)).max())


def compute_window(t: float, hr_ts, hr_bpm, hr_rr, acc_ts, acc_x, acc_y, acc_z):
    ws = t - WINDOW_SEC

    # HR window
    lo = np.searchsorted(hr_ts, ws, "left")
    hi = np.searchsorted(hr_ts, t,  "right")
    if hi - lo < MIN_HR_ROWS:
        return None

    hrs  = hr_bpm[lo:hi].astype(float)
    rrs_lists = [r for r in hr_rr[lo:hi] if len(r)]
    rrs  = np.concatenate(rrs_lists) if rrs_lists else np.array([])
    mean_hr  = float(np.mean(hrs))
    max_hr   = float(np.max(hrs))
    min_hr   = float(np.min(hrs))
    hr_delta = max_hr - min_hr

    # Linear slope of HR over the window (BPM/sec) — captures acceleration
    if len(hrs) >= 3:
        t_rel = hr_ts[lo:hi] - hr_ts[lo]
        hr_slope = float(np.polyfit(t_rel, hrs, 1)[0])
    else:
        hr_slope = 0.0
    win_rmssd  = rmssd(rrs)
    win_pnn50  = pnn50(rrs)
    sdnn       = float(np.std(rrs))  if len(rrs) >= 2 else 0.0
    mean_rr    = float(np.mean(rrs)) if len(rrs)  > 0 else 0.0

    # ACC window
    al = np.searchsorted(acc_ts, ws, "left")
    ah = np.searchsorted(acc_ts, t,  "right")
    if ah - al >= 4:
        ax, ay, az = acc_x[al:ah].astype(float), acc_y[al:ah].astype(float), acc_z[al:ah].astype(float)
        mags = np.sqrt(ax**2 + ay**2 + az**2)
        n4   = max(1, len(az) // 4)
        acc_mag_mean  = float(np.mean(mags))
        acc_mag_std   = float(np.std(mags))  if len(mags) >= 2 else 0.0
        acc_vert_delta = float(abs(np.mean(az[-n4:]) - np.mean(az[:n4])))
        posture_jerk = peak_angular_velocity(acc_ts[al:ah], ax, ay, az)
    else:
        acc_mag_mean = acc_mag_std = acc_vert_delta = posture_jerk = 0.0

    # 30-min baseline
    bl = np.searchsorted(hr_ts, t - BASELINE_SEC, "left")
    hr_rise = rmssd_pct = 0.0
    if hi - bl >= 30:
        b_hr    = float(np.mean(hr_bpm[bl:hi].astype(float)))
        hr_rise = mean_hr - b_hr
        b_rrs_lists = [r for r in hr_rr[bl:hi] if len(r)]
        b_rrs   = np.concatenate(b_rrs_lists) if b_rrs_lists else np.array([])
        b_rmssd = rmssd(b_rrs)
        if b_rmssd > 0 and len(rrs) >= 4:
            rmssd_pct = (win_rmssd - b_rmssd) / b_rmssd

    return (mean_hr, max_hr, min_hr, hr_delta, win_rmssd, sdnn, mean_rr,
            acc_mag_mean, acc_mag_std, acc_vert_delta, hr_rise, rmssd_pct,
            hr_slope, win_pnn50, posture_jerk)


def label_window(t: float, f_starts, f_ends):
    """
    Any window within PREDICT_HORIZON before a flareup onset is positive — the goal
    is to warn *any time* within 10 min of onset, not exactly 10 min out, so partial
    pre-window coverage is fine. Returns 1 (pre-flareup), 0 (normal), or None if the
    window falls inside a flareup (excluded from training).
    """
    for fs, fe in zip(f_starts, f_ends):
        if fs <= t <= fe:
            return None          # during a flareup — exclude from training
        if t < fs and t >= fs - PREDICT_HORIZON:
            return 1             # pre-flareup prediction window
    return 0


def main():
    print("Loading HR …")
    hr = load_hr(DATA_DIR)
    print(f"  {len(hr):,} samples  ({hr['timestamp'].min().date()} → {hr['timestamp'].max().date()})")

    print("Loading ACC (subsampled 4×) …")
    acc = load_acc(DATA_DIR)
    print(f"  {len(acc):,} samples")

    print("Loading flareups …")
    flu = load_flareups(DATA_DIR)
    print(f"  {len(flu)} flareups")

    # Pre-extract numpy arrays for fast indexing
    hr_ts  = hr["ts"].values
    hr_bpm = hr["hr_bpm"].values
    hr_rr  = hr["rr_list"].values

    acc_ts = acc["ts"].values if len(acc) else np.array([])
    acc_x  = acc["x_mG"].values if len(acc) else np.array([])
    acc_y  = acc["y_mG"].values if len(acc) else np.array([])
    acc_z  = acc["z_mG"].values if len(acc) else np.array([])

    f_starts = flu["start"].values
    f_ends   = flu["end"].values

    t_start = hr_ts.min() + WINDOW_SEC
    t_end   = hr_ts.max()
    windows = np.arange(t_start, t_end, SLIDE_SEC)
    print(f"\nComputing features for {len(windows):,} candidate windows …")

    # These 15 features must stay in exact parity with FeatureEngine.swift /
    # POTSPredictor.featureDict. `date`/`ts` are NOT model features: `date` is the
    # grouping key so overlapping sliding windows from the same day never straddle
    # the train/test boundary (prevents leakage); `ts` is the window-end epoch.
    cols = ["meanHR","maxHR","minHR","hrDelta","rmssd","sdnn","meanRR",
            "accMagMean","accMagStd","accVertDelta","hrRiseFromBaseline","rmssdPctChange",
            "hrSlope","pNN50","postureJerkPeak","label","date","ts"]
    rows = []
    for i, t in enumerate(windows):
        if i % 10_000 == 0 and i > 0:
            print(f"  {i:,}/{len(windows):,}")
        feat = compute_window(t, hr_ts, hr_bpm, hr_rr, acc_ts, acc_x, acc_y, acc_z)
        if feat is None:
            continue
        lbl = label_window(t, f_starts, f_ends)
        if lbl is None:
            continue             # skip windows inside a flareup
        day = pd.to_datetime(t, unit="s").strftime("%Y-%m-%d")
        rows.append(feat + (lbl, day, float(t)))

    df = pd.DataFrame(rows, columns=cols)
    pos = (df["label"] == 1).sum()
    neg = (df["label"] == 0).sum()
    print(f"\n{len(df):,} windows — {pos} flareup, {neg} normal "
          f"across {df['date'].nunique()} days")

    # Save raw (unbalanced) data. Balancing is applied downstream on the TRAIN
    # split only (sweep.py / train_model.swift) so it never leaks into eval.
    df.to_csv(OUT_CSV, index=False)
    print(f"Saved: {OUT_CSV}")


if __name__ == "__main__":
    main()
