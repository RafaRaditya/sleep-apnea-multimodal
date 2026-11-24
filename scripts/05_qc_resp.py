"""QC for the respiration channel: respiratory-band SNR, breathing rate, and
cross-channel scale sanity. Read-only over artifacts/signals."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal

ROOT = Path(__file__).resolve().parents[1]
SIG = ROOT / "artifacts" / "signals"
RESP_FS = 25.0
ECG_FS = 100.0


def band_power(x: np.ndarray, fs: float, lo: float, hi: float) -> float:
    f, p = signal.welch(x, fs=fs, nperseg=min(len(x), int(fs * 64)))
    mask = (f >= lo) & (f <= hi)
    return float(np.trapezoid(p[mask], f[mask])) if mask.any() else 0.0


def breathing_rate(x: np.ndarray, fs: float) -> float:
    sos = signal.butter(2, [0.1, 0.6], btype="bandpass", fs=fs, output="sos")
    y = signal.sosfiltfilt(sos, x)
    peaks, _ = signal.find_peaks(y, distance=int(fs * 1.2), prominence=np.std(y) * 0.5)
    minutes = len(x) / fs / 60
    return len(peaks) / minutes if minutes > 0 else 0.0


rows = []
qc = pd.read_csv(SIG / "signal_qc.csv")
for _, r in qc.iterrows():
    rec = r["record_id"]
    resp = np.load(SIG / f"{rec}_resp25.npy").astype(np.float64)
    ecg = np.load(SIG / f"{rec}_ecg100.npy").astype(np.float64)

    resp_band = band_power(resp, RESP_FS, 0.1, 0.6)
    noise_band = band_power(resp, RESP_FS, 1.5, 6.0)
    snr_db = 10 * np.log10(resp_band / noise_band) if noise_band > 0 else float("inf")
    ecg_band = band_power(ecg, ECG_FS, 0.5, 40.0)
    ecg_noise = band_power(ecg, ECG_FS, 45.0, ECG_FS / 2 - 1) if ECG_FS > 92 else 0.0

    rows.append({
        "dataset": r["dataset"], "record_id": rec,
        "resp_band_power": resp_band, "resp_noise_power": noise_band,
        "resp_snr_db": round(snr_db, 2),
        "breaths_per_min": round(breathing_rate(resp, RESP_FS), 1),
        "resp_p1_p99_range": round(float(np.percentile(resp, 99) - np.percentile(resp, 1)), 4),
        "ecg_snr_db": round(10 * np.log10(ecg_band / ecg_noise), 2) if ecg_noise > 0 else None,
    })

with (SIG / "resp_qc.csv").open("w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

d = pd.DataFrame(rows)
print(d.to_string(index=False))
print()
print(d.groupby("dataset").agg(
    snr_min=("resp_snr_db", "min"), snr_med=("resp_snr_db", "median"),
    bpm_min=("breaths_per_min", "min"), bpm_med=("breaths_per_min", "median"),
    bpm_max=("breaths_per_min", "max")).round(2).to_string())
print()
print("records with resp SNR < 0 dB:", d[d.resp_snr_db < 0].record_id.tolist())
print("records with implausible breathing rate (<8 or >30 /min):",
      d[(d.breaths_per_min < 8) | (d.breaths_per_min > 30)].record_id.tolist())
