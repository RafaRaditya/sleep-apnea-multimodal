"""Stage 3a: filtering, detrending, resampling, and signal QC.

Per record, loads the native signals, filters them, resamples to the common
rates, and caches the result. No normalization is applied here: z-score uses
training-only statistics and is applied when windows are materialised.

Provisional settings (paper leaves the numeric values to audit/preliminary work):
  ECG   : band-pass 0.5-40 Hz, zero-phase Butterworth (order 4), notch at the
          detected mains frequency, resampled to 100 Hz.
  Resp  : low-pass 0.7 Hz (order 4) + linear detrend, resampled to 25 Hz.

Outputs:
  artifacts/signals/{record_id}_ecg100.npy
  artifacts/signals/{record_id}_resp25.npy
  artifacts/signals/signal_qc.csv
  artifacts/signals/preprocessing_config.json
  artifacts/signals/qc_<record>.png
"""

from __future__ import annotations

import json
import re
from fractions import Fraction
from pathlib import Path

import numpy as np
import pyedflib
import wfdb
from scipy import signal

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dataset-rm" / "dataset-rm"
POLICY = json.loads((ROOT / "config" / "policy.json").read_text(encoding="utf-8"))
EXCLUDED_RECORDS = set(POLICY["multimodal_policy"]["excluded_records"])
OUT = ROOT / "artifacts" / "signals"
OUT.mkdir(parents=True, exist_ok=True)

ECG_FS = 100.0
RESP_FS = 25.0
ECG_BAND = (0.5, 40.0)
RESP_LOWPASS = 0.7
FILTER_ORDER = 4

CONFIG = {
    "ecg": {
        "bandpass_hz": list(ECG_BAND),
        "order": FILTER_ORDER,
        "zero_phase": True,
        "notch": "detected mains frequency, Q=30",
        "target_fs_hz": ECG_FS,
    },
    "resp": {
        "lowpass_hz": RESP_LOWPASS,
        "order": FILTER_ORDER,
        "detrend": "linear",
        "target_fs_hz": RESP_FS,
    },
    "normalization": "not applied here; training-only z-score at window materialisation",
    "status": "provisional",
}


def bandpass(x: np.ndarray, fs: float, low: float, high: float) -> np.ndarray:
    sos = signal.butter(FILTER_ORDER, [low, high], btype="bandpass", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, x)


def lowpass(x: np.ndarray, fs: float, high: float) -> np.ndarray:
    sos = signal.butter(FILTER_ORDER, high, btype="lowpass", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, x)


def notch(x: np.ndarray, fs: float, f0: float) -> np.ndarray:
    b, a = signal.iirnotch(f0, 30.0, fs=fs)
    return signal.filtfilt(b, a, x)


def detect_mains(x: np.ndarray, fs: float) -> tuple[float | None, float]:
    """Return (mains_hz, ratio) for the stronger of 50/60 Hz, if it stands out."""
    freqs, psd = signal.welch(x, fs=fs, nperseg=min(len(x), int(fs * 8)))
    best, best_ratio = None, 1.0
    for f0 in (50.0, 60.0):
        if f0 >= fs / 2:
            continue
        peak = psd[(freqs >= f0 - 1) & (freqs <= f0 + 1)].mean()
        neigh = psd[((freqs >= f0 - 6) & (freqs < f0 - 2))
                    | ((freqs > f0 + 2) & (freqs <= f0 + 6))].mean()
        if neigh <= 0:
            continue
        ratio = float(peak / neigh)
        if ratio > best_ratio:
            best, best_ratio = f0, ratio
    return (best if best_ratio >= 3.0 else None), best_ratio


def resample(x: np.ndarray, fs_in: float, fs_out: float) -> np.ndarray:
    ratio = Fraction(fs_out / fs_in).limit_denominator(1000)
    return signal.resample_poly(x, ratio.numerator, ratio.denominator)


def flatline_fraction(x: np.ndarray, fs: float) -> float:
    """Fraction of the signal inside runs of near-constant samples >= 1 s."""
    diff = np.abs(np.diff(x))
    scale = np.median(np.abs(x - np.median(x))) + 1e-12
    flat = diff < 1e-6 * max(scale, 1e-12) * 1e3
    if not flat.any():
        return 0.0
    run, runs = 0, []
    for f in flat:
        run = run + 1 if f else 0
        runs.append(run)
    runs = np.asarray(runs)
    return float((runs >= fs).sum() / len(x))


def load_ucddb(rec_id: str):
    path = DATA / "ucddb" / "DESTINATION" / f"{rec_id}.rec"
    edf = pyedflib.EdfReader(str(path))
    labels = [edf.getLabel(i) for i in range(edf.signals_in_file)]
    ecg_i = labels.index("ECG")
    resp_i = labels.index("Flow")
    ecg = edf.readSignal(ecg_i).astype(np.float64)
    resp = edf.readSignal(resp_i).astype(np.float64)
    fs_ecg = float(edf.getSampleFrequency(ecg_i))
    fs_resp = float(edf.getSampleFrequency(resp_i))
    duration = float(edf.file_duration)
    edf.close()
    return ecg, fs_ecg, resp, fs_resp, duration, "Flow", "airflow(thermistor)"


def load_psg(rec_id: str):
    base = DATA / "slpdb" / "DESTINATION" / rec_id
    rec = wfdb.rdrecord(str(base))
    names = rec.sig_name
    ecg_i = next(i for i, n in enumerate(names) if n.upper().startswith("ECG"))
    resp_i = next(i for i, n in enumerate(names) if "resp" in n.lower() and "nasal" in n.lower())
    ecg = rec.p_signal[:, ecg_i].astype(np.float64)
    resp = rec.p_signal[:, resp_i].astype(np.float64)
    fs = float(rec.fs)
    return ecg, fs, resp, fs, float(rec.sig_len / rec.fs), names[resp_i], "airflow(nasal)"


def process(rec_id: str, dataset: str) -> dict:
    if dataset == "UCDDB":
        ecg, fs_ecg, resp, fs_resp, duration, resp_name, resp_type = load_ucddb(rec_id)
    else:
        ecg, fs_ecg, resp, fs_resp, duration, resp_name, resp_type = load_psg(rec_id)

    ecg_raw_nan = int(np.isnan(ecg).sum() + np.isinf(ecg).sum())
    resp_raw_nan = int(np.isnan(resp).sum() + np.isinf(resp).sum())
    ecg = np.nan_to_num(ecg, nan=0.0, posinf=0.0, neginf=0.0)
    resp = np.nan_to_num(resp, nan=0.0, posinf=0.0, neginf=0.0)

    mains, mains_ratio = detect_mains(ecg, fs_ecg)
    ecg_f = bandpass(ecg, fs_ecg, *ECG_BAND)
    if mains is not None:
        ecg_f = notch(ecg_f, fs_ecg, mains)
    resp_f = signal.detrend(lowpass(resp, fs_resp, RESP_LOWPASS))

    ecg_r = resample(ecg_f, fs_ecg, ECG_FS)
    resp_r = resample(resp_f, fs_resp, RESP_FS)

    n = min(len(ecg_r), int(round(len(resp_r) * ECG_FS / RESP_FS)))
    ecg_r = ecg_r[:n]
    resp_r = resp_r[: int(round(n * RESP_FS / ECG_FS))]

    np.save(OUT / f"{rec_id}_ecg100.npy", ecg_r.astype(np.float32))
    np.save(OUT / f"{rec_id}_resp25.npy", resp_r.astype(np.float32))

    drift_before = float(np.ptp(np.polyval(np.polyfit(
        np.arange(len(resp)), resp, 1), [0, len(resp) - 1])))
    return {
        "dataset": dataset, "record_id": rec_id,
        "resp_channel": resp_name, "resp_type": resp_type,
        "fs_ecg_native": fs_ecg, "fs_resp_native": fs_resp,
        "fs_ecg_out": ECG_FS, "fs_resp_out": RESP_FS,
        "duration_s": round(duration, 1),
        "ecg_out_samples": len(ecg_r), "resp_out_samples": len(resp_r),
        "ecg_out_duration_s": round(len(ecg_r) / ECG_FS, 2),
        "resp_out_duration_s": round(len(resp_r) / RESP_FS, 2),
        "duration_mismatch_ms": round(abs(len(ecg_r) / ECG_FS - len(resp_r) / RESP_FS) * 1000, 1),
        "mains_hz": mains, "mains_ratio": round(mains_ratio, 2),
        "notch_applied": mains is not None,
        "ecg_nan_raw": ecg_raw_nan, "resp_nan_raw": resp_raw_nan,
        "ecg_flatline_frac": round(flatline_fraction(ecg_r, ECG_FS), 4),
        "resp_flatline_frac": round(flatline_fraction(resp_r, RESP_FS), 4),
        "ecg_std_raw": round(float(np.std(ecg)), 4),
        "ecg_std_filt": round(float(np.std(ecg_f)), 4),
        "resp_std_raw": round(float(np.std(resp)), 4),
        "resp_std_filt": round(float(np.std(resp_f)), 4),
        "resp_drift_range_raw": round(drift_before, 4),
        "ecg_min": round(float(ecg_r.min()), 4), "ecg_max": round(float(ecg_r.max()), 4),
        "resp_min": round(float(resp_r.min()), 4), "resp_max": round(float(resp_r.max()), 4),
    }


def qc_plot(rec_id: str, dataset: str, stats: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ecg = np.load(OUT / f"{rec_id}_ecg100.npy")
    resp = np.load(OUT / f"{rec_id}_resp25.npy")
    t_e = np.arange(len(ecg)) / ECG_FS
    t_r = np.arange(len(resp)) / RESP_FS
    win = (0, 120)

    fig, ax = plt.subplots(3, 1, figsize=(12, 7))
    ax[0].plot(t_e, ecg, lw=0.5)
    ax[0].set_xlim(*win)
    ax[0].set_title(f"{dataset} {rec_id} - ECG {ECG_FS:g} Hz, band-pass {ECG_BAND}")
    ax[1].plot(t_r, resp, lw=0.8, color="tab:green")
    ax[1].set_xlim(*win)
    ax[1].set_title(f"Resp ({stats['resp_type']}) {RESP_FS:g} Hz, low-pass {RESP_LOWPASS} + detrend")
    ax[2].specgram(ecg, NFFT=512, Fs=ECG_FS, noverlap=384)
    ax[2].set_title("ECG spectrogram")
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / f"qc_{rec_id}.png", dpi=110)
    plt.close(fig)


def main() -> None:
    rows = []
    for rec_path in sorted((DATA / "ucddb" / "DESTINATION").glob("ucddb*.rec")):
        if rec_path.stem in EXCLUDED_RECORDS:
            continue
        rows.append(process(rec_path.stem, "UCDDB"))
        print("UCDDB", rec_path.stem, flush=True)

    for rec_id in (DATA / "slpdb" / "DESTINATION" / "RECORDS").read_text().split():
        if rec_id in EXCLUDED_RECORDS:
            continue
        header = wfdb.rdheader(str(DATA / "slpdb" / "DESTINATION" / rec_id))
        if not any("nasal" in n.lower() for n in header.sig_name):
            continue
        rows.append(process(rec_id, "MIT-BIH_PSG"))
        print("PSG", rec_id, flush=True)

    import csv
    with (OUT / "signal_qc.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    (OUT / "preprocessing_config.json").write_text(json.dumps(CONFIG, indent=2), encoding="utf-8")

    for rec in ("ucddb002", "slp02a"):
        stats = next((r for r in rows if r["record_id"] == rec), None)
        if stats:
            qc_plot(rec, stats["dataset"], stats)

    keys = ["duration_mismatch_ms", "ecg_flatline_frac", "resp_flatline_frac",
            "ecg_nan_raw", "resp_nan_raw", "ecg_std_filt", "resp_std_filt"]
    print("\nsummary")
    for k in keys:
        vals = np.array([r[k] for r in rows], dtype=float)
        print(f"  {k}: min={vals.min():.4g} median={np.median(vals):.4g} max={vals.max():.4g}")
    print("  notch applied:", sum(r["notch_applied"] for r in rows), "/", len(rows))
    print("  mains detected:", {r["mains_hz"] for r in rows})


if __name__ == "__main__":
    main()
