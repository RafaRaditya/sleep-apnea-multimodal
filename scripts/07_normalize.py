"""Stage 3b-1: training-only normalization statistics.

Computes per-(seed, dataset, modality) mean and standard deviation from the
training windows only, then reports the fraction of values that amplitude
clipping at +/-c would affect.

The paper says "each recording is z-score normalized per channel" and also that
"normalization statistics are computed from the training data only". Those two
readings conflict for test records. This implements the training-statistics
reading, grouped per dataset so that the large scale difference between UCDDB
(airflow in L/min) and MIT-BIH PSG (uncalibrated nasal airflow) does not let one
dataset dominate the other. Recorded as provisional in the config.

Output: artifacts/normalization/norm_stats.json, clipping_report.csv
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
POLICY = json.loads((ROOT / "config" / "policy.json").read_text(encoding="utf-8"))
SIG = ROOT / "artifacts" / "signals"
WIN = ROOT / "artifacts" / "windows"
OUT = ROOT / "artifacts" / "normalization"
OUT.mkdir(parents=True, exist_ok=True)

ECG_FS = POLICY["preprocessing"]["ecg"]["target_fs_hz"]
RESP_FS = POLICY["preprocessing"]["resp"]["target_fs_hz"]
SEEDS = POLICY["splits"]["seeds"]
DURATIONS = [10, 30]
CLIP_C = 5.0
CHUNK = 2000


def load(rec: str, modality: str) -> np.ndarray:
    fs = ECG_FS if modality == "ecg" else RESP_FS
    return np.load(SIG / f"{rec}_{modality}{int(fs)}.npy")


def accumulate(recs: pd.DataFrame, duration: int, dataset: str) -> dict:
    acc = {m: {"n": 0, "sum": 0.0, "sumsq": 0.0} for m in ("ecg", "resp")}
    for rec_id, group in recs.groupby("record_id"):
        starts = group["window_start_s"].to_numpy()
        for modality in ("ecg", "resp"):
            fs = ECG_FS if modality == "ecg" else RESP_FS
            x = load(rec_id, modality).astype(np.float64)
            width = int(round(duration * fs))
            for i in range(0, len(starts), CHUNK):
                chunk = starts[i:i + CHUNK]
                idx = (chunk[:, None] * fs).astype(np.int64) + np.arange(width)[None, :]
                if idx.max() >= len(x):
                    idx = idx[idx.max(axis=1) < len(x)]
                vals = x[idx]
                acc[modality]["n"] += vals.size
                acc[modality]["sum"] += float(vals.sum())
                acc[modality]["sumsq"] += float((vals.astype(np.float64) ** 2).sum())
    out = {}
    for modality, a in acc.items():
        mean = a["sum"] / a["n"]
        var = max(a["sumsq"] / a["n"] - mean * mean, 0.0)
        out[modality] = {"mean": round(mean, 6), "std": round(float(np.sqrt(var)), 6),
                         "n_values": a["n"], "dataset": dataset}
    return out


def clipping_report(stats: dict, all_windows: pd.DataFrame) -> list[dict]:
    rows = []
    for (dataset, modality), st in stats.items():
        z_hi = (CLIP_C * st["std"]) + st["mean"]
        z_lo = (-CLIP_C * st["std"]) + st["mean"]
        rows.append({"dataset": dataset, "modality": modality, "clip_c": CLIP_C,
                     "zscore_value_low": round(z_lo, 4), "zscore_value_high": round(z_hi, 4),
                     "std": st["std"], "mean": st["mean"]})
    return rows


def main() -> None:
    manifest = {d: pd.read_csv(WIN / f"windows_{d}s.csv", low_memory=False) for d in DURATIONS}
    result = {}
    for seed in SEEDS:
        result[str(seed)] = {}
        for duration in DURATIONS:
            df = manifest[duration]
            df = df[(df.seed == seed) & (df.split == "train")]
            result[str(seed)][f"{duration}s"] = {}
            for dataset in ("UCDDB", "MIT-BIH_PSG"):
                sub = df[df.dataset == dataset]
                result[str(seed)][f"{duration}s"][dataset] = accumulate(sub, duration, dataset)
                print(f"seed {seed} {duration}s {dataset}: "
                      f"ecg mean={result[str(seed)][f'{duration}s'][dataset]['ecg']['mean']:.4f} "
                      f"std={result[str(seed)][f'{duration}s'][dataset]['ecg']['std']:.4f} | "
                      f"resp std={result[str(seed)][f'{duration}s'][dataset]['resp']['std']:.4f}",
                      flush=True)

    (OUT / "norm_stats.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    flat = {}
    for seed_stats in result.values():
        for dur_stats in seed_stats.values():
            for dataset, mods in dur_stats.items():
                for modality, st in mods.items():
                    flat.setdefault((dataset, modality), []).append(st)
    averaged = {k: {"mean": float(np.mean([s["mean"] for s in v])),
                    "std": float(np.mean([s["std"] for s in v]))} for k, v in flat.items()}
    with (OUT / "clipping_report.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(clipping_report(averaged, None)[0].keys()))
        w.writeheader()
        w.writerows(clipping_report(averaged, None))
    print(json.dumps({f"{k[0]}/{k[1]}": v for k, v in averaged.items()}, indent=2))


if __name__ == "__main__":
    main()
