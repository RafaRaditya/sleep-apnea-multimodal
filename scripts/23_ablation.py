"""Stage 7d: preprocessing ablation loader.

Experiment 3 of the paper asks for filtering, normalization, and amplitude clipping to be
added/removed one at a time. Everything else must stay fixed: same windows, same splits,
same architecture, same budget. So this module only changes the transform applied when a
window is materialised, never the window list.

Arms (each changes exactly one thing relative to the baseline):

    baseline          main cache, dataset-level z-score, clip +/-5 sigma  (= current pipeline)
    no_clip           same, clip disabled
    clip_per_record   clip threshold scaled to each record's own std
    norm_per_record   z-score using each record's own mean/std instead of the dataset's
    no_filter         unfiltered cache, dataset-level z-score, clip +/-5 sigma

Why `clip_per_record` exists: Stage 6c found that per-record scale ranges 0.55-5.92 in
dataset-std units, so a fixed 5 sigma clip removes 8-10% of respiration samples in
slp02a/slp02b but almost nothing elsewhere. A record-aware threshold is the alternative
that finding points to.

Why `norm_per_record` exists: the paper says "each recording is z-score normalized per
channel" and also "statistics are computed from the training data only". Per-record
normalisation resolves the first reading without leaking val/test statistics, because it
uses only that record's own samples.

Caveat that must travel with `no_filter`: the unfiltered cache is resampled with
resample_poly, which is itself an anti-alias low-pass (ECG to 100 Hz, respiration to
25 Hz). So it is "no added band-pass/low-pass/detrend", not "no filtering".
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[1]
POLICY = json.loads((ROOT / "config" / "policy.json").read_text(encoding="utf-8"))
SIG = ROOT / "artifacts" / "signals"
SIG_UNFILTERED = ROOT / "artifacts" / "signals_unfiltered"

ECG_FS = POLICY["preprocessing"]["ecg"]["target_fs_hz"]
RESP_FS = POLICY["preprocessing"]["resp"]["target_fs_hz"]
DEFAULT_CLIP = 5.0

ARMS = {
    "baseline": {"source": "main", "clip": "fixed", "norm": "dataset"},
    "no_clip": {"source": "main", "clip": None, "norm": "dataset"},
    "clip_per_record": {"source": "main", "clip": "per_record", "norm": "dataset"},
    "norm_per_record": {"source": "main", "clip": "fixed", "norm": "per_record"},
    "no_filter": {"source": "unfiltered", "clip": "fixed", "norm": "dataset"},
}
ARM_NAMES = list(ARMS)


def record_stats(records: list[tuple[str, str]], duration: int,
                 source: str = "main") -> dict:
    """Mean/std per (record, modality), computed from every sample of that record.

    Used by `norm_per_record` and `clip_per_record`. Deliberately label-free and
    split-blind: it uses only the record's own signal, never other records, so it cannot
    leak information from validation or test records into training.
    """
    base = SIG if source == "main" else SIG_UNFILTERED
    out = {}
    for record_id, _dataset in records:
        entry = {}
        for modality, fs in (("ecg", ECG_FS), ("resp", RESP_FS)):
            x = np.load(base / f"{record_id}_{modality}{int(fs)}.npy", mmap_mode="r")
            arr = np.asarray(x, dtype=np.float64)
            entry[modality] = {"mean": float(arr.mean()), "std": float(arr.std())}
        out[record_id] = entry
    return out


class AblationDataset(Dataset):
    def __init__(self, manifest: pd.DataFrame, norm: dict, duration: int, arm: str,
                 stats: dict | None = None, cache_size: int = 8):
        if arm not in ARMS:
            raise ValueError(f"unknown arm {arm}; choose from {ARM_NAMES}")
        self.rows = manifest.reset_index(drop=True)
        self.norm = norm
        self.duration = duration
        self.arm = arm
        self.spec = ARMS[arm]
        self.stats = stats or {}
        self.cache_size = cache_size
        self.base = SIG if self.spec["source"] == "main" else SIG_UNFILTERED
        if not self.base.exists():
            raise FileNotFoundError(
                f"{self.base} is missing; run scripts/22_make_unfiltered_cache.py")
        self._cache: dict[tuple[str, str], np.ndarray] = {}
        self._order: list[tuple[str, str]] = []

    def _signal(self, record_id: str, modality: str) -> np.ndarray:
        key = (record_id, modality)
        arr = self._cache.get(key)
        if arr is None:
            fs = ECG_FS if modality == "ecg" else RESP_FS
            arr = np.load(self.base / f"{record_id}_{modality}{int(fs)}.npy")
            self._cache[key] = arr
            self._order.append(key)
            if len(self._order) > self.cache_size:
                self._cache.pop(self._order.pop(0), None)
        return arr

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows.iloc[idx]
        out = {}
        for modality in ("ecg", "resp"):
            fs = ECG_FS if modality == "ecg" else RESP_FS
            x = self._signal(row.record_id, modality)
            start = int(round(row.window_start_s * fs))
            width = int(round(self.duration * fs))
            seg = x[start:start + width].astype(np.float32)
            if seg.shape[0] < width:
                seg = np.pad(seg, (0, width - seg.shape[0]), mode="constant")

            if self.spec["norm"] == "dataset":
                stats = self.norm[row.dataset][modality]
                mean, std = stats["mean"], stats["std"]
            else:
                stats = self.stats[row.record_id][modality]
                mean, std = stats["mean"], stats["std"]
            seg = (seg - mean) / max(std, 1e-8)

            clip = self.spec["clip"]
            if clip == "fixed":
                seg = np.clip(seg, -DEFAULT_CLIP, DEFAULT_CLIP)
            elif clip == "per_record":
                # Scale shown relative to the dataset std, so a record whose own std is
                # 5.9x the dataset std gets a proportionally wider bound.
                own = self.stats[row.record_id][modality]["std"]
                ds_std = self.norm[row.dataset][modality]["std"]
                ratio = own / max(ds_std, 1e-8)
                bound = DEFAULT_CLIP * max(ratio, 1e-6)
                seg = np.clip(seg, -bound, bound)
            out[modality] = torch.from_numpy(seg).unsqueeze(0)

        out["label"] = torch.tensor(int(row.label), dtype=torch.long)
        return out
