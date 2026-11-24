"""Stage 3b-2: windowed dataset and loader smoke test.

Slices filtered signals on demand using the Stage 2 window manifest, applies
training-only z-score statistics, then amplitude clipping. Nothing is cached to
disk, so memory stays flat regardless of window count.

Run directly for a smoke test of shapes, label agreement with the manifest,
normalization provenance, and throughput.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[1]
POLICY = json.loads((ROOT / "config" / "policy.json").read_text(encoding="utf-8"))
SIG = ROOT / "artifacts" / "signals"
WIN = ROOT / "artifacts" / "windows"
NORM = ROOT / "artifacts" / "normalization" / "norm_stats.json"

ECG_FS = POLICY["preprocessing"]["ecg"]["target_fs_hz"]
RESP_FS = POLICY["preprocessing"]["resp"]["target_fs_hz"]
CLIP_C = 5.0


class WindowDataset(Dataset):
    def __init__(self, manifest: pd.DataFrame, norm: dict, duration: int,
                 clip_c: float = CLIP_C, cache_size: int = 8):
        self.rows = manifest.reset_index(drop=True)
        self.norm = norm
        self.duration = duration
        self.clip_c = clip_c
        self.cache_size = cache_size
        self._cache: dict[tuple[str, str], np.ndarray] = {}
        self._order: list[tuple[str, str]] = []

    def _signal(self, record_id: str, dataset: str, modality: str) -> np.ndarray:
        key = (record_id, modality)
        arr = self._cache.get(key)
        if arr is None:
            fs = ECG_FS if modality == "ecg" else RESP_FS
            arr = np.load(SIG / f"{record_id}_{modality}{int(fs)}.npy")
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
            x = self._signal(row.record_id, row.dataset, modality)
            start = int(round(row.window_start_s * fs))
            width = int(round(self.duration * fs))
            seg = x[start:start + width].astype(np.float32)
            if seg.shape[0] < width:
                seg = np.pad(seg, (0, width - seg.shape[0]), mode="constant")
            stats = self.norm[row.dataset][modality]
            seg = (seg - stats["mean"]) / max(stats["std"], 1e-8)
            seg = np.clip(seg, -self.clip_c, self.clip_c)
            out[modality] = torch.from_numpy(seg).unsqueeze(0)

        out["label"] = torch.tensor(int(row.label), dtype=torch.long)
        return out


def load_norm(seed: int, duration: int) -> dict:
    raw = json.loads(NORM.read_text(encoding="utf-8"))
    return raw[str(seed)][f"{duration}s"]


def smoke(seed: int = 42, duration: int = 10, n: int = 512) -> None:
    path = WIN / f"windows_{duration}s.csv"
    manifest = pd.read_csv(path, low_memory=False)
    norm = load_norm(seed, duration)

    print(f"=== smoke test: seed={seed}, window={duration}s ===")
    print("manifest rows:", len(manifest))

    checks = []
    for split in ("train", "val", "test"):
        sub = manifest[(manifest.seed == seed) & (manifest.split == split)]
        ds = WindowDataset(sub.iloc[:n], norm, duration)
        item = ds[0]
        shapes = {k: tuple(v.shape) for k, v in item.items() if k != "label"}
        checks.append({
            "split": split,
            "windows": len(sub),
            "subjects": sub.subject_id.nunique(),
            "records": sub.record_id.nunique(),
            "ecg_shape": shapes["ecg"],
            "resp_shape": shapes["resp"],
            "label_dtype": str(item["label"].dtype),
        })
    print(pd.DataFrame(checks).to_string(index=False))

    # expected sample counts
    exp_ecg = int(duration * ECG_FS)
    exp_resp = int(duration * RESP_FS)
    assert checks[0]["ecg_shape"] == (1, exp_ecg), checks[0]["ecg_shape"]
    assert checks[0]["resp_shape"] == (1, exp_resp), checks[0]["resp_shape"]
    print(f"shape contract ok: ecg=(1,{exp_ecg}) resp=(1,{exp_resp})")

    # label parity + value range over a larger sample
    sub = manifest[(manifest.seed == seed) & (manifest.split == "test")]
    ds = WindowDataset(sub, norm, duration)
    labels = np.array([int(ds[i]["label"]) for i in range(min(len(ds), 2000))])
    expected = sub.label.to_numpy()[:len(labels)]
    print("label parity:", bool((labels == expected).all()),
          f"| positive rate sampled: {labels.mean():.3f}")

    batch = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=False)
    b = next(iter(batch))
    print("batch ecg:", tuple(b["ecg"].shape), "resp:", tuple(b["resp"].shape),
          "label:", tuple(b["label"].shape))
    print("value range after norm+clip: ecg [{:.2f}, {:.2f}] resp [{:.2f}, {:.2f}]".format(
        b["ecg"].min(), b["ecg"].max(), b["resp"].min(), b["resp"].max()))
    print("any nan:", bool(torch.isnan(b["ecg"]).any() or torch.isnan(b["resp"]).any()))

    # how much clipping actually bites
    ecg_all = torch.cat([ds[i]["ecg"].flatten() for i in range(min(len(ds), 300))])
    resp_all = torch.cat([ds[i]["resp"].flatten() for i in range(min(len(ds), 300))])
    print(f"clipped at +/-{CLIP_C}: ecg {float((ecg_all.abs() >= CLIP_C - 1e-6).float().mean()):.4f}, "
          f"resp {float((resp_all.abs() >= CLIP_C - 1e-6).float().mean()):.4f}")

    t0 = time.perf_counter()
    count = 0
    for batch in batch:
        count += batch["label"].shape[0]
    dt = time.perf_counter() - t0
    print(f"throughput: {count / dt:.0f} windows/s (1 process, batch 64)")


if __name__ == "__main__":
    import sys
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    duration = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    smoke(seed, duration)
