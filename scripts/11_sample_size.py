"""How much data is there really? Raw counts, per-subject counts, and an
autocorrelation-adjusted effective sample size."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
WIN = ROOT / "artifacts" / "windows"
SPLIT = ROOT / "artifacts" / "splits" / "split_manifest.csv"

for duration in (10, 30):
    d = pd.read_csv(WIN / f"windows_{duration}s.csv", low_memory=False)
    d = d[d.seed == 42]
    print(f"\n================ {duration}s windows, seed 42 ================")

    g = d.groupby(["dataset", "split"]).agg(
        windows=("label", "size"),
        subjects=("subject_id", "nunique"),
        records=("record_id", "nunique"),
        positives=("label", "sum"),
    ).reset_index()
    g["win_per_subject"] = (g.windows / g.subjects).round(0)
    g["pos_per_subject"] = (g.positives / g.subjects).round(0)
    print(g.to_string(index=False))

    print("\n-- per-subject detail (test split) --")
    t = d[d.split == "test"].groupby(["dataset", "subject_id"]).agg(
        records=("record_id", "nunique"),
        windows=("label", "size"),
        positive=("label", "sum"),
        pos_rate=("label", "mean"),
    ).reset_index()
    print(t.to_string(index=False))

    print("\n-- independent segments: windows are not independent samples --")
    # with 50% train overlap, one window's information overlaps its neighbour
    for dataset in ("UCDDB", "MIT-BIH_PSG"):
        for split in ("train", "val", "test"):
            s = d[(d.dataset == dataset) & (d.split == split)]
            if not len(s):
                continue
            hours = (s.groupby("record_id").window_end_s.max().sum())
            # non-overlapping equivalents
            nonoverlap = hours / duration
            print(f"{dataset:<12} {split:<6} window-hours={hours:8.0f}  "
                  f"raw windows={len(s):7d}  non-overlapping equivalents={nonoverlap:8.0f}")
