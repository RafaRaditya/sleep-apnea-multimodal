"""Verify windowing end-to-end: manifest coverage, real slicing, tail handling."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SIG = ROOT / "artifacts" / "signals"
WIN = ROOT / "artifacts" / "windows"
POLICY = json.loads((ROOT / "config" / "policy.json").read_text(encoding="utf-8"))
ECG_FS = POLICY["preprocessing"]["ecg"]["target_fs_hz"]
RESP_FS = POLICY["preprocessing"]["resp"]["target_fs_hz"]

print("=== 1. every window must fit inside its signal (no padding) ===")
for duration in (10, 30):
    df = pd.read_csv(WIN / f"windows_{duration}s.csv", low_memory=False)
    bad, per_rec = [], []
    for rec, g in df[df.seed == 42].groupby("record_id"):
        ecg_len = len(np.load(SIG / f"{rec}_ecg100.npy"))
        resp_len = len(np.load(SIG / f"{rec}_resp25.npy"))
        ecg_cap = ecg_len / ECG_FS
        resp_cap = resp_len / RESP_FS
        over = g[g.window_end_s > min(ecg_cap, resp_cap) + 1e-6]
        if len(over):
            bad.append((rec, len(over), float(over.window_end_s.max()), min(ecg_cap, resp_cap)))
        per_rec.append({"record": rec, "windows": len(g),
                        "last_end": float(g.window_end_s.max()),
                        "signal_cap": round(min(ecg_cap, resp_cap), 2),
                        "unused_tail_s": round(min(ecg_cap, resp_cap) - g.window_end_s.max(), 2)})
    print(f"{duration}s: windows exceeding signal = {len(bad)}", bad[:3])
    pr = pd.DataFrame(per_rec)
    print(f"  unused tail: median={pr.unused_tail_s.median():.1f}s max={pr.unused_tail_s.max():.1f}s")
    print(f"  records with tail >= window length: "
          f"{int((pr.unused_tail_s >= duration).sum())} of {len(pr)}")

print("\n=== 2. real slicing check: index arithmetic ===")
df = pd.read_csv(WIN / "windows_10s.csv", low_memory=False)
row = df[(df.seed == 42) & (df.record_id == "ucddb002")].iloc[700]
ecg = np.load(SIG / "ucddb002_ecg100.npy")
resp = np.load(SIG / "ucddb002_resp25.npy")
s = int(round(row.window_start_s * ECG_FS))
seg = ecg[s:s + 1000]
print(f"window start={row.window_start_s}s end={row.window_end_s}s label={row.label}")
print(f"ecg slice samples [{s}:{s+1000}] len={len(seg)} "
      f"mean={seg.mean():.4f} std={seg.std():.4f}")
r = int(round(row.window_start_s * RESP_FS))
print(f"resp slice samples [{r}:{r+250}] len={len(resp[r:r+250])} "
      f"std={resp[r:r+250].std():.4f}")

print("\n=== 3. temporal alignment: resp slice must equal ECG slice rescaled in time ===")
for rec in ("ucddb002", "slp02a"):
    e = np.load(SIG / f"{rec}_ecg100.npy")
    rr = np.load(SIG / f"{rec}_resp25.npy")
    if len(e) / ECG_FS > len(rr) / RESP_FS:
        print(f"  {rec}: ecg longer by {len(e)/ECG_FS - len(rr)/RESP_FS:.3f}s")
    elif len(rr) / RESP_FS > len(e) / ECG_FS:
        print(f"  {rec}: resp longer by {len(rr)/RESP_FS - len(e)/ECG_FS:.3f}s")
    else:
        print(f"  {rec}: equal duration {len(e)/ECG_FS:.2f}s")

print("\n=== 4. label spot-check against source annotations ===")
# ucddb002 first annotated event is at 00:29:13 with duration 16 s, record start 00:11:04
offset = 1089.0
g = df[(df.seed == 42) & (df.record_id == "ucddb002")]
hits = g[(g.window_start_s < offset + 16) & (g.window_end_s > offset)]
print(f"windows overlapping the event at t={offset}s: {len(hits)}")
print(hits[["window_start_s", "window_end_s", "label", "positive_overlap_fraction"]].to_string(index=False))

print("\n=== 5. stride policy actually used ===")
for duration in (10, 30):
    d = pd.read_csv(WIN / f"windows_{duration}s.csv", low_memory=False)
    d = d[(d.seed == 42) & (d.record_id == "ucddb002")]
    for split in ("train", "val", "test"):
        g = d[d.split == split].sort_values("window_start_s")
        diffs = np.diff(g.window_start_s.to_numpy())
        uniq = np.unique(np.round(diffs, 2))
        print(f"{duration}s {split:<6} stride(s)={uniq[:4]} n={len(g)}")
