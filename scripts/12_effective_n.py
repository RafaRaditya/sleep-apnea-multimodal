"""Autocorrelation of the per-window label series -> effective sample size.

Windows along a recording are strongly dependent; treating 80k windows as 80k
independent samples overstates evidence. This estimates N_eff with the standard
variance-inflation correction N_eff = N / (1 + 2*sum(rho_k)).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
WIN = ROOT / "artifacts" / "windows"


def acf(x: np.ndarray, nlags: int) -> np.ndarray:
    x = x - x.mean()
    denom = np.dot(x, x)
    if denom == 0:
        return np.zeros(nlags + 1)
    return np.array([np.dot(x[:-k], x[k:]) / denom if k else 1.0 for k in range(nlags + 1)])


def n_eff_per_series(x: np.ndarray, max_lag: int = 200) -> tuple[float, float]:
    rho = acf(x.astype(float), max_lag)
    # first lag where ACF goes non-positive, capped
    cut = max_lag
    for k in range(1, len(rho)):
        if rho[k] <= 0:
            cut = k
            break
    tau = 1.0 + 2.0 * rho[1:cut].sum()
    return len(x) / max(tau, 1.0), tau


print(f"{'duration':<9}{'dataset':<13}{'split':<7}{'windows':>8}{'N_eff':>10}{'tau':>8}{'N_eff/subj':>12}")
for duration in (10, 30):
    d = pd.read_csv(WIN / f"windows_{duration}s.csv", low_memory=False)
    d = d[d.seed == 42]
    for dataset in ("UCDDB", "MIT-BIH_PSG"):
        for split in ("train", "val", "test"):
            s = d[(d.dataset == dataset) & (d.split == split)]
            if not len(s):
                continue
            n_eff_total, taus, n_subj = 0.0, [], 0
            for _, g in s.groupby("record_id"):
                g = g.sort_values("window_start_s")
                ne, tau = n_eff_per_series(g.label.to_numpy())
                n_eff_total += ne
                taus.append(tau)
            n_subj = s.subject_id.nunique()
            print(f"{duration:<9}{dataset:<13}{split:<7}{len(s):>8}{n_eff_total:>10.0f}"
                  f"{np.mean(taus):>8.1f}{n_eff_total / n_subj:>12.0f}")

print("\n-- positive-count reality check (10 s, seed 42) --")
d = pd.read_csv(WIN / "windows_10s.csv", low_memory=False)
d = d[d.seed == 42]
for split in ("train", "val", "test"):
    s = d[d.split == split]
    print(f"{split:<6} windows={len(s):>7}  positives={int(s.label.sum()):>6}  "
          f"positive subjects={s[s.label == 1].subject_id.nunique():>3}/{s.subject_id.nunique()}")

print("\n-- how many distinct apnea episodes do the positives represent? --")
# count label runs (contiguous positive windows) per recording as a proxy for events
runs_total, run_pos = 0, 0
for dataset in ("UCDDB", "MIT-BIH_PSG"):
    for split in ("train", "val", "test"):
        s = d[(d.dataset == dataset) & (d.split == split)]
        for _, g in s.groupby("record_id"):
            lab = g.sort_values("window_start_s").label.to_numpy()
            starts = np.sum((lab[1:] == 1) & (lab[:-1] == 0)) + (1 if lab[0] == 1 else 0)
            runs_total += starts
            if split == "test":
                run_pos += starts
print(f"contiguous positive runs: all splits={runs_total}, test split={run_pos}")
