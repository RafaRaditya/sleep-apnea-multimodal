"""Stage 4b: diagnostic for the pilot checkpoints.

Separates "pipeline broken" from "model has not generalized yet" by scoring the
same held-out-style subsets with the saved pilot checkpoints. Reports AUC on a
training subset and on the validation split, plus per-subject positive rates and
per-subject AUC. No test split is read.

Interpretation guide: a training AUC well above the validation AUC means the
model memorizes subject-specific waveform patterns rather than apnea morphology.
That is a budget/regularization issue for Stage 5, not a plumbing bug.

Output: artifacts/pilot/pilot_diagnostics.csv, artifacts/pilot/pilot_subject_auc.csv
"""

from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models import MODEL_NAMES, build  # noqa: E402

dataset_mod = import_module("08_dataset")
WindowDataset = dataset_mod.WindowDataset
load_norm = dataset_mod.load_norm

ROOT = Path(__file__).resolve().parents[1]
WIN = ROOT / "artifacts" / "windows"
PILOT = ROOT / "artifacts" / "pilot"
SEED = 42
DURATION = 10
N = 3000


def subset(manifest: pd.DataFrame, split: str, n: int, rng) -> pd.DataFrame:
    pool = manifest[(manifest.seed == SEED) & (manifest.split == split)]
    idx = np.sort(rng.choice(len(pool), size=min(n, len(pool)), replace=False))
    return pool.iloc[idx]


def scores(model_name: str, rows: pd.DataFrame, norm: dict) -> tuple[np.ndarray, np.ndarray]:
    ds = WindowDataset(rows, norm, DURATION, cache_size=64)
    ecg = torch.stack([ds[i]["ecg"] for i in range(len(ds))])
    resp = torch.stack([ds[i]["resp"] for i in range(len(ds))])
    y = np.array([int(ds[i]["label"]) for i in range(len(ds))])

    model = build(model_name, DURATION)
    state = torch.load(PILOT / "checkpoints" / f"{model_name}.pt", map_location="cpu")
    model.load_state_dict(state["state_dict"])
    model.eval()
    with torch.no_grad():
        p = torch.cat([torch.softmax(model(ecg[i:i + 256], resp[i:i + 256]), dim=1)[:, 1]
                       for i in range(0, len(y), 256)]).numpy()
    return y, p


def main() -> None:
    manifest = pd.read_csv(WIN / f"windows_{DURATION}s.csv", low_memory=False)
    norm = load_norm(SEED, DURATION)
    rng = np.random.default_rng(SEED)
    rows = {s: subset(manifest, s, N, rng) for s in ("train", "val")}

    summary, per_subject = [], []
    for name in MODEL_NAMES:
        for split, sub in rows.items():
            y, p = scores(name, sub, norm)
            summary.append({
                "model": name, "split": split, "windows": len(y),
                "subjects": int(sub.subject_id.nunique()),
                "positive_rate": round(float(y.mean()), 4),
                "roc_auc": round(float(roc_auc_score(y, p)), 4),
                "mean_score_positive": round(float(p[y == 1].mean()), 4) if (y == 1).any() else None,
                "mean_score_negative": round(float(p[y == 0].mean()), 4) if (y == 0).any() else None,
                "score_min": round(float(p.min()), 4), "score_max": round(float(p.max()), 4),
            })
            for subject, grp in sub.groupby("subject_id"):
                sy = grp.label.to_numpy()
                sp = p[grp.index.map(lambda i: sub.index.get_loc(i))]
                auc = (round(float(roc_auc_score(sy, sp)), 4)
                       if len(np.unique(sy)) == 2 else None)
                per_subject.append({
                    "model": name, "split": split, "subject": subject,
                    "dataset": grp.dataset.iloc[0], "windows": len(grp),
                    "positive_rate": round(float(sy.mean()), 4), "roc_auc": auc,
                })

    summary_df = pd.DataFrame(summary)
    subject_df = pd.DataFrame(per_subject)
    summary_df.to_csv(PILOT / "pilot_diagnostics.csv", index=False)
    subject_df.to_csv(PILOT / "pilot_subject_auc.csv", index=False)
    print(summary_df.to_string(index=False))
    print("\nval split per subject:")
    print(subject_df[subject_df.split == "val"].to_string(index=False))


if __name__ == "__main__":
    main()
