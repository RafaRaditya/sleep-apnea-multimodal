"""Stage 6b: cross-subject score calibration analysis (diagnostic only).

Stage 6 showed that for `proposed` the pooled test AUC (0.526) is far below the mean
per-subject AUC (0.733). This script characterises that gap instead of hiding it.

Three questions it answers:
  1. How large is the pooled-vs-per-subject gap on validation and on test, per model?
     If the gap is small on validation and large on test, the problem is the test
     subjects, not the architecture.
  2. Is the offset related to subject prevalence? That is, does the model give the
     highest scores to the subject with the lowest apnea rate (a prior inversion)?
  3. Would a label-free per-record score normalisation close the gap? This is reported
     as a SECONDARY, EXPLORATORY analysis: the choice to run it was made after seeing
     test results, so it cannot be presented as a pre-registered protocol. The primary
     metric stays the pooled AUC with the frozen validation threshold.

Everything here is post-hoc and uses no test labels to define the transform (z-scoring
is per record, label-free). It still cannot be used to select the model.

Run:
    python scripts/17_calibration.py

Outputs (artifacts/calibration/):
    calibration_summary.csv, calibration_by_subject.csv, calibration_run.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
TRAIN_DIR = ROOT / "artifacts" / "train"
TEST_DIR = ROOT / "artifacts" / "test"
OUT = ROOT / "artifacts" / "calibration"
MODELS = ["proposed", "simple_fusion", "ecg_only", "resp_only"]


def auc_or_nan(y: np.ndarray, s: np.ndarray) -> float:
    return float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else float("nan")


def zscore_grouped(df: pd.DataFrame, key: str) -> pd.Series:
    return df.groupby(key).score.transform(lambda s: (s - s.mean()) / (s.std() + 1e-8))


def describe(df: pd.DataFrame, label: str) -> dict:
    per_subject = [auc_or_nan(g.label.to_numpy(), g.score.to_numpy())
                   for _, g in df.groupby("subject_id") if g.label.nunique() == 2]
    pooled = auc_or_nan(df.label.to_numpy(), df.score.to_numpy())
    within = auc_or_nan(df.label.to_numpy(), zscore_grouped(df, "subject_id").to_numpy())
    per_record = [auc_or_nan(g.label.to_numpy(), g.score.to_numpy())
                  for _, g in df.groupby("record_id") if g.label.nunique() == 2]
    means = df.groupby("subject_id").score.mean()
    return {
        "split": label,
        "pooled_auc": round(pooled, 4),
        "mean_subject_auc": round(float(np.mean(per_subject)), 4),
        "mean_record_auc": round(float(np.mean(per_record)), 4),
        "gap": round(float(np.mean(per_subject)) - pooled, 4),
        "within_subject_auc": round(within, 4),
        "subject_score_spread": round(float(means.max() - means.min()), 4),
    }


def bootstrap_subject_auc(df: pd.DataFrame, key: str | None, n_boot: int, seed: int) -> dict:
    """Resample subjects with replacement; window-level overlap is ignored on purpose
    so the interval is not narrower than the data supports. Six subjects means this
    interval is wide and should be read as a warning, not a precise estimate."""
    rng = np.random.default_rng(seed)
    subjects = sorted(df.subject_id.unique())
    scores = df.score.to_numpy() if key is None else zscore_grouped(df, key).to_numpy()
    labels = df.label.to_numpy()
    by_subject = {s: np.where(df.subject_id.to_numpy() == s)[0] for s in subjects}

    vals = []
    for _ in range(n_boot):
        pick = rng.choice(subjects, size=len(subjects), replace=True)
        idx = np.concatenate([by_subject[s] for s in pick])
        a = auc_or_nan(labels[idx], scores[idx])
        if np.isfinite(a):
            vals.append(a)
    if not vals:
        return {"n_boot": 0}
    return {"median": round(float(np.median(vals)), 4),
            "lo95": round(float(np.percentile(vals, 2.5)), 4),
            "hi95": round(float(np.percentile(vals, 97.5)), 4),
            "n_effective_subjects": len(subjects)}


def prevalence_correlation(df: pd.DataFrame) -> dict:
    """Spearman(prevalence, mean score) across subjects, plus leave-one-out stability.
    Subjects with zero positives stay in: their mean score is real evidence, and dropping
    them silently is exactly how slp41 escaped every AUC table."""
    g = df.groupby("subject_id").agg(prevalence=("label", "mean"), mean_score=("score", "mean"))
    rho, p = spearmanr(g.prevalence, g.mean_score)
    basis = {"n_subjects": len(g), "spearman": round(float(rho), 3), "p": round(float(p), 3)}
    if len(g) > 3:
        loo = []
        for drop in g.index:
            r, _ = spearmanr(g.drop(drop).prevalence, g.drop(drop).mean_score)
            loo.append(float(r))
        basis["leave_one_out_range"] = [round(min(loo), 3), round(max(loo), 3)]
    return basis


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--duration", type=int, default=10)
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    tr = TRAIN_DIR / f"seed{args.seed}_{args.duration}s"
    te = TEST_DIR / f"seed{args.seed}_{args.duration}s"
    frozen = json.loads((tr / "frozen.json").read_text(encoding="utf-8"))

    rows, subject_rows, boot_rows, transform_rows, corr_rows = [], [], [], [], []
    for name in MODELS:
        val = pd.read_csv(tr / f"val_predictions_{name}.csv")
        test = pd.read_csv(te / f"test_predictions_{name}.csv")
        threshold = frozen["models"][name]["threshold"]

        for split, df in (("val", val), ("test", test)):
            rows.append({"model": name, **describe(df, split)})

        # question 2: does the offset track prevalence?
        g = test.groupby(["dataset", "subject_id"]).agg(
            windows=("label", "size"), positive_rate=("label", "mean"),
            score_mean=("score", "mean"), score_std=("score", "std"),
            pred_positive_rate=("pred", "mean")).reset_index()
        rho, p = spearmanr(g.positive_rate, g.score_mean)
        g["model"] = name
        g["spearman_prev_vs_score"] = round(float(rho), 3)
        g["spearman_p"] = round(float(p), 3)
        g["threshold"] = threshold
        subject_rows.append(g)

        # correlation on both splits, zero-prevalence subjects kept on purpose
        for split, df in (("val", val), ("test", test)):
            corr_rows.append({"model": name, "split": split, **prevalence_correlation(df)})

        # question 3: label-free per-record / per-subject normalisation.
        # Applied to test AND to validation: if the transform hurts validation, it is a
        # post-hoc rescue rather than a principled improvement, and must be said so.
        for key, label in ((None, "none"), ("record_id", "z-score per record"),
                           ("subject_id", "z-score per subject")):
            val_s = (zscore_grouped(val, key) if key else val.score).to_numpy()
            test_s = (zscore_grouped(test, key) if key else test.score).to_numpy()
            transform_rows.append({
                "model": name, "transform": label,
                "val_pooled_auc": round(auc_or_nan(val.label.to_numpy(), val_s), 4),
                "test_pooled_auc": round(auc_or_nan(test.label.to_numpy(), test_s), 4),
            })
            boot_rows.append({
                "model": name, "transform": label,
                **bootstrap_subject_auc(test, key, args.n_boot, args.seed),
            })

    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "calibration_summary.csv", index=False)
    by_subject = pd.concat(subject_rows, ignore_index=True)
    by_subject.to_csv(OUT / "calibration_by_subject.csv", index=False)
    transforms = pd.DataFrame(transform_rows)
    transforms.to_csv(OUT / "calibration_transforms.csv", index=False)
    boot = pd.DataFrame(boot_rows)
    boot.to_csv(OUT / "calibration_bootstrap.csv", index=False)
    corr = pd.DataFrame(corr_rows)
    corr.to_csv(OUT / "calibration_prevalence.csv", index=False)

    pd.set_option("display.width", 220)
    print("=== A. pooled vs per-subject gap (question 1) ===")
    print(summary.to_string(index=False))

    print("\n=== B. per-subject prevalence vs mean score, test (question 2) ===")
    print(by_subject[["model", "dataset", "subject_id", "windows", "positive_rate",
                      "score_mean", "pred_positive_rate", "threshold"]]
          .to_string(index=False))
    print("\nSpearman(prevalence, mean score), zero-prevalence subjects kept:")
    print(corr.to_string(index=False))
    print("\nZero-prevalence subjects are the trap: slp41 (val) and any test equivalent have no "
          "defined AUC, so they vanish from every AUC table while still carrying a mean score.")

    print("\n=== C. label-free z-scoring (question 3, SECONDARY/EXPLORATORY) ===")
    print(transforms.to_string(index=False))

    print("\n=== D. subject-level bootstrap 95% CI, test ===")
    print(boot.to_string(index=False))

    (OUT / "calibration_run.json").write_text(json.dumps({
        "stage": "6b",
        "run_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seed": args.seed, "duration_s": args.duration,
        "source": {"val": str(tr.relative_to(ROOT)) if tr.is_relative_to(ROOT) else str(tr),
                   "test": str(te.relative_to(ROOT)) if te.is_relative_to(ROOT) else str(te)},
        "status": "post-hoc diagnostic; the per-record normalisation was chosen after "
                  "seeing test results and is reported as secondary, never as the headline",
        "n_boot": args.n_boot,
        "caution": "bootstrap resamples 6 subjects and ignores within-subject autocorrelation, "
                   "so intervals are optimistic in width and must not be read as power",
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
