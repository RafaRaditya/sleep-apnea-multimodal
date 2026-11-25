"""Stage 7b: aggregate the per-seed test results into mean +/- SD.

Stage 7a aggregated validation. This aggregates test, which each seed reads exactly
once with frozen checkpoints and frozen validation thresholds. No model, epoch, or
threshold is selected here; the numbers are reported as they came out.

Also recomputes a subject-level bootstrap per seed from the stored predictions, so the
interval reported alongside a seed comes from that seed's own six test subjects.

Run:
    python scripts/20_aggregate_test.py
    python scripts/20_aggregate_test.py --seeds 42,1337,2024 --duration 10

Outputs (artifacts/aggregate/seed<list>_<duration>s/):
    test_summary_mean_sd.csv, test_by_dataset_mean_sd.csv,
    test_auc_decomposition_mean_sd.csv, test_bootstrap_by_seed.csv,
    test_per_seed.csv
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
TEST_DIR = ROOT / "artifacts" / "test"
OUT_ROOT = ROOT / "artifacts" / "aggregate"
MODELS = ["proposed", "simple_fusion", "ecg_only", "resp_only"]
METRICS = ["accuracy", "sensitivity", "specificity", "precision", "f1", "roc_auc"]


def auc_or_nan(y: np.ndarray, s: np.ndarray) -> float:
    return float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else float("nan")


def bootstrap_subject_auc(pred: pd.DataFrame, n_boot: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    subjects = sorted(pred.subject_id.unique())
    labels = pred.label.to_numpy()
    scores = pred.score.to_numpy()
    by_subject = {s: np.where(pred.subject_id.to_numpy() == s)[0] for s in subjects}
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(subjects, size=len(subjects), replace=True)
        idx = np.concatenate([by_subject[s] for s in pick])
        a = auc_or_nan(labels[idx], scores[idx])
        if np.isfinite(a):
            vals.append(a)
    if not vals:
        return {"n_boot": 0}
    return {"n_subjects": len(subjects),
            "median": round(float(np.median(vals)), 4),
            "lo95": round(float(np.percentile(vals, 2.5)), 4),
            "hi95": round(float(np.percentile(vals, 97.5)), 4)}


def load_seed(seed: int, duration: int, n_boot: int) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    run = TEST_DIR / f"seed{seed}_{duration}s"
    if not run.exists():
        raise SystemExit(f"missing test results for seed {seed}: {run}")
    rows, dataset_rows, decomp_rows, boot_rows = [], [], [], []
    for name in MODELS:
        path = run / f"metrics_test_{name}.json"
        if not path.exists():
            raise SystemExit(f"missing metrics: {path}")
        m = json.loads(path.read_text(encoding="utf-8"))
        row = {"seed": seed, "model": name, "params": m["params"],
               "threshold": m["frozen_threshold"], "best_epoch": m["best_epoch"],
               "test_windows": m["test_windows"], "test_subjects": m["test_subjects"],
               "test_positive_rate": m["test_positive_rate"],
               "inference_ms_per_window": m["inference_ms_per_window"]}
        row.update({f"test_{k}": m["test_overall"][k] for k in METRICS})
        rows.append(row)

        for dataset, dm in m["test_by_dataset"].items():
            dataset_rows.append({"seed": seed, "model": name, "dataset": dataset,
                                 "roc_auc": dm["roc_auc"], "f1": dm["f1"],
                                 "sensitivity": dm["sensitivity"],
                                 "specificity": dm["specificity"]})

        d = m["auc_decomposition"]
        decomp_rows.append({"seed": seed, "model": name, **d})

        pred = pd.read_csv(run / f"test_predictions_{name}.csv")
        boot_rows.append({"seed": seed, "model": name,
                          **bootstrap_subject_auc(pred, n_boot, seed)})
    return rows, dataset_rows, decomp_rows, boot_rows


def mean_sd(df: pd.DataFrame, group: list[str], cols: list[str]) -> pd.DataFrame:
    out = df.groupby(group)[cols].agg(["mean", "std"])
    out.columns = [f"{base}_{stat}" for base, stat in out.columns]
    out = out.reset_index()
    for c in cols:
        out[c] = out[f"{c}_mean"].round(4)
        out[f"{c}_sd"] = out[f"{c}_std"].round(4)
        out = out.drop(columns=[f"{c}_mean", f"{c}_std"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,1337,2024")
    ap.add_argument("--duration", type=int, default=10)
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    rows, dataset_rows, decomp_rows, boot_rows = [], [], [], []
    for seed in seeds:
        r, d, c, b = load_seed(seed, args.duration, args.n_boot)
        rows += r
        dataset_rows += d
        decomp_rows += c
        boot_rows += b

    per_seed = pd.DataFrame(rows)
    out = OUT_ROOT / f"seed{'-'.join(str(s) for s in seeds)}_{args.duration}s"
    out.mkdir(parents=True, exist_ok=True)

    test_cols = [f"test_{k}" for k in METRICS]
    summary = mean_sd(per_seed, ["model"], test_cols + ["threshold"])
    summary = summary.sort_values("test_roc_auc", ascending=False)
    summary.to_csv(out / "test_summary_mean_sd.csv", index=False)

    ds = pd.DataFrame(dataset_rows)
    ds_agg = mean_sd(ds, ["model", "dataset"], ["roc_auc", "f1"]).sort_values(["model", "dataset"])
    ds_agg.to_csv(out / "test_by_dataset_mean_sd.csv", index=False)

    decomp = pd.DataFrame(decomp_rows)
    decomp_agg = mean_sd(decomp, ["model"],
                         ["pooled_auc", "mean_subject_auc", "within_subject_auc"])
    decomp_agg.to_csv(out / "test_auc_decomposition_mean_sd.csv", index=False)

    boot = pd.DataFrame(boot_rows)
    boot.to_csv(out / "test_bootstrap_by_seed.csv", index=False)

    per_seed.to_csv(out / "test_per_seed.csv", index=False)

    pd.set_option("display.width", 250)
    print(f"=== test, seeds {seeds}, {args.duration}s: mean +/- SD ===")
    print(summary[["model", "test_roc_auc", "test_roc_auc_sd", "test_f1", "test_f1_sd",
                   "test_sensitivity", "test_sensitivity_sd",
                   "test_specificity", "test_specificity_sd",
                   "threshold", "threshold_sd"]].to_string(index=False))
    print("\n=== per-dataset test ROC-AUC (mean +/- SD across seeds) ===")
    print(ds_agg.to_string(index=False))
    print("\n=== pooled vs per-subject AUC (mean across seeds) ===")
    print(decomp_agg.to_string(index=False))
    print("\n=== subject-level bootstrap 95% CI, per seed ===")
    print(boot[["seed", "model", "n_subjects", "median", "lo95", "hi95"]].to_string(index=False))
    print("\n=== test ROC-AUC per seed ===")
    print(per_seed[["seed", "model", "test_roc_auc", "test_f1", "threshold"]]
          .to_string(index=False))

    (out / "test_aggregate_run.json").write_text(json.dumps({
        "stage": "7b",
        "run_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seeds": seeds, "duration_s": args.duration,
        "source": [f"artifacts/test/seed{s}_{args.duration}s" for s in seeds],
        "summary": "mean and sample SD (ddof=1) across seeds",
        "n_boot": args.n_boot,
        "caution": "each seed has 6 test subjects; bootstrap intervals are wide and ignore "
                   "within-subject autocorrelation, so they are optimistic in width",
    }, indent=2), encoding="utf-8")

    print(f"\nwrote -> {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out}")


if __name__ == "__main__":
    main()
