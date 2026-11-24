"""Stage 7a: aggregate the per-seed validation results into mean +/- SD.

Section 3.5.2 asks for repeated random seeds reported as mean +/- standard deviation.
Each seed is a separate subject-level split, so the spread across seeds measures how
sensitive the conclusions are to which subjects land in validation.

Only validation outputs are read here. Test predictions are aggregated separately
(stage 7b) and are never used to choose a model, threshold, or hyperparameter.

Run:
    python scripts/19_aggregate.py
    python scripts/19_aggregate.py --seeds 42,1337,2024 --duration 10

Outputs (artifacts/aggregate/seed<list>_<duration>s/):
    val_summary_mean_sd.csv, val_by_dataset_mean_sd.csv, val_ranking.csv,
    val_per_seed.csv, aggregate_run.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TRAIN_DIR = ROOT / "artifacts" / "train"
OUT_ROOT = ROOT / "artifacts" / "aggregate"
MODELS = ["proposed", "simple_fusion", "ecg_only", "resp_only"]
METRICS = ["accuracy", "sensitivity", "specificity", "precision", "f1", "roc_auc"]


def load_seed(seed: int, duration: int) -> list[dict]:
    run = TRAIN_DIR / f"seed{seed}_{duration}s"
    if not (run / "frozen.json").exists():
        raise SystemExit(f"missing freeze record for seed {seed}: {run / 'frozen.json'}")
    rows = []
    for name in MODELS:
        path = run / f"metrics_{name}.json"
        if not path.exists():
            raise SystemExit(f"missing metrics: {path}")
        m = json.loads(path.read_text(encoding="utf-8"))
        row = {"seed": seed, "model": name, "params": m["params"],
               "epochs_run": m["epochs_run"], "best_epoch": m["best_epoch"],
               "best_val_loss": m["best_val_loss"], "threshold": m["threshold"],
               "val_windows": m["val_windows"], "val_subjects": m["val_subjects"],
               "inference_ms_per_window": m["inference_ms_per_window"]}
        row.update({f"val_{k}": m["val_overall"][k] for k in METRICS})
        for dataset, dm in m["val_by_dataset"].items():
            row[f"auc_{dataset}"] = dm["roc_auc"]
        rows.append(row)
    return rows


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
    args = ap.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    per_seed = pd.DataFrame([r for s in seeds for r in load_seed(s, args.duration)])

    out = OUT_ROOT / f"seed{'-'.join(str(s) for s in seeds)}_{args.duration}s"
    out.mkdir(parents=True, exist_ok=True)

    val_cols = [f"val_{k}" for k in METRICS]
    summary = mean_sd(per_seed, ["model"], val_cols + ["threshold", "best_epoch", "best_val_loss"])
    summary = summary[["model"] + [c for c in summary.columns if c != "model"]]
    summary = summary.sort_values("val_roc_auc", ascending=False)
    summary.to_csv(out / "val_summary_mean_sd.csv", index=False)

    auc_cols = [c for c in per_seed.columns if c.startswith("auc_")]
    by_dataset = mean_sd(per_seed, ["model"], auc_cols)
    by_dataset = by_dataset.sort_values("model")
    by_dataset.to_csv(out / "val_by_dataset_mean_sd.csv", index=False)

    ranking = per_seed.pivot_table(index="seed", columns="model", values="val_roc_auc")
    rank = ranking.rank(axis=1, ascending=False).astype(int)
    rank_out = pd.concat([ranking.round(4).add_suffix("_auc"),
                          rank.add_suffix("_rank")], axis=1).reset_index()
    rank_out.to_csv(out / "val_ranking.csv", index=False)

    per_seed.to_csv(out / "val_per_seed.csv", index=False)

    pd.set_option("display.width", 250)
    print(f"=== validation, seeds {seeds}, {args.duration}s: mean +/- SD ===")
    print(summary[["model", "val_roc_auc", "val_roc_auc_sd", "val_f1", "val_f1_sd",
                   "val_sensitivity", "val_sensitivity_sd", "val_specificity",
                   "val_specificity_sd", "threshold", "threshold_sd"]]
          .to_string(index=False))
    print("\n=== per-dataset ROC-AUC (mean +/- SD across seeds) ===")
    print(by_dataset.to_string(index=False))
    print("\n=== ROC-AUC by seed and rank (1 = best that seed) ===")
    print(rank_out.to_string(index=False))
    print("\nPer-seed numbers:")
    print(per_seed[["seed", "model", "val_roc_auc", "val_f1", "threshold",
                    "best_epoch", "epochs_run"]].to_string(index=False))

    (out / "aggregate_run.json").write_text(json.dumps({
        "stage": "7a",
        "run_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seeds": seeds, "duration_s": args.duration,
        "source": [f"artifacts/train/seed{s}_{args.duration}s" for s in seeds],
        "split": "validation only; test not read by this script",
        "summary": "mean and sample SD (ddof=1) across seeds",
        "caution": "each seed is a different subject split, so SD across seeds measures "
                   "split sensitivity, not run-to-run variance of a fixed split",
    }, indent=2), encoding="utf-8")

    print(f"\nwrote -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
