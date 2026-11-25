"""Stage 7d: run the preprocessing ablation (Experiment 3).

Each arm changes exactly one preprocessing decision; windows, splits, architecture,
budget, and the frozen protocol are unchanged. Validation drives early stopping and the
decision threshold; test is read once per arm with those frozen values.

Run:
    # quick check, one seed, two models
    python scripts/23_run_ablation.py --seeds 42 --models proposed,resp_only

    # full
    python scripts/23_run_ablation.py --seeds 42,1337,2024 \\
        --models proposed,resp_only --duration 30 --device cuda --workers 2

Outputs (artifacts/ablation/seed<list>_<duration>s/):
    ablation_val.csv, ablation_test.csv, ablation_per_seed.csv, ablation_run.json
    runs/<arm>_<model>_seed<seed>/...
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models import build, n_params  # noqa: E402

train_mod = import_module("15_train")
abl = import_module("23_ablation")
dataset_mod = import_module("08_dataset")

ROOT = Path(__file__).resolve().parents[1]
WIN = ROOT / "artifacts" / "windows"
TRAIN_CFG = json.loads((ROOT / "config" / "model.json").read_text(encoding="utf-8"))["training"]


def run_arm(arm: str, model_name: str, seed: int, duration: int, manifest: pd.DataFrame,
            norm: dict, out_dir: Path, device: torch.device, workers: int,
            max_epochs: int, patience: int, use_test: bool,
            train_cap: int | None = None) -> dict:
    run_dir = out_dir / "runs" / f"{arm}_{model_name}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    records = sorted(set(zip(manifest.record_id, manifest.dataset)))
    stats = abl.record_stats(records, duration, source=abl.ARMS[arm]["source"])

    def make_train_ds(rows: pd.DataFrame):
        return abl.AblationDataset(rows, norm, duration, arm, stats)

    val_rows = manifest[(manifest.seed == seed) & (manifest.split == "val")].reset_index(drop=True)
    val_ds = abl.AblationDataset(val_rows, norm, duration, arm, stats)
    val_loader = DataLoader(val_ds, batch_size=TRAIN_CFG["batch_size"], shuffle=False,
                            num_workers=workers, drop_last=False,
                            pin_memory=(device.type == "cuda"),
                            persistent_workers=workers > 0)

    result = train_mod.train_one(
        model_name, manifest, norm, duration, val_loader, val_rows, run_dir,
        max_epochs, patience, train_cap, device=device, num_workers=workers,
        train_ds_factory=make_train_ds)
    result["arm"] = arm
    result["arm_spec"] = abl.ARMS[arm]
    result["seed"] = seed
    result["duration_s"] = duration
    (run_dir / f"metrics_{model_name}.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")

    row = {"arm": arm, "model": model_name, "seed": seed, "duration_s": duration,
           "params": result["params"], "best_epoch": result["best_epoch"],
           "epochs_run": result["epochs_run"], "threshold": result["threshold"]}
    for k in ("accuracy", "sensitivity", "specificity", "precision", "f1", "roc_auc"):
        row[f"val_{k}"] = result["val_overall"][k]

    if use_test:
        test_rows = manifest[(manifest.seed == seed) & (manifest.split == "test")].reset_index(drop=True)
        test_ds = abl.AblationDataset(test_rows, norm, duration, arm, stats)
        test_loader = DataLoader(test_ds, batch_size=TRAIN_CFG["batch_size"], shuffle=False,
                                 num_workers=workers, drop_last=False,
                                 pin_memory=(device.type == "cuda"),
                                 persistent_workers=workers > 0)
        ckpt = torch.load(train_mod.resolve_recorded(
            json.loads((run_dir / f"metrics_{model_name}.json").read_text(encoding="utf-8"))["checkpoint"]),
            map_location=device, weights_only=False)
        model = build(model_name, duration).to(device)
        model.load_state_dict(ckpt["state_dict"])
        score, y, sec_per_window = train_mod.score_batches(model, test_loader, device)
        expected = test_rows.label.to_numpy()
        if not np.array_equal(y, expected):
            raise RuntimeError(f"[{arm}/{model_name}] label order mismatch on test")
        test_m = train_mod.metrics_from_scores(y, score, result["threshold"])
        for k in ("accuracy", "sensitivity", "specificity", "precision", "f1", "roc_auc"):
            row[f"test_{k}"] = test_m[k]
        row["test_windows"] = int(len(y))
        row["test_positive_rate"] = round(float(y.mean()), 4)
        row["inference_ms_per_window"] = round(sec_per_window * 1000, 3)
        pred = pd.DataFrame({"dataset": test_rows.dataset.to_numpy(),
                             "subject_id": test_rows.subject_id.to_numpy(),
                             "record_id": test_rows.record_id.to_numpy(),
                             "window_start_s": test_rows.window_start_s.to_numpy(),
                             "label": y, "score": score,
                             "pred": (score >= result["threshold"]).astype(int)})
        pred.to_csv(run_dir / f"test_predictions_{model_name}.csv", index=False)

    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--models", default="proposed,resp_only")
    ap.add_argument("--arms", default=",".join(abl.ARM_NAMES))
    ap.add_argument("--duration", type=int, default=30)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--max-epochs", type=int, default=TRAIN_CFG.get("max_epochs", 12))
    ap.add_argument("--patience", type=int, default=TRAIN_CFG.get("patience", 3))
    ap.add_argument("--out", default=None)
    ap.add_argument("--val-only", action="store_true",
                    help="skip the test read; use for a first pass before committing")
    ap.add_argument("--train-cap", type=int, default=None,
                    help="smoke testing only: cap training windows")
    args = ap.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    for a in arms:
        if a not in abl.ARMS:
            raise SystemExit(f"unknown arm {a}; choose from {abl.ARM_NAMES}")

    device = train_mod.pick_device(args.device)
    out_dir = (Path(args.out) if args.out
               else ROOT / "artifacts" / "ablation"
               / f"seed{'-'.join(str(s) for s in seeds)}_{args.duration}s").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(WIN / f"windows_{args.duration}s.csv", low_memory=False)

    print(f"ablation -> {out_dir}", flush=True)
    print(f"device={device} workers={args.workers} seeds={seeds} models={models} "
          f"arms={arms} duration={args.duration}s val_only={args.val_only}\n", flush=True)

    rows = []
    for seed in seeds:
        TRAIN_CFG["seed"] = seed
        TRAIN_CFG["duration_s"] = args.duration
        norm = dataset_mod.load_norm(seed, args.duration)
        for arm in arms:
            for model_name in models:
                t0 = time.perf_counter()
                print(f"[seed {seed}] {arm} / {model_name} ...", flush=True)
                row = run_arm(arm, model_name, seed, args.duration, manifest, norm,
                              out_dir, device, args.workers, args.max_epochs,
                              args.patience, use_test=not args.val_only,
                              train_cap=args.train_cap)
                row["wall_minutes"] = round((time.perf_counter() - t0) / 60, 2)
                rows.append(row)
                pd.DataFrame(rows).to_csv(out_dir / "ablation_runs.csv", index=False)
                print(f"[seed {seed}] {arm} / {model_name} "
                      f"val_auc={row['val_roc_auc']:.4f} "
                      f"({row['wall_minutes']:.1f} min)", flush=True)

    df = pd.DataFrame(rows)
    metric_cols = [c for c in df.columns if c.startswith(("val_", "test_"))]
    agg = df.groupby(["arm", "model"])[metric_cols].agg(["mean", "std"])
    agg.columns = [f"{b}_{s}" for b, s in agg.columns]
    agg = agg.reset_index()
    for c in metric_cols:
        if f"{c}_mean" in agg:
            agg[c] = agg[f"{c}_mean"].round(4)
            agg[f"{c}_sd"] = agg[f"{c}_std"].round(4)
            agg = agg.drop(columns=[f"{c}_mean", f"{c}_std"])

    if not args.val_only:
        test_cols = [c for c in agg.columns if c.startswith("test_")]
        if test_cols:
            agg.to_csv(out_dir / "ablation_test.csv", index=False)
    agg.to_csv(out_dir / "ablation_val.csv", index=False)
    df.to_csv(out_dir / "ablation_per_seed.csv", index=False)

    (out_dir / "ablation_run.json").write_text(json.dumps({
        "stage": "7d",
        "run_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seeds": seeds, "models": models, "arms": arms,
        "duration_s": args.duration, "device": str(device), "workers": args.workers,
        "val_only": args.val_only,
        "arm_definitions": abl.ARMS,
        "caveat_no_filter": "unfiltered cache is resample_poly only, and that anti-alias "
                            "step is itself a low-pass (ECG ~50 Hz, respiration ~12.5 Hz). "
                            "Arm means 'no added filter', not 'no filtering'.",
        "rule": "one decision changed per arm; windows, splits, architecture, budget, and "
                "threshold selection are identical across arms",
    }, indent=2), encoding="utf-8")

    pd.set_option("display.width", 220)
    keep = ["arm", "model", "val_roc_auc", "val_roc_auc_sd", "val_f1"]
    if not args.val_only:
        keep += ["test_roc_auc", "test_roc_auc_sd"]
    keep = [c for c in keep if c in agg.columns]
    print("\n=== ablation, mean +/- SD across seeds ===")
    print(agg[keep].sort_values(["model", "val_roc_auc"], ascending=[True, False])
          .to_string(index=False))
    print(f"\nwrote -> {train_mod.rel_or_abs(out_dir)}")


if __name__ == "__main__":
    main()
