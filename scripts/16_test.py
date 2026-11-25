"""Stage 6: test prediction with frozen models, then the six evaluation metrics.

Reads artifacts/train/seed<seed>_<duration>s/frozen.json and uses each checkpoint
exactly as frozen: no retraining, no threshold tuning, no re-selection of epochs.
The decision threshold comes from validation and is applied unchanged, so this is
the single test-set read for the seed (Section 3.5.2, "test used once").

Metrics: accuracy, sensitivity, specificity, precision, F1, ROC-AUC. AUC is computed
from the probability scores, not from thresholded labels.

AUC is decomposed two extra ways because pooled AUC and per-subject AUC disagree when
each subject has its own score offset: pooled (all windows together, the headline),
mean per-subject AUC (each subject ranked separately, then averaged), and
within-subject AUC (scores centred per subject, then pooled). These are diagnostics
for the cross-subject calibration limitation, not additional headline metrics.

Run:
    python scripts/16_test.py
    python scripts/16_test.py --models proposed,simple_fusion

Outputs (artifacts/test/seed<seed>_<duration>s/):
    metrics_test_<model>.json, test_predictions_<model>.csv,
    test_by_subject_<model>.csv, test_by_dataset_<model>.csv,
    test_auc_decomposition.csv, test_summary.csv, test_run.json
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

dataset_mod = import_module("08_dataset")
WindowDataset = dataset_mod.WindowDataset
load_norm = dataset_mod.load_norm

train_mod = import_module("15_train")
metrics_from_scores = train_mod.metrics_from_scores

ROOT = Path(__file__).resolve().parents[1]
WIN = ROOT / "artifacts" / "windows"
TRAIN_CFG = json.loads((ROOT / "config" / "model.json").read_text(encoding="utf-8"))["training"]


def round_metrics(m: dict) -> dict:
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()}


def auc_decomposition(pred: pd.DataFrame) -> dict:
    from sklearn.metrics import roc_auc_score
    per_subject = [roc_auc_score(g.label, g.score)
                   for _, g in pred.groupby("subject_id") if g.label.nunique() == 2]
    centred = pred.groupby("subject_id").score.transform(
        lambda s: (s - s.mean()) / (s.std() + 1e-8))
    return {
        "pooled_auc": round(float(roc_auc_score(pred.label, pred.score)), 4),
        "mean_subject_auc": round(float(np.mean(per_subject)), 4),
        "min_subject_auc": round(float(np.min(per_subject)), 4),
        "max_subject_auc": round(float(np.max(per_subject)), 4),
        "within_subject_auc": round(float(roc_auc_score(pred.label, centred)), 4),
        "subjects_scored": len(per_subject),
    }


@torch.no_grad()
def score_batches(model: torch.nn.Module, loader: DataLoader,
                  device: torch.device | None = None) -> tuple[np.ndarray, np.ndarray, float]:
    device = device or torch.device("cpu")
    model.eval()
    scores, labels, n = [], [], 0
    t0 = time.perf_counter()
    for batch in loader:
        logits = model(batch["ecg"].to(device), batch["resp"].to(device))
        scores.append(torch.softmax(logits, dim=1)[:, 1].cpu())
        labels.append(batch["label"])
        n += batch["label"].shape[0]
    return (torch.cat(scores).numpy(), torch.cat(labels).numpy(),
            (time.perf_counter() - t0) / max(n, 1))


def evaluate_one(name: str, frozen_entry: dict, test_rows: pd.DataFrame, norm: dict,
                 duration: int, out_dir: Path, device: torch.device | None = None,
                 workers: int = 0) -> dict:
    device = device or torch.device("cpu")
    ckpt_path = train_mod.resolve_recorded(frozen_entry["checkpoint"])
    threshold = float(frozen_entry["threshold"])
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    model = build(name, duration).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    loader = DataLoader(WindowDataset(test_rows, norm, duration, cache_size=64),
                        batch_size=TRAIN_CFG["batch_size"], shuffle=False,
                        num_workers=workers, drop_last=False,
                        pin_memory=(device.type == "cuda"),
                        persistent_workers=workers > 0)
    score, y, sec_per_window = score_batches(model, loader, device)

    expected = test_rows.label.to_numpy()
    if not np.array_equal(y, expected):
        raise RuntimeError(f"[{name}] label order mismatch between loader and manifest")

    overall = metrics_from_scores(y, score, threshold)
    per_dataset = {d: metrics_from_scores(y[test_rows.dataset.to_numpy() == d],
                                          score[test_rows.dataset.to_numpy() == d], threshold)
                   for d in sorted(test_rows.dataset.unique())}

    pred = pd.DataFrame({"dataset": test_rows.dataset.to_numpy(),
                         "subject_id": test_rows.subject_id.to_numpy(),
                         "record_id": test_rows.record_id.to_numpy(),
                         "window_start_s": test_rows.window_start_s.to_numpy(),
                         "label": y, "score": score,
                         "pred": (score >= threshold).astype(int)})
    pred.to_csv(out_dir / f"test_predictions_{name}.csv", index=False)

    per_subject = []
    for subject, grp in pred.groupby("subject_id"):
        m = metrics_from_scores(grp.label.to_numpy(), grp.score.to_numpy(), threshold)
        per_subject.append({"subject_id": subject, "dataset": grp.dataset.iloc[0],
                            "windows": len(grp), "positives": int(grp.label.sum()),
                            "positive_rate": round(float(grp.label.mean()), 4),
                            **round_metrics(m)})
    pd.DataFrame(per_subject).to_csv(out_dir / f"test_by_subject_{name}.csv", index=False)

    pd.DataFrame([{"dataset": d, **round_metrics(m)} for d, m in per_dataset.items()]) \
        .to_csv(out_dir / f"test_by_dataset_{name}.csv", index=False)

    decomposition = auc_decomposition(pred)

    result = {
        "model": name, "params": n_params(model), "duration_s": duration,
        "checkpoint": frozen_entry["checkpoint"],
        "frozen_threshold": threshold, "best_epoch": frozen_entry.get("best_epoch"),
        "test_windows": int(len(pred)),
        "test_subjects": int(pred.subject_id.nunique()),
        "test_records": int(pred.record_id.nunique()),
        "test_positive_rate": round(float(pred.label.mean()), 4),
        "test_overall": round_metrics(overall),
        "test_by_dataset": {d: round_metrics(m) for d, m in per_dataset.items()},
        "auc_decomposition": decomposition,
        "inference_ms_per_window": round(sec_per_window * 1000, 3),
    }
    (out_dir / f"metrics_test_{name}.json").write_text(json.dumps(result, indent=2),
                                                       encoding="utf-8")
    print(f"[{name}] TEST acc={overall['accuracy']:.3f} sens={overall['sensitivity']:.3f} "
          f"spec={overall['specificity']:.3f} prec={overall['precision']:.3f} "
          f"f1={overall['f1']:.3f} auc={overall['roc_auc']:.3f} "
          f"(threshold={threshold:.3f})", flush=True)
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=None,
                    help="comma list; default = every model recorded in frozen.json")
    ap.add_argument("--seed", type=int, default=TRAIN_CFG["seed"])
    ap.add_argument("--duration", type=int, default=TRAIN_CFG["duration_s"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda | cuda:0")
    ap.add_argument("--workers", type=int, default=0)
    args = ap.parse_args()

    device = train_mod.pick_device(args.device)

    train_dir = ROOT / "artifacts" / "train" / f"seed{args.seed}_{args.duration}s"
    frozen_path = train_dir / "frozen.json"
    if not frozen_path.exists():
        raise SystemExit(f"missing freeze record: {frozen_path}")
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    if frozen.get("test_split_read"):
        raise SystemExit("frozen.json reports the test split was already read; refusing to rerun")

    names = ([n.strip() for n in args.models.split(",") if n.strip()]
             if args.models else list(frozen["models"]))
    missing = [n for n in names if n not in frozen["models"]]
    if missing:
        raise SystemExit(f"model(s) not in frozen.json: {missing}")

    out_dir = (Path(args.out) if args.out
               else ROOT / "artifacts" / "test" / f"seed{args.seed}_{args.duration}s").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(WIN / f"windows_{args.duration}s.csv", low_memory=False)
    norm = load_norm(args.seed, args.duration)
    test_rows = manifest[(manifest.seed == args.seed) & (manifest.split == "test")].reset_index(drop=True)
    if test_rows.empty:
        raise SystemExit(f"no test rows for seed={args.seed} duration={args.duration}s")

    print(f"stage 6 -> {out_dir}\n"
          f"test windows={len(test_rows):,} subjects={test_rows.subject_id.nunique()} "
          f"records={test_rows.record_id.nunique()} "
          f"positive_rate={test_rows.label.mean():.3f}", flush=True)
    print(f"frozen at {frozen['frozen_at']} (stage {frozen['stage']}), "
          f"thresholds taken as-is\n", flush=True)

    results = []
    for name in names:
        results.append(evaluate_one(name, frozen["models"][name], test_rows, norm,
                                    args.duration, out_dir, device=device,
                                    workers=args.workers))

    summary = pd.DataFrame([{
        "model": r["model"], "windows": r["test_windows"], "positives": r["test_overall"]["positives"],
        "accuracy": r["test_overall"]["accuracy"], "sensitivity": r["test_overall"]["sensitivity"],
        "specificity": r["test_overall"]["specificity"], "precision": r["test_overall"]["precision"],
        "f1": r["test_overall"]["f1"], "roc_auc": r["test_overall"]["roc_auc"],
        "threshold": r["frozen_threshold"], "ms_per_window": r["inference_ms_per_window"],
    } for r in results]).sort_values("roc_auc", ascending=False)
    summary.to_csv(out_dir / "test_summary.csv", index=False)
    print("\n" + summary.to_string(index=False), flush=True)

    decomp = pd.DataFrame([{"model": r["model"], **r["auc_decomposition"]} for r in results]) \
        .sort_values("pooled_auc", ascending=False)
    decomp.to_csv(out_dir / "test_auc_decomposition.csv", index=False)
    print("\nAUC decomposition (pooled vs per-subject; diagnostic, not headline):", flush=True)
    print(decomp.to_string(index=False), flush=True)

    (out_dir / "test_run.json").write_text(json.dumps({
        "stage": 6,
        "run_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seed": args.seed, "duration_s": args.duration,
        "source_frozen": train_mod.rel_or_abs(frozen_path),
        "frozen_at": frozen["frozen_at"],
        "rule": "checkpoint and validation threshold applied unchanged; test read once",
        "models": names,
        "metrics": ["accuracy", "sensitivity", "specificity", "precision", "f1", "roc_auc"],
        "windows_file": f"artifacts/windows/windows_{args.duration}s.csv",
        "test_windows": int(len(test_rows)),
        "test_subjects": int(test_rows.subject_id.nunique()),
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
