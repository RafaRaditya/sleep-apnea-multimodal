"""Stage 5: full training on all training subjects, then freeze.

Uses every training window of the requested seed/duration (no subset), keeps the
validation split for early stopping, checkpoint selection, and threshold
selection, and writes a freeze record. The test split is not read anywhere in this
script, so "test used once" (Section 3.5.2) still holds for Stage 6.

Signals are streamed from the per-record cache instead of materialized: 116,199
10 s windows would need ~580 MB of RAM, and this machine has ~2 GB available.

Run:
    python scripts/15_train.py --models proposed
    python scripts/15_train.py --models proposed,ecg_only,resp_only,simple_fusion

Outputs (default artifacts/train/seed<seed>_<duration>s/):
    run_config.json, history_<model>.csv, metrics_<model>.json,
    val_predictions_<model>.csv, checkpoints/<model>.pt, frozen.json
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
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models import MODEL_NAMES, build, n_params  # noqa: E402

dataset_mod = import_module("08_dataset")
WindowDataset = dataset_mod.WindowDataset
load_norm = dataset_mod.load_norm

ROOT = Path(__file__).resolve().parents[1]
WIN = ROOT / "artifacts" / "windows"
MODEL_CFG = json.loads((ROOT / "config" / "model.json").read_text(encoding="utf-8"))
TRAIN_CFG = MODEL_CFG["training"]


def metrics_from_scores(y: np.ndarray, score: np.ndarray, threshold: float) -> dict:
    from sklearn.metrics import roc_auc_score
    pred = (score >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    sens = tp / (tp + fn) if tp + fn else float("nan")
    spec = tn / (tn + fp) if tn + fp else float("nan")
    prec = tp / (tp + fp) if tp + fp else float("nan")
    f1 = 2 * prec * sens / (prec + sens) if prec + sens else float("nan")
    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    auc = float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else float("nan")
    return {"accuracy": acc, "sensitivity": sens, "specificity": spec, "precision": prec,
            "f1": f1, "roc_auc": auc, "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "n": int(len(y)), "positives": int(y.sum())}


def best_threshold(y: np.ndarray, score: np.ndarray) -> tuple[float, float]:
    grid = np.round(np.linspace(0.01, 0.99, 197), 4)
    best_t, best_f1 = 0.5, -1.0
    for t in grid:
        f1 = metrics_from_scores(y, score, float(t))["f1"]
        if np.isfinite(f1) and f1 > best_f1:
            best_t, best_f1 = float(t), f1
    return best_t, best_f1


@torch.no_grad()
def score_batches(model: nn.Module, loader: DataLoader) -> tuple[np.ndarray, np.ndarray, float]:
    model.eval()
    scores, labels, n = [], [], 0
    t0 = time.perf_counter()
    for batch in loader:
        scores.append(torch.softmax(model(batch["ecg"], batch["resp"]), dim=1)[:, 1])
        labels.append(batch["label"])
        n += batch["label"].shape[0]
    return (torch.cat(scores).numpy(), torch.cat(labels).numpy(),
            (time.perf_counter() - t0) / max(n, 1))


def train_one(name: str, manifest: pd.DataFrame, norm: dict, duration: int,
              val_loader: DataLoader, val_rows: pd.DataFrame, out_dir: Path,
              max_epochs: int, patience: int, train_cap: int | None = None) -> dict:
    seed = TRAIN_CFG["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_rows = manifest[(manifest.seed == seed) & (manifest.split == "train")]
    if train_cap:
        rng = np.random.default_rng(seed)
        keep = np.sort(rng.choice(len(train_rows), size=min(train_cap, len(train_rows)),
                                  replace=False))
        train_rows = train_rows.iloc[keep].reset_index(drop=True)
    train_ds = WindowDataset(train_rows, norm, duration, cache_size=64)
    train_loader = DataLoader(train_ds, batch_size=TRAIN_CFG["batch_size"],
                              shuffle=True, num_workers=0, drop_last=True)

    counts = torch.bincount(torch.from_numpy(train_rows.label.to_numpy()), minlength=2).float()
    weights = counts.sum() / (2 * counts.clamp(min=1))
    criterion = nn.CrossEntropyLoss(weight=weights)

    model = build(name, duration)
    optimizer = torch.optim.AdamW(model.parameters(), lr=TRAIN_CFG["lr"],
                                  weight_decay=TRAIN_CFG["weight_decay"])
    total_steps = len(train_loader) * max_epochs
    warmup_steps = max(1, int(TRAIN_CFG["warmup_fraction"] * total_steps))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    print(f"[{name}] train windows={len(train_rows):,} subjects="
          f"{train_rows.subject_id.nunique()} val windows={len(val_rows):,} "
          f"subjects={val_rows.subject_id.nunique()} params={n_params(model):,} "
          f"steps/epoch={len(train_loader)}", flush=True)

    best = {"val_loss": float("inf"), "epoch": -1, "state": None}
    history = []
    for epoch in range(max_epochs):
        model.train()
        t0 = time.perf_counter()
        running, seen = 0.0, 0
        for batch in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(batch["ecg"], batch["resp"]), batch["label"])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), TRAIN_CFG["grad_clip_norm"])
            optimizer.step()
            scheduler.step()
            running += float(loss.detach()) * batch["label"].shape[0]
            seen += batch["label"].shape[0]
        train_loss = running / max(seen, 1)
        train_s = time.perf_counter() - t0

        model.eval()
        t0 = time.perf_counter()
        v_loss, v_n = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                v_loss += float(criterion(model(batch["ecg"], batch["resp"]),
                                          batch["label"])) * batch["label"].shape[0]
                v_n += batch["label"].shape[0]
        val_loss = v_loss / v_n
        val_s = time.perf_counter() - t0

        history.append({"epoch": epoch + 1, "train_loss": round(train_loss, 4),
                        "val_loss": round(val_loss, 4),
                        "lr": round(scheduler.get_last_lr()[0], 6),
                        "train_s": round(train_s, 1), "val_s": round(val_s, 1)})
        pd.DataFrame(history).to_csv(out_dir / f"history_{name}.csv", index=False)
        print(f"[{name}] epoch {epoch + 1}/{max_epochs} train_loss={train_loss:.4f} "
              f"val_loss={val_loss:.4f} lr={scheduler.get_last_lr()[0]:.2e} "
              f"({train_s:.0f}s/{val_s:.0f}s)", flush=True)

        if val_loss < best["val_loss"] - 1e-6:
            best = {"val_loss": val_loss, "epoch": epoch + 1,
                    "state": {k: v.detach().clone() for k, v in model.state_dict().items()}}
        elif epoch + 1 - best["epoch"] >= patience:
            print(f"[{name}] early stop after epoch {epoch + 1} "
                  f"(no improvement for {patience} epochs)", flush=True)
            break

    model.load_state_dict(best["state"])
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"{name}.pt"
    torch.save({"model": name, "duration_s": duration, "seed": seed,
                "epoch": best["epoch"], "val_loss": best["val_loss"],
                "state_dict": best["state"], "config": MODEL_CFG,
                "train_windows": int(len(train_rows))}, ckpt_path)

    score, y, sec_per_window = score_batches(model, val_loader)
    threshold, val_f1 = best_threshold(y, score)

    overall = metrics_from_scores(y, score, threshold)
    per_dataset = {d: metrics_from_scores(y[val_rows.dataset.to_numpy() == d],
                                          score[val_rows.dataset.to_numpy() == d], threshold)
                   for d in sorted(val_rows.dataset.unique())}

    pred = pd.DataFrame({"dataset": val_rows.dataset.to_numpy(),
                         "subject_id": val_rows.subject_id.to_numpy(),
                         "record_id": val_rows.record_id.to_numpy(),
                         "window_start_s": val_rows.window_start_s.to_numpy(),
                         "label": y, "score": score,
                         "pred": (score >= threshold).astype(int)})
    pred.to_csv(out_dir / f"val_predictions_{name}.csv", index=False)

    per_subject = []
    for subject, grp in pred.groupby("subject_id"):
        m = metrics_from_scores(grp.label.to_numpy(), grp.score.to_numpy(), threshold)
        per_subject.append({"subject_id": subject, "dataset": grp.dataset.iloc[0],
                            "windows": len(grp), "positive_rate": round(float(grp.label.mean()), 4),
                            **{k: (round(v, 4) if isinstance(v, float) else v)
                               for k, v in m.items()}})
    pd.DataFrame(per_subject).to_csv(out_dir / f"val_by_subject_{name}.csv", index=False)

    result = {
        "model": name, "params": n_params(model), "duration_s": duration, "seed": seed,
        "train_windows": int(len(train_rows)),
        "train_subjects": int(train_rows.subject_id.nunique()),
        "val_windows": int(len(val_rows)),
        "val_subjects": int(val_rows.subject_id.nunique()),
        "best_epoch": best["epoch"], "best_val_loss": round(best["val_loss"], 4),
        "epochs_run": len(history), "history": history,
        "class_weights": [round(float(w), 4) for w in weights],
        "threshold": threshold, "threshold_val_f1": round(val_f1, 4),
        "val_overall": {k: (round(v, 4) if isinstance(v, float) else v)
                        for k, v in overall.items()},
        "val_by_dataset": {d: {k: (round(v, 4) if isinstance(v, float) else v)
                               for k, v in m.items()} for d, m in per_dataset.items()},
        "inference_ms_per_window": round(sec_per_window * 1000, 3),
        "checkpoint": str(ckpt_path.relative_to(ROOT)),
    }
    (out_dir / f"metrics_{name}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[{name}] frozen: epoch={best['epoch']} val_loss={best['val_loss']:.4f} "
          f"threshold={threshold:.3f} val_auc={overall['roc_auc']:.4f} "
          f"val_f1={overall['f1']:.4f}", flush=True)
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="proposed")
    ap.add_argument("--duration", type=int, default=TRAIN_CFG["duration_s"])
    ap.add_argument("--seed", type=int, default=TRAIN_CFG["seed"])
    ap.add_argument("--max-epochs", type=int, default=TRAIN_CFG.get("max_epochs", 12))
    ap.add_argument("--patience", type=int, default=TRAIN_CFG.get("patience", 3))
    ap.add_argument("--out", default=None)
    ap.add_argument("--train-cap", type=int, default=None,
                    help="smoke testing only: cap training windows (recorded in run_config)")
    args = ap.parse_args()

    names = [n.strip() for n in args.models.split(",") if n.strip()]
    for n in names:
        if n not in MODEL_NAMES:
            raise SystemExit(f"unknown model {n}; choose from {MODEL_NAMES}")

    out_dir = (Path(args.out) if args.out
               else ROOT / "artifacts" / "train" / f"seed{args.seed}_{args.duration}s").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(WIN / f"windows_{args.duration}s.csv", low_memory=False)
    norm = load_norm(args.seed, args.duration)
    TRAIN_CFG["seed"] = args.seed
    TRAIN_CFG["duration_s"] = args.duration

    val_rows = manifest[(manifest.seed == args.seed) & (manifest.split == "val")].reset_index(drop=True)
    val_ds = WindowDataset(val_rows, norm, args.duration, cache_size=64)
    val_loader = DataLoader(val_ds, batch_size=TRAIN_CFG["batch_size"], shuffle=False,
                            num_workers=0, drop_last=False)

    run_config = {
        "stage": 5, "seed": args.seed, "duration_s": args.duration,
        "models": names, "max_epochs": args.max_epochs, "patience": args.patience,
        "train_cap": args.train_cap,
        "model_config": MODEL_CFG,
        "windows_file": f"artifacts/windows/windows_{args.duration}s.csv",
        "norm_file": "artifacts/normalization/norm_stats.json",
        "test_split_read": False,
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (out_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    print(f"stage 5 -> {out_dir}", flush=True)

    results = []
    for name in names:
        t0 = time.perf_counter()
        res = train_one(name, manifest, norm, args.duration, val_loader, val_rows,
                        out_dir, args.max_epochs, args.patience, args.train_cap)
        res["wall_minutes"] = round((time.perf_counter() - t0) / 60, 2)
        results.append(res)
        (out_dir / f"metrics_{name}.json").write_text(json.dumps(res, indent=2), encoding="utf-8")

    frozen = {
        "stage": 5, "frozen_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seed": args.seed, "duration_s": args.duration,
        "rule": "checkpoint, decision threshold, and hyperparameters are fixed before any test prediction",
        "test_split_read": False,
        "models": {r["model"]: {"checkpoint": r["checkpoint"], "threshold": r["threshold"],
                                "best_epoch": r["best_epoch"],
                                "best_val_loss": r["best_val_loss"]} for r in results},
    }
    (out_dir / "frozen.json").write_text(json.dumps(frozen, indent=2), encoding="utf-8")
    print(json.dumps(frozen["models"], indent=2))


if __name__ == "__main__":
    main()
