"""Stage 4: pilot training run (pipeline health check, not a result).

Purpose, per the work plan: prove that the training pipeline is alive before
committing to a larger budget. Four configurations (proposed, ECG-only,
respiration-only, simple fusion), one seed, 10 s windows, reduced training
subset, CPU only.

Deliberately excluded:
  * the test split is never read, so the paper's "test used once" rule holds;
  * no hyperparameter search, no multi-seed, no ablation - those belong to
    Stages 5 and 7.

Validation is used for early stopping (validation loss), checkpoint selection,
and threshold selection (maximum F1), as Section 3.4.9 / 3.5.2 require.
Numbers printed here are pipeline diagnostics on the validation split, not
test-set results.

Outputs: artifacts/pilot/pilot_summary.csv, artifacts/pilot/pilot_<model>.json,
artifacts/pilot/checkpoints/<model>.pt
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models import ECG_FS, RESP_FS, MODEL_NAMES, build, n_params  # noqa: E402
from importlib import import_module

_dataset = import_module("08_dataset")
WindowDataset = _dataset.WindowDataset
load_norm = _dataset.load_norm

ROOT = Path(__file__).resolve().parents[1]
WIN = ROOT / "artifacts" / "windows"
OUT = ROOT / "artifacts" / "pilot"
CKPT = OUT / "checkpoints"
MODEL_CFG = json.loads((ROOT / "config" / "model.json").read_text(encoding="utf-8"))
TRAIN_CFG = MODEL_CFG["training"]


def materialize(manifest: pd.DataFrame, norm: dict, duration: int) -> dict:
    """Slice a subset into memory: ~40 MB for 8k 10 s windows, so RAM stays flat."""
    ds = WindowDataset(manifest, norm, duration, cache_size=64)
    ecg = torch.empty(len(ds), 1, duration * ECG_FS)
    resp = torch.empty(len(ds), 1, duration * RESP_FS)
    labels = torch.empty(len(ds), dtype=torch.long)
    t0 = time.perf_counter()
    for i in range(len(ds)):
        item = ds[i]
        ecg[i], resp[i], labels[i] = item["ecg"], item["resp"], item["label"]
    print(f"  materialized {len(ds):,} windows in {time.perf_counter() - t0:.1f}s "
          f"({ecg.numel() * 4 / 1e6:.0f} MB ecg)", flush=True)
    return {"ecg": ecg, "resp": resp, "label": labels}


def make_loader(data: dict, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        TensorDataset(data["ecg"], data["resp"], data["label"]),
        batch_size=batch_size, shuffle=shuffle, num_workers=0, drop_last=False)


def metrics_from_scores(y: np.ndarray, score: np.ndarray, threshold: float) -> dict:
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
    return {"accuracy": acc, "sensitivity": sens, "specificity": spec,
            "precision": prec, "f1": f1, "roc_auc": auc,
            "tp": tp, "tn": tn, "fp": fp, "fn": fn}


def best_threshold(y: np.ndarray, score: np.ndarray) -> float:
    grid = np.unique(np.round(np.linspace(0.0, 1.0, 201), 4))
    best_t, best_f1 = 0.5, -1.0
    for t in grid:
        f1 = metrics_from_scores(y, score, t)["f1"]
        if np.isfinite(f1) and f1 > best_f1:
            best_t, best_f1 = float(t), f1
    return best_t


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader) -> tuple[np.ndarray, np.ndarray, float]:
    model.eval()
    scores, labels = [], []
    n = 0
    t0 = time.perf_counter()
    for ecg, resp, y in loader:
        logits = model(ecg, resp)
        scores.append(torch.softmax(logits, dim=1)[:, 1])
        labels.append(y)
        n += y.shape[0]
    dt = time.perf_counter() - t0
    return (torch.cat(scores).numpy(), torch.cat(labels).numpy(), dt / n)


def run(name: str, train_data: dict, val_data: dict, duration_s: int, out: dict) -> dict:
    torch.manual_seed(TRAIN_CFG["seed"])
    np.random.seed(TRAIN_CFG["seed"])

    model = build(name, duration_s)
    bs = TRAIN_CFG["batch_size"]
    train_loader = make_loader(train_data, bs, shuffle=True)
    val_loader = make_loader(val_data, bs, shuffle=False)

    counts = torch.bincount(train_data["label"], minlength=2).float()
    weights = counts.sum() / (2 * counts.clamp(min=1))
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=TRAIN_CFG["lr"],
                                  weight_decay=TRAIN_CFG["weight_decay"])

    steps_per_epoch = len(train_loader)
    max_epochs = out["max_epochs"]
    total_steps = steps_per_epoch * max_epochs
    warmup_steps = max(1, int(TRAIN_CFG["warmup_fraction"] * total_steps))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best = {"val_loss": float("inf"), "epoch": -1, "state": None}
    history = []
    step = 0
    for epoch in range(max_epochs):
        model.train()
        t0 = time.perf_counter()
        running, seen = 0.0, 0
        for ecg, resp, y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(ecg, resp), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), TRAIN_CFG["grad_clip_norm"])
            optimizer.step()
            scheduler.step()
            step += 1
            running += float(loss.detach()) * y.shape[0]
            seen += y.shape[0]
        train_loss = running / seen
        train_time = time.perf_counter() - t0

        model.eval()
        t0 = time.perf_counter()
        v_loss, v_n = 0.0, 0
        with torch.no_grad():
            for ecg, resp, y in val_loader:
                v_loss += float(criterion(model(ecg, resp), y)) * y.shape[0]
                v_n += y.shape[0]
        val_loss = v_loss / v_n
        val_time = time.perf_counter() - t0

        history.append({"epoch": epoch + 1, "train_loss": round(train_loss, 4),
                        "val_loss": round(val_loss, 4),
                        "lr": round(scheduler.get_last_lr()[0], 6),
                        "train_s": round(train_time, 1), "val_s": round(val_time, 1)})
        print(f"  epoch {epoch + 1}/{max_epochs} train_loss={train_loss:.4f} "
              f"val_loss={val_loss:.4f} lr={scheduler.get_last_lr()[0]:.2e} "
              f"({train_time:.0f}s train / {val_time:.0f}s val)", flush=True)

        if val_loss < best["val_loss"] - 1e-6:
            best = {"val_loss": val_loss, "epoch": epoch + 1,
                    "state": {k: v.detach().clone() for k, v in model.state_dict().items()}}
        elif epoch + 1 - best["epoch"] >= TRAIN_CFG["patience"]:
            print(f"  early stop: no val improvement for {TRAIN_CFG['patience']} epochs")
            break

    model.load_state_dict(best["state"])
    CKPT.mkdir(parents=True, exist_ok=True)
    ckpt_path = CKPT / f"{name}.pt"
    torch.save({"model": name, "duration_s": TRAIN_CFG["duration_s"],
                "seed": TRAIN_CFG["seed"], "epoch": best["epoch"],
                "val_loss": best["val_loss"], "state_dict": best["state"],
                "config": MODEL_CFG}, ckpt_path)

    # reload proof: fresh model + saved weights must reproduce the same logits
    reloaded = build(name, duration_s)
    reloaded.load_state_dict(torch.load(ckpt_path, map_location="cpu")["state_dict"])
    reloaded.eval()
    sample = {k: v[:64] for k, v in val_data.items() if k != "label"}
    with torch.no_grad():
        a = model.eval()(sample["ecg"], sample["resp"])
        b = reloaded(sample["ecg"], sample["resp"])
    max_diff = float((a - b).abs().max())

    score, y_val, sec_per_window = predict(model, val_loader)
    threshold = best_threshold(y_val, score)
    val_metrics = metrics_from_scores(y_val, score, threshold)
    val_metrics_default = metrics_from_scores(y_val, score, 0.5)

    return {
        "model": name,
        "params": n_params(model),
        "best_epoch": best["epoch"],
        "best_val_loss": round(best["val_loss"], 4),
        "epochs_run": len(history),
        "history": history,
        "class_counts_train": counts.tolist(),
        "class_weights": [round(float(w), 4) for w in weights],
        "threshold_val_f1": threshold,
        "val_metrics_at_val_threshold": val_metrics,
        "val_metrics_at_0.5": val_metrics_default,
        "inference_s_per_window": round(sec_per_window, 6),
        "checkpoint_reload_max_logit_diff": max_diff,
        "checkpoint": str(ckpt_path.relative_to(ROOT)) if ckpt_path.is_relative_to(ROOT) else str(ckpt_path),
    }


def main() -> None:
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else TRAIN_CFG["seed"]
    duration = int(sys.argv[2]) if len(sys.argv) > 2 else TRAIN_CFG["duration_s"]
    n_train = int(sys.argv[3]) if len(sys.argv) > 3 else 8000
    n_val = int(sys.argv[4]) if len(sys.argv) > 4 else 3000
    max_epochs = int(sys.argv[5]) if len(sys.argv) > 5 else 4

    OUT.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(WIN / f"windows_{duration}s.csv", low_memory=False)
    norm = load_norm(seed, duration)
    rng = np.random.default_rng(seed)

    print(f"=== stage 4 pilot: seed={seed} window={duration}s "
          f"models={MODEL_NAMES} ===", flush=True)
    subsets = {}
    for split, n in (("train", n_train), ("val", n_val)):
        pool = manifest[(manifest.seed == seed) & (manifest.split == split)]
        idx = rng.choice(len(pool), size=min(n, len(pool)), replace=False)
        sub = pool.iloc[np.sort(idx)]
        print(f"{split}: {len(sub):,} of {len(pool):,} windows, "
              f"{sub.subject_id.nunique()} subjects, "
              f"positive rate {sub.label.mean():.3f}")
        subsets[split] = materialize(sub, norm, duration)

    print(f"val window count: {len(subsets['val']['label']):,}", flush=True)

    results = []
    for name in MODEL_NAMES:
        print(f"\n--- {name} ---", flush=True)
        t0 = time.perf_counter()
        res = run(name, subsets["train"], subsets["val"], duration, {"max_epochs": max_epochs})
        res["wall_minutes"] = round((time.perf_counter() - t0) / 60, 2)
        results.append(res)
        print(f"  params={res['params']:,} best_epoch={res['best_epoch']} "
              f"val_loss={res['best_val_loss']:.4f} "
              f"val_auc={res['val_metrics_at_val_threshold']['roc_auc']:.3f} "
              f"val_f1={res['val_metrics_at_val_threshold']['f1']:.3f} "
              f"reload_diff={res['checkpoint_reload_max_logit_diff']:.2e} "
              f"({res['wall_minutes']:.1f} min)", flush=True)
        (OUT / f"pilot_{name}.json").write_text(json.dumps(res, indent=2), encoding="utf-8")

    rows = []
    for r in results:
        m = r["val_metrics_at_val_threshold"]
        rows.append({
            "model": r["model"], "params": r["params"], "best_epoch": r["best_epoch"],
            "epochs_run": r["epochs_run"], "best_val_loss": r["best_val_loss"],
            "threshold": r["threshold_val_f1"],
            "val_accuracy": round(m["accuracy"], 4),
            "val_sensitivity": round(m["sensitivity"], 4),
            "val_specificity": round(m["specificity"], 4),
            "val_precision": round(m["precision"], 4),
            "val_f1": round(m["f1"], 4),
            "val_roc_auc": round(m["roc_auc"], 4),
            "inference_ms_per_window": round(r["inference_s_per_window"] * 1000, 2),
            "checkpoint_reload_max_logit_diff": r["checkpoint_reload_max_logit_diff"],
            "wall_minutes": r["wall_minutes"],
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "pilot_summary.csv", index=False)
    print("\n" + summary.to_string(index=False))
    print("\nNOTE: validation-split diagnostics only. Test split untouched.")


if __name__ == "__main__":
    main()
