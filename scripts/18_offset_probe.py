"""Stage 6c: probe what produces the per-record score offset (diagnostic only).

Stage 6b left one mechanism unresolved: the model gives its highest scores to subjects with
the LEAST apnea (subject 22: 1.9% prevalence, mean score 0.450; slp41: 0% prevalence, mean
score 0.504). This script tests the leading explanation without training anything.

Hypothesis: normalization is applied PER DATASET, not per record (Stage 3b-1). So every
record still enters the model with its own offset from the dataset mean. That offset is a
pure record-level constant the model can use as a shortcut, and it can correlate with how
apneic a record is.

Two checks:
  1. Measure the per-record offset that survives dataset-level normalization, i.e. the mean
     of (x - dataset_mean) / dataset_std for each record and modality. If it varies a lot
     across records, the model is fed a record identifier in disguise.
  2. Correlate that offset with (a) apnea prevalence and (b) the model's mean score, on the
     records we have scores for (validation + test). Correlations are computed with and
     without controlling for prevalence, because prevalence is the obvious confounder.

Interpretation rules written into the output: n is ~14 records, so a correlation here is a
lead, not a conclusion. Nothing in this script may be used to select a model.

Run:
    python scripts/18_offset_probe.py

Outputs (artifacts/offset_probe/):
    record_features.csv, offset_correlations.csv, offset_probe.json
"""

from __future__ import annotations

import json
import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
SIG = ROOT / "artifacts" / "signals"
ART = ROOT / "artifacts"
OUT = ART / "offset_probe"
MODELS = ["proposed", "simple_fusion", "ecg_only", "resp_only"]

sys.path.insert(0, str(Path(__file__).resolve().parent))
dataset_mod = import_module("08_dataset")

POLICY = json.loads((ROOT / "config" / "policy.json").read_text(encoding="utf-8"))
ECG_FS = POLICY["preprocessing"]["ecg"]["target_fs_hz"]
RESP_FS = POLICY["preprocessing"]["resp"]["target_fs_hz"]


def record_signal_stats(record: str, dataset: str, norm: dict) -> dict:
    """What the model actually receives from this record: post-normalization mean/std.
    mean_z != 0 means the record is offset from its dataset, a free record-level cue."""
    out = {}
    for modality, fs in (("ecg", ECG_FS), ("resp", RESP_FS)):
        x = np.load(SIG / f"{record}_{modality}{int(fs)}.npy").astype(np.float64)
        st = norm[dataset][modality]
        z = (x - st["mean"]) / max(st["std"], 1e-8)
        out[f"{modality}_mean_z"] = round(float(z.mean()), 4)
        out[f"{modality}_std_z"] = round(float(z.std()), 4)
        out[f"{modality}_p01_z"] = round(float(np.percentile(z, 1)), 4)
        out[f"{modality}_p99_z"] = round(float(np.percentile(z, 99)), 4)
        out[f"{modality}_clip_frac"] = round(float(((z <= -5) | (z >= 5)).mean()), 5)
    return out


def partial_spearman(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple[float, float]:
    """Rank-based partial correlation of x,y controlling for z, via residualising ranks."""
    rx = pd.Series(x).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    rz = pd.Series(z).rank().to_numpy()
    A = np.column_stack([np.ones_like(rz), rz])

    def resid(v: np.ndarray) -> np.ndarray:
        beta, *_ = np.linalg.lstsq(A, v, rcond=None)
        return v - A @ beta

    ex, ey = resid(rx), resid(ry)
    r = float(np.corrcoef(ex, ey)[0, 1])
    n, k = len(x), 3
    if n <= k or abs(r) >= 1:
        return r, float("nan")
    from scipy.stats import t as tdist
    tval = r * np.sqrt((n - k) / (1 - r * r))
    return r, float(2 * (1 - tdist.cdf(abs(tval), n - k)))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    seed, duration = 42, 10
    norm = dataset_mod.load_norm(seed, duration)

    qc = pd.read_csv(ART / "signals" / "signal_qc.csv")
    rq = pd.read_csv(ART / "signals" / "resp_qc.csv")
    man = pd.read_csv(ART / "audit" / "audit_manifest.csv")

    # scores we actually have: validation and test
    frames = []
    for split, base, fname in (
        ("val", ART / "train" / f"seed{seed}_{duration}s", "val_predictions_{}.csv"),
        ("test", ART / "test" / f"seed{seed}_{duration}s", "test_predictions_{}.csv"),
    ):
        for m in MODELS:
            p = base / fname.format(m)
            if not p.exists():
                continue
            d = pd.read_csv(p)
            g = d.groupby(["dataset", "record_id", "subject_id"]).agg(
                windows=("label", "size"), prevalence=("label", "mean"),
                mean_score=("score", "mean"), std_score=("score", "std")).reset_index()
            g["model"] = m
            g["split"] = split
            frames.append(g)
    scores = pd.concat(frames, ignore_index=True)

    feats = qc[["dataset", "record_id", "ecg_std_filt", "resp_std_filt", "ecg_flatline_frac",
                "resp_flatline_frac", "resp_drift_range_raw", "duration_s"]].merge(
        rq[["dataset", "record_id", "resp_snr_db", "breaths_per_min", "ecg_snr_db"]],
        on=["dataset", "record_id"], how="left")
    stats = []
    for _, r in feats.iterrows():
        s = record_signal_stats(r.record_id, r.dataset, norm)
        stats.append({"dataset": r.dataset, "record_id": r.record_id, **s})
    feats = feats.merge(pd.DataFrame(stats), on=["dataset", "record_id"])

    rec = scores.merge(feats, on=["dataset", "record_id"], how="left")
    keep = ["dataset", "record_id", "subject_id", "split", "model", "windows", "prevalence",
            "mean_score", "std_score", "ecg_mean_z", "resp_mean_z", "ecg_std_z", "resp_std_z",
            "ecg_clip_frac", "resp_clip_frac", "ecg_flatline_frac", "resp_flatline_frac",
            "resp_snr_db", "breaths_per_min", "duration_s"]
    rec[keep].to_csv(OUT / "record_features.csv", index=False)

    print("=== per-record post-normalization offset (what the model sees) ===")
    show = feats[["dataset", "record_id", "ecg_mean_z", "resp_mean_z", "ecg_std_z", "resp_std_z"]]
    print(show.round(3).to_string(index=False))
    print("\nspread of ecg_mean_z across records: "
          f"{feats.ecg_mean_z.min():.2f} to {feats.ecg_mean_z.max():.2f} "
          f"(std {feats.ecg_mean_z.std():.2f})")
    print("spread of resp_mean_z across records: "
          f"{feats.resp_mean_z.min():.2f} to {feats.resp_mean_z.max():.2f} "
          f"(std {feats.resp_mean_z.std():.2f})")

    features = ["ecg_mean_z", "resp_mean_z", "ecg_std_z", "resp_std_z", "ecg_clip_frac",
                "resp_clip_frac", "ecg_flatline_frac", "resp_flatline_frac", "resp_snr_db",
                "breaths_per_min", "duration_s"]
    rows = []
    for m in MODELS:
        for split in ("val", "test"):
            sub = rec[(rec.model == m) & (rec.split == split)].dropna(subset=["mean_score"])
            if len(sub) < 4:
                continue
            for feat in features:
                s2 = sub.dropna(subset=[feat])
                if len(s2) < 4 or s2[feat].nunique() < 3:
                    continue
                rho, p = spearmanr(s2[feat], s2.mean_score)
                prho, pp = partial_spearman(s2[feat].to_numpy(), s2.mean_score.to_numpy(),
                                            s2.prevalence.to_numpy())
                rows.append({"model": m, "split": split, "feature": feat, "n_records": len(s2),
                             "spearman": round(float(rho), 3), "p": round(float(p), 3),
                             "partial_spearman_controlling_prevalence": round(prho, 3),
                             "partial_p": round(pp, 3) if np.isfinite(pp) else None})
    corr = pd.DataFrame(rows)
    corr.to_csv(OUT / "offset_correlations.csv", index=False)

    print("\n=== correlation of record features with the model's mean score ===")
    print("(partial column controls for prevalence: that is the honest test, since a record "
          "with more apnea naturally scores higher)")
    top = corr.reindex(corr.partial_spearman_controlling_prevalence.abs()
                       .sort_values(ascending=False).index).head(20)
    print(top.to_string(index=False))

    (OUT / "offset_probe.json").write_text(json.dumps({
        "stage": "6c",
        "seed": seed, "duration_s": duration,
        "status": "post-hoc diagnostic, no training, must not be used to select a model",
        "exposure": "per-record offset AFTER dataset-level z-score; the model receives this",
        "record_count": int(len(feats)),
        "caution": "n is the number of records (~14 with scores). These correlations are leads, "
                   "not conclusions, and multiple features are tested without correction.",
        "next_step_if_confirmed": "refit normalization per record inside the training split, "
                                  "then retrain all models before touching test again",
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
