"""Stage 2: subject-level split and window/label manifest.

No training. Produces:
  artifacts/splits/split_manifest.csv
  artifacts/windows/windows_10s.csv
  artifacts/windows/windows_30s.csv
  artifacts/windows/window_summary.json

Decisions (provisional unless stated):
  - Apnea-ECG excluded (ECG-only, no respiratory channel).
  - UCDDB respiration channel: Sum (effort sum), matching slpdb Resp (sum).
  - Positive = apnea OR hypopnea only; POSSIBLE and PB EVENT are not positive.
  - Window positive if >=50% overlap with union of positive intervals.
  - Train stride = W/2 (provisional, paper leaves stride TBD);
    validation/test stride = W (non-overlapping, per paper).
  - Splits: 70/15/15, stratified by severity, seeds 42/1337/2024.
"""

from __future__ import annotations

import collections
import csv
import json
import random
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyedflib
import wfdb

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dataset-rm" / "dataset-rm"
POLICY = json.loads((ROOT / "config" / "policy.json").read_text(encoding="utf-8"))
EXCLUDED_RECORDS = set(POLICY["multimodal_policy"]["excluded_records"])
OUT_SPLIT = ROOT / "artifacts" / "splits"
OUT_WIN = ROOT / "artifacts" / "windows"
OUT_SPLIT.mkdir(parents=True, exist_ok=True)
OUT_WIN.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 1337, 2024]
DURATIONS = [10, 30]
TRAIN_STRIDE_FRACTION = 0.5
POSITIVE_OVERLAP = 0.5
APNEA_TOKENS = {"H", "HA", "OA", "X", "CA", "CAA"}
UCDDB_POSITIVE_TYPES = ("HYP", "APNEA")


def rel_seconds(event_time, record_start) -> float:
    delta = (datetime.combine(record_start.date(), event_time)
             - datetime.combine(record_start.date(), record_start.time())).total_seconds()
    if delta < -12 * 3600:
        delta += 24 * 3600
    elif delta > 12 * 3600:
        delta -= 24 * 3600
    return delta


def parse_ucddb_events(path: Path) -> list[dict]:
    events = []
    for line in path.read_text(errors="ignore").splitlines()[3:]:
        if len(line) < 34 or not re.match(r"\s*\d{2}:\d{2}:\d{2}", line[:11]):
            continue
        etype = line[10:18].strip()
        try:
            duration = int(line[26:34].strip())
        except ValueError:
            continue
        events.append({"time": line[0:10].strip(), "type": etype, "duration": duration})
    return events


def merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not intervals:
        return []
    intervals = sorted((max(0.0, s), e) for s, e in intervals if e > s)
    merged = [list(intervals[0])]
    for start, end in intervals[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(s, e) for s, e in merged]


def overlap_seconds(start: float, end: float, intervals: list[tuple[float, float]]) -> float:
    total = 0.0
    for s, e in intervals:
        if e <= start:
            continue
        if s >= end:
            break
        total += min(e, end) - max(s, start)
    return max(0.0, total)


def ucddb_records() -> list[dict]:
    details = pd.read_excel(DATA / "ucddb" / "DESTINATION" / "SubjectDetails.xls")
    ahi = {f"{int(r['Study Number'].replace('UCDDB', ''))}": float(r["PSG AHI"])
           for _, r in details.iterrows()}

    records = []
    for rec_path in sorted((DATA / "ucddb" / "DESTINATION").glob("ucddb*.rec")):
        rec_id = rec_path.stem
        if rec_id in EXCLUDED_RECORDS:
            continue
        subject = str(int(rec_id.replace("ucddb", "")))
        edf = pyedflib.EdfReader(str(rec_path))
        duration = float(edf.file_duration)
        start = edf.getStartdatetime()
        edf.close()

        stages = [int(x) for x in
                  (DATA / "ucddb" / "DESTINATION" / f"{rec_id}_stage.txt").read_text().split()]
        valid_end = min(duration, len(stages) * 30.0)

        events = parse_ucddb_events(DATA / "ucddb" / "DESTINATION" / f"{rec_id}_respevt.txt")
        positive, ambiguous = [], []
        for e in events:
            rel = rel_seconds(datetime.strptime(e["time"], "%H:%M:%S").time(), start)
            interval = (rel, rel + e["duration"])
            if e["type"].startswith(UCDDB_POSITIVE_TYPES):
                positive.append(interval)
            elif e["type"] in ("POSSIBLE", "PB EVENT"):
                ambiguous.append(interval)
        records.append({
            "dataset": "UCDDB", "record_id": rec_id, "subject_id": subject,
            "duration_s": duration, "valid_start_s": 0.0, "valid_end_s": valid_end,
            "positive_intervals": merge_intervals(positive),
            "ambiguous_intervals": merge_intervals(ambiguous),
            "label_source": "event",
        })
    return records, ahi


def psg_records() -> list[dict]:
    records = []
    skipped = []
    for rec_id in (DATA / "slpdb" / "DESTINATION" / "RECORDS").read_text().split():
        if rec_id in EXCLUDED_RECORDS:
            skipped.append(rec_id)
            continue
        subject = re.match(r"slp(\d+)", rec_id).group(1)
        header = wfdb.rdheader(str(DATA / "slpdb" / "DESTINATION" / rec_id))
        resp_channels = [n for n in header.sig_name if "resp" in n.lower()]
        if not any("nasal" in n.lower() for n in resp_channels):
            skipped.append(rec_id)
            continue
        ann = wfdb.rdann(str(DATA / "slpdb" / "DESTINATION" / rec_id), "st")
        times = ann.sample / header.fs
        epochs = [aux.split() for aux in ann.aux_note if aux.split()]
        if len(epochs) != len(times):
            raise RuntimeError(f"{rec_id}: annotation/token length mismatch")

        valid_start = float(times[0])
        valid_end = min(header.sig_len / header.fs, float(times[-1]) + 30.0)
        positive = []
        for i, tokens in enumerate(epochs):
            if any(tok in APNEA_TOKENS for tok in tokens[1:]):
                positive.append((float(times[i]), min(float(times[i]) + 30.0, valid_end)))
        records.append({
            "dataset": "MIT-BIH_PSG", "record_id": rec_id, "subject_id": f"slp{subject}",
            "duration_s": header.sig_len / header.fs,
            "valid_start_s": valid_start, "valid_end_s": valid_end,
            "positive_intervals": merge_intervals(positive),
            "ambiguous_intervals": [],
            "label_source": "30s_epoch",
        })
    return records, skipped


def subject_severity(records: list[dict]) -> dict[str, float]:
    severity = {}
    for r in records:
        positive_time = sum(e - s for s, e in r["positive_intervals"])
        annotated = r["valid_end_s"] - r["valid_start_s"]
        value = positive_time / annotated if annotated else 0.0
        severity.setdefault(r["subject_id"], []).append(value)
    return {s: float(np.mean(v)) for s, v in severity.items()}


def stratified_split(subjects: list[str], severity: dict[str, float], seed: int) -> dict[str, str]:
    """70/15/15 by severity tercile, allocated proportionally with integer rounding."""
    rng = random.Random(seed)
    ordered = sorted(subjects, key=lambda s: (severity[s], s))
    bins = [list(b) for b in np.array_split(np.array(ordered, dtype=object), 3)]
    n = len(ordered)

    def allocate(k: int) -> list[int]:
        sizes = np.array([len(b) for b in bins], dtype=float)
        if sizes.sum() == 0 or k <= 0:
            return [0] * len(bins)
        quota = sizes / sizes.sum() * k
        counts = np.minimum(np.floor(quota).astype(int),
                            np.array([max(0, len(b) - 1) for b in bins]))
        remainder = k - int(counts.sum())
        for i in np.argsort(-(quota - counts)):
            if remainder <= 0:
                break
            if counts[i] < len(bins[i]) - 1:
                counts[i] += 1
                remainder -= 1
        return counts.tolist()

    assignment = {}
    for split in ("val", "test"):
        k = max(2, round(0.15 * n))
        counts = allocate(k)
        for i, b in enumerate(bins):
            take = min(counts[i], len(b))
            for s in rng.sample(b, take):
                assignment[s] = split
                b.remove(s)

    for b in bins:
        for s in b:
            assignment[s] = "train"

    if len({v for v in assignment.values()}) < 3:
        raise RuntimeError(f"split {seed} failed: {assignment}")
    return assignment


def window_rows(record: dict, split: str, window_s: int, train_stride_s: float) -> tuple[list, dict]:
    stride = train_stride_s if split == "train" else float(window_s)
    start = float(record["valid_start_s"])
    rows = []
    stats = {"n": 0, "positive": 0, "ambiguous_only": 0, "near_threshold": 0,
             "excluded_lead_in_s": round(record["valid_start_s"], 1)}
    while start + window_s <= record["valid_end_s"] + 1e-9:
        end = start + window_s
        frac = overlap_seconds(start, end, record["positive_intervals"]) / window_s
        label = int(frac >= POSITIVE_OVERLAP)
        ambiguous = overlap_seconds(start, end, record["ambiguous_intervals"]) / window_s
        rows.append((round(start, 1), round(end, 1), label, round(frac, 4), round(ambiguous, 4)))
        stats["n"] += 1
        stats["positive"] += label
        if not label and ambiguous > 0:
            stats["ambiguous_only"] += 1
        if 0.4 <= frac <= 0.6:
            stats["near_threshold"] += 1
        start += stride
    stats["unused_tail_s"] = round(max(0.0, record["valid_end_s"] - start), 1)
    return rows, stats


def main() -> None:
    ucddb, ucddb_ahi = ucddb_records()
    psg, skipped_psg = psg_records()
    records = ucddb + psg

    severity = subject_severity(records)
    for rec in ucddb:
        severity[rec["subject_id"]] = ucddb_ahi[rec["subject_id"]]

    subjects_by_dataset = collections.defaultdict(set)
    for r in records:
        subjects_by_dataset[r["dataset"]].add(r["subject_id"])

    split_rows = []
    assignments = {}
    for seed in SEEDS:
        for dataset, subjects in subjects_by_dataset.items():
            assign = stratified_split(sorted(subjects), severity, seed)
            for subject, split in sorted(assign.items()):
                assignments[(seed, dataset, subject)] = split
                recs = sorted(r["record_id"] for r in records
                              if r["dataset"] == dataset and r["subject_id"] == subject)
                split_rows.append({
                    "seed": seed, "dataset": dataset, "subject_id": subject,
                    "split": split, "n_records": len(recs), "records": ";".join(recs),
                    "severity": round(severity[subject], 4),
                })

    with (OUT_SPLIT / "split_manifest.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(split_rows[0].keys()))
        writer.writeheader()
        writer.writerows(split_rows)

    summary = {}
    for window_s in DURATIONS:
        train_stride = window_s * TRAIN_STRIDE_FRACTION
        path = OUT_WIN / f"windows_{window_s}s.csv"
        counters = collections.defaultdict(lambda: collections.Counter())
        per_record_near = {}
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["seed", "dataset", "record_id", "subject_id", "split",
                             "window_start_s", "window_end_s", "label",
                             "positive_overlap_fraction", "ambiguous_overlap_fraction"])
            for seed in SEEDS:
                for rec in records:
                    split = assignments[(seed, rec["dataset"], rec["subject_id"])]
                    rows, stats = window_rows(rec, split, window_s, train_stride)
                    key = (seed, rec["dataset"], split)
                    counters[key]["windows"] += stats["n"]
                    counters[key]["positive"] += stats["positive"]
                    counters[key]["ambiguous_only"] += stats["ambiguous_only"]
                    counters[key]["lead_in_s"] += stats["excluded_lead_in_s"]
                    counters[key]["tail_s"] += stats["unused_tail_s"]
                    per_record_near[(seed, rec["record_id"])] = stats["near_threshold"]
                    for row in rows:
                        writer.writerow([seed, rec["dataset"], rec["record_id"],
                                         rec["subject_id"], split, *row])

        summary[f"{window_s}s"] = {}
        for (seed, dataset, split), c in sorted(counters.items()):
            summary[f"{window_s}s"][f"seed{seed}|{dataset}|{split}"] = {
                "windows": c["windows"],
                "positive": c["positive"],
                "positive_fraction": round(c["positive"] / c["windows"], 4) if c["windows"] else None,
                "ambiguous_only_windows": c["ambiguous_only"],
                "excluded_lead_in_s": round(c["lead_in_s"], 1),
                "unused_tail_s": round(c["tail_s"], 1),
            }
        summary[f"{window_s}s"]["near_threshold_windows"] = int(sum(per_record_near.values()))

    checks = {
        "subject_overlap_between_splits": {
            str(seed): {
                dataset: sorted(set.intersection(
                    *[{s for (sd, ds, s), sp in assignments.items() if sd == seed and ds == dataset and sp == split}
                      for split in ("train", "val", "test")]))
                for dataset in subjects_by_dataset
            } for seed in SEEDS
        },
        "split_subject_counts": {
            str(seed): {
                dataset: dict(collections.Counter(
                    assignments[(seed, dataset, s)] for s in subjects_by_dataset[dataset]))
                for dataset in subjects_by_dataset
            } for seed in SEEDS
        },
    }
    summary["checks"] = checks
    summary["excluded_psg_records"] = skipped_psg

    (OUT_WIN / "window_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
