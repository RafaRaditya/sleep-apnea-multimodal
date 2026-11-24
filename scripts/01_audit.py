"""Stage 1: record and annotation audit.

Read-only audit of UCDDB, MIT-BIH PSG (slpdb), and Apnea-ECG.
Writes artifacts/audit/audit_manifest.csv and audit_summary.md.
"""

from __future__ import annotations

import collections
import csv
import json
import re
from datetime import datetime
from pathlib import Path

import pyedflib
import wfdb

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dataset-rm" / "dataset-rm"
POLICY = json.loads((ROOT / "config" / "policy.json").read_text(encoding="utf-8"))
EXCLUDED_RECORDS = POLICY["multimodal_policy"]["excluded_records"]
OUT = ROOT / "artifacts" / "audit"
OUT.mkdir(parents=True, exist_ok=True)

APNEA_EVENTS = {"H", "HA", "OA", "X", "CA", "CAA"}

MANIFEST_FIELDS = [
    "dataset", "dataset_version", "record_id", "subject_id",
    "ecg_channel", "resp_channel", "resp_type",
    "sampling_rate_ecg", "sampling_rate_resp", "duration_s",
    "time_offset_s", "annotation_type", "annotation_resolution",
    "annotation_available", "annotated_duration_s",
    "positive_fraction", "eligible", "exclusion_reason",
]


def parse_ucddb_events(path: Path) -> list[dict]:
    """Parse _respevt.txt fixed-width rows: time | type | PB/CS | duration."""
    events = []
    lines = path.read_text(errors="ignore").splitlines()
    for line in lines[3:]:
        if len(line) < 34 or not re.match(r"\s*\d{2}:\d{2}:\d{2}", line[:11]):
            continue
        time_str = line[0:10].strip()
        etype = line[10:18].strip()
        pbcs = line[18:26].strip()
        dur_raw = line[26:34].strip()
        try:
            duration = int(dur_raw)
        except ValueError:
            continue
        if etype.startswith("HYP"):
            category = "hypopnea"
        elif etype.startswith("APNEA"):
            category = "apnea"
        elif etype.startswith("PB"):
            category = "periodic_breathing"
        else:
            category = "possible"
        events.append({"time": time_str, "type": etype, "pbcs": pbcs,
                       "category": category, "duration": duration})
    return events


def rel_seconds(event_time, record_start) -> float:
    """Wall-clock annotation time -> seconds from record start, handling midnight wrap."""
    delta = (datetime.combine(record_start.date(), event_time)
             - datetime.combine(record_start.date(), record_start.time())).total_seconds()
    if delta < -12 * 3600:
        delta += 24 * 3600
    elif delta > 12 * 3600:
        delta -= 24 * 3600
    return delta


def ucddb_audit() -> tuple[list[dict], dict]:
    rows = []
    stats = {"records": 0, "event_types": collections.Counter(), "durations_h": [],
             "positive_fraction": [], "signal_sets": collections.Counter(),
             "unannotated_tail_s": [], "per_record_channels": {},
             "event_categories": collections.Counter(), "positive_seconds": 0,
             "total_seconds": 0, "possible_seconds": 0, "pb_seconds": 0,
             "event_count": 0, "bound_violations": 0}

    for rec_path in sorted((DATA / "ucddb" / "DESTINATION").glob("ucddb*.rec")):
        rec_id = rec_path.stem
        subject = rec_id.replace("ucddb", "").lstrip("0")
        edf = pyedflib.EdfReader(str(rec_path))
        labels = [edf.getLabel(i) for i in range(edf.signals_in_file)]
        fs = [edf.getSampleFrequency(i) for i in range(edf.signals_in_file)]
        duration = edf.file_duration
        start = edf.getStartdatetime()
        n_sig = edf.signals_in_file
        edf.close()

        label_fs = dict(zip(labels, fs))
        stats["per_record_channels"][rec_id] = list(zip(labels, fs))
        ecg_ch = "ECG" if "ECG" in label_fs else None
        resp_candidates = [c for c in ("Flow", "ribcage", "abdo", "Sum") if c in label_fs]
        resp_ch = resp_candidates[0] if resp_candidates else None
        stats["signal_sets"][n_sig] += 1

        ev_path = DATA / "ucddb" / "DESTINATION" / f"{rec_id}_respevt.txt"
        events = parse_ucddb_events(ev_path) if ev_path.exists() else []
        for e in events:
            stats["event_types"][e["type"]] += 1
            stats["event_categories"][e["category"]] += 1

        stage_path = DATA / "ucddb" / "DESTINATION" / f"{rec_id}_stage.txt"
        stages = [int(x) for x in stage_path.read_text().split()] if stage_path.exists() else []

        offset_s = None
        violations = 0
        if events:
            t0 = datetime.strptime(events[0]["time"], "%H:%M:%S").time()
            offset_s = rel_seconds(t0, start)
            for e in events:
                rel = rel_seconds(datetime.strptime(e["time"], "%H:%M:%S").time(), start)
                e["rel_start_s"] = rel
                if rel < 0 or rel + e["duration"] > duration + 30:
                    violations += 1
        stats["bound_violations"] += violations

        positive_s = sum(e["duration"] for e in events
                         if e["category"] in ("apnea", "hypopnea"))
        stats["possible_seconds"] += sum(e["duration"] for e in events
                                        if e["category"] == "possible")
        stats["pb_seconds"] += sum(e["duration"] for e in events
                                   if e["category"] == "periodic_breathing")
        stats["positive_seconds"] += positive_s
        stats["total_seconds"] += duration
        stats["event_count"] += len(events)
        frac = positive_s / duration if duration else None
        stats["durations_h"].append(duration / 3600)
        if frac is not None:
            stats["positive_fraction"].append(frac)
        annotated_s = len(stages) * 30
        stats["unannotated_tail_s"].append(duration - annotated_s)

        rows.append({
            "dataset": "UCDDB", "dataset_version": "1.0.0", "record_id": rec_id,
            "subject_id": subject, "ecg_channel": ecg_ch,
            "resp_channel": resp_ch,
            "resp_type": "airflow(thermistor)+effort(ribcage/abdo)" if resp_ch == "Flow"
                        else ("effort" if resp_ch else None),
            "sampling_rate_ecg": label_fs.get(ecg_ch) if ecg_ch else None,
            "sampling_rate_resp": label_fs.get(resp_ch) if resp_ch else None,
            "duration_s": round(duration, 1), "time_offset_s": offset_s,
            "annotation_type": "respiratory event (onset+duration); sleep stage 30s",
            "annotation_resolution": "event-onset/duration; stage=30s",
            "annotation_available": True, "annotated_duration_s": annotated_s,
            "positive_fraction": round(frac, 4) if frac is not None else None,
            "eligible": bool(ecg_ch and resp_ch) and rec_id not in EXCLUDED_RECORDS,
            "exclusion_reason": EXCLUDED_RECORDS.get(rec_id)
            or ("" if (ecg_ch and resp_ch) else "missing ECG or respiratory channel"),
        })

    stats["records"] = len(rows)
    return rows, stats


def pick_resp_channel(channels: list[str]) -> tuple[str | None, str | None]:
    """Airflow first, then effort sum, then single effort channel."""
    for name in channels:
        low = name.lower()
        if "nasal" in low:
            return name, "airflow (nasal)"
    for name in channels:
        if "sum" in name.lower():
            return name, "effort sum"
    for name in channels:
        low = name.lower()
        if "chest" in low:
            return name, "thoracic effort"
    for name in channels:
        low = name.lower()
        if "abdom" in low:
            return name, "abdominal effort"
    return (channels[0], "respiratory (unspecified)") if channels else (None, None)


def slpdb_audit() -> tuple[list[dict], dict]:
    rows = []
    stats = {"records": 0, "subjects": set(), "epoch_events": collections.Counter(),
             "durations_h": [], "positive_fraction": [], "channel_sets": collections.Counter(),
             "anno_epochs": 0, "signal_epochs": 0, "unannotated_tail_s": [],
             "apnea_epochs": 0, "total_epochs": 0,
             "resp_channel_counts": collections.Counter(),
             "resp_type_counts": collections.Counter(), "eligible": 0,
             "excluded_records": []}

    for rec_id in (DATA / "slpdb" / "DESTINATION" / "RECORDS").read_text().split():
        record = wfdb.rdheader(str(DATA / "slpdb" / "DESTINATION" / rec_id))
        subject = re.match(r"slp(\d+)", rec_id).group(1)
        fs = record.fs
        dur = record.sig_len / fs
        names = record.sig_name
        ecg_ch = next((n for n in names if n.upper().startswith("ECG")), None)
        resp_channels = [n for n in names if "resp" in n.lower()]
        has_airflow = any("nasal" in n.lower() for n in resp_channels)
        resp_ch, resp_type = pick_resp_channel(resp_channels)
        stats["resp_channel_counts"][resp_ch] += 1
        stats["resp_type_counts"][resp_type] += 1
        stats["channel_sets"][len(names)] += 1
        stats["subjects"].add(subject)

        ann = wfdb.rdann(str(DATA / "slpdb" / "DESTINATION" / rec_id), "st")
        epochs = []
        for aux in ann.aux_note:
            tokens = aux.split()
            if not tokens:
                continue
            epochs.append(tokens)
        for tokens in epochs:
            for tok in tokens[1:]:
                stats["epoch_events"][tok] += 1
        apnea_epochs = 0
        for tokens in epochs:
            if any(tok in APNEA_EVENTS for tok in tokens[1:]):
                apnea_epochs += 1
        epoch_seconds = 30
        annotated_s = len(epochs) * epoch_seconds
        frac = apnea_epochs * epoch_seconds / annotated_s if annotated_s else None

        stats["records"] += 1
        if ecg_ch and has_airflow:
            stats["eligible"] += 1
        else:
            stats["excluded_records"].append(rec_id)
        stats["durations_h"].append(dur / 3600)
        if frac is not None:
            stats["positive_fraction"].append(frac)
        stats["anno_epochs"] += len(ann.sample)
        stats["apnea_epochs"] += apnea_epochs
        stats["total_epochs"] += len(epochs)
        stats["signal_epochs"] += int(dur // epoch_seconds)
        stats["unannotated_tail_s"].append(dur - len(ann.sample) * epoch_seconds)

        rows.append({
            "dataset": "MIT-BIH_PSG", "dataset_version": "1.0.0", "record_id": rec_id,
            "subject_id": f"slp{subject}", "ecg_channel": ecg_ch, "resp_channel": resp_ch,
            "resp_type": resp_type,
            "sampling_rate_ecg": fs if ecg_ch else None,
            "sampling_rate_resp": fs if resp_ch else None,
            "duration_s": round(dur, 1), "time_offset_s": 0,
            "annotation_type": "30s epoch: sleep stage + event indicators (H/OA/X/CA/L/A...)",
            "annotation_resolution": "30s epoch, no onset/duration",
            "annotation_available": True,
            "annotated_duration_s": annotated_s,
            "positive_fraction": round(frac, 4) if frac is not None else None,
            "eligible": bool(ecg_ch and has_airflow) and rec_id not in EXCLUDED_RECORDS,
            "exclusion_reason": EXCLUDED_RECORDS.get(rec_id)
            or ("" if (ecg_ch and has_airflow)
                else ("no airflow channel (effort only) under strict airflow policy"
                      if ecg_ch else "missing ECG channel")),
        })
    stats["n_subjects"] = len(stats["subjects"])
    return rows, stats


def apnea_ecg_audit() -> tuple[list[dict], dict]:
    rows = []
    stats = {"records": 0, "durations_h": [], "positive_fraction": [],
             "classes": collections.Counter(), "groups": collections.Counter(),
             "unannotated_tail_s": [], "derived_records": 0}

    for rec_id in (DATA / "apnea-ecg" / "DESTINATION" / "RECORDS").read_text().split():
        if re.fullmatch(r"[abc]\d{2}(r|er)", rec_id):
            stats["derived_records"] += 1
            continue
        record = wfdb.rdheader(str(DATA / "apnea-ecg" / "DESTINATION" / rec_id))
        fs = record.fs
        dur = record.sig_len / fs
        ann = wfdb.rdann(str(DATA / "apnea-ecg" / "DESTINATION" / rec_id), "apn")
        counts = collections.Counter(ann.symbol)
        stats["classes"].update(counts)
        minute = 60
        positive = counts.get("A", 0) / len(ann.symbol) if len(ann.symbol) else None
        group = rec_id[0]
        stats["groups"][group] += 1
        stats["records"] += 1
        stats["durations_h"].append(dur / 3600)
        if positive is not None:
            stats["positive_fraction"].append(positive)
        stats["unannotated_tail_s"].append(dur - len(ann.symbol) * minute)

        rows.append({
            "dataset": "Apnea-ECG", "dataset_version": "1.0.0", "record_id": rec_id,
            "subject_id": rec_id, "ecg_channel": record.sig_name[0],
            "resp_channel": None, "resp_type": None,
            "sampling_rate_ecg": fs, "sampling_rate_resp": None,
            "duration_s": round(dur, 1), "time_offset_s": 0,
            "annotation_type": "per-minute apnea label (A/N)",
            "annotation_resolution": "60s minute", "annotation_available": True,
            "annotated_duration_s": len(ann.symbol) * minute,
            "positive_fraction": round(positive, 4) if positive is not None else None,
            "eligible": False,
            "exclusion_reason": "no respiratory channel; ECG-only database",
        })
    return rows, stats


def fmt_counter(counter: collections.Counter) -> str:
    return ", ".join(f"{k}={v}" for k, v in counter.most_common())


def mean_sd(values: list[float]) -> str:
    if not values:
        return "n/a"
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    return f"{mean:.2f} +/- {var ** 0.5:.2f}"


def main() -> None:
    ucddb_rows, ucddb_stats = ucddb_audit()
    slpdb_rows, slpdb_stats = slpdb_audit()
    apnea_rows, apnea_stats = apnea_ecg_audit()
    rows = ucddb_rows + slpdb_rows + apnea_rows

    with (OUT / "audit_manifest.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    with (OUT / "ucddb_channels.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["record_id", "channel_index", "label", "sampling_rate"])
        for rec_id, channels in ucddb_stats["per_record_channels"].items():
            for idx, (label, fs) in enumerate(channels):
                writer.writerow([rec_id, idx, label, fs])

    summary = {
        "UCDDB": {
            "records": ucddb_stats["records"],
            "total_hours": round(sum(ucddb_stats["durations_h"]), 1),
            "duration_h_mean_sd": mean_sd(ucddb_stats["durations_h"]),
            "event_positive_fraction_mean": round(
                sum(ucddb_stats["positive_fraction"]) / len(ucddb_stats["positive_fraction"]), 4),
            "event_positive_fraction_pooled": round(
                ucddb_stats["positive_seconds"] / ucddb_stats["total_seconds"], 4),
            "event_count": ucddb_stats["event_count"],
            "event_categories": dict(ucddb_stats["event_categories"]),
            "possible_event_seconds": ucddb_stats["possible_seconds"],
            "periodic_breathing_seconds": ucddb_stats["pb_seconds"],
            "event_types": dict(ucddb_stats["event_types"]),
            "signal_count_distribution": dict(ucddb_stats["signal_sets"]),
            "unannotated_tail_s_mean": round(
                sum(ucddb_stats["unannotated_tail_s"]) / len(ucddb_stats["unannotated_tail_s"]), 1),
            "event_bounds_violations": ucddb_stats["bound_violations"],
        },
        "MIT-BIH_PSG": {
            "records": slpdb_stats["records"],
            "subjects": slpdb_stats["n_subjects"],
            "total_hours": round(sum(slpdb_stats["durations_h"]), 1),
            "duration_h_mean_sd": mean_sd(slpdb_stats["durations_h"]),
            "apnea_epoch_fraction_mean": round(
                sum(slpdb_stats["positive_fraction"]) / len(slpdb_stats["positive_fraction"]), 4),
            "apnea_epoch_fraction_pooled": round(
                slpdb_stats["apnea_epochs"] / slpdb_stats["total_epochs"], 4),
            "event_tokens": dict(slpdb_stats["epoch_events"]),
            "resp_channel_counts": dict(slpdb_stats["resp_channel_counts"]),
            "resp_type_counts": dict(slpdb_stats["resp_type_counts"]),
            "eligible_records": slpdb_stats["eligible"],
            "excluded_records": slpdb_stats["excluded_records"],
            "annotation_coverage_epochs": f"{slpdb_stats['anno_epochs']}/{slpdb_stats['signal_epochs']}",
            "unannotated_tail_s_mean": round(
                sum(slpdb_stats["unannotated_tail_s"]) / len(slpdb_stats["unannotated_tail_s"]), 1),
        },
        "Apnea-ECG": {
            "records": apnea_stats["records"],
            "total_hours": round(sum(apnea_stats["durations_h"]), 1),
            "duration_h_mean_sd": mean_sd(apnea_stats["durations_h"]),
            "minute_positive_fraction_mean": round(
                sum(apnea_stats["positive_fraction"]) / len(apnea_stats["positive_fraction"]), 4),
            "minute_positive_fraction_pooled": round(
                apnea_stats["classes"].get("A", 0) / sum(apnea_stats["classes"].values()), 4),
            "label_counts": dict(apnea_stats["classes"]),
            "record_groups": dict(apnea_stats["groups"]),
            "derived_records_excluded": apnea_stats["derived_records"],
        },
        "eligible_manifest_rows": sum(1 for r in rows if r["eligible"]),
    }

    (OUT / "audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = ["# Stage 1 Audit Summary", ""]
    for ds, s in summary.items():
        if isinstance(s, int):
            continue
        lines += [f"## {ds}", ""]
        for k, v in s.items():
            lines.append(f"- {k}: {v}")
        lines.append("")
    lines.append(f"Eligible multimodal records: {summary['eligible_manifest_rows']}")
    (OUT / "audit_summary.md").write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"\nWrote {OUT / 'audit_manifest.csv'}")
    print(f"Wrote {OUT / 'audit_summary.md'}")


if __name__ == "__main__":
    main()
