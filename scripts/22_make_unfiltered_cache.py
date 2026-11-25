"""Stage 3a-b: build an unfiltered cache for the preprocessing ablation.

The main cache (scripts/04_preprocess.py) applies ECG band-pass 0.5-40 Hz with notch,
and respiration low-pass 0.7 Hz with linear detrend. Ablation arm A4 asks what happens
without that signal filtering, so this script rebuilds the cache with resampling only.

IMPORTANT LIMITATION, must be reported with any A4 result:
resample_poly is an anti-alias filter by construction. Bringing ECG to 100 Hz already
attenuates above 50 Hz, and respiration to 25 Hz already attenuates above 12.5 Hz. So
this is "no added filter" (no explicit band-pass, low-pass, or detrend), not "no
filtering at all". The `detect_mains`/notch stage is also skipped.

Output goes to artifacts/signals_unfiltered/ with the same file naming as the main
cache, so the loader can switch directory without other changes.

Run:
    python scripts/22_make_unfiltered_cache.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pyedflib
import wfdb

sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib

prep = importlib.import_module("04_preprocess")

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dataset-rm" / "dataset-rm"
OUT = ROOT / "artifacts" / "signals_unfiltered"
EXCLUDED = prep.EXCLUDED_RECORDS

CONFIG = {
    "variant": "unfiltered_resample_only",
    "ecg": {"explicit_filter": None, "notch": "skipped", "target_fs_hz": 100.0,
            "caveat": "resample_poly anti-alias attenuates above ~50 Hz"},
    "resp": {"explicit_filter": None, "detrend": "skipped", "target_fs_hz": 25.0,
             "caveat": "resample_poly anti-alias attenuates above ~12.5 Hz"},
    "note": "Ablation arm A4. This is 'no added filter', not 'no filtering': the "
            "anti-alias step required by resampling remains, and it is itself a low-pass.",
}


def process_unfiltered(rec_id: str, dataset: str) -> dict:
    if dataset == "UCDDB":
        ecg, fs_ecg, resp, fs_resp, duration, resp_name, resp_type = prep.load_ucddb(rec_id)
    else:
        ecg, fs_ecg, resp, fs_resp, duration, resp_name, resp_type = prep.load_psg(rec_id)

    ecg = np.nan_to_num(ecg, nan=0.0, posinf=0.0, neginf=0.0)
    resp = np.nan_to_num(resp, nan=0.0, posinf=0.0, neginf=0.0)

    ecg_r = prep.resample(ecg, fs_ecg, prep.ECG_FS)
    resp_r = prep.resample(resp, fs_resp, prep.RESP_FS)

    n = min(len(ecg_r), int(round(len(resp_r) * prep.ECG_FS / prep.RESP_FS)))
    ecg_r = ecg_r[:n]
    resp_r = resp_r[: int(round(n * prep.RESP_FS / prep.ECG_FS))]

    np.save(OUT / f"{rec_id}_ecg100.npy", ecg_r.astype(np.float32))
    np.save(OUT / f"{rec_id}_resp25.npy", resp_r.astype(np.float32))

    return {
        "dataset": dataset, "record_id": rec_id,
        "resp_channel": resp_name, "resp_type": resp_type,
        "duration_s": round(duration, 1),
        "ecg_out_samples": len(ecg_r), "resp_out_samples": len(resp_r),
        "duration_mismatch_ms": round(
            abs(len(ecg_r) / prep.ECG_FS - len(resp_r) / prep.RESP_FS) * 1000, 1),
        "ecg_std": round(float(np.std(ecg_r)), 4),
        "resp_std": round(float(np.std(resp_r)), 4),
        "ecg_mean": round(float(np.mean(ecg_r)), 4),
        "resp_mean": round(float(np.mean(resp_r)), 4),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []

    for rec_path in sorted((DATA / "ucddb" / "DESTINATION").glob("ucddb*.rec")):
        if rec_path.stem in EXCLUDED:
            continue
        rows.append(process_unfiltered(rec_path.stem, "UCDDB"))
        print("UCDDB", rec_path.stem, flush=True)

    for rec_id in (DATA / "slpdb" / "DESTINATION" / "RECORDS").read_text().split():
        if rec_id in EXCLUDED:
            continue
        header = wfdb.rdheader(str(DATA / "slpdb" / "DESTINATION" / rec_id))
        if not any("nasal" in n.lower() for n in header.sig_name):
            continue
        rows.append(process_unfiltered(rec_id, "MIT-BIH_PSG"))
        print("PSG", rec_id, flush=True)

    with (OUT / "signal_qc_unfiltered.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    (OUT / "preprocessing_config_unfiltered.json").write_text(
        json.dumps(CONFIG, indent=2), encoding="utf-8")

    # Pairwise check: the unfiltered cache must have identical length to the main cache,
    # otherwise window start times would no longer line up with the manifest.
    main_dir = ROOT / "artifacts" / "signals"
    mismatches = []
    for r in rows:
        for mod, fs in (("ecg", 100), ("resp", 25)):
            a = np.load(main_dir / f"{r['record_id']}_{mod}{fs}.npy", mmap_mode="r")
            b = np.load(OUT / f"{r['record_id']}_{mod}{fs}.npy", mmap_mode="r")
            if a.shape != b.shape:
                mismatches.append({"record": r["record_id"], "modality": mod,
                                   "main": list(a.shape), "unfiltered": list(b.shape)})

    print(f"\nrecords: {len(rows)}")
    print(f"length mismatches vs main cache: {len(mismatches)}")
    if mismatches:
        print(json.dumps(mismatches[:10], indent=2))
        raise SystemExit("length mismatch would break window alignment; aborting")
    print("alignment ok: every record has identical sample counts to the main cache")


if __name__ == "__main__":
    main()
