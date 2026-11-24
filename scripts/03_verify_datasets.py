"""Verify local dataset copies: checksums, completeness, channel inventory."""

from __future__ import annotations

import collections
import csv
import hashlib
import json
import re
from pathlib import Path

import pyedflib
import wfdb

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dataset-rm" / "dataset-rm"
OUT = ROOT / "artifacts" / "verify"
OUT.mkdir(parents=True, exist_ok=True)


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def verify_checksums(ds_dir: Path) -> dict:
    sums = ds_dir / "SHA256SUMS.txt"
    listed, ok, mismatch, missing = 0, 0, [], []
    for line in sums.read_text().splitlines():
        parts = line.split()
        if len(parts) != 2:
            continue
        expected, name = parts
        listed += 1
        target = ds_dir / name
        if not target.exists():
            missing.append(name)
            continue
        if sha256(target) == expected:
            ok += 1
        else:
            mismatch.append(name)
    return {"listed": listed, "ok": ok, "mismatch": mismatch, "missing": missing}


def verify_ucddb() -> dict:
    combos = collections.Counter()
    no_ecg, no_resp = [], []
    for rec in sorted((DATA / "ucddb" / "DESTINATION").glob("ucddb*.rec")):
        f = pyedflib.EdfReader(str(rec))
        labels = tuple(f.getLabel(i) for i in range(f.signals_in_file))
        fs = tuple(f.getSampleFrequency(i) for i in range(f.signals_in_file))
        combos[(labels, fs)] += 1
        if "ECG" not in labels:
            no_ecg.append(rec.stem)
        if not any(c in labels for c in ("Flow", "ribcage", "abdo", "Sum")):
            no_resp.append(rec.stem)
        f.close()
    return {
        "records": len(list((DATA / "ucddb" / "DESTINATION").glob("ucddb*.rec"))),
        "channel_layouts": len(combos),
        "records_missing_ecg": no_ecg,
        "records_missing_resp": no_resp,
        "layouts": [{"n_channels": len(labels), "ecg_fs": dict(zip(labels, fs)).get("ECG"),
                     "resp_fs": [dict(zip(labels, fs))[c] for c in ("Flow", "ribcage", "abdo", "Sum")
                                 if c in labels],
                     "count": n} for (labels, fs), n in combos.items()],
    }


def verify_slpdb() -> dict:
    combos = collections.Counter()
    no_ecg, no_resp = [], []
    records = (DATA / "slpdb" / "DESTINATION" / "RECORDS").read_text().split()
    per_record = []
    for rec in records:
        h = wfdb.rdheader(str(DATA / "slpdb" / "DESTINATION" / rec))
        combos[(tuple(h.sig_name), h.fs)] += 1
        resp = [n for n in h.sig_name if "resp" in n.lower()]
        per_record.append({"record": rec, "subject": re.match(r"slp(\d+)", rec).group(1),
                           "resp_channels": ";".join(resp), "n_channels": len(h.sig_name),
                           "fs": h.fs})
        if not any(n.upper().startswith("ECG") for n in h.sig_name):
            no_ecg.append(rec)
        if not resp:
            no_resp.append(rec)

    with (OUT / "slpdb_resp_channels.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(per_record[0].keys()))
        writer.writeheader()
        writer.writerows(per_record)

    return {
        "records": len(records),
        "channel_layouts": len(combos),
        "records_with_nasal": sum("nasal" in r["resp_channels"] for r in per_record),
        "records_missing_ecg": no_ecg,
        "records_missing_resp": no_resp,
        "layouts": [{"channels": list(names), "fs": fs, "count": n}
                    for (names, fs), n in combos.items()],
    }


def verify_apnea_ecg() -> dict:
    combos = collections.Counter()
    records = (DATA / "apnea-ecg" / "DESTINATION" / "RECORDS").read_text().split()
    base = [r for r in records if not re.fullmatch(r"[abc]\d{2}(r|er)", r)]
    for rec in base:
        h = wfdb.rdheader(str(DATA / "apnea-ecg" / "DESTINATION" / rec))
        combos[(tuple(h.sig_name), h.fs)] += 1
    return {
        "records_listed": len(records),
        "base_records": len(base),
        "derived_records": len(records) - len(base),
        "channel_layouts": len(combos),
        "layouts": [{"channels": list(names), "fs": fs, "count": n}
                    for (names, fs), n in combos.items()],
        "n_signal_channel_counts": sorted({len(k[0]) for k in combos}),
    }


def main() -> None:
    result = {
        "ucddb": {"checksums": verify_checksums(DATA / "ucddb" / "DESTINATION"),
                  "channels": verify_ucddb()},
        "slpdb": {"checksums": verify_checksums(DATA / "slpdb" / "DESTINATION"),
                  "channels": verify_slpdb()},
        "apnea_ecg": {"checksums": verify_checksums(DATA / "apnea-ecg" / "DESTINATION"),
                      "channels": verify_apnea_ecg()},
    }
    (OUT / "dataset_verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    for ds, payload in result.items():
        c = payload["checksums"]
        print(f"{ds}: checksums {c['ok']}/{c['listed']} ok, "
              f"mismatch={len(c['mismatch'])}, missing={len(c['missing'])}")
        print("  channels:", json.dumps(payload["channels"])[:600])


if __name__ == "__main__":
    main()
