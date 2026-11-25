"""Stage 7c: bundle the files Colab needs into one zip.

The raw PhysioNet datasets are not included: the signal cache is already filtered and
resampled, so Colab never needs pyedflib or wfdb.

Run:
    python scripts/21_make_cache_zip.py
    python scripts/21_make_cache_zip.py --out D:\\research_cache.zip

Output: research_cache.zip at the repo root (about 200 MB compressed).
Upload it to Google Drive as MyDrive/research/research_cache.zip.
"""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"

INCLUDE_DIRS = [
    ("scripts", "scripts", "*.py"),
    ("config", "config", "*.json"),
    ("artifacts/signals", "artifacts/signals", "*.npy"),
    ("artifacts/signals", "artifacts/signals", "*.csv"),
    ("artifacts/signals", "artifacts/signals", "*.json"),
    ("artifacts/windows", "artifacts/windows", "*.csv"),
    ("artifacts/windows", "artifacts/windows", "*.json"),
    ("artifacts/normalization", "artifacts/normalization", "*.json"),
]

SKIP_PARTS = {"__pycache__"}


def collect() -> list[tuple[Path, str]]:
    items: list[tuple[Path, str]] = []
    for src_rel, arc_root, pattern in INCLUDE_DIRS:
        src = ROOT / src_rel
        if not src.exists():
            raise SystemExit(f"missing directory: {src}")
        for path in sorted(src.glob(pattern)):
            if SKIP_PARTS & set(path.parts):
                continue
            items.append((path, f"{arc_root}/{path.name}"))
    return items


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "research_cache.zip"))
    args = ap.parse_args()

    items = collect()
    total = sum(p.stat().st_size for p, _ in items)
    print(f"{len(items)} files, {total / 1e6:.1f} MB uncompressed")

    manifest = []
    out = Path(args.out)
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for path, arcname in items:
            z.write(path, arcname)
            manifest.append({"arcname": arcname, "bytes": path.stat().st_size})

    size = out.stat().st_size / 1e6
    print(f"wrote {out} ({size:.1f} MB compressed)")

    (out.with_suffix(".manifest.json")).write_text(json.dumps({
        "created_from": str(ROOT),
        "file_count": len(items),
        "uncompressed_mb": round(total / 1e6, 1),
        "compressed_mb": round(size, 1),
        "files": manifest,
        "excluded": "raw datasets (apnea-ecg, slpdb, ucddb, dataset-rm), checkpoints, "
                    "train/test outputs",
    }, indent=2), encoding="utf-8")
    print(f"wrote {out.with_suffix('.manifest.json')}")
    print("\nNext: upload the zip to Google Drive at MyDrive/research/research_cache.zip, "
          "then open notebooks/colab_train_30s.ipynb in Colab.")


if __name__ == "__main__":
    main()
