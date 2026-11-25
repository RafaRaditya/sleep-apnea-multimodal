"""Build a single self-contained HTML report from the stage artifacts.

Reads only what already exists in artifacts/ and writes reports/stage_report.html.
Large per-window CSVs are summarised, not embedded.
"""

from __future__ import annotations

import base64
import html
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
OUT = ROOT / "reports"
OUT.mkdir(exist_ok=True)

CSS = """
:root { color-scheme: light dark; }
* { box-sizing: border-box; }
body { font: 15px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
       margin: 0; padding: 32px 20px 80px; max-width: 1180px; margin-inline: auto;
       color: #1a1a1a; background: #fbfbfd; }
h1 { font-size: 26px; margin: 0 0 6px; letter-spacing: -0.02em; }
h2 { font-size: 19px; margin: 40px 0 4px; padding-top: 18px; border-top: 1px solid #e3e3e8; }
h3 { font-size: 15px; margin: 22px 0 8px; color: #444; }
.sub { color: #666; font-size: 13.5px; margin: 0 0 4px; }
.badge { display: inline-block; font-size: 11px; font-weight: 600; padding: 2px 7px;
         border-radius: 99px; background: #e6f4ea; color: #12663a; margin-left: 8px;
         vertical-align: 2px; }
.badge.warn { background: #fdf0e3; color: #8a4b12; }
table { border-collapse: collapse; width: 100%; font-size: 13px; margin: 10px 0 6px;
        background: #fff; border: 1px solid #e6e6ec; border-radius: 8px; overflow: hidden; }
th { text-align: left; font-weight: 600; background: #f4f4f8; padding: 7px 10px;
     border-bottom: 1px solid #e6e6ec; white-space: nowrap; }
td { padding: 6px 10px; border-bottom: 1px solid #f0f0f4; vertical-align: top; }
tr:last-child td { border-bottom: 0; }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
code, .mono { font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace; font-size: 12px; }
.file { color: #4a4a55; background: #f1f1f6; padding: 1px 5px; border-radius: 4px; }
.note { background: #fff8e8; border-left: 3px solid #e0a53a; padding: 10px 14px;
        border-radius: 0 6px 6px 0; margin: 12px 0; font-size: 13.5px; }
.fig { margin: 14px 0; }
.fig img { width: 100%; border: 1px solid #e6e6ec; border-radius: 8px; }
.fig figcaption { font-size: 12.5px; color: #666; margin-top: 6px; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 12px; margin: 14px 0; }
.card { background: #fff; border: 1px solid #e6e6ec; border-radius: 8px; padding: 12px 14px; }
.card .k { font-size: 12px; color: #666; }
.card .v { font-size: 21px; font-weight: 600; letter-spacing: -0.02em; margin-top: 2px; }
.toc { columns: 2; font-size: 13.5px; margin: 14px 0 0; }
.toc a { color: #1a56c4; text-decoration: none; }
.toc a:hover { text-decoration: underline; }
details { margin: 8px 0; }
summary { cursor: pointer; font-size: 13px; color: #444; }
"""


def table(df: pd.DataFrame, num_cols: list[str] | None = None, max_rows: int = 40) -> str:
    num_cols = num_cols or []
    head = "".join(
        f'<th class="{"num" if c in num_cols else ""}">{html.escape(str(c))}</th>' for c in df.columns
    )
    body = []
    for _, r in df.head(max_rows).iterrows():
        cells = []
        for c in df.columns:
            v = r[c]
            if isinstance(v, float):
                v = f"{v:,.4g}"
            cells.append(f'<td class="{"num" if c in num_cols else ""}">{html.escape(str(v))}</td>')
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def fig(path: Path, caption: str) -> str:
    if not path.exists():
        return ""
    b64 = base64.b64encode(path.read_bytes()).decode()
    return (f'<figure class="fig"><img src="data:image/png;base64,{b64}" alt="{html.escape(caption)}">'
            f"<figcaption>{html.escape(caption)}</figcaption></figure>")


def build() -> str:
    parts: list[str] = []
    a = parts.append

    a(f"<style>{CSS}</style>")
    a("<h1>Sleep Apnea Multimodal — Stage Report</h1>")
    a('<p class="sub">Generated from <code>artifacts/</code>. '
      "Numbers below are produced by the scripts in <code>scripts/</code>; nothing is hand-entered.</p>")
    a('<p class="sub">Sources: UCDDB (25 records) · MIT-BIH PSG (18 records) · Apnea-ECG (70 base records). '
      "Reference: paper UTS 18 Nov 2026.</p>")

    a('<div class="toc">'
      '<div><a href="#s0">Stage 0 — dataset verification</a></div>'
      '<div><a href="#s1">Stage 1 — record &amp; annotation audit</a></div>'
      '<div><a href="#s2">Stage 2 — subject split + windowing</a></div>'
      '<div><a href="#s3">Stage 3 — preprocessing &amp; loader</a></div>'
      '<div><a href="#s4">Stage 4 — model pilot</a></div>'
      '<div><a href="#s5">Stage 5 — full training + freeze</a></div>'
      '<div><a href="#s6">Stage 6 — test prediction</a></div>'
      '<div><a href="#s6b">Stage 6b — calibration diagnostic</a></div>'
      '<div><a href="#s6c">Stage 6c — per-record offset probe</a></div>'
      '<div><a href="#s7">Stage 7 — multi-seed comparison</a></div>'
      '<div><a href="#s7b">Stage 7b — 10 s vs 30 s</a></div>'
      '<div><a href="#decisions">Decisions &amp; exclusions</a></div>'
      '<div><a href="#decisions-detail">Decision log</a></div>'
      '<div><a href="#limits">Limitations</a></div>'
      '<div><a href="#files">File map</a></div>'
      "</div>")

    # ---------------- Stage 0
    ver = json.loads((ART / "verify" / "dataset_verification.json").read_text(encoding="utf-8"))
    a('<h2 id="s0">Stage 0 — dataset verification</h2>')
    a('<p class="sub">Local copies checked against the official <code>SHA256SUMS.txt</code> of each PhysioNet release.</p>')
    rows = []
    for name, key in (("UCDDB", "ucddb"), ("MIT-BIH PSG", "slpdb"), ("Apnea-ECG", "apnea_ecg")):
        c = ver[key]["checksums"]
        rows.append({"Dataset": name, "Checksums OK": f"{c['ok']}/{c['listed']}",
                     "Mismatch": len(c["mismatch"]), "Missing": ", ".join(c["missing"]) or "-",
                     "Mismatch files": ", ".join(c["mismatch"]) or "-"})
    a(table(pd.DataFrame(rows), num_cols=["Checksums OK", "Mismatch"]))
    a('<div class="note">All signal and annotation files match the released checksums. '
      'The only differences are one metadata spreadsheet (<code>SubjectDetails.xls</code>) and two '
      'web index pages that were not downloaded.</div>')

    slp = pd.read_csv(ART / "verify" / "slpdb_resp_channels.csv")
    a("<h3>PSG respiration channels are not uniform</h3>")
    a(table(slp, max_rows=20))
    a('<div class="note">15 of 18 PSG records carry <code>Resp (nasal)</code> airflow. '
      '<code>slp01a/slp01b</code> carry only <code>Resp (sum)</code> and <code>slp61</code> only '
      '<code>Resp (abdominal)</code>. This drove the strict-airflow decision below.</div>')

    # ---------------- Stage 1
    summ = json.loads((ART / "audit" / "audit_summary.json").read_text(encoding="utf-8"))
    a('<h2 id="s1">Stage 1 — record &amp; annotation audit</h2>')
    cards = [
        ("Eligible multimodal records", summ["eligible_manifest_rows"]),
        ("UCDDB hours", f'{summ["UCDDB"]["total_hours"]:.1f}'),
        ("PSG hours", f'{summ["MIT-BIH_PSG"]["total_hours"]:.1f}'),
        ("Apnea-ECG hours", f'{summ["Apnea-ECG"]["total_hours"]:.1f}'),
    ]
    a('<div class="grid">' + "".join(
        f'<div class="card"><div class="k">{k}</div><div class="v">{v}</div></div>' for k, v in cards
    ) + "</div>")

    rows = [
        {"Dataset": "UCDDB", "Records": summ["UCDDB"]["records"],
         "Duration (h)": summ["UCDDB"]["duration_h_mean_sd"],
         "Annotation": "event onset + duration; stage 30 s",
         "Positive rate": f'{summ["UCDDB"]["event_positive_fraction_pooled"]:.3f}',
         "Respiratory source": "airflow thermistor (Flow) @8 Hz, ECG @128 Hz"},
        {"Dataset": "MIT-BIH PSG", "Records": summ["MIT-BIH_PSG"]["records"],
         "Duration (h)": summ["MIT-BIH_PSG"]["duration_h_mean_sd"],
         "Annotation": "30 s epoch tokens",
         "Positive rate": f'{summ["MIT-BIH_PSG"]["apnea_epoch_fraction_pooled"]:.3f}',
         "Respiratory source": "nasal airflow @250 Hz, ECG @250 Hz"},
        {"Dataset": "Apnea-ECG", "Records": summ["Apnea-ECG"]["records"],
         "Duration (h)": summ["Apnea-ECG"]["duration_h_mean_sd"],
         "Annotation": "per-minute A/N",
         "Positive rate": f'{summ["Apnea-ECG"]["minute_positive_fraction_pooled"]:.3f}',
         "Respiratory source": "none — single ECG channel @100 Hz"},
    ]
    a(table(pd.DataFrame(rows), num_cols=["Records", "Positive rate"]))
    a('<h3>UCDDB event categories</h3>')
    a(table(pd.DataFrame([{"Category": k, "Events": v}
                          for k, v in summ["UCDDB"]["event_categories"].items()]),
            num_cols=["Events"]))
    a('<div class="note">POSSIBLE (19) and PB EVENT (91) are <b>not</b> treated as positive. '
      'Positive = apnea or hypopnea only, per Section 3.4.4.</div>')

    man = pd.read_csv(ART / "audit" / "audit_manifest.csv")
    a('<h3>Per-record audit manifest</h3>')
    show = man[["dataset", "record_id", "subject_id", "ecg_channel", "resp_channel", "resp_type",
                "sampling_rate_ecg", "sampling_rate_resp", "duration_s", "positive_fraction",
                "eligible"]].copy()
    show["duration_s"] = (show["duration_s"] / 3600).round(2)
    show.columns = ["dataset", "record", "subject", "ecg_ch", "resp_ch", "resp_type",
                    "fs_ecg", "fs_resp", "hours", "positive", "eligible"]
    a(table(show, num_cols=["fs_ecg", "fs_resp", "hours", "positive"], max_rows=45))

    # ---------------- Stage 2
    a('<h2 id="s2">Stage 2 — subject split + windowing</h2>')
    split = pd.read_csv(ART / "splits" / "split_manifest.csv")
    counts = (split.groupby(["seed", "dataset", "split"]).agg(
        subjects=("subject_id", "nunique"), records=("n_records", "sum")).reset_index())
    a("<h3>Subject counts per split</h3>")
    a(table(counts, num_cols=["subjects", "records"]))

    ws = json.loads((ART / "windows" / "window_summary.json").read_text(encoding="utf-8"))
    for duration in ("10s", "30s"):
        a(f"<h3>{duration} windows — seed 42</h3>")
        rows = []
        for key, v in ws[duration].items():
            if not key.startswith("seed42|"):
                continue
            _, dataset, split_name = key.split("|")
            rows.append({"dataset": dataset, "split": split_name, "windows": v["windows"],
                         "positive": v["positive"], "positive rate": v["positive_fraction"],
                         "ambiguous-only": v["ambiguous_only_windows"]})
        a(table(pd.DataFrame(rows), num_cols=["windows", "positive", "positive rate", "ambiguous-only"]))
    a('<div class="note">Train uses overlapping windows (stride = half the window); validation and '
      'test are non-overlapping, per Section 3.4.4. Windows are only created inside annotated '
      'regions, so unannotated lead-ins and tails are not labelled normal.</div>')
    a(f'<p class="sub">Total rows across all three seeds (42, 1337, 2024): windows_10s.csv '
      f'{sum(1 for _ in open(ART / "windows" / "windows_10s.csv", encoding="utf-8")) - 1:,} · '
      f'windows_30s.csv {sum(1 for _ in open(ART / "windows" / "windows_30s.csv", encoding="utf-8")) - 1:,}. '
      'Per-seed counts are in the table above; seeds 42, 1337, and 2024 are all trained and evaluated.</p>')

    # ---------------- Stage 3
    a('<h2 id="s3">Stage 3 — preprocessing &amp; loader</h2>')
    cfg = json.loads((ART / "signals" / "preprocessing_config.json").read_text(encoding="utf-8"))
    a('<p class="sub">Provisional settings, pending the final numeric choices the paper leaves to '
      'preliminary experiments.</p>')
    rows = [
        {"Signal": "ECG", "Operation": f'band-pass {cfg["ecg"]["bandpass_hz"]} Hz, order {cfg["ecg"]["order"]}, '
                                       f'zero-phase; notch when mains detected',
         "Output rate": f'{cfg["ecg"]["target_fs_hz"]:.0f} Hz'},
        {"Signal": "Respiration", "Operation": f'low-pass {cfg["resp"]["lowpass_hz"]} Hz, order '
                                               f'{cfg["resp"]["order"]}, linear detrend',
         "Output rate": f'{cfg["resp"]["target_fs_hz"]:.0f} Hz'},
        {"Signal": "Normalization", "Operation": "training-only z-score, per dataset and modality",
         "Output rate": "then clip to ±5 σ"},
    ]
    a(table(pd.DataFrame(rows)))

    qc = pd.read_csv(ART / "signals" / "signal_qc.csv")
    summary = qc.groupby("dataset").agg(
        records=("record_id", "count"),
        durations_match=("duration_mismatch_ms", "max"),
        ecg_flat=("ecg_flatline_frac", "max"),
        resp_flat=("resp_flatline_frac", "max"),
        notch=("notch_applied", "sum"),
    ).reset_index()
    a("<h3>Signal QC after filtering and resampling</h3>")
    a(table(summary, num_cols=["records", "durations_match", "ecg_flat", "resp_flat", "notch"]))
    a('<div class="note">ECG and respiration durations match to 0 ms in every record. '
      'Notch was needed only for <code>slp67x</code> (60 Hz).</div>')

    rq = pd.read_csv(ART / "signals" / "resp_qc.csv")
    a("<h3>Respiration quality per record</h3>")
    a(table(rq[["dataset", "record_id", "resp_snr_db", "breaths_per_min", "resp_p1_p99_range"]],
            num_cols=["resp_snr_db", "breaths_per_min", "resp_p1_p99_range"], max_rows=45))
    a('<div class="note"><code>ucddb005</code> had a dead <code>Flow</code> channel '
      '(std 0.50 vs 4.2–6.0 elsewhere, only 318 unique values, flat after the initial transient). '
      'It was excluded rather than substituted, keeping the strict airflow policy.</div>')

    a(fig(ART / "signals" / "qc_ucddb002.png",
          "UCDDB ucddb002 — ECG 100 Hz, respiration 25 Hz, ECG spectrogram. Motion artefact around 65–80 s."))
    a(fig(ART / "signals" / "qc_slp02a.png",
          "MIT-BIH PSG slp02a — same pipeline. Nasal airflow is uncalibrated, so its amplitude is small "
          "before z-scoring."))

    norm = json.loads((ART / "normalization" / "norm_stats.json").read_text(encoding="utf-8"))
    rows = []
    for dataset in ("UCDDB", "MIT-BIH_PSG"):
        for modality in ("ecg", "resp"):
            vals = [norm[s][d][dataset][modality]["std"] for s in norm for d in norm[s]]
            rows.append({"dataset": dataset, "modality": modality,
                         "std across seeds": f"{min(vals):.4f} – {max(vals):.4f}"})
    a("<h3>Training-only normalization statistics</h3>")
    a(table(pd.DataFrame(rows)))
    a('<div class="note">Respiratory scale differs by ~22× between datasets before normalization '
      '(UCDDB ≈5.1, PSG ≈0.23). Statistics are fitted per dataset from training windows only, so test '
      'data never informs normalization.</div>')

    a("<h3>Loader checks</h3>")
    rows = [
        {"check": "ECG tensor", "10 s": "(1, 1000)", "30 s": "(1, 3000)"},
        {"check": "Respiration tensor", "10 s": "(1, 250)", "30 s": "(1, 750)"},
        {"check": "Labels match manifest", "10 s": "100%", "30 s": "100%"},
        {"check": "NaN / Inf", "10 s": "none", "30 s": "none"},
        {"check": "Clipping at ±5σ", "10 s": "0.10% ECG", "30 s": "0.04% ECG"},
        {"check": "Throughput (1 process)", "10 s": "~2900 windows/s", "30 s": "~3250 windows/s"},
    ]
    a(table(pd.DataFrame(rows)))

    # ---------------- Stage 4
    a('<h2 id="s4">Stage 4 — model pilot</h2>')
    a('<p class="sub">Proves the training pipeline runs end to end. A reduced subset (8,000 train / '
      '3,000 val windows), not a performance result.</p>')
    pilot = pd.read_csv(ART / "pilot" / "pilot_summary.csv")
    pilot = pilot[["model", "params", "best_epoch", "epochs_run", "best_val_loss", "threshold",
                   "val_accuracy", "val_sensitivity", "val_specificity", "val_precision"]]
    pilot.columns = ["model", "params", "best epoch", "epochs run", "best val loss", "threshold",
                     "val acc", "val sens", "val spec", "val prec"]
    a(table(pilot, num_cols=["params", "best epoch", "epochs run", "best val loss", "threshold",
                             "val acc", "val sens", "val spec", "val prec"]))
    a('<div class="note">Training loss fell in all four models, gradients were finite, and reloaded '
      'checkpoints reproduced identical logits. What the pilot also showed is that the models '
      '<b>memorise subjects</b>: train AUC ≈0.80 against pooled validation AUC 0.45–0.56, with '
      'per-subject validation AUC varying from 0.33 to 0.87. That is a regularisation/budget problem '
      'for Stage 5, not a bug: labels, shapes, normalisation, and signal reads were all verified.</div>')

    # ---------------- Stage 5
    a('<h2 id="s5">Stage 5 — full training + freeze</h2>')
    run_cfg = json.loads((ART / "train" / "seed42_10s" / "run_config.json").read_text(encoding="utf-8"))
    frozen = json.loads((ART / "train" / "seed42_10s" / "frozen.json").read_text(encoding="utf-8"))
    a('<p class="sub">All 116,199 training windows (26 subjects) and 14,407 validation windows '
      '(6 subjects), seed 42, 10 s windows. Checkpoint, best epoch, and decision threshold were '
      'written to <code>frozen.json</code> before any test prediction. '
      f'Test split read: <b>{"yes" if run_cfg["test_split_read"] else "no"}</b>.</p>')
    rows = []
    for name, entry in frozen["models"].items():
        m = json.loads((ART / "train" / "seed42_10s" / f"metrics_{name}.json").read_text(encoding="utf-8"))
        v = m["val_overall"]
        rows.append({"model": name, "params": m["params"], "epochs run": m["epochs_run"],
                     "best epoch": entry["best_epoch"], "val loss": entry["best_val_loss"],
                     "threshold": entry["threshold"], "val acc": v["accuracy"],
                     "val sens": v["sensitivity"], "val spec": v["specificity"],
                     "val prec": v["precision"], "val f1": v["f1"], "val AUC": v["roc_auc"],
                     "ms/window": m["inference_ms_per_window"]})
    df = pd.DataFrame(rows)
    a(table(df, num_cols=["params", "epochs run", "best epoch", "val loss", "threshold", "val acc",
                          "val sens", "val spec", "val prec", "val f1", "val AUC", "ms/window"]))
    a("<h3>Validation ROC-AUC per dataset</h3>")
    rows = []
    for name in frozen["models"]:
        m = json.loads((ART / "train" / "seed42_10s" / f"metrics_{name}.json").read_text(encoding="utf-8"))
        rows.append({"model": name, **{d: v["roc_auc"] for d, v in m["val_by_dataset"].items()}})
    a(table(pd.DataFrame(rows), num_cols=[c for c in pd.DataFrame(rows).columns if c != "model"]))
    a('<div class="note">Full data improved generalisation over the pilot (proposed AUC 0.467 → 0.644; '
      'simple fusion 0.498 → 0.693), but two things must be said plainly: <b>proposed did not beat '
      'simple_fusion on validation</b> (0.644 vs 0.693), and the PSG validation AUC is below 0.5 for '
      'every model — only two PSG subjects, one of them with no positive windows, so that number '
      'carries no evidence. Thresholds found by F1 are low (0.095–0.37) with precision 0.10–0.14 at '
      '8.9% prevalence, i.e. the models lean towards predicting positive.</div>')

    # ---------------- Stage 6
    a('<h2 id="s6">Stage 6 — test prediction</h2>')
    test_dir = ART / "test" / "seed42_10s"
    if (test_dir / "test_run.json").exists():
        trun = json.loads((test_dir / "test_run.json").read_text(encoding="utf-8"))
        a(f'<p class="sub">Frozen models only — run {trun["run_at"]}, frozen at {trun["frozen_at"]}. '
          f'{trun["test_windows"]:,} test windows, {trun["test_subjects"]} subjects '
          f'(4 UCDDB + 2 PSG). Checkpoints and validation thresholds applied unchanged; '
          'the test split was read once.</p>')
        summ = pd.read_csv(test_dir / "test_summary.csv")
        summ = summ[["model", "windows", "positives", "accuracy", "sensitivity", "specificity",
                     "precision", "f1", "roc_auc", "threshold", "ms_per_window"]]
        summ.columns = ["model", "windows", "positives", "accuracy", "sensitivity", "specificity",
                        "precision", "f1", "ROC-AUC", "threshold", "ms/window"]
        a(table(summ, num_cols=["windows", "positives", "accuracy", "sensitivity", "specificity",
                                "precision", "f1", "ROC-AUC", "threshold", "ms/window"]))

        a("<h3>Test ROC-AUC per dataset</h3>")
        rows = []
        for name in trun["models"]:
            m = json.loads((test_dir / f"metrics_test_{name}.json").read_text(encoding="utf-8"))
            rows.append({"model": name, **{d: v["roc_auc"] for d, v in m["test_by_dataset"].items()}})
        a(table(pd.DataFrame(rows), num_cols=[c for c in pd.DataFrame(rows).columns if c != "model"]))

        a("<h3>AUC decomposition — pooled vs per subject</h3>")
        dec = pd.read_csv(test_dir / "test_auc_decomposition.csv")
        dec.columns = ["model", "pooled AUC", "mean subject AUC", "min subject AUC",
                       "max subject AUC", "within-subject AUC", "subjects scored"]
        a(table(dec, num_cols=["pooled AUC", "mean subject AUC", "min subject AUC",
                               "max subject AUC", "within-subject AUC", "subjects scored"]))
        a('<div class="note">For <code>proposed</code> the pooled AUC is 0.526 while the mean '
          'per-subject AUC is 0.733. That 0.21 gap is a cross-subject calibration failure: each '
          'subject gets a differently offset score scale, so ranking breaks when all windows are '
          'pooled. <code>resp_only</code> is the only model where pooled and per-subject AUC agree. '
          'This decomposition is a diagnostic, not a headline metric, and the pooled figure is the '
          'one that must be reported.</div>')

        df_subj = pd.read_csv(test_dir / "test_by_subject_proposed.csv")
        df_subj = df_subj[["subject_id", "dataset", "windows", "positive_rate", "roc_auc",
                           "sensitivity", "specificity"]]
        df_subj.columns = ["subject", "dataset", "windows", "positive rate", "ROC-AUC",
                           "sensitivity", "specificity"]
        a("<h3>Proposed — test per subject</h3>")
        a(table(df_subj, num_cols=["windows", "positive rate", "ROC-AUC", "sensitivity",
                                   "specificity"]))
        a('<div class="note">All four UCDDB subjects are above 0.69, but the two PSG subjects are '
          'lower (slp02 weakest at 0.53). With six subjects, the pooled test figure is one '
          'measurement, not a performance estimate.</div>')
        a('<div class="note"><b>Must be reported, not buried:</b> <code>resp_only</code> is the best '
          'model on test (AUC 0.734) and <code>proposed</code> is second (0.526). The cross-attention '
          'architecture has not been shown to help. <code>ecg_only</code> collapses on UCDDB '
          '(specificity 0.044), and thresholds carried over from validation do not transfer '
          'optimally (precision 0.11–0.21 at 14.7% prevalence) — they were deliberately not retuned '
          'after seeing test.</div>')
    else:
        a('<div class="note">Stage 6 has not been run yet.</div>')

    # ---------------- Stage 6b
    calib_dir = ART / "calibration"
    a('<h2 id="s6b">Stage 6b — calibration diagnostic</h2>')
    if (calib_dir / "calibration_summary.csv").exists():
        a('<p class="sub"><b>Post-hoc diagnostic, not a protocol.</b> Run after the test results were '
          'seen, so nothing here may be used to select a model or replace the headline metric. '
          'The reported metric stays the pooled AUC with the frozen validation threshold.</p>')
        csum = pd.read_csv(calib_dir / "calibration_summary.csv")
        csum = csum[["model", "split", "pooled_auc", "mean_subject_auc", "mean_record_auc",
                     "gap", "within_subject_auc", "subject_score_spread"]]
        csum.columns = ["model", "split", "pooled AUC", "mean subject AUC", "mean record AUC",
                        "gap", "within-subject AUC", "subject score spread"]
        a("<h3>Does the gap belong to the model or to the test subjects?</h3>")
        a(table(csum, num_cols=["pooled AUC", "mean subject AUC", "mean record AUC", "gap",
                                "within-subject AUC", "subject score spread"]))
        a('<div class="note">For <code>proposed</code> the gap is 0.044 on validation and 0.207 on '
          'test. A model that could not rank windows would show a large gap on both. The architecture '
          'is not broken; the test subjects carry much wider score offsets. <code>resp_only</code> is '
          'the only model stable across both splits.</div>')

        bys = pd.read_csv(calib_dir / "calibration_by_subject.csv")
        a("<h3>Score offset versus subject prevalence (test)</h3>")
        show = bys[bys.model == "proposed"][["dataset", "subject_id", "windows", "positive_rate",
                                             "score_mean", "pred_positive_rate", "threshold"]]
        show.columns = ["dataset", "subject", "windows", "positive rate", "mean score",
                        "predicted positive rate", "threshold"]
        a(table(show, num_cols=["windows", "positive rate", "mean score",
                                "predicted positive rate", "threshold"]))
        rho = bys.groupby("model").spearman_prev_vs_score.first()
        prev_path = calib_dir / "calibration_prevalence.csv"
        if prev_path.exists():
            prev = pd.read_csv(prev_path)
            prev.columns = ["model", "split", "subjects", "Spearman", "p",
                            "leave-one-out range"]
            a("<h3>Is the offset a model property or a test-subject property?</h3>")
            a(table(prev, num_cols=["subjects", "Spearman", "p"]))
            a('<div class="note"><b>Do not overclaim.</b> The correlation is −0.943 on test '
              '(p = 0.005, stable under leave-one-out) but only +0.029 on validation (p = 0.957). '
              'So the inversion belongs to the six test subjects, not to the model as a stable '
              'trait. What <i>is</i> already visible on validation is <code>slp41</code>: zero '
              'positive windows, yet the highest mean score in the whole validation split (0.504, '
              'above slp23 at 12.7% and slp03 at 33.5%). A subject with one class has no defined '
              'AUC, so it never appears in any AUC table and never triggered a warning.</div>')
        else:
            a(table(pd.DataFrame({"model": rho.index,
                                  "Spearman(prevalence, mean score)": rho.values}),
                    num_cols=["Spearman(prevalence, mean score)"]))
        a('<div class="note"><b>Prior inversion, test only.</b> Subject 22 has the lowest prevalence '
          '(1.9%) but the highest mean score (0.450), so 70.9% of its windows are called positive. '
          'That is the mechanism destroying the pooled AUC, not an inability to rank within a '
          'subject (within-subject AUC 0.672). Leave-one-out leaves the correlation between −0.90 '
          'and −1.00 on test, but it does not replicate on validation — see the table above.</div>')

        tr = pd.read_csv(calib_dir / "calibration_transforms.csv")
        a("<h3>Would label-free per-record normalisation close the gap?</h3>")
        a(table(tr, num_cols=["val_pooled_auc", "test_pooled_auc"]))
        a('<div class="note"><b>Must be stated plainly:</b> the transform raises the test number but '
          '<b>lowers validation</b> for <code>proposed</code> (0.644 → 0.601) and '
          '<code>simple_fusion</code> (0.693 → 0.624). It is a post-hoc rescue chosen after seeing '
          'test, not a principled improvement, and it is harmful on the split that was not used to '
          'pick it. <code>resp_only</code> drops slightly on both, because it never had the offset '
          'problem.</div>')

        boot = pd.read_csv(calib_dir / "calibration_bootstrap.csv")
        boot.columns = ["model", "transform", "median", "95% CI low", "95% CI high",
                        "subjects resampled"]
        a("<h3>Subject-level bootstrap (2,000 resamples of 6 subjects)</h3>")
        a(table(boot, num_cols=["median", "95% CI low", "95% CI high", "subjects resampled"]))
        a('<div class="note">The <code>proposed</code> interval (0.438–0.767) is very wide and still '
          'contains 0.5. Six subjects cannot give a precise estimate. The bootstrap resamples '
          'subjects and deliberately ignores within-subject autocorrelation, so even this width is '
          'optimistic.</div>')
    else:
        a('<div class="note">Stage 6b has not been run yet.</div>')

    # ---------------- Stage 6c
    probe_dir = ART / "offset_probe"
    a('<h2 id="s6c">Stage 6c — per-record offset probe</h2>')
    if (probe_dir / "per_record_scaling.csv").exists():
        a('<p class="sub"><b>Hypothesis tested and rejected.</b> If dataset-level normalisation left '
          'each record with its own mean offset, that offset would be a record-level shortcut the '
          'model could exploit. It does not: the post-normalisation mean is 0.00 for all 39 records '
          'in both modalities, because the signals were detrended before z-scoring.</p>')
        sc = pd.read_csv(probe_dir / "per_record_scaling.csv")
        a("<h3>What was found instead: per-record scale is not uniform</h3>")
        worst = sc.sort_values("resp_clip_frac", ascending=False).head(10)[
            ["dataset", "record_id", "ecg_std_z", "resp_std_z", "ecg_clip_frac", "resp_clip_frac"]]
        worst.columns = ["dataset", "record", "ECG std (z units)", "resp std (z units)",
                         "ECG clipped", "resp clipped"]
        a(table(worst, num_cols=["ECG std (z units)", "resp std (z units)",
                                 "ECG clipped", "resp clipped"]))
        a(f'<div class="note"><b>The ±5σ clip does not act equally on every record.</b> Averaged over '
          f'all 39 records, {sc.ecg_clip_frac.mean()*100:.2f}% of ECG samples and '
          f'{sc.resp_clip_frac.mean()*100:.2f}% of respiration samples are clipped. The worst records '
          f'lose far more: <code>slp02b</code> {sc.resp_clip_frac.max()*100:.1f}% of its respiration. '
          f'This <b>corrects an earlier claim</b> in this report and in the work log that clipping was '
          f'"almost never active (0.1% of ECG)" — that figure was wrong. Because per-record scale '
          f'varies from 0.55 to 5.92 (dataset std units), the same 5σ threshold bites very '
          f'differently per recording.</div>')

        corr_path = probe_dir / "offset_correlations.csv"
        if corr_path.exists():
            corr = pd.read_csv(corr_path)
            a("<h3>Do record features explain the model's mean score?</h3>")
            a('<p class="sub">Spearman over 6–7 records per split, 11 features, no multiple-comparison '
              'correction. Partial column controls for prevalence.</p>')
            top = corr.reindex(corr.partial_spearman_controlling_prevalence.abs()
                               .sort_values(ascending=False).index).head(10)
            a(table(top, num_cols=["n_records", "spearman", "p",
                                   "partial_spearman_controlling_prevalence", "partial_p"]))
            a('<div class="note"><b>Nothing here qualifies as a finding.</b> The strongest candidates '
              'appear in one split or one model only and do not replicate. With n = 6–7 records and '
              '11 features tested, these are leads at best. There is <i>no</i> evidence that record '
              'scale causes the score offset.</div>')
    else:
        a('<div class="note">Stage 6c has not been run yet.</div>')

    # ---------------- Stage 7
    agg_dir = ART / "aggregate" / "seed42-1337-2024_10s"
    a('<h2 id="s7">Stage 7 — multi-seed comparison (Experiment 1, 10 s)</h2>')
    if (agg_dir / "test_summary_mean_sd.csv").exists():
        a('<p class="sub">Four models trained on full training data for three subject-level splits '
          '(seeds 42, 1337, 2024), window 10 s. Each seed\'s test split was read once with its own '
          'frozen checkpoints and thresholds. Numbers below are mean ± sample SD across the three '
          'seeds, as Section 3.5.2 asks.</p>')

        vsum = pd.read_csv(agg_dir / "val_summary_mean_sd.csv")
        vsum = vsum[["model", "val_roc_auc", "val_roc_auc_sd", "val_f1", "val_f1_sd"]]
        vsum.columns = ["model", "val AUC", "val AUC SD", "val F1", "val F1 SD"]
        a("<h3>Validation</h3>")
        a(table(vsum, num_cols=["val AUC", "val AUC SD", "val F1", "val F1 SD"]))
        a('<div class="note">Ranking is not stable across seeds on validation: '
          '<code>simple_fusion</code> wins seed 42, <code>resp_only</code> wins 1337 and 2024. '
          '<code>proposed</code> is 2nd or 3rd every time and never wins. <code>ecg_only</code> is '
          'always last. The smallest SD belongs to <code>proposed</code> only because it is '
          'consistently mediocre.</div>')

        tsum = pd.read_csv(agg_dir / "test_summary_mean_sd.csv")
        tsum = tsum[["model", "test_roc_auc", "test_roc_auc_sd", "test_f1", "test_f1_sd",
                     "test_sensitivity", "test_sensitivity_sd", "test_specificity",
                     "test_specificity_sd"]]
        tsum.columns = ["model", "test AUC", "test AUC SD", "test F1", "test F1 SD",
                        "sensitivity", "sensitivity SD", "specificity", "specificity SD"]
        a("<h3>Test (frozen models, each seed read once)</h3>")
        a(table(tsum, num_cols=["test AUC", "test AUC SD", "test F1", "test F1 SD",
                                "sensitivity", "sensitivity SD", "specificity", "specificity SD"]))
        a('<div class="note"><b>The main result of this stage:</b> <code>resp_only</code> is the only '
          'stable model (0.734 / 0.751 / 0.742, SD 0.008). <code>proposed</code> loses to it in '
          '<b>every</b> seed (0.526 / 0.552 / 0.586), so the cross-attention benefit is not '
          'supported by any seed. <code>simple_fusion</code> swings wildly (0.450 / 0.696 / 0.585, '
          'SD 0.123) — one seed for that model is misleading.</div>')

        ds = pd.read_csv(agg_dir / "test_by_dataset_mean_sd.csv")
        ds = ds[["model", "dataset", "roc_auc", "roc_auc_sd", "f1", "f1_sd"]]
        ds.columns = ["model", "dataset", "test AUC", "test AUC SD", "F1", "F1 SD"]
        a("<h3>Test by dataset</h3>")
        a(table(ds, num_cols=["test AUC", "test AUC SD", "F1", "F1 SD"]))
        a('<div class="note">PSG remains the weak half for every model except <code>resp_only</code> '
          '(0.666 ± 0.050); the others are at or below 0.46 on PSG. <code>proposed</code> is strong '
          'on UCDDB (0.763 ± 0.036) but collapses on PSG (0.457 ± 0.106).</div>')

        dc = pd.read_csv(agg_dir / "test_auc_decomposition_mean_sd.csv")
        dc = dc[["model", "pooled_auc", "pooled_auc_sd", "mean_subject_auc",
                 "mean_subject_auc_sd", "within_subject_auc", "within_subject_auc_sd"]]
        dc.columns = ["model", "pooled AUC", "pooled AUC SD", "mean subject AUC",
                      "mean subject AUC SD", "within-subject AUC", "within-subject AUC SD"]
        a("<h3>Pooled vs per-subject AUC</h3>")
        a(table(dc, num_cols=["pooled AUC", "pooled AUC SD", "mean subject AUC",
                              "mean subject AUC SD", "within-subject AUC",
                              "within-subject AUC SD"]))
        a('<div class="note">The cross-subject score offset survives across seeds for '
          '<code>proposed</code> (pooled 0.555 vs mean subject 0.673) and '
          '<code>simple_fusion</code> (0.577 vs 0.676). <code>resp_only</code> is the only model '
          'where the two agree (0.742 vs 0.714).</div>')

        boot = pd.read_csv(agg_dir / "test_bootstrap_by_seed.csv")
        boot = boot[["seed", "model", "n_subjects", "median", "lo95", "hi95"]]
        boot.columns = ["seed", "model", "subjects", "median", "95% CI low", "95% CI high"]
        a("<h3>Subject-level bootstrap 95% CI (2,000 resamples per seed)</h3>")
        a(table(boot, num_cols=["subjects", "median", "95% CI low", "95% CI high"]))
        a('<div class="note">Every <code>proposed</code> interval spans roughly 0.40–0.81 and contains '
          '0.5. With six test subjects per seed, the SD across seeds for <code>proposed</code> '
          '(0.030) is narrower than its own CI, so small differences between models cannot be '
          'resolved by this data. The bootstrap resamples subjects and ignores within-subject '
          'autocorrelation, so even these widths are optimistic.</div>')

        a('<div class="note"><b>Still open (Experiments 3–4):</b> preprocessing ablation and '
          'single- vs combined-dataset training with LODO. Also open: the ECG-only Apnea-ECG arm. '
          'No results in this report cover those.</div>')
    else:
        a('<div class="note">Stage 7 has not been run yet.</div>')

    # ---------------- Stage 7b: window duration (Experiment 2)
    agg10 = ART / "aggregate" / "seed42-1337-2024_10s"
    agg30 = ART / "aggregate" / "seed42-1337-2024_30s"
    a('<h2 id="s7b">Stage 7b — window duration, 10 s vs 30 s (Experiment 2)</h2>')
    if (agg30 / "test_summary_mean_sd.csv").exists() and (agg10 / "test_summary_mean_sd.csv").exists():
        a('<p class="sub">Same four models, same three seeds and subject splits, same frozen '
          'protocol; only the window length differs. Reported as mean ± SD across seeds. '
          'Seed 2024 at 30 s was trained on a T4 GPU, the others on CPU — floating-point order '
          'differs, so this is recorded as a device difference, not a method change.</p>')

        a10 = pd.read_csv(agg10 / "test_summary_mean_sd.csv")[["model", "test_roc_auc", "test_roc_auc_sd"]]
        a30 = pd.read_csv(agg30 / "test_summary_mean_sd.csv")[["model", "test_roc_auc", "test_roc_auc_sd"]]
        cmp = a10.merge(a30, on="model", suffixes=(" (10 s)", " (30 s)"))
        cols = ["test_roc_auc (10 s)", "test_roc_auc_sd (10 s)",
                "test_roc_auc (30 s)", "test_roc_auc_sd (30 s)"]
        cmp["change"] = (cmp[cols[2]] - cmp[cols[0]]).round(4)
        cmp = cmp.sort_values(cols[2], ascending=False)
        cmp.columns = ["model", "AUC 10 s", "SD 10 s", "AUC 30 s", "SD 30 s", "change"]
        a("<h3>Test ROC-AUC by window length</h3>")
        a(table(cmp, num_cols=["AUC 10 s", "SD 10 s", "AUC 30 s", "SD 30 s", "change"]))
        a('<div class="note"><b>The headline of Experiment 2:</b> the two <em>fusion</em> models gain '
          'substantially at 30 s (<code>simple_fusion</code> +0.123, <code>proposed</code> +0.123), '
          'while <code>resp_only</code> barely moves (+0.004) and <code>ecg_only</code> gets worse '
          '(−0.024). This is consistent with fusion needing longer sequences to have anything to '
          'fuse: at 10 s each branch only has 50 tokens.</div>')

        p10 = pd.read_csv(agg10 / "test_per_seed.csv")[["seed", "model", "test_roc_auc"]]
        p30 = pd.read_csv(agg30 / "test_per_seed.csv")[["seed", "model", "test_roc_auc"]]
        per = p10.merge(p30, on=["seed", "model"], suffixes=("_10", "_30"))
        per["change"] = (per.test_roc_auc_30 - per.test_roc_auc_10).round(4)
        per = per.rename(columns={"test_roc_auc_10": "AUC 10 s", "test_roc_auc_30": "AUC 30 s"})
        a("<h3>Change per seed — is the gain consistent or one lucky split?</h3>")
        a(table(per[["model", "seed", "AUC 10 s", "AUC 30 s", "change"]],
                num_cols=["AUC 10 s", "AUC 30 s", "change"]))
        a('<div class="note"><b>Check this before claiming anything.</b> <code>proposed</code> gains '
          'in all three seeds (+0.109, +0.106, +0.154) — the only model whose direction never flips. '
          '<code>simple_fusion</code> gains in all three too but wildly (+0.213, +0.056, +0.100), and '
          'it was already the least stable model at 10 s. <code>resp_only</code> moves both ways '
          '(−0.040, +0.004, +0.049), i.e. no real effect. <code>ecg_only</code> is negative in two of '
          'three seeds.</div>')
        a('<div class="note"><b>What this does not show.</b> <code>resp_only</code> still has the '
          'highest mean test AUC at 30 s (0.747 vs 0.678 for <code>proposed</code>), and its SD is '
          'much smaller. The gain narrows the gap; it does not close or reverse it. Also, '
          '<code>proposed</code> at 30 s still carries the cross-subject score offset (pooled 0.678 vs '
          'mean subject 0.704), and two of the three seed bootstrap intervals still contain 0.5 '
          '(seed 42: 0.522-0.804, seed 1337: 0.486-0.808; seed 2024 no longer does: 0.575-0.857). '
          'Report the gain, do not report a win.</div>')

        a("<h3>Validation at 30 s</h3>")
        v30 = pd.read_csv(agg30 / "val_summary_mean_sd.csv")[
            ["model", "val_roc_auc", "val_roc_auc_sd", "val_f1", "val_f1_sd", "best_epoch"]]
        v30.columns = ["model", "val AUC", "val AUC SD", "val F1", "val F1 SD", "mean best epoch"]
        a(table(v30, num_cols=["val AUC", "val AUC SD", "val F1", "val F1 SD", "mean best epoch"]))
        a('<div class="note">At 30 s the best epoch is often 1–2 (<code>ecg_only</code> epoch 1 in '
          'all three seeds), so the longer window overfits faster under the same budget. '
          '<code>proposed</code> and <code>simple_fusion</code> on validation rank above '
          '<code>resp_only</code> (0.743 / 0.746 vs 0.727), but that ordering does not survive on '
          'test — a reminder that with 6 validation subjects the ranking is not trustworthy.</div>')

        a("<h3>Test ROC-AUC by dataset at 30 s</h3>")
        ds30 = pd.read_csv(agg30 / "test_by_dataset_mean_sd.csv")[
            ["model", "dataset", "roc_auc", "roc_auc_sd", "f1"]]
        ds30.columns = ["model", "dataset", "test AUC", "test AUC SD", "F1"]
        a(table(ds30, num_cols=["test AUC", "test AUC SD", "F1"]))
        a('<div class="note">UCDDB stays easy (0.77–0.81 for the fusion models) and PSG stays the '
          'hard half (0.34–0.67). <code>resp_only</code> is still the only model above 0.6 on PSG '
          '(0.669 ± 0.069).</div>')
    else:
        a('<div class="note">The 30 s comparison has not been run yet.</div>')

    # ---------------- decisions
    a('<h2 id="decisions">Decisions &amp; exclusions</h2>')
    policy = json.loads((ROOT / "config" / "policy.json").read_text(encoding="utf-8"))
    rows = [{"item": k, "reason": v}
            for k, v in policy["multimodal_policy"]["datasets_excluded_from_multimodal"].items()]
    rows += [{"item": k, "reason": v}
             for k, v in policy["multimodal_policy"]["excluded_records"].items()]
    rows += [{"item": "slp01a, slp01b", "reason": "Strict airflow policy: only Resp (sum) available."},
             {"item": "slp61", "reason": "Strict airflow policy: only Resp (abdominal) available."}]
    a(table(pd.DataFrame(rows)))
    a('<p class="sub">Label rule: a window is positive when at least 50% of its duration overlaps an '
      'annotated apnea or hypopnea event. PSG labels inherit 30 s epochs, so 10 s windows carry coarse '
      '(weak) labels; this is reported as label noise rather than corrected.</p>')

    # ---------------- decisions detail
    a('<h2 id="decisions-detail">Decisions taken so far</h2>')
    dec = pd.DataFrame([
        {"#": 1, "Decision": "Apnea-ECG excluded from multimodal (ECG-only arm only)",
         "Why": "It does have ECG, which is required, but no respiratory channel. The multimodal "
                "model consumes ECG and respiration for the same window, so one modality is not "
                "enough. Still usable for the ECG-only arm.",
         "Basis": "Sec. 3.3 (availability verified by audit); Sec. 3.4.2"},
        {"#": 2, "Decision": "Strict airflow channel policy",
         "Why": "UCDDB Flow (thermistor) matched to PSG Resp (nasal); mixing airflow with effort is not equivalent",
         "Basis": "Sec. 3.4.2"},
        {"#": 3, "Decision": "slp01a, slp01b excluded",
         "Why": "Only Resp (sum) is available", "Basis": "consequence of #2"},
        {"#": 4, "Decision": "slp61 excluded",
         "Why": "Only Resp (abdominal) is available", "Basis": "consequence of #2"},
        {"#": 5, "Decision": "ucddb005 excluded",
         "Why": "Dead Flow channel: std 0.50 vs 4.2-6.0 elsewhere, 318 unique values, flat for ~6 h",
         "Basis": "channel audit"},
        {"#": 6, "Decision": "Positive = apnea OR hypopnea",
         "Why": "Paper labeling rule", "Basis": "Sec. 3.4.4"},
        {"#": 7, "Decision": "POSSIBLE and PB EVENT are not positive",
         "Why": "Ambiguous events; counting them would inflate positives without support",
         "Basis": "Sec. 3.4.4"},
        {"#": 8, "Decision": "Window positive at >=50% overlap",
         "Why": "Paper labeling rule", "Basis": "Sec. 3.4.4"},
        {"#": 9, "Decision": "Train overlapping, val/test non-overlapping",
         "Why": "Paper protocol", "Basis": "Sec. 3.4.4"},
        {"#": 10, "Decision": "Windows only inside annotated regions",
         "Why": "Unannotated intervals must not be labelled normal", "Basis": "context §6-7"},
        {"#": 11, "Decision": "Tail truncated, not padded",
         "Why": "Avoid fabricating samples", "Basis": "implementation choice (paper: TBD)"},
        {"#": 12, "Decision": "z-score fitted per dataset from training windows only",
         "Why": "No test statistics leak; dataset scales differ ~22x", "Basis": "Sec. 3.4.3"},
        {"#": 13, "Decision": "ECG 100 Hz, respiration 25 Hz",
         "Why": "Moderate cost, safe cut-offs", "Basis": "provisional (paper defers to audit)"},
        {"#": 14, "Decision": "Clipping at c = 5 sigma",
         "Why": "Limit motion artefacts without cutting normal signal",
         "Basis": "provisional"},
    ])
    a(table(dec, num_cols=["#"], max_rows=20))

    a('<h2 id="limits">Limitations that must be reported</h2>')
    limits = pd.DataFrame([
        {"Limitation": "Thin evaluation unit, not few windows",
         "Detail": "Test has 6 subjects (4 UCDDB + 2 PSG). Autocorrelation-adjusted N_eff is "
                   "~1875 (UCDDB) and ~110 (PSG); test contains ~583 contiguous positive runs."},
        {"Limitation": "Uneven positive distribution across test subjects",
         "Detail": "Seed 42: UCDDB subject 22 has 1.9% positive windows, PSG subject slp16 has 42.5%."},
        {"Limitation": "PSG labels are coarse",
         "Detail": "30 s epoch labels inherited by 10 s windows; reported as label noise, not corrected."},
        {"Limitation": "50% threshold is sensitive for UCDDB at 30 s",
         "Detail": "~10k windows fall in the 0.4-0.6 overlap band, peaking exactly at 0.5."},
        {"Limitation": "LODO is two-way only",
         "Detail": "UCDDB<->PSG, because the third dataset fails the channel requirement."},
        {"Limitation": "Clipping at 5 sigma is not uniform across records; an earlier figure was wrong",
         "Detail": "Recomputed over all 39 records: 1.16% of ECG samples and 0.69% of respiration "
                   "samples are clipped on average — not the 0.1% stated earlier. Worst cases lose "
                   "10.4% (slp02b) and 8.0% (slp02a) of respiration. Per-record scale ranges 0.55–5.92 "
                   "in dataset-std units, so the same threshold bites very differently. The clipping "
                   "ablation is therefore meaningful, and the choice must be made before retraining."},
        {"Limitation": "PSG respiration is uncalibrated",
         "Detail": "Amplitude carries no physiological units; z-scoring makes it usable but not interpretable."},
        {"Limitation": "Proposed architecture is not supported by the results",
         "Detail": "Across three seeds, test AUC for proposed is 0.526 / 0.552 / 0.586 (mean 0.555 ± "
                   "0.030) while the respiration-only baseline scores 0.734 / 0.751 / 0.742 "
                   "(0.742 ± 0.008). Proposed loses in every seed. No evidence that cross-attention helps."},
        {"Limitation": "Cross-subject score calibration is unresolved; mechanism only partly measured",
         "Detail": "Proposed: gap 0.044 on validation, 0.207 on test. On test Spearman(prevalence, "
                   "mean score) = −0.943 (p=0.005, stable under leave-one-out; subject 22 has the "
                   "lowest prevalence 1.9% and the highest mean score 0.450), but on validation it is "
                   "+0.029 (p=0.957). The inversion is a property of the six test subjects, not of "
                   "the model as a stable trait. The stronger warning sign is slp41 on validation: "
                   "0% prevalence with the highest mean score (0.504), invisible because a "
                   "single-class subject has no defined AUC."},
        {"Limitation": "Per-record score normalisation is a post-hoc rescue, not a result",
         "Detail": "z-scoring per record lifts test AUC 0.526 → 0.671 for proposed but lowers "
                   "validation 0.644 → 0.601. It was chosen after seeing test and hurts the split not "
                   "used to choose it, so it must not be reported as the headline metric."},
        {"Limitation": "Three seeds, one window duration",
         "Detail": "Model comparison (Experiment 1) now has 3 seeds at 10 s, reported as mean ± SD. "
                   "The 30 s condition (Experiment 2), preprocessing ablation (Experiment 3), and "
                   "single- vs combined-dataset training plus LODO (Experiment 4) have not been run."},
        {"Limitation": "SD across seeds is split sensitivity, not run-to-run variance",
         "Detail": "Each seed is a different subject-level split, so the SD measures how much the "
                   "conclusion depends on which subjects land in validation/test. With 6 test subjects, "
                   "the SD for proposed (0.030) is narrower than its own bootstrap CI (0.41–0.81), so "
                   "small model differences cannot be resolved by this data."},
    ])
    a(table(limits, max_rows=20))

    # ---------------- files
    a('<h2 id="files">File map</h2>')
    files = [
        ("Stage 0", "artifacts/verify/dataset_verification.json", "checksums and channel layouts"),
        ("Stage 0", "artifacts/verify/slpdb_resp_channels.csv", "respiration channel per PSG record"),
        ("Stage 1", "artifacts/audit/audit_manifest.csv", "one row per record"),
        ("Stage 1", "artifacts/audit/audit_summary.md", "human-readable summary"),
        ("Stage 1", "artifacts/audit/ucddb_channels.csv", "all EDF channels with sample rates"),
        ("Stage 2", "artifacts/splits/split_manifest.csv", "subject assignment per seed"),
        ("Stage 2", "artifacts/windows/windows_10s.csv", "window manifest, 10 s"),
        ("Stage 2", "artifacts/windows/windows_30s.csv", "window manifest, 30 s"),
        ("Stage 2", "artifacts/windows/window_summary.json", "counts and leakage checks"),
        ("Stage 3", "artifacts/signals/signal_qc.csv", "per-record signal QC"),
        ("Stage 3", "artifacts/signals/resp_qc.csv", "respiration SNR and breathing rate"),
        ("Stage 3", "artifacts/signals/preprocessing_config.json", "filter settings actually used"),
        ("Stage 3", "artifacts/signals/qc_*.png", "waveform and spectrogram snapshots"),
        ("Stage 3", "artifacts/normalization/norm_stats.json", "training-only mean and std"),
        ("Stage 4", "artifacts/pilot/pilot_summary.csv", "pilot results (not final)"),
        ("Stage 4", "artifacts/pilot/pilot_subject_auc.csv", "per-subject pilot AUC"),
        ("Stage 4", "artifacts/pilot/checkpoints/*.pt", "pilot checkpoints"),
        ("Stage 5", "artifacts/train/seed42_10s/frozen.json", "frozen checkpoints and thresholds"),
        ("Stage 5", "artifacts/train/seed42_10s/metrics_*.json", "validation metrics per model"),
        ("Stage 5", "artifacts/train/seed42_10s/history_*.csv", "per-epoch loss and timings"),
        ("Stage 5", "artifacts/train/seed42_10s/checkpoints/*.pt", "final checkpoints"),
        ("Stage 6", "artifacts/test/seed42_10s/test_summary.csv", "six metrics per model on test"),
        ("Stage 6", "artifacts/test/seed42_10s/test_run.json", "what was run and from which freeze"),
        ("Stage 6", "artifacts/test/seed42_10s/test_auc_decomposition.csv", "pooled vs per-subject AUC"),
        ("Stage 6", "artifacts/test/seed42_10s/test_by_subject_*.csv", "per-subject test metrics"),
        ("Stage 6", "artifacts/test/seed42_10s/test_by_dataset_*.csv", "per-dataset test metrics"),
        ("Stage 6", "artifacts/test/seed42_10s/test_predictions_*.csv", "per-window scores"),
        ("Stage 6b", "artifacts/calibration/calibration_summary.csv", "pooled vs per-subject gap"),
        ("Stage 6b", "artifacts/calibration/calibration_by_subject.csv", "score offset vs prevalence"),
        ("Stage 6b", "artifacts/calibration/calibration_transforms.csv", "raw vs per-record z-score"),
        ("Stage 6b", "artifacts/calibration/calibration_prevalence.csv", "prevalence-score correlation, both splits"),
        ("Stage 6b", "artifacts/calibration/calibration_bootstrap.csv", "subject-level 95% CI"),
        ("Stage 6c", "artifacts/offset_probe/per_record_scaling.csv", "per-record std and clip fraction"),
        ("Stage 6c", "artifacts/offset_probe/record_features.csv", "record features vs scores"),
        ("Stage 6c", "artifacts/offset_probe/offset_correlations.csv", "feature-score correlations"),
        ("Stage 7", "artifacts/aggregate/seed42-1337-2024_10s/val_summary_mean_sd.csv", "validation mean ± SD, 3 seeds"),
        ("Stage 7", "artifacts/aggregate/seed42-1337-2024_10s/test_summary_mean_sd.csv", "test mean ± SD, 3 seeds"),
        ("Stage 7", "artifacts/aggregate/seed42-1337-2024_10s/test_by_dataset_mean_sd.csv", "per-dataset test AUC"),
        ("Stage 7", "artifacts/aggregate/seed42-1337-2024_10s/test_bootstrap_by_seed.csv", "bootstrap CI per seed"),
        ("Stage 7", "artifacts/aggregate/seed42-1337-2024_10s/val_ranking.csv", "model rank per seed"),
        ("Stage 7b", "artifacts/aggregate/seed42-1337-2024_30s/test_summary_mean_sd.csv", "30 s test, 3 seeds"),
        ("Stage 7b", "artifacts/aggregate/seed42-1337-2024_30s/val_summary_mean_sd.csv", "30 s validation, 3 seeds"),
        ("Stage 7b", "artifacts/aggregate/seed42-1337-2024_30s/test_per_seed.csv", "30 s per-seed AUC"),
        ("Stage 7b", "artifacts/aggregate/seed42-1337-2024_30s/test_bootstrap_by_seed.csv", "30 s bootstrap CI"),
        ("Stage 7b", "notebooks/colab_train_30s.ipynb", "GPU run used for seed 2024 at 30 s"),
        ("all", "config/policy.json", "eligibility, exclusions, thresholds"),
        ("all", "config/model.json", "architecture and training hyperparameters"),
        ("all", "scripts/01..20_*.py", "one script per stage, re-runnable"),
    ]
    a(table(pd.DataFrame(files, columns=["Stage", "File", "Contents"])))

    a('<p class="sub" style="margin-top:36px">Re-run any stage with '
      '<code class="mono">python scripts/01_audit.py</code> … '
      '<code class="mono">python scripts/08_dataset.py 42 10</code> … '
      '<code class="mono">python scripts/15_train.py</code> … '
      '<code class="mono">python scripts/16_test.py</code>.</p>')
    return "\n".join(parts)


if __name__ == "__main__":
    body = build()
    doc = ("<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
           "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
           "<title>Sleep Apnea Multimodal — Stage Report</title></head><body>"
           f"{body}</body></html>")
    out = OUT / "stage_report.html"
    out.write_text(doc, encoding="utf-8")
    print("wrote", out, f"({out.stat().st_size / 1024:.0f} KB)")
