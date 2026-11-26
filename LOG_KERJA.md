# Log Kerja — Sleep Apnea Multimodal

**Terakhir diperbarui:** 26 November 2026
**Acuan:** `paper RM (1).pdf` (paper UTS) + `Konteks_Penelitian_Sleep_Apnea.md`
**Status keseluruhan:** Eksperimen 1 (perbandingan model) dan Eksperimen 2 (durasi window)
selesai untuk **4 model × 3 seed × 2 durasi (10 s dan 30 s)**. Test tiap seed dibaca **sekali**
dengan model beku. Eksperimen 3 (ablation preprocessing) sudah **dijalankan** (30 run, T4 GPU)
tetapi **belum layak dikutip**: arm `baseline` tidak mereproduksi Eksperimen 2, jadi selisih
antar arm belum bisa dipisahkan dari variasi run (lihat Tahap 7d). Belum: Eksperimen 4
(single vs combined-dataset + LODO) dan lengan ECG-only Apnea-ECG.

Dokumen ini mencatat apa yang **sudah benar-benar dijalankan**, angka aktualnya, dan alasan
setiap keputusan — supaya bisa dipertanggungjawabkan kalau ditanya. Semua angka berasal dari
file di `artifacts/`, bukan dari ingatan. Untuk tampilan visual: buka `reports/stage_report.html`.
Untuk serah-terima ke orang lain: lihat `HANDOFF.md` (urutan baca, status, aturan, biaya komputasi).

---

## 1. Ringkasan satu halaman

| Item | Angka aktual |
|---|---|
| Dataset dipertimbangkan | 3 (UCDDB, MIT-BIH PSG, Apnea-ECG) |
| Dataset dipakai untuk multimodal | 2 (UCDDB + MIT-BIH PSG) |
| Record layak | **39** (24 UCDDB + 15 PSG) |
| Subjek layak | **38** (24 + 14) |
| Total durasi layak | **±240,9 jam** (UCDDB 166,5 h + PSG 74,4 h) |
| Window 10 s (seed 42) | 144.599 (train 116.199 / val 14.407 / test 13.993) |
| Window 30 s (seed 42) | 48.175 (train 38.711 / val 4.801 / test 4.663) |
| Split subjek | 70/15/15, 3 seed (42, 1337, 2024), stratifikasi severitas |
| Kebocoran subjek | 0 (diverifikasi per seed) |
| Rate positif window 10 s | UCDDB 9,8% · PSG 32,8% |
| Model dilatih | 4 konfigurasi × 3 seed × **2 durasi** (10 s dan 30 s), data penuh |
| Status training | 10 s: 3 seed CPU · 30 s: seed 42 & 1337 CPU, **seed 2024 GPU (T4)** |
| Status test | 10 s dan 30 s, 3 seed, 6 subjek/split; pooled AUC `resp_only` 0,742/0,747 (Tahap 7b, 7c) |
| Status Eksperimen 2 | `proposed` +0,123 di 30 s, naik di **3/3 seed**; `resp_only` tidak terpengaruh (Tahap 7c) |
| Status Eksperimen 3 | 30 run selesai (5 arm × 2 model × 3 seed, T4 GPU) tetapi **belum dapat ditafsirkan**: `baseline` menyimpang −0,164…+0,229 dari Eksperimen 2 (Tahap 7d) |

---

## 2. Keputusan yang sudah diambil

Semua juga tersimpan mesin-terbaca di `config/policy.json`.

| # | Keputusan | Alasan | Dasar |
|---|---|---|---|
| 1 | Apnea-ECG **tidak** dipakai untuk multimodal | ECG-nya **ada** (dan memang dibutuhkan), tetapi respirasi tidak ada. Model multimodal menerima ECG **dan** respirasi untuk window yang sama, jadi kekurangan satu modalitas membuatnya tidak bisa masuk jalur multimodal. Tetap berguna untuk lengan ECG-only | Sec. 3.3 ("availability ... will be verified through a dataset audit"), Sec. 3.4.2 |
| 2 | Policy respirasi = **strict airflow** | UCDDB `Flow` (thermistor) ↔ PSG `Resp (nasal)`. Menyetarakan jenis sinyal, menghindari mencampur airflow dengan effort | Sec. 3.4.2 (airflow/thoracic/abdominal berbeda informasi) |
| 3 | `slp01a`, `slp01b` keluar | Hanya `Resp (sum)`, tanpa kanal airflow | konsekuensi keputusan 2 |
| 4 | `slp61` keluar | Hanya `Resp (abdominal)`, tanpa kanal airflow | konsekuensi keputusan 2 |
| 5 | `ucddb005` keluar | Kanal `Flow` rusak: std 0,503 (record lain 4,2–6,0), hanya 318 nilai unik, datar sepanjang ±6 jam. Kanal `Sum`/`ribcage`/`abdo` masih sehat (puncak 15,9–16,4 napas/menit) tetapi tidak dipakai agar policy airflow tetap konsisten | audit kanal |
| 6 | Positif = apnea **atau** hypopnea | Sesuai aturan paper | Sec. 3.4.4 |
| 7 | `POSSIBLE` (19 event) dan `PB EVENT` (91) **bukan** positif | Event ambigu; menandainya positif menambah label noise tanpa dasar | Sec. 3.4.4 |
| 8 | Ambang overlap window ≥ **50%** | Aturan paper | Sec. 3.4.4 |
| 9 | Train overlap (stride = ½ window), val/test non-overlap | Paper: training pakai overlapping, evaluasi tidak | Sec. 3.4.4 |
| 10 | Window hanya dibuat di rentang beranotasi | Tidak boleh melabeli normal interval tanpa anotasi | konteks §6, §7 |
| 11 | Tail dibuang (truncate), bukan padding | Menghindari data palsu di window | keputusan implementasi (paper: TBD) |
| 12 | Statistik z-score di-fit **per dataset**, dari window training saja | Mencegah kebocoran statistik dari val/test; skala UCDDB vs PSG beda ~22× | Sec. 3.4.3 |
| 13 | ECG 100 Hz, respirasi 25 Hz | Bitrate moderat; cut-off aman untuk QRS (≤40 Hz) dan napas (≤0,7 Hz) | provisional, paper menyerahkan ke audit |
| 14 | Clipping c = 5 σ | Membatasi artefak gerak tanpa memotong sinyal normal | provisional |
| 15 | Patch 0,2 s di kedua cabang (ECG 20 sampel, respirasi 5 sampel) | Kedua cabang menghasilkan jumlah token dan rentang waktu per token yang sama (50 token/10 s). Paper membolehkan rate berbeda asal waktu sejajar | Sec. 3.4.2, 3.4.6 |
| 16 | Arsitektur numerik: d=64, 2 layer, 4 head, FFN 128, dropout 0,1, head 64 | Semua angka ini TBD di paper; dipilih agar muat di CPU. Wajib dituning di validation sebelum test | Sec. 3.4.5–3.4.7, 3.4.9 |
| 17 | Pilot memakai subset 8.000 train / 3.000 val, 4 epoch, batch 64 | Mesin hanya CPU (8 core, RAM bebas ~2 GB). Pilot bukan eksperimen final, jadi subset dicatat dan test tidak disentuh | keputusan implementasi |
| 18 | Threshold keputusan dipilih di validation dengan F1 maksimum | Paper menyebut F1 maksimal sebagai contoh aturan, bukan satu-satunya; dipakai apa adanya lalu dicatat | Sec. 3.4.9 |

**Catatan penting soal keputusan 1:** paper UTS tidak menyuruh membuang dataset. Paper menulis
ketiga dataset "**considered**" dan kelayakannya ditentukan audit. Jadi ini **hasil audit**,
bukan perubahan metode. Konsekuensinya harus ditulis di laporan akhir: combined-dataset training
mencakup 2 dataset, LODO jadi 2 arah. Jangan menulis "all three datasets" untuk eksperimen yang
tidak menjalankannya.

---

## 3. Penjelasan tiap tahap — ngapain dan kenapa

Bagian ini menjelaskan **tujuan** tiap tahap, bukan hanya keluarannya. Angka lengkap ada di
bagian 4.

### Tahap 0 — Verifikasi dataset

**Tujuan:** memastikan file yang dipakai memang rilis resmi, bukan versi korup atau termodifikasi,
dan memetakan struktur kanal tiap dataset.

**Cara:** hitung SHA-256 tiap file lalu bandingkan dengan `SHA256SUMS.txt` resmi PhysioNet;
baca header semua record untuk mendata kanal, sampling rate, dan durasi.

**Kenapa perlu:** tanpa ini, klaim "memakai dataset publik X" hanya asumsi. Verifikasi juga yang
memunculkan fakta bahwa PSG punya 12 layout kanal berbeda — fakta yang kemudian mengubah policy
respirasi.

### Tahap 1 — Audit record & anotasi

**Tujuan:** menentukan record mana yang **layak** masuk model, dan memahami arti anotasi tiap
dataset sebelum dipakai.

**Cara:**
- baca header tiap record (kanal, fs, durasi);
- parse anotasi per dataset (UCDDB event onset+durasi; PSG token per epoch 30 s; Apnea-ECG per menit);
- cek semua event jatuh di dalam rentang rekaman;
- hitung proporsi positif dan jumlah event.

**Kenapa perlu:** ini gate yang paper tulis sendiri ("availability ... will be verified through a
dataset audit"). Hasilnya juga menemukan hal yang tidak terlihat dari katalog: 5 record PSG
anotasinya mulai belakangan (lead-in 60–780 s), sehingga interval itu tidak boleh dilabeli normal.

**Keluaran:** tabel `eligible` per record, alasan pengecualian, dan distribusi label.

### Tahap 2 — Split subjek + windowing

Dua pekerjaan berbeda dalam satu skrip karena keduanya bergantung pada anotasi.

**Split subjek** — membagi **subjek** (bukan window) ke train/val/test 70/15/15, distratifikasi
berdasarkan severitas, untuk 3 seed. Tujuannya mencegah kebocoran: window dari orang yang sama
tidak boleh muncul di dua split, karena itu membuat skor test terlalu optimistis. Pasangan record
milik satu orang (`slp02a` + `slp02b`) dijaga tetap satu split.

**Windowing** — memotong rekaman menjadi window 10 s dan 30 s, lalu menetapkan label:
- positif bila ≥50% durasi window beririsan dengan apnea/hypopnea;
- train memakai stride setengah window (overlap), val/test tidak overlap;
- window hanya dibuat di dalam rentang beranotasi; lead-in dan ekor tanpa anotasi dibuang,
  bukan dilabeli normal.

**Keluaran:** dua manifest window (10 s dan 30 s) berisi seed, dataset, record, subjek, split,
waktu mulai/akhir, label, dan fraksi overlap.

### Tahap 3 — Preprocessing, normalisasi, loader

**3a — Filter dan resample.** Tujuan: membersihkan sinyal dan menyamakan sampling rate agar
semua window punya panjang identik dalam sampel.
- ECG: band-pass 0,5–40 Hz (zero-phase agar fase tidak bergeser) + notch bila gangguan listrik
  terdeteksi → 100 Hz.
- Respirasi: low-pass 0,7 Hz + detrend untuk membuang baseline → 25 Hz.
- Resampling memakai filter anti-alias, sesuai syarat paper.

QC-nya membuktikan keselarasan: durasi kedua modalitas cocok sampai 0 ms di 39/39 record,
tidak ada NaN, dan notch hanya perlu di `slp67x`.

**3b — Normalisasi.** Tujuan: menyamakan skala antar kanal dan antar dataset.
- Statistik (mean, std) dihitung **hanya dari window training**, per dataset per modalitas.
- Dataset dipisah karena skala UCDDB (L/min) dan PSG (tak terkalibrasi) berbeda ~22×.
- Setelah z-score, nilai di-clip ke ±5σ untuk meredam artefak gerak.

**3c — Loader.** Tujuan: menyajikan pasangan (ECG, respirasi, label) ke model tanpa memuat
seluruh data ke memori. Sinyal dipotong on-the-fly dari cache per record.

**Keluaran:** cache sinyal, `norm_stats.json`, dan `WindowDataset` yang sudah lolos uji bentuk,
paritas label, dan throughput.

### Tahap 4 — Model pilot (sudah dijalankan, hasilnya bukan performa final)

**Tujuan:** membuktikan pipeline training hidup sebelum mengeluarkan biaya besar.
- Empat konfigurasi: proposed, ECG-only, respiration-only, simple fusion.
- 1 seed, window 10 s, budget kecil (subset 8.000 train / 3.000 val, 4 epoch).
- Yang diperiksa: loss turun, gradien wajar, checkpoint bisa dimuat dan dipulihkan.
- **Bukan** untuk angka final, dan `test` tidak dibaca sama sekali.

### Tahap 5 — Training penuh + freeze (sudah dijalankan, seed 42, 10 s)

**Tujuan:** mendapatkan model final dan mengunci keputusan sebelum menyentuh test.
- Seluruh 116.199 window training (26 subjek), validasi 14.407 window (6 subjek).
- Early stopping pada validation loss; tuning hyperparameter hanya di validation.
- Decision threshold dipilih di validation (F1 maksimum).
- Checkpoint dan threshold **dibekukan** sebelum prediksi test.

### Tahap 6 — Prediksi test + metrik (sudah dijalankan, seed 42, 10 s)

**Tujuan:** evaluasi tak bias pada subjek yang belum pernah dilihat model.
- Prediksi dengan model beku (`frozen.json` dibaca apa adanya: checkpoint, epoch, dan threshold tidak diubah).
- Enam metrik: accuracy, sensitivity, specificity, precision, F1, ROC-AUC.
- ROC-AUC dihitung dari skor/probabilitas, bukan dari label hasil threshold.
- Test dibaca **sekali**; tidak ada keputusan yang diambil dari hasil test.

### Tahap 7 — Analisis komparatif (belum)

**Tujuan:** menjawab empat eksperimen paper.
1. Proposed vs ECG-only vs respiration-only vs simple fusion.
2. Window 10 s vs 30 s.
3. Ablation preprocessing (filtering, normalization, clipping).
4. Single-dataset vs combined-dataset training; LODO 2 arah sebagai protokol terpisah.
- Dilaporkan sebagai mean ± SD antar seed, dengan bootstrap di level subjek.
- Ditambah parameter count dan inference time per window.

---

## 4. Apa yang sudah dikerjakan, per tahap

### Tahap 0 — Verifikasi dataset (`scripts/03_verify_datasets.py`)

- Membandingkan seluruh file dengan `SHA256SUMS.txt` resmi PhysioNet.
- **Hasil:** UCDDB 101/102 OK · PSG 113/114 OK · Apnea-ECG 383/384 OK.
- Selisih hanya file non-sinyal: `SubjectDetails.xls` (hash beda), `slpdb.shtml` dan
  `annotations.shtml` (halaman index tidak diunduh).
- Ditemukan PSG **tidak seragam**: 12 layout kanal berbeda; 15 dari 18 record punya
  `Resp (nasal)`.
- Output: `artifacts/verify/dataset_verification.json`, `slpdb_resp_channels.csv`.

### Tahap 1 — Audit record & anotasi (`scripts/01_audit.py`)

| Dataset | Record | Subjek | Durasi rata-rata | Anotasi | Positif |
|---|---|---|---|---|---|
| UCDDB | 25 | 25 | 6,93 ± 0,53 h | event onset+durasi, stage 30 s | 9,8% waktu |
| MIT-BIH PSG | 18 | 16 | 4,76 ± 1,71 h | token per epoch 30 s | 32,8% epoch |
| Apnea-ECG | 70 | 70 | 8,20 ± 0,54 h | label per menit A/N | 38,1% menit |

- UCDDB: 3.428 event — 2.623 hypopnea, 695 apnea, 91 PB, 19 POSSIBLE.
- 0 event UCDDB jatuh di luar rentang rekaman (bukti anotasi sejajar waktu).
- 5 record PSG punya anotasi yang mulai belakangan (lead-in 60–780 s) → tidak dilabeli normal.
- 77 epoch PSG di ekor rekaman tanpa anotasi → dibuang.
- Record turunan Apnea-ECG (`a01r`, `a01er`, dst — 16 file) dikecualikan dari hitungan agar
  tidak menduplikasi subjek.
- Output: `artifacts/audit/audit_manifest.csv`, `audit_summary.md/json`, `ucddb_channels.csv`.

### Tahap 2 — Split subjek + windowing (`scripts/02_split_windows.py`)

- Split 70/15/15 per dataset, stratifikasi berdasarkan severitas (AHI untuk UCDDB, fraksi epoch
  apnea untuk PSG), 3 seed.
- Realisasi (seed 42): UCDDB 16/4/4 · PSG 10/2/2.
- **Kebocoran subjek = 0** di semua seed (cek irisan himpunan subjek antar split).
- Pasangan record satu subjek tetap satu split (`slp02a`+`slp02b` bersama).
- Window 10 s: 144.599 · 30 s: 48.175 (seed 42, semua dataset).
- Label: positif bila ≥50% durasi window beririsan dengan union interval apnea/hypopnea.
- Output: `artifacts/splits/split_manifest.csv`, `artifacts/windows/windows_{10,30}s.csv`,
  `window_summary.json`.

### Tahap 3 — Preprocessing, normalisasi, loader

**3a. Filter + resample + QC (`scripts/04_preprocess.py`)**

- ECG: band-pass 0,5–40 Hz (Butterworth order 4, zero-phase) + notch bila mains terdeteksi
  → **100 Hz**.
- Respirasi: low-pass 0,7 Hz (order 4) + detrend linear → **25 Hz**.
- **Durasi ECG dan respirasi cocok sampai 0 ms di 39/39 record.**
- Tidak ada NaN/Inf di sinyal mentah.
- Notch hanya aktif di `slp67x` (60 Hz kuat); 38 record lain tidak menunjukkan peak mains.
- Flatline maksimum: ECG 4,75% (`ucddb008`), respirasi 14,7% (`slp60`) — dicatat, tidak digugurkan.
- Output: cache `.npy` per record, `signal_qc.csv`, `preprocessing_config.json`, gambar QC.

**3b-1. Statistik normalisasi (`scripts/07_normalize.py`)**

- Dihitung dari window **training saja**, per dataset per modalitas.
- std train: UCDDB ECG ≈0,35 / resp ≈5,10; PSG ECG ≈0,18 / resp ≈0,23.
- Perbedaan skala respirasi ~22× antar dataset → disamakan oleh z-score per dataset.
- Output: `artifacts/normalization/norm_stats.json`, `clipping_report.csv`.

**3b-2. Loader (`scripts/08_dataset.py`)**

- `WindowDataset` memotong sinyal on-the-fly dari manifest → memori datar.
- Verifikasi: bentuk (1,1000)/(1,250) untuk 10 s dan (1,3000)/(1,750) untuk 30 s;
  label 100% sama dengan manifest; tanpa NaN; throughput ~2.900–3.250 window/detik (1 proses).

### Tahap 4 — Model pilot (`scripts/models.py`, `scripts/13_pilot.py`, `scripts/14_pilot_diag.py`)

**Arsitektur yang dibangun (sesuai Sec. 3.4.1, 3.4.5–3.4.8):**

| Model | Parameter | Komposisi |
|---|---|---|
| proposed | 217.602 (10 s) | 2 encoder Transformer + bidirectional cross-attention + mean pooling + concat + head 2 kelas |
| simple_fusion | 150.658 | 2 encoder + concat pooled, tanpa cross-attention |
| ecg_only | 75.906 | 1 encoder ECG |
| resp_only | 74.946 | 1 encoder respirasi |

- Patch 0,2 s → 50 token/cabang pada 10 s, 150 token/cabang pada 30 s (diverifikasi untuk kedua durasi).
- Batch 64, AdamW, warmup 5% + cosine, class-weighted CE (bobot dari subset training saja), early stopping pada validation loss (patience 2).

**Hasil yang benar-benar didapat (seed 42, 10 s, subset):**

| Model | Epoch jalan | Best val loss | AUC train (3.000) | AUC val (3.000) | Reload checkpoint |
|---|---|---|---|---|---|
| proposed | 3 | 0,580 | 0,799 | 0,467 | diff 0,0 |
| ecg_only | 4 | 0,619 | 0,797 | 0,453 | diff 0,0 |
| resp_only | 4 | 0,679 | 0,795 | 0,557 | diff 0,0 |
| simple_fusion | 4 | 0,637 | 0,823 | 0,498 | diff 0,0 |

**Yang dibuktikan:** loss training turun di keempat model, gradien finite, checkpoint disimpan lalu
dimuat ulang dan menghasilkan logit identik, inference ~0,2–0,7 ms/window. Pipeline hidup.

**Yang juga terbukti (dan harus ditulis sebagai batasan, bukan kegagalan tersembunyi):** model
**menghafal subjek**. AUC di window training 0,80 tetapi AUC validasi gabungan 0,45–0,56. Bila dihitung
per subjek, AUC validasi justru 0,84–0,87 untuk subjek 18 dan 19 (UCDDB) dan 0,60–0,66 untuk slp03,
sedangkan subjek 8 hanya 0,33–0,61 dan slp41 tidak punya window positif sama sekali. Artinya offset
skor antar subjek berbeda-beda, dan validasi hanya berisi 6 subjek. Ini masalah anggaran/regularisasi
Tahap 5, bukan bug pada kode: label, bentuk, normalisasi, dan pembacaan sinyal sudah diverifikasi.

Dampak pada pemilihan threshold: pada val gabungan, F1 terbaik dicapai dengan memprediksi hampir semua
window positif (threshold 0,11 untuk proposed). Angka itu adalah gejala kalibrasi lintas subjek, bukan
bukti bahwa model bisa dipakai.

Output: `artifacts/pilot/pilot_summary.csv`, `pilot_{model}.json`, `pilot_diagnostics.csv`,
`pilot_subject_auc.csv`, `checkpoints/*.pt`, `model_smoke.json`.

### Tahap 5 — Training penuh + freeze (`scripts/15_train.py`)

Dijalankan pada seed 42, window 10 s, **seluruh** 116.199 window training (26 subjek) dan 14.407
window validasi (6 subjek). `test` tidak dibaca oleh skrip ini.

| Model | Param | Epoch jalan | Best epoch | Val loss | Threshold |
|---|---|---|---|---|---|
| proposed | 217.602 | 6 | 3 | 0,5095 | 0,210 |
| simple_fusion | 150.658 | 7 | 4 | 0,5630 | 0,370 |
| ecg_only | 75.906 | 6 | 3 | 0,5821 | 0,095 |
| resp_only | 74.946 | 4 | 1 | 0,5644 | 0,140 |

**Metrik validasi pada threshold terpilih (bukan hasil test):**

| Model | Acc | Sens | Spec | Prec | F1 | ROC-AUC | ms/window |
|---|---|---|---|---|---|---|---|
| proposed | 0,582 | 0,672 | 0,573 | 0,131 | 0,219 | 0,644 | 1,07 |
| simple_fusion | 0,567 | 0,785 | 0,546 | 0,142 | 0,241 | 0,693 | 0,73 |
| ecg_only | 0,272 | 0,957 | 0,207 | 0,104 | 0,187 | 0,543 | 0,48 |
| resp_only | 0,386 | 0,840 | 0,343 | 0,109 | 0,193 | 0,591 | 0,51 |

**Dipecah per dataset (ROC-AUC):**

| Model | UCDDB | MIT-BIH PSG |
|---|---|---|
| proposed | 0,761 | 0,364 |
| simple_fusion | 0,821 | 0,458 |
| ecg_only | 0,577 | 0,330 |
| resp_only | 0,685 | 0,377 |

**Per subjek (proposed):** s18 AUC 0,838 · s19 0,873 · s23 0,494 · s8 0,632 · slp03 0,599 ·
slp41 tidak punya window positif sehingga AUC tidak terdefinisi.

**Yang benar-benar terbukti:**
- Pipeline penuh jalan: 26 subjek training, early stopping aktif, checkpoint + threshold dibekukan
  di `artifacts/train/seed42_10s/frozen.json`.
- Data penuh menaikkan generalisasi dibanding pilot (proposed AUC val 0,467 → 0,644;
  simple fusion 0,498 → 0,693).
- Biaya inference terukur: 0,48–1,07 ms per window pada CPU 8 core, batch 64.

**Yang harus diakui:**
- AUC PSG di bawah 0,5 pada semua model. Validasi PSG hanya 2 subjek, salah satunya (slp41) tanpa
  window positif, sehingga angka PSG tidak stabil dan tidak bisa dijadikan bukti apa pun.
- Threshold hasil F1 sangat rendah (0,095–0,37) dan precision hanya 0,10–0,14 pada prevalensi 8,9%.
  Artinya model masih condong memprediksi positif; F1 yang dilaporkan berasal dari trade-off itu.
- `proposed` **tidak** lebih baik dari `simple_fusion` pada validasi ini (AUC 0,644 vs 0,693).
  Itu temuan yang harus dilaporkan apa adanya, bukan alasan menyesuaikan hasil.
- Satu epoch `ecg_only` memakan 13.939 s karena mesin sempat dibebani/di-suspend. Tidak ada
  keputusan training yang bergantung pada waktu, jadi hasilnya tidak terpengaruh; durasi lain
  ~280–1.090 s/epoch.

Output: `artifacts/train/seed42_10s/{run_config.json, history_*.csv, metrics_*.json,
val_predictions_*.csv, val_by_subject_*.csv, checkpoints/*.pt, frozen.json}`.

### Tahap 6 — Prediksi test + metrik (`scripts/16_test.py`)

Dijalankan 22 November 2026 dengan `frozen.json` dari Tahap 5 dibaca apa adanya. Checkpoint,
best epoch, dan threshold tidak diubah, tidak ada retraining, tidak ada tuning setelah melihat test.
Test: **13.993 window, 7 record, 6 subjek** (UCDDB 4 subjek · PSG 2 subjek), rate positif 14,7%
(2.057 window).

| Model | Acc | Sens | Spec | Prec | F1 | ROC-AUC | Threshold | ms/window |
|---|---|---|---|---|---|---|---|---|
| resp_only | 0,506 | 0,866 | 0,443 | 0,211 | 0,340 | **0,734** | 0,140 | 0,39 |
| proposed | 0,517 | 0,556 | 0,510 | 0,163 | 0,253 | 0,526 | 0,210 | 0,73 |
| simple_fusion | 0,456 | 0,459 | 0,456 | 0,127 | 0,199 | 0,450 | 0,370 | 0,55 |
| ecg_only | 0,210 | 0,631 | 0,138 | 0,112 | 0,190 | 0,359 | 0,095 | 0,38 |

**ROC-AUC per dataset:**

| Model | UCDDB | MIT-BIH PSG |
|---|---|---|
| resp_only | 0,720 | 0,723 |
| proposed | 0,738 | 0,579 |
| simple_fusion | 0,693 | 0,502 |
| ecg_only | 0,592 | 0,418 |

**Dekomposisi AUC (diagnostik, bukan metrik utama):**

| Model | Pooled AUC | Mean AUC per subjek | Min–max subjek | Within-subject AUC |
|---|---|---|---|---|
| resp_only | 0,734 | 0,748 | 0,635–0,833 | 0,712 |
| proposed | 0,526 | 0,733 | 0,530–0,852 | 0,672 |
| simple_fusion | 0,450 | 0,671 | 0,475–0,860 | 0,604 |
| ecg_only | 0,359 | 0,554 | 0,427–0,618 | 0,505 |

Pooled AUC dan AUC per subjek berbeda jauh untuk `proposed` (0,526 vs 0,733). Penyebabnya
offset skor antar subjek: model memberi skala skor berbeda per orang, sehingga saat semua window
digabung, peringkatnya rusak walau di dalam satu subjek peringkatnya cukup baik. `resp_only`
paling stabil karena pooled ≈ per subjek (0,734 vs 0,748).

**Per subjek (proposed):** s22 0,694 · s24 0,793 · s26 0,812 · s6 0,852 · slp02 0,530 · slp16 0,716.
Keempat subjek UCDDB di atas 0,69; kedua subjek PSG lebih rendah, dan slp02 paling lemah.

**Yang benar-benar terbukti:**
- Test dibaca sekali dengan model beku; keluaran deterministik dan bisa direproduksi.
- Enam metrik dihitung dari skor kontinu untuk AUC, bukan dari label ter-threshold.
- Inference 0,38–0,73 ms/window pada CPU, jauh di bawah durasi window 10 s.

**Yang harus diakui (jangan dibungkus):**
- **`proposed` bukan model terbaik di test.** `resp_only` unggul jauh (AUC 0,734 vs 0,526).
  Cross-attention belum terbukti memberi manfaat; ini temuan, bukan alasan mengubah metode.
- **Pooled AUC `proposed` (0,526) hampir setara tebak acak**, padahal mean AUC per subjek 0,733.
  Selisih 0,21 poin itu adalah masalah kalibrasi lintas subjek yang belum diselesaikan, dan harus
  dilaporkan sebagai batasan utama, bukan disembunyikan di balik rata-rata per subjek.
- **`ecg_only` praktis tidak berguna di UCDDB** (AUC 0,592, specificity 0,044): model memprediksi
  hampir semua window positif. Sensitivitas tinggi di sini bukan prestasi.
- **Hanya 2 subjek PSG di test**, jadi angka PSG tetap tipis. AUC PSG `proposed` 0,579 lebih baik
  dari validasi (0,364), tetapi dengan n=2 subjek jangan dijadikan klaim generalisasi.
- Threshold dari validasi tidak optimal di test (precision 0,11–0,21 pada prevalensi 14,7%).
  Threshold **tidak** disesuaikan setelah melihat test; itu memang aturan protokolnya.

Output: `artifacts/test/seed42_10s/{test_run.json, test_summary.csv,
test_auc_decomposition.csv, metrics_test_*.json, test_predictions_*.csv,
test_by_subject_*.csv, test_by_dataset_*.csv}` dan `artifacts/test/stage6_console.log`.

### Tahap 6b — Analisis kalibrasi lintas subjek (`scripts/17_calibration.py`)

**Sifat skrip ini: post-hoc diagnostik, bukan protokol.** Dijalankan setelah test dilihat, jadi tidak
satu pun keputusannya boleh dipakai untuk memilih model atau mengganti metrik utama. Yang dilaporkan
tetap pooled AUC dengan threshold beku.

**Pertanyaan 1 — apakah gap pooled vs per subjek itu sifat model atau sifat subjek test?**

| Model | Pooled val | Mean subjek val | Gap val | Pooled test | Mean subjek test | Gap test |
|---|---|---|---|---|---|---|
| proposed | 0,644 | 0,687 | 0,044 | 0,526 | 0,733 | **0,207** |
| simple_fusion | 0,693 | 0,681 | −0,012 | 0,450 | 0,671 | 0,221 |
| ecg_only | 0,543 | 0,604 | 0,061 | 0,359 | 0,554 | 0,195 |
| resp_only | 0,591 | 0,695 | 0,105 | 0,734 | 0,748 | 0,014 |

Gap `proposed` nyaris nol di validasi (0,044) lalu meledak di test (0,207). Jadi bukan arsitekturnya
yang rusak; subjek test punya offset skor yang jauh lebih lebar. Tiga dari empat model menunjukkan
pola sama; `resp_only` satu-satunya yang stabil di kedua split.

**Pertanyaan 2 — apakah offset itu berkorelasi dengan prevalensi subjek?**

Spearman(prevalensi, skor rata-rata) di 6 subjek test: proposed **−0,943** (p=0,005) · ecg_only −0,657 ·
simple_fusion −0,257 · resp_only −0,029.

Untuk `proposed` hubungannya hampir monoton terbalik di **test**: subjek 22 punya prevalensi **1,9%**
(terendah) tetapi skor rata-rata **0,450** (tertinggi), sehingga 70,9% window-nya diprediksi positif.
Leave-one-out menunjukkan ini tidak bergantung pada satu subjek: membuang subjek mana pun
menyisakan ρ antara −0,90 dan −1,00.

**Batas klaim ini (jangan dilebihkan):** korelasi yang sama di **validasi** hanya +0,029 (p=0,957).
Jadi −0,943 adalah sifat **subjek test**, bukan sifat model yang stabil di semua split. Menyebutnya
"model selalu membalik prior" tidak didukung data.

**Yang justru sudah terlihat di validasi:** subjek `slp41` (prevalensi **0,0%**) mendapat skor rata-rata
**0,504** — tertinggi di seluruh validasi, di atas `slp23` (12,7%) dan `slp03` (33,5%). Subjek tanpa
window positif tidak punya AUC (tidak terdefinisi), jadi ia hilang dari semua tabel AUC dan tidak
pernah memicu peringatan, padahal skornya abnormal dan berpola sama dengan subjek 22 di test.

**Pertanyaan 3 — apakah z-score per rekaman menutup gap? (sekunder, eksploratif)**

| Model | Val raw | Val z-record | Test raw | Test z-record |
|---|---|---|---|---|
| proposed | 0,644 | **0,601** | 0,526 | 0,671 |
| simple_fusion | 0,693 | **0,624** | 0,450 | 0,600 |
| ecg_only | 0,543 | 0,537 | 0,359 | 0,503 |
| resp_only | 0,591 | 0,592 | 0,734 | 0,714 |

**Ini temuan yang harus ditulis apa adanya:** transformasi itu menaikkan angka test tetapi
**menurunkan** angka validasi pada `proposed` (0,644 → 0,601) dan `simple_fusion` (0,693 → 0,624).
Artinya bukan perbaikan yang berprinsip, melainkan penyelamat post-hoc. Kalau dilaporkan sebagai
hasil, harus dinyatakan bahwa transformasi dipilih setelah melihat test dan bahwa ia merugikan split
yang tidak dipakai untuk memilihnya. `resp_only` malah sedikit turun di kedua split (0,734 → 0,714,
0,591 → 0,592) karena model itu memang tidak punya masalah offset.

**Bootstrap level subjek (6 subjek, 2.000 resample):**

| Model | Pooled test | Median | 95% CI | z-record median | 95% CI |
|---|---|---|---|---|---|
| resp_only | 0,734 | 0,737 | 0,629–0,810 | 0,719 | 0,672–0,796 |
| proposed | 0,526 | 0,547 | 0,438–0,767 | 0,678 | 0,595–0,804 |
| simple_fusion | 0,450 | 0,458 | 0,366–0,713 | 0,608 | 0,523–0,742 |
| ecg_only | 0,359 | 0,369 | 0,302–0,620 | 0,508 | 0,462–0,584 |

Interval `proposed` sangat lebar (0,438–0,767) dan masih memuat 0,5. Dengan 6 subjek, CI ini
memang selebar itu; jangan dibaca sebagai presisi. Catatan teknis: bootstrap me-resample subjek
dan sengaja mengabaikan autokorelasi di dalam subjek, jadi lebarnya masih optimistis.

**Kesimpulan yang boleh diambil:**
- Kegagalan `proposed` di test adalah kegagalan **transport skor antar subjek**, bukan buta terhadap apnea.
- Mekanismenya terukur: korelasi prevalensi–skor −0,943, dengan subjek berprevalensi terendah
  mendapat skor tertinggi.
- Belum ada bukti bahwa cross-attention membantu; `resp_only` tetap yang terbaik dan paling stabil.
- Jangan pakai z-score per rekaman sebagai hasil utama. Kalau dipakai, sebut status post-hoc-nya.

Output: `artifacts/calibration/{calibration_summary.csv, calibration_by_subject.csv,
calibration_transforms.csv, calibration_bootstrap.csv, calibration_prevalence.csv,
calibration_run.json, console.log}`.

### Tahap 6c — Probe mekanisme offset per rekaman (`scripts/18_offset_probe.py`)

**Hipotesis yang diuji:** karena normalisasi dilakukan per **dataset**, bukan per rekaman, tiap
rekaman masih membawa offset sendiri. Offset itu konstanta tingkat-rekaman yang bisa dipakai model
sebagai pintasan, dan bisa berkorelasi dengan seberapa apneik rekaman tersebut.

**Hasil: hipotesis ini gugur.** Rata-rata (mean) ternyata nol untuk semua 39 record di kedua
modalitas (rentang −0,00 sampai 0,00; std 0,00). Sinyal sudah di-detrend di Tahap 3a, jadi komponen
DC tidak tersisa dan z-score per dataset memang menghapus selisih mean. Tidak ada offset rata-rata
yang bisa dijadikan pintasan.

| Yang diukur | Hasil |
|---|---|
| `ecg_mean_z` per record | 0,00 di semua 39 record |
| `resp_mean_z` per record | 0,00 (maks 0,0002) |
| `ecg_std_z` per record | 0,55 – 1,71 |
| `resp_std_z` per record | 0,55 – **5,92** |

**Yang justru ditemukan (dan ini cacat nyata): skala per rekaman tidak seragam, dan clipping ±5σ
tidak berperilaku sama di semua rekaman.** Karena std tiap rekaman berbeda jauh dari std dataset,
ambang 5σ memotong porsi sampel yang sangat berbeda:

| Record | `resp_std_z` | Porsi sampel respirasi ter-clip |
|---|---|---|
| slp02b | 5,92 | **10,4%** |
| slp02a | 5,26 | **8,0%** |
| slp03 | 2,47 | 3,7% |
| slp32 | 1,63 | 1,7% |
| slp41 | 1,13 | 1,5% |

ECG juga tidak sekata: rata-rata 1,16% sampel ter-clip, maksimum 4,07% (`ucddb008`).

**Koreksi klaim lama.** Catatan sebelumnya (batasan #6) menyebut "clipping c=5 hampir tidak aktif
(0,1% ECG)". Itu **salah** — dihitung ulang pada seluruh 39 record, rata-rata ECG ter-clip adalah
**1,16%** dan respirasi **0,69%**, dengan ekor sampai 10,4%. Angka 0,1% tampaknya berasal dari
sampel window yang tidak representatif.

**Batas klaim.** Uji korelasi fitur rekaman terhadap skor model dilakukan (n = 6–7 rekaman per
split, 11 fitur, tanpa koreksi multiple comparison). Tidak ada hasil yang layak disebut temuan:
kandidat terkuat hanya muncul di satu split atau satu model, dan tidak konsisten. Jadi **belum ada
bukti bahwa skala rekaman yang menyebabkan offset skor**. Yang pasti hanya: clipping tidak seragam,
dan `slp02a`/`slp02b` kehilangan ~8–10% sampel respirasi sebelum masuk model.

**Konsekuensi untuk langkah berikut:** perbaikan yang kini punya dasar empiris bukan normalisasi
mean per rekaman (tidak ada yang bisa diperbaiki, meannya sudah nol), melainkan **normalisasi skala
per rekaman** dan/atau clipping yang sadar-skala, diputuskan sebelum training ulang.

Output: `artifacts/offset_probe/{record_features.csv, offset_correlations.csv,
per_record_scaling.csv, offset_probe.json, console.log}`.

### Tahap 7a — Agregasi validasi multi-seed (`scripts/19_aggregate.py`)

**Tujuan:** memenuhi rencana paper Section 3.5.2 — beberapa seed dilaporkan sebagai mean ± SD,
bukan satu angka. Tiap seed adalah split subjek yang berbeda, jadi SD antar seed mengukur
sensitivitas kesimpulan terhadap subjek mana yang jatuh di validasi.

**Dijalankan 24 November 2026:** training penuh 4 model × 3 seed (42, 1337, 2024), 10 s,
seluruh window training, test tidak dibaca oleh skrip ini.

**Validasi, mean ± SD antar 3 seed (ROC-AUC):**

| Model | Val AUC | Rank tiap seed | Val F1 |
|---|---|---|---|
| resp_only | 0,7226 ± 0,1185 | 3 / 1 / 1 | 0,3543 ± 0,1476 |
| simple_fusion | 0,6894 ± 0,0471 | 1 / 2 / 2 | 0,3242 ± 0,1003 |
| proposed | 0,6369 ± 0,0153 | 2 / 3 / 3 | 0,3001 ± 0,0796 |
| ecg_only | 0,5513 ± 0,0419 | 4 / 4 / 4 | 0,2398 ± 0,0456 |

**Per dataset (val ROC-AUC, mean ± SD):**

| Model | UCDDB | MIT-BIH PSG |
|---|---|---|
| resp_only | 0,7438 ± 0,0801 | 0,5630 ± 0,1637 |
| simple_fusion | 0,7075 ± 0,1007 | 0,5367 ± 0,0726 |
| proposed | 0,7039 ± 0,0537 | 0,5146 ± 0,1304 |
| ecg_only | 0,5148 ± 0,1731 | 0,4201 ± 0,0871 |

**Yang terbukti:** peringkat model **tidak stabil antar seed di validasi** — `simple_fusion` juara
di seed 42, `resp_only` juara di seed 1337 & 2024. `proposed` konsisten peringkat 2–3 dan tidak
pernah menang. `ecg_only` selalu terakhir. SD `proposed` paling kecil (0,0153) — bukan karena bagus,
tapi karena selalu mediocre.

Catatan: seed 2024 lebih cepat overfit (val loss melonjak setelah epoch 1–3), jadi best epoch-nya
sangat awal (1–4) dan `simple_fusion`/`ecg_only` hanya jalan 4 epoch.

Output: `artifacts/aggregate/seed42-1337-2024_10s/{val_summary_mean_sd.csv,
val_by_dataset_mean_sd.csv, val_ranking.csv, val_per_seed.csv, aggregate_run.json}`.

### Tahap 7b — Prediksi test 3 seed + agregasi (`scripts/16_test.py`, `scripts/20_aggregate_test.py`)

**Prosedur:** tiap seed dibaca **sekali** dengan `frozen.json` seed tersebut (checkpoint, best epoch,
threshold tidak diubah). Tidak ada retraining, tidak ada tuning setelah melihat test. Bootstrap
dilakukan di level subjek (2.000 resample, 6 subjek per seed).

**Test, mean ± SD antar 3 seed:**

| Model | Test AUC | Test F1 | Sens | Spec |
|---|---|---|---|---|
| resp_only | **0,7424 ± 0,0083** | 0,4020 ± 0,0623 | 0,7331 ± 0,1156 | 0,6088 ± 0,1437 |
| simple_fusion | 0,5767 ± 0,1231 | 0,3030 ± 0,0902 | 0,5647 ± 0,0967 | 0,5402 ± 0,1240 |
| proposed | 0,5548 ± 0,0299 | 0,3023 ± 0,0732 | 0,4940 ± 0,1058 | 0,6365 ± 0,1180 |
| ecg_only | 0,4513 ± 0,1365 | 0,2311 ± 0,0397 | 0,6967 ± 0,2757 | 0,1666 ± 0,1741 |

**Per seed (test ROC-AUC):**

| Model | seed 42 | seed 1337 | seed 2024 |
|---|---|---|---|
| resp_only | 0,7345 | 0,7510 | 0,7417 |
| proposed | 0,5261 | 0,5524 | 0,5858 |
| simple_fusion | 0,4498 | 0,6957 | 0,5847 |
| ecg_only | 0,3590 | 0,6081 | 0,3867 |

**Per dataset (test ROC-AUC, mean ± SD):**

| Model | UCDDB | MIT-BIH PSG |
|---|---|---|
| resp_only | 0,7449 ± 0,0294 | 0,6656 ± 0,0497 |
| proposed | 0,7634 ± 0,0359 | 0,4567 ± 0,1062 |
| simple_fusion | 0,7020 ± 0,0077 | 0,4595 ± 0,0447 |
| ecg_only | 0,5419 ± 0,1140 | 0,4409 ± 0,0580 |

**Pooled vs per subjek (mean antar seed):**

| Model | Pooled AUC | Mean AUC subjek | Within-subject AUC |
|---|---|---|---|
| resp_only | 0,7424 ± 0,0083 | 0,7140 ± 0,0302 | 0,6887 ± 0,0267 |
| proposed | 0,5548 ± 0,0299 | 0,6730 ± 0,0518 | 0,6212 ± 0,0461 |
| simple_fusion | 0,5767 ± 0,1231 | 0,6755 ± 0,0227 | 0,6266 ± 0,0237 |
| ecg_only | 0,4513 ± 0,1365 | 0,5279 ± 0,0324 | 0,4975 ± 0,0287 |

**Bootstrap 95% CI per seed (subject-level, 6 subjek):**

| Model | seed 42 | seed 1337 | seed 2024 |
|---|---|---|---|
| resp_only | 0,629–0,810 | 0,651–0,810 | 0,638–0,802 |
| proposed | 0,438–0,767 | 0,414–0,780 | 0,407–0,810 |
| simple_fusion | 0,366–0,713 | 0,583–0,767 | 0,451–0,739 |
| ecg_only | 0,302–0,620 | 0,480–0,655 | 0,329–0,588 |

**Yang benar-benar terbukti (ini temuan utama tahap 7):**
- **`resp_only` adalah satu-satunya model yang konsisten dan stabil.** Test AUC 0,7345 / 0,7510 /
  0,7417 dengan SD hanya 0,0083 — jauh lebih kecil dari model lain. Ini bukan satu titik pengukuran
  yang kebetulan bagus; tiga split subjek berbeda memberi angka hampir sama.
- **`proposed` konsisten di bawah `resp_only` di ketiga seed** (0,526 / 0,552 / 0,586). Tidak ada
  satu seed pun di mana cross-attention unggul. Klaim manfaat cross-attention **tidak didukung**.
- **`simple_fusion` sangat tidak stabil** (0,450 → 0,696 → 0,585, SD 0,1231). Peringkatnya berubah
  drastis antar split, jadi angka satu seed untuk model ini tidak bisa dipercaya.
- **`ecg_only` praktis tidak berguna di test** (0,4513 ± 0,1365; dua seed di bawah 0,5).
  Sensitivitas 0,997 dengan spesifisitas 0,009 di seed 1337 — model memprediksi hampir semua
  window positif. Sensitivitas tinggi itu bukan prestasi.
- **Gap pooled-vs-per-subjek masih ada untuk `proposed`** (0,555 vs 0,673 mean subjek) dan bertahan
  di ketiga seed; `resp_only` satu-satunya yang gap-nya kecil (0,742 vs 0,714).
- **PSG tetap lemah dan berisiko.** `resp_only` PSG 0,666 ± 0,050 (terbaik), model lain ≤0,46.

**Yang harus diakui:** 6 subjek test per seed, N_eff tipis, dan CI masih selebar 0,3–0,4 poin.
SD antar seed untuk `proposed` (0,0299) lebih kecil dari lebar CI-nya, artinya perbedaan antar
seed tidak bisa dipisahkan dari noise subjek. Jangan menyimpulkan urutan model dari selisih kecil.

Output: `artifacts/test/seed{1337,2024}_10s/*` dan
`artifacts/aggregate/seed42-1337-2024_10s/{test_summary_mean_sd.csv,
test_by_dataset_mean_sd.csv, test_auc_decomposition_mean_sd.csv, test_bootstrap_by_seed.csv,
test_per_seed.csv}`.

### Tahap 7c — Eksperimen 2: window 30 s (`scripts/19_aggregate.py`, `scripts/20_aggregate_test.py`)

**Tujuan:** menjawab Eksperimen 2 paper — 10 s (reduced window utama) vs 30 s (pembanding).
Semua hal lain dijaga sama: 4 model, 3 seed, split subjek identik, arsitektur, budget, dan
protokol beku.

**Perangkat:** seed 42 dan 1337 di CPU; **seed 2024 di T4 GPU** (`device: cuda`, `workers: 2`,
via `notebooks/colab_train_30s.ipynb`). Perbedaan urutan operasi floating-point bisa menggeser
angka tipis; ini dicatat sebagai beda perangkat, bukan beda metode.

**Test ROC-AUC, mean ± SD antar 3 seed:**

| Model | 10 s | 30 s | Perubahan |
|---|---|---|---|
| resp_only | 0,7424 ± 0,0083 | **0,7465 ± 0,0486** | +0,004 |
| simple_fusion | 0,5767 ± 0,1231 | 0,6995 ± 0,0462 | **+0,123** |
| proposed | 0,5548 ± 0,0299 | 0,6777 ± 0,0551 | **+0,123** |
| ecg_only | 0,4513 ± 0,1365 | 0,4269 ± 0,0948 | −0,024 |

**Perubahan per seed — ini yang menentukan klaim apa yang boleh dibuat:**

| Model | seed 42 | seed 1337 | seed 2024 | Arah konsisten? |
|---|---|---|---|---|
| proposed | +0,109 | +0,106 | +0,154 | **ya, 3/3 naik** |
| simple_fusion | +0,213 | +0,056 | +0,100 | ya, tapi rentang 4× |
| resp_only | −0,040 | +0,004 | +0,049 | tidak (campur) |
| ecg_only | +0,048 | −0,078 | −0,043 | tidak (2/3 turun) |

**Validasi 30 s (mean ± SD):**

| Model | Val AUC | Val F1 | Best epoch rata-rata |
|---|---|---|---|
| simple_fusion | 0,7462 ± 0,0738 | 0,3848 | 3,3 |
| proposed | 0,7432 ± 0,1066 | 0,3777 | 3,7 |
| resp_only | 0,7274 ± 0,1802 | 0,3692 | 5,3 |
| ecg_only | 0,5883 ± 0,0572 | 0,2693 | 1,0 |

**Test per dataset 30 s:**

| Model | UCDDB | MIT-BIH PSG |
|---|---|---|
| simple_fusion | 0,8064 ± 0,0354 | 0,5810 ± 0,1181 |
| proposed | 0,8000 ± 0,0494 | 0,5443 ± 0,1549 |
| resp_only | 0,7660 ± 0,0775 | **0,6693 ± 0,0693** |
| ecg_only | 0,4883 ± 0,1444 | 0,3402 ± 0,0146 |

**Pooled vs per subjek 30 s:** proposed 0,678 vs 0,704 · resp_only 0,747 vs 0,712 ·
simple_fusion 0,700 vs 0,716 · ecg_only 0,427 vs 0,494.

**Yang benar-benar terbukti:**
- **Window 30 s secara sistematis membantu model fusion.** `proposed` naik di **ketiga** seed
  (+0,109 / +0,106 / +0,154). Ini satu-satunya model yang arahnya tidak pernah berbalik.
  `simple_fusion` juga naik di ketiga seed tetapi rentangnya lebar (+0,056 sampai +0,213).
- **`resp_only` tidak terpengaruh durasi** (−0,040 / +0,004 / +0,049 — praktis nol). Masuk akal:
  informasi respirasi ada di skala napas, tidak butuh konteks 30 s.
- **`ecg_only` memburuk** di 2 dari 3 seed. Tidak ada bukti ECG saja memanfaatkan window panjang.
- **Gap `resp_only` vs `proposed` menyempit tajam**: dari 0,188 (10 s) menjadi 0,069 (30 s).
  Mekanismenya konsisten dengan kebutuhan token: pada 10 s tiap cabang hanya punya 50 token,
  pada 30 s ada 150, sehingga cross-attention punya konteks untuk bekerja.
- **Overfit lebih cepat di 30 s**: best epoch `ecg_only` = 1 di ketiga seed, `proposed` 2–5,
  padahal budget epoch sama. Window lebih panjang memperbesar risiko hafal subjek.

**Yang harus diakui (jangan dibungkus):**
- **`proposed` tetap bukan model terbaik.** Di 30 s `resp_only` masih tertinggi
  (0,7465 vs 0,6777) dan SD-nya jauh lebih kecil (0,049 vs 0,055 dengan rata-rata lebih tinggi).
  Yang terjadi adalah *gap menyempit*, bukan *cross-attention menang*. Menulis "cross-attention
  lebih baik" dari angka ini adalah salah.
- **Offset antar subjek pada `proposed` masih ada** (pooled 0,678 vs per subjek 0,704), meski
  lebih kecil daripada 10 s (0,555 vs 0,673). Belum selesai.
- **Bootstrap per seed:** seed 42 (0,522–0,804) dan seed 1337 (0,486–0,808) masih memuat 0,5;
  seed 2024 (0,575–0,857) tidak lagi. Dengan 6 subjek, arah perbaikan lebih bisa dipercaya
  daripada besarnya.
- **PSG tetap lemah** untuk semua model kecuali `resp_only`. UCDDB ~0,80, PSG 0,34–0,67.

Output: `artifacts/train/seed{42,1337,2024}_30s/`, `artifacts/test/seed{42,1337,2024}_30s/`, dan
`artifacts/aggregate/seed42-1337-2024_30s/{val_summary_mean_sd.csv, test_summary_mean_sd.csv,
test_by_dataset_mean_sd.csv, test_auc_decomposition_mean_sd.csv, test_bootstrap_by_seed.csv,
test_per_seed.csv, val_ranking.csv, aggregate_run.json, test_aggregate_run.json}`.

### Tahap 7d — Eksperimen 3: ablation preprocessing (DIJALANKAN, TAPI BELUM LAYAK DIKUTIP)

**Dijalankan 25 November 2026 di Colab T4:** 5 arm × 2 model × 3 seed, window 30 s,
`device=cuda`, `workers=2`, 30/30 run lengkap dengan checkpoint.

**Kenapa hasil ini BELUM boleh dipakai sebagai hasil Eksperimen 3:**

Arm `baseline` seharusnya mereproduksi Eksperimen 2 — pipeline, data, split, arsitektur, dan
budget identik. Kenyataannya tidak:

| Model | seed | baseline (ablation) | Eksperimen 2 | selisih |
|---|---|---|---|---|
| proposed | 42 | 0,4707 | 0,6348 | **−0,164** |
| proposed | 1337 | 0,8879 | 0,6586 | **+0,229** |
| proposed | 2024 | 0,8050 | 0,7398 | +0,065 |
| resp_only | 42 | 0,7844 | 0,6944 | +0,090 |
| resp_only | 1337 | 0,7993 | 0,7545 | +0,045 |
| resp_only | 2024 | 0,8080 | 0,7905 | +0,018 |

Penyebabnya sudah dilacak, dan **bukan bug data**: saya bandingkan tensor yang dihasilkan
`AblationDataset(baseline)` dengan `WindowDataset` pada 200 window — perbedaan maksimum
**0,000000** di kedua modalitas, label sama, dan jumlah parameter identik (230.402 untuk
`proposed` 30 s di kedua jalur). Set test juga identik (4.663 baris, kunci dan label sama).

Yang berbeda adalah **hasil training**: val_loss epoch 1 `resp_only` seed 42 = 0,6059 di ablation
vs 0,5827 di Eksperimen 2, berjalan di perangkat dan jumlah worker berbeda. Latihan yang sama
sekali pun bisa berbeda karena (a) urutan batch, (b) non-determinisme kernel GPU, (c) titik
early-stop yang berbeda. Karena keputusan model diambil dari validation loss, selisih kecil di
epoch awal bisa menggeser epoch terpilih dan mengubah banyak hal.

Efeknya terlihat di tabel: `proposed` seed 42 memilih **epoch 1** sebagai terbaik (val 0,443) lalu
berhenti di epoch 4, sedangkan Eksperimen 2 memilih epoch 5 dari 8 epoch. Model berbeda, jadi
angka test-nya berbeda. Artinya selisih ablation-vs-baseline **tenggelam oleh variasi run**,
terutama karena validation hanya 6 subjek.

**Konsekuensi yang harus dinyatakan:** dengan `baseline` saja menyimpang sampai 0,23 poin,
selisih antar arm yang besarnya 0,01–0,09 **tidak bisa ditafsirkan**. Hasil ini hanya menunjukkan
bahwa protokol ablation belum stabil, bukan efek preprocessing.

**Bukti pendukung bahwa sinyalnya belum terpisah dari noise** (test AUC per seed):

| Model | arm | seed 42 | seed 1337 | seed 2024 |
|---|---|---|---|---|
| proposed | baseline | 0,471 | 0,888 | 0,805 |
| proposed | no_clip | 0,710 | 0,892 | 0,752 |
| proposed | clip_per_record | 0,751 | 0,896 | 0,761 |
| proposed | norm_per_record | 0,654 | 0,902 | 0,842 |
| proposed | no_filter | 0,698 | 0,895 | 0,849 |
| resp_only | baseline | 0,784 | 0,799 | 0,808 |
| resp_only | no_clip | 0,780 | 0,798 | 0,812 |
| resp_only | clip_per_record | 0,773 | 0,793 | 0,816 |
| resp_only | norm_per_record | 0,797 | 0,801 | 0,813 |
| resp_only | no_filter | 0,714 | 0,830 | 0,841 |

`resp_only` — model yang paling sensitif terhadap skala karena std per rekamannya 0,55–5,92 —
menunjukkan perubahan hanya ±0,01 untuk `no_clip` dan `clip_per_record` di semua seed. Itu
konsisten dengan "clipping hampir tidak berpengaruh pada model ini", tetapi juga konsisten dengan
"efeknya lebih kecil dari noise run". Dua tafsir itu belum bisa dipisahkan.

**Bug performa yang ditemukan dan diperbaiki (26 Nov 2026).** `AblationDataset` dipakai dengan
`cache_size` bawaan 8, sedangkan `WindowDataset` memakai 64. Akibatnya setiap epoch membaca ulang
file `.npy` dari disk: 112 detik/epoch di GPU untuk 605 step, padahal Eksperimen 2 di GPU yang
sama mencatat 14–24 detik/epoch. Sudah diperbaiki (`cache_size=64` di train/val/test). Ini tidak
mengubah angka yang sudah ada, tetapi membuat run ulang jauh lebih murah.

**Yang harus dilakukan sebelum hasil ini dipakai:**
1. Jalankan ulang dengan `cache_size` yang sudah diperbaiki.
2. Tetapkan protokol determinisme: `torch.use_deterministic_algorithms`, seed untuk DataLoader
   worker, dan `workers=0` bila konsistensi lebih penting daripada kecepatan.
3. Verifikasi ulang bahwa `baseline` mereproduksi Eksperimen 2 dalam toleransi yang dinyatakan
   (misalnya selisih test AUC < 0,02) **sebelum** menafsirkan arm lain.
4. Bila reproduksi tetap tidak tercapai, laporkan variabilitas run sebagai batas, dan naikkan
   jumlah seed alih-alih menafsirkan selisih kecil.

Output: `artifacts/ablation/seed42-1337-2024_30s/{ablation_val.csv, ablation_test.csv,
ablation_per_seed.csv, ablation_runs.csv, ablation_run.json, runs/<arm>_<model>_seed<seed>/}`.

**Definisi lima arm, masing-masing mengubah satu keputusan saja:**

| Arm | Yang berubah vs baseline | Cache sinyal |
|---|---|---|
| `baseline` | tidak ada (pipeline sekarang) | filtered |
| `no_clip` | clipping ±5σ dihapus | filtered |
| `clip_per_record` | batas clip diskalakan ke std rekaman itu sendiri | filtered |
| `norm_per_record` | z-score per rekaman, bukan per dataset | filtered |
| `no_filter` | band-pass/low-pass/detrend dihapus | unfiltered |

Arm `clip_per_record` dan `norm_per_record` dipilih karena temuan Tahap 6c: skala per
rekaman berkisar 0,55–5,92 dalam satuan std dataset, sehingga ambang 5σ tetap memotong
8–10% sampel respirasi di `slp02a/b` tetapi hampir tidak menyentuh rekaman lain.
Diverifikasi: 37 dari 78 entri (record × modalitas) mendapat batas lebih longgar, 40 lebih
ketat — jadi arm ini bukan duplikat `no_clip`.

**Cache unfiltered** (`scripts/22_make_unfiltered_cache.py`): dibuat dari data mentah
dengan resampling saja. Diverifikasi **0 mismatch panjang sampel** pada 39 record di kedua
modalitas, sinkronisasi 0 ms, sehingga window manifest tetap valid.

**Batasan yang wajib ikut pada `no_filter`:** `resample_poly` adalah anti-alias, dan itu
sendiri low-pass (ECG ~50 Hz, respirasi ~12,5 Hz). Jadi arm ini berarti "tanpa filter
tambahan", bukan "tanpa filter sama sekali". Jangan menulis klaim lebih kuat dari itu.

**Sudah diuji:** smoke test end-to-end kelima arm (1 model, 1 epoch, 512 window) berjalan,
menghitung validasi + test + agregasi. Itu membuktikan kode, bukan efek ablation.

**Bug resume yang ditemukan dan diperbaiki (25 Nov 2026).** Smoke test di Colab gagal dengan
`KeyError: 'state'`. Penyebabnya dua cacat di kode resume Tahap 5: (a) `save_resume` menulis
`best` tanpa kunci `state` (disimpan terpisah sebagai `best_state`), tetapi saat dibaca kode
memeriksa `best.get("state")`, yang selalu `None`, sehingga checkpoint terbaik tidak pernah
dipulihkan; (b) kalau `next_epoch >= max_epochs`, loop epoch tidak jalan sama sekali dan
`best["state"]` tetap `None`. Perbaikan: `best` direkonstruksi eksplisit dari `best_state`,
state scheduler ikut disimpan (menghilangkan warning `lr_scheduler.step()`), dan run yang
sudah selesai **di-skip** alih-alih dilatih ulang — supaya sweep Colab yang terputus lalu
dijalankan ulang tidak membuang GPU. Diuji tiga skenario: run selesai lalu diulang (skip),
resume di tengah (lanjut dari epoch benar, best epoch dipertahankan), dan run normal (tidak
berubah).

**Cara menjalankan:** lokal cepat untuk cek konfigurasi, atau
`notebooks/colab_ablation_30s.ipynb` untuk run penuh. Dengan `cache_size` yang sudah diperbaiki,
waktu GPU kembali normal (14–24 detik/epoch, bukan 112 detik).

**Bug kedua: `--out` di luar repo (25 Nov 2026).** Smoke test Colab kedua gagal dengan
`ValueError: ... is not in the subpath of '/content/Research_v2'` karena skrip memaksa
`path.relative_to(ROOT)` saat mencatat path checkpoint, sedangkan `--out /content/smoke_ablation`
berada di luar proyek. Diperbaiki dengan `rel_or_abs()` (relatif bila memungkinkan, absolut bila
tidak) dan `resolve_recorded()` untuk membacanya kembali. `resolve_recorded()` perlu karena
`ROOT / "/abs/path"` hanya menghasilkan path absolut di POSIX; di Windows huruf drive-nya hilang,
jadi path absolut kini dipakai apa adanya. Diuji: output di luar repo dengan dan tanpa pembacaan
test, serta reproduksi angka seed 42 30 s yang sudah di-commit dengan nilai identik.

---

### Pemeriksaan tambahan yang sudah dijalankan

| Skrip | Yang dibuktikan |
|---|---|
| `05_qc_resp.py` | SNR pita napas median 92,5 dB (UCDDB), 80,2 dB (PSG); laju napas median 14,3 dan 12,5 /menit → sinyal respirasi informatif, bukan noise |
| `06_qc_flow.py` | Menemukan `ucddb005` sebagai satu-satunya record dengan kanal `Flow` rusak |
| `09_verify_windowing.py` | 0 window melewati batas sinyal; atribusi label cocok dengan anotasi sumber; stride sesuai policy |
| `11_sample_size.py` | Menghitung jumlah window, subjek, dan ekuivalen non-overlap per split |
| `12_effective_n.py` | Menghitung N_eff dari autokorelasi label |
| `17_calibration.py` | Memisahkan gap pooled-vs-per-subjek menjadi sifat subjek test, menunjukkan korelasi prevalensi–skor −0,943 di test tetapi hanya +0,029 di validasi, dan menguji z-score per rekaman (merugikan validasi) |
| `18_offset_probe.py` | Menggugurkan hipotesis offset mean per rekaman (mean_z = 0,00 di 39 record); menemukan skala per rekaman 0,55–5,92 dan clipping 5σ tidak seragam (sampai 10,4%) |
| `10_report.py` | Merender `reports/stage_report.html` dari artefak |

---

## 5. Batas yang harus diakui (jangan disembunyikan)

1. **Unit evaluasi tipis, bukan jumlah window.** Test hanya 6 subjek (4 UCDDB + 2 PSG).
   13.993 window test bukan 13.993 sampel independen: autokorelasi label memberi N_eff
   ≈1.875 (UCDDB) dan ≈110 (PSG). Episode apnea di test ≈583 runs.
2. **Sebaran positif tidak rata.** Contoh seed 42: subjek UCDDB-22 hanya 1,9% window positif,
   subjek PSG-16 42,5%. Metrik test sangat bergantung pada 1–2 subjek.
3. **Label PSG kasar.** Epoch 30 s diwariskan ke window 10 s → weak label. Dilaporkan sebagai
   label noise, tidak dikoreksi.
4. **Ambang 50% sensitif di UCDDB 30 s.** ~10 ribu window punya overlap 0,4–0,6 (puncak di
   tepat 0,5). Hypopnea 13–16 s sering menghasilkan ambang batas.
5. **LODO hanya 2 arah** (UCDDB↔PSG), karena dataset ketiga tidak memenuhi syarat kanal.
6. **Clipping c=5 tidak seragam, dan klaim lama soal ini salah.** Dihitung ulang pada 39 record,
   rata-rata 1,16% sampel ECG dan 0,69% respirasi ter-clip — bukan 0,1% seperti tertulis sebelumnya.
   Puncaknya `slp02b` 10,4% dan `slp02a` 8,0% sampel respirasi. Penyebabnya skala per rekaman
   berbeda jauh dari skala dataset (`resp_std_z` 0,55–5,92), jadi ambang 5σ memotong sangat berbeda
   di tiap rekaman. Ablation clipping kini justru bermakna, tapi harus diputuskan sebelum training
   ulang.
7. **Respirasi PSG tidak terkalibrasi** (amplitudo kecil, satuan tak bermakna fisiologis).
   z-score menanganinya untuk model, tapi arti amplitudo tidak boleh diinterpretasi klinis.
8. **Kalibrasi antar subjek belum beres.** Data penuh menaikkan AUC gabungan (proposed 0,47 → 0,64),
   tetapi varian per subjek masih lebar (0,49–0,87) dan AUC dataset PSG di bawah 0,5. Dengan 6 subjek
   validasi, angka gabungan tidak boleh dilaporkan sebagai performa; laporkan per dataset dan per subjek.
9. **`proposed` belum unggul atas baseline.** Validasi seed 42: AUC 0,644 vs `simple_fusion` 0,693.
   Test seed 42 malah menunjukkan `resp_only` terbaik (0,734 vs `proposed` 0,526). Jangan klaim
   cross-attention lebih baik sebelum ada bukti multi-seed; saat ini tidak ada bukti sama sekali.
10. **Kalibrasi lintas subjek belum beres; mekanismenya sebagian terukur, jangan dilebihkan.** Gap
    pooled vs per subjek pada `proposed`: 0,044 di validasi menjadi 0,207 di test. Di test
    korelasi prevalensi–skor −0,943 (p=0,005, tahan leave-one-out), dengan subjek 22 (prevalensi 1,9%)
    mendapat skor tertinggi 0,450. **Tetapi di validasi korelasi itu hanya +0,029**, jadi ini sifat
    subjek test, bukan sifat model yang stabil. Bukti yang lebih kuat justru `slp41` di validasi:
    prevalensi 0%, skor rata-rata 0,504 (tertinggi) — subjek tanpa window positif tidak punya AUC
    sehingga luput dari semua tabel. Z-score per rekaman menutup gap di test tetapi merugikan validasi
    (0,644 → 0,601), jadi itu penyelamat post-hoc, bukan perbaikan berprinsip.
11. **Jumlah window bukan jumlah sampel independen.** Test 30 s punya ~4.600 window, 10 s ~14.000,
    tetapi N_eff-nya ≈2.000 (UCDDB) dan ≈110 (PSG) setelah koreksi autokorelasi label, dengan
    ~583 episode apnea berurutan. Jangan membaca jumlah window sebagai ukuran sampel.
12. **Multi-seed menegaskan `proposed` tidak unggul di 10 s, tapi gap menyempit di 30 s.** Di 10 s,
    `proposed` kalah dari `resp_only` di **ketiga** seed (0,555 vs 0,742). Di 30 s, `proposed` naik
    di ketiga seed (+0,109 / +0,106 / +0,154) sehingga gap menyempit dari 0,188 ke 0,069 — tetapi
    **`resp_only` masih tertinggi** (0,7465 vs 0,6777) dengan SD lebih kecil. Ini perbaikan yang
    konsisten, bukan kemenangan. `simple_fusion` di 10 s sangat tidak stabil (SD 0,123) dan membaik
    di 30 s (SD 0,046), jadi angka satu durasi untuk model itu menyesatkan.
13. **SD antar seed bukan variance run-ulang.** Tiap seed adalah split subjek berbeda, jadi SD
    mengukur sensitivitas ke komposisi split. Dengan 6 subjek test, SD `proposed` di 10 s (0,030)
    lebih kecil dari lebar CI-nya, sehingga urutan model dari selisih kecil tidak bermakna.
14. **30 s sudah dijalankan, tapi dengan beda perangkat.** Seed 42 dan 1337 di CPU, seed 2024 di
    T4 GPU. Urutan operasi floating-point GPU berbeda, jadi angka seed 2024 tidak sepenuhnya
    sebanding bit-per-bit dengan dua seed lain; ini dicatat sebagai beda perangkat. Selain itu,
    window 30 s overfit lebih cepat (best epoch `ecg_only` = 1 di ketiga seed) di bawah budget
    epoch yang sama, sehingga perbandingan durasi juga tercampur perbedaan jumlah epoch efektif.
15. **Variasi run lebih besar dari efek ablation — ini yang membatalkan Tahap 7d.** Arm
    `baseline` ablation (pipeline identik dengan Eksperimen 2) menghasilkan test AUC yang
    menyimpang **−0,164 sampai +0,229** dari Eksperimen 2. Data loader sudah terbukti identik
    (selisih tensor 0,000000) dan jumlah parameter sama, jadi penyebabnya adalah variasi
    training: urutan batch, non-determinisme GPU, dan titik early-stop. Karena validation hanya
    6 subjek, epoch terbaik yang terpilih bisa berbeda jauh (seed 42: epoch 1 vs epoch 5), dan
    model yang berbeda menghasilkan angka test berbeda. **Konsekuensinya: selisih antar arm
    ablation yang besarnya 0,01–0,09 tidak boleh ditafsirkan sampai reproduksi baseline
    ditegakkan.** Ini bukan alasan menyembunyikan hasil — hasilnya dilaporkan sebagai "protokol
    belum stabil", bukan sebagai efek preprocessing.

---

## 6. Yang belum dikerjakan

| Tahap | Isi | Status |
|---|---|---|
| 4 | Model pilot: proposed (dual Transformer + bidirectional cross-attention) + 3 baseline, 1 seed, 10 s, budget kecil | **selesai** — pipeline hidup, tetapi belum generalize (lihat Tahap 4) |
| 5 | Training penuh + freeze checkpoint/threshold | **selesai** — seed 42, 10 s, 4 model, test belum disentuh |
| 6 | Prediksi test dengan model beku + 6 metrik | **selesai** — seed 42, 10 s, test dibaca sekali (lihat Tahap 6) |
| 7 | Analisis komparatif: 4 eksperimen, multi-seed, bootstrap subjek, ablation | **sebagian** — Eksperimen 1 dan 2 selesai (3 seed × 2 durasi). Eksperimen 3 dijalankan tapi protokolnya belum stabil sehingga belum dapat ditafsirkan (lihat Tahap 7d). Eksperimen 4 belum |
| opsional | Lengan ECG-only Apnea-ECG terpisah (disetujui dosen) | belum |

---

## 7. Jawaban siap pakai kalau ditanya

**"Kenapa hanya 2 dataset?"**
> Apnea-ECG punya ECG (yang memang kita butuhkan), tetapi tidak punya kanal respirasi —
> 70 record, semuanya 1 kanal ECG 100 Hz. Model multimodal menerima ECG **dan** respirasi untuk
> window yang sama (Sec. 3.4.2), jadi satu modalitas saja tidak cukup. Paper menyerahkan
> kelayakan ini ke audit (Sec. 3.3). Ini hasil audit, bukan perubahan metode.

**"Sudah tahu tidak punya respirasi, kenapa dimasukkan?"**
> Paper tidak menyatakan dataset mana pun memenuhi syarat; yang tertulis "considered" dan
> "availability is not assumed but will be verified". Mengaudit kandidat adalah prosedur yang
> paper tetapkan. Yang salah adalah memakai tanpa audit atau membuang tanpa bukti.

**"Berarti Experiment 4 berubah?"**
> Konfigurasi (ii) ditulis "across all datasets" dengan syarat audit. Dengan dua dataset yang
> lolos, klaimnya menyempit ke dua dataset dan kami laporkan sebagai batasan.

**"Kenapa slp01, slp61, ucddb005 dibuang?"**
> Policy strict airflow: slp01 hanya effort sum, slp61 hanya abdominal effort, jadi tidak sejenis
> dengan airflow. ucddb005 punya kanal Flow yang mati (std 0,5 vs 4,2–6,0) sehingga tidak bisa
> dipakai sebagai airflow. Semua tercatat per record di manifest audit.

**"Datanya cukup?"**
> Untuk training cukup: 39 record, ±241 jam, 116 ribu window 10 s dari 26 subjek training, dan
> 25 dari 26 subjek punya contoh apnea. Yang tipis adalah unit evaluasi: 6 subjek test, dengan
> N_eff ≈2.000 (UCDDB) dan ≈110 (PSG) setelah koreksi autokorelasi. Karena itu bootstrap dilakukan
> di level subjek, dan metrik akan dilaporkan per subjek juga.

**"Kenapa normalisasi tidak per rekaman seperti di paper?"**
> Paper menulis dua hal yang bertentangan: "each recording is z-score normalized per channel"
> dan "statistics are computed from the training data only". Kami memilih bacaan kedua agar tidak
> ada statistik test yang bocor ke training, dan mengelompokkannya per dataset karena skala
> UCDDB dan PSG berbeda ~22×. Perbedaan bacaan ini dicatat sebagai keputusan, bukan diabaikan.

**"Jadi cross-attention-nya berhasil atau tidak?"**
> Jawabannya tergantung durasi window, dan ini justru temuan Eksperimen 2. Di **10 s**: tidak.
> `proposed` mendapat test AUC 0,526 / 0,552 / 0,586, kalah dari `resp_only` (0,735 / 0,751 / 0,742)
> di ketiga seed. Di **30 s**: `proposed` naik konsisten di ketiga seed — 0,635 / 0,659 / 0,740,
> yaitu +0,109 / +0,106 / +0,154 — sehingga rata-ratanya 0,678 dan gap ke `resp_only` menyempit dari
> 0,188 menjadi 0,069. Yang perlu ditegaskan: itu perbaikan yang konsisten dan masuk akal (pada 10 s
> tiap cabang hanya punya 50 token, pada 30 s ada 150 sehingga ada konteks untuk dipertemukan), tapi
> `resp_only` **masih yang tertinggi dan paling stabil**. Jadi bukan "cross-attention menang",
> melainkan "cross-attention baru mulai berguna ketika window cukup panjang, dan belum menyalip
> baseline".

**"Kenapa `simple_fusion` angkanya naik-turun?"**
> Karena split subjeknya berubah antar seed, dan efeknya paling besar saat window pendek. Di 10 s
> test AUC-nya 0,450 / 0,696 / 0,585 (SD 0,123); di 30 s 0,663 / 0,751 / 0,685 (SD 0,046). Jadi
> ketidakstabilannya sebagian memang hilang ketika window diperpanjang — masuk akal, karena model
> fusion butuh token untuk bekerja. Tetap saja, satu seed untuk model ini menyesatkan; itulah alasan
> paper meminta mean ± SD beberapa seed.

**"Jadi 10 s atau 30 s yang lebih baik?"**
> Tidak ada satu jawaban untuk semua model, dan itu hasilnya. Pada 30 s, `proposed` naik di ketiga
> seed (+0,109 / +0,106 / +0,154) dan `simple_fusion` juga naik meski tidak sekonsisten itu;
> `resp_only` praktis tidak berubah (−0,040 / +0,004 / +0,049); `ecg_only` malah turun di 2 dari 3
> seed. Jadi window panjang menolong model yang **menggabungkan dua modalitas**, dan tidak menolong
> model satu modalitas. Untuk klaim "reduced window 10 s cukup", jawaban jujurnya: cukup untuk
> `resp_only` (0,742 vs 0,747 — selisih dalam noise), tidak cukup untuk model fusion (0,555 vs 0,678).
> Ini juga berarti klaim paper bahwa window pendek mempertahankan akurasi harus dibatasi per model,
> bukan dinyatakan umum.

**"Kenapa `resp_only` bisa terbaik, padahal modelnya multimodal?"**
> Itu memang hasil yang keluar, bukan yang kami rencanakan. Ada dua pembacaan yang jujur. Pertama,
> sinyal respirasi mungkin memang membawa informasi apnea paling langsung pada dataset ini, dan
> ECG 10 detik lebih banyak memuat variasi antar-subjek yang tidak relevan. Kedua, jalur fusion
> yang kami pakai masih gagal menstransfer skor antar subjek — terbukti dari gap pooled AUC vs
> rata-rata AUC per subjek pada `proposed` (0,555 vs 0,673). Jadi pertanyaannya bukan "respirasi
> lebih baik dari multimodal", melainkan "fusion kami belum berhasil memakai kedua modalitas
> sebaik respirasi sendirian". Ini yang harus ditulis sebagai temuan dan batasan, dan menjadi
> dasar pekerjaan lanjutan, bukan disembunyikan.

---

## 8. Cara memeriksa sendiri

```powershell
# tampilan visual semua tahap
start reports\stage_report.html

# angka mentah
python -c "import pandas as pd; print(pd.read_csv('artifacts/audit/audit_manifest.csv').head())"
python -c "import pandas as pd; print(pd.read_csv('artifacts/splits/split_manifest.csv').head())"

# jalankan ulang tahap
python scripts\01_audit.py
python scripts\02_split_windows.py
python scripts\04_preprocess.py
python scripts\07_normalize.py
python scripts\08_dataset.py 42 10
python scripts\15_train.py --models proposed,ecg_only,resp_only,simple_fusion
python scripts\16_test.py
python scripts\10_report.py
```

Peta file lengkap ada di `reports/stage_report.html` bagian **File map**.
