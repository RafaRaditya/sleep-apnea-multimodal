# HANDOFF — Sleep Apnea Multimodal

Dokumen ini untuk orang yang melanjutkan pekerjaan ini. Bukan pengganti log; ini pintu masuknya.
Ditulis 24 November 2026.

---

## 1. Baca ini dulu, dalam urutan ini

| # | Berkas | Untuk apa | Perkiraan waktu baca |
|---|---|---|---|
| 1 | `Konteks_Penelitian_Sleep_Apnea.md` | Aturan acuan: apa yang tertulis di paper vs TBD vs hasil aktual. Ini yang mencegah kamu mengklaim hal yang tidak dikerjakan. | 25 menit |
| 2 | `paper RM (1).pdf` | Sumber primer. Bagian penting: Sec. 3.1 (4 eksperimen), 3.2 (flowchart), 3.4 (metode), 3.5 (evaluasi). | 30 menit |
| 3 | `LOG_KERJA.md` | Apa yang benar-benar dijalankan, angka aktual, alasan tiap keputusan. Bagian 3 (penjelasan tiap tahap) dan 5 (batasan) paling penting. | 45 menit |
| 4 | `reports/stage_report.html` | Tampilan visual semua tahap. Buka di browser. | 10 menit |
| 5 | `config/policy.json`, `config/model.json` | Keputusan eligibility/exclusion dan hyperparameter. | 5 menit |

**Jangan** memakai draft lama, slide, atau chat sebelumnya sebagai sumber metode. `Konteks_Penelitian_Sleep_Apnea.md` §15 mendaftar hal-hal yang sudah digantikan paper UTS.

---

## 2. Status: di mana kita berhenti

### Selesai

| Tahap | Isi | Bukti |
|---|---|---|
| 0 | Verifikasi SHA-256 dataset | `artifacts/verify/` |
| 1 | Audit record & anotasi | `artifacts/audit/` |
| 2 | Split subjek (3 seed) + windowing 10/30 s | `artifacts/splits/`, `artifacts/windows/` |
| 3 | Preprocessing, normalisasi, loader | `artifacts/signals/`, `artifacts/normalization/` |
| 4 | Pilot 4 model | `artifacts/pilot/` |
| 5 | Training penuh + freeze, 4 model × 3 seed × 2 durasi | `artifacts/train/seed{42,1337,2024}_{10,30}s/` |
| 6 | Test beku, 3 seed × 2 durasi | `artifacts/test/seed{42,1337,2024}_{10,30}s/` |
| 6b/6c | Diagnostik kalibrasi & offset per rekaman | `artifacts/calibration/`, `artifacts/offset_probe/` |
| **7a/7b** | **Eksperimen 1: agregasi multi-seed, validasi + test, 10 s** | `artifacts/aggregate/seed42-1337-2024_10s/` |
| **7c** | **Eksperimen 2: window 30 s, 3 seed, agregasi** | `artifacts/aggregate/seed42-1337-2024_30s/` |
| **7d** | **Eksperimen 3: ablation preprocessing, 30 run di T4** | `artifacts/ablation/seed42-1337-2024_30s/` — **dijalankan, belum dapat ditafsirkan** |

### Belum dikerjakan (ini pekerjaan berikutnya)

| Eksperimen paper | Status | Catatan |
|---|---|---|
| **Eksperimen 3** — ablation preprocessing | **dijalankan, perlu diulang** | Arm `baseline` menyimpang −0,164…+0,229 dari Eksperimen 2 padahal pipeline-nya identik. Data loader terbukti sama (selisih tensor 0,000000), jadi penyebabnya variasi run. Perbaiki determinisme dulu, lalu ulangi. Rincian di `LOG_KERJA.md` Tahap 7d |
| **Eksperimen 4** — single- vs combined-dataset training, LODO 2 arah | belum | Perlu kohort pembanding yang didefinisikan dulu |
| Lengan ECG-only Apnea-ECG | belum | Disetujui dosen, terpisah dari multimodal |

### Persis di titik mana kita berhenti

Eksperimen 1 dan 2 selesai dan terverifikasi (3 seed × 2 durasi, agregasi lengkap).
Eksperimen 3 sudah **dijalankan penuh** (30 run, T4 GPU) tetapi **hasilnya belum dapat
ditafsirkan**: arm `baseline`, yang pipeline-nya identik dengan Eksperimen 2, menyimpang
−0,164…+0,229 di test AUC. Data loader sudah diverifikasi identik sampai 0,000000, jadi ini
variasi training, bukan bug data. Sebelum mengutip angka ablation, tegakkan determinisme dan
buktikan `baseline` mereproduksi Eksperimen 2.

---

## 3. Hasil kunci yang harus kamu tahu (jangan salah mengutip)

**Eksperimen 1 (3 seed, window 10 s, mean ± SD):**

| Model | Validasi AUC | Test AUC |
|---|---|---|
| resp_only | 0,7226 ± 0,1185 | **0,7424 ± 0,0083** |
| simple_fusion | 0,6894 ± 0,0471 | 0,5767 ± 0,1231 |
| proposed | 0,6369 ± 0,0153 | 0,5548 ± 0,0299 |
| ecg_only | 0,5513 ± 0,0419 | 0,4513 ± 0,1365 |

**Eksperimen 2 (3 seed, perbandingan 10 s vs 30 s, test AUC):**

| Model | 10 s | 30 s | Perubahan |
|---|---|---|---|
| resp_only | 0,7424 ± 0,0083 | **0,7465 ± 0,0486** | +0,004 |
| simple_fusion | 0,5767 ± 0,1231 | 0,6995 ± 0,0462 | +0,123 |
| proposed | 0,5548 ± 0,0299 | 0,6777 ± 0,0551 | +0,123 |
| ecg_only | 0,4513 ± 0,1365 | 0,4269 ± 0,0948 | −0,024 |

Empat hal yang **harus** dilaporkan apa adanya:

1. **Window 30 s menolong model fusion, bukan model satu modalitas.** `proposed` naik di
   ketiga seed (+0,109 / +0,106 / +0,154); `resp_only` praktis tidak berubah; `ecg_only` turun.
2. **`proposed` tetap bukan model terbaik.** Gap ke `resp_only` menyempit dari 0,188 ke 0,069,
   tetapi `resp_only` masih tertinggi dan jauh lebih stabil. Jangan menulis "cross-attention menang".
3. **`simple_fusion` sangat tidak stabil di 10 s** (0,450 / 0,696 / 0,585) dan membaik di 30 s
   (SD 0,123 → 0,046). Satu durasi untuk model ini menyesatkan.
4. **Kalibrasi antar subjek belum beres.** `proposed` 30 s: pooled 0,678 vs mean per subjek 0,704;
   dua dari tiga bootstrap seed masih memuat 0,5.

Detail lengkap + tabel per dataset + bootstrap CI: `LOG_KERJA.md` Tahap 7a/7b/7c.

**Eksperimen 3 (ablation) — ADA ANGKANYA, JANGAN DIKUTIP DULU.**

30 run selesai di T4 (5 arm × 2 model × 3 seed, 30 s). Tapi arm `baseline`, yang seharusnya
identik dengan Eksperimen 2, memberi test AUC yang berbeda:

| Model | seed | baseline (ablation) | Eksperimen 2 | selisih |
|---|---|---|---|---|
| proposed | 42 | 0,471 | 0,635 | −0,164 |
| proposed | 1337 | 0,888 | 0,659 | +0,229 |
| resp_only | 42 | 0,784 | 0,694 | +0,090 |

Penyebabnya sudah diperiksa dan **bukan bug data**: loader menghasilkan tensor identik
(selisih maksimum 0,000000 di 200 window), parameter sama, set test sama. Yang berbeda adalah
hasil training — urutan batch, non-determinisme GPU, dan epoch terbaik yang terpilih berbeda
(seed 42 memilih epoch 1, bukan epoch 5). Dengan validation hanya 6 subjek, itu cukup untuk
menggeser angka test jauh. **Artinya variasi run (sampai 0,23) lebih besar daripada efek
ablation (0,01–0,09), sehingga selisih antar arm belum bisa ditafsirkan.**

Rincian + tabel per seed: `LOG_KERJA.md` Tahap 7d.

---

## 4. Aturan yang tidak boleh dilanggar

1. **Test dibaca sekali per seed.** Threshold dan checkpoint dibekukan di `frozen.json`
   **sebelum** test disentuh. Jangan tuning setelah melihat test.
2. **Split di level subjek.** Tidak boleh ada window dari subjek yang sama di dua split.
3. **Statistik normalisasi hanya dari training.** Bukan dari val/test.
4. **Jangan mengarang angka.** Kalau belum dijalankan, tulis "belum".
5. **Jangan menambah dataset, modalitas, baseline, atau loss utama** tanpa arahan lingkup.
   SHHS/MESA/OSASUD/SpO₂ bukan bagian eksperimen utama paper UTS.
6. **Bedakan RENCANA vs HASIL.** Paper UTS belum punya Results; semua angka kita adalah
   eksekusi kelompok, dan hanya boleh diklaim sebatas yang terukur.

---

## 5. Cara menjalankan

Semua skrip di `scripts/`, dijalankan dari root proyek, Python 3.12.

**CPU (default, laptop):** tidak ada argumen tambahan, `--device auto` akan memilih CPU.

```powershell
# Training penuh (per seed, per durasi)
python scripts\15_train.py --models proposed,simple_fusion,ecg_only,resp_only --seed 42 --duration 30

# Test prediksi dengan model beku (setelah training selesai)
python scripts\16_test.py --seed 42 --duration 30

# Agregasi antar seed (butuh semua seed selesai)
python scripts\19_aggregate.py --seeds 42,1337,2024 --duration 30
python scripts\20_aggregate_test.py --seeds 42,1337,2024 --duration 30

# Render ulang laporan HTML
python scripts\10_report.py
```

**GPU (Colab):** dua notebook tersedia.

| Notebook | Untuk | Zip yang dibutuhkan |
|---|---|---|
| `notebooks/colab_train_30s.ipynb` | training seed 2024 30 s (sudah selesai) | 400 MB |
| `notebooks/colab_ablation_30s.ipynb` | Eksperimen 3, 5 arm ablation | **786 MB** (butuh `signals_unfiltered/`) |

Zip dibuat oleh `python scripts\21_make_cache_zip.py`, berisi scripts + config + kedua cache
sinyal + manifest window + norm stats. Dataset mentah tidak perlu di Colab.

**Eksperimen 3 (ablation) — kode sudah siap, belum dijalankan:**

```powershell
# lokal, cepat (1 seed, subset kecil) untuk cek konfigurasi
python scripts\23_run_ablation.py --seeds 42 --models resp_only --duration 30 --max-epochs 1 --train-cap 512 --val-only

# penuh (Colab, ±80 menit di T4)
python scripts\23_run_ablation.py --seeds 42,1337,2024 --models proposed,resp_only --duration 30 --device cuda --workers 2
```

Lima arm: `baseline`, `no_clip`, `clip_per_record`, `norm_per_record`, `no_filter`
(definisi di `scripts/23_ablation.py`). Cache unfiltered dibuat oleh
`scripts/22_make_unfiltered_cache.py`; sudah diverifikasi 0 mismatch panjang sampel di 39
record, jadi keselarasan window terjaga.

**Batasan yang wajib ikut pada arm `no_filter`:** cache unfiltered tetap memakai
`resample_poly`, dan anti-alias itu sendiri adalah low-pass (ECG ~50 Hz, respirasi ~12.5 Hz).
Jadi arm ini berarti "tanpa filter tambahan", bukan "tanpa filter sama sekali".

**Resume:** kalau sesi Colab terputus, jalankan perintah yang sama lagi. Skrip membaca
`resume_<model>.pt` di direktori output dan melanjutkan dari epoch terakhir yang selesai,
termasuk state optimizer dan posisi learning-rate scheduler.

**Catatan perangkat:** hasil GPU bisa berbeda tipis dari CPU karena urutan operasi
floating-point. Catat di log seed mana yang jalan di perangkat apa.

**Fakta mesin (laptop):** CPU 8 core, RAM bebas ~2 GB. Sinyal di-stream dari cache, bukan
dimuat semua. Satu `ecg_only` pernah 13.939 s/epoch karena mesin di-suspend — itu anomali,
bukan pola. Jalankan `python scripts\keep_awake.py` untuk menahan Modern Standby selama
training panjang di laptop.

---

## 6. Biaya komputasi terukur (bukan tebakan)

Benchmark nyata, `ms/window` training:

| Model | 10 s | 30 s | rasio |
|---|---|---|---|
| proposed | 6,0 | 33,2 | 5,5× |
| simple_fusion | 4,2 | 21,4 | 5,1× |
| resp_only | 2,2 | 10,7 | 4,9× |
| ecg_only | 2,2 | 10,9 | 5,0× |

Jumlah train window 30 s per seed ≈ 38.700 (10 s ≈ 116.500).
Proyeksi epoch 30 s: proposed ±22 menit, simple_fusion ±14, resp_only/ecg_only ±7.

Total 3 seed × 4 model 30 s: **±15 jam (epoch sama seperti 10 s) sampai ±30 jam (worst case)**.
Total 10 s yang sudah selesai: ±11 jam.

---

## 7. Keputusan yang masih terbuka

| # | Pertanyaan | Kenapa penting |
|---|---|---|
| 1 | ~~Berapa seed untuk 30 s~~ | **Terjawab** — 3 seed, 4 model, selesai (Tahap 7c) |
| 2 | ~~Apakah preprocessing diperbaiki dulu~~ | **Terjawab sebagian** — ablation dijalankan, tapi protokolnya belum stabil; perbaiki determinisme dulu |
| 3 | Kohort pembanding untuk Eksperimen 4 | Tanpa ini, klaim efek harmonisasi tidak bisa dipisahkan dari perubahan komposisi cohort |
| 4 | Apakah ablation pakai arsitektur `proposed` saja atau juga baseline | **Terjawab** — `proposed` + `resp_only` dipakai |
| 5 | Durasi mana yang jadi konfigurasi utama untuk Eksperimen 3 dan 4 | **Terjawab** — 30 s, karena 10 s meremehkan model fusion |
| 6 | **Determinisme training** — sekarang yang paling mendesak | Tanpa ini, hasil ablation tidak bisa ditafsirkan. Opsi: `torch.use_deterministic_algorithms(True)`, seed untuk DataLoader worker, dan/atau `workers=0`. Setelah itu, buktikan `baseline` mereproduksi Eksperimen 2 (mis. selisih test AUC < 0,02) sebelum menafsirkan arm lain |

**Rekomendasi saya (belum dijalankan):** jangan menambah arm baru dulu. Urutannya:
1. Tegakkan determinisme (lihat #6) dan buktikan `baseline` mereproduksi Eksperimen 2.
2. Setelah itu baru jalankan ulang 5 arm yang sudah ada — kodenya sudah siap, cukup
   `notebooks/colab_ablation_30s.ipynb`.
3. Kalau reproduksi tetap tidak tercapai meski deterministik, laporkan variabilitas run
   sebagai batas metodologis, dan naikkan jumlah seed alih-alih menafsirkan selisih kecil.

Catatan: `cache_size=8` di loader ablation sudah diperbaiki menjadi 64, jadi run ulang akan
jauh lebih cepat (sebelumnya 112 detik/epoch di GPU, seharusnya 14–24 detik).

---

## 8. Jebakan yang sudah pernah menggigit

1. **Klaim clipping 0,1% itu salah.** Setelah dihitung ulang di 39 record: rata-rata 1,16% (ECG)
   dan 0,69% (respirasi) ter-clip, ekor sampai 10,4% di `slp02b`. Sudah dikoreksi di log #6.
2. **`slp41` tidak punya window positif.** Ia hilang dari semua tabel AUC karena AUC tidak
   terdefinisi, padahal skor rata-ratanya paling tinggi di validasi. Subjek satu-kelas harus
   selalu dicek terpisah.
3. **Pooled AUC menipu.** Selalu laporkan pooled **dan** per subjek. Untuk `proposed` selisihnya
   0,12–0,21 poin.
4. **PSG validasi hanya 2 subjek**, salah satunya tanpa kasus positif. Angka PSG tidak stabil.
5. **Label PSG kasar**: epoch 30 s diwariskan ke window 10 s. Ini label noise, bukan bug.
6. **Jangan pakai z-score per rekaman sebagai hasil utama.** Itu dipilih setelah melihat test dan
   merugikan validasi (0,644 → 0,601).

---

## 9. Peta berkas singkat

```
Konteks_Penelitian_Sleep_Apnea.md   aturan acuan (rencana vs hasil)
LOG_KERJA.md                        log lengkap per tahap + angka aktual
paper RM (1).pdf                    paper UTS
config/                             policy.json (eligibility), model.json (hyperparameter)
scripts/                            01..20, satu skrip per tahap
artifacts/                          semua keluaran: audit, splits, windows, signals,
                                    normalization, pilot, train, test, calibration,
                                    offset_probe, aggregate
reports/stage_report.html           tampilan visual semua tahap
dataset-rm/, ucddb/, slpdb/, apnea-ecg/   data mentah
```

---

## 10. Kalau kamu hanya punya 15 menit

Baca `LOG_KERJA.md` bagian **1** (ringkasan satu halaman), **5** (batasan), dan **Tahap 7b**
(hasil multi-seed). Lalu buka `reports/stage_report.html` bagian Stage 7. Itu sudah cukup untuk
tahu posisi dan hasil utamanya. Setelah itu baru baca aturan acuan sebelum mengubah apa pun.
