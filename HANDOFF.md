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
| 5 | Training penuh + freeze, 3 seed × 4 model, 10 s | `artifacts/train/seed{42,1337,2024}_10s/` |
| 6 | Test beku, 3 seed, 10 s | `artifacts/test/seed{42,1337,2024}_10s/` |
| 6b/6c | Diagnostik kalibrasi & offset per rekaman | `artifacts/calibration/`, `artifacts/offset_probe/` |
| **7a/7b** | **Eksperimen 1: agregasi multi-seed, validasi + test** | `artifacts/aggregate/seed42-1337-2024_10s/` |

### Belum dikerjakan (ini pekerjaan berikutnya)

| Eksperimen paper | Status | Catatan |
|---|---|---|
| **Eksperimen 2** — window 30 s vs 10 s | **belum** | Manifest dan `norm_stats` 30 s sudah ada, tinggal training |
| **Eksperimen 3** — ablation preprocessing (filter / normalisasi / clipping) | belum | Butuh training ulang per konfigurasi |
| **Eksperimen 4** — single- vs combined-dataset training, LODO 2 arah | belum | Perlu kohort pembanding yang didefinisikan dulu |
| Lengan ECG-only Apnea-ECG | belum | Disetujui dosen, terpisah dari multimodal |

### Persis di titik mana kita berhenti

Tahap 7a/7b selesai dan terverifikasi. Saya (AI) sudah menghitung biaya Eksperimen 2 dan
menyerahkan keputusan eksekusinya — belum ada satu pun training 30 s yang dijalankan.
Jadi **semua angka 30 s adalah estimasi, bukan hasil**.

---

## 3. Hasil kunci yang harus kamu tahu (jangan salah mengutip)

**Eksperimen 1 (3 seed, window 10 s, mean ± SD):**

| Model | Validasi AUC | Test AUC |
|---|---|---|
| resp_only | 0,7226 ± 0,1185 | **0,7424 ± 0,0083** |
| simple_fusion | 0,6894 ± 0,0471 | 0,5767 ± 0,1231 |
| proposed | 0,6369 ± 0,0153 | 0,5548 ± 0,0299 |
| ecg_only | 0,5513 ± 0,0419 | 0,4513 ± 0,1365 |

Empat hal yang **harus** dilaporkan apa adanya:

1. **`proposed` (cross-attention) tidak lebih baik.** Kalah dari `resp_only` di ketiga seed.
   Tidak ada satu seed pun yang mendukung manfaat cross-attention.
2. **`resp_only` yang paling stabil** (SD test 0,008 di tiga split subjek berbeda).
3. **`simple_fusion` sangat tidak stabil** (test 0,450 / 0,696 / 0,585). Satu seed untuk model
   ini menyesatkan.
4. **Kalibrasi antar subjek belum beres.** `proposed` pooled AUC 0,555 vs rata-rata AUC per
   subjek 0,673. Gap ini bertahan di ketiga seed.

Detail lengkap + tabel per dataset + bootstrap CI: `LOG_KERJA.md` Tahap 7a/7b.

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

Semua skrip di `scripts/`, dijalankan dari root proyek, Python 3.12, CPU-only.

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

Skrip berurutan: `01_audit` → `02_split_windows` → `03_verify_datasets` → `04_preprocess` →
`07_normalize` → `08_dataset` → `15_train` → `16_test` → `19/20_aggregate` → `10_report`.

**Fakta mesin:** CPU 8 core, RAM bebas ~2 GB (harus). Sinyal di-stream dari cache, bukan dimuat
semua. Satu `ecg_only` pernah 13.939 s/epoch karena mesin di-suspend — itu anomali, bukan pola.

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
| 1 | Berapa seed untuk 30 s, dan model mana | Menentukan 6 jam vs 30 jam |
| 2 | Apakah preprocessing diperbaiki dulu (normalisasi skala per rekaman, clipping sadar-skala) | Temuan Tahap 6c: `slp02a/b` kehilangan 8–10% sampel respirasi karena clip 5σ tidak seragam. Kalau diperbaiki sekarang, semua training harus diulang |
| 3 | Kohort pembanding untuk Eksperimen 4 | Tanpa ini, klaim efek harmonisasi tidak bisa dipisahkan dari perubahan komposisi cohort |
| 4 | Apakah ablation pakai arsitektur `proposed` saja atau juga baseline | Menentukan jumlah run |

**Rekomendasi saya (belum dijalankan):** 30 s dengan 1 seed dulu (±6 jam) untuk memastikan
pipeline 30 s hidup dan hasilnya wajar, baru lanjutkan seed lain. Jangan mengubah preprocessing
di tengah perbandingan 10 s vs 30 s, karena itu mengubah dua variabel sekaligus.

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
