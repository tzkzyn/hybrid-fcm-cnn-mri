# Pipeline Segmentasi Tumor Otak MRI (BraTS 2020)

Repositori ini berisi implementasi kode penelitian untuk segmentasi citra medis multimodal MRI (FLAIR, T1, T1ce, T2) menggunakan integrasi metode **Fuzzy C-Means**, **Reduksi Fitur PCA**, **Optimasi Algoritma Metaheuristik (Marine Predators Algorithm / MPA)**, dan **CNN Refinement**.

---

## 📋 Daftar Isi
1. [Fitur Utama](#-fitur-utama)
2. [Struktur Folder](#-struktur-folder)
3. [Instalasi](#-instalasi)
4. [Persiapan Dataset](#-persiapan-dataset)
5. [Cara Penggunaan](#-cara-penggunaan)
6. [Hasil Evaluasi & Metrik](#-hasil-evaluasi--metrik)

---

## 🌟 Fitur Utama
* **Prapemrosesan Otomatis**: Normalisasi intensitas z-score, skull stripping (masking otak), dan penanganan noise.
* **Ekstraksi Fitur Spasial & Tekstur**: Fitur intensitas multi-modalitas dan tekstur GLCM yang direduksi dimensinya menggunakan PCA.
* **Hybrid Fuzzy C-Means (FCM)**: Inisialisasi centroid terarah menggunakan hasil PCA serta optimasi swarm intelligence via Mealpy MPA.
* **Penyempurnaan CNN (Refiner)**: Model konvolusional untuk mengurangi false positive pada area batas tumor.
* **Pengujian Robustness**: Modul pengujian ketahanan terhadap degradasi noise Gaussian dan Salt-and-Pepper.

---

## 📁 Struktur Folder
```text
.
├── colab_upload/             # Utilitas pendukung eksekusi di GPU Google Colab
├── docs/                     # Dokumentasi arsitektur dan teknis
│   └── ARCHITECTURE.md
├── src/                      # Source code utama
│   ├── preprocessing.py      # Pembacaan citra NIfTI (.nii/.nii.gz) & normalisasi
│   ├── feature_extraction.py # GLCM, ekstraksi patch, dan reduksi PCA
│   ├── hybrid_fcm.py         # Algoritma Hybrid FCM & integrasi MPA
│   ├── cnn_module.py         # Arsitektur jaringan saraf konvolusi & pelatihan
│   ├── postprocessing.py     # Operasi morfologi, pembersihan komponen kecil
│   └── evaluation.py         # Perhitungan DSC, IoU, HD95, sensitivitas, spesifisitas
├── tests/                    # Pengujian modular untuk setiap modul
│   ├── test_preprocessing.py
│   ├── test_feature_extraction.py
│   ├── test_hybrid_fcm.py
│   ├── test_cnn_module.py
│   ├── test_postprocessing.py
│   └── test_evaluation.py
├── train_full_pipeline.py    # Skrip pipeline pelatihan end-to-end
├── requirements.txt          # Daftar pustaka Python yang dibutuhkan
└── .gitignore                # Pengecualian file besar dan virtual environment
```

---

## ⚙️ Instalasi

Pastikan menggunakan Python 3.9 atau versi yang lebih baru:

```bash
# 1. Clone repositori ini
git clone https://github.com/tazkiyah-m/proyek_Jurnal.git
cd proyek_Jurnal

# 2. Buat dan aktifkan virtual environment (opsional tetapi disarankan)
python3 -m venv venv
source venv/bin/activate  # Di Linux/macOS
# atau: .\venv\Scripts\activate  # Di Windows

# 3. Pasang seluruh dependensi
pip install -r requirements.txt
```

---

## 🗂️ Persiapan Dataset
Dataset yang digunakan adalah **BraTS 2020** (Brain Tumor Segmentation Challenge 2020). Struktur direktori yang diharapkan oleh pipeline:
```text
data/
└── Brats2020/
    └── training/
        ├── BraTS20_Training_001/
        │   ├── BraTS20_Training_001_flair.nii
        │   ├── BraTS20_Training_001_t1.nii
        │   ├── BraTS20_Training_001_t1ce.nii
        │   ├── BraTS20_Training_001_t2.nii
        │   └── BraTS20_Training_001_seg.nii
        └── ...
```

---

## 🚀 Cara Penggunaan

### 1. Menjalankan Uji Modular (Unit Tests)
Anda dapat memverifikasi tiap fungsi dengan script uji coba yang tersedia:
```bash
python tests/test_preprocessing.py
python tests/test_feature_extraction.py
python tests/test_hybrid_fcm.py
python tests/test_cnn_module.py
python tests/test_postprocessing.py
python tests/test_evaluation.py
```

### 2. Menjalankan Training Penuh
Untuk menjalankan seluruh tahapan pipeline dari pembacaan data hingga pelaporan hasil:
```bash
python train_full_pipeline.py
```

---

## 📊 Hasil Evaluasi (Ablation Study)

Berdasarkan pengujian evaluasi pada data uji:
* **Full Pipeline**: Menghasilkan rata-rata Dice Similarity Coefficient (DSC) sebesar **0.6448**, IoU **0.5208**, Sensitivitas **0.9223**, dan Spesifisitas **0.9750**.
* **Uji Signifikansi**: Wilcoxon Signed-Rank Test menunjukkan peningkatan metrik yang signifikan secara statistik (*p < 0.001*).
