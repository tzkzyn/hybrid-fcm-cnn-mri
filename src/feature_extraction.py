#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
feature_extraction.py — Modul 2: Ekstraksi Fitur & Reduksi Dimensi PCA

Bagian dari riset:
  "An AI-Driven Hybrid Fuzzy C-Means Framework for Robust Brain Tumor
   MRI Segmentation via PCA-Based Feature Reduction"

Modul ini mengekstrak fitur tekstur dan statistik dari setiap piksel
MRI menggunakan pendekatan patch-based, lalu mereduksi dimensi dengan
PCA untuk inisialisasi centroid FCM yang optimal.

Fitur per piksel (6 dimensi):
  1. Mean lokal (patch)
  2. Varians lokal (patch)
  3. Intensitas piksel asli
  4. GLCM Kontras   (rata-rata 4 arah)
  5. GLCM Energi    (rata-rata 4 arah)
  6. GLCM Homogenitas (rata-rata 4 arah)

Author  : [Peneliti]
Created : 2026-05-01
Python  : 3.9+
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.ndimage import uniform_filter
from skimage.feature import graycomatrix, graycoprops
from sklearn.decomposition import PCA

from src.preprocessing import (
    extract_slice,
    normalize_intensity,
)

# ---------------------------------------------------------------------------
# Logger setup
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)

# ---------------------------------------------------------------------------
# Konstanta
# ---------------------------------------------------------------------------
FEATURE_NAMES: List[str] = [
    "mean_lokal",
    "varians_lokal",
    "intensitas",
    "glcm_kontras",
    "glcm_energi",
    "glcm_homogenitas",
]

GLCM_ANGLES = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]  # 0°, 45°, 90°, 135°
GLCM_DISTANCE = [1]
GLCM_LEVELS = 256


# ===================================================================
# 1. extract_patch_features
# ===================================================================
def extract_patch_features(
    slice_2d: np.ndarray,
    patch_size: int = 5,
) -> np.ndarray:
    """Mengekstrak 6 fitur tekstur/statistik per piksel dari slice MRI 2-D.

    Setiap piksel direpresentasikan oleh fitur yang dihitung dari patch
    lokal berukuran ``patch_size × patch_size`` yang berpusat pada
    piksel tersebut.

    Fitur yang diekstrak:
      0. **Mean lokal** — rata-rata intensitas dalam patch.
      1. **Varians lokal** — varians intensitas dalam patch.
      2. **Intensitas piksel** — nilai intensitas piksel itu sendiri.
      3. **GLCM Kontras** — rata-rata kontras GLCM pada 4 arah.
      4. **GLCM Energi** — rata-rata energi (angular second moment)
         GLCM pada 4 arah.
      5. **GLCM Homogenitas** — rata-rata homogenitas (inverse
         difference moment) GLCM pada 4 arah.

    Parameters
    ----------
    slice_2d : np.ndarray
        Slice MRI 2-D (H, W) yang sudah dinormalisasi ke [0, 1],
        dtype float32.
    patch_size : int, default 5
        Ukuran sisi patch lokal (harus ganjil agar simetris).

    Returns
    -------
    np.ndarray
        Array berdimensi (H, W, 6) dengan dtype float32, berisi 6
        fitur per piksel.

    Raises
    ------
    ValueError
        Jika input bukan 2-D atau patch_size genap.
    """
    if slice_2d.ndim != 2:
        raise ValueError(
            f"Input harus 2-D, diterima ndim={slice_2d.ndim}"
        )
    if patch_size % 2 == 0:
        raise ValueError(
            f"patch_size harus ganjil, diterima {patch_size}"
        )

    H, W = slice_2d.shape
    img = slice_2d.astype(np.float32)

    # ------------------------------------------------------------------
    # Fitur 0: Mean lokal  (uniform filter = rata-rata kotak)
    # ------------------------------------------------------------------
    mean_local = uniform_filter(img, size=patch_size).astype(np.float32)

    # ------------------------------------------------------------------
    # Fitur 1: Varians lokal = E[X^2] - (E[X])^2
    # ------------------------------------------------------------------
    mean_sq = uniform_filter(img ** 2, size=patch_size).astype(np.float32)
    var_local = np.maximum(mean_sq - mean_local ** 2, 0.0)

    # ------------------------------------------------------------------
    # Fitur 2: Intensitas piksel asli
    # ------------------------------------------------------------------
    intensity = img.copy()

    # ------------------------------------------------------------------
    # Fitur 3-5: GLCM (kontras, energi, homogenitas)
    # ------------------------------------------------------------------
    # GLCM membutuhkan uint8
    img_uint8 = np.clip(img * 255, 0, 255).astype(np.uint8)

    # Komputasi GLCM per patch — untuk efisiensi, gunakan pendekatan
    # sliding window pada ukuran kecil lalu hitung GLCM per patch.
    # Karena GLCM per-piksel sangat mahal (240*240=57600 GLCM),
    # kita gunakan pendekatan block-wise untuk mempercepat.
    pad = patch_size // 2

    glcm_contrast = np.zeros((H, W), dtype=np.float32)
    glcm_energy = np.zeros((H, W), dtype=np.float32)
    glcm_homogeneity = np.zeros((H, W), dtype=np.float32)

    # Padding agar border piksel bisa punya patch penuh
    img_padded = np.pad(img_uint8, pad, mode="reflect")

    logger.info(
        "Menghitung GLCM per piksel (patch=%d×%d, %d piksel total)...",
        patch_size, patch_size, H * W,
    )

    # Strategi: proses per baris untuk balance memory/speed
    # Batch GLCM: kumpulkan patch per baris, hitung satu-satu
    for i in range(H):
        for j in range(W):
            patch = img_padded[i:i + patch_size, j:j + patch_size]

            # Skip patch yang sepenuhnya nol (background)
            if patch.max() == 0:
                continue

            try:
                glcm = graycomatrix(
                    patch,
                    distances=GLCM_DISTANCE,
                    angles=GLCM_ANGLES,
                    levels=GLCM_LEVELS,
                    symmetric=True,
                    normed=True,
                )
                # Rata-rata 4 arah (axis angles)
                glcm_contrast[i, j] = graycoprops(glcm, "contrast").mean()
                glcm_energy[i, j] = graycoprops(glcm, "energy").mean()
                glcm_homogeneity[i, j] = graycoprops(glcm, "homogeneity").mean()
            except Exception:
                # Jika patch bermasalah, biarkan nol
                pass

        # Progress log setiap 20% baris
        if (i + 1) % max(1, H // 5) == 0:
            logger.info("  GLCM progress: %d/%d baris (%.0f%%)", i + 1, H,
                        100 * (i + 1) / H)

    # ------------------------------------------------------------------
    # Normalisasi GLCM fitur ke [0, 1] untuk konsistensi skala
    # ------------------------------------------------------------------
    for arr in (glcm_contrast, glcm_energy, glcm_homogeneity):
        arr_max = arr.max()
        if arr_max > 0:
            arr /= arr_max

    # ------------------------------------------------------------------
    # Stack semua fitur: (H, W, 6)
    # ------------------------------------------------------------------
    features = np.stack([
        mean_local,       # 0
        var_local,        # 1
        intensity,        # 2
        glcm_contrast,    # 3
        glcm_energy,      # 4
        glcm_homogeneity, # 5
    ], axis=-1)

    logger.info(
        "Fitur diekstrak  |  shape=%s  |  per-feature range: %s",
        features.shape,
        {FEATURE_NAMES[k]: f"[{features[:,:,k].min():.3f}, "
                           f"{features[:,:,k].max():.3f}]"
         for k in range(6)},
    )
    return features


# ===================================================================
# 2. reduce_with_pca
# ===================================================================
def reduce_with_pca(
    features_2d: np.ndarray,
    n_components: int = 3,
    n_clusters: int = 3,
) -> Tuple[np.ndarray, PCA, np.ndarray, float]:
    """Mereduksi dimensi fitur per-piksel dengan PCA.

    Melakukan PCA pada fitur (H, W, F) yang di-reshape ke (H*W, F),
    menghasilkan proyeksi berdimensi rendah yang mempertahankan
    ≥ 95% variansi data.

    Parameters
    ----------
    features_2d : np.ndarray
        Array fitur berdimensi (H, W, F) dari ``extract_patch_features``.
    n_components : int, default 3
        Jumlah komponen PCA yang dipertahankan.
    n_clusters : int, default 3
        Jumlah kluster untuk inisialisasi centroid FCM.
        - 2: background vs tumor
        - 3: background, tumor core, edema
        - 4: background, necrosis, edema, enhancing tumor

    Returns
    -------
    projected : np.ndarray
        Array (H, W, n_components) — fitur tereduksi PCA.
    pca_model : sklearn.decomposition.PCA
        Objek PCA yang sudah di-fit (untuk analisis/inverse nanti).
    initial_centroids : np.ndarray
        Array (n_clusters, n_components) — centroid awal untuk FCM,
        diinisialisasi dengan metode berbasis distribusi PC1.
    explained_variance : float
        Total explained variance ratio (0–1).

    Raises
    ------
    ValueError
        Jika input bukan 3-D atau n_components > jumlah fitur.
    """
    if features_2d.ndim != 3:
        raise ValueError(
            f"features_2d harus 3-D (H, W, F), diterima ndim={features_2d.ndim}"
        )

    H, W, F = features_2d.shape

    if n_components > F:
        raise ValueError(
            f"n_components ({n_components}) > jumlah fitur ({F})"
        )

    # Reshape ke (N, F) — satu baris per piksel
    flat = features_2d.reshape(-1, F).astype(np.float64)

    # Fit PCA
    pca = PCA(n_components=n_components, random_state=42)
    projected_flat = pca.fit_transform(flat)  # (N, n_components)

    # Reshape kembali ke (H, W, C)
    projected = projected_flat.reshape(H, W, n_components).astype(np.float32)

    # Total explained variance
    explained = float(pca.explained_variance_ratio_.sum())

    logger.info(
        "PCA selesai  |  %d fitur -> %d komponen  |  "
        "explained_variance=%.4f (%.1f%%)  |  "
        "per-komponen: %s",
        F, n_components, explained, explained * 100,
        [f"{v:.4f}" for v in pca.explained_variance_ratio_],
    )

    # Inisialisasi centroid FCM berbasis distribusi PC1
    centroids = compute_initial_centroids(projected, n_clusters=n_clusters)

    return projected, pca, centroids, explained


# ===================================================================
# 3. compute_initial_centroids
# ===================================================================
def compute_initial_centroids(
    projected: np.ndarray,
    n_clusters: int = 3,
) -> np.ndarray:
    """Menginisialisasi centroid FCM berdasarkan distribusi statistik PC1.

    Strategi inisialisasi cerdas yang memanfaatkan properti PCA:
      - PC1 menangkap variasi terbesar → gradien intensitas brain vs tumor
      - Piksel dengan PC1 jauh dari median cenderung merupakan struktur
        yang berbeda secara radiometrik (misal tumor vs jaringan sehat)

    Algoritma:
      1. Hitung median dan std dari PC1.
      2. **Cluster 0** (non-tumor/background):
         mean dari piksel dengan PC1 < median - 1×std.
      3. **Cluster 1** (tumor core):
         mean dari piksel dengan PC1 > median + 1×std.
      4. **Cluster 2** (edema/intermediate):
         mean dari piksel di antara kedua threshold.
      5. Jika ``n_clusters=4``, cluster 1 dipecah berdasarkan PC2.
      6. Jika ``n_clusters=2``, cluster 1 dan 2 digabung.

    Parameters
    ----------
    projected : np.ndarray
        Hasil proyeksi PCA berdimensi (H, W, C).
    n_clusters : int, default 3
        Jumlah kluster yang diinginkan (2, 3, atau 4).

    Returns
    -------
    np.ndarray
        Array (n_clusters, C) berisi koordinat centroid awal di ruang PCA.

    Raises
    ------
    ValueError
        Jika n_clusters bukan 2, 3, atau 4.
    """
    if n_clusters not in (2, 3, 4):
        raise ValueError(
            f"n_clusters harus 2, 3, atau 4, diterima {n_clusters}"
        )

    H, W, C = projected.shape
    flat = projected.reshape(-1, C)  # (N, C)

    # Ambil PC1
    pc1 = flat[:, 0]
    median_pc1 = np.median(pc1)
    std_pc1 = np.std(pc1)

    # Batas region
    low_threshold = median_pc1 - 1.0 * std_pc1
    high_threshold = median_pc1 + 1.0 * std_pc1

    # Masking
    mask_low = pc1 < low_threshold    # Region non-tumor / background
    mask_high = pc1 > high_threshold  # Region tumor core
    mask_mid = (~mask_low) & (~mask_high)  # Region edema / peralihan

    # Hitung centroid per region (fallback ke overall mean jika kosong)
    overall_mean = flat.mean(axis=0)

    def safe_mean(mask):
        if mask.sum() > 0:
            return flat[mask].mean(axis=0)
        return overall_mean.copy()

    centroid_bg = safe_mean(mask_low)       # Cluster 0: background
    centroid_tumor = safe_mean(mask_high)   # Cluster 1: tumor core
    centroid_edema = safe_mean(mask_mid)    # Cluster 2: edema

    if n_clusters == 2:
        # Gabung tumor + edema
        mask_combined = mask_high | mask_mid
        centroid_combined = safe_mean(mask_combined)
        centroids = np.stack([centroid_bg, centroid_combined], axis=0)

    elif n_clusters == 3:
        centroids = np.stack([centroid_bg, centroid_tumor, centroid_edema],
                             axis=0)

    elif n_clusters == 4:
        # Pecah tumor region berdasarkan PC2
        tumor_pixels = flat[mask_high]
        if tumor_pixels.shape[0] >= 2 and C >= 2:
            pc2_tumor = tumor_pixels[:, 1]
            pc2_median = np.median(pc2_tumor)
            sub_low = pc2_tumor < pc2_median
            sub_high = ~sub_low

            if sub_low.sum() > 0 and sub_high.sum() > 0:
                centroid_tumor_a = tumor_pixels[sub_low].mean(axis=0)
                centroid_tumor_b = tumor_pixels[sub_high].mean(axis=0)
            else:
                # Fallback: geser centroid tumor sedikit
                centroid_tumor_a = centroid_tumor.copy()
                centroid_tumor_b = centroid_tumor.copy()
                if C >= 2:
                    centroid_tumor_a[1] -= 0.1 * std_pc1
                    centroid_tumor_b[1] += 0.1 * std_pc1
        else:
            centroid_tumor_a = centroid_tumor.copy()
            centroid_tumor_b = centroid_tumor.copy()

        centroids = np.stack([
            centroid_bg,
            centroid_tumor_a,   # Necrosis
            centroid_edema,     # Edema
            centroid_tumor_b,   # Enhancing tumor
        ], axis=0)

    centroids = centroids.astype(np.float32)

    logger.info(
        "Centroid awal dihitung  |  n_clusters=%d  |  "
        "PC1 median=%.4f  std=%.4f  |  "
        "threshold=[%.4f, %.4f]  |  "
        "region sizes: low=%d mid=%d high=%d",
        n_clusters, median_pc1, std_pc1,
        low_threshold, high_threshold,
        mask_low.sum(), mask_mid.sum(), mask_high.sum(),
    )
    for i, c in enumerate(centroids):
        logger.info("  Centroid %d: %s", i, np.array2string(c, precision=4))

    return centroids


# ===================================================================
# 4. analyze_pca_components
# ===================================================================
def analyze_pca_components(
    pca_model: PCA,
    feature_names: Optional[List[str]] = None,
) -> Dict:
    """Menganalisis komponen PCA untuk interpretasi dan pelaporan paper.

    Mengekstrak informasi tentang kontribusi setiap fitur asli terhadap
    komponen utama, yang berguna untuk menjelaskan mengapa PCA efektif
    dalam memisahkan struktur tumor.

    Parameters
    ----------
    pca_model : sklearn.decomposition.PCA
        Model PCA yang sudah di-fit (dari ``reduce_with_pca``).
    feature_names : list of str, optional
        Nama fitur asli. Jika None, gunakan ``FEATURE_NAMES`` default.

    Returns
    -------
    dict
        Dictionary berisi:
        - ``'explained_variance_ratio'``: array (n_components,) —
          proporsi variansi yang dijelaskan tiap komponen.
        - ``'cumulative_variance'``: array (n_components,) —
          variansi kumulatif.
        - ``'component_loadings'``: array (n_components, n_features) —
          bobot/loading setiap fitur pada setiap komponen. Nilai absolut
          yang besar menunjukkan kontribusi besar.
        - ``'feature_names'``: list of str — nama fitur.
        - ``'n_components'``: int — jumlah komponen.
        - ``'dominant_features'``: list of str — fitur dominan per
          komponen (loading absolut terbesar).
    """
    if feature_names is None:
        feature_names = FEATURE_NAMES.copy()

    evr = pca_model.explained_variance_ratio_
    cumulative = np.cumsum(evr)
    loadings = pca_model.components_  # (n_components, n_features)

    # Fitur dominan per komponen
    dominant = []
    for i in range(loadings.shape[0]):
        idx = np.argmax(np.abs(loadings[i]))
        name = feature_names[idx] if idx < len(feature_names) else f"feat_{idx}"
        dominant.append(name)

    analysis = {
        "explained_variance_ratio": evr,
        "cumulative_variance": cumulative,
        "component_loadings": loadings,
        "feature_names": feature_names,
        "n_components": len(evr),
        "dominant_features": dominant,
    }

    logger.info("Analisis PCA:")
    for i in range(len(evr)):
        logger.info(
            "  PC%d: explained=%.4f (cum=%.4f)  dominant=%s  "
            "loadings=%s",
            i + 1, evr[i], cumulative[i], dominant[i],
            np.array2string(loadings[i], precision=3),
        )

    return analysis


# ===================================================================
# 5. extract_features_from_patient
# ===================================================================
def extract_features_from_patient(
    patient_dict: Dict[str, np.ndarray],
    modality: str = "flair",
    slice_idx: int = -1,
    patch_size: int = 5,
    n_components: int = 3,
    n_clusters: int = 3,
) -> Dict:
    """Convenience function: preprocessing + feature extraction + PCA.

    Menggabungkan seluruh pipeline dari slice MRI mentah hingga
    proyeksi PCA siap pakai, dalam satu panggilan fungsi.

    Parameters
    ----------
    patient_dict : dict[str, np.ndarray]
        Hasil dari ``load_patient_modalities()`` atau
        ``create_dummy_patient_data()``.
    modality : str, default 'flair'
        Nama modalitas yang diproses ('flair', 't1', 't1ce', 't2').
    slice_idx : int, default -1
        Indeks slice axial. Jika -1, gunakan slice tengah.
    patch_size : int, default 5
        Ukuran patch untuk ekstraksi fitur.
    n_components : int, default 3
        Jumlah komponen PCA.
    n_clusters : int, default 3
        Jumlah kluster untuk inisialisasi centroid.

    Returns
    -------
    dict
        Dictionary berisi:
        - ``'slice_original'``: np.ndarray (H, W) — slice sebelum
          normalisasi.
        - ``'slice_normalized'``: np.ndarray (H, W) — slice setelah
          normalisasi intensitas.
        - ``'features'``: np.ndarray (H, W, 6) — fitur per piksel.
        - ``'projected'``: np.ndarray (H, W, n_components) — proyeksi PCA.
        - ``'initial_centroids'``: np.ndarray (n_clusters, n_components).
        - ``'explained_variance'``: float.
        - ``'pca_model'``: objek PCA.
        - ``'pca_analysis'``: dict dari ``analyze_pca_components``.

    Raises
    ------
    KeyError
        Jika modalitas tidak ditemukan dalam patient_dict.
    """
    if modality not in patient_dict:
        available = list(patient_dict.keys())
        raise KeyError(
            f"Modalitas '{modality}' tidak ditemukan. "
            f"Tersedia: {available}"
        )

    volume = patient_dict[modality]

    # Tentukan slice index
    if slice_idx < 0:
        slice_idx = volume.shape[0] // 2

    logger.info(
        "Pipeline fitur  |  modality=%s  slice=%d  "
        "patch=%d  PCA_comp=%d  clusters=%d",
        modality, slice_idx, patch_size, n_components, n_clusters,
    )

    # 1. Ambil slice asli
    slice_original = extract_slice(volume, slice_idx=slice_idx, axis=0)

    # 2. Normalisasi volume lalu ambil slice
    volume_norm = normalize_intensity(volume)
    slice_normalized = extract_slice(volume_norm, slice_idx=slice_idx, axis=0)

    # 3. Ekstraksi fitur
    features = extract_patch_features(slice_normalized, patch_size=patch_size)

    # 4. Reduksi PCA
    projected, pca_model, centroids, explained = reduce_with_pca(
        features, n_components=n_components, n_clusters=n_clusters,
    )

    # 5. Analisis PCA
    pca_analysis = analyze_pca_components(pca_model)

    result = {
        "slice_original": slice_original,
        "slice_normalized": slice_normalized,
        "features": features,
        "projected": projected,
        "initial_centroids": centroids,
        "explained_variance": explained,
        "pca_model": pca_model,
        "pca_analysis": pca_analysis,
        "modality": modality,
        "slice_idx": slice_idx,
    }

    logger.info(
        "Pipeline selesai  |  features=%s  projected=%s  "
        "centroids=%s  explained=%.4f",
        features.shape, projected.shape, centroids.shape, explained,
    )
    return result


# ===================================================================
# CLI — Quick self-test
# ===================================================================
if __name__ == "__main__":
    from src.preprocessing import create_dummy_patient_data

    print("=" * 60)
    print("  Modul Feature Extraction — Self-Test (mini)")
    print("=" * 60)

    # Buat data dummy kecil untuk self-test cepat
    dummy = create_dummy_patient_data(shape=(10, 32, 32), seed=42)
    flair = dummy["flair"]
    flair_norm = normalize_intensity(flair)
    mid = extract_slice(flair_norm, axis=0)

    print(f"  Slice shape: {mid.shape}")

    feats = extract_patch_features(mid, patch_size=5)
    print(f"  Features shape: {feats.shape}")

    projected, pca, centroids, ev = reduce_with_pca(feats, n_components=3,
                                                     n_clusters=3)
    print(f"  Projected shape: {projected.shape}")
    print(f"  Centroids shape: {centroids.shape}")
    print(f"  Explained variance: {ev:.4f}")

    analysis = analyze_pca_components(pca)
    print(f"  Dominant features: {analysis['dominant_features']}")

    print("\n" + "=" * 60)
    print("  Self-test PASSED ✓")
    print("=" * 60)
