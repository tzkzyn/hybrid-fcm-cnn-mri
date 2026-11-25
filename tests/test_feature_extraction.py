#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_feature_extraction.py — Skrip Uji Komprehensif untuk Modul 2

Menguji seluruh fungsi di src/feature_extraction.py:
  1. Ekstraksi fitur patch-based (6 fitur per piksel)
  2. Reduksi PCA ke 3 komponen
  3. Inisialisasi centroid FCM berbasis PC1
  4. Analisis kontribusi fitur ke komponen PCA
  5. Visualisasi: fitur map, scatter PCA, centroid, loadings

Output: Figure matplotlib disimpan ke output/test_feature_*.png

Author  : [Peneliti]
Created : 2026-05-01
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from src.preprocessing import (
    create_dummy_patient_data,
    extract_slice,
    load_patient_modalities,
    normalize_intensity,
    BRATS_SHAPE,
)
from src.feature_extraction import (
    FEATURE_NAMES,
    analyze_pca_components,
    compute_initial_centroids,
    extract_features_from_patient,
    extract_patch_features,
    reduce_with_pca,
)

# ---------------------------------------------------------------------------
# Konfigurasi
# ---------------------------------------------------------------------------
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

DATASET_PATH = PROJECT_ROOT / "data" / "Brats2020" / "training"
SAMPLE_PATIENT = "BraTS20_Training_001"

# Gunakan shape kecil untuk dummy test (GLCM per-pixel mahal)
# 155×240×240 slice → 240×240 = 57600 GLCM. Kita pakai 64×64 untuk test.
DUMMY_SHAPE = (30, 64, 64)

plt.rcParams.update({
    "figure.facecolor": "#1a1a2e",
    "axes.facecolor": "#16213e",
    "axes.edgecolor": "#e94560",
    "axes.labelcolor": "#eaeaea",
    "text.color": "#eaeaea",
    "xtick.color": "#aaaaaa",
    "ytick.color": "#aaaaaa",
    "font.size": 10,
    "axes.titlesize": 12,
    "figure.titlesize": 14,
})


def separator(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ===================================================================
# TEST 1: Ekstraksi Fitur Patch-Based
# ===================================================================
def test_patch_features(dummy: dict) -> tuple:
    """Ekstrak fitur dari slice FLAIR dummy dan cetak informasi."""
    separator("TEST 1: Ekstraksi Fitur Patch-Based (6 Fitur)")

    flair = dummy["flair"]
    flair_norm = normalize_intensity(flair)
    mid_idx = flair.shape[0] // 2
    slice_norm = extract_slice(flair_norm, slice_idx=mid_idx, axis=0)
    slice_orig = extract_slice(flair, slice_idx=mid_idx, axis=0)

    print(f"  Slice shape: {slice_norm.shape}")
    print(f"  Slice range: [{slice_norm.min():.4f}, {slice_norm.max():.4f}]")

    t0 = time.time()
    features = extract_patch_features(slice_norm, patch_size=5)
    elapsed = time.time() - t0

    print(f"\n  Features shape: {features.shape}")
    print(f"  Waktu komputasi: {elapsed:.1f} detik")
    print(f"\n  Range per fitur:")
    for k, name in enumerate(FEATURE_NAMES):
        f_min = features[:, :, k].min()
        f_max = features[:, :, k].max()
        f_mean = features[:, :, k].mean()
        print(f"    [{k}] {name:20s}  "
              f"min={f_min:8.4f}  max={f_max:8.4f}  mean={f_mean:8.4f}")

    # ---- Plot: Slice asli, normalized, mean feature ----
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        "FLAIR Slice — Original, Normalized, Mean Lokal",
        fontweight="bold", fontsize=14, color="#e94560",
    )

    axes[0].imshow(slice_orig, cmap="gray", aspect="equal")
    axes[0].set_title("Original", fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(slice_norm, cmap="gray", aspect="equal", vmin=0, vmax=1)
    axes[1].set_title("Normalized [0,1]", fontweight="bold")
    axes[1].axis("off")

    im2 = axes[2].imshow(features[:, :, 0], cmap="viridis", aspect="equal")
    axes[2].set_title("Mean Lokal (patch 5×5)", fontweight="bold")
    axes[2].axis("off")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    path = OUTPUT_DIR / "test_feature_slice_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  ✓ Plot disimpan: {path}")

    # ---- Plot: Semua 6 fitur ----
    fig2, axes2 = plt.subplots(2, 3, figsize=(16, 10))
    fig2.suptitle(
        "6 Fitur Per-Piksel (Patch 5×5)",
        fontweight="bold", fontsize=14, color="#e94560",
    )

    for k in range(6):
        ax = axes2[k // 3, k % 3]
        im = ax.imshow(features[:, :, k], cmap="inferno", aspect="equal")
        ax.set_title(f"[{k}] {FEATURE_NAMES[k]}", fontweight="bold")
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    path2 = OUTPUT_DIR / "test_feature_all_6.png"
    fig2.savefig(path2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"  ✓ Plot disimpan: {path2}")

    return features, slice_norm, slice_orig


# ===================================================================
# TEST 2: PCA Reduction
# ===================================================================
def test_pca(features: np.ndarray) -> tuple:
    """Reduksi fitur ke 3 komponen PCA."""
    separator("TEST 2: Reduksi PCA (6 → 3 Komponen)")

    projected, pca_model, centroids, explained = reduce_with_pca(
        features, n_components=3, n_clusters=3,
    )

    print(f"  Input features shape  : {features.shape}")
    print(f"  Projected shape       : {projected.shape}")
    print(f"  Centroids shape       : {centroids.shape}")
    print(f"  Total explained var   : {explained:.4f} ({explained*100:.1f}%)")
    print(f"\n  Explained variance per komponen:")
    for i, v in enumerate(pca_model.explained_variance_ratio_):
        cum = pca_model.explained_variance_ratio_[:i+1].sum()
        print(f"    PC{i+1}: {v:.4f} (kumulatif: {cum:.4f})")

    print(f"\n  Centroid koordinat (ruang PCA):")
    labels = ["Background/Non-tumor", "Tumor Core", "Edema"]
    for i, c in enumerate(centroids):
        label = labels[i] if i < len(labels) else f"Cluster {i}"
        print(f"    Cluster {i} ({label}): {np.array2string(c, precision=4)}")

    return projected, pca_model, centroids, explained


# ===================================================================
# TEST 3: Centroid Inisialisasi (2, 3, 4 cluster)
# ===================================================================
def test_centroids(projected: np.ndarray) -> None:
    """Uji inisialisasi centroid untuk 2, 3, dan 4 cluster."""
    separator("TEST 3: Inisialisasi Centroid (2, 3, 4 Cluster)")

    for k in (2, 3, 4):
        c = compute_initial_centroids(projected, n_clusters=k)
        print(f"\n  n_clusters = {k}  →  centroids shape: {c.shape}")
        for i, row in enumerate(c):
            print(f"    Cluster {i}: {np.array2string(row, precision=4)}")


# ===================================================================
# TEST 4: Analisis PCA Components
# ===================================================================
def test_analysis(pca_model, features: np.ndarray) -> dict:
    """Analisis kontribusi fitur ke komponen PCA."""
    separator("TEST 4: Analisis Komponen PCA")

    analysis = analyze_pca_components(pca_model, FEATURE_NAMES)

    print(f"  Jumlah komponen: {analysis['n_components']}")
    print(f"  Fitur dominan per komponen: {analysis['dominant_features']}")
    print(f"\n  Component loadings:")
    loadings = analysis["component_loadings"]
    header = "  {:>12s}".format("") + "".join(
        f"  {name:>14s}" for name in FEATURE_NAMES
    )
    print(header)
    for i in range(loadings.shape[0]):
        row = f"  {'PC'+str(i+1):>12s}"
        for j in range(loadings.shape[1]):
            row += f"  {loadings[i, j]:>14.4f}"
        print(row)

    return analysis


# ===================================================================
# TEST 5: Visualisasi — Scatter PCA + Centroid
# ===================================================================
def test_scatter_plot(
    projected: np.ndarray,
    centroids: np.ndarray,
    dummy: dict,
) -> None:
    """Scatter plot PC1 vs PC2 dengan warna ground truth."""
    separator("TEST 5: Scatter Plot PCA (PC1 vs PC2)")

    seg = dummy["seg"]
    mid_idx = seg.shape[0] // 2
    seg_slice = extract_slice(seg, slice_idx=mid_idx, axis=0)

    H, W, _ = projected.shape
    pc1 = projected[:, :, 0].flatten()
    pc2 = projected[:, :, 1].flatten()
    labels = seg_slice.flatten().astype(int)

    # Subsample untuk plot cepat
    n_total = len(pc1)
    max_points = 5000
    if n_total > max_points:
        rng = np.random.default_rng(42)
        idx = rng.choice(n_total, max_points, replace=False)
        pc1_s, pc2_s, labels_s = pc1[idx], pc2[idx], labels[idx]
    else:
        pc1_s, pc2_s, labels_s = pc1, pc2, labels

    print(f"  Total piksel: {n_total}")
    print(f"  Subsampled: {len(pc1_s)}")
    print(f"  Label unik: {np.unique(labels_s)}")

    # ---- Plot scatter ----
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    fig.suptitle(
        "Proyeksi PCA — PC1 vs PC2\n(warna = ground truth segmentasi)",
        fontweight="bold", fontsize=14, color="#e94560",
    )

    label_names = {0: "Background", 1: "Necrosis", 2: "Edema", 3: "Enhancing"}
    colors = {0: "#4a90d9", 1: "#e94560", 2: "#f5a623", 3: "#7ed321"}

    for lbl in sorted(np.unique(labels_s)):
        mask = labels_s == lbl
        name = label_names.get(lbl, f"Label {lbl}")
        color = colors.get(lbl, "#cccccc")
        ax.scatter(
            pc1_s[mask], pc2_s[mask],
            c=color, label=name, alpha=0.4, s=8, edgecolors="none",
        )

    # Centroid
    ax.scatter(
        centroids[:, 0], centroids[:, 1],
        c="red", marker="X", s=300, edgecolors="white",
        linewidths=2, zorder=10, label="Centroid Awal",
    )

    # Annotate centroid
    cluster_labels = ["BG", "Tumor", "Edema", "Enh"]
    for i, c in enumerate(centroids):
        lbl = cluster_labels[i] if i < len(cluster_labels) else f"C{i}"
        ax.annotate(
            lbl, (c[0], c[1]),
            textcoords="offset points", xytext=(10, 10),
            fontsize=11, fontweight="bold", color="white",
            arrowprops=dict(arrowstyle="->", color="white", lw=1.5),
        )

    ax.set_xlabel("PC1", fontsize=12, fontweight="bold")
    ax.set_ylabel("PC2", fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10, framealpha=0.8)
    ax.grid(alpha=0.2, color="#555555")

    plt.tight_layout()
    path = OUTPUT_DIR / "test_feature_pca_scatter.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  ✓ Plot disimpan: {path}")


# ===================================================================
# TEST 6: Bar Chart Explained Variance + Heatmap Loadings
# ===================================================================
def test_variance_and_loadings(analysis: dict) -> None:
    """Bar chart explained variance + heatmap component loadings."""
    separator("TEST 6: Explained Variance & Component Loadings")

    evr = analysis["explained_variance_ratio"]
    cum = analysis["cumulative_variance"]
    loadings = analysis["component_loadings"]
    feat_names = analysis["feature_names"]
    n_comp = analysis["n_components"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        "Analisis PCA — Explained Variance & Component Loadings",
        fontweight="bold", fontsize=14, color="#e94560",
    )

    # ---- Bar chart: explained variance ----
    ax0 = axes[0]
    x = np.arange(n_comp)
    bars = ax0.bar(x, evr, color="#e94560", alpha=0.8, edgecolor="white",
                   label="Individual")
    ax0.plot(x, cum, "o-", color="#f5a623", linewidth=2, markersize=8,
             label="Kumulatif")

    # Label di atas bar
    for bar, val in zip(bars, evr):
        ax0.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=10,
                 color="#eaeaea", fontweight="bold")

    # Garis threshold 95%
    ax0.axhline(y=0.95, color="#7ed321", linestyle="--", alpha=0.7,
                label="95% threshold")

    ax0.set_xlabel("Komponen PCA", fontsize=11)
    ax0.set_ylabel("Explained Variance Ratio", fontsize=11)
    ax0.set_title("Explained Variance", fontweight="bold")
    ax0.set_xticks(x)
    ax0.set_xticklabels([f"PC{i+1}" for i in range(n_comp)])
    ax0.legend(fontsize=9, loc="center right")
    ax0.set_ylim(0, 1.1)

    # ---- Heatmap: component loadings ----
    ax1 = axes[1]
    im = ax1.imshow(np.abs(loadings), cmap="YlOrRd", aspect="auto",
                    vmin=0)
    ax1.set_xticks(range(len(feat_names)))
    ax1.set_xticklabels(feat_names, rotation=45, ha="right", fontsize=9)
    ax1.set_yticks(range(n_comp))
    ax1.set_yticklabels([f"PC{i+1}" for i in range(n_comp)])
    ax1.set_title("Component Loadings (|nilai|)", fontweight="bold")

    # Annotate cells
    for i in range(n_comp):
        for j in range(len(feat_names)):
            val = loadings[i, j]
            color = "white" if np.abs(val) > 0.4 else "#333333"
            ax1.text(j, i, f"{val:.3f}", ha="center", va="center",
                     fontsize=9, color=color, fontweight="bold")

    plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)

    plt.tight_layout()
    path = OUTPUT_DIR / "test_feature_pca_analysis.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Plot disimpan: {path}")


# ===================================================================
# TEST 7: Convenience Pipeline
# ===================================================================
def test_convenience_pipeline(dummy: dict) -> None:
    """Test extract_features_from_patient convenience function."""
    separator("TEST 7: Convenience Pipeline (extract_features_from_patient)")

    t0 = time.time()
    result = extract_features_from_patient(
        dummy, modality="flair", slice_idx=-1,
        patch_size=5, n_components=3, n_clusters=3,
    )
    elapsed = time.time() - t0

    print(f"  Waktu total pipeline: {elapsed:.1f} detik")
    print(f"\n  Output keys: {list(result.keys())}")
    print(f"  slice_original   : {result['slice_original'].shape}")
    print(f"  slice_normalized : {result['slice_normalized'].shape}")
    print(f"  features         : {result['features'].shape}")
    print(f"  projected        : {result['projected'].shape}")
    print(f"  initial_centroids: {result['initial_centroids'].shape}")
    print(f"  explained_var    : {result['explained_variance']:.4f}")
    print(f"  modality         : {result['modality']}")
    print(f"  slice_idx        : {result['slice_idx']}")


# ===================================================================
# BONUS: Dataset Asli
# ===================================================================
def test_real_dataset() -> None:
    """Uji dengan pasien BraTS 2020 asli (jika tersedia)."""
    separator("BONUS: Feature Extraction pada Dataset Asli")

    patient_path = DATASET_PATH / SAMPLE_PATIENT
    if not patient_path.is_dir():
        print(f"  ⚠ Dataset asli tidak ditemukan: {patient_path}")
        print("  ⚠ Lewati tes dataset asli.")
        return

    print(f"  Memuat pasien: {SAMPLE_PATIENT}")
    data = load_patient_modalities(patient_path)

    # Gunakan convenience function — slice tengah FLAIR
    # Karena 240×240 GLCM sangat lama, kita crop ke area tumor saja
    flair_norm = normalize_intensity(data["flair"])
    seg = data["seg"]
    mid_idx = flair_norm.shape[0] // 2

    flair_slice = extract_slice(flair_norm, slice_idx=mid_idx, axis=0)
    seg_slice = extract_slice(seg, slice_idx=mid_idx, axis=0)

    # Crop ke ROI 64×64 di sekitar pusat (area tumor biasanya di tengah)
    H, W = flair_slice.shape
    cy, cx = H // 2, W // 2
    roi_size = 64
    y1, y2 = cy - roi_size // 2, cy + roi_size // 2
    x1, x2 = cx - roi_size // 2, cx + roi_size // 2

    flair_roi = flair_slice[y1:y2, x1:x2]
    seg_roi = seg_slice[y1:y2, x1:x2]

    print(f"  ROI crop: [{y1}:{y2}, {x1}:{x2}] → shape={flair_roi.shape}")

    t0 = time.time()
    features = extract_patch_features(flair_roi, patch_size=5)
    elapsed_feat = time.time() - t0
    print(f"  Feature extraction time: {elapsed_feat:.1f}s")

    projected, pca_model, centroids, explained = reduce_with_pca(
        features, n_components=3, n_clusters=3,
    )
    print(f"  Explained variance: {explained:.4f}")

    # ---- Plot: ROI fitur + PCA scatter ----
    fig = plt.figure(figsize=(20, 10))
    fig.suptitle(
        f"Feature Extraction — {SAMPLE_PATIENT} — ROI 64×64 Slice #{mid_idx}",
        fontweight="bold", fontsize=14, color="#e94560",
    )

    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.35, wspace=0.3)

    # Baris atas: FLAIR ROI + 6 fitur (hanya 3 ditampilkan + seg)
    ax_flair = fig.add_subplot(gs[0, 0])
    ax_flair.imshow(flair_roi, cmap="gray", aspect="equal")
    ax_flair.set_title("FLAIR ROI", fontweight="bold")
    ax_flair.axis("off")

    for k, idx in enumerate([0, 3, 5]):  # mean, kontras, homogenitas
        ax = fig.add_subplot(gs[0, k + 1])
        im = ax.imshow(features[:, :, idx], cmap="inferno", aspect="equal")
        ax.set_title(FEATURE_NAMES[idx], fontweight="bold")
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Baris bawah: PC1 map, PC2 map, scatter, seg overlay
    for k in range(min(3, projected.shape[2])):
        ax = fig.add_subplot(gs[1, k])
        im = ax.imshow(projected[:, :, k], cmap="coolwarm", aspect="equal")
        ax.set_title(f"PC{k+1}", fontweight="bold")
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax_seg = fig.add_subplot(gs[1, 3])
    ax_seg.imshow(flair_roi, cmap="gray", aspect="equal")
    seg_display = np.ma.masked_where(seg_roi == 0, seg_roi)
    ax_seg.imshow(seg_display, cmap="Set1", alpha=0.5, aspect="equal",
                  vmin=0, vmax=4)
    ax_seg.set_title("Seg Overlay", fontweight="bold")
    ax_seg.axis("off")

    path = OUTPUT_DIR / "test_feature_real_patient.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  ✓ Plot disimpan: {path}")

    # ---- Scatter PC1 vs PC2 dengan label asli ----
    fig2, ax2 = plt.subplots(1, 1, figsize=(10, 8))
    fig2.suptitle(
        f"PCA Scatter — {SAMPLE_PATIENT} ROI (Ground Truth Overlay)",
        fontweight="bold", fontsize=14, color="#e94560",
    )

    pc1 = projected[:, :, 0].flatten()
    pc2 = projected[:, :, 1].flatten()
    labels = seg_roi.flatten().astype(int)

    label_cfg = {
        0: ("Background", "#4a90d9"),
        1: ("Necrosis", "#e94560"),
        2: ("Edema", "#f5a623"),
        4: ("Enhancing", "#7ed321"),
    }

    for lbl in sorted(np.unique(labels)):
        mask = labels == lbl
        name, color = label_cfg.get(int(lbl), (f"Label {lbl}", "#cccccc"))
        ax2.scatter(pc1[mask], pc2[mask], c=color, label=name,
                    alpha=0.5, s=10, edgecolors="none")

    ax2.scatter(centroids[:, 0], centroids[:, 1],
                c="red", marker="X", s=300, edgecolors="white",
                linewidths=2, zorder=10, label="Centroid Awal")

    ax2.set_xlabel("PC1", fontsize=12, fontweight="bold")
    ax2.set_ylabel("PC2", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=10, framealpha=0.8)
    ax2.grid(alpha=0.2, color="#555555")

    plt.tight_layout()
    path2 = OUTPUT_DIR / "test_feature_real_scatter.png"
    fig2.savefig(path2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"  ✓ Plot disimpan: {path2}")


# ===================================================================
# MAIN
# ===================================================================
def main():
    print("\n" + "█" * 60)
    print("█  TEST FEATURE EXTRACTION — Modul 2")
    print("█" * 60)

    # Buat data dummy (ukuran kecil agar GLCM cepat)
    print(f"\n  Menggunakan shape dummy: {DUMMY_SHAPE}")
    print(f"  (Shape kecil untuk efisiensi GLCM per-piksel)")
    dummy = create_dummy_patient_data(shape=DUMMY_SHAPE, seed=42)

    # Test 1: Patch features
    features, slice_norm, slice_orig = test_patch_features(dummy)

    # Test 2: PCA
    projected, pca_model, centroids, explained = test_pca(features)

    # Test 3: Centroid (2, 3, 4 cluster)
    test_centroids(projected)

    # Test 4: Analisis PCA
    analysis = test_analysis(pca_model, features)

    # Test 5: Scatter plot
    test_scatter_plot(projected, centroids, dummy)

    # Test 6: Variance & loadings plots
    test_variance_and_loadings(analysis)

    # Test 7: Convenience pipeline
    test_convenience_pipeline(dummy)

    # BONUS: Dataset asli
    test_real_dataset()

    separator("SEMUA TEST MODUL 2 SELESAI ✓")
    print(f"\n  Output plots disimpan di: {OUTPUT_DIR}/")
    print(f"  File feature_extraction yang dihasilkan:")
    for f in sorted(OUTPUT_DIR.glob("test_feature_*.png")):
        size_kb = f.stat().st_size / 1024
        print(f"    • {f.name} ({size_kb:.0f} KB)")
    print()


if __name__ == "__main__":
    main()
