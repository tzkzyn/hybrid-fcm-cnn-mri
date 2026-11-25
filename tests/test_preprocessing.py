#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_preprocessing.py — Skrip Uji Komprehensif untuk Modul Preprocessing

Skrip ini menguji seluruh fungsi di src/preprocessing.py:
  1. Membuat data dummy (tanpa dataset asli)
  2. Menampilkan slice FLAIR sebelum & sesudah normalisasi
  3. Menampilkan 3 jenis noise: gaussian, salt_pepper, motion_blur
  4. Menampilkan brain mask
  5. Mencetak shape semua array
  6. BONUS: Jika dataset BraTS 2020 asli ditemukan, memuat satu pasien
           dan menampilkan slice tengah semua modalitas

Output: Figure matplotlib disimpan sebagai PNG di folder output/.

Author  : [Peneliti]
Created : 2026-05-01
"""

from __future__ import annotations

import sys
import os
from pathlib import Path

# Pastikan src/ bisa di-import
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Backend non-interaktif agar bisa jalan tanpa display
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from src.preprocessing import (
    add_noise,
    bias_field_correction,
    create_dummy_patient_data,
    extract_slice,
    get_brain_mask,
    load_patient_modalities,
    normalize_intensity,
    BRATS_SHAPE,
)

# ---------------------------------------------------------------------------
# Konfigurasi
# ---------------------------------------------------------------------------
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

DATASET_PATH = PROJECT_ROOT / "data" / "Brats2020" / "training"
SAMPLE_PATIENT = "BraTS20_Training_001"

# Style plot
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

CMAP_MRI = "gray"
CMAP_MASK = "hot"
CMAP_SEG = "Set1"


def separator(title: str) -> None:
    """Cetak separator visual ke terminal."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ===================================================================
# TEST 1: Data Dummy + Print Shapes
# ===================================================================
def test_dummy_data() -> dict:
    """Buat data dummy dan cetak informasi shape/dtype/range."""
    separator("TEST 1: Membuat Data Dummy")

    dummy = create_dummy_patient_data(shape=BRATS_SHAPE, seed=42)

    print(f"\n  Jumlah modalitas: {len(dummy)}")
    print(f"  Keys: {list(dummy.keys())}\n")

    for key, arr in dummy.items():
        unique_vals = np.unique(arr).shape[0] if key == "seg" else "—"
        print(
            f"  {key:6s} | shape={str(arr.shape):18s} | "
            f"dtype={str(arr.dtype):8s} | "
            f"range=[{arr.min():7.3f}, {arr.max():7.3f}] | "
            f"unique_labels={unique_vals}"
        )

    # Verifikasi shape
    for key in ("flair", "t1", "t1ce", "t2"):
        assert dummy[key].shape == BRATS_SHAPE, (
            f"Shape {key} salah: {dummy[key].shape} != {BRATS_SHAPE}"
        )
        assert dummy[key].dtype == np.float32
    assert dummy["seg"].shape == BRATS_SHAPE
    assert dummy["seg"].dtype == np.int32

    print("\n  ✓ Semua shape dan dtype terverifikasi!")
    return dummy


# ===================================================================
# TEST 2: Normalisasi Intensitas (sebelum vs sesudah)
# ===================================================================
def test_normalization(dummy: dict) -> np.ndarray:
    """Tampilkan slice FLAIR sebelum & sesudah normalisasi."""
    separator("TEST 2: Normalisasi Intensitas")

    flair_raw = dummy["flair"]
    flair_norm = normalize_intensity(flair_raw, lower_percent=1, upper_percent=99)

    print(f"  Raw   FLAIR  range: [{flair_raw.min():.4f}, {flair_raw.max():.4f}]")
    print(f"  Norm  FLAIR  range: [{flair_norm.min():.4f}, {flair_norm.max():.4f}]")

    # Ambil slice tengah (axial)
    mid_idx = flair_raw.shape[0] // 2
    raw_slice = extract_slice(flair_raw, slice_idx=mid_idx, axis=0)
    norm_slice = extract_slice(flair_norm, slice_idx=mid_idx, axis=0)

    print(f"  Slice index: {mid_idx} (axial)")
    print(f"  Raw  slice shape: {raw_slice.shape}")
    print(f"  Norm slice shape: {norm_slice.shape}")

    # ---- Plot ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        "FLAIR — Sebelum vs Sesudah Normalisasi Intensitas",
        fontweight="bold", fontsize=14, color="#e94560",
    )

    im0 = axes[0].imshow(raw_slice, cmap=CMAP_MRI, aspect="equal")
    axes[0].set_title("Sebelum Normalisasi", fontweight="bold")
    axes[0].axis("off")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(norm_slice, cmap=CMAP_MRI, aspect="equal",
                         vmin=0, vmax=1)
    axes[1].set_title("Sesudah Normalisasi [0, 1]", fontweight="bold")
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    save_path = OUTPUT_DIR / "test_normalization.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  ✓ Plot disimpan: {save_path}")

    return flair_norm


# ===================================================================
# TEST 3: Noise Augmentation (3 jenis)
# ===================================================================
def test_noise(flair_norm: np.ndarray) -> None:
    """Tampilkan contoh 3 jenis noise pada slice FLAIR."""
    separator("TEST 3: Noise Augmentation")

    mid_slice = extract_slice(flair_norm, axis=0)

    noise_configs = [
        ("gaussian", 0.08),
        ("salt_pepper", 0.05),
        ("motion_blur", 0.15),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        "Augmentasi Noise — Uji Robustness",
        fontweight="bold", fontsize=14, color="#e94560",
    )

    for i, (ntype, level) in enumerate(noise_configs):
        noisy = add_noise(mid_slice, noise_type=ntype, level=level)
        print(
            f"  {ntype:15s} (level={level:.2f}) | "
            f"shape={noisy.shape} | range=[{noisy.min():.4f}, {noisy.max():.4f}]"
        )

        im = axes[i].imshow(noisy, cmap=CMAP_MRI, aspect="equal",
                            vmin=0, vmax=1)
        axes[i].set_title(
            f"{ntype.replace('_', ' ').title()}\n(level={level})",
            fontweight="bold",
        )
        axes[i].axis("off")
        plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)

    plt.tight_layout()
    save_path = OUTPUT_DIR / "test_noise.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  ✓ Plot disimpan: {save_path}")


# ===================================================================
# TEST 4: Brain Mask
# ===================================================================
def test_brain_mask(flair_norm: np.ndarray) -> None:
    """Tampilkan brain mask dari volume FLAIR ternormalisasi."""
    separator("TEST 4: Brain Mask")

    mask = get_brain_mask(flair_norm, threshold=0.1, closing_radius=5)
    print(f"  Mask shape : {mask.shape}")
    print(f"  Mask dtype : {mask.dtype}")
    print(f"  Mask unique: {np.unique(mask)}")
    print(f"  Brain pct  : {100 * mask.sum() / mask.size:.2f}%")

    # Ambil slice tengah
    mid_idx = mask.shape[0] // 2
    mask_slice = extract_slice(mask, slice_idx=mid_idx, axis=0)
    flair_slice = extract_slice(flair_norm, slice_idx=mid_idx, axis=0)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        "Brain Mask — Thresholding + Morphological Closing",
        fontweight="bold", fontsize=14, color="#e94560",
    )

    axes[0].imshow(flair_slice, cmap=CMAP_MRI, aspect="equal")
    axes[0].set_title("FLAIR (normalized)", fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(mask_slice, cmap=CMAP_MASK, aspect="equal",
                   vmin=0, vmax=1)
    axes[1].set_title("Brain Mask", fontweight="bold")
    axes[1].axis("off")

    # Overlay
    axes[2].imshow(flair_slice, cmap=CMAP_MRI, aspect="equal")
    axes[2].imshow(mask_slice, cmap="Reds", alpha=0.3, aspect="equal")
    axes[2].set_title("Overlay (FLAIR + Mask)", fontweight="bold")
    axes[2].axis("off")

    plt.tight_layout()
    save_path = OUTPUT_DIR / "test_brain_mask.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  ✓ Plot disimpan: {save_path}")


# ===================================================================
# TEST 5: Extract Slice (3 orientasi)
# ===================================================================
def test_extract_slice(dummy: dict) -> None:
    """Test ekstraksi slice pada 3 orientasi: axial, coronal, sagittal."""
    separator("TEST 5: Extract Slice (3 Orientasi)")

    flair = dummy["flair"]
    orientations = [
        (0, "Axial (axis=0)"),
        (1, "Coronal (axis=1)"),
        (2, "Sagittal (axis=2)"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        "Ekstraksi Slice — 3 Orientasi Anatomi",
        fontweight="bold", fontsize=14, color="#e94560",
    )

    for i, (ax, label) in enumerate(orientations):
        sliced = extract_slice(flair, axis=ax)
        print(f"  {label:25s} -> shape={sliced.shape}")

        axes[i].imshow(sliced, cmap=CMAP_MRI, aspect="auto")
        axes[i].set_title(label, fontweight="bold")
        axes[i].axis("off")

    plt.tight_layout()
    save_path = OUTPUT_DIR / "test_slice_orientations.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  ✓ Plot disimpan: {save_path}")


# ===================================================================
# BONUS: Muat dataset asli (jika tersedia)
# ===================================================================
def test_real_dataset() -> None:
    """Muat satu pasien dari BraTS 2020 dan tampilkan semua modalitas."""
    separator("BONUS: Dataset BraTS 2020 Asli")

    patient_path = DATASET_PATH / SAMPLE_PATIENT
    if not patient_path.is_dir():
        print(f"  ⚠ Dataset asli tidak ditemukan di: {patient_path}")
        print("  ⚠ Lewati tes dataset asli.")
        return

    print(f"  Dataset ditemukan! Memuat pasien: {SAMPLE_PATIENT}")

    # Muat semua modalitas
    data = load_patient_modalities(patient_path)

    print(f"\n  Modalitas yang dimuat:")
    for key, arr in data.items():
        print(
            f"    {key:6s} | shape={str(arr.shape):18s} | "
            f"dtype={str(arr.dtype):8s} | "
            f"range=[{arr.min():8.2f}, {arr.max():8.2f}]"
        )

    # Normalisasi setiap modalitas (kecuali seg)
    data_norm = {}
    for key in ("flair", "t1", "t1ce", "t2"):
        data_norm[key] = normalize_intensity(data[key])
    data_norm["seg"] = data["seg"]

    # Plot slice tengah semua modalitas + segmentasi
    mid_idx = data["flair"].shape[0] // 2
    print(f"\n  Menampilkan slice tengah (idx={mid_idx}) semua modalitas...")

    fig = plt.figure(figsize=(20, 8))
    fig.suptitle(
        f"BraTS 2020 — {SAMPLE_PATIENT} — Slice Axial #{mid_idx}",
        fontweight="bold", fontsize=16, color="#e94560",
    )

    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.3, wspace=0.15)

    # Baris atas: 4 modalitas (raw)
    modalities_order = ["flair", "t1", "t1ce", "t2"]
    for i, mod in enumerate(modalities_order):
        ax = fig.add_subplot(gs[0, i])
        raw_slice = extract_slice(data[mod], slice_idx=mid_idx, axis=0)
        ax.imshow(raw_slice, cmap=CMAP_MRI, aspect="equal")
        ax.set_title(f"{mod.upper()} (raw)", fontweight="bold")
        ax.axis("off")

    # Baris bawah: 4 modalitas (normalized) — kecuali kolom 4 jadi seg
    for i, mod in enumerate(modalities_order[:3]):
        ax = fig.add_subplot(gs[1, i])
        norm_slice = extract_slice(data_norm[mod], slice_idx=mid_idx, axis=0)
        ax.imshow(norm_slice, cmap=CMAP_MRI, aspect="equal", vmin=0, vmax=1)
        ax.set_title(f"{mod.upper()} (norm)", fontweight="bold")
        ax.axis("off")

    # Segmentasi ground truth
    ax_seg = fig.add_subplot(gs[1, 3])
    seg_slice = extract_slice(data["seg"], slice_idx=mid_idx, axis=0)
    im_seg = ax_seg.imshow(seg_slice, cmap=CMAP_SEG, aspect="equal",
                           vmin=0, vmax=3, interpolation="nearest")
    ax_seg.set_title("Ground Truth Seg", fontweight="bold")
    ax_seg.axis("off")

    # Colorbar untuk segmentasi
    cbar = plt.colorbar(im_seg, ax=ax_seg, fraction=0.046, pad=0.04,
                        ticks=[0, 1, 2, 3])
    cbar.ax.set_yticklabels(["BG", "Necrosis", "Edema", "Enhancing"])
    cbar.ax.tick_params(labelsize=8)

    save_path = OUTPUT_DIR / "test_real_patient.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  ✓ Plot disimpan: {save_path}")

    # --- Brain mask pada data asli ---
    mask = get_brain_mask(data_norm["flair"], threshold=0.05)
    mask_slice = extract_slice(mask, slice_idx=mid_idx, axis=0)
    flair_slice = extract_slice(data_norm["flair"], slice_idx=mid_idx, axis=0)

    fig2, axes2 = plt.subplots(1, 2, figsize=(10, 5))
    fig2.suptitle(
        f"Brain Mask — {SAMPLE_PATIENT} — Slice #{mid_idx}",
        fontweight="bold", fontsize=14, color="#e94560",
    )

    axes2[0].imshow(flair_slice, cmap=CMAP_MRI, aspect="equal")
    axes2[0].set_title("FLAIR (normalized)", fontweight="bold")
    axes2[0].axis("off")

    axes2[1].imshow(flair_slice, cmap=CMAP_MRI, aspect="equal")
    axes2[1].imshow(mask_slice, cmap="Reds", alpha=0.3, aspect="equal")
    axes2[1].set_title("FLAIR + Brain Mask Overlay", fontweight="bold")
    axes2[1].axis("off")

    plt.tight_layout()
    save_path2 = OUTPUT_DIR / "test_real_brain_mask.png"
    fig2.savefig(save_path2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"  ✓ Plot disimpan: {save_path2}")

    # --- Segmentasi label distribution ---
    seg = data["seg"]
    print(f"\n  Distribusi label segmentasi (seluruh volume):")
    for label_val, label_name in [(0, "Background"), (1, "Necrosis"),
                                   (2, "Edema"), (3, "Enhancing Tumor")]:
        count = int((seg == label_val).sum())
        pct = 100.0 * count / seg.size
        print(f"    Label {label_val} ({label_name:18s}): "
              f"{count:>10,} voxels ({pct:5.2f}%)")


# ===================================================================
# MAIN
# ===================================================================
def main():
    """Jalankan semua tes preprocessing secara berurutan."""
    print("\n" + "█" * 60)
    print("█  TEST PREPROCESSING — BraTS 2020 MRI Pipeline")
    print("█" * 60)

    # Test 1: Data dummy
    dummy = test_dummy_data()

    # Test 2: Normalisasi
    flair_norm = test_normalization(dummy)

    # Test 3: Noise
    test_noise(flair_norm)

    # Test 4: Brain mask
    test_brain_mask(flair_norm)

    # Test 5: Extract slice (3 orientasi)
    test_extract_slice(dummy)

    # BONUS: Dataset asli
    test_real_dataset()

    separator("SEMUA TEST SELESAI ✓")
    print(f"\n  Output plots disimpan di: {OUTPUT_DIR}/")
    print(f"  File yang dihasilkan:")
    for f in sorted(OUTPUT_DIR.glob("test_*.png")):
        size_kb = f.stat().st_size / 1024
        print(f"    • {f.name} ({size_kb:.0f} KB)")
    print()


if __name__ == "__main__":
    main()
