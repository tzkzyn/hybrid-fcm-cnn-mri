#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
preprocessing.py — Modul 1: Preprocessing Pipeline untuk BraTS 2020 MRI Data

Bagian dari riset:
  "An AI-Driven Hybrid Fuzzy C-Means Framework for Robust Brain Tumor
   MRI Segmentation via PCA-Based Feature Reduction"

Modul ini menyediakan fungsi-fungsi preprocessing untuk data MRI
multimodal BraTS 2020 dalam format NIfTI (.nii), meliputi:
  - Pembacaan file NIfTI
  - N4ITK bias field correction
  - Normalisasi intensitas berbasis persentil
  - Augmentasi noise (Gaussian, salt & pepper, motion blur)
  - Pemuatan otomatis semua modalitas per pasien
  - Ekstraksi brain mask via thresholding + morfologi
  - Pembuatan data dummy untuk pengujian
  - Ekstraksi slice 2D dari volume 3D

Author  : [Peneliti]
Created : 2026-05-01
Python  : 3.9+
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import SimpleITK as sitk
from scipy.ndimage import binary_closing, uniform_filter

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
MODALITY_SUFFIXES = ("flair", "t1", "t1ce", "t2")
SEG_SUFFIX = "seg"
BRATS_SHAPE = (155, 240, 240)  # (slices, height, width)


# ===================================================================
# 1. load_nifti
# ===================================================================
def load_nifti(path: Union[str, Path]) -> np.ndarray:
    """Membaca file NIfTI (.nii) dan mengembalikan array NumPy float32 3-D.

    Menggunakan SimpleITK untuk kompatibilitas maksimal dengan format
    NIfTI-1 yang dipakai BraTS 2020.

    Parameters
    ----------
    path : str | Path
        Path absolut atau relatif ke file .nii.

    Returns
    -------
    np.ndarray
        Volume 3-D berdimensi (depth, height, width) dengan dtype float32.

    Raises
    ------
    FileNotFoundError
        Jika file tidak ditemukan di path yang diberikan.
    RuntimeError
        Jika SimpleITK gagal membaca file (format korup, dsb.).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File NIfTI tidak ditemukan: {path}")

    try:
        sitk_image = sitk.ReadImage(str(path))
        array = sitk.GetArrayFromImage(sitk_image)  # shape: (D, H, W)
        logger.info(
            "Loaded %s  |  shape=%s  dtype=%s",
            path.name, array.shape, array.dtype,
        )
        return array.astype(np.float32)
    except Exception as exc:
        raise RuntimeError(
            f"Gagal membaca file NIfTI '{path}': {exc}"
        ) from exc


# ===================================================================
# 2. bias_field_correction
# ===================================================================
def bias_field_correction(
    image: np.ndarray,
    shrink_factor: int = 4,
    num_iterations: Tuple[int, ...] = (50, 50, 50, 50),
    num_fitting_levels: int = 4,
) -> np.ndarray:
    """Menerapkan N4ITK Bias Field Correction pada volume MRI 3-D.

    Bias field (inhomogenitas medan magnet) menyebabkan variasi
    intensitas artifisial di seluruh volume. Koreksi N4ITK mengestimasi
    dan menghapus bias field tersebut secara iteratif.

    Parameters
    ----------
    image : np.ndarray
        Volume MRI 3-D (float32). Nilai harus ≥ 0.
    shrink_factor : int, default 4
        Faktor down-sampling untuk mempercepat koreksi.
        Semakin besar = semakin cepat tapi kurang presisi.
    num_iterations : tuple of int, default (50, 50, 50, 50)
        Jumlah iterasi per level resolusi.
    num_fitting_levels : int, default 4
        Jumlah level multi-resolusi pada B-spline fitting.

    Returns
    -------
    np.ndarray
        Volume MRI 3-D (float32) yang sudah dikoreksi bias field-nya.

    Raises
    ------
    ValueError
        Jika input bukan array 3-D.
    RuntimeError
        Jika proses N4ITK gagal.
    """
    if image.ndim != 3:
        raise ValueError(
            f"bias_field_correction membutuhkan array 3-D, "
            f"diterima ndim={image.ndim}"
        )

    try:
        # Pastikan float32 dan non-negatif
        img = np.clip(image, 0, None).astype(np.float32)

        # Konversi ke SimpleITK Image
        sitk_image = sitk.GetImageFromArray(img)
        sitk_image = sitk.Cast(sitk_image, sitk.sitkFloat32)

        # Buat mask: area dengan intensitas > 0
        mask_image = sitk.OtsuThreshold(sitk_image, 0, 1, 200)

        # Shrink untuk kecepatan
        shrunk_image = sitk.Shrink(
            sitk_image, [shrink_factor] * sitk_image.GetDimension()
        )
        shrunk_mask = sitk.Shrink(
            mask_image, [shrink_factor] * mask_image.GetDimension()
        )

        # N4 Bias Field Correction
        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        corrector.SetMaximumNumberOfIterations(list(num_iterations))
        corrector.SetNumberOfFittingLevels(num_fitting_levels)

        _ = corrector.Execute(shrunk_image, shrunk_mask)

        # Ambil log bias field dan terapkan ke resolusi penuh
        log_bias_field = corrector.GetLogBiasFieldAsImage(sitk_image)
        corrected = sitk_image / sitk.Exp(log_bias_field)

        result = sitk.GetArrayFromImage(corrected).astype(np.float32)
        logger.info(
            "Bias field correction selesai  |  "
            "input range=[%.2f, %.2f] -> output range=[%.2f, %.2f]",
            img.min(), img.max(), result.min(), result.max(),
        )
        return result

    except Exception as exc:
        raise RuntimeError(
            f"N4ITK bias field correction gagal: {exc}"
        ) from exc


# ===================================================================
# 3. normalize_intensity
# ===================================================================
def normalize_intensity(
    image: np.ndarray,
    lower_percent: float = 1.0,
    upper_percent: float = 99.0,
) -> np.ndarray:
    """Normalisasi intensitas volume MRI ke rentang [0, 1].

    Strategi:
      1. Hitung persentil bawah dan atas dari voxel non-zero.
      2. Clip intensitas ke rentang [p_low, p_high].
      3. Min-max scaling ke [0, 1].

    Metode ini robust terhadap outlier intensitas yang lazim pada MRI.

    Parameters
    ----------
    image : np.ndarray
        Volume MRI 3-D (float32).
    lower_percent : float, default 1.0
        Persentil bawah untuk clipping (0–100).
    upper_percent : float, default 99.0
        Persentil atas untuk clipping (0–100).

    Returns
    -------
    np.ndarray
        Volume ternormalisasi [0, 1] berdtype float32.
        Voxel background (aslinya 0) tetap 0.

    Raises
    ------
    ValueError
        Jika lower_percent >= upper_percent.
    """
    if lower_percent >= upper_percent:
        raise ValueError(
            f"lower_percent ({lower_percent}) harus < "
            f"upper_percent ({upper_percent})"
        )

    result = image.copy().astype(np.float32)

    # Ambil voxel non-zero saja (abaikan background)
    non_zero_mask = result > 0
    if non_zero_mask.sum() == 0:
        logger.warning("Volume seluruhnya nol — normalisasi dilewati.")
        return result

    non_zero_vals = result[non_zero_mask]
    p_low = np.percentile(non_zero_vals, lower_percent)
    p_high = np.percentile(non_zero_vals, upper_percent)

    # Clip dan scale
    result[non_zero_mask] = np.clip(non_zero_vals, p_low, p_high)
    denom = p_high - p_low
    if denom < 1e-8:
        logger.warning("Rentang persentil sangat kecil (≈0). Mengembalikan nol.")
        result[non_zero_mask] = 0.0
    else:
        result[non_zero_mask] = (result[non_zero_mask] - p_low) / denom

    logger.info(
        "Normalisasi intensitas  |  percentile [%.1f, %.1f] = [%.2f, %.2f]  "
        "-> range [0, 1]",
        lower_percent, upper_percent, p_low, p_high,
    )
    return result


# ===================================================================
# 4. add_noise
# ===================================================================
def add_noise(
    image: np.ndarray,
    noise_type: str = "gaussian",
    level: float = 0.05,
) -> np.ndarray:
    """Menambahkan noise sintetis ke volume/slice MRI untuk uji robustness.

    Tiga jenis noise didukung:
      - ``'gaussian'``     : Additive Gaussian noise ~ N(0, level).
      - ``'salt_pepper'``  : Salt-and-pepper noise (fraction = level).
      - ``'motion_blur'``  : Motion blur 1-D via uniform filter pada axis
                             terakhir (kernel size = int(level * 20) + 3).

    Parameters
    ----------
    image : np.ndarray
        Array 2-D atau 3-D berdtype float32, diharapkan dalam rentang [0, 1].
    noise_type : str
        Salah satu dari ``'gaussian'``, ``'salt_pepper'``,
        ``'motion_blur'``.
    level : float, default 0.05
        Intensitas noise.
        - Gaussian  : standar deviasi noise.
        - S&P       : fraksi piksel yang terkena (0–1).
        - Motion    : menentukan ukuran kernel (kernel = int(level*20) + 3).

    Returns
    -------
    np.ndarray
        Array dengan noise yang ditambahkan, di-clip ke [0, 1].

    Raises
    ------
    ValueError
        Jika ``noise_type`` tidak dikenali.
    """
    noisy = image.copy().astype(np.float32)
    noise_type = noise_type.lower().strip()

    if noise_type == "gaussian":
        rng = np.random.default_rng()
        noise = rng.normal(loc=0.0, scale=level, size=image.shape).astype(
            np.float32
        )
        noisy += noise

    elif noise_type == "salt_pepper":
        rng = np.random.default_rng()
        random_matrix = rng.random(image.shape)
        # Salt (putih)
        noisy[random_matrix < (level / 2.0)] = 1.0
        # Pepper (hitam)
        noisy[random_matrix > (1.0 - level / 2.0)] = 0.0

    elif noise_type == "motion_blur":
        kernel_size = int(level * 20) + 3
        # Blur pada axis terakhir (horizontal) untuk simulasi gerakan
        if noisy.ndim == 2:
            noisy = uniform_filter(noisy, size=(1, kernel_size))
        elif noisy.ndim == 3:
            noisy = uniform_filter(noisy, size=(1, 1, kernel_size))
        else:
            noisy = uniform_filter(noisy, size=kernel_size)
    else:
        raise ValueError(
            f"noise_type tidak dikenali: '{noise_type}'. "
            f"Gunakan 'gaussian', 'salt_pepper', atau 'motion_blur'."
        )

    noisy = np.clip(noisy, 0.0, 1.0)
    logger.info(
        "Noise '%s' ditambahkan (level=%.4f)  |  shape=%s",
        noise_type, level, noisy.shape,
    )
    return noisy


# ===================================================================
# 5. load_patient_modalities
# ===================================================================
def load_patient_modalities(
    patient_folder: Union[str, Path],
) -> Dict[str, np.ndarray]:
    """Memuat seluruh file .nii dari folder satu pasien BraTS 2020.

    Secara otomatis mendeteksi dan memuat 4 modalitas MRI (flair, t1,
    t1ce, t2) serta ground truth segmentasi (seg) berdasarkan suffix
    nama file.

    Parameters
    ----------
    patient_folder : str | Path
        Path ke folder pasien, misal:
        ``data/Brats2020/training/BraTS20_Training_001/``

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary dengan key = nama modalitas ('flair', 't1', 't1ce',
        't2', 'seg') dan value = array NumPy float32 3-D.

    Raises
    ------
    FileNotFoundError
        Jika folder tidak ada atau file modalitas tidak ditemukan.
    RuntimeError
        Jika gagal membaca salah satu file NIfTI.

    Examples
    --------
    >>> data = load_patient_modalities("data/Brats2020/training/BraTS20_Training_001")
    >>> data.keys()
    dict_keys(['flair', 't1', 't1ce', 't2', 'seg'])
    >>> data['flair'].shape
    (155, 240, 240)
    """
    patient_folder = Path(patient_folder)
    if not patient_folder.is_dir():
        raise FileNotFoundError(
            f"Folder pasien tidak ditemukan: {patient_folder}"
        )

    patient_id = patient_folder.name  # e.g. "BraTS20_Training_001"
    all_suffixes = list(MODALITY_SUFFIXES) + [SEG_SUFFIX]
    modalities: Dict[str, np.ndarray] = {}

    for suffix in all_suffixes:
        filename = f"{patient_id}_{suffix}.nii"
        filepath = patient_folder / filename

        if not filepath.exists():
            # Coba juga .nii.gz sebagai fallback
            filepath_gz = patient_folder / f"{filename}.gz"
            if filepath_gz.exists():
                filepath = filepath_gz
                logger.info(
                    "File .nii tidak ada, menggunakan .nii.gz: %s",
                    filepath.name,
                )
            else:
                raise FileNotFoundError(
                    f"File modalitas tidak ditemukan: {filepath} "
                    f"(juga tidak ada .nii.gz)"
                )

        modalities[suffix] = load_nifti(filepath)

    logger.info(
        "Semua modalitas pasien '%s' berhasil dimuat: %s",
        patient_id, list(modalities.keys()),
    )
    return modalities


# ===================================================================
# 6. get_brain_mask
# ===================================================================
def get_brain_mask(
    image: np.ndarray,
    threshold: float = 0.1,
    closing_radius: int = 5,
) -> np.ndarray:
    """Membuat binary brain mask dari volume MRI via thresholding + morfologi.

    Strategi:
      1. Threshold: voxel > threshold → 1, sisanya 0.
      2. Morphological closing (binary) untuk menutup lubang kecil
         di dalam area otak.

    Parameters
    ----------
    image : np.ndarray
        Volume MRI 3-D, idealnya sudah ternormalisasi [0, 1].
    threshold : float, default 0.1
        Nilai ambang batas. Voxel di atas threshold dianggap otak.
    closing_radius : int, default 5
        Radius structuring element untuk morphological closing.
        Semakin besar = lubang yang ditutup semakin besar.

    Returns
    -------
    np.ndarray
        Binary mask 3-D (dtype uint8, nilai 0 atau 1) berukuran sama
        dengan input.
    """
    # Threshold
    mask = (image > threshold).astype(np.uint8)

    # Morphological closing (per slice untuk efisiensi)
    struct = np.ones((1, closing_radius, closing_radius), dtype=bool)
    mask_closed = binary_closing(mask, structure=struct, iterations=2)
    mask_closed = mask_closed.astype(np.uint8)

    brain_voxels = mask_closed.sum()
    total_voxels = mask_closed.size
    pct = 100.0 * brain_voxels / total_voxels
    logger.info(
        "Brain mask  |  threshold=%.2f  closing_radius=%d  "
        "brain_voxels=%d (%.1f%%)",
        threshold, closing_radius, brain_voxels, pct,
    )
    return mask_closed


# ===================================================================
# 7. create_dummy_patient_data
# ===================================================================
def create_dummy_patient_data(
    shape: Tuple[int, int, int] = BRATS_SHAPE,
    num_classes: int = 4,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """Membuat data dummy yang menyerupai volume MRI BraTS untuk testing.

    Menghasilkan 4 modalitas (flair, t1, t1ce, t2) dan 1 segmentasi
    ground truth (seg) dengan struktur tumor sintetis berupa blob
    elipsoid di tengah volume.

    Parameters
    ----------
    shape : tuple of int, default (155, 240, 240)
        Dimensi volume (depth, height, width). Defaultnya sesuai
        BraTS 2020.
    num_classes : int, default 4
        Jumlah kelas segmentasi (0=bg, 1=necrosis, 2=edema,
        3=enhancing tumor).
    seed : int, default 42
        Random seed untuk reprodusibilitas.

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary berisi:
        - 'flair', 't1', 't1ce', 't2': array float32 [0, 1]
        - 'seg': array int32 dengan label {0, 1, 2, 3}

    Notes
    -----
    Data dummy ini BUKAN data medis sungguhan. Digunakan hanya untuk
    verifikasi fungsionalitas pipeline tanpa membutuhkan dataset asli.
    """
    rng = np.random.default_rng(seed)
    D, H, W = shape
    center = np.array([D // 2, H // 2, W // 2])

    # ---- Buat koordinat grid ----
    zz, yy, xx = np.mgrid[0:D, 0:H, 0:W]

    # ---- Buat ellipsoid bertingkat untuk simulasi tumor ----
    # Jarak ternormalisasi dari pusat
    dist = np.sqrt(
        ((zz - center[0]) / (D * 0.15)) ** 2
        + ((yy - center[1]) / (H * 0.10)) ** 2
        + ((xx - center[2]) / (W * 0.10)) ** 2
    )

    # Segmentasi berlapis: enhancing core < necrosis < edema < brain
    seg = np.zeros(shape, dtype=np.int32)
    seg[dist < 3.0] = 2   # Edema (label 2) — lapisan terluar tumor
    seg[dist < 2.0] = 1   # Necrosis (label 1) — lapisan tengah
    seg[dist < 1.0] = 3   # Enhancing tumor (label 3) — inti

    # ---- Buat brain region (ellipsoid besar) ----
    brain_dist = np.sqrt(
        ((zz - center[0]) / (D * 0.45)) ** 2
        + ((yy - center[1]) / (H * 0.40)) ** 2
        + ((xx - center[2]) / (W * 0.40)) ** 2
    )
    brain_region = brain_dist < 1.0

    # ---- Generate modalitas dengan karakteristik berbeda ----
    modalities: Dict[str, np.ndarray] = {}
    base_intensities = {
        "flair": {"brain": 0.4, "edema": 0.85, "necrosis": 0.50, "enhancing": 0.70},
        "t1":    {"brain": 0.5, "edema": 0.45, "necrosis": 0.35, "enhancing": 0.55},
        "t1ce":  {"brain": 0.5, "edema": 0.50, "necrosis": 0.30, "enhancing": 0.90},
        "t2":    {"brain": 0.4, "edema": 0.80, "necrosis": 0.60, "enhancing": 0.55},
    }

    for mod_name, intensities in base_intensities.items():
        vol = np.zeros(shape, dtype=np.float32)
        # Jaringan otak normal
        vol[brain_region] = intensities["brain"]
        # Overlay tumor
        vol[seg == 2] = intensities["edema"]
        vol[seg == 1] = intensities["necrosis"]
        vol[seg == 3] = intensities["enhancing"]
        # Tambah noise realistis
        noise = rng.normal(0, 0.03, size=shape).astype(np.float32)
        vol += noise * brain_region.astype(np.float32)
        vol = np.clip(vol, 0.0, 1.0)
        modalities[mod_name] = vol

    modalities["seg"] = seg

    logger.info(
        "Data dummy dibuat  |  shape=%s  |  label dist: %s",
        shape,
        {i: int((seg == i).sum()) for i in range(num_classes)},
    )
    return modalities


# ===================================================================
# 8. extract_slice
# ===================================================================
def extract_slice(
    volume: np.ndarray,
    slice_idx: Optional[int] = None,
    axis: int = 0,
) -> np.ndarray:
    """Mengekstrak satu slice 2-D dari volume 3-D.

    Parameters
    ----------
    volume : np.ndarray
        Volume 3-D (depth, height, width).
    slice_idx : int | None, default None
        Indeks slice yang diambil. Jika None, diambil slice tengah
        pada axis yang dipilih.
    axis : int, default 0
        Axis pengambilan slice:
        - 0: axial   (ambil dari dimensi depth / slice)
        - 1: coronal (ambil dari dimensi height)
        - 2: sagittal (ambil dari dimensi width)

    Returns
    -------
    np.ndarray
        Array 2-D hasil slice.

    Raises
    ------
    ValueError
        Jika volume bukan 3-D atau axis di luar rentang.
    IndexError
        Jika slice_idx di luar batas dimensi.
    """
    if volume.ndim != 3:
        raise ValueError(
            f"extract_slice membutuhkan volume 3-D, "
            f"diterima ndim={volume.ndim}"
        )
    if axis not in (0, 1, 2):
        raise ValueError(f"axis harus 0, 1, atau 2, diterima {axis}")

    max_idx = volume.shape[axis]
    if slice_idx is None:
        slice_idx = max_idx // 2

    if not (0 <= slice_idx < max_idx):
        raise IndexError(
            f"slice_idx={slice_idx} di luar batas untuk "
            f"axis={axis} (max={max_idx - 1})"
        )

    sliced = np.take(volume, indices=slice_idx, axis=axis)
    logger.debug(
        "Extracted slice  |  axis=%d  idx=%d  shape=%s",
        axis, slice_idx, sliced.shape,
    )
    return sliced


# ===================================================================
# CLI — Quick self-test
# ===================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  Modul Preprocessing — Self-Test")
    print("=" * 60)

    # Test dengan data dummy
    dummy = create_dummy_patient_data(shape=(30, 64, 64))  # kecil
    for key, arr in dummy.items():
        print(f"  {key:6s} -> shape={arr.shape}  dtype={arr.dtype}  "
              f"range=[{arr.min():.3f}, {arr.max():.3f}]")

    # Test normalisasi
    flair_norm = normalize_intensity(dummy["flair"])
    print(f"\n  Normalized flair range: [{flair_norm.min():.4f}, "
          f"{flair_norm.max():.4f}]")

    # Test noise
    mid_slice = extract_slice(flair_norm)
    for nt in ("gaussian", "salt_pepper", "motion_blur"):
        noisy = add_noise(mid_slice, noise_type=nt, level=0.05)
        print(f"  Noise '{nt}' -> shape={noisy.shape}")

    # Test brain mask
    mask = get_brain_mask(flair_norm, threshold=0.1)
    print(f"\n  Brain mask shape={mask.shape}  "
          f"brain_pct={100 * mask.sum() / mask.size:.1f}%")

    print("\n" + "=" * 60)
    print("  Self-test PASSED ✓")
    print("=" * 60)
