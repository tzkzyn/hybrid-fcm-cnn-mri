#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
postprocessing.py — Modul 5: Light U-Net Refiner & Morphological Post-processing

Bagian dari riset:
  "An AI-Driven Hybrid Fuzzy C-Means Framework for Robust Brain Tumor
   MRI Segmentation via PCA-Based Feature Reduction"

Modul ini memperbaiki segmentasi kasar dari Hybrid FCM menjadi
segmentasi halus melalui:
  1. Light U-Net Refiner — CNN encoder-decoder dengan skip connections
  2. Morphological post-processing — CCA, hole filling, small object removal

Author  : [Peneliti]
Created : 2026-05-01
Python  : 3.9+
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.ndimage import binary_fill_holes, label as ndimage_label

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf
from tensorflow import keras

# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)


# ===================================================================
# 1. build_refiner_unet
# ===================================================================
def _conv_block(x, filters, name_prefix):
    """Double Conv2D + BN + ReLU block."""
    x = keras.layers.Conv2D(filters, 3, padding="same",
                            name=f"{name_prefix}_conv1")(x)
    x = keras.layers.BatchNormalization(name=f"{name_prefix}_bn1")(x)
    x = keras.layers.ReLU(name=f"{name_prefix}_relu1")(x)
    x = keras.layers.Conv2D(filters, 3, padding="same",
                            name=f"{name_prefix}_conv2")(x)
    x = keras.layers.BatchNormalization(name=f"{name_prefix}_bn2")(x)
    x = keras.layers.ReLU(name=f"{name_prefix}_relu2")(x)
    return x


def build_refiner_unet(
    input_shape: Tuple[int, int, int] = (240, 240, 2),
    n_classes: int = 4,
) -> keras.Model:
    """Membangun Light U-Net Refiner.

    Input 2-channel: [MRI_normalized, coarse_segmentation].
    Output: segmentasi halus n_classes kelas.

    Parameters
    ----------
    input_shape : tuple, default (240, 240, 2)
        (H, W, 2). H dan W harus habis dibagi 8.
    n_classes : int, default 4
        Jumlah kelas segmentasi (0=bg, 1=necrosis, 2=edema, 3=enhancing).

    Returns
    -------
    keras.Model
        Model U-Net yang sudah di-compile.
    """
    H, W, C = input_shape
    if H % 8 != 0 or W % 8 != 0:
        raise ValueError(f"H ({H}) dan W ({W}) harus habis dibagi 8.")

    inputs = keras.Input(shape=input_shape, name="refiner_input")

    # ---- Encoder ----
    e1 = _conv_block(inputs, 32, "enc1")
    p1 = keras.layers.MaxPool2D(2, name="enc1_pool")(e1)

    e2 = _conv_block(p1, 64, "enc2")
    p2 = keras.layers.MaxPool2D(2, name="enc2_pool")(e2)

    e3 = _conv_block(p2, 128, "enc3")
    p3 = keras.layers.MaxPool2D(2, name="enc3_pool")(e3)

    # ---- Bottleneck ----
    b = keras.layers.Conv2D(256, 3, padding="same", name="bottle_conv1")(p3)
    b = keras.layers.BatchNormalization(name="bottle_bn1")(b)
    b = keras.layers.ReLU(name="bottle_relu1")(b)
    b = keras.layers.Dropout(0.3, name="bottle_drop")(b)
    b = keras.layers.Conv2D(256, 3, padding="same", name="bottle_conv2")(b)
    b = keras.layers.BatchNormalization(name="bottle_bn2")(b)
    b = keras.layers.ReLU(name="bottle_relu2")(b)

    # ---- Decoder ----
    d3 = keras.layers.Conv2DTranspose(128, 2, strides=2, padding="same",
                                       name="dec3_up")(b)
    d3 = keras.layers.Concatenate(name="dec3_cat")([d3, e3])
    d3 = _conv_block(d3, 128, "dec3")

    d2 = keras.layers.Conv2DTranspose(64, 2, strides=2, padding="same",
                                       name="dec2_up")(d3)
    d2 = keras.layers.Concatenate(name="dec2_cat")([d2, e2])
    d2 = _conv_block(d2, 64, "dec2")

    d1 = keras.layers.Conv2DTranspose(32, 2, strides=2, padding="same",
                                       name="dec1_up")(d2)
    d1 = keras.layers.Concatenate(name="dec1_cat")([d1, e1])
    d1 = _conv_block(d1, 32, "dec1")

    # ---- Output ----
    outputs = keras.layers.Conv2D(
        n_classes, 1, activation="softmax", name="refined_seg"
    )(d1)

    model = keras.Model(inputs=inputs, outputs=outputs, name="LightUNet_Refiner")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    logger.info(
        "Refiner U-Net dibangun  |  input=%s  classes=%d  params=%s",
        input_shape, n_classes, f"{model.count_params():,}",
    )
    return model


# ===================================================================
# 2. prepare_refiner_data
# ===================================================================
def prepare_refiner_data(
    patient_dict: Dict[str, np.ndarray],
    coarse_seg_volume: np.ndarray,
    modality: str = "flair",
    n_classes: int = 4,
) -> Tuple[np.ndarray, np.ndarray]:
    """Siapkan data training refiner dari satu pasien.

    Parameters
    ----------
    patient_dict : dict
        Hasil load_patient_modalities() atau create_dummy_patient_data().
    coarse_seg_volume : np.ndarray
        Volume segmentasi kasar (D, H, W) dari Hybrid FCM, label integer.
    modality : str
        Modalitas MRI.
    n_classes : int
        Jumlah kelas untuk normalisasi label coarse seg.

    Returns
    -------
    X : np.ndarray
        Input (N, H, W, 2) — channel 0=MRI norm, channel 1=coarse seg float.
    y : np.ndarray
        Ground truth (N, H, W) integer labels.
    """
    from src.preprocessing import normalize_intensity

    volume = patient_dict[modality]
    seg_gt = patient_dict["seg"]
    vol_norm = normalize_intensity(volume)

    D = volume.shape[0]
    X_list, y_list = [], []

    for idx in range(D):
        coarse = coarse_seg_volume[idx].astype(np.float32)
        gt = seg_gt[idx].astype(np.int32)

        # Skip empty slices
        if coarse.max() == 0 and gt.max() == 0:
            continue

        mri = vol_norm[idx].astype(np.float32)
        # Normalize coarse seg to [0, 1]
        coarse_norm = coarse / max(n_classes - 1, 1)

        x = np.stack([mri, coarse_norm], axis=-1)  # (H, W, 2)
        X_list.append(x)
        y_list.append(gt)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)

    logger.info(
        "Refiner data  |  X=%s  y=%s  slices=%d", X.shape, y.shape, len(X_list),
    )
    return X, y


# ===================================================================
# 3. train_refiner
# ===================================================================
def train_refiner(
    model: keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    epochs: int = 30,
    batch_size: int = 8,
    patience: int = 10,
    checkpoint_path: Optional[Union[str, Path]] = None,
) -> keras.callbacks.History:
    """Melatih Light U-Net Refiner.

    Parameters
    ----------
    model : keras.Model
        Model dari build_refiner_unet().
    X_train, y_train : np.ndarray
        Data training (N, H, W, 2) dan label (N, H, W).
    X_val, y_val : np.ndarray, optional
        Data validasi.
    epochs : int
        Jumlah epoch.
    batch_size : int
        Ukuran batch.
    patience : int
        Early stopping patience.
    checkpoint_path : str | Path, optional
        Path checkpoint.

    Returns
    -------
    keras.callbacks.History
    """
    validation_data = None
    if X_val is not None and y_val is not None:
        validation_data = (X_val, y_val)

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss" if validation_data else "loss",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
    ]
    if checkpoint_path:
        callbacks.append(keras.callbacks.ModelCheckpoint(
            str(checkpoint_path),
            monitor="val_loss" if validation_data else "loss",
            save_best_only=True, verbose=1,
        ))

    history = model.fit(
        X_train, y_train,
        validation_data=validation_data,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    logger.info("Refiner training selesai  |  epochs=%d  final_loss=%.4f",
                len(history.history["loss"]), history.history["loss"][-1])
    return history


# ===================================================================
# 4. refine_segmentation
# ===================================================================
def refine_segmentation(
    model: keras.Model,
    mri_slice: np.ndarray,
    coarse_seg: np.ndarray,
    n_classes: int = 4,
) -> np.ndarray:
    """Refine satu slice segmentasi kasar dengan U-Net.

    Parameters
    ----------
    model : keras.Model
        Refiner U-Net terlatih.
    mri_slice : np.ndarray
        Slice MRI (H, W) ternormalisasi [0, 1].
    coarse_seg : np.ndarray
        Segmentasi kasar (H, W) label integer.
    n_classes : int
        Jumlah kelas.

    Returns
    -------
    np.ndarray
        Segmentasi halus (H, W) label integer.
    """
    coarse_norm = coarse_seg.astype(np.float32) / max(n_classes - 1, 1)
    x = np.stack([mri_slice.astype(np.float32), coarse_norm], axis=-1)
    x = x[np.newaxis, ...]  # (1, H, W, 2)

    pred = model.predict(x, verbose=0)  # (1, H, W, n_classes)
    refined = np.argmax(pred[0], axis=-1).astype(np.int32)
    return refined


# ===================================================================
# 5. postprocess_morphology
# ===================================================================
def postprocess_morphology(
    seg_map: np.ndarray,
    brain_mask: Optional[np.ndarray] = None,
    min_hole_size: int = 100,
    min_object_size: int = 50,
) -> np.ndarray:
    """Membersihkan segmentasi dengan operasi morfologi.

    Operasi:
      1. Per label: hapus komponen kecil (< min_object_size).
      2. Per label: fill holes (< min_hole_size).
      3. Jika brain_mask: hapus prediksi di luar otak.

    Parameters
    ----------
    seg_map : np.ndarray
        Peta segmentasi (H, W) label integer.
    brain_mask : np.ndarray, optional
        Binary brain mask (H, W).
    min_hole_size : int
        Ukuran lubang minimum yang di-fill.
    min_object_size : int
        Ukuran objek minimum yang dipertahankan.

    Returns
    -------
    np.ndarray
        Segmentasi yang dibersihkan (H, W).
    """
    cleaned = seg_map.copy().astype(np.int32)
    unique_labels = np.unique(cleaned)

    for lbl in unique_labels:
        if lbl == 0:
            continue  # skip background

        binary = (cleaned == lbl).astype(np.int32)

        # Remove small objects
        labeled, n_comp = ndimage_label(binary)
        for comp_id in range(1, n_comp + 1):
            comp_mask = labeled == comp_id
            if comp_mask.sum() < min_object_size:
                cleaned[comp_mask] = 0

        # Fill holes
        binary_updated = (cleaned == lbl)
        filled = binary_fill_holes(binary_updated)
        # Only fill small holes
        holes = filled & ~binary_updated
        if holes.any():
            hole_labeled, n_holes = ndimage_label(holes)
            for h_id in range(1, n_holes + 1):
                h_mask = hole_labeled == h_id
                if h_mask.sum() < min_hole_size:
                    cleaned[h_mask] = lbl

    # Apply brain mask
    if brain_mask is not None:
        cleaned[brain_mask == 0] = 0

    removed = int((seg_map != cleaned).sum())
    logger.info("Morphology postprocessing  |  pixels_changed=%d (%.2f%%)",
                removed, 100 * removed / seg_map.size)
    return cleaned


# ===================================================================
# 6. refine_patient_volume
# ===================================================================
def refine_patient_volume(
    model: keras.Model,
    patient_dict: Dict[str, np.ndarray],
    coarse_seg_volume: np.ndarray,
    modality: str = "flair",
    n_classes: int = 4,
    apply_morphology: bool = True,
    brain_mask_volume: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Refine seluruh volume 3-D slice per slice.

    Parameters
    ----------
    model : keras.Model
        Refiner U-Net terlatih.
    patient_dict : dict
        Data pasien.
    coarse_seg_volume : np.ndarray
        Volume segmentasi kasar (D, H, W).
    modality : str
        Modalitas MRI.
    n_classes : int
        Jumlah kelas.
    apply_morphology : bool
        Terapkan postprocess_morphology per slice.
    brain_mask_volume : np.ndarray, optional
        Volume brain mask (D, H, W).

    Returns
    -------
    np.ndarray
        Volume segmentasi refined (D, H, W) int32.
    """
    from src.preprocessing import normalize_intensity

    volume = patient_dict[modality]
    vol_norm = normalize_intensity(volume)
    D, H, W = volume.shape

    refined_volume = np.zeros((D, H, W), dtype=np.int32)

    for idx in range(D):
        mri_slice = vol_norm[idx]
        coarse = coarse_seg_volume[idx]

        refined = refine_segmentation(model, mri_slice, coarse, n_classes)

        if apply_morphology:
            bmask = brain_mask_volume[idx] if brain_mask_volume is not None else None
            refined = postprocess_morphology(refined, brain_mask=bmask)

        refined_volume[idx] = refined

    logger.info("Volume refined  |  shape=%s  labels=%s",
                refined_volume.shape,
                {int(v): int((refined_volume == v).sum())
                 for v in np.unique(refined_volume)})
    return refined_volume


# ===================================================================
# 7. save/load refiner
# ===================================================================
def save_refiner_model(model: keras.Model, path: Union[str, Path]) -> None:
    """Simpan refiner model."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(path))
    logger.info("Refiner disimpan  |  %s  (%.1f MB)",
                path, path.stat().st_size / 1048576)


def load_refiner_model(path: Union[str, Path]) -> keras.Model:
    """Muat refiner model."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model tidak ditemukan: {path}")
    model = keras.models.load_model(str(path))
    logger.info("Refiner dimuat  |  %s  params=%s",
                path, f"{model.count_params():,}")
    return model


# ===================================================================
# CLI self-test
# ===================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  Modul Postprocessing — Quick Self-Test")
    print("=" * 60)

    model = build_refiner_unet(input_shape=(64, 64, 2), n_classes=4)
    model.summary(line_length=90, print_fn=lambda s: None)  # suppress
    print(f"  Params: {model.count_params():,}")

    dummy_x = np.random.rand(1, 64, 64, 2).astype(np.float32)
    out = model.predict(dummy_x, verbose=0)
    print(f"  Output shape: {out.shape}")

    seg = np.random.randint(0, 4, (64, 64), dtype=np.int32)
    cleaned = postprocess_morphology(seg, min_object_size=10, min_hole_size=10)
    print(f"  Morphology: changed={int((seg != cleaned).sum())} pixels")

    print("\n  Self-test PASSED ✓")
    print("=" * 60)
