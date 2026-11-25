#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cnn_module.py — Modul 3: Lightweight Dual-Output CNN

Bagian dari riset:
  "An AI-Driven Hybrid Fuzzy C-Means Framework for Robust Brain Tumor
   MRI Segmentation via PCA-Based Feature Reduction"

CNN ini memiliki dua cabang output:
  1. **Cluster Count** — Prediksi jumlah kluster optimal (k=2, 3, atau 4)
     untuk Fuzzy C-Means, berdasarkan kompleksitas anatomi slice.
  2. **Confidence Map** — Peta probabilitas spasial (240×240) yang
     menunjukkan keyakinan model bahwa suatu piksel merupakan tumor,
     digunakan sebagai bobot keanggotaan awal FCM.

Arsitektur:
  Shared backbone (4× Conv2D+BN+ReLU+Pool → GAP → Dense)
  ├─ Cabang 1: Dense(64) → Dense(3, softmax)  [cluster_count]
  └─ Cabang 2: Dense → Reshape → 3× Conv2DTranspose → sigmoid  [confidence_map]

Author  : [Peneliti]
Created : 2026-05-01
Python  : 3.9+
Deps    : TensorFlow 2.x, NumPy
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

# TensorFlow — suppress verbose warnings
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf
from tensorflow import keras

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
CLUSTER_MAP = {1: 0, 2: 1, 3: 2}  # n_tumor_labels → class index (0,1,2)
CLUSTER_INV = {0: 2, 1: 3, 2: 4}  # class index → actual k


# ===================================================================
# 1. build_cnn_model
# ===================================================================
def build_cnn_model(
    input_shape: Tuple[int, int, int] = (240, 240, 1),
) -> keras.Model:
    """Membangun lightweight dual-output CNN menggunakan Keras Functional API.

    Arsitektur:
      - Shared backbone: 4 blok Conv2D + BatchNorm + ReLU + MaxPool,
        diakhiri GlobalAveragePooling2D + Dense(128).
      - Cabang 1 (cluster_count): Dense(64) → Dense(3, softmax).
      - Cabang 2 (confidence_map): Dense → Reshape → 3× Conv2DTranspose
        → Conv2D(1, sigmoid).

    Parameters
    ----------
    input_shape : tuple of int, default (240, 240, 1)
        Dimensi input (height, width, channels). Height dan width
        harus habis dibagi 8 (karena 3 MaxPool stride-2).

    Returns
    -------
    keras.Model
        Model Keras yang belum di-compile, dengan dua output:
        ``'cluster_count'`` dan ``'confidence_map'``.

    Raises
    ------
    ValueError
        Jika dimensi input tidak kompatibel (tidak habis dibagi 8).
    """
    H, W, C = input_shape
    if H % 8 != 0 or W % 8 != 0:
        raise ValueError(
            f"Height ({H}) dan Width ({W}) harus habis dibagi 8. "
            f"Diterima input_shape={input_shape}"
        )

    # ---- Input ----
    inputs = keras.Input(shape=input_shape, name="mri_input")

    # ---- Shared Backbone ----
    # Block 1: (H, W, 1) -> (H/2, W/2, 32)
    x = keras.layers.Conv2D(32, 3, padding="same", name="conv1")(inputs)
    x = keras.layers.BatchNormalization(name="bn1")(x)
    x = keras.layers.ReLU(name="relu1")(x)
    x = keras.layers.MaxPool2D(2, name="pool1")(x)

    # Block 2: -> (H/4, W/4, 64)
    x = keras.layers.Conv2D(64, 3, padding="same", name="conv2")(x)
    x = keras.layers.BatchNormalization(name="bn2")(x)
    x = keras.layers.ReLU(name="relu2")(x)
    x = keras.layers.MaxPool2D(2, name="pool2")(x)

    # Block 3: -> (H/8, W/8, 128)
    x = keras.layers.Conv2D(128, 3, padding="same", name="conv3")(x)
    x = keras.layers.BatchNormalization(name="bn3")(x)
    x = keras.layers.ReLU(name="relu3")(x)
    x = keras.layers.MaxPool2D(2, name="pool3")(x)

    # Block 4: -> (H/8, W/8, 256) -> GAP -> (256,)
    x = keras.layers.Conv2D(256, 3, padding="same", name="conv4")(x)
    x = keras.layers.BatchNormalization(name="bn4")(x)
    x = keras.layers.ReLU(name="relu4")(x)
    backbone_spatial = x  # simpan untuk referensi
    x = keras.layers.GlobalAveragePooling2D(name="gap")(x)

    # Shared dense
    shared = keras.layers.Dense(128, activation="relu", name="shared_dense")(x)
    shared = keras.layers.Dropout(0.5, name="shared_dropout")(shared)

    # ---- Cabang 1: Cluster Count (k=2,3,4) ----
    c1 = keras.layers.Dense(64, activation="relu", name="cluster_dense")(shared)
    cluster_output = keras.layers.Dense(
        3, activation="softmax", name="cluster_count"
    )(c1)

    # ---- Cabang 2: Confidence Map (spatial decoder) ----
    # Dimensi setelah backbone spatial: (H/8, W/8, 256)
    feat_h, feat_w = H // 8, W // 8
    decoder_units = feat_h * feat_w * 64

    c2 = keras.layers.Dense(
        decoder_units, activation="relu", name="decoder_dense"
    )(shared)
    c2 = keras.layers.Reshape(
        (feat_h, feat_w, 64), name="decoder_reshape"
    )(c2)

    # Upsample 3×: (H/8,W/8) -> (H/4,W/4) -> (H/2,W/2) -> (H,W)
    c2 = keras.layers.Conv2DTranspose(
        64, 3, strides=2, padding="same", name="deconv1"
    )(c2)
    c2 = keras.layers.BatchNormalization(name="deconv_bn1")(c2)
    c2 = keras.layers.ReLU(name="deconv_relu1")(c2)

    c2 = keras.layers.Conv2DTranspose(
        32, 3, strides=2, padding="same", name="deconv2"
    )(c2)
    c2 = keras.layers.BatchNormalization(name="deconv_bn2")(c2)
    c2 = keras.layers.ReLU(name="deconv_relu2")(c2)

    c2 = keras.layers.Conv2DTranspose(
        16, 3, strides=2, padding="same", name="deconv3"
    )(c2)
    c2 = keras.layers.BatchNormalization(name="deconv_bn3")(c2)
    c2 = keras.layers.ReLU(name="deconv_relu3")(c2)

    confidence_output = keras.layers.Conv2D(
        1, 1, activation="sigmoid", name="confidence_map"
    )(c2)

    # ---- Build Model ----
    model = keras.Model(
        inputs=inputs,
        outputs=[cluster_output, confidence_output],
        name="DualOutput_CNN",
    )

    logger.info(
        "Model dibangun  |  input=%s  |  params=%s  |  "
        "outputs: cluster_count(3), confidence_map(%d,%d,1)",
        input_shape, f"{model.count_params():,}", H, W,
    )
    return model


# ===================================================================
# 2. get_cluster_count_from_seg
# ===================================================================
def get_cluster_count_from_seg(seg_slice: np.ndarray) -> int:
    """Menentukan jumlah kluster dari ground truth segmentasi.

    Menghitung berapa jenis label tumor (label > 0) yang ada pada
    slice segmentasi, lalu memetakan ke jumlah kluster FCM:
      - 1 jenis label tumor → k=2 (background + 1 tumor)
      - 2 jenis label tumor → k=3 (background + 2 tumor)
      - 3 jenis label tumor → k=4 (background + 3 tumor)

    Parameters
    ----------
    seg_slice : np.ndarray
        Slice segmentasi 2-D dengan label integer (0, 1, 2, 3, atau 4).

    Returns
    -------
    int
        Jumlah kluster: 2, 3, atau 4.
    """
    unique_labels = np.unique(seg_slice)
    tumor_labels = unique_labels[unique_labels > 0]
    n_tumor = len(tumor_labels)

    if n_tumor == 0:
        # Tidak ada tumor — default 2 (bg + 1 kemungkinan)
        return 2
    elif n_tumor >= 3:
        return 4
    else:
        return n_tumor + 1  # 1->2, 2->3


# ===================================================================
# 3. prepare_training_data_from_patient
# ===================================================================
def prepare_training_data_from_patient(
    patient_dict: Dict[str, np.ndarray],
    modality: str = "flair",
    include_empty: bool = False,
) -> List[Dict[str, np.ndarray]]:
    """Menyiapkan data training CNN dari satu pasien BraTS.

    Mengiterasi setiap slice axial dari volume MRI pasien, melakukan
    normalisasi, menghitung cluster_count dari segmentasi, dan membuat
    confidence map (binary: tumor vs background).

    Parameters
    ----------
    patient_dict : dict[str, np.ndarray]
        Hasil dari ``load_patient_modalities()`` atau
        ``create_dummy_patient_data()``. Harus berisi key ``modality``
        dan ``'seg'``.
    modality : str, default 'flair'
        Modalitas MRI yang digunakan sebagai input CNN.
    include_empty : bool, default False
        Jika True, sertakan slice tanpa tumor (seg seluruhnya 0).
        Jika False, skip slice tanpa tumor.

    Returns
    -------
    list of dict
        Setiap dict berisi:
        - ``'slice'``: np.ndarray (H, W) — slice ternormalisasi [0, 1].
        - ``'seg'``: np.ndarray (H, W) — segmentasi asli.
        - ``'cluster_count'``: int — jumlah kluster (2, 3, atau 4).
        - ``'cluster_class'``: int — class index (0, 1, atau 2).
        - ``'confidence_map'``: np.ndarray (H, W) — binary (seg > 0).
        - ``'slice_idx'``: int — indeks slice dalam volume.

    Raises
    ------
    KeyError
        Jika modalitas atau 'seg' tidak ada dalam patient_dict.
    """
    if modality not in patient_dict:
        raise KeyError(f"Modalitas '{modality}' tidak ditemukan.")
    if "seg" not in patient_dict:
        raise KeyError("'seg' tidak ditemukan dalam patient_dict.")

    from src.preprocessing import normalize_intensity

    volume = patient_dict[modality]
    seg_volume = patient_dict["seg"]
    n_slices = volume.shape[0]

    # Normalisasi seluruh volume
    volume_norm = normalize_intensity(volume)

    samples: List[Dict[str, np.ndarray]] = []

    for idx in range(n_slices):
        seg_slice = seg_volume[idx].astype(np.float32)
        has_tumor = (seg_slice > 0).any()

        if not has_tumor and not include_empty:
            continue

        mri_slice = volume_norm[idx].astype(np.float32)
        cluster_k = get_cluster_count_from_seg(seg_slice)
        cluster_class = cluster_k - 2  # k=2->0, k=3->1, k=4->2
        confidence = (seg_slice > 0).astype(np.float32)

        samples.append({
            "slice": mri_slice,
            "seg": seg_slice,
            "cluster_count": cluster_k,
            "cluster_class": cluster_class,
            "confidence_map": confidence,
            "slice_idx": idx,
        })

    logger.info(
        "Data training disiapkan  |  modality=%s  |  "
        "total_slices=%d  valid_slices=%d (include_empty=%s)  |  "
        "cluster_dist: {2: %d, 3: %d, 4: %d}",
        modality, n_slices, len(samples), include_empty,
        sum(1 for s in samples if s["cluster_count"] == 2),
        sum(1 for s in samples if s["cluster_count"] == 3),
        sum(1 for s in samples if s["cluster_count"] == 4),
    )
    return samples


# ===================================================================
# 4. train_cnn_model
# ===================================================================
def train_cnn_model(
    model: keras.Model,
    train_slices: np.ndarray,
    train_segs: np.ndarray,
    val_slices: Optional[np.ndarray] = None,
    val_segs: Optional[np.ndarray] = None,
    epochs: int = 50,
    batch_size: int = 8,
    checkpoint_path: Optional[Union[str, Path]] = None,
) -> keras.callbacks.History:
    """Melatih dual-output CNN dengan data MRI dan segmentasi.

    Secara otomatis menyiapkan label untuk kedua cabang output:
      - cluster_count: dihitung dari ``get_cluster_count_from_seg``
        per slice.
      - confidence_map: ``(seg > 0).astype(float32)``.

    Parameters
    ----------
    model : keras.Model
        Model dari ``build_cnn_model()`` (belum atau sudah di-compile).
    train_slices : np.ndarray
        Array training MRI (N, H, W) sudah dinormalisasi [0, 1].
    train_segs : np.ndarray
        Array segmentasi training (N, H, W) dengan label integer.
    val_slices : np.ndarray, optional
        Array validasi MRI (M, H, W).
    val_segs : np.ndarray, optional
        Array segmentasi validasi (M, H, W).
    epochs : int, default 50
        Jumlah epoch training.
    batch_size : int, default 8
        Ukuran batch.
    checkpoint_path : str | Path, optional
        Path untuk menyimpan checkpoint model terbaik.

    Returns
    -------
    keras.callbacks.History
        Objek history training.
    """
    # ---- Compile model jika belum ----
    needs_compile = False
    try:
        needs_compile = model.optimizer is None
    except (AttributeError, ValueError):
        needs_compile = True

    if needs_compile:
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-4),
            loss={
                "cluster_count": "sparse_categorical_crossentropy",
                "confidence_map": "binary_crossentropy",
            },
            loss_weights={
                "cluster_count": 0.3,
                "confidence_map": 0.7,
            },
            metrics={
                "cluster_count": ["accuracy"],
                "confidence_map": ["accuracy"],
            },
        )
        logger.info("Model di-compile dengan Adam(lr=1e-4)")

    # ---- Siapkan data training ----
    X_train = train_slices[..., np.newaxis].astype(np.float32)  # (N,H,W,1)

    y_cluster_train = np.array([
        get_cluster_count_from_seg(train_segs[i]) - 2
        for i in range(len(train_segs))
    ], dtype=np.int32)

    y_conf_train = (train_segs > 0).astype(np.float32)[..., np.newaxis]  # (N,H,W,1)

    logger.info(
        "Training data  |  X=%s  y_cluster=%s (dist=%s)  y_conf=%s",
        X_train.shape, y_cluster_train.shape,
        {k: int((y_cluster_train == k).sum()) for k in range(3)},
        y_conf_train.shape,
    )

    # ---- Siapkan data validasi ----
    validation_data = None
    if val_slices is not None and val_segs is not None:
        X_val = val_slices[..., np.newaxis].astype(np.float32)
        y_cluster_val = np.array([
            get_cluster_count_from_seg(val_segs[i]) - 2
            for i in range(len(val_segs))
        ], dtype=np.int32)
        y_conf_val = (val_segs > 0).astype(np.float32)[..., np.newaxis]
        validation_data = (
            X_val,
            {"cluster_count": y_cluster_val, "confidence_map": y_conf_val},
        )
        logger.info("Validation data  |  X=%s", X_val.shape)

    # ---- Callbacks ----
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss" if validation_data else "loss",
            patience=10,
            restore_best_weights=True,
            verbose=1,
        ),
    ]

    if checkpoint_path:
        checkpoint_path = str(checkpoint_path)
        callbacks.append(
            keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_path,
                monitor="val_loss" if validation_data else "loss",
                save_best_only=True,
                verbose=1,
            )
        )

    # ---- Training ----
    history = model.fit(
        X_train,
        {"cluster_count": y_cluster_train, "confidence_map": y_conf_train},
        validation_data=validation_data,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    logger.info(
        "Training selesai  |  epochs_run=%d  "
        "final_loss=%.4f",
        len(history.history["loss"]),
        history.history["loss"][-1],
    )
    return history


# ===================================================================
# 5. predict_cluster_count
# ===================================================================
def predict_cluster_count(
    model: keras.Model,
    slice_2d: np.ndarray,
) -> int:
    """Memprediksi jumlah kluster optimal dari satu slice MRI.

    Parameters
    ----------
    model : keras.Model
        Model CNN yang sudah dilatih (dual-output).
    slice_2d : np.ndarray
        Slice MRI 2-D (H, W), sudah dinormalisasi [0, 1].

    Returns
    -------
    int
        Jumlah kluster: 2, 3, atau 4.
    """
    # Reshape ke batch of 1: (1, H, W, 1)
    x = slice_2d[np.newaxis, ..., np.newaxis].astype(np.float32)
    predictions = model.predict(x, verbose=0)

    # Output pertama = cluster_count probabilities
    cluster_probs = predictions[0][0]  # (3,)
    class_idx = int(np.argmax(cluster_probs))
    k = CLUSTER_INV[class_idx]

    logger.debug(
        "Prediksi cluster  |  probs=%s  class=%d  k=%d",
        np.array2string(cluster_probs, precision=3), class_idx, k,
    )
    return k


# ===================================================================
# 6. predict_confidence_map
# ===================================================================
def predict_confidence_map(
    model: keras.Model,
    slice_2d: np.ndarray,
) -> np.ndarray:
    """Memprediksi confidence map dari satu slice MRI.

    Confidence map menunjukkan probabilitas setiap piksel merupakan
    tumor (1.0 = yakin tumor, 0.0 = yakin bukan tumor).

    Parameters
    ----------
    model : keras.Model
        Model CNN yang sudah dilatih (dual-output).
    slice_2d : np.ndarray
        Slice MRI 2-D (H, W), sudah dinormalisasi [0, 1].

    Returns
    -------
    np.ndarray
        Confidence map (H, W) dengan nilai [0, 1], dtype float32.
    """
    x = slice_2d[np.newaxis, ..., np.newaxis].astype(np.float32)
    predictions = model.predict(x, verbose=0)

    # Output kedua = confidence_map
    conf_map = predictions[1][0, :, :, 0]  # (H, W)

    logger.debug(
        "Prediksi confidence map  |  shape=%s  range=[%.4f, %.4f]",
        conf_map.shape, conf_map.min(), conf_map.max(),
    )
    return conf_map.astype(np.float32)


# ===================================================================
# 7. save_cnn_model
# ===================================================================
def save_cnn_model(
    model: keras.Model,
    path: Union[str, Path],
) -> None:
    """Menyimpan model CNN ke disk.

    Parameters
    ----------
    model : keras.Model
        Model yang akan disimpan.
    path : str | Path
        Path file output (misal 'output/cnn_model.keras').
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    model.save(str(path))
    size_mb = path.stat().st_size / (1024 * 1024)
    logger.info("Model disimpan  |  path=%s  size=%.1f MB", path, size_mb)


# ===================================================================
# 8. load_cnn_model
# ===================================================================
def load_cnn_model(
    path: Union[str, Path],
) -> keras.Model:
    """Memuat model CNN dari disk.

    Parameters
    ----------
    path : str | Path
        Path ke file model (.keras atau .h5).

    Returns
    -------
    keras.Model
        Model yang dimuat.

    Raises
    ------
    FileNotFoundError
        Jika file tidak ditemukan.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model tidak ditemukan: {path}")

    model = keras.models.load_model(str(path))
    logger.info(
        "Model dimuat  |  path=%s  params=%s",
        path, f"{model.count_params():,}",
    )
    return model


# ===================================================================
# CLI — Quick self-test
# ===================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  Modul CNN — Quick Self-Test")
    print("=" * 60)

    model = build_cnn_model(input_shape=(64, 64, 1))
    model.summary(line_length=90)

    # Dummy forward pass
    dummy_x = np.random.rand(2, 64, 64, 1).astype(np.float32)
    cluster_out, conf_out = model.predict(dummy_x, verbose=0)
    print(f"\n  cluster_out shape: {cluster_out.shape}")
    print(f"  conf_out shape   : {conf_out.shape}")
    print(f"  cluster_probs[0] : {cluster_out[0]}")

    print("\n" + "=" * 60)
    print("  Self-test PASSED ✓")
    print("=" * 60)
