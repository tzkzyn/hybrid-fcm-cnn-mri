#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_cnn_module.py — Skrip Uji Komprehensif untuk Modul 3 (CNN)

Menguji seluruh fungsi di src/cnn_module.py:
  1. build_cnn_model — arsitektur & dimensi output
  2. prepare_training_data_from_patient — siapkan data dari dummy/real
  3. train_cnn_model — training loop (epoch kecil)
  4. predict_cluster_count & predict_confidence_map — inference
  5. save/load model
  6. Visualisasi: loss curves, confidence map, cluster predictions
  7. BONUS: Training dengan dataset BraTS 2020 asli

Output: Figure matplotlib disimpan ke output/test_cnn_*.png

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

from src.preprocessing import (
    create_dummy_patient_data,
    extract_slice,
    load_patient_modalities,
    normalize_intensity,
)
from src.cnn_module import (
    build_cnn_model,
    get_cluster_count_from_seg,
    load_cnn_model,
    predict_cluster_count,
    predict_confidence_map,
    prepare_training_data_from_patient,
    save_cnn_model,
    train_cnn_model,
)

# ---------------------------------------------------------------------------
# Konfigurasi
# ---------------------------------------------------------------------------
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

DATASET_PATH = PROJECT_ROOT / "data" / "Brats2020" / "training"
SAMPLE_PATIENT = "BraTS20_Training_001"

# Ukuran kecil untuk dummy test (CPU training cepat)
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
# TEST 1: Build Model & Summary
# ===================================================================
def test_build_model() -> None:
    """Bangun model dan cetak summary."""
    separator("TEST 1: Build CNN Model")

    H, W = DUMMY_SHAPE[1], DUMMY_SHAPE[2]
    model = build_cnn_model(input_shape=(H, W, 1))
    model.summary(line_length=90)

    # Verifikasi output shapes
    print(f"\n  Input shape  : {model.input_shape}")
    print(f"  Output names : {[o.name for o in model.outputs]}")
    for i, o in enumerate(model.outputs):
        print(f"  Output[{i}] shape: {o.shape}")

    print(f"  Total params : {model.count_params():,}")

    # Quick forward pass
    dummy_x = np.random.rand(2, H, W, 1).astype(np.float32)
    outs = model.predict(dummy_x, verbose=0)
    print(f"\n  Forward pass (batch=2):")
    print(f"    cluster_count: {outs[0].shape}  → {outs[0][0]}")
    print(f"    confidence_map: {outs[1].shape}  → "
          f"range=[{outs[1].min():.4f}, {outs[1].max():.4f}]")

    print(f"\n  ✓ Model build & forward pass OK")


# ===================================================================
# TEST 2: get_cluster_count_from_seg
# ===================================================================
def test_cluster_count_logic() -> None:
    """Uji logika penentuan jumlah cluster dari segmentasi."""
    separator("TEST 2: get_cluster_count_from_seg")

    test_cases = [
        (np.array([[0, 0], [0, 0]]), 2, "Semua background"),
        (np.array([[0, 1], [0, 0]]), 2, "Hanya 1 label tumor"),
        (np.array([[0, 1], [2, 0]]), 3, "2 label tumor"),
        (np.array([[0, 1], [2, 3]]), 4, "3 label tumor"),
        (np.array([[1, 2], [3, 4]]), 4, "4 label tumor (BraTS: 1,2,4)"),
    ]

    all_pass = True
    for seg, expected, desc in test_cases:
        result = get_cluster_count_from_seg(seg)
        status = "✓" if result == expected else "✗"
        if result != expected:
            all_pass = False
        print(f"  {status}  {desc:30s}  → k={result} (expected {expected})")

    print(f"\n  {'✓ Semua test passed!' if all_pass else '✗ Ada test yang gagal!'}")


# ===================================================================
# TEST 3: Prepare Training Data
# ===================================================================
def test_prepare_data(dummy: dict) -> list:
    """Siapkan data training dari dummy patient."""
    separator("TEST 3: Prepare Training Data")

    samples = prepare_training_data_from_patient(
        dummy, modality="flair", include_empty=False,
    )

    print(f"  Total slice dummy    : {dummy['flair'].shape[0]}")
    print(f"  Slice valid (tumor)  : {len(samples)}")

    if samples:
        s = samples[0]
        print(f"\n  Contoh sample[0]:")
        print(f"    slice shape       : {s['slice'].shape}")
        print(f"    seg shape         : {s['seg'].shape}")
        print(f"    cluster_count     : {s['cluster_count']}")
        print(f"    cluster_class     : {s['cluster_class']}")
        print(f"    confidence_map    : {s['confidence_map'].shape} "
              f"(tumor_pct={100*s['confidence_map'].mean():.1f}%)")
        print(f"    slice_idx         : {s['slice_idx']}")

    # Distribusi cluster count
    cluster_dist = {}
    for s in samples:
        k = s["cluster_count"]
        cluster_dist[k] = cluster_dist.get(k, 0) + 1
    print(f"\n  Distribusi cluster count: {cluster_dist}")

    return samples


# ===================================================================
# TEST 4: Training dengan Dummy Data
# ===================================================================
def test_training(dummy: dict) -> tuple:
    """Latih model CNN dengan data dummy."""
    separator("TEST 4: Training CNN (Dummy Data, 5 epochs)")

    H, W = DUMMY_SHAPE[1], DUMMY_SHAPE[2]
    model = build_cnn_model(input_shape=(H, W, 1))

    samples = prepare_training_data_from_patient(
        dummy, modality="flair", include_empty=True,
    )

    # Siapkan array
    slices = np.array([s["slice"] for s in samples], dtype=np.float32)
    segs = np.array([s["seg"] for s in samples], dtype=np.float32)

    # Split train/val (80/20)
    n = len(slices)
    split = max(1, int(n * 0.8))
    train_s, val_s = slices[:split], slices[split:]
    train_seg, val_seg = segs[:split], segs[split:]

    print(f"  Train: {train_s.shape}  Val: {val_s.shape}")

    t0 = time.time()
    history = train_cnn_model(
        model,
        train_slices=train_s,
        train_segs=train_seg,
        val_slices=val_s,
        val_segs=val_seg,
        epochs=5,
        batch_size=4,
    )
    elapsed = time.time() - t0
    print(f"\n  Waktu training: {elapsed:.1f} detik")

    # Print history keys
    print(f"  History keys: {list(history.history.keys())}")

    return model, history


# ===================================================================
# TEST 5: Plot Training History
# ===================================================================
def test_plot_history(history) -> None:
    """Plot loss curves dan accuracy."""
    separator("TEST 5: Plot Training History")

    h = history.history

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        "CNN Training History",
        fontweight="bold", fontsize=14, color="#e94560",
    )

    # ---- Total Loss ----
    ax = axes[0]
    ax.plot(h["loss"], "o-", color="#e94560", label="Train Loss", linewidth=2)
    if "val_loss" in h:
        ax.plot(h["val_loss"], "s--", color="#f5a623", label="Val Loss",
                linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Total Loss", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2, color="#555555")

    # ---- Cluster Count Accuracy ----
    ax = axes[1]
    acc_key = None
    for k in h:
        if "cluster_count" in k and "accuracy" in k and "val" not in k:
            acc_key = k
            break
    val_acc_key = None
    for k in h:
        if "cluster_count" in k and "accuracy" in k and "val" in k:
            val_acc_key = k
            break

    if acc_key:
        ax.plot(h[acc_key], "o-", color="#7ed321",
                label="Train Acc", linewidth=2)
    if val_acc_key:
        ax.plot(h[val_acc_key], "s--", color="#4a90d9",
                label="Val Acc", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Cluster Count Accuracy", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2, color="#555555")
    ax.set_ylim(0, 1.05)

    # ---- Confidence Map Loss ----
    ax = axes[2]
    conf_key = None
    for k in h:
        if "confidence_map" in k and "loss" in k and "val" not in k:
            conf_key = k
            break
    val_conf_key = None
    for k in h:
        if "confidence_map" in k and "loss" in k and "val" in k:
            val_conf_key = k
            break

    if conf_key:
        ax.plot(h[conf_key], "o-", color="#e94560",
                label="Train", linewidth=2)
    if val_conf_key:
        ax.plot(h[val_conf_key], "s--", color="#f5a623",
                label="Val", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Confidence Map Loss", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2, color="#555555")

    plt.tight_layout()
    path = OUTPUT_DIR / "test_cnn_training_history.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Plot disimpan: {path}")


# ===================================================================
# TEST 6: Prediction (inference)
# ===================================================================
def test_prediction(model, dummy: dict) -> None:
    """Uji prediksi pada satu slice."""
    separator("TEST 6: Prediction (Inference)")

    flair_norm = normalize_intensity(dummy["flair"])
    seg = dummy["seg"]

    # Ambil slice dengan tumor
    for idx in range(flair_norm.shape[0]):
        seg_slice = seg[idx]
        if (seg_slice > 0).any():
            break

    mri_slice = flair_norm[idx].astype(np.float32)
    seg_slice = seg[idx].astype(np.float32)

    print(f"  Slice idx: {idx}")
    print(f"  MRI shape: {mri_slice.shape}  range=[{mri_slice.min():.3f}, "
          f"{mri_slice.max():.3f}]")

    # Prediksi cluster count
    k = predict_cluster_count(model, mri_slice)
    k_true = get_cluster_count_from_seg(seg_slice)
    print(f"\n  Cluster count:")
    print(f"    Predicted: k={k}")
    print(f"    Ground truth: k={k_true}")

    # Prediksi confidence map
    conf_map = predict_confidence_map(model, mri_slice)
    gt_conf = (seg_slice > 0).astype(np.float32)

    print(f"\n  Confidence map:")
    print(f"    Shape: {conf_map.shape}")
    print(f"    Range: [{conf_map.min():.4f}, {conf_map.max():.4f}]")
    print(f"    GT tumor pct: {100*gt_conf.mean():.1f}%")
    print(f"    Pred tumor pct (>0.5): "
          f"{100*(conf_map > 0.5).sum()/conf_map.size:.1f}%")

    # ---- Plot: MRI, GT conf, Predicted conf, Overlay ----
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(
        f"CNN Inference — Slice #{idx}  |  "
        f"Predicted k={k}, True k={k_true}",
        fontweight="bold", fontsize=14, color="#e94560",
    )

    axes[0].imshow(mri_slice, cmap="gray", aspect="equal")
    axes[0].set_title("Input MRI (FLAIR)", fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(gt_conf, cmap="hot", aspect="equal", vmin=0, vmax=1)
    axes[1].set_title("Ground Truth\n(seg > 0)", fontweight="bold")
    axes[1].axis("off")

    im2 = axes[2].imshow(conf_map, cmap="hot", aspect="equal", vmin=0, vmax=1)
    axes[2].set_title("Predicted\nConfidence Map", fontweight="bold")
    axes[2].axis("off")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    axes[3].imshow(mri_slice, cmap="gray", aspect="equal")
    conf_overlay = np.ma.masked_where(conf_map < 0.3, conf_map)
    axes[3].imshow(conf_overlay, cmap="jet", alpha=0.6, aspect="equal",
                   vmin=0, vmax=1)
    axes[3].set_title("MRI + Confidence\nOverlay", fontweight="bold")
    axes[3].axis("off")

    plt.tight_layout()
    path = OUTPUT_DIR / "test_cnn_prediction.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  ✓ Plot disimpan: {path}")


# ===================================================================
# TEST 7: Save & Load Model
# ===================================================================
def test_save_load(model) -> None:
    """Uji simpan dan muat model."""
    separator("TEST 7: Save & Load Model")

    model_path = OUTPUT_DIR / "cnn_model.keras"

    save_cnn_model(model, model_path)
    print(f"  Model disimpan: {model_path}")
    print(f"  File size: {model_path.stat().st_size / 1024:.0f} KB")

    loaded = load_cnn_model(model_path)
    print(f"  Model dimuat: params={loaded.count_params():,}")

    # Verifikasi output sama
    dummy_x = np.random.rand(1, DUMMY_SHAPE[1], DUMMY_SHAPE[2], 1).astype(
        np.float32
    )
    orig_out = model.predict(dummy_x, verbose=0)
    load_out = loaded.predict(dummy_x, verbose=0)

    diff_cluster = np.abs(orig_out[0] - load_out[0]).max()
    diff_conf = np.abs(orig_out[1] - load_out[1]).max()
    print(f"  Cluster diff (max): {diff_cluster:.6f}")
    print(f"  Conf diff (max)   : {diff_conf:.6f}")
    print(f"\n  ✓ Save/load verified (diff < 1e-5: "
          f"{diff_cluster < 1e-5 and diff_conf < 1e-5})")


# ===================================================================
# BONUS: Dataset Asli
# ===================================================================
def test_real_dataset() -> None:
    """Training ringan dengan satu pasien BraTS asli."""
    separator("BONUS: CNN Training dengan BraTS 2020 Asli")

    patient_path = DATASET_PATH / SAMPLE_PATIENT
    if not patient_path.is_dir():
        print(f"  ⚠ Dataset tidak ditemukan: {patient_path}")
        return

    print(f"  Memuat pasien: {SAMPLE_PATIENT}")
    data = load_patient_modalities(patient_path)

    # Siapkan data — hanya slice dengan tumor
    samples = prepare_training_data_from_patient(
        data, modality="flair", include_empty=False,
    )
    print(f"  Slice valid (tumor): {len(samples)}")

    if len(samples) < 4:
        print("  ⚠ Terlalu sedikit slice tumor, skip training.")
        return

    # Ambil subset kecil (max 20 slice) agar cepat
    max_samples = min(20, len(samples))
    samples = samples[:max_samples]

    slices = np.array([s["slice"] for s in samples], dtype=np.float32)
    segs = np.array([s["seg"] for s in samples], dtype=np.float32)

    # Resize ke 240×240 (sudah native BraTS)
    H, W = slices.shape[1], slices.shape[2]
    print(f"  Slice shape: ({max_samples}, {H}, {W})")

    # Pastikan H dan W habis dibagi 8
    # BraTS = 240×240, 240/8=30 ✓
    model = build_cnn_model(input_shape=(H, W, 1))

    split = max(1, int(max_samples * 0.8))
    t0 = time.time()
    history = train_cnn_model(
        model,
        train_slices=slices[:split],
        train_segs=segs[:split],
        val_slices=slices[split:],
        val_segs=segs[split:],
        epochs=3,
        batch_size=4,
    )
    elapsed = time.time() - t0
    print(f"  Training time: {elapsed:.1f}s")

    # Prediksi pada slice pertama
    test_slice = slices[0]
    k = predict_cluster_count(model, test_slice)
    conf = predict_confidence_map(model, test_slice)
    seg_slice = segs[0]
    k_true = get_cluster_count_from_seg(seg_slice)

    print(f"\n  Prediksi slice[0]: k={k} (true={k_true})")
    print(f"  Confidence map: range=[{conf.min():.4f}, {conf.max():.4f}]")

    # Plot
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(
        f"CNN Real Data — {SAMPLE_PATIENT}  |  k_pred={k}, k_true={k_true}",
        fontweight="bold", fontsize=14, color="#e94560",
    )

    axes[0].imshow(test_slice, cmap="gray", aspect="equal")
    axes[0].set_title("FLAIR", fontweight="bold")
    axes[0].axis("off")

    gt_conf = (seg_slice > 0).astype(np.float32)
    axes[1].imshow(gt_conf, cmap="hot", aspect="equal", vmin=0, vmax=1)
    axes[1].set_title("GT (seg>0)", fontweight="bold")
    axes[1].axis("off")

    im = axes[2].imshow(conf, cmap="hot", aspect="equal", vmin=0, vmax=1)
    axes[2].set_title("Pred Confidence", fontweight="bold")
    axes[2].axis("off")
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    axes[3].imshow(test_slice, cmap="gray", aspect="equal")
    overlay = np.ma.masked_where(conf < 0.3, conf)
    axes[3].imshow(overlay, cmap="jet", alpha=0.6, aspect="equal",
                   vmin=0, vmax=1)
    axes[3].set_title("Overlay", fontweight="bold")
    axes[3].axis("off")

    plt.tight_layout()
    path = OUTPUT_DIR / "test_cnn_real_patient.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  ✓ Plot disimpan: {path}")

    # Simpan model real
    save_path = OUTPUT_DIR / "cnn_model_real.keras"
    save_cnn_model(model, save_path)


# ===================================================================
# MAIN
# ===================================================================
def main():
    print("\n" + "█" * 60)
    print("█  TEST CNN MODULE — Modul 3")
    print("█" * 60)

    # Test 1: Build model
    test_build_model()

    # Test 2: Cluster count logic
    test_cluster_count_logic()

    # Buat dummy data
    print(f"\n  Menggunakan dummy shape: {DUMMY_SHAPE}")
    dummy = create_dummy_patient_data(shape=DUMMY_SHAPE, seed=42)

    # Test 3: Prepare data
    samples = test_prepare_data(dummy)

    # Test 4: Training
    model, history = test_training(dummy)

    # Test 5: Plot history
    test_plot_history(history)

    # Test 6: Prediction
    test_prediction(model, dummy)

    # Test 7: Save/load
    test_save_load(model)

    # BONUS: Real dataset
    test_real_dataset()

    separator("SEMUA TEST MODUL 3 (CNN) SELESAI ✓")
    print(f"\n  Output plots disimpan di: {OUTPUT_DIR}/")
    print(f"  File CNN yang dihasilkan:")
    for f in sorted(OUTPUT_DIR.glob("test_cnn_*")):
        size_kb = f.stat().st_size / 1024
        print(f"    • {f.name} ({size_kb:.0f} KB)")
    for f in sorted(OUTPUT_DIR.glob("cnn_model*")):
        size_kb = f.stat().st_size / 1024
        print(f"    • {f.name} ({size_kb:.0f} KB)")
    print()


if __name__ == "__main__":
    main()
