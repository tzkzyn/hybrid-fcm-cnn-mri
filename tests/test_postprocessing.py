#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_postprocessing.py — Skrip Uji Modul 5: Light U-Net Refiner

Menguji:
  1. Build refiner U-Net & summary
  2. Simulasi coarse segmentation
  3. Prepare refiner data
  4. Training refiner (5 epochs dummy)
  5. Refine segmentation inference
  6. Morphological postprocessing
  7. Visualisasi perbandingan coarse vs refined vs GT

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
    normalize_intensity,
    get_brain_mask,
)
from src.postprocessing import (
    build_refiner_unet,
    prepare_refiner_data,
    train_refiner,
    refine_segmentation,
    postprocess_morphology,
    save_refiner_model,
    load_refiner_model,
)

# ---------------------------------------------------------------------------
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)
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


def separator(title: str):
    print(f"\n{'='*60}\n  {title}\n{'='*60}")


def dice_score(pred, gt):
    pred_bin = (pred > 0).astype(bool)
    gt_bin = (gt > 0).astype(bool)
    inter = (pred_bin & gt_bin).sum()
    total = pred_bin.sum() + gt_bin.sum()
    return 2.0 * inter / total if total > 0 else 1.0


# ===================================================================
# TEST 1: Build Model
# ===================================================================
def test_build():
    separator("TEST 1: Build Refiner U-Net")
    H, W = DUMMY_SHAPE[1], DUMMY_SHAPE[2]
    model = build_refiner_unet(input_shape=(H, W, 2), n_classes=4)
    model.summary(line_length=90)

    dummy_x = np.random.rand(2, H, W, 2).astype(np.float32)
    out = model.predict(dummy_x, verbose=0)
    print(f"\n  Output shape: {out.shape}  (expected (2, {H}, {W}, 4))")
    print(f"  Params: {model.count_params():,}")
    print(f"  ✓ Build OK")
    return model


# ===================================================================
# TEST 2: Simulate coarse seg & prepare data
# ===================================================================
def test_prepare_data(dummy):
    separator("TEST 2: Simulasi Coarse Seg & Prepare Data")

    seg_gt = dummy["seg"]
    # Simulasi coarse seg: add noise to GT (swap ~10% of labels)
    rng = np.random.default_rng(42)
    coarse = seg_gt.copy()
    noise_mask = rng.random(coarse.shape) < 0.10
    coarse[noise_mask] = rng.integers(0, 4, size=noise_mask.sum())

    print(f"  GT seg shape: {seg_gt.shape}")
    print(f"  Coarse seg shape: {coarse.shape}")
    agreement = (coarse == seg_gt).mean()
    print(f"  Agreement GT vs coarse: {agreement:.2%}")

    X, y = prepare_refiner_data(dummy, coarse, modality="flair", n_classes=4)
    print(f"  X shape: {X.shape}  y shape: {y.shape}")
    print(f"  X[0] channels: MRI range=[{X[0,:,:,0].min():.3f}, {X[0,:,:,0].max():.3f}]"
          f"  coarse range=[{X[0,:,:,1].min():.3f}, {X[0,:,:,1].max():.3f}]")

    return X, y, coarse


# ===================================================================
# TEST 3: Training
# ===================================================================
def test_training(model, X, y):
    separator("TEST 3: Training Refiner (5 epochs)")

    n = len(X)
    split = max(1, int(n * 0.8))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    print(f"  Train: {X_train.shape}  Val: {X_val.shape}")

    t0 = time.time()
    history = train_refiner(
        model, X_train, y_train, X_val, y_val,
        epochs=5, batch_size=4, patience=10,
    )
    elapsed = time.time() - t0
    print(f"  Waktu: {elapsed:.1f}s")
    print(f"  Final loss: {history.history['loss'][-1]:.4f}")

    return model, history


# ===================================================================
# TEST 4: Inference
# ===================================================================
def test_inference(model, dummy, coarse):
    separator("TEST 4: Refine Segmentation (Inference)")

    flair_norm = normalize_intensity(dummy["flair"])
    seg_gt = dummy["seg"]
    mid = flair_norm.shape[0] // 2

    mri_slice = extract_slice(flair_norm, mid, axis=0)
    coarse_slice = extract_slice(coarse, mid, axis=0)
    gt_slice = extract_slice(seg_gt, mid, axis=0)

    refined = refine_segmentation(model, mri_slice, coarse_slice, n_classes=4)

    d_coarse = dice_score(coarse_slice, gt_slice)
    d_refined = dice_score(refined, gt_slice)

    print(f"  Dice coarse: {d_coarse:.4f}")
    print(f"  Dice refined: {d_refined:.4f}")
    print(f"  Improvement: {d_refined - d_coarse:+.4f}")

    return mri_slice, coarse_slice, refined, gt_slice


# ===================================================================
# TEST 5: Morphological postprocessing
# ===================================================================
def test_morphology(refined, dummy):
    separator("TEST 5: Morphological Postprocessing")

    flair_norm = normalize_intensity(dummy["flair"])
    mid = flair_norm.shape[0] // 2
    flair_slice = extract_slice(flair_norm, mid, axis=0)
    brain = get_brain_mask(flair_norm, threshold=0.1)
    brain_slice = extract_slice(brain, mid, axis=0)

    cleaned = postprocess_morphology(
        refined, brain_mask=brain_slice,
        min_hole_size=50, min_object_size=20,
    )

    changed = int((refined != cleaned).sum())
    print(f"  Pixels changed: {changed}")
    print(f"  Labels before: {np.unique(refined)}")
    print(f"  Labels after:  {np.unique(cleaned)}")

    return cleaned


# ===================================================================
# TEST 6: Plots
# ===================================================================
def test_plots(mri, coarse, refined, cleaned, gt, history):
    separator("TEST 6: Visualisasi")

    # --- Comparison plot ---
    fig, axes = plt.subplots(1, 5, figsize=(24, 4.5))
    fig.suptitle("Postprocessing — Coarse → Refined → Cleaned",
                 fontweight="bold", fontsize=14, color="#e94560")

    titles = ["Input FLAIR", "Coarse (FCM)", "Refined (U-Net)",
              "Cleaned (Morph)", "Ground Truth"]
    images = [mri, coarse, refined, cleaned, gt]
    cmaps = ["gray", "Set1", "Set1", "Set1", "Set1"]

    for i, (img, t, cm) in enumerate(zip(images, titles, cmaps)):
        axes[i].imshow(img, cmap=cm, aspect="equal",
                       vmin=0 if cm == "Set1" else None,
                       vmax=3 if cm == "Set1" else None,
                       interpolation="nearest" if cm == "Set1" else "bilinear")
        if i > 0:
            d = dice_score(img, gt)
            axes[i].set_title(f"{t}\nDice={d:.3f}", fontweight="bold")
        else:
            axes[i].set_title(t, fontweight="bold")
        axes[i].axis("off")

    plt.tight_layout()
    p = OUTPUT_DIR / "test_postprocessing_comparison.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {p}")

    # --- Loss curve ---
    fig2, ax = plt.subplots(1, 1, figsize=(8, 5))
    fig2.suptitle("Refiner Training Loss",
                  fontweight="bold", fontsize=14, color="#e94560")
    h = history.history
    ax.plot(h["loss"], "o-", color="#e94560", linewidth=2, label="Train")
    if "val_loss" in h:
        ax.plot(h["val_loss"], "s--", color="#f5a623", linewidth=2, label="Val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(alpha=0.2, color="#555555")
    plt.tight_layout()
    p2 = OUTPUT_DIR / "test_postprocessing_loss.png"
    fig2.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"  ✓ {p2}")


# ===================================================================
# TEST 7: Save/Load
# ===================================================================
def test_save_load(model):
    separator("TEST 7: Save & Load Refiner")
    path = OUTPUT_DIR / "refiner_model.keras"
    save_refiner_model(model, path)
    print(f"  Saved: {path}  ({path.stat().st_size/1024:.0f} KB)")

    loaded = load_refiner_model(path)
    print(f"  Loaded params: {loaded.count_params():,}")

    dummy_x = np.random.rand(1, DUMMY_SHAPE[1], DUMMY_SHAPE[2], 2).astype(np.float32)
    diff = np.abs(model.predict(dummy_x, verbose=0) -
                  loaded.predict(dummy_x, verbose=0)).max()
    print(f"  Max diff: {diff:.8f}  ✓")


# ===================================================================
# MAIN
# ===================================================================
def main():
    print("\n" + "█" * 60)
    print("█  TEST POSTPROCESSING — Modul 5")
    print("█" * 60)

    dummy = create_dummy_patient_data(shape=DUMMY_SHAPE, seed=42)

    model = test_build()
    X, y, coarse = test_prepare_data(dummy)
    model, history = test_training(model, X, y)
    mri, coarse_s, refined, gt = test_inference(model, dummy, coarse)
    cleaned = test_morphology(refined, dummy)
    test_plots(mri, coarse_s, refined, cleaned, gt, history)
    test_save_load(model)

    separator("SEMUA TEST MODUL 5 SELESAI ✓")
    print(f"\n  Output:")
    for f in sorted(OUTPUT_DIR.glob("test_postprocessing_*")):
        print(f"    • {f.name} ({f.stat().st_size/1024:.0f} KB)")
    for f in sorted(OUTPUT_DIR.glob("refiner_*")):
        print(f"    • {f.name} ({f.stat().st_size/1024:.0f} KB)")
    print()


if __name__ == "__main__":
    main()
