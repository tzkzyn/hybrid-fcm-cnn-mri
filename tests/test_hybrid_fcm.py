#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_hybrid_fcm.py — Skrip Uji Modul 4: Hybrid FCM + Metaheuristic

Menguji:
  1. FCM standar (tanpa confidence, tanpa MPA)
  2. Hybrid FCM + confidence map (tanpa MPA)
  3. Hybrid FCM + confidence map + MPA
  4. Perbandingan ketiganya (J, iterasi, Dice)
  5. Visualisasi dan plot konvergensi

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
)
from src.feature_extraction import (
    extract_patch_features,
    reduce_with_pca,
    FEATURE_NAMES,
)
from src.hybrid_fcm import (
    initialize_membership_soft,
    fuzzy_c_means_single_step,
    fcm_objective_function,
    hybrid_fcm_segmentation,
    reconstruct_segmentation_map,
    segment_patient_slice,
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


def dice_score(pred: np.ndarray, gt: np.ndarray) -> float:
    """Dice Score kasar: 2*|A∩B| / (|A|+|B|) pada binary masks."""
    pred_bin = (pred > 0).astype(bool)
    gt_bin = (gt > 0).astype(bool)
    intersection = (pred_bin & gt_bin).sum()
    total = pred_bin.sum() + gt_bin.sum()
    if total == 0:
        return 1.0
    return 2.0 * intersection / total


# ===================================================================
# Prepare data (shared across tests)
# ===================================================================
def prepare_test_data():
    """Buat data dummy, ekstrak fitur & PCA, siapkan confidence map."""
    separator("PERSIAPAN DATA")
    dummy = create_dummy_patient_data(shape=DUMMY_SHAPE, seed=42)

    flair = dummy["flair"]
    seg = dummy["seg"]
    flair_norm = normalize_intensity(flair)

    mid = flair_norm.shape[0] // 2
    mri_slice = extract_slice(flair_norm, slice_idx=mid, axis=0)
    seg_slice = extract_slice(seg, slice_idx=mid, axis=0)

    print(f"  Slice shape: {mri_slice.shape}")
    print(f"  Seg labels: {np.unique(seg_slice)}")

    # Feature extraction (GLCM mahal, tapi 64x64 feasible)
    t0 = time.time()
    features = extract_patch_features(mri_slice, patch_size=5)
    print(f"  Feature extraction: {time.time()-t0:.1f}s  shape={features.shape}")

    # PCA
    projected, pca_model, init_centroids, expl_var = reduce_with_pca(
        features, n_components=3, n_clusters=3,
    )
    print(f"  PCA projected: {projected.shape}  explained={expl_var:.4f}")
    print(f"  Initial centroids:\n{init_centroids}")

    # Confidence map from GT
    conf_map = (seg_slice > 0).astype(np.float32)
    print(f"  Confidence map (GT): tumor_pct={100*conf_map.mean():.1f}%")

    return {
        "mri_slice": mri_slice,
        "seg_slice": seg_slice,
        "features": features,
        "projected": projected,
        "init_centroids": init_centroids,
        "conf_map": conf_map,
        "dummy": dummy,
    }


# ===================================================================
# TEST 1: FCM standar
# ===================================================================
def test_standard_fcm(td: dict) -> dict:
    separator("TEST 1: FCM Standar (tanpa confidence, tanpa MPA)")

    t0 = time.time()
    result = hybrid_fcm_segmentation(
        data_2d_features=td["projected"],
        initial_centroids=td["init_centroids"],
        confidence_map=None,          # no CNN
        n_clusters=3,
        fuzzifier=2.0,
        max_fcm_iter=100,
        tol=1e-5,
        use_metaheuristic=False,      # no MPA
    )
    elapsed = time.time() - t0

    seg = result["seg_labels"]
    d = dice_score(seg, td["seg_slice"])
    print(f"  Waktu: {elapsed:.1f}s  Iterasi: {result['n_iterations']}")
    print(f"  Final J: {result['objective_history'][-1]:.4f}")
    print(f"  Dice Score (binary): {d:.4f}")
    result["dice"] = d
    result["elapsed"] = elapsed
    return result


# ===================================================================
# TEST 2: Hybrid FCM + confidence map
# ===================================================================
def test_hybrid_conf(td: dict) -> dict:
    separator("TEST 2: Hybrid FCM + Confidence Map (tanpa MPA)")

    t0 = time.time()
    result = hybrid_fcm_segmentation(
        data_2d_features=td["projected"],
        initial_centroids=td["init_centroids"],
        confidence_map=td["conf_map"],  # with CNN confidence
        n_clusters=3,
        fuzzifier=2.0,
        max_fcm_iter=100,
        tol=1e-5,
        use_metaheuristic=False,
    )
    elapsed = time.time() - t0

    seg = result["seg_labels"]
    d = dice_score(seg, td["seg_slice"])
    print(f"  Waktu: {elapsed:.1f}s  Iterasi: {result['n_iterations']}")
    print(f"  Final J: {result['objective_history'][-1]:.4f}")
    print(f"  Dice Score (binary): {d:.4f}")
    result["dice"] = d
    result["elapsed"] = elapsed
    return result


# ===================================================================
# TEST 3: Hybrid FCM + confidence + MPA
# ===================================================================
def test_hybrid_mpa(td: dict) -> dict:
    separator("TEST 3: Hybrid FCM + Confidence + MPA")

    t0 = time.time()
    result = hybrid_fcm_segmentation(
        data_2d_features=td["projected"],
        initial_centroids=td["init_centroids"],
        confidence_map=td["conf_map"],
        n_clusters=3,
        fuzzifier=2.0,
        max_fcm_iter=30,
        tol=1e-5,
        use_metaheuristic=True,       # MPA enabled
        mpa_every=5,
        mpa_epoch=5,
        mpa_pop=10,
    )
    elapsed = time.time() - t0

    seg = result["seg_labels"]
    d = dice_score(seg, td["seg_slice"])
    print(f"  Waktu: {elapsed:.1f}s  Iterasi: {result['n_iterations']}")
    print(f"  Final J: {result['objective_history'][-1]:.4f}")
    print(f"  Fuzzifier final: {result['fuzzifier_final']:.3f}")
    print(f"  Dice Score (binary): {d:.4f}")
    result["dice"] = d
    result["elapsed"] = elapsed
    return result


# ===================================================================
# TEST 4: Perbandingan
# ===================================================================
def test_comparison(r1, r2, r3):
    separator("TEST 4: Perbandingan 3 Varian")

    print(f"  {'Metode':<30s} {'J_final':>10s} {'Iters':>6s} "
          f"{'Dice':>7s} {'Time':>7s}")
    print(f"  {'-'*65}")
    for name, r in [("FCM Standar", r1),
                     ("Hybrid+Conf", r2),
                     ("Hybrid+Conf+MPA", r3)]:
        print(f"  {name:<30s} {r['objective_history'][-1]:>10.4f} "
              f"{r['n_iterations']:>6d} {r['dice']:>7.4f} "
              f"{r['elapsed']:>6.1f}s")


# ===================================================================
# TEST 5: Visualisasi
# ===================================================================
def test_plots(td, r1, r2, r3):
    separator("TEST 5: Visualisasi")

    seg_gt = td["seg_slice"]
    mri = td["mri_slice"]

    # --- Plot 1: Segmentasi comparison ---
    fig, axes = plt.subplots(1, 5, figsize=(22, 4))
    fig.suptitle("Perbandingan Segmentasi — 3 Varian FCM",
                 fontweight="bold", fontsize=14, color="#e94560")

    axes[0].imshow(mri, cmap="gray", aspect="equal")
    axes[0].set_title("Input FLAIR", fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(seg_gt, cmap="Set1", aspect="equal", vmin=0, vmax=3,
                   interpolation="nearest")
    axes[1].set_title("Ground Truth", fontweight="bold")
    axes[1].axis("off")

    titles = ["FCM Standar", "Hybrid+Conf", "Hybrid+Conf+MPA"]
    for i, (r, t) in enumerate(zip([r1, r2, r3], titles)):
        axes[i+2].imshow(r["seg_labels"], cmap="Set1", aspect="equal",
                         vmin=0, vmax=3, interpolation="nearest")
        axes[i+2].set_title(f"{t}\nDice={r['dice']:.3f}", fontweight="bold")
        axes[i+2].axis("off")

    plt.tight_layout()
    p = OUTPUT_DIR / "test_hybrid_fcm_comparison.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {p}")

    # --- Plot 2: Convergence ---
    fig2, ax2 = plt.subplots(1, 1, figsize=(10, 6))
    fig2.suptitle("Konvergensi Fungsi Objektif J",
                  fontweight="bold", fontsize=14, color="#e94560")

    for label, r, color, ls in [
        ("FCM Standar", r1, "#4a90d9", "-"),
        ("Hybrid+Conf", r2, "#f5a623", "--"),
        ("Hybrid+Conf+MPA", r3, "#e94560", "-."),
    ]:
        ax2.plot(r["objective_history"], color=color, linestyle=ls,
                 linewidth=2, label=f"{label} (J={r['objective_history'][-1]:.2f})")

    ax2.set_xlabel("Iterasi FCM", fontsize=12)
    ax2.set_ylabel("J (Objektif)", fontsize=12)
    ax2.legend(fontsize=10, framealpha=0.8)
    ax2.grid(alpha=0.2, color="#555555")

    plt.tight_layout()
    p2 = OUTPUT_DIR / "test_hybrid_fcm_convergence.png"
    fig2.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"  ✓ {p2}")

    # --- Plot 3: Confidence map + overlay ---
    fig3, axes3 = plt.subplots(1, 3, figsize=(16, 5))
    fig3.suptitle("Confidence Map & FCM Overlay",
                  fontweight="bold", fontsize=14, color="#e94560")

    axes3[0].imshow(td["conf_map"], cmap="hot", aspect="equal", vmin=0, vmax=1)
    axes3[0].set_title("Confidence Map (GT)", fontweight="bold")
    axes3[0].axis("off")

    axes3[1].imshow(mri, cmap="gray", aspect="equal")
    seg_overlay = np.ma.masked_where(r3["seg_labels"] == 0, r3["seg_labels"])
    axes3[1].imshow(seg_overlay, cmap="Set1", alpha=0.5, aspect="equal",
                    vmin=0, vmax=3)
    axes3[1].set_title("MRI + Hybrid Seg", fontweight="bold")
    axes3[1].axis("off")

    # Membership heatmap for cluster 1 (tumor)
    U = r3["U_final"]
    H, W = td["seg_slice"].shape
    tumor_membership = U[:, 1].reshape(H, W)
    im = axes3[2].imshow(tumor_membership, cmap="hot", aspect="equal",
                         vmin=0, vmax=1)
    axes3[2].set_title("Membership Cluster 1\n(Tumor)", fontweight="bold")
    axes3[2].axis("off")
    plt.colorbar(im, ax=axes3[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    p3 = OUTPUT_DIR / "test_hybrid_fcm_overlay.png"
    fig3.savefig(p3, dpi=150, bbox_inches="tight")
    plt.close(fig3)
    print(f"  ✓ {p3}")


# ===================================================================
# TEST 6: segment_patient_slice end-to-end
# ===================================================================
def test_end_to_end(dummy):
    separator("TEST 6: End-to-End segment_patient_slice")

    t0 = time.time()
    result = segment_patient_slice(
        dummy,
        cnn_model=None,   # will use GT-based confidence
        modality="flair",
        slice_idx=-1,
        use_metaheuristic=False,  # fast test
        patch_size=5,
        max_fcm_iter=30,
    )
    elapsed = time.time() - t0

    print(f"  Waktu: {elapsed:.1f}s")
    print(f"  n_clusters: {result['n_clusters']}")
    print(f"  Iterasi: {result['n_iterations']}")
    print(f"  seg_labels shape: {result['seg_labels'].shape}")
    if result["seg_gt"] is not None:
        d = dice_score(result["seg_labels"], result["seg_gt"])
        print(f"  Dice (binary): {d:.4f}")


# ===================================================================
# BONUS: CNN model if available
# ===================================================================
def test_with_cnn(dummy):
    separator("BONUS: End-to-End dengan CNN Model")

    model_path = OUTPUT_DIR / "cnn_model.keras"
    if not model_path.exists():
        print(f"  ⚠ CNN model tidak ditemukan: {model_path}")
        return

    from src.cnn_module import load_cnn_model
    cnn = load_cnn_model(model_path)

    result = segment_patient_slice(
        dummy,
        cnn_model=cnn,
        modality="flair",
        slice_idx=-1,
        use_metaheuristic=False,
        patch_size=5,
        max_fcm_iter=30,
    )
    print(f"  CNN predicted k={result['n_clusters']}")
    print(f"  Iterasi: {result['n_iterations']}")
    if result["seg_gt"] is not None:
        d = dice_score(result["seg_labels"], result["seg_gt"])
        print(f"  Dice (binary): {d:.4f}")


# ===================================================================
# MAIN
# ===================================================================
def main():
    print("\n" + "█" * 60)
    print("█  TEST HYBRID FCM — Modul 4")
    print("█" * 60)

    td = prepare_test_data()
    r1 = test_standard_fcm(td)
    r2 = test_hybrid_conf(td)
    r3 = test_hybrid_mpa(td)
    test_comparison(r1, r2, r3)
    test_plots(td, r1, r2, r3)
    test_end_to_end(td["dummy"])
    test_with_cnn(td["dummy"])

    separator("SEMUA TEST MODUL 4 SELESAI ✓")
    print(f"\n  Output:")
    for f in sorted(OUTPUT_DIR.glob("test_hybrid_fcm_*.png")):
        print(f"    • {f.name} ({f.stat().st_size/1024:.0f} KB)")
    print()


if __name__ == "__main__":
    main()
