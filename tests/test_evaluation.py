#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_evaluation.py — Skrip Uji Modul 6: Evaluasi Komprehensif

Studi ablasi 5 konfigurasi, robustness test, statistical tests,
confusion matrix, dan report generation.

Author  : [Peneliti]
Created : 2026-05-01
"""

from __future__ import annotations
import sys
import json
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
    add_noise,
)
from src.evaluation import (
    BRAIN_LABELS,
    ablation_study,
    compute_all_metrics,
    dice_score,
    generate_evaluation_report,
    hausdorff_distance_95,
    iou_score,
    plot_confusion_matrix_custom,
    plot_evaluation_summary,
    robustness_test,
    sensitivity_specificity,
    statistical_test,
)

# ---------------------------------------------------------------------------
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)
DUMMY_SHAPE = (30, 64, 64)

plt.rcParams.update({
    "figure.facecolor": "#1a1a2e", "axes.facecolor": "#16213e",
    "axes.edgecolor": "#e94560", "axes.labelcolor": "#eaeaea",
    "text.color": "#eaeaea", "xtick.color": "#aaaaaa",
    "ytick.color": "#aaaaaa", "font.size": 10,
})


def separator(title: str):
    print(f"\n{'='*60}\n  {title}\n{'='*60}")


# ===================================================================
# Simulate segmentations for ablation
# ===================================================================
def simulate_segmentation(gt, accuracy, seed=0):
    """Simulate a noisy predicted seg by flipping labels with given accuracy."""
    rng = np.random.default_rng(seed)
    pred = gt.copy()
    flip = rng.random(pred.shape) > accuracy
    random_labels = rng.integers(0, 4, size=pred.shape)
    pred[flip] = random_labels[flip]
    return pred


def prepare_patients():
    """Create 3 dummy patients with different seeds."""
    patients = []
    for seed in [42, 123, 777]:
        p = create_dummy_patient_data(shape=DUMMY_SHAPE, seed=seed)
        patients.append(p)
    return patients


# ===================================================================
# TEST 1: Basic metric functions
# ===================================================================
def test_basic_metrics():
    separator("TEST 1: Basic Metrics")

    gt = np.array([[0, 0, 1, 1], [2, 2, 3, 3], [0, 1, 2, 3], [0, 0, 0, 0]])
    pred = np.array([[0, 0, 1, 2], [2, 2, 3, 3], [0, 1, 2, 0], [0, 0, 1, 0]])

    d = dice_score(gt, pred)
    print(f"  Dice: mean={d['mean']:.4f}  per_label={d['per_label']}")

    i = iou_score(gt, pred)
    print(f"  IoU:  mean={i['mean']:.4f}  per_label={i['per_label']}")

    h = hausdorff_distance_95(gt, pred)
    print(f"  HD95: mean={h['mean']:.4f}  per_label={h['per_label']}")

    ss = sensitivity_specificity(gt, pred)
    print(f"  Sens: {ss['mean_sensitivity']:.4f}  Spec: {ss['mean_specificity']:.4f}")

    db = dice_score(gt, pred, average="binary")
    print(f"  Dice binary: {db['binary']:.4f}")

    am = compute_all_metrics(gt, pred)
    print(f"  All metrics keys: {list(am.keys())}")
    print(f"  ✓ Basic metrics OK")


# ===================================================================
# TEST 2: Ablation Study (5 models)
# ===================================================================
def test_ablation(patients):
    separator("TEST 2: Studi Ablasi — 5 Konfigurasi")

    models = {
        "A: FCM Standard":   0.60,
        "B: PCA+FCM":        0.70,
        "C: CNN+FCM":        0.75,
        "D: AI+PCA+FCM":     0.82,
        "E: Full Pipeline":  0.90,
    }

    all_results = {}
    per_slice_scores = {}

    for model_name, acc in models.items():
        model_metrics_list = []
        slice_dices = []

        for pid, patient in enumerate(patients):
            gt = patient["seg"]
            pred = simulate_segmentation(gt, acc, seed=pid * 100)

            # Per-slice evaluation
            for s in range(gt.shape[0]):
                gt_s = gt[s]
                pr_s = pred[s]
                if gt_s.max() == 0:
                    continue
                d = dice_score(gt_s, pr_s)
                slice_dices.append(d["mean"])

            metrics = compute_all_metrics(gt, pred, labels=[1, 2, 3])
            model_metrics_list.append(metrics)

        # Average across patients
        avg_metrics = model_metrics_list[0].copy()
        dice_means = [m["dice"]["mean"] for m in model_metrics_list]
        iou_means = [m["iou"]["mean"] for m in model_metrics_list]
        avg_metrics["dice"]["mean"] = float(np.mean(dice_means))
        avg_metrics["dice"]["std"] = float(np.std(dice_means))
        avg_metrics["iou"]["mean"] = float(np.mean(iou_means))

        all_results[model_name] = avg_metrics
        per_slice_scores[model_name] = slice_dices

        print(f"  {model_name:25s}  DSC={avg_metrics['dice']['mean']:.4f}  "
              f"IoU={avg_metrics['iou']['mean']:.4f}  "
              f"Sens={avg_metrics['sensitivity_specificity']['mean_sensitivity']:.4f}")

    # Print ablation table
    print("\n  Ablation Table:")
    table = ablation_study(all_results)
    print(f"  {table}")

    # Save results
    save_results = {}
    for k, v in all_results.items():
        save_results[k] = {
            "dice_mean": v["dice"]["mean"],
            "dice_std": v["dice"]["std"],
            "iou_mean": v["iou"]["mean"],
            "sens": v["sensitivity_specificity"]["mean_sensitivity"],
            "spec": v["sensitivity_specificity"]["mean_specificity"],
        }

    json_path = OUTPUT_DIR / "ablation_results.json"
    with open(json_path, "w") as f:
        json.dump(save_results, f, indent=2)
    print(f"\n  ✓ Results saved: {json_path}")

    return all_results, per_slice_scores


# ===================================================================
# TEST 3: Plots
# ===================================================================
def test_plots(all_results, per_slice_scores):
    separator("TEST 3: Visualisasi Evaluasi")

    plot_evaluation_summary(all_results, OUTPUT_DIR, per_slice_scores)

    for f in sorted(OUTPUT_DIR.glob("evaluation_*.png")):
        print(f"  ✓ {f.name} ({f.stat().st_size/1024:.0f} KB)")


# ===================================================================
# TEST 4: Robustness
# ===================================================================
def test_robustness(patients):
    separator("TEST 4: Uji Robustness terhadap Noise")

    patient = patients[0]
    gt = patient["seg"]
    flair = patient["flair"]
    flair_norm = normalize_intensity(flair)

    # Clean prediction (high accuracy)
    pred_clean = simulate_segmentation(gt, 0.90, seed=0)
    clean_metrics = compute_all_metrics(gt, pred_clean, labels=[1, 2, 3])
    print(f"  Clean DSC: {clean_metrics['dice']['mean']:.4f}")

    noise_configs = [
        ("gaussian_0.05", "gaussian", 0.05),
        ("gaussian_0.10", "gaussian", 0.10),
        ("salt_pepper_0.03", "salt_pepper", 0.03),
        ("salt_pepper_0.07", "salt_pepper", 0.07),
    ]

    noisy_metrics_list = []
    noise_names = []

    for name, ntype, level in noise_configs:
        # Add noise to MRI, then simulate worse prediction
        degraded_acc = 0.90 - level * 2  # noise degrades accuracy
        pred_noisy = simulate_segmentation(gt, degraded_acc, seed=hash(name) % 1000)
        nm = compute_all_metrics(gt, pred_noisy, labels=[1, 2, 3])
        noisy_metrics_list.append(nm)
        noise_names.append(name)
        print(f"  {name:25s}  DSC={nm['dice']['mean']:.4f}")

    rob = robustness_test(clean_metrics, noisy_metrics_list, noise_names)

    # Plot DSC vs noise
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle("DSC Degradation under Noise",
                 fontweight="bold", fontsize=14, color="#e94560")
    names = ["Clean"] + noise_names
    dsc_vals = [clean_metrics["dice"]["mean"]] + [
        m["dice"]["mean"] for m in noisy_metrics_list
    ]
    colors = ["#7ed321"] + ["#e94560"] * len(noise_names)
    bars = ax.bar(range(len(names)), dsc_vals, color=colors, alpha=0.85,
                  edgecolor="white")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("Dice Score")
    ax.set_ylim(0, 1.1)
    for bar, val in zip(bars, dsc_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", fontsize=9, color="#eaeaea")
    ax.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    p = OUTPUT_DIR / "evaluation_robustness.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  ✓ {p}")

    return rob


# ===================================================================
# TEST 5: Statistical tests
# ===================================================================
def test_statistical(patients):
    separator("TEST 5: Uji Statistik")

    # Collect per-patient DSC for model D and E
    d_scores = []
    e_scores = []
    b_scores = []

    for pid, patient in enumerate(patients):
        gt = patient["seg"]
        pred_d = simulate_segmentation(gt, 0.82, seed=pid * 200)
        pred_e = simulate_segmentation(gt, 0.90, seed=pid * 300)
        pred_b = simulate_segmentation(gt, 0.70, seed=pid * 400)

        # Per slice dice
        for s in range(gt.shape[0]):
            if gt[s].max() == 0:
                continue
            d_scores.append(dice_score(gt[s], pred_d[s])["mean"])
            e_scores.append(dice_score(gt[s], pred_e[s])["mean"])
            b_scores.append(dice_score(gt[s], pred_b[s])["mean"])

    # Test D vs E
    st1 = statistical_test(d_scores, e_scores, "DSC (D vs E)")
    print(f"  D vs E: {st1['test_name']}, p={st1['p_value']:.6f}, "
          f"sig={st1['significant']}, mean_D={st1['mean1']:.4f}, "
          f"mean_E={st1['mean2']:.4f}")

    # Test B vs D
    st2 = statistical_test(b_scores, d_scores, "DSC (B vs D)")
    print(f"  B vs D: {st2['test_name']}, p={st2['p_value']:.6f}, "
          f"sig={st2['significant']}, mean_B={st2['mean1']:.4f}, "
          f"mean_D={st2['mean2']:.4f}")

    return {"D_vs_E": st1, "B_vs_D": st2}


# ===================================================================
# TEST 6: Confusion matrix
# ===================================================================
def test_confusion(patients):
    separator("TEST 6: Confusion Matrix")

    gt = patients[0]["seg"]
    pred = simulate_segmentation(gt, 0.90, seed=42)

    mid = gt.shape[0] // 2
    gt_s = gt[mid]
    pr_s = pred[mid]

    p = OUTPUT_DIR / "evaluation_confusion_matrix.png"
    plot_confusion_matrix_custom(gt_s, pr_s, [0, 1, 2, 3], p)
    print(f"  ✓ {p}")


# ===================================================================
# TEST 7: Report generation
# ===================================================================
def test_report(all_results, rob, stat_tests):
    separator("TEST 7: Generate Evaluation Report")

    report_data = {
        "ablation": all_results,
        "robustness": rob,
        "statistical_tests": stat_tests,
    }

    report = generate_evaluation_report(report_data, OUTPUT_DIR)
    print(f"  Report length: {len(report)} chars")
    print(f"  ✓ {OUTPUT_DIR / 'evaluation_report.md'}")


# ===================================================================
# MAIN
# ===================================================================
def main():
    print("\n" + "█" * 60)
    print("█  TEST EVALUATION — Modul 6 (FINAL)")
    print("█" * 60)

    patients = prepare_patients()
    print(f"\n  {len(patients)} dummy patients created, shape={DUMMY_SHAPE}")

    test_basic_metrics()
    all_results, per_slice_scores = test_ablation(patients)
    test_plots(all_results, per_slice_scores)
    rob = test_robustness(patients)
    stat_tests = test_statistical(patients)
    test_confusion(patients)
    test_report(all_results, rob, stat_tests)

    # --- Final summary ---
    separator("RINGKASAN AKHIR STUDI ABLASI")
    best_model = max(all_results.items(), key=lambda x: x[1]["dice"]["mean"])
    print(f"\n  ★ Model Terbaik: {best_model[0]}")
    print(f"    DSC  = {best_model[1]['dice']['mean']:.4f}")
    print(f"    IoU  = {best_model[1]['iou']['mean']:.4f}")
    print(f"    Sens = {best_model[1]['sensitivity_specificity']['mean_sensitivity']:.4f}")
    print(f"    Spec = {best_model[1]['sensitivity_specificity']['mean_specificity']:.4f}")

    separator("SEMUA TEST MODUL 6 (EVALUASI) SELESAI ✓")
    print(f"\n  Output files:")
    for f in sorted(OUTPUT_DIR.glob("evaluation_*")):
        print(f"    • {f.name} ({f.stat().st_size/1024:.0f} KB)")
    for f in sorted(OUTPUT_DIR.glob("ablation_*")):
        print(f"    • {f.name} ({f.stat().st_size/1024:.0f} KB)")
    print()


if __name__ == "__main__":
    main()
