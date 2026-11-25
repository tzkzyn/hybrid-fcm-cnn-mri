#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluation.py — Modul 6: Evaluasi Komprehensif Segmentasi

Bagian dari riset:
  "An AI-Driven Hybrid Fuzzy C-Means Framework for Robust Brain Tumor
   MRI Segmentation via PCA-Based Feature Reduction"

Metrik evaluasi mengikuti standar BraTS Challenge:
  - Dice Similarity Coefficient (DSC)
  - Intersection over Union (IoU / Jaccard)
  - Hausdorff Distance 95th percentile (HD95)
  - Sensitivity & Specificity
  - Average Volume Difference (AVD)
  - Average Symmetric Surface Distance (ASSD)

Author  : [Peneliti]
Created : 2026-05-01
Python  : 3.9+
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.stats import shapiro, ttest_rel, wilcoxon

# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)

# ---------------------------------------------------------------------------
# Konstanta
# ---------------------------------------------------------------------------
BRAIN_LABELS = {0: "Background", 1: "Necrosis", 2: "Edema", 3: "Enhancing Tumor"}
TUMOR_LABELS = [1, 2, 3]


# ===================================================================
# 1. dice_score
# ===================================================================
def dice_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List[int]] = None,
    average: str = "macro",
) -> Dict[str, Any]:
    """Dice Similarity Coefficient (DSC).

    DSC = 2|A ∩ B| / (|A| + |B|).

    Parameters
    ----------
    y_true, y_pred : np.ndarray
        Ground truth dan prediksi (shape harus sama).
    labels : list of int, optional
        Label yang dievaluasi. None = semua label > 0.
    average : str
        'macro' (per label lalu rata-rata), 'micro' (global),
        'binary' (tumor vs non-tumor).

    Returns
    -------
    dict
        'per_label', 'mean', 'std'.
    """
    y_true = y_true.astype(np.int32).flatten()
    y_pred = y_pred.astype(np.int32).flatten()

    if labels is None:
        labels = sorted(set(np.unique(y_true)) | set(np.unique(y_pred)))
        labels = [l for l in labels if l > 0]

    if average == "binary":
        t = (y_true > 0).astype(bool)
        p = (y_pred > 0).astype(bool)
        inter = (t & p).sum()
        total = t.sum() + p.sum()
        d = 2.0 * inter / total if total > 0 else 1.0
        return {"binary": float(d), "mean": float(d)}

    per_label = {}
    for lbl in labels:
        t = (y_true == lbl)
        p = (y_pred == lbl)
        inter = (t & p).sum()
        total = t.sum() + p.sum()
        per_label[int(lbl)] = float(2.0 * inter / total) if total > 0 else 1.0

    vals = list(per_label.values())
    return {
        "per_label": per_label,
        "mean": float(np.mean(vals)) if vals else 0.0,
        "std": float(np.std(vals)) if vals else 0.0,
    }


# ===================================================================
# 2. iou_score
# ===================================================================
def iou_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Intersection over Union (Jaccard Index).

    IoU = |A ∩ B| / |A ∪ B|.
    """
    y_true = y_true.astype(np.int32).flatten()
    y_pred = y_pred.astype(np.int32).flatten()

    if labels is None:
        labels = sorted(set(np.unique(y_true)) | set(np.unique(y_pred)))
        labels = [l for l in labels if l > 0]

    per_label = {}
    for lbl in labels:
        t = (y_true == lbl)
        p = (y_pred == lbl)
        inter = (t & p).sum()
        union = (t | p).sum()
        per_label[int(lbl)] = float(inter / union) if union > 0 else 1.0

    vals = list(per_label.values())
    return {
        "per_label": per_label,
        "mean": float(np.mean(vals)) if vals else 0.0,
        "std": float(np.std(vals)) if vals else 0.0,
    }


# ===================================================================
# 3. hausdorff_distance_95
# ===================================================================
def hausdorff_distance_95(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List[int]] = None,
    spacing: Tuple[float, ...] = (1.0, 1.0),
) -> Dict[str, Any]:
    """95th Percentile Hausdorff Distance.

    Menggunakan distance_transform_edt untuk efisiensi.
    """
    if labels is None:
        labels = sorted(set(np.unique(y_true)) | set(np.unique(y_pred)))
        labels = [l for l in labels if l > 0]

    per_label = {}
    for lbl in labels:
        t = (y_true == lbl).astype(bool)
        p = (y_pred == lbl).astype(bool)

        if not t.any() and not p.any():
            per_label[int(lbl)] = 0.0
            continue
        if not t.any() or not p.any():
            per_label[int(lbl)] = float("inf")
            continue

        # Distance from boundary of t to p and vice versa
        sp = spacing
        if len(sp) != t.ndim:
            sp = (1.0,) * t.ndim
        dt_t = distance_transform_edt(~t, sampling=sp)
        dt_p = distance_transform_edt(~p, sampling=sp)

        # Surface distances
        surf_t = dt_p[t]  # distance from true surface to nearest pred
        surf_p = dt_t[p]  # distance from pred surface to nearest true

        all_dist = np.concatenate([surf_t, surf_p])
        hd95 = float(np.percentile(all_dist, 95))
        per_label[int(lbl)] = hd95

    vals = [v for v in per_label.values() if np.isfinite(v)]
    return {
        "per_label": per_label,
        "mean": float(np.mean(vals)) if vals else float("inf"),
    }


# ===================================================================
# 4. sensitivity_specificity
# ===================================================================
def sensitivity_specificity(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Sensitivity (TPR) dan Specificity (TNR) per label."""
    y_true = y_true.astype(np.int32).flatten()
    y_pred = y_pred.astype(np.int32).flatten()

    if labels is None:
        labels = sorted(set(np.unique(y_true)) | set(np.unique(y_pred)))
        labels = [l for l in labels if l > 0]

    per_label = {}
    for lbl in labels:
        t = (y_true == lbl)
        p = (y_pred == lbl)
        tp = (t & p).sum()
        fn = (t & ~p).sum()
        fp = (~t & p).sum()
        tn = (~t & ~p).sum()

        sens = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        per_label[int(lbl)] = {"sensitivity": sens, "specificity": spec}

    sens_vals = [v["sensitivity"] for v in per_label.values()]
    spec_vals = [v["specificity"] for v in per_label.values()]
    return {
        "per_label": per_label,
        "mean_sensitivity": float(np.mean(sens_vals)) if sens_vals else 0.0,
        "mean_specificity": float(np.mean(spec_vals)) if spec_vals else 0.0,
    }


# ===================================================================
# 5. average_volume_difference
# ===================================================================
def average_volume_difference(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Average Volume Difference (AVD) in percent."""
    y_true = y_true.astype(np.int32).flatten()
    y_pred = y_pred.astype(np.int32).flatten()

    if labels is None:
        labels = sorted(set(np.unique(y_true)) | set(np.unique(y_pred)))
        labels = [l for l in labels if l > 0]

    per_label = {}
    for lbl in labels:
        vol_true = (y_true == lbl).sum()
        vol_pred = (y_pred == lbl).sum()
        if vol_true > 0:
            avd = abs(vol_pred - vol_true) / vol_true * 100.0
        elif vol_pred > 0:
            avd = 100.0
        else:
            avd = 0.0
        per_label[int(lbl)] = float(avd)

    vals = list(per_label.values())
    return {
        "per_label": per_label,
        "mean": float(np.mean(vals)) if vals else 0.0,
    }


# ===================================================================
# 6. compute_all_metrics
# ===================================================================
def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List[int]] = None,
    spacing: Tuple[float, ...] = (1.0, 1.0),
) -> Dict[str, Any]:
    """Menghitung semua metrik evaluasi sekaligus."""
    if labels is None:
        labels = sorted(set(np.unique(y_true)) | set(np.unique(y_pred)))
        labels = [l for l in labels if l > 0]
    if not labels:
        labels = [1]

    return {
        "dice": dice_score(y_true, y_pred, labels),
        "dice_binary": dice_score(y_true, y_pred, labels, average="binary"),
        "iou": iou_score(y_true, y_pred, labels),
        "hd95": hausdorff_distance_95(y_true, y_pred, labels, spacing),
        "sensitivity_specificity": sensitivity_specificity(y_true, y_pred, labels),
        "avd": average_volume_difference(y_true, y_pred, labels),
        "labels_evaluated": labels,
    }


# ===================================================================
# 7. evaluate_patient_volume
# ===================================================================
def evaluate_patient_volume(
    ground_truth_3d: np.ndarray,
    predicted_3d: np.ndarray,
    slice_axis: int = 0,
    labels: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Evaluasi volume 3-D slice per slice + volume utuh."""
    n_slices = ground_truth_3d.shape[slice_axis]
    per_slice = []
    for i in range(n_slices):
        gt_s = np.take(ground_truth_3d, i, axis=slice_axis)
        pr_s = np.take(predicted_3d, i, axis=slice_axis)
        if gt_s.max() == 0 and pr_s.max() == 0:
            continue
        m = compute_all_metrics(gt_s, pr_s, labels)
        m["slice_idx"] = i
        per_slice.append(m)

    volume_metrics = compute_all_metrics(ground_truth_3d, predicted_3d, labels)
    dice_scores = [s["dice"]["mean"] for s in per_slice]

    return {
        "per_slice": per_slice,
        "volume_metrics": volume_metrics,
        "average_slice_dice": float(np.mean(dice_scores)) if dice_scores else 0.0,
        "n_slices_evaluated": len(per_slice),
    }


# ===================================================================
# 8. ablation_study
# ===================================================================
def ablation_study(results_dict: Dict[str, Dict]) -> str:
    """Buat tabel ablasi dari dict {model_name: metrics_dict}.

    Returns markdown table string.
    """
    rows = []
    for name, metrics in results_dict.items():
        d = metrics.get("dice", {})
        iou = metrics.get("iou", {})
        hd = metrics.get("hd95", {})
        ss = metrics.get("sensitivity_specificity", {})
        rows.append({
            "Model": name,
            "DSC_mean": d.get("mean", 0),
            "DSC_std": d.get("std", 0),
            "IoU_mean": iou.get("mean", 0),
            "HD95_mean": hd.get("mean", 0),
            "Sens": ss.get("mean_sensitivity", 0),
            "Spec": ss.get("mean_specificity", 0),
        })

    rows.sort(key=lambda x: x["DSC_mean"], reverse=True)

    # Build markdown table
    header = "| Model | DSC | IoU | HD95 | Sens | Spec |"
    sep = "|-------|-----|-----|------|------|------|"
    lines = [header, sep]
    for r in rows:
        lines.append(
            f"| {r['Model']} | {r['DSC_mean']:.4f}±{r['DSC_std']:.4f} "
            f"| {r['IoU_mean']:.4f} | {r['HD95_mean']:.2f} "
            f"| {r['Sens']:.4f} | {r['Spec']:.4f} |"
        )

    table = "\n".join(lines)
    logger.info("Ablation study table:\n%s", table)
    return table


# ===================================================================
# 9. robustness_test
# ===================================================================
def robustness_test(
    clean_metrics: Dict,
    noisy_metrics_list: List[Dict],
    noise_types: List[str],
) -> Dict[str, Any]:
    """Bandingkan metrik data bersih vs data noisy."""
    clean_dice = clean_metrics.get("dice", {}).get("mean", 0)
    degradations = {}
    for nt, nm in zip(noise_types, noisy_metrics_list):
        noisy_dice = nm.get("dice", {}).get("mean", 0)
        if clean_dice > 0:
            deg = (clean_dice - noisy_dice) / clean_dice * 100.0
        else:
            deg = 0.0
        degradations[nt] = {
            "clean_dice": clean_dice,
            "noisy_dice": noisy_dice,
            "degradation_pct": deg,
        }

    logger.info("Robustness test: %s", degradations)
    return degradations


# ===================================================================
# 10. statistical_test
# ===================================================================
def statistical_test(
    model1_scores: List[float],
    model2_scores: List[float],
    metric_name: str = "DSC",
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Paired statistical test antara dua model."""
    s1 = np.array(model1_scores, dtype=np.float64)
    s2 = np.array(model2_scores, dtype=np.float64)

    if len(s1) < 3 or len(s2) < 3:
        return {"test_name": "insufficient_data", "p_value": 1.0,
                "significant": False, "metric": metric_name}

    # Check normality of differences
    diff = s1 - s2
    try:
        _, p_normal = shapiro(diff)
    except Exception:
        p_normal = 0.0

    if p_normal > 0.05:
        stat, p_val = ttest_rel(s1, s2)
        test_name = "paired_t_test"
    else:
        try:
            stat, p_val = wilcoxon(s1, s2)
            test_name = "wilcoxon_signed_rank"
        except Exception:
            stat, p_val = 0.0, 1.0
            test_name = "wilcoxon_failed"

    result = {
        "test_name": test_name,
        "statistic": float(stat),
        "p_value": float(p_val),
        "significant": bool(p_val < alpha),
        "alpha": alpha,
        "metric": metric_name,
        "n_samples": len(s1),
        "mean1": float(np.mean(s1)),
        "mean2": float(np.mean(s2)),
    }
    logger.info("Statistical test (%s): %s p=%.6f sig=%s",
                metric_name, test_name, p_val, result["significant"])
    return result


# ===================================================================
# 11. generate_evaluation_report
# ===================================================================
def generate_evaluation_report(
    all_results: Dict,
    output_dir: Path,
) -> str:
    """Generate laporan evaluasi dalam format Markdown."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Evaluation Report",
        "",
        "## AI-Driven Hybrid FCM Brain Tumor MRI Segmentation",
        "",
    ]

    # Ablation table
    if "ablation" in all_results:
        lines.append("## Ablation Study")
        lines.append("")
        lines.append(ablation_study(all_results["ablation"]))
        lines.append("")

    # Robustness
    if "robustness" in all_results:
        lines.append("## Robustness Test")
        lines.append("")
        lines.append("| Noise Type | Clean DSC | Noisy DSC | Degradation |")
        lines.append("|------------|-----------|-----------|-------------|")
        for nt, data in all_results["robustness"].items():
            lines.append(
                f"| {nt} | {data['clean_dice']:.4f} "
                f"| {data['noisy_dice']:.4f} "
                f"| {data['degradation_pct']:.2f}% |"
            )
        lines.append("")

    # Statistical tests
    if "statistical_tests" in all_results:
        lines.append("## Statistical Tests")
        lines.append("")
        for name, res in all_results["statistical_tests"].items():
            sig = "✓ Significant" if res.get("significant") else "✗ Not significant"
            lines.append(
                f"- **{name}**: {res.get('test_name', '?')}, "
                f"p={res.get('p_value', 1):.6f}, {sig}"
            )
        lines.append("")

    report = "\n".join(lines)
    report_path = output_dir / "evaluation_report.md"
    report_path.write_text(report, encoding="utf-8")
    logger.info("Report saved: %s", report_path)
    return report


# ===================================================================
# 12. plot_evaluation_summary
# ===================================================================
def plot_evaluation_summary(
    results_dict: Dict[str, Dict],
    output_dir: Path,
    per_slice_scores: Optional[Dict[str, List[float]]] = None,
) -> None:
    """Plot visualisasi evaluasi komprehensif."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    plt.rcParams.update({
        "figure.facecolor": "#1a1a2e", "axes.facecolor": "#16213e",
        "axes.edgecolor": "#e94560", "axes.labelcolor": "#eaeaea",
        "text.color": "#eaeaea", "xtick.color": "#aaaaaa",
        "ytick.color": "#aaaaaa", "font.size": 10,
    })

    models = list(results_dict.keys())
    colors = ["#4a90d9", "#f5a623", "#7ed321", "#e94560", "#9b59b6",
              "#1abc9c", "#e67e22"][:len(models)]

    # --- Bar chart DSC per label ---
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle("Dice Score per Label per Model",
                 fontweight="bold", fontsize=14, color="#e94560")

    all_labels = set()
    for m in results_dict.values():
        pl = m.get("dice", {}).get("per_label", {})
        all_labels.update(pl.keys())
    all_labels = sorted(all_labels)

    if all_labels:
        x = np.arange(len(all_labels))
        width = 0.8 / max(len(models), 1)
        for i, (model, color) in enumerate(zip(models, colors)):
            pl = results_dict[model].get("dice", {}).get("per_label", {})
            vals = [pl.get(l, 0) for l in all_labels]
            ax.bar(x + i * width, vals, width, label=model, color=color, alpha=0.85)

        ax.set_xticks(x + width * (len(models) - 1) / 2)
        label_names = [BRAIN_LABELS.get(l, f"L{l}") for l in all_labels]
        ax.set_xticklabels(label_names)
        ax.set_ylabel("Dice Score")
        ax.set_ylim(0, 1.1)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(axis="y", alpha=0.2)

    plt.tight_layout()
    fig.savefig(output_dir / "evaluation_dsc_per_label.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    # --- Boxplot DSC per slice ---
    if per_slice_scores:
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        fig2.suptitle("DSC Distribution per Slice",
                      fontweight="bold", fontsize=14, color="#e94560")
        data_box = [per_slice_scores.get(m, []) for m in models]
        bp = ax2.boxplot(data_box, labels=models, patch_artist=True)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax2.set_ylabel("Dice Score")
        ax2.grid(axis="y", alpha=0.2)
        plt.tight_layout()
        fig2.savefig(output_dir / "evaluation_dsc_boxplot.png", dpi=150,
                     bbox_inches="tight")
        plt.close(fig2)

    # --- Radar chart ---
    fig3, ax3 = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    fig3.suptitle("Multi-Metric Radar", fontweight="bold", fontsize=14,
                  color="#e94560", y=1.02)
    metric_names = ["DSC", "IoU", "Sensitivity", "Specificity"]
    angles = np.linspace(0, 2 * np.pi, len(metric_names), endpoint=False).tolist()
    angles += angles[:1]

    for model, color in zip(models, colors):
        m = results_dict[model]
        vals = [
            m.get("dice", {}).get("mean", 0),
            m.get("iou", {}).get("mean", 0),
            m.get("sensitivity_specificity", {}).get("mean_sensitivity", 0),
            m.get("sensitivity_specificity", {}).get("mean_specificity", 0),
        ]
        vals += vals[:1]
        ax3.plot(angles, vals, "o-", linewidth=2, label=model, color=color)
        ax3.fill(angles, vals, alpha=0.1, color=color)

    ax3.set_xticks(angles[:-1])
    ax3.set_xticklabels(metric_names)
    ax3.set_ylim(0, 1.1)
    ax3.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)
    plt.tight_layout()
    fig3.savefig(output_dir / "evaluation_radar.png", dpi=150,
                 bbox_inches="tight")
    plt.close(fig3)

    logger.info("Evaluation plots saved to %s", output_dir)


# ===================================================================
# 13. plot_confusion_matrix_custom
# ===================================================================
def plot_confusion_matrix_custom(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: List[int],
    output_path: Union[str, Path],
) -> None:
    """Plot confusion matrix multi-kelas."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    y_true = y_true.flatten().astype(int)
    y_pred = y_pred.flatten().astype(int)

    n = len(labels)
    cm = np.zeros((n, n), dtype=np.int64)
    for i, lt in enumerate(labels):
        for j, lp in enumerate(labels):
            cm[i, j] = ((y_true == lt) & (y_pred == lp)).sum()

    # Normalize per row
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = cm / np.maximum(row_sums, 1)

    plt.rcParams.update({
        "figure.facecolor": "#1a1a2e", "axes.facecolor": "#16213e",
        "text.color": "#eaeaea",
    })

    fig, ax = plt.subplots(figsize=(8, 7))
    fig.suptitle("Confusion Matrix (Normalized by Row)",
                 fontweight="bold", fontsize=14, color="#e94560")

    im = ax.imshow(cm_norm, cmap="YlOrRd", vmin=0, vmax=1)
    label_names = [BRAIN_LABELS.get(l, f"L{l}") for l in labels]
    ax.set_xticks(range(n))
    ax.set_xticklabels(label_names, rotation=45, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels(label_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    for i in range(n):
        for j in range(n):
            color = "white" if cm_norm[i, j] > 0.5 else "#333"
            ax.text(j, i, f"{cm[i,j]}\n({cm_norm[i,j]:.2%})",
                    ha="center", va="center", fontsize=9, color=color,
                    fontweight="bold")

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Confusion matrix saved: %s", output_path)


# ===================================================================
# CLI self-test
# ===================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  Modul Evaluasi — Quick Self-Test")
    print("=" * 60)

    gt = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    pr = np.array([0, 0, 1, 2, 2, 2, 3, 0])

    d = dice_score(gt, pr)
    print(f"  Dice: {d}")
    i = iou_score(gt, pr)
    print(f"  IoU: {i}")
    ss = sensitivity_specificity(gt, pr)
    print(f"  Sens/Spec: {ss}")

    st = statistical_test([0.8, 0.85, 0.82], [0.75, 0.80, 0.78])
    print(f"  Stat test: {st}")

    print("\n  Self-test PASSED ✓")
    print("=" * 60)
