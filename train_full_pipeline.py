#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_full_pipeline.py — Full Training on 369 BraTS 2020 Patients (Google Colab GPU)

Usage in Colab:
  1. Upload src/ folder to /content/src/
  2. Mount Google Drive with dataset at /content/drive/MyDrive/Brats2020/training/
  3. Run: !python train_full_pipeline.py

Author  : [Peneliti]
Created : 2026-05-01
"""

import gc
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# ── Environment ──────────────────────────────────────────────────────────────
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# ── Paths ────────────────────────────────────────────────────────────────────
DRIVE_ROOT    = Path("/content/drive/MyDrive")
DRIVE_DATASET = DRIVE_ROOT / "Brats2020" / "training"
LOCAL_DATASET = Path("/content/data/Brats2020/training")
SRC_DIR       = Path("/content/src")
OUTPUT_DIR    = DRIVE_ROOT / "training_output"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
LOG_FILE      = Path("/content/training.log")

# ── Fallback for local testing ───────────────────────────────────────────────
IS_COLAB = "google.colab" in sys.modules or Path("/content").exists()
if not IS_COLAB:
    # Local dev mode
    PROJECT = Path(__file__).resolve().parent
    SRC_DIR = PROJECT / "src"
    LOCAL_DATASET = PROJECT / "data" / "Brats2020" / "training"
    DRIVE_DATASET = LOCAL_DATASET
    OUTPUT_DIR = PROJECT / "output" / "full_training"
    CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
    LOG_FILE = PROJECT / "output" / "training.log"

sys.path.insert(0, str(SRC_DIR.parent))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(str(LOG_FILE), mode="a"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("train_full")

# ── Imports (after path setup) ───────────────────────────────────────────────
import tensorflow as tf
from tensorflow import keras

from src.preprocessing import load_patient_modalities, normalize_intensity
from src.cnn_module import (
    build_cnn_model, get_cluster_count_from_seg,
    predict_cluster_count, predict_confidence_map, train_cnn_model,
)
from src.feature_extraction import extract_patch_features, reduce_with_pca
from src.hybrid_fcm import hybrid_fcm_segmentation
from src.postprocessing import (
    build_refiner_unet, postprocess_morphology,
    prepare_refiner_data, refine_segmentation, train_refiner,
    save_refiner_model,
)
from src.evaluation import (
    compute_all_metrics, ablation_study, generate_evaluation_report,
    plot_evaluation_summary, plot_confusion_matrix_custom,
)

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kw):
        total = kw.get("total", "?")
        for i, x in enumerate(it):
            print(f"  [{i+1}/{total}]", end="\r")
            yield x


# ═══════════════════════════════════════════════════════════════════════════
# 0. SETUP & GPU CHECK
# ═══════════════════════════════════════════════════════════════════════════
def setup():
    log.info("=" * 70)
    log.info("  FULL PIPELINE TRAINING — BraTS 2020")
    log.info("=" * 70)

    gpus = tf.config.list_physical_devices("GPU")
    log.info("TensorFlow %s  |  GPUs: %s", tf.__version__, gpus)
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        log.info("GPU memory growth enabled")
    else:
        log.warning("⚠ No GPU detected — training will be SLOW")

    # Copy dataset to local if on Colab
    if IS_COLAB and DRIVE_DATASET.exists() and not LOCAL_DATASET.exists():
        log.info("Copying dataset from Drive to local storage...")
        LOCAL_DATASET.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(str(DRIVE_DATASET), str(LOCAL_DATASET))
        log.info("Dataset copied to %s", LOCAL_DATASET)


# ═══════════════════════════════════════════════════════════════════════════
# 1. SPLIT PATIENTS
# ═══════════════════════════════════════════════════════════════════════════
def split_patients() -> Tuple[List[Path], List[Path], List[Path]]:
    dataset = LOCAL_DATASET if LOCAL_DATASET.exists() else DRIVE_DATASET
    patients = sorted([d for d in dataset.iterdir() if d.is_dir()
                       and d.name.startswith("BraTS20_Training")])
    log.info("Total patients found: %d in %s", len(patients), dataset)

    rng = np.random.default_rng(42)
    idx = rng.permutation(len(patients))
    n = len(patients)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)

    train = [patients[i] for i in idx[:n_train]]
    val   = [patients[i] for i in idx[n_train:n_train + n_val]]
    test  = [patients[i] for i in idx[n_train + n_val:]]

    log.info("Split: train=%d  val=%d  test=%d", len(train), len(val), len(test))

    # Save split for reproducibility
    split_info = {
        "train": [p.name for p in train],
        "val":   [p.name for p in val],
        "test":  [p.name for p in test],
    }
    with open(OUTPUT_DIR / "data_split.json", "w") as f:
        json.dump(split_info, f, indent=2)

    return train, val, test


# ═══════════════════════════════════════════════════════════════════════════
# 2. COLLECT CNN TRAINING DATA
# ═══════════════════════════════════════════════════════════════════════════
def collect_cnn_data(patient_paths: List[Path], max_patients: int = None,
                     ) -> Tuple[np.ndarray, np.ndarray]:
    """Load patients and collect slices with tumor for CNN training."""
    slices_list, segs_list = [], []
    paths = patient_paths[:max_patients] if max_patients else patient_paths

    for p in tqdm(paths, desc="Loading CNN data", total=len(paths)):
        try:
            data = load_patient_modalities(p)
            flair = data["flair"]
            seg = data["seg"]
            flair_norm = normalize_intensity(flair)

            for idx in range(flair.shape[0]):
                seg_s = seg[idx]
                if seg_s.max() == 0:
                    continue
                slices_list.append(flair_norm[idx].astype(np.float32))
                segs_list.append(seg_s.astype(np.float32))
            del data, flair, seg, flair_norm
            gc.collect()
        except Exception as e:
            log.warning("Skip %s: %s", p.name, e)

    X = np.array(slices_list, dtype=np.float32)
    y = np.array(segs_list, dtype=np.float32)
    log.info("CNN data collected: X=%s  y=%s", X.shape, y.shape)
    return X, y


# ═══════════════════════════════════════════════════════════════════════════
# 3. TRAIN CNN
# ═══════════════════════════════════════════════════════════════════════════
def train_cnn_phase(train_paths, val_paths):
    log.info("\n" + "=" * 70)
    log.info("  PHASE 1: TRAINING CNN (cluster count + confidence map)")
    log.info("=" * 70)

    cnn_path = OUTPUT_DIR / "cnn_full_model.keras"
    if cnn_path.exists():
        log.info("CNN model found, loading: %s", cnn_path)
        return keras.models.load_model(str(cnn_path))

    X_train, y_train = collect_cnn_data(train_paths)
    X_val, y_val = collect_cnn_data(val_paths)

    H, W = X_train.shape[1], X_train.shape[2]
    model = build_cnn_model(input_shape=(H, W, 1))

    batch_size = 16
    try:
        history = train_cnn_model(
            model, X_train, y_train, X_val, y_val,
            epochs=50, batch_size=batch_size,
            checkpoint_path=str(CHECKPOINT_DIR / "cnn_best.keras"),
        )
    except tf.errors.ResourceExhaustedError:
        log.warning("OOM with batch=%d, retrying with batch=4", batch_size)
        gc.collect()
        history = train_cnn_model(
            model, X_train, y_train, X_val, y_val,
            epochs=50, batch_size=4,
            checkpoint_path=str(CHECKPOINT_DIR / "cnn_best.keras"),
        )

    model.save(str(cnn_path))
    log.info("CNN saved: %s", cnn_path)

    # Save history
    hist = {k: [float(v) for v in vs] for k, vs in history.history.items()}
    with open(OUTPUT_DIR / "cnn_history.json", "w") as f:
        json.dump(hist, f, indent=2)

    del X_train, y_train, X_val, y_val
    gc.collect()
    return model


# ═══════════════════════════════════════════════════════════════════════════
# 4. GENERATE COARSE SEGMENTATION + TRAIN REFINER
# ═══════════════════════════════════════════════════════════════════════════
def generate_coarse_seg(patient_path, cnn_model, modality="flair"):
    """Run FCM on each slice to produce coarse segmentation volume."""
    data = load_patient_modalities(patient_path)
    flair = data[modality]
    seg = data["seg"]
    flair_norm = normalize_intensity(flair)
    D, H, W = flair.shape
    coarse_vol = np.zeros((D, H, W), dtype=np.int32)

    for idx in range(D):
        seg_s = seg[idx]
        mri_s = flair_norm[idx].astype(np.float32)

        if seg_s.max() == 0:
            continue

        try:
            k = predict_cluster_count(cnn_model, mri_s)
            conf = predict_confidence_map(cnn_model, mri_s)
            feats = extract_patch_features(mri_s, patch_size=3)
            proj, _, cents, _ = reduce_with_pca(feats, n_components=3, n_clusters=k)
            result = hybrid_fcm_segmentation(
                proj, cents, conf, n_clusters=k,
                max_fcm_iter=30, use_metaheuristic=False,
            )
            coarse_vol[idx] = result["seg_labels"]
        except Exception:
            coarse_vol[idx] = 0

    return data, coarse_vol


def train_refiner_phase(train_paths, val_paths, cnn_model):
    log.info("\n" + "=" * 70)
    log.info("  PHASE 2: TRAINING REFINER U-NET")
    log.info("=" * 70)

    refiner_path = OUTPUT_DIR / "refiner_full_model.keras"
    if refiner_path.exists():
        log.info("Refiner found, loading: %s", refiner_path)
        return keras.models.load_model(str(refiner_path))

    X_all, y_all = [], []
    max_train = min(len(train_paths), 50)  # limit for memory
    for p in tqdm(train_paths[:max_train], desc="Refiner data", total=max_train):
        try:
            data, coarse = generate_coarse_seg(p, cnn_model)
            X, y = prepare_refiner_data(data, coarse, modality="flair")
            X_all.append(X)
            y_all.append(y)
            del data, coarse, X, y
            gc.collect()
        except Exception as e:
            log.warning("Skip refiner data %s: %s", p.name, e)

    X_train = np.concatenate(X_all, axis=0)
    y_train = np.concatenate(y_all, axis=0)
    del X_all, y_all
    gc.collect()
    log.info("Refiner train data: X=%s y=%s", X_train.shape, y_train.shape)

    # Validation
    X_v, y_v = [], []
    for p in tqdm(val_paths[:10], desc="Refiner val", total=min(10, len(val_paths))):
        try:
            data, coarse = generate_coarse_seg(p, cnn_model)
            X, y = prepare_refiner_data(data, coarse)
            X_v.append(X)
            y_v.append(y)
            del data, coarse
            gc.collect()
        except Exception as e:
            log.warning("Skip val %s: %s", p.name, e)

    X_val = np.concatenate(X_v, axis=0) if X_v else None
    y_val = np.concatenate(y_v, axis=0) if y_v else None

    H, W = X_train.shape[1], X_train.shape[2]
    model = build_refiner_unet(input_shape=(H, W, 2), n_classes=4)

    batch_size = 8
    try:
        history = train_refiner(
            model, X_train, y_train, X_val, y_val,
            epochs=50, batch_size=batch_size, patience=10,
            checkpoint_path=CHECKPOINT_DIR / "refiner_best.keras",
        )
    except tf.errors.ResourceExhaustedError:
        log.warning("OOM, retrying with batch=2")
        gc.collect()
        history = train_refiner(
            model, X_train, y_train, X_val, y_val,
            epochs=50, batch_size=2, patience=10,
        )

    save_refiner_model(model, refiner_path)
    hist = {k: [float(v) for v in vs] for k, vs in history.history.items()}
    with open(OUTPUT_DIR / "refiner_history.json", "w") as f:
        json.dump(hist, f, indent=2)

    del X_train, y_train, X_val, y_val
    gc.collect()
    return model


# ═══════════════════════════════════════════════════════════════════════════
# 5. FULL PIPELINE INFERENCE ON ONE PATIENT
# ═══════════════════════════════════════════════════════════════════════════
def run_full_pipeline(patient_path, cnn_model, refiner_model, modality="flair"):
    """Run complete pipeline on one patient, return metrics dict."""
    data = load_patient_modalities(patient_path)
    flair = data[modality]
    seg_gt = data["seg"]
    flair_norm = normalize_intensity(flair)
    D, H, W = flair.shape
    pred_vol = np.zeros((D, H, W), dtype=np.int32)

    for idx in range(D):
        mri_s = flair_norm[idx].astype(np.float32)
        if seg_gt[idx].max() == 0 and mri_s.max() < 0.01:
            continue
        try:
            k = predict_cluster_count(cnn_model, mri_s)
            conf = predict_confidence_map(cnn_model, mri_s)
            feats = extract_patch_features(mri_s, patch_size=3)
            proj, _, cents, _ = reduce_with_pca(feats, n_components=3, n_clusters=k)
            fcm_res = hybrid_fcm_segmentation(
                proj, cents, conf, n_clusters=k,
                max_fcm_iter=30, use_metaheuristic=False,
            )
            coarse = fcm_res["seg_labels"]
            refined = refine_segmentation(refiner_model, mri_s, coarse, n_classes=4)
            cleaned = postprocess_morphology(refined, min_object_size=30)
            pred_vol[idx] = cleaned
        except Exception:
            pred_vol[idx] = 0

    metrics = compute_all_metrics(seg_gt, pred_vol, labels=[1, 2, 3])
    return metrics, pred_vol


# ═══════════════════════════════════════════════════════════════════════════
# 6. EVALUATE TEST SET
# ═══════════════════════════════════════════════════════════════════════════
def evaluate_test_set(test_paths, cnn_model, refiner_model):
    log.info("\n" + "=" * 70)
    log.info("  PHASE 3: EVALUATION ON TEST SET (%d patients)", len(test_paths))
    log.info("=" * 70)

    results_path = OUTPUT_DIR / "test_results.json"
    all_metrics = []

    for p in tqdm(test_paths, desc="Evaluating", total=len(test_paths)):
        try:
            metrics, _ = run_full_pipeline(p, cnn_model, refiner_model)
            metrics["patient"] = p.name
            all_metrics.append(metrics)
            dsc = metrics["dice"]["mean"]
            log.info("  %s  DSC=%.4f", p.name, dsc)
        except Exception as e:
            log.warning("  %s FAILED: %s", p.name, e)

    # Aggregate
    dice_means = [m["dice"]["mean"] for m in all_metrics]
    iou_means  = [m["iou"]["mean"] for m in all_metrics]
    hd95_means = [m["hd95"]["mean"] for m in all_metrics]
    sens_means = [m["sensitivity_specificity"]["mean_sensitivity"] for m in all_metrics]
    spec_means = [m["sensitivity_specificity"]["mean_specificity"] for m in all_metrics]

    summary = {
        "n_patients": len(all_metrics),
        "DSC_mean":  float(np.mean(dice_means)),
        "DSC_std":   float(np.std(dice_means)),
        "IoU_mean":  float(np.mean(iou_means)),
        "IoU_std":   float(np.std(iou_means)),
        "HD95_mean": float(np.mean([v for v in hd95_means if np.isfinite(v)])),
        "Sens_mean": float(np.mean(sens_means)),
        "Spec_mean": float(np.mean(spec_means)),
        "per_patient": [{
            "patient": m.get("patient", "?"),
            "dice_mean": m["dice"]["mean"],
            "iou_mean": m["iou"]["mean"],
        } for m in all_metrics],
    }

    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    log.info("Results saved: %s", results_path)

    # CSV
    csv_path = OUTPUT_DIR / "test_results.csv"
    with open(csv_path, "w") as f:
        f.write("patient,DSC,IoU,HD95,Sensitivity,Specificity\n")
        for m in all_metrics:
            f.write(f"{m.get('patient','?')},"
                    f"{m['dice']['mean']:.4f},"
                    f"{m['iou']['mean']:.4f},"
                    f"{m['hd95']['mean']:.4f},"
                    f"{m['sensitivity_specificity']['mean_sensitivity']:.4f},"
                    f"{m['sensitivity_specificity']['mean_specificity']:.4f}\n")
    log.info("CSV saved: %s", csv_path)

    log.info("\n  ══════════════════════════════════════")
    log.info("  TEST SET RESULTS (%d patients)", len(all_metrics))
    log.info("  DSC:  %.4f ± %.4f", summary["DSC_mean"], summary["DSC_std"])
    log.info("  IoU:  %.4f ± %.4f", summary["IoU_mean"], summary["IoU_std"])
    log.info("  HD95: %.4f", summary["HD95_mean"])
    log.info("  Sens: %.4f  Spec: %.4f", summary["Sens_mean"], summary["Spec_mean"])
    log.info("  ══════════════════════════════════════")

    return summary, all_metrics


# ═══════════════════════════════════════════════════════════════════════════
# 7. GENERATE FINAL REPORT
# ═══════════════════════════════════════════════════════════════════════════
def generate_final_report(summary):
    log.info("\n  Generating final report...")

    lines = [
        "# Final Evaluation Report",
        "",
        "## AI-Driven Hybrid FCM Brain Tumor MRI Segmentation",
        f"## BraTS 2020 — {summary['n_patients']} Test Patients",
        "",
        "### Overall Performance",
        "",
        "| Metric | Mean ± Std |",
        "|--------|-----------|",
        f"| DSC | {summary['DSC_mean']:.4f} ± {summary['DSC_std']:.4f} |",
        f"| IoU | {summary['IoU_mean']:.4f} ± {summary['IoU_std']:.4f} |",
        f"| HD95 | {summary['HD95_mean']:.4f} |",
        f"| Sensitivity | {summary['Sens_mean']:.4f} |",
        f"| Specificity | {summary['Spec_mean']:.4f} |",
        "",
    ]

    report = "\n".join(lines)
    report_path = OUTPUT_DIR / "final_report.md"
    report_path.write_text(report, encoding="utf-8")
    log.info("Report saved: %s", report_path)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════
def main():
    t_start = time.time()
    setup()

    train_paths, val_paths, test_paths = split_patients()

    # Phase 1: CNN
    cnn_model = train_cnn_phase(train_paths, val_paths)

    # Phase 2: Refiner
    refiner_model = train_refiner_phase(train_paths, val_paths, cnn_model)

    # Phase 3: Evaluate
    summary, all_metrics = evaluate_test_set(test_paths, cnn_model, refiner_model)

    # Phase 4: Report
    generate_final_report(summary)

    elapsed = time.time() - t_start
    log.info("\n" + "=" * 70)
    log.info("  TRAINING COMPLETE  |  Total time: %.1f min (%.1f hrs)",
             elapsed / 60, elapsed / 3600)
    log.info("  Models saved to: %s", OUTPUT_DIR)
    log.info("=" * 70)


if __name__ == "__main__":
    main()
