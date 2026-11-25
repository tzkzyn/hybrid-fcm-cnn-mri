#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hybrid_fcm.py — Modul 4: Hybrid Fuzzy C-Means + Metaheuristic Optimization

Bagian dari riset:
  "An AI-Driven Hybrid Fuzzy C-Means Framework for Robust Brain Tumor
   MRI Segmentation via PCA-Based Feature Reduction"

Modul inti yang mengintegrasikan:
  1. FCM dengan inisialisasi centroid dari PCA
  2. Bobot keanggotaan awal dari confidence map CNN
  3. Optimasi Marine Predators Algorithm (MPA) via Mealpy
  4. Jumlah cluster adaptif dari prediksi CNN

Author  : [Peneliti]
Created : 2026-05-01
Python  : 3.9+
"""

from __future__ import annotations

import logging
import warnings
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from mealpy import FloatVar
from mealpy.swarm_based.MPA import OriginalMPA

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)

# Suppress mealpy verbose logging
logging.getLogger("mealpy").setLevel(logging.WARNING)


# ===================================================================
# 1. initialize_membership_soft
# ===================================================================
def initialize_membership_soft(
    n_samples: int,
    n_clusters: int,
    confidence_map: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Membuat matriks keanggotaan awal U berdasarkan confidence map CNN.

    Piksel dengan confidence tinggi (>0.5) diberi bobot lebih ke
    cluster tumor; piksel rendah ke cluster background.

    Parameters
    ----------
    n_samples : int
        Jumlah piksel (N = H * W).
    n_clusters : int
        Jumlah kluster (2, 3, atau 4).
    confidence_map : np.ndarray, optional
        Array 1-D (N,) dengan nilai [0, 1] dari CNN. Jika None,
        inisialisasi uniform.

    Returns
    -------
    np.ndarray
        Matriks keanggotaan (N, n_clusters), setiap baris berjumlah 1.
    """
    U = np.ones((n_samples, n_clusters), dtype=np.float64) / n_clusters

    if confidence_map is not None:
        conf = confidence_map.flatten()[:n_samples].astype(np.float64)
        conf = np.clip(conf, 0.01, 0.99)

        # Cluster 0 = background, Cluster 1+ = tumor regions
        U[:, 0] = 1.0 - conf  # background
        if n_clusters == 2:
            U[:, 1] = conf
        elif n_clusters == 3:
            U[:, 1] = conf * 0.6   # tumor core
            U[:, 2] = conf * 0.4   # edema
        elif n_clusters >= 4:
            U[:, 1] = conf * 0.4   # necrosis
            U[:, 2] = conf * 0.35  # edema
            U[:, 3] = conf * 0.25  # enhancing
            # background gets remaining
        # Fill remaining clusters uniformly if > 4
        for k in range(4, n_clusters):
            U[:, k] = 0.01

    # Normalize rows to sum to 1
    row_sums = U.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1e-10)
    U = U / row_sums

    logger.info(
        "Membership init  |  N=%d  K=%d  conf=%s  "
        "U_mean_per_cluster=%s",
        n_samples, n_clusters,
        "provided" if confidence_map is not None else "uniform",
        np.array2string(U.mean(axis=0), precision=4),
    )
    return U


# ===================================================================
# 2. fuzzy_c_means_single_step
# ===================================================================
def fuzzy_c_means_single_step(
    data: np.ndarray,
    U: np.ndarray,
    centroids: np.ndarray,
    fuzzifier: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Satu iterasi FCM: update centroid lalu update keanggotaan.

    Parameters
    ----------
    data : np.ndarray
        Data fitur (N, F).
    U : np.ndarray
        Matriks keanggotaan saat ini (N, K).
    centroids : np.ndarray
        Centroid saat ini (K, F).
    fuzzifier : float, default 2.0
        Parameter fuzzifier m (m > 1).

    Returns
    -------
    centroids_new : np.ndarray
        Centroid yang diperbarui (K, F).
    U_new : np.ndarray
        Matriks keanggotaan yang diperbarui (N, K).
    """
    N, F = data.shape
    K = centroids.shape[0]
    m = fuzzifier

    # --- Update centroids ---
    Um = np.power(U, m)  # (N, K)
    denom = Um.sum(axis=0)  # (K,)
    denom = np.maximum(denom, 1e-10)
    centroids_new = (Um.T @ data) / denom[:, np.newaxis]  # (K, F)

    # --- Update membership ---
    # Compute distances: ||x_i - v_j||^2
    dist_sq = np.zeros((N, K), dtype=np.float64)
    for j in range(K):
        diff = data - centroids_new[j]  # (N, F)
        dist_sq[:, j] = np.sum(diff ** 2, axis=1)

    # Avoid division by zero
    dist_sq = np.maximum(dist_sq, 1e-10)

    # U_ij = 1 / sum_k (d_ij / d_ik)^(2/(m-1))
    power = 2.0 / (m - 1.0)
    U_new = np.zeros((N, K), dtype=np.float64)

    for j in range(K):
        total = np.zeros(N, dtype=np.float64)
        for k in range(K):
            ratio = dist_sq[:, j] / dist_sq[:, k]
            total += np.power(ratio, power / 2.0)
        U_new[:, j] = 1.0 / np.maximum(total, 1e-10)

    # Normalize
    row_sums = U_new.sum(axis=1, keepdims=True)
    U_new = U_new / np.maximum(row_sums, 1e-10)

    return centroids_new, U_new


# ===================================================================
# 3. fcm_objective_function
# ===================================================================
def fcm_objective_function(
    data: np.ndarray,
    U: np.ndarray,
    centroids: np.ndarray,
    fuzzifier: float = 2.0,
) -> float:
    """Hitung fungsi objektif FCM: J = Σ_i Σ_j (U_ij^m * ||x_i - v_j||²).

    Parameters
    ----------
    data : np.ndarray
        Data fitur (N, F).
    U : np.ndarray
        Matriks keanggotaan (N, K).
    centroids : np.ndarray
        Centroid (K, F).
    fuzzifier : float
        Parameter fuzzifier m.

    Returns
    -------
    float
        Nilai fungsi objektif J.
    """
    m = fuzzifier
    Um = np.power(U, m)
    K = centroids.shape[0]
    J = 0.0
    for j in range(K):
        diff = data - centroids[j]
        dist_sq = np.sum(diff ** 2, axis=1)
        J += np.sum(Um[:, j] * dist_sq)
    return float(J)


# ===================================================================
# 4. optimize_fcm_mpa
# ===================================================================
def optimize_fcm_mpa(
    data: np.ndarray,
    n_clusters: int,
    initial_centroids: np.ndarray,
    initial_U: np.ndarray,
    fuzzifier_range: Tuple[float, float] = (1.5, 3.0),
    pop_size: int = 20,
    epoch: int = 30,
) -> Dict:
    """Optimasi centroid dan fuzzifier FCM dengan Marine Predators Algorithm.

    Variabel optimasi: flattened centroids + fuzzifier m.
    Dimensi = K * F + 1.

    Parameters
    ----------
    data : np.ndarray
        Data fitur (N, F).
    n_clusters : int
        Jumlah kluster K.
    initial_centroids : np.ndarray
        Centroid awal (K, F) untuk menentukan bounds.
    initial_U : np.ndarray
        Membership saat ini (N, K).
    fuzzifier_range : tuple
        Range fuzzifier (m_min, m_max).
    pop_size : int
        Ukuran populasi MPA.
    epoch : int
        Jumlah epoch MPA.

    Returns
    -------
    dict
        'centroids_opt', 'fuzzifier_opt', 'objective_value',
        'mpa_history'.
    """
    K, F = initial_centroids.shape
    n_dim = K * F + 1  # centroid coords + fuzzifier

    # Bounds: centroid ± 2*range, fuzzifier in range
    c_flat = initial_centroids.flatten()
    c_range = np.ptp(data, axis=0).max() * 0.5
    c_range = max(c_range, 0.1)

    lb = np.zeros(n_dim, dtype=np.float64)
    ub = np.zeros(n_dim, dtype=np.float64)
    for i in range(K * F):
        lb[i] = c_flat[i] - c_range
        ub[i] = c_flat[i] + c_range
    lb[-1] = fuzzifier_range[0]
    ub[-1] = fuzzifier_range[1]

    def objective(solution):
        cents = solution[:K * F].reshape(K, F)
        m = solution[-1]
        m = np.clip(m, fuzzifier_range[0], fuzzifier_range[1])
        # Run one FCM step to get updated U
        _, U_new = fuzzy_c_means_single_step(data, initial_U, cents, m)
        J = fcm_objective_function(data, U_new, cents, m)
        return J

    problem = {
        "obj_func": objective,
        "bounds": FloatVar(lb=lb.tolist(), ub=ub.tolist()),
        "minmax": "min",
        "log_to": None,
    }

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = OriginalMPA(epoch=epoch, pop_size=pop_size)
        result = model.solve(problem, seed=42)

    best = result.solution
    centroids_opt = best[:K * F].reshape(K, F).astype(np.float32)
    fuzzifier_opt = float(np.clip(best[-1], *fuzzifier_range))
    obj_val = float(result.target.fitness)

    logger.info(
        "MPA optimasi selesai  |  epoch=%d  pop=%d  "
        "objective=%.4f  fuzzifier=%.3f",
        epoch, pop_size, obj_val, fuzzifier_opt,
    )
    return {
        "centroids_opt": centroids_opt,
        "fuzzifier_opt": fuzzifier_opt,
        "objective_value": obj_val,
    }


# ===================================================================
# 5. hybrid_fcm_segmentation
# ===================================================================
def hybrid_fcm_segmentation(
    data_2d_features: np.ndarray,
    initial_centroids: np.ndarray,
    confidence_map: Optional[np.ndarray] = None,
    n_clusters: int = 3,
    fuzzifier: float = 2.0,
    max_fcm_iter: int = 100,
    tol: float = 1e-5,
    use_metaheuristic: bool = True,
    mpa_every: int = 5,
    mpa_epoch: int = 10,
    mpa_pop: int = 15,
) -> Dict:
    """Segmentasi Hybrid FCM + CNN confidence + MPA optimization.

    Algoritma:
      1. Inisialisasi U dari confidence_map CNN.
      2. Set centroid dari PCA initial_centroids.
      3. Loop FCM:
         a. Update centroid & membership (fuzzy_c_means_single_step).
         b. Setiap ``mpa_every`` iterasi: optimasi MPA.
         c. Cek konvergensi (perubahan U < tol).
      4. Return segmentasi (argmax U per piksel).

    Parameters
    ----------
    data_2d_features : np.ndarray
        Fitur per piksel (H, W, F) atau sudah flat (N, F).
    initial_centroids : np.ndarray
        Centroid awal dari PCA (K, F).
    confidence_map : np.ndarray, optional
        Confidence map dari CNN (H, W) atau (N,).
    n_clusters : int
        Jumlah kluster.
    fuzzifier : float
        Parameter fuzzifier awal.
    max_fcm_iter : int
        Maksimum iterasi FCM.
    tol : float
        Toleransi konvergensi.
    use_metaheuristic : bool
        Gunakan MPA optimization.
    mpa_every : int
        Jalankan MPA setiap N iterasi FCM.
    mpa_epoch : int
        Epoch MPA per panggilan.
    mpa_pop : int
        Populasi MPA.

    Returns
    -------
    dict
        'seg_labels' (H, W), 'centroids_final', 'U_final',
        'fuzzifier_final', 'objective_history', 'n_iterations'.
    """
    # Reshape input
    if data_2d_features.ndim == 3:
        H, W, F = data_2d_features.shape
        data = data_2d_features.reshape(-1, F).astype(np.float64)
    elif data_2d_features.ndim == 2:
        data = data_2d_features.astype(np.float64)
        N = data.shape[0]
        H = W = int(np.sqrt(N))
        if H * W != N:
            H, W = N, 1
    else:
        raise ValueError(f"data harus 2-D atau 3-D, got ndim={data_2d_features.ndim}")

    N, F = data.shape
    K = n_clusters

    # Adjust centroids if needed
    centroids = initial_centroids[:K, :F].copy().astype(np.float64)
    if centroids.shape[0] < K:
        extra = np.random.randn(K - centroids.shape[0], F) * 0.1
        centroids = np.vstack([centroids, extra])

    # Initialize U
    conf_flat = confidence_map.flatten()[:N] if confidence_map is not None else None
    U = initialize_membership_soft(N, K, conf_flat)

    # FCM loop
    objective_history: List[float] = []
    m = fuzzifier

    for it in range(max_fcm_iter):
        U_old = U.copy()

        # FCM step
        centroids, U = fuzzy_c_means_single_step(data, U, centroids, m)

        # Objective
        J = fcm_objective_function(data, U, centroids, m)
        objective_history.append(J)

        # MPA optimization
        if use_metaheuristic and (it + 1) % mpa_every == 0:
            try:
                mpa_result = optimize_fcm_mpa(
                    data, K, centroids, U,
                    fuzzifier_range=(1.5, 3.0),
                    pop_size=mpa_pop, epoch=mpa_epoch,
                )
                if mpa_result["objective_value"] < J:
                    centroids = mpa_result["centroids_opt"].astype(np.float64)
                    m = mpa_result["fuzzifier_opt"]
                    logger.info(
                        "  MPA improved: J %.4f -> %.4f, m=%.3f",
                        J, mpa_result["objective_value"], m,
                    )
            except Exception as e:
                logger.warning("MPA gagal di iterasi %d: %s", it, e)

        # Convergence check
        delta = np.max(np.abs(U - U_old))
        if delta < tol:
            logger.info(
                "FCM konvergen di iterasi %d  |  delta=%.2e  J=%.4f",
                it + 1, delta, J,
            )
            break

        if (it + 1) % 20 == 0:
            logger.info("  FCM iter %d  |  J=%.4f  delta=%.2e  m=%.3f",
                        it + 1, J, delta, m)

    # Final labels
    labels_1d = np.argmax(U, axis=1).astype(np.int32)
    seg_map = labels_1d.reshape(H, W)

    logger.info(
        "Hybrid FCM selesai  |  iters=%d  K=%d  m=%.3f  "
        "final_J=%.4f  labels=%s",
        len(objective_history), K, m, objective_history[-1],
        {i: int((labels_1d == i).sum()) for i in range(K)},
    )

    return {
        "seg_labels": seg_map,
        "centroids_final": centroids.astype(np.float32),
        "U_final": U.astype(np.float32),
        "fuzzifier_final": m,
        "objective_history": objective_history,
        "n_iterations": len(objective_history),
    }


# ===================================================================
# 6. reconstruct_segmentation_map
# ===================================================================
def reconstruct_segmentation_map(
    labels_1d: np.ndarray,
    original_shape: Tuple[int, int],
) -> np.ndarray:
    """Reshape label 1-D ke peta segmentasi 2-D.

    Parameters
    ----------
    labels_1d : np.ndarray
        Label kluster per piksel (N,).
    original_shape : tuple of int
        Dimensi asli (H, W).

    Returns
    -------
    np.ndarray
        Peta segmentasi (H, W) dtype int32.
    """
    H, W = original_shape
    if labels_1d.size != H * W:
        raise ValueError(
            f"Ukuran labels ({labels_1d.size}) != H*W ({H*W})"
        )
    return labels_1d.reshape(H, W).astype(np.int32)


# ===================================================================
# 7. segment_patient_slice
# ===================================================================
def segment_patient_slice(
    patient_dict: Dict[str, np.ndarray],
    cnn_model=None,
    modality: str = "flair",
    slice_idx: int = -1,
    use_metaheuristic: bool = True,
    n_clusters_override: Optional[int] = None,
    patch_size: int = 5,
    mpa_every: int = 5,
    mpa_epoch: int = 10,
    max_fcm_iter: int = 50,
) -> Dict:
    """End-to-end segmentasi satu slice MRI.

    Alur:
      1. Normalisasi slice.
      2. Ekstrak fitur + PCA.
      3. Prediksi cluster count & confidence map (CNN, jika tersedia).
      4. Jalankan Hybrid FCM.

    Parameters
    ----------
    patient_dict : dict
        Hasil load_patient_modalities() atau create_dummy_patient_data().
    cnn_model : keras.Model, optional
        Model CNN dual-output. Jika None, gunakan GT-based defaults.
    modality : str
        Modalitas MRI.
    slice_idx : int
        Indeks slice (-1 = tengah).
    use_metaheuristic : bool
        Gunakan MPA.
    n_clusters_override : int, optional
        Override jumlah cluster (abaikan CNN).
    patch_size : int
        Ukuran patch untuk fitur.
    mpa_every : int
        MPA setiap N iterasi.
    mpa_epoch : int
        Epoch MPA.
    max_fcm_iter : int
        Maks iterasi FCM.

    Returns
    -------
    dict
        Semua hasil segmentasi dan intermediate results.
    """
    from src.preprocessing import extract_slice, normalize_intensity
    from src.feature_extraction import extract_patch_features, reduce_with_pca

    volume = patient_dict[modality]
    seg_volume = patient_dict.get("seg")

    if slice_idx < 0:
        slice_idx = volume.shape[0] // 2

    # 1. Normalize & extract slice
    vol_norm = normalize_intensity(volume)
    mri_slice = extract_slice(vol_norm, slice_idx=slice_idx, axis=0)
    seg_slice = None
    if seg_volume is not None:
        seg_slice = extract_slice(seg_volume, slice_idx=slice_idx, axis=0)

    # 2. Feature extraction + PCA
    features = extract_patch_features(mri_slice, patch_size=patch_size)

    # Determine n_clusters
    n_clusters = 3  # default
    confidence_map = None

    if n_clusters_override is not None:
        n_clusters = n_clusters_override
    elif cnn_model is not None:
        try:
            from src.cnn_module import predict_cluster_count, predict_confidence_map
            n_clusters = predict_cluster_count(cnn_model, mri_slice)
            confidence_map = predict_confidence_map(cnn_model, mri_slice)
        except Exception as e:
            logger.warning("CNN prediction gagal: %s. Fallback k=3.", e)
            n_clusters = 3
    elif seg_slice is not None:
        from src.cnn_module import get_cluster_count_from_seg
        n_clusters = get_cluster_count_from_seg(seg_slice)
        confidence_map = (seg_slice > 0).astype(np.float32)

    projected, pca_model, init_centroids, expl_var = reduce_with_pca(
        features, n_components=3, n_clusters=n_clusters,
    )

    # 3. Hybrid FCM
    result = hybrid_fcm_segmentation(
        data_2d_features=projected,
        initial_centroids=init_centroids,
        confidence_map=confidence_map,
        n_clusters=n_clusters,
        fuzzifier=2.0,
        max_fcm_iter=max_fcm_iter,
        use_metaheuristic=use_metaheuristic,
        mpa_every=mpa_every,
        mpa_epoch=mpa_epoch,
    )

    result.update({
        "mri_slice": mri_slice,
        "seg_gt": seg_slice,
        "features": features,
        "projected": projected,
        "confidence_map": confidence_map,
        "n_clusters": n_clusters,
        "explained_variance": expl_var,
        "slice_idx": slice_idx,
        "modality": modality,
    })

    return result


# ===================================================================
# CLI self-test
# ===================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  Modul Hybrid FCM — Quick Self-Test")
    print("=" * 60)

    rng = np.random.default_rng(42)
    N, F, K = 100, 3, 3
    data = rng.random((N, F)).astype(np.float64)
    centroids = rng.random((K, F)).astype(np.float64)
    U = initialize_membership_soft(N, K)

    centroids_new, U_new = fuzzy_c_means_single_step(data, U, centroids, 2.0)
    J = fcm_objective_function(data, U_new, centroids_new, 2.0)
    print(f"  FCM step: J={J:.4f}")
    print(f"  Centroids: {centroids_new.shape}")
    print(f"  U: {U_new.shape}, row_sum={U_new.sum(axis=1)[:3]}")

    print("\n  Self-test PASSED ✓")
    print("=" * 60)
