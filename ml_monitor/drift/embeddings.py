"""
Maximum Mean Discrepancy (MMD) for embedding / high-dimensional drift detection.
Uses an RBF kernel. Permutation test gives an empirical p-value.
"""
from __future__ import annotations

import logging

import numpy as np

from .statistical import DriftResult

logger = logging.getLogger(__name__)


def _rbf_kernel(X: np.ndarray, Y: np.ndarray, sigma: float) -> np.ndarray:
    """Compute RBF kernel between rows of X and Y."""
    XX = np.sum(X ** 2, axis=1, keepdims=True)
    YY = np.sum(Y ** 2, axis=1, keepdims=True)
    dist = XX + YY.T - 2 * X @ Y.T
    return np.exp(-dist / (2 * sigma ** 2))


def _mmd_statistic(X: np.ndarray, Y: np.ndarray, sigma: float) -> float:
    """Unbiased MMD² estimator."""
    n, m = len(X), len(Y)
    Kxx = _rbf_kernel(X, X, sigma)
    Kyy = _rbf_kernel(Y, Y, sigma)
    Kxy = _rbf_kernel(X, Y, sigma)

    np.fill_diagonal(Kxx, 0)
    np.fill_diagonal(Kyy, 0)

    mmd2 = (
        Kxx.sum() / (n * (n - 1))
        + Kyy.sum() / (m * (m - 1))
        - 2 * Kxy.sum() / (n * m)
    )
    return float(mmd2)


def mmd_test(
    reference: np.ndarray,
    production: np.ndarray,
    feature_name: str = "embeddings",
    mmd_threshold: float = 0.05,
    n_permutations: int = 200,
    sigma: float | None = None,
) -> DriftResult:
    """
    MMD-based drift test for continuous, possibly high-dimensional arrays.

    Parameters
    ----------
    reference   : (N, D) reference embeddings
    production  : (M, D) production embeddings
    mmd_threshold: threshold on MMD² statistic (used when p-value unavailable)
    n_permutations: permutations for empirical p-value (0 = skip)
    sigma       : RBF bandwidth; if None, uses median heuristic
    """
    ref = np.atleast_2d(reference)
    prod = np.atleast_2d(production)

    if ref.ndim == 1:
        ref = ref.reshape(-1, 1)
    if prod.ndim == 1:
        prod = prod.reshape(-1, 1)

    if len(ref) < 10 or len(prod) < 10:
        return DriftResult(
            test_name="mmd",
            feature=feature_name,
            statistic=0.0,
            p_value=None,
            drift_detected=False,
            threshold=mmd_threshold,
            details={"error": "insufficient_samples"},
        )

    # Median heuristic for sigma
    if sigma is None:
        combined = np.vstack([ref, prod])
        pairwise = np.sum((combined[:, None] - combined[None, :]) ** 2, axis=-1)
        sigma = float(np.sqrt(np.median(pairwise[pairwise > 0]) / 2))
        sigma = max(sigma, 1e-6)

    observed_mmd = _mmd_statistic(ref, prod, sigma)

    p_value = None
    if n_permutations > 0:
        combined = np.vstack([ref, prod])
        n = len(ref)
        null_stats = []
        rng = np.random.default_rng(42)
        for _ in range(n_permutations):
            idx = rng.permutation(len(combined))
            null_stats.append(
                _mmd_statistic(combined[idx[:n]], combined[idx[n:]], sigma)
            )
        p_value = float(np.mean(np.array(null_stats) >= observed_mmd))

    drift = (p_value < 0.05) if p_value is not None else (observed_mmd > mmd_threshold)

    logger.debug(
        f"MMD [{feature_name}] mmd²={observed_mmd:.6f} p={p_value} drift={drift}"
    )
    return DriftResult(
        test_name="mmd",
        feature=feature_name,
        statistic=observed_mmd,
        p_value=p_value,
        drift_detected=drift,
        threshold=mmd_threshold,
        details={
            "sigma": sigma,
            "n_permutations": n_permutations,
            "ref_shape": list(ref.shape),
            "prod_shape": list(prod.shape),
        },
    )