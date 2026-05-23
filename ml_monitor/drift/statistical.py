"""
Statistical drift tests:
  - Kolmogorov-Smirnov   (continuous features)
  - Population Stability Index (continuous features)
  - Chi-Square           (categorical features)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class DriftResult:
    test_name: str
    feature: str
    statistic: float
    p_value: Optional[float]
    drift_detected: bool
    threshold: float
    details: dict


# ------------------------------------------------------------------ #
# Kolmogorov-Smirnov test
# ------------------------------------------------------------------ #

def ks_test(
    reference: np.ndarray,
    production: np.ndarray,
    feature_name: str = "feature",
    p_threshold: float = 0.05,
) -> DriftResult:
    """Two-sample KS test for continuous features."""
    ref = reference[~np.isnan(reference)]
    prod = production[~np.isnan(production)]

    if len(ref) < 2 or len(prod) < 2:
        logger.warning(f"KS test: insufficient samples for {feature_name}")
        return DriftResult(
            test_name="ks",
            feature=feature_name,
            statistic=0.0,
            p_value=None,
            drift_detected=False,
            threshold=p_threshold,
            details={"error": "insufficient_samples"},
        )

    stat, p_value = stats.ks_2samp(ref, prod)
    drift = p_value < p_threshold

    logger.debug(
        f"KS [{feature_name}] stat={stat:.4f} p={p_value:.4f} drift={drift}"
    )
    return DriftResult(
        test_name="ks",
        feature=feature_name,
        statistic=float(stat),
        p_value=float(p_value),
        drift_detected=drift,
        threshold=p_threshold,
        details={
            "ref_mean": float(np.mean(ref)),
            "prod_mean": float(np.mean(prod)),
            "ref_std": float(np.std(ref)),
            "prod_std": float(np.std(prod)),
            "ref_n": len(ref),
            "prod_n": len(prod),
        },
    )


# ------------------------------------------------------------------ #
# Population Stability Index
# ------------------------------------------------------------------ #

def psi(
    reference: np.ndarray,
    production: np.ndarray,
    feature_name: str = "feature",
    n_bins: int = 10,
    psi_threshold: float = 0.2,
) -> DriftResult:
    """
    PSI < 0.1   → no significant change
    PSI 0.1–0.2 → moderate change
    PSI > 0.2   → significant drift
    """
    ref = reference[~np.isnan(reference)]
    prod = production[~np.isnan(production)]

    eps = 1e-6
    breakpoints = np.percentile(ref, np.linspace(0, 100, n_bins + 1))
    breakpoints = np.unique(breakpoints)

    ref_counts = np.histogram(ref, bins=breakpoints)[0].astype(float)
    prod_counts = np.histogram(prod, bins=breakpoints)[0].astype(float)

    ref_pct = ref_counts / len(ref) + eps
    prod_pct = prod_counts / len(prod) + eps

    psi_value = float(np.sum((prod_pct - ref_pct) * np.log(prod_pct / ref_pct)))
    drift = psi_value > psi_threshold

    logger.debug(f"PSI [{feature_name}] psi={psi_value:.4f} drift={drift}")
    return DriftResult(
        test_name="psi",
        feature=feature_name,
        statistic=psi_value,
        p_value=None,
        drift_detected=drift,
        threshold=psi_threshold,
        details={
            "n_bins": n_bins,
            "interpretation": (
                "stable" if psi_value < 0.1
                else "moderate" if psi_value < 0.2
                else "significant"
            ),
        },
    )


# ------------------------------------------------------------------ #
# Chi-Square test (categorical)
# ------------------------------------------------------------------ #

def chi2_test(
    reference: np.ndarray,
    production: np.ndarray,
    feature_name: str = "feature",
    p_threshold: float = 0.05,
) -> DriftResult:
    """Chi-square goodness-of-fit test for categorical features."""
    # Align categories
    all_cats = np.union1d(np.unique(reference), np.unique(production))

    ref_counts = np.array(
        [np.sum(reference == c) for c in all_cats], dtype=float
    )
    prod_counts = np.array(
        [np.sum(production == c) for c in all_cats], dtype=float
    )

    # Scale reference to same total as production
    if ref_counts.sum() == 0:
        return DriftResult(
            test_name="chi2",
            feature=feature_name,
            statistic=0.0,
            p_value=None,
            drift_detected=False,
            threshold=p_threshold,
            details={"error": "empty_reference"},
        )

    expected = ref_counts / ref_counts.sum() * prod_counts.sum()
    # Drop zero-expected bins
    mask = expected > 0
    if mask.sum() < 2:
        return DriftResult(
            test_name="chi2",
            feature=feature_name,
            statistic=0.0,
            p_value=None,
            drift_detected=False,
            threshold=p_threshold,
            details={"error": "insufficient_categories"},
        )

    stat, p_value = stats.chisquare(prod_counts[mask], f_exp=expected[mask])
    drift = p_value < p_threshold

    logger.debug(f"Chi2 [{feature_name}] stat={stat:.4f} p={p_value:.4f} drift={drift}")
    return DriftResult(
        test_name="chi2",
        feature=feature_name,
        statistic=float(stat),
        p_value=float(p_value),
        drift_detected=drift,
        threshold=p_threshold,
        details={
            "categories": all_cats.tolist(),
            "ref_distribution": (ref_counts / ref_counts.sum()).tolist(),
            "prod_distribution": (prod_counts / prod_counts.sum()).tolist(),
        },
    )