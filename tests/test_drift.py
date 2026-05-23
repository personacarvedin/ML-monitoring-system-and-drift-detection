"""Tests for drift detection."""
import numpy as np
import pandas as pd
import pytest

from ml_monitor.drift.statistical import chi2_test, ks_test, psi
from ml_monitor.drift.embeddings import mmd_test
from ml_monitor.drift.detector import DriftDetector


# ------------------------------------------------------------------ #
# KS test
# ------------------------------------------------------------------ #
def test_ks_no_drift():
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, 500)
    prod = rng.normal(0, 1, 300)
    result = ks_test(ref, prod, p_threshold=0.05)
    assert not result.drift_detected


def test_ks_drift():
    rng = np.random.default_rng(1)
    ref = rng.normal(0, 1, 500)
    prod = rng.normal(5, 1, 300)    # large shift
    result = ks_test(ref, prod, p_threshold=0.05)
    assert result.drift_detected


# ------------------------------------------------------------------ #
# PSI
# ------------------------------------------------------------------ #
def test_psi_stable():
    rng = np.random.default_rng(2)
    ref = rng.normal(0, 1, 1000)
    prod = rng.normal(0, 1, 500)
    result = psi(ref, prod, psi_threshold=0.2)
    assert result.statistic < 0.2


def test_psi_drifted():
    rng = np.random.default_rng(3)
    ref = rng.normal(0, 1, 1000)
    prod = rng.normal(10, 1, 500)
    result = psi(ref, prod, psi_threshold=0.2)
    assert result.drift_detected


# ------------------------------------------------------------------ #
# Chi-square
# ------------------------------------------------------------------ #
def test_chi2_no_drift():
    rng = np.random.default_rng(4)
    cats = ["A", "B", "C"]
    ref = rng.choice(cats, 500, p=[0.5, 0.3, 0.2])
    prod = rng.choice(cats, 300, p=[0.5, 0.3, 0.2])
    result = chi2_test(ref, prod)
    assert not result.drift_detected


def test_chi2_drift():
    rng = np.random.default_rng(5)
    cats = ["A", "B", "C"]
    ref = rng.choice(cats, 500, p=[0.5, 0.3, 0.2])
    prod = rng.choice(cats, 300, p=[0.1, 0.1, 0.8])   # very different
    result = chi2_test(ref, prod)
    assert result.drift_detected


# ------------------------------------------------------------------ #
# MMD
# ------------------------------------------------------------------ #
def test_mmd_no_drift():
    rng = np.random.default_rng(6)
    ref = rng.normal(0, 1, (200, 4))
    prod = rng.normal(0, 1, (150, 4))
    result = mmd_test(ref, prod, n_permutations=100)
    assert not result.drift_detected


def test_mmd_drift():
    rng = np.random.default_rng(7)
    ref = rng.normal(0, 1, (200, 4))
    prod = rng.normal(5, 1, (150, 4))
    result = mmd_test(ref, prod, n_permutations=100)
    assert result.drift_detected


# ------------------------------------------------------------------ #
# DriftDetector (integration)
# ------------------------------------------------------------------ #
def test_detector_integration():
    cols = [f"f{i}" for i in range(5)]
    rng = np.random.default_rng(8)
    ref_df = pd.DataFrame(rng.normal(0, 1, (500, 5)), columns=cols)
    prod_df = pd.DataFrame(rng.normal(0, 1, (300, 5)), columns=cols)
    prod_df["f0"] += 10          # inject drift on f0

    config = {"drift": {"ks_threshold": 0.05, "psi_threshold": 0.2,
                         "chi2_threshold": 0.05, "min_samples": 10}}
    detector = DriftDetector(config)
    detector.set_reference(ref_df)
    results = detector.detect(prod_df)

    drifted_features = {
        r.feature for r in results["feature_drift"] if r.drift_detected
    }
    assert "f0" in drifted_features