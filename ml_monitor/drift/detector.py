"""
DriftDetector – orchestrates all drift tests across a dataset.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .embeddings import mmd_test
from .statistical import DriftResult, chi2_test, ks_test, psi

logger = logging.getLogger(__name__)

# Features with ≤ this many unique values are treated as categorical
CATEGORICAL_CARDINALITY_THRESHOLD = 20


class DriftDetector:
    """
    Run drift detection between a reference dataset and a production window.

    Usage
    -----
    detector = DriftDetector(config)
    detector.set_reference(reference_df)
    results = detector.detect(production_df)
    """

    def __init__(self, config: dict):
        drift_cfg = config.get("drift", {})
        self.ks_threshold = drift_cfg.get("ks_threshold", 0.05)
        self.psi_threshold = drift_cfg.get("psi_threshold", 0.2)
        self.chi2_threshold = drift_cfg.get("chi2_threshold", 0.05)
        self.mmd_threshold = drift_cfg.get("mmd_threshold", 0.05)
        self.min_samples = drift_cfg.get("min_samples", 50)

        self._reference: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------ #
    # Reference
    # ------------------------------------------------------------------ #
    def set_reference(self, df: pd.DataFrame) -> None:
        """Store the reference (baseline) dataset."""
        self._reference = df.copy()
        logger.info(f"Reference set: {df.shape[0]} rows, {df.shape[1]} cols")

    # ------------------------------------------------------------------ #
    # Main entry point
    # ------------------------------------------------------------------ #
    def detect(
        self,
        production: pd.DataFrame,
        embeddings_ref: Optional[np.ndarray] = None,
        embeddings_prod: Optional[np.ndarray] = None,
    ) -> Dict[str, List[DriftResult]]:
        """
        Run all applicable drift tests.

        Returns
        -------
        {
            "feature_drift": [...],
            "embedding_drift": [...],
        }
        """
        if self._reference is None:
            raise RuntimeError("Call set_reference() before detect()")

        if len(production) < self.min_samples:
            logger.warning(
                f"Only {len(production)} production samples; "
                f"need ≥{self.min_samples} for reliable tests. Skipping."
            )
            return {"feature_drift": [], "embedding_drift": []}

        results: Dict[str, List[DriftResult]] = {
            "feature_drift": [],
            "embedding_drift": [],
        }

        # Run per-feature tests
        common_cols = [
            c for c in self._reference.columns if c in production.columns
        ]
        for col in common_cols:
            col_results = self._test_feature(col, production)
            results["feature_drift"].extend(col_results)

        # Embedding / MMD drift
        if embeddings_ref is not None and embeddings_prod is not None:
            emb_result = mmd_test(
                embeddings_ref,
                embeddings_prod,
                feature_name="embeddings",
                mmd_threshold=self.mmd_threshold,
            )
            results["embedding_drift"].append(emb_result)

        n_drift = sum(
            1
            for r in results["feature_drift"] + results["embedding_drift"]
            if r.drift_detected
        )
        logger.info(
            f"Drift detection complete: {n_drift} / "
            f"{len(results['feature_drift']) + len(results['embedding_drift'])} "
            "tests flagged drift"
        )
        return results

    # ------------------------------------------------------------------ #
    # Per-feature dispatch
    # ------------------------------------------------------------------ #
    def _test_feature(
        self, col: str, production: pd.DataFrame
    ) -> List[DriftResult]:
        ref_series = self._reference[col].dropna()
        prod_series = production[col].dropna()

        if len(ref_series) == 0 or len(prod_series) == 0:
            return []

        is_categorical = (
            ref_series.dtype == "object"
            or ref_series.dtype.name == "category"
            or ref_series.nunique() <= CATEGORICAL_CARDINALITY_THRESHOLD
        )

        if is_categorical:
            return [
                chi2_test(
                    ref_series.astype(str).values,
                    prod_series.astype(str).values,
                    feature_name=col,
                    p_threshold=self.chi2_threshold,
                )
            ]
        else:
            ref_arr = ref_series.astype(float).values
            prod_arr = prod_series.astype(float).values
            return [
                ks_test(ref_arr, prod_arr, col, self.ks_threshold),
                psi(ref_arr, prod_arr, col, psi_threshold=self.psi_threshold),
            ]

    # ------------------------------------------------------------------ #
    # Summary helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def summarise(results: Dict[str, List[DriftResult]]) -> pd.DataFrame:
        rows = []
        for category, res_list in results.items():
            for r in res_list:
                rows.append(
                    {
                        "category": category,
                        "feature": r.feature,
                        "test": r.test_name,
                        "statistic": round(r.statistic, 6),
                        "p_value": round(r.p_value, 4) if r.p_value else None,
                        "drift": r.drift_detected,
                    }
                )
        return pd.DataFrame(rows)