"""
Data quality checks: null rates, outliers, schema validation.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def null_rate_report(df: pd.DataFrame) -> Dict[str, float]:
    """Return null rate (0-1) per column."""
    return {col: float(df[col].isna().mean()) for col in df.columns}


def outlier_report(
    df: pd.DataFrame,
    reference: Optional[pd.DataFrame] = None,
    z_threshold: float = 3.0,
) -> Dict[str, Dict]:
    """
    Flag outliers using z-score.  If reference is provided, compute
    mean/std from reference; otherwise use the data itself.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    report: Dict[str, Dict] = {}

    for col in numeric_cols:
        src = reference[col].dropna() if reference is not None else df[col].dropna()
        mu, sigma = float(src.mean()), float(src.std())
        if sigma == 0:
            continue
        z = np.abs((df[col].dropna() - mu) / sigma)
        outlier_rate = float((z > z_threshold).mean())
        report[col] = {
            "outlier_rate": outlier_rate,
            "z_threshold": z_threshold,
            "ref_mean": mu,
            "ref_std": sigma,
        }
    return report


def schema_check(
    df: pd.DataFrame,
    expected_schema: Dict[str, str],
) -> Dict[str, Any]:
    """
    Validate that df matches expected_schema.

    Parameters
    ----------
    expected_schema : {"col_name": "dtype_kind"}, e.g. {"age": "float", "label": "object"}
    """
    issues: List[str] = []
    missing_cols = [c for c in expected_schema if c not in df.columns]
    extra_cols = [c for c in df.columns if c not in expected_schema]

    for col in missing_cols:
        issues.append(f"Missing column: {col}")

    for col in extra_cols:
        issues.append(f"Unexpected column: {col}")

    for col, expected_dtype in expected_schema.items():
        if col not in df.columns:
            continue
        actual = str(df[col].dtype)
        if expected_dtype not in actual:
            issues.append(
                f"Type mismatch for '{col}': expected '{expected_dtype}', got '{actual}'"
            )

    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "missing_columns": missing_cols,
        "extra_columns": extra_cols,
    }


def full_quality_report(
    df: pd.DataFrame,
    reference: Optional[pd.DataFrame] = None,
    expected_schema: Optional[Dict[str, str]] = None,
    null_threshold: float = 0.1,
    outlier_threshold: float = 0.05,
    z_threshold: float = 3.0,
) -> Dict[str, Any]:
    """Aggregate quality report with pass/fail flags."""
    report: Dict[str, Any] = {}

    # Null rates
    null_rates = null_rate_report(df)
    report["null_rates"] = null_rates
    high_null_cols = {c: v for c, v in null_rates.items() if v > null_threshold}
    report["high_null_columns"] = high_null_cols
    report["null_check_passed"] = len(high_null_cols) == 0

    # Outliers
    outliers = outlier_report(df, reference, z_threshold)
    report["outliers"] = outliers
    high_outlier_cols = {
        c: v for c, v in outliers.items() if v["outlier_rate"] > outlier_threshold
    }
    report["high_outlier_columns"] = high_outlier_cols
    report["outlier_check_passed"] = len(high_outlier_cols) == 0

    # Schema
    if expected_schema:
        schema_result = schema_check(df, expected_schema)
        report["schema"] = schema_result
        report["schema_check_passed"] = schema_result["valid"]
    else:
        report["schema_check_passed"] = True

    report["overall_passed"] = (
        report["null_check_passed"]
        and report["outlier_check_passed"]
        and report["schema_check_passed"]
    )
    return report