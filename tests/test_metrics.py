"""Tests for performance and data quality metrics."""
import numpy as np
import pandas as pd
import pytest

from ml_monitor.metrics.performance import compute_metrics
from ml_monitor.metrics.data_quality import (
    null_rate_report, outlier_report, schema_check, full_quality_report
)


def test_classification_metrics():
    y_true = np.array([0, 1, 1, 0, 1, 0])
    y_pred = np.array([0, 1, 0, 0, 1, 1])
    m = compute_metrics("classification", y_true, y_pred)
    assert "accuracy" in m and "f1" in m
    assert 0.0 <= m["accuracy"] <= 1.0


def test_regression_metrics():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.1, 1.9, 3.2, 3.8])
    m = compute_metrics("regression", y_true, y_pred)
    assert "rmse" in m and "r2" in m
    assert m["rmse"] >= 0


def test_null_rate():
    df = pd.DataFrame({"a": [1, None, 3], "b": [None, None, 3]})
    rates = null_rate_report(df)
    assert abs(rates["a"] - 1/3) < 1e-9
    assert abs(rates["b"] - 2/3) < 1e-9


def test_schema_check_valid():
    df = pd.DataFrame({"age": [25.0, 30.0], "cat": ["A", "B"]})
    result = schema_check(df, {"age": "float", "cat": "object"})
    assert result["valid"]


def test_schema_check_missing_col():
    df = pd.DataFrame({"age": [25.0]})
    result = schema_check(df, {"age": "float", "income": "float"})
    assert not result["valid"]
    assert "income" in result["missing_columns"]


def test_full_quality_report_passes():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(0, 1, (100, 3)), columns=["a", "b", "c"])
    report = full_quality_report(df)
    assert report["null_check_passed"]
    assert report["overall_passed"]