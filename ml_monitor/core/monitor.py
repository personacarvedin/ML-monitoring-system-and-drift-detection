"""
MLMonitor – main orchestrator that wires together drift detection,
metrics, data quality checks, alerting, and storage.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

from ..alerts.alerter import Alerter
from ..drift.detector import DriftDetector
from ..metrics.data_quality import full_quality_report
from ..metrics.performance import compute_metrics
from ..storage.store import MonitorStore
from .registry import ModelRegistry

logger = logging.getLogger(__name__)


def _load_config(path: str = "config/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


class MLMonitor:
    """
    Central monitoring system.

    Quick start
    -----------
    monitor = MLMonitor()
    monitor.register_model("my_model", task="classification",
                           reference_data=ref_df)
    report = monitor.run(
        model_id="my_model",
        production_data=prod_df,
        y_true=y_true,
        y_pred=y_pred,
    )
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        try:
            self.config = _load_config(config_path)
        except FileNotFoundError:
            logger.warning(f"Config not found at {config_path}; using defaults")
            self.config = {}

        self.store = MonitorStore(
            db_path=self.config.get("storage", {}).get("db_path", "ml_monitor.db")
        )
        self.registry = ModelRegistry()
        self.alerter = Alerter(self.config, self.store)
        self._detectors: Dict[str, DriftDetector] = {}

    # ------------------------------------------------------------------ #
    # Registration
    # ------------------------------------------------------------------ #
    def register_model(
        self,
        model_id: str,
        task: str,
        reference_data: Optional[pd.DataFrame] = None,
        reference_embeddings: Optional[np.ndarray] = None,
        expected_schema: Optional[Dict[str, str]] = None,
        performance_thresholds: Optional[Dict[str, float]] = None,
        metadata: Optional[Dict] = None,
    ) -> None:
        self.registry.register(
            model_id=model_id,
            task=task,
            reference_data=reference_data,
            reference_embeddings=reference_embeddings,
            expected_schema=expected_schema,
            performance_thresholds=performance_thresholds,
            metadata=metadata,
        )

        # Build + prime drift detector
        detector = DriftDetector(self.config)
        if reference_data is not None:
            detector.set_reference(reference_data)
        self._detectors[model_id] = detector

    # ------------------------------------------------------------------ #
    # Main run
    # ------------------------------------------------------------------ #
    def run(
        self,
        model_id: str,
        production_data: pd.DataFrame,
        y_true: Optional[np.ndarray] = None,
        y_pred: Optional[np.ndarray] = None,
        y_proba: Optional[np.ndarray] = None,
        production_embeddings: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Run a full monitoring check and return a report dict.
        """
        entry = self.registry.get(model_id)
        report: Dict[str, Any] = {"model_id": model_id, "sections": {}}

        # 1. Data quality
        quality = full_quality_report(
            production_data,
            reference=entry.reference_data,
            expected_schema=entry.expected_schema,
        )
        report["sections"]["data_quality"] = quality
        if not quality["overall_passed"]:
            issues = quality.get("schema", {}).get("issues", [])
            issues += list(quality.get("high_null_columns", {}).keys())
            issues += list(quality.get("high_outlier_columns", {}).keys())
            self.alerter.fire_quality_alert(model_id, issues)

        # 2. Drift detection
        detector = self._detectors.get(model_id)
        drift_results = {}
        if detector and entry.reference_data is not None:
            raw = detector.detect(
                production_data,
                embeddings_ref=entry.reference_embeddings,
                embeddings_prod=production_embeddings,
            )
            drift_results = raw
            summary = DriftDetector.summarise(raw)
            report["sections"]["drift"] = summary.to_dict(orient="records")

            # Log and alert drifted features
            for res in raw.get("feature_drift", []) + raw.get("embedding_drift", []):
                self.store.log_drift(
                    model_id=model_id,
                    feature=res.feature,
                    test_name=res.test_name,
                    statistic=res.statistic,
                    p_value=res.p_value,
                    drift_detected=res.drift_detected,
                )
                if res.drift_detected:
                    self.alerter.fire_drift_alert(
                        model_id, res.feature, res.test_name, res.statistic
                    )

        # 3. Performance metrics
        if y_true is not None and y_pred is not None:
            perf = compute_metrics(entry.task, y_true, y_pred, y_proba)
            report["sections"]["performance"] = perf

            for metric, value in perf.items():
                self.store.log_metric(model_id, metric, value)

            # Check thresholds
            for metric, threshold in entry.performance_thresholds.items():
                value = perf.get(metric)
                if value is not None and value < threshold:
                    self.alerter.fire_performance_alert(
                        model_id, metric, value, threshold
                    )

        logger.info(f"Monitoring run complete for '{model_id}'")
        return report

    # ------------------------------------------------------------------ #
    # Convenience getters
    # ------------------------------------------------------------------ #
    def get_drift_history(self, model_id: str, days: int = 7) -> List[Dict]:
        return self.store.get_drift_history(model_id, days)

    def get_metric_history(
        self, model_id: str, metric: str, days: int = 7
    ) -> List[Dict]:
        return self.store.get_metric_history(model_id, metric, days)

    def get_open_alerts(self, model_id: Optional[str] = None) -> List[Dict]:
        return self.store.get_open_alerts(model_id)