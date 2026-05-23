"""
Model registry – stores metadata and reference data per model.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ModelEntry:
    model_id: str
    task: str                          # "classification" | "regression"
    reference_data: Optional[pd.DataFrame] = None
    reference_embeddings: Optional[np.ndarray] = None
    expected_schema: Optional[Dict[str, str]] = None
    performance_thresholds: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ModelRegistry:
    def __init__(self):
        self._models: Dict[str, ModelEntry] = {}

    def register(
        self,
        model_id: str,
        task: str,
        reference_data: Optional[pd.DataFrame] = None,
        reference_embeddings: Optional[np.ndarray] = None,
        expected_schema: Optional[Dict[str, str]] = None,
        performance_thresholds: Optional[Dict[str, float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ModelEntry:
        entry = ModelEntry(
            model_id=model_id,
            task=task,
            reference_data=reference_data,
            reference_embeddings=reference_embeddings,
            expected_schema=expected_schema,
            performance_thresholds=performance_thresholds or {},
            metadata=metadata or {},
        )
        self._models[model_id] = entry
        logger.info(f"Registered model '{model_id}' (task={task})")
        return entry

    def get(self, model_id: str) -> ModelEntry:
        if model_id not in self._models:
            raise KeyError(f"Model '{model_id}' not found in registry")
        return self._models[model_id]

    def list_models(self):
        return list(self._models.keys())

    def update_reference(
        self,
        model_id: str,
        reference_data: pd.DataFrame,
        reference_embeddings: Optional[np.ndarray] = None,
    ) -> None:
        entry = self.get(model_id)
        entry.reference_data = reference_data
        if reference_embeddings is not None:
            entry.reference_embeddings = reference_embeddings
        logger.info(f"Reference updated for '{model_id}'")