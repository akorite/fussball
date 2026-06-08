"""Model layer: training, tuning, evaluation, and serialization."""

from __future__ import annotations

from fussball.models.evaluator import EvaluationResult, ROIResults, evaluate
from fussball.models.trainer import (
    DEFAULT_PARAMS,
    FEATURE_IMPORTANCE_AVG_GAIN,
    ROI_BET_FRACTION,
    TrainedModel,
    TuningResult,
    load_model,
    save_model,
    train,
    tune,
)

__all__ = [
    "DEFAULT_PARAMS",
    "FEATURE_IMPORTANCE_AVG_GAIN",
    "ROI_BET_FRACTION",
    "EvaluationResult",
    "ROIResults",
    "TrainedModel",
    "TuningResult",
    "evaluate",
    "load_model",
    "save_model",
    "train",
    "tune",
]
