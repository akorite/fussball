"""Model training and hyperparameter tuning.

Provides ``train()`` for LightGBM training with early stopping,
``tune()`` for Optuna-based hyperparameter search, and
``save_model()`` / ``load_model()`` for serialization.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from optuna.samplers import TPESampler
from sklearn.model_selection import TimeSeriesSplit

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_PARAMS: dict[str, Any] = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "verbosity": -1,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "lambda_l1": 0.1,
    "lambda_l2": 0.1,
    "seed": 42,
}

FEATURE_IMPORTANCE_AVG_GAIN: str = "gain"
ROI_BET_FRACTION: float = 0.05  # fraction of bankroll per bet (for reference)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class TrainedModel:
    """Wraps a trained LightGBM classifier with its parameters and metadata."""

    model: lgb.LGBMClassifier
    params: dict[str, Any]
    best_iteration: int | None = None


@dataclass
class TuningResult:
    """Result of an Optuna hyperparameter search."""

    study: optuna.Study
    best_params: dict[str, Any]
    best_score: float
    best_model: TrainedModel


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train(
    X: np.ndarray | Any,
    y: np.ndarray | Any,
    params: dict[str, Any] | None = None,
    eval_set: tuple[Any, Any] | None = None,
    num_boost_round: int = 1000,
    early_stopping_rounds: int = 50,
) -> TrainedModel:
    """Train a LightGBM classifier.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Training features.
    y : array-like, shape (n_samples,)
        Binary target labels.
    params : dict or None
        Overrides for ``DEFAULT_PARAMS``.
    eval_set : tuple (X_val, y_val) or None
        Optional held-out validation set for early stopping.
    num_boost_round : int
        Maximum number of boosting iterations.
    early_stopping_rounds : int
        Early stopping patience (used only when ``eval_set`` is provided).

    Returns
    -------
    TrainedModel
        Wrapper holding the fitted classifier, merged parameters, and best
        iteration number.

    Raises
    ------
    ValueError
        If fewer than 50 samples are provided, or if X / y have mismatched
        lengths.
    """
    _validate_inputs(X, y)

    # Ensure X is a DataFrame to avoid sklearn feature-name warnings
    X, _columns = _ensure_dataframe(X)

    merged: dict[str, Any] = {**DEFAULT_PARAMS}
    if params is not None:
        merged.update(params)

    clf = lgb.LGBMClassifier(**merged, n_estimators=num_boost_round)

    if eval_set is not None:
        X_val, y_val = eval_set
        X_val, _ = _ensure_dataframe(X_val, force_columns=_columns)
        clf.fit(
            X,
            y,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(early_stopping_rounds, first_metric_only=True)],
        )
        best_iter = clf.best_iteration_
    else:
        clf.fit(X, y)
        # Without early stopping, best_iteration_ may return 0 as a fallback.
        # Use the actual number of fitted trees instead.
        best_iter = (
            num_boost_round
            if (not hasattr(clf, "best_iteration_") or clf.best_iteration_ == 0)
            else clf.best_iteration_
        )

    return TrainedModel(model=clf, params=merged, best_iteration=best_iter)


# ---------------------------------------------------------------------------
# Hyperparameter tuning
# ---------------------------------------------------------------------------


def tune(
    X: np.ndarray | Any,
    y: np.ndarray | Any,
    n_trials: int = 20,
    seed: int = 42,
) -> TuningResult:
    """Run Optuna hyperparameter search with time-series cross-validation.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Training features.
    y : array-like, shape (n_samples,)
        Binary target labels.
    n_trials : int
        Number of Optuna trials (keep small for fast tests).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    TuningResult
        Dataclass with the completed study, best parameters, best score, and a
        model trained on the full dataset with the best parameters.
    """

    _validate_inputs(X, y)

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: optuna.Trial) -> float:
        trial_params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-3, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 10.0, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
        }

        merged: dict[str, Any] = {**DEFAULT_PARAMS, **trial_params}

        # Convert to numpy arrays for consistent positional indexing
        X_arr = np.asarray(X)
        y_arr = np.asarray(y).ravel()
        # Preserve column names from original DataFrame if available
        columns = getattr(X, "columns", [f"f{i}" for i in range(X_arr.shape[1])])

        tscv = TimeSeriesSplit(n_splits=3)
        log_losses: list[float] = []
        for train_idx, val_idx in tscv.split(X_arr):
            X_fold = pd.DataFrame(X_arr[train_idx], columns=columns)
            X_val_fold = pd.DataFrame(X_arr[val_idx], columns=columns)
            y_fold = y_arr[train_idx]
            y_val_fold = y_arr[val_idx]

            clf = lgb.LGBMClassifier(**merged, n_estimators=1000)
            clf.fit(
                X_fold,
                y_fold,
                eval_set=[(X_val_fold, y_val_fold)],
                callbacks=[lgb.early_stopping(10, first_metric_only=True)],
            )
            y_prob = clf.predict_proba(X_val_fold)[:, 1]
            from sklearn.metrics import log_loss

            log_losses.append(log_loss(y_val_fold, y_prob))

        return float(np.mean(log_losses))

    study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)

    # Train a final model on the full dataset with best params (no early stopping)
    best_params = study.best_params
    merged_best: dict[str, Any] = {**DEFAULT_PARAMS, **best_params}
    # Ensure X has feature names to avoid sklearn warnings
    X_final = pd.DataFrame(np.asarray(X)) if not hasattr(X, "columns") else X
    full_clf = lgb.LGBMClassifier(**merged_best, n_estimators=1000)
    full_clf.fit(X_final, y)

    return TuningResult(
        study=study,
        best_params=best_params,
        best_score=study.best_value,
        best_model=TrainedModel(
            model=full_clf,
            params=merged_best,
            best_iteration=full_clf.best_iteration_,
        ),
    )


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def save_model(model: TrainedModel, path: str | Path) -> None:
    """Serialize a ``TrainedModel`` with ``joblib``."""
    joblib.dump(model, path)


def load_model(path: str | Path) -> TrainedModel:
    """Deserialize a ``TrainedModel``."""
    return joblib.load(path)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _ensure_dataframe(X, force_columns: list[str] | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Convert X to a DataFrame with column names if it isn't already.

    Returns (X_as_df, column_names).
    """
    if hasattr(X, "columns"):
        columns = list(X.columns)
        return X, columns
    X_arr = np.asarray(X)
    if force_columns is not None:
        columns = force_columns
    else:
        columns = [f"f{i}" for i in range(X_arr.shape[1])]
    return pd.DataFrame(X_arr, columns=columns), columns


def _validate_inputs(X, y) -> None:
    """Validate training inputs."""
    n = X.shape[0] if hasattr(X, "shape") else len(X)
    m = y.shape[0] if hasattr(y, "shape") else len(y)

    if n != m:
        raise ValueError(f"X and y have mismatched lengths: X has {n} samples, y has {m}")
    if n < 50:
        raise ValueError(f"At least 50 samples are required for training, got {n}")
