"""Model evaluation metrics and ROI simulation.

Provides ``evaluate()`` that computes log loss, Brier score, accuracy,
calibration curve, and optionally an ROI simulation against market odds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, log_loss

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ROIResults:
    """Results of a betting ROI simulation."""

    edge_threshold: float
    total_bets: int
    win_rate: float
    profit_units: float  # net profit assuming 1 unit stake per bet
    roi_percent: float  # profit / total_staked * 100


@dataclass
class EvaluationResult:
    """Comprehensive model evaluation metrics."""

    log_loss: float
    brier_score: float
    accuracy_at_0_5: float
    calibration_curve: tuple[np.ndarray, np.ndarray]
    n_samples: int
    roi_results: ROIResults | None = None


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate(
    y_true: np.ndarray | Any,
    y_prob: np.ndarray | Any,
    market_over_odds: np.ndarray | Any | None = None,
    edge_threshold: float = 0.05,
) -> EvaluationResult:
    """Compute evaluation metrics for binary classifier predictions.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground-truth binary labels (0 or 1).
    y_prob : array-like of shape (n_samples,)
        Predicted probability of the positive class (over 2.5 goals).
    market_over_odds : array-like of shape (n_samples,) or None
        Market decimal odds for over 2.5. If provided, an ROI simulation is
        also computed.
    edge_threshold : float
        Minimum edge (model_prob - market_implied_prob) required to place a
        bet.

    Returns
    -------
    EvaluationResult
        Aggregated metrics and optional ROI results.

    Raises
    ------
    ValueError
        If ``y_true`` and ``y_prob`` have mismatched lengths, or if values
        are outside valid ranges.
    """
    y_true_arr = np.asarray(y_true, dtype=float).ravel()
    y_prob_arr = np.asarray(y_prob, dtype=float).ravel()

    if y_true_arr.shape != y_prob_arr.shape:
        raise ValueError(
            f"y_true and y_prob have mismatched shapes: {y_true_arr.shape} vs {y_prob_arr.shape}"
        )

    if not np.all(np.isin(y_true_arr, [0, 1])):
        raise ValueError("y_true must contain only values 0 and 1")

    if np.any((y_prob_arr < 0) | (y_prob_arr > 1)):
        raise ValueError("y_prob must contain values in [0, 1]")

    n = len(y_true_arr)

    # Log loss
    ll = float(log_loss(y_true_arr, y_prob_arr))

    # Brier score
    bs = float(brier_score_loss(y_true_arr, y_prob_arr))

    # Accuracy at threshold 0.5
    preds = (y_prob_arr >= 0.5).astype(int)
    acc = float(np.mean(preds == y_true_arr))

    # Calibration curve (10 bins, quantile strategy)
    prob_true, prob_pred = calibration_curve(y_true_arr, y_prob_arr, n_bins=10, strategy="quantile")

    # Optional ROI simulation
    roi_results: ROIResults | None = None
    if market_over_odds is not None:
        market_arr = np.asarray(market_over_odds, dtype=float).ravel()
        if market_arr.shape != y_true_arr.shape:
            raise ValueError(
                f"market_over_odds has mismatched shape: {market_arr.shape} vs {y_true_arr.shape}"
            )
        roi_results = _simulate_roi(y_true_arr, y_prob_arr, market_arr, edge_threshold)

    return EvaluationResult(
        log_loss=ll,
        brier_score=bs,
        accuracy_at_0_5=acc,
        calibration_curve=(prob_true, prob_pred),
        n_samples=n,
        roi_results=roi_results,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _simulate_roi(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    market_odds: np.ndarray,
    edge_threshold: float,
) -> ROIResults:
    """Simulate betting ROI given model predictions and market odds.

    Places a 1-unit bet on ``over 2.5`` whenever the model's probability
    exceeds the market-implied probability by at least ``edge_threshold``.
    """
    market_prob = 1.0 / market_odds
    edge = y_prob - market_prob
    bet_mask = edge > edge_threshold

    total_bets = int(bet_mask.sum())
    if total_bets == 0:
        return ROIResults(
            edge_threshold=edge_threshold,
            total_bets=0,
            win_rate=0.0,
            profit_units=0.0,
            roi_percent=0.0,
        )

    wins = y_true[bet_mask]  # 1 if over 2.5 hits
    odds_bet = market_odds[bet_mask]

    # Per-bet profit: if win, profit = odds - 1 (since 1 unit stake is returned + profit)
    # If loss, profit = -1
    profits = np.where(wins == 1, odds_bet - 1.0, -1.0)
    total_profit = float(profits.sum())
    total_staked = float(total_bets)
    win_rate = float(wins.mean())

    return ROIResults(
        edge_threshold=edge_threshold,
        total_bets=total_bets,
        win_rate=win_rate,
        profit_units=total_profit,
        roi_percent=(total_profit / total_staked) * 100.0 if total_staked > 0 else 0.0,
    )
