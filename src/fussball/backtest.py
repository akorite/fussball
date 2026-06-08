"""Backtest pipeline: time-series cross-validation across leagues and seasons.

The integration pattern is:
    records → build_feature_matrix → select_xy → train → predict_proba → evaluate
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.calibration import calibration_curve

from fussball.data.schema import MatchRecord
from fussball.features import build_feature_matrix
from fussball.models import evaluate, train
from fussball.models.evaluator import EvaluationResult, ROIResults

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration and result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class BacktestConfig:
    """Configuration for the backtest pipeline.

    Attributes
    ----------
    min_train_seasons:
        Minimum number of seasons of history required before testing a fold.
    edge_thresholds:
        Edge thresholds (model_prob - market_implied_prob) to evaluate for ROI curves.
    default_edge_threshold:
        The threshold used for the primary evaluation on each fold.
    min_samples_per_fold:
        Minimum number of test samples required for a fold to be included.
    output_dir:
        Directory where artifacts (JSON, PNG) are written.
    save_artifacts:
        If True, write JSON and PNG artifacts to ``output_dir``.
    tune:
        If True, run Optuna hyperparameter tuning before training.
    seed:
        Random seed for reproducibility of any internal shuffling.
    """

    min_train_seasons: int = 1
    edge_thresholds: tuple[float, ...] = (0.0, 0.02, 0.05, 0.10)
    default_edge_threshold: float = 0.05
    min_samples_per_fold: int = 100
    output_dir: Path = Path("data")
    save_artifacts: bool = True
    tune: bool = False
    seed: int = 42


@dataclass
class FoldResult:
    """Results for a single train/test fold (one league, one test season)."""

    league: str
    train_seasons: list[str]
    test_season: str
    n_train: int
    n_test: int
    evaluation: EvaluationResult
    roi_by_threshold: dict[float, ROIResults] = field(default_factory=dict)


@dataclass
class BacktestResult:
    """Aggregated backtest results across all leagues and folds."""

    folds: list[FoldResult] = field(default_factory=list)
    aggregate_log_loss: float = float("nan")
    aggregate_brier: float = float("nan")
    aggregate_accuracy: float = float("nan")
    per_league_log_loss: dict[str, float] = field(default_factory=dict)
    per_league_roi: dict[str, float] = field(default_factory=dict)
    aggregate_roi_curve: list[tuple[float, float]] = field(default_factory=list)
    calibration_data: tuple[np.ndarray, np.ndarray] = field(
        default_factory=lambda: (np.array([], dtype=float), np.array([], dtype=float))
    )
    config: BacktestConfig = field(default_factory=BacktestConfig)
    n_folds_skipped: int = 0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_backtest(
    records: list[MatchRecord],
    config: BacktestConfig | None = None,
) -> BacktestResult:
    """Run an end-to-end time-series backtest across all leagues and seasons.

    For each league with ``≥ min_train_seasons + 1`` seasons, walk forward
    season by season: train on seasons ``1..k``, test on season ``k+1``.

    Parameters
    ----------
    records:
        All match records to use in the backtest.
    config:
        Configuration overrides.  Uses defaults when ``None``.

    Returns
    -------
    BacktestResult
        Aggregated metrics, per-fold details, and calibration / ROI data.
    """
    if config is None:
        config = BacktestConfig()

    # -- Group records by (league, season) ---------------------------------
    league_data: dict[str, dict[str, list[MatchRecord]]] = defaultdict(lambda: defaultdict(list))
    for r in records:
        league_data[r.league][r.season].append(r)

    fold_results: list[FoldResult] = []
    all_y_true: list[np.ndarray] = []
    all_y_prob: list[np.ndarray] = []
    n_folds_skipped = 0

    # -- Walk forward per league --------------------------------------------
    for league, season_dict in sorted(league_data.items()):
        sorted_seasons = sorted(season_dict.keys())
        n_seasons = len(sorted_seasons)
        min_required = config.min_train_seasons + 1

        if n_seasons < min_required:
            logger.warning(
                "Skipping league %s: %d seasons available, need at least %d",
                league,
                n_seasons,
                min_required,
            )
            n_folds_skipped += 1
            continue

        for k in range(config.min_train_seasons, n_seasons):
            train_seasons = sorted_seasons[:k]
            test_season = sorted_seasons[k]

            train_records: list[MatchRecord] = []
            for s in train_seasons:
                train_records.extend(season_dict[s])
            test_records = list(season_dict[test_season])

            n_train = len(train_records)
            n_test = len(test_records)

            # Skip fold if test set is too small
            if n_test < config.min_samples_per_fold:
                logger.warning(
                    "Skipping fold %s/%s: test set has %d samples (min %d)",
                    league,
                    test_season,
                    n_test,
                    config.min_samples_per_fold,
                )
                n_folds_skipped += 1
                continue

            logger.info(
                "Fold %s/%s: train on %d seasons (%d samples), test on %s (%d samples)",
                league,
                test_season,
                len(train_seasons),
                n_train,
                test_season,
                n_test,
            )

            # -- Feature engineering (combined train+test for correct rolling features)
            all_records = train_records + test_records
            try:
                matrix = build_feature_matrix(all_records)
            except Exception as exc:
                logger.warning(
                    "Skipping fold %s/%s: feature engineering failed: %s",
                    league,
                    test_season,
                    exc,
                )
                n_folds_skipped += 1
                continue

            train_ids = {r.match_id for r in train_records}
            test_ids = {r.match_id for r in test_records}

            # Split features
            id_cols = matrix.identifier_columns
            train_mask = matrix.X.index.isin(train_ids)
            test_mask = matrix.X.index.isin(test_ids)

            X_train = matrix.X.loc[train_mask]
            y_train = matrix.y.loc[train_mask]
            X_test = matrix.X.loc[test_mask]
            y_test = matrix.y.loc[test_mask]

            # Drop identifier columns
            cols_to_drop = [c for c in id_cols if c in X_train.columns]
            X_train_feat = X_train.drop(columns=cols_to_drop)
            X_test_feat = X_test.drop(columns=cols_to_drop)

            # Skip if training data is insufficient (train() requires >= 50)
            if len(X_train_feat) < 50:
                logger.warning(
                    "Skipping fold %s/%s: only %d training samples (need >= 50)",
                    league,
                    test_season,
                    len(X_train_feat),
                )
                n_folds_skipped += 1
                continue

            # -- Train model --------------------------------------------------
            try:
                trained = train(X_train_feat, y_train)
            except Exception as exc:
                logger.warning(
                    "Skipping fold %s/%s: training failed: %s",
                    league,
                    test_season,
                    exc,
                )
                n_folds_skipped += 1
                continue

            # -- Predict on test set ------------------------------------------
            p_test = trained.model.predict_proba(X_test_feat)[:, 1]

            # -- Build market odds aligned with test predictions --------------
            odds_map: dict[str, float | None] = {
                r.match_id: r.closing_odds_over_2_5 for r in test_records
            }
            match_ids_test = X_test_feat.index.tolist()
            odds_test = np.array([odds_map.get(mid) for mid in match_ids_test], dtype=float)

            # -- Evaluate at default threshold --------------------------------
            try:
                eval_result = evaluate(
                    y_test.values if hasattr(y_test, "values") else y_test,
                    p_test,
                    market_over_odds=odds_test,
                    edge_threshold=config.default_edge_threshold,
                )
            except Exception as exc:
                logger.warning(
                    "Skipping fold %s/%s: evaluation failed: %s",
                    league,
                    test_season,
                    exc,
                )
                n_folds_skipped += 1
                continue

            # -- Re-evaluate at each edge threshold for ROI curve -------------
            roi_by_threshold: dict[float, ROIResults] = {}
            for threshold in config.edge_thresholds:
                try:
                    r_eval = evaluate(
                        y_test.values if hasattr(y_test, "values") else y_test,
                        p_test,
                        market_over_odds=odds_test,
                        edge_threshold=threshold,
                    )
                    r_val = r_eval.roi_results
                    if r_val is None:
                        r_val = ROIResults(
                            edge_threshold=threshold,
                            total_bets=0,
                            win_rate=0.0,
                            profit_units=0.0,
                            roi_percent=0.0,
                        )
                    roi_by_threshold[threshold] = r_val
                except Exception:
                    roi_by_threshold[threshold] = ROIResults(
                        edge_threshold=threshold,
                        total_bets=0,
                        win_rate=0.0,
                        profit_units=0.0,
                        roi_percent=0.0,
                    )

            fold_result = FoldResult(
                league=league,
                train_seasons=list(train_seasons),
                test_season=test_season,
                n_train=n_train,
                n_test=n_test,
                evaluation=eval_result,
                roi_by_threshold=roi_by_threshold,
            )
            fold_results.append(fold_result)

            # Collect for aggregate calibration
            all_y_true.append(y_test.values if hasattr(y_test, "values") else np.asarray(y_test))
            all_y_prob.append(p_test)

            logger.info(
                "Fold %s/%s complete: log_loss=%.4f, n_test=%d",
                league,
                test_season,
                eval_result.log_loss,
                n_test,
            )

    # -- Compute aggregate metrics -----------------------------------------
    result = _compute_aggregates(fold_results, all_y_true, all_y_prob, config)
    result.n_folds_skipped = n_folds_skipped

    # -- Save artifacts -----------------------------------------------------
    if config.save_artifacts and fold_results:
        _save_artifacts(result, config)

    return result


# ---------------------------------------------------------------------------
# Aggregate computation
# ---------------------------------------------------------------------------


def _compute_aggregates(
    fold_results: list[FoldResult],
    all_y_true: list[np.ndarray],
    all_y_prob: list[np.ndarray],
    config: BacktestConfig,
) -> BacktestResult:
    """Compute sample-weighted aggregate metrics from fold results."""
    if not fold_results:
        return BacktestResult(
            folds=[],
            aggregate_log_loss=float("nan"),
            aggregate_brier=float("nan"),
            aggregate_accuracy=float("nan"),
            per_league_log_loss={},
            per_league_roi={},
            aggregate_roi_curve=[(t, 0.0) for t in config.edge_thresholds],
            calibration_data=(np.array([], dtype=float), np.array([], dtype=float)),
            config=config,
            n_folds_skipped=0,
        )

    total_samples = sum(f.n_test for f in fold_results)

    # Sample-weighted averages
    agg_ll = sum(f.n_test * f.evaluation.log_loss for f in fold_results) / total_samples
    agg_bs = sum(f.n_test * f.evaluation.brier_score for f in fold_results) / total_samples
    agg_acc = sum(f.n_test * f.evaluation.accuracy_at_0_5 for f in fold_results) / total_samples

    # Per-league aggregates
    pl_data: dict[str, dict[str, float]] = defaultdict(
        lambda: {"n_test": 0.0, "log_loss": 0.0, "profit": 0.0, "stakes": 0.0}
    )
    for f in fold_results:
        pl = pl_data[f.league]
        pl["n_test"] += f.n_test
        pl["log_loss"] += f.n_test * f.evaluation.log_loss
        roi = f.roi_by_threshold.get(config.default_edge_threshold)
        if roi is not None:
            pl["profit"] += roi.profit_units
            pl["stakes"] += roi.total_bets

    per_league_log_loss = {lg: d["log_loss"] / d["n_test"] for lg, d in sorted(pl_data.items())}
    per_league_roi = {
        lg: (d["profit"] / d["stakes"] * 100.0) if d["stakes"] > 0 else 0.0
        for lg, d in sorted(pl_data.items())
    }

    # Aggregate ROI curve (aggregate profits and stakes across all folds)
    agg_roi_curve: list[tuple[float, float]] = []
    for threshold in config.edge_thresholds:
        total_profit = 0.0
        total_bets = 0
        for f in fold_results:
            roi = f.roi_by_threshold.get(threshold)
            if roi is not None:
                total_profit += roi.profit_units
                total_bets += roi.total_bets
        roi_pct = (total_profit / total_bets * 100.0) if total_bets > 0 else 0.0
        agg_roi_curve.append((threshold, roi_pct))

    # Calibration curve on concatenated predictions
    if all_y_true and all_y_prob:
        concat_y_true = np.concatenate(all_y_true)
        concat_y_prob = np.concatenate(all_y_prob)
        if len(concat_y_true) > 0:
            prob_true, prob_pred = calibration_curve(
                concat_y_true, concat_y_prob, n_bins=10, strategy="quantile"
            )
            calibration_data: tuple[np.ndarray, np.ndarray] = (prob_pred, prob_true)
        else:
            calibration_data = (np.array([], dtype=float), np.array([], dtype=float))
    else:
        calibration_data = (np.array([], dtype=float), np.array([], dtype=float))

    return BacktestResult(
        folds=fold_results,
        aggregate_log_loss=agg_ll,
        aggregate_brier=agg_bs,
        aggregate_accuracy=agg_acc,
        per_league_log_loss=per_league_log_loss,
        per_league_roi=per_league_roi,
        aggregate_roi_curve=agg_roi_curve,
        calibration_data=calibration_data,
        config=config,
        n_folds_skipped=0,
    )


# ---------------------------------------------------------------------------
# Artifact persistence
# ---------------------------------------------------------------------------


def _save_artifacts(result: BacktestResult, config: BacktestConfig) -> None:
    """Write backtest artifacts (JSON, calibration plot, ROI plot)."""
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON
    json_path = output_dir / "backtest_results.json"
    data = _records_to_dict(result)
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    logger.info("Wrote %s", json_path)

    # Calibration curve
    cal_path = output_dir / "calibration_curve.png"
    mean_pred, frac_pos = result.calibration_data
    if len(mean_pred) > 0 and len(frac_pos) > 0:
        _plot_calibration(mean_pred, frac_pos, cal_path)
        logger.info("Wrote %s", cal_path)

    # ROI curve
    roi_path = output_dir / "roi_by_edge_threshold.png"
    thresholds = [p[0] for p in result.aggregate_roi_curve]
    rois = [p[1] for p in result.aggregate_roi_curve]
    if thresholds:
        _plot_roi_curve(thresholds, rois, roi_path)
        logger.info("Wrote %s", roi_path)


def _records_to_dict(result: BacktestResult) -> dict:
    """Convert a ``BacktestResult`` to a JSON-serialisable dict.

    Handles numpy arrays, tuples, dataclasses, and ``Path`` objects.
    """
    d = asdict(result)
    return _convert_numpy(d)


def _convert_numpy(obj: Any) -> Any:
    """Recursively convert numpy types to Python primitives."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, dict):
        return {k: _convert_numpy(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_convert_numpy(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def _plot_calibration(mean_pred: np.ndarray, frac_pos: np.ndarray, path: Path) -> None:
    """Plot calibration curve (mean predicted prob vs fraction of positives)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(mean_pred, frac_pos, "o-", label="Model")
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Calibration Curve")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def _plot_roi_curve(thresholds: list[float], rois: list[float], path: Path) -> None:
    """Plot ROI percent vs edge threshold."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(thresholds, rois, "s-", markersize=6)
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Edge threshold")
    ax.set_ylabel("ROI (%)")
    ax.set_title("ROI vs Edge Threshold")
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
