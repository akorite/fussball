"""Tests for the model training, tuning, evaluation, and serialization modules.

These tests construct synthetic (X, y) directly using NumPy/Pandas without
importing from ``fussball.features``, since U3 (feature engineering) is being
built in parallel.
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import log_loss

from fussball.models import (
    DEFAULT_PARAMS,
    EvaluationResult,
    ROIResults,
    TrainedModel,
    TuningResult,
    evaluate,
    load_model,
    save_model,
    train,
    tune,
)
from fussball.models.evaluator import _simulate_roi

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_X_y():
    """Deterministic synthetic dataset with a learnable signal.

    y depends linearly on 3 of 10 features — LightGBM should easily capture
    this pattern.
    """
    rng = np.random.default_rng(42)
    n = 500
    X = pd.DataFrame(rng.normal(size=(n, 10)), columns=[f"f{i}" for i in range(10)])
    signal = 0.5 * X["f0"] + 0.3 * X["f1"] - 0.2 * X["f2"]
    prob = 1 / (1 + np.exp(-signal))
    y = pd.Series((rng.uniform(size=n) < prob).astype(int), name="over_2_5", index=X.index)
    return X, y


@pytest.fixture
def synthetic_X_y_numpy():
    """Same as synthetic_X_y but returns raw numpy arrays."""
    rng = np.random.default_rng(42)
    n = 500
    X = rng.normal(size=(n, 10))
    signal = 0.5 * X[:, 0] + 0.3 * X[:, 1] - 0.2 * X[:, 2]
    prob = 1 / (1 + np.exp(-signal))
    y = (rng.uniform(size=n) < prob).astype(int)
    return X, y


@pytest.fixture
def all_zero_X_y():
    """Edge case: all-zero features — no signal."""
    rng = np.random.default_rng(42)
    n = 200
    X = pd.DataFrame(np.zeros((n, 5)), columns=[f"f{i}" for i in range(5)])
    # Balanced binary target with no feature signal
    y = pd.Series(rng.binomial(1, 0.5, size=n), name="over_2_5", index=X.index)
    return X, y


# =========================================================================
# Test: train — happy path
# =========================================================================


class TestTrain:
    """Happy-path tests for ``train()``."""

    def test_converges_lower_than_random_baseline(self, synthetic_X_y):
        """Trained model should beat the ~0.693 random-guess log loss."""
        X, y = synthetic_X_y
        result = train(X, y)
        y_prob = result.model.predict_proba(X)[:, 1]
        ll = log_loss(y, y_prob)
        assert ll < 0.7, f"Log loss {ll:.4f} should be < 0.7 (random baseline)"

    def test_returns_trained_model_dataclass(self, synthetic_X_y):
        """train() returns a TrainedModel with expected attributes."""
        X, y = synthetic_X_y
        result = train(X, y)
        assert isinstance(result, TrainedModel)
        assert isinstance(result.model, object)  # LGBMClassifier
        assert isinstance(result.params, dict)
        assert isinstance(result.best_iteration, int)

    def test_merged_params_include_defaults(self, synthetic_X_y):
        """Merged params should contain DEFAULT_PARAMS values."""
        X, y = synthetic_X_y
        result = train(X, y)
        assert result.params["objective"] == "binary"
        assert result.params["seed"] == 42

    def test_params_override_defaults(self, synthetic_X_y):
        """Passing custom params overrides default values."""
        X, y = synthetic_X_y
        result = train(X, y, params={"learning_rate": 0.2})
        assert result.params["learning_rate"] == 0.2

    def test_predict_proba_output_range(self, synthetic_X_y):
        """predict_proba returns values in [0, 1]."""
        X, y = synthetic_X_y
        result = train(X, y)
        y_prob = result.model.predict_proba(X)[:, 1]
        assert np.all(y_prob >= 0.0) and np.all(y_prob <= 1.0)

    def test_with_validation_set(self, synthetic_X_y):
        """Training with eval_set should enable early stopping."""
        X, y = synthetic_X_y
        n = len(X)
        split = int(n * 0.8)
        X_train, X_val = X.iloc[:split], X.iloc[split:]
        y_train, y_val = y.iloc[:split], y.iloc[split:]
        result = train(X_train, y_train, eval_set=(X_val, y_val))
        assert isinstance(result.best_iteration, int)
        assert result.best_iteration > 0

    def test_reproducibility(self, synthetic_X_y_numpy):
        """Same data + same seed + same params produces identical gain."""
        X, y = synthetic_X_y_numpy
        r1 = train(X, y, params={"seed": 42})
        r2 = train(X, y, params={"seed": 42})
        imp1 = r1.model.booster_.feature_importance(importance_type="gain")
        imp2 = r2.model.booster_.feature_importance(importance_type="gain")
        np.testing.assert_array_equal(imp1, imp2)

    def test_numpy_input(self, synthetic_X_y_numpy):
        """train() accepts raw numpy arrays."""
        import warnings

        X, y = synthetic_X_y_numpy
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = train(X, y)
        assert isinstance(result, TrainedModel)
        # Use a DataFrame for predict to avoid feature-name warnings
        X_pred = pd.DataFrame(X)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            y_prob = result.model.predict_proba(X_pred)[:, 1]
        assert y_prob.shape[0] == len(y)


# =========================================================================
# Test: train — edge cases
# =========================================================================


class TestTrainEdgeCases:
    def test_fewer_than_50_samples_raises(self):
        """Training with <50 samples raises ValueError."""
        X = np.random.default_rng(42).normal(size=(30, 5))
        y = np.random.default_rng(42).binomial(1, 0.5, size=30)
        with pytest.raises(ValueError, match="At least 50 samples"):
            train(X, y)

    def test_mismatched_lengths_raises(self):
        """Mismatched X/y lengths raises ValueError."""
        X = np.random.default_rng(42).normal(size=(100, 5))
        y = np.random.default_rng(42).binomial(1, 0.5, size=99)
        with pytest.raises(ValueError, match="mismatched lengths"):
            train(X, y)

    def test_all_zero_features_produces_valid_probs(self, all_zero_X_y):
        """All-zero features: model trains and probs in [0, 1]."""
        X, y = all_zero_X_y
        result = train(X, y)
        y_prob = result.model.predict_proba(X)[:, 1]
        assert np.all(y_prob >= 0.0) and np.all(y_prob <= 1.0)
        # With no signal, predictions should be near 0.5 on average
        avg_prob = float(np.mean(y_prob))
        assert 0.2 < avg_prob < 0.8, f"Mean prob {avg_prob:.3f} too extreme"


# =========================================================================
# Test: tune
# =========================================================================


class TestTune:
    """Tests for hyperparameter tuning via ``tune()``."""

    def test_tune_returns_tuning_result(self, synthetic_X_y):
        """tune() returns a TuningResult with all expected fields."""
        X, y = synthetic_X_y
        # Use very few trials for speed
        result = tune(X, y, n_trials=3, seed=42)
        assert isinstance(result, TuningResult)
        assert isinstance(result.study, object)  # optuna.Study
        assert isinstance(result.best_params, dict)
        assert isinstance(result.best_score, float)
        assert isinstance(result.best_model, TrainedModel)

    def test_best_params_contain_expected_keys(self, synthetic_X_y):
        """Best params should include the hyperparameters we searched over."""
        X, y = synthetic_X_y
        result = tune(X, y, n_trials=3, seed=42)
        expected_keys = {
            "learning_rate",
            "num_leaves",
            "min_child_samples",
            "feature_fraction",
            "bagging_fraction",
            "lambda_l1",
            "lambda_l2",
            "max_depth",
        }
        assert expected_keys.issubset(result.best_params.keys()), (
            f"Missing keys: {expected_keys - set(result.best_params.keys())}"
        )

    def test_best_model_predicts(self, synthetic_X_y):
        """Best model from tune can predict and probs are in [0,1]."""
        X, y = synthetic_X_y
        result = tune(X, y, n_trials=3, seed=42)
        y_prob = result.best_model.model.predict_proba(X)[:, 1]
        assert np.all(y_prob >= 0.0) and np.all(y_prob <= 1.0)

    def test_tune_with_single_league_data(self, synthetic_X_y):
        """Tuning on a single-league-sized dataset works."""
        X, y = synthetic_X_y
        # Use a subset (rows 0-99) to simulate a single-league scenario
        X_sub = X.iloc[:100]
        y_sub = y.iloc[:100]
        result = tune(X_sub, y_sub, n_trials=2, seed=42)
        assert isinstance(result, TuningResult)
        assert result.best_model.model.predict_proba(X_sub).shape[0] == 100


# =========================================================================
# Test: evaluate
# =========================================================================


class TestEvaluate:
    def test_basic_metrics(self):
        """evaluate returns proper metrics for known inputs."""
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.8, 0.9])
        result = evaluate(y_true, y_prob)
        assert isinstance(result, EvaluationResult)
        assert result.n_samples == 4
        assert result.roi_results is None
        assert 0.0 < result.log_loss < 1.0
        assert 0.0 <= result.brier_score <= 1.0
        assert result.accuracy_at_0_5 == 1.0  # all correct at 0.5 threshold

    def test_perfect_predictions(self):
        """Perfect predictions give log_loss ~0, brier ~0, acc=1."""
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.0, 0.0, 1.0, 1.0])
        result = evaluate(y_true, y_prob)
        assert result.log_loss < 0.01
        assert result.brier_score < 0.01
        assert result.accuracy_at_0_5 == 1.0

    def test_calibration_curve_structure(self):
        """Calibration curve returns two arrays of same length (<=10)."""
        rng = np.random.default_rng(42)
        y_true = rng.binomial(1, 0.5, size=200)
        y_prob = rng.uniform(size=200)
        result = evaluate(y_true, y_prob)
        prob_true, prob_pred = result.calibration_curve
        assert isinstance(prob_true, np.ndarray)
        assert isinstance(prob_pred, np.ndarray)
        assert len(prob_true) == len(prob_pred)
        assert len(prob_true) <= 10

    def test_market_odds_roi(self):
        """Providing market_over_odds populates roi_results."""
        y_true = np.array([1, 0, 1, 0])
        y_prob = np.array([0.9, 0.1, 0.8, 0.2])
        market_odds = np.array([2.0, 2.0, 2.0, 2.0])
        result = evaluate(y_true, y_prob, market_over_odds=market_odds)
        assert result.roi_results is not None
        assert isinstance(result.roi_results, ROIResults)
        assert result.roi_results.total_bets > 0

    def test_market_odds_none_skips_roi(self):
        """When market_over_odds is None, roi_results is None."""
        y_true = np.array([0, 1])
        y_prob = np.array([0.3, 0.7])
        result = evaluate(y_true, y_prob)
        assert result.roi_results is None

    def test_mismatched_length_raises(self):
        """Mismatched y_true/y_prob raises ValueError."""
        y_true = np.array([0, 1, 0])
        y_prob = np.array([0.3, 0.7])
        with pytest.raises(ValueError, match="mismatched shapes"):
            evaluate(y_true, y_prob)

    def test_market_odds_mismatched_length_raises(self):
        """Mismatched market_over_odds raises ValueError."""
        y_true = np.array([0, 1])
        y_prob = np.array([0.3, 0.7])
        market_odds = np.array([2.0])
        with pytest.raises(ValueError, match="mismatched shape"):
            evaluate(y_true, y_prob, market_over_odds=market_odds)

    def test_invalid_y_true_raises(self):
        """y_true containing values other than 0/1 raises ValueError."""
        y_true = np.array([0, 2, 1])
        y_prob = np.array([0.3, 0.5, 0.7])
        with pytest.raises(ValueError, match="y_true must contain only values 0 and 1"):
            evaluate(y_true, y_prob)

    def test_invalid_y_prob_raises(self):
        """y_prob outside [0,1] raises ValueError."""
        y_true = np.array([0, 1])
        y_prob = np.array([-0.1, 1.2])
        with pytest.raises(ValueError, match="y_prob must contain values in"):
            evaluate(y_true, y_prob)

    def test_roi_always_right_vs_always_wrong(self):
        """ROI: always-right predictions show profit, always-wrong show loss."""
        n = 100
        rng = np.random.default_rng(42)
        y_true = rng.binomial(1, 0.5, size=n)

        # Always right: model predicts perfectly (prob = true label)
        y_prob_right = y_true.astype(float)
        # Always wrong: model predicts opposite
        y_prob_wrong = 1.0 - y_true.astype(float)

        market_odds = np.full(n, 2.0)  # fair odds, implied prob = 0.5

        result_right = evaluate(y_true, y_prob_right, market_over_odds=market_odds)
        result_wrong = evaluate(y_true, y_prob_wrong, market_over_odds=market_odds)

        assert result_right.roi_results is not None
        assert result_wrong.roi_results is not None
        # Right predictions should yield positive profit; wrong should yield negative
        assert result_right.roi_results.profit_units > 0, (
            f"Right predictions should profit, got {result_right.roi_results.profit_units}"
        )
        assert result_wrong.roi_results.profit_units < 0, (
            f"Wrong predictions should lose, got {result_wrong.roi_results.profit_units}"
        )

    def test_pandas_series_input(self):
        """evaluate() accepts pd.Series for both y_true and y_prob."""
        y_true = pd.Series([0, 1, 1, 0])
        y_prob = pd.Series([0.2, 0.8, 0.9, 0.3])
        result = evaluate(y_true, y_prob)
        assert result.n_samples == 4
        assert result.log_loss > 0


# =========================================================================
# Test: ROI simulation (unit)
# =========================================================================


class TestROISimulation:
    def test_all_bets_win(self):
        """When model is always correct with edge > threshold, all bets win."""
        y_true = np.array([1, 1, 1])
        y_prob = np.array([0.8, 0.8, 0.8])
        market_odds = np.array([1.5, 1.5, 1.5])  # implied prob ~0.67, edge ~0.13 > 0.05
        roi = _simulate_roi(y_true, y_prob, market_odds, edge_threshold=0.05)
        assert roi.total_bets == 3
        assert roi.win_rate == 1.0
        # Each winning bet at odds 1.5: profit = odds - 1 = 0.5 per unit
        assert roi.profit_units == pytest.approx(1.5, rel=1e-3)
        assert roi.roi_percent == pytest.approx(50.0, rel=1e-3)

    def test_no_bets_when_no_edge(self):
        """When model never exceeds edge threshold, no bets placed."""
        y_true = np.array([1, 0, 1])
        y_prob = np.array([0.51, 0.49, 0.52])
        market_odds = np.array([2.0, 2.0, 2.0])  # implied prob = 0.5, edge <= 0.02
        roi = _simulate_roi(y_true, y_prob, market_odds, edge_threshold=0.05)
        assert roi.total_bets == 0
        assert roi.win_rate == 0.0
        assert roi.profit_units == 0.0
        assert roi.roi_percent == 0.0

    def test_edge_threshold_filtering(self):
        """Increasing edge threshold reduces number of bets."""
        y_true = np.array([1, 1, 1, 1])
        y_prob = np.array([0.55, 0.60, 0.70, 0.80])
        market_odds = np.full(4, 2.0)  # implied = 0.5

        roi_low = _simulate_roi(y_true, y_prob, market_odds, edge_threshold=0.02)
        roi_high = _simulate_roi(y_true, y_prob, market_odds, edge_threshold=0.25)

        assert roi_low.total_bets > roi_high.total_bets


# =========================================================================
# Test: serialization
# =========================================================================


class TestSerialization:
    def test_save_and_load_model(self, synthetic_X_y, tmp_path):
        """Model saved to disk and loaded back produces same predictions."""
        X, y = synthetic_X_y
        trained = train(X, y)

        path = tmp_path / "model.joblib"
        save_model(trained, path)

        loaded = load_model(path)
        assert isinstance(loaded, TrainedModel)
        assert loaded.params == trained.params
        assert loaded.best_iteration == trained.best_iteration

        # Predictions should match
        prob_orig = trained.model.predict_proba(X)[:, 1]
        prob_loaded = loaded.model.predict_proba(X)[:, 1]
        np.testing.assert_array_almost_equal(prob_orig, prob_loaded)

    def test_save_model_with_joblib(self, synthetic_X_y, tmp_path):
        """TrainedModel can be loaded via raw joblib as well."""
        X, y = synthetic_X_y
        trained = train(X, y)

        path = tmp_path / "model.joblib"
        save_model(trained, path)

        # Verify it's a valid joblib file
        loaded = joblib.load(path)
        assert isinstance(loaded, TrainedModel)


# =========================================================================
# Test: constants
# =========================================================================


class TestConstants:
    def test_default_params_structure(self):
        """DEFAULT_PARAMS has all expected keys."""
        expected = {
            "objective",
            "metric",
            "boosting_type",
            "verbosity",
            "learning_rate",
            "num_leaves",
            "min_child_samples",
            "feature_fraction",
            "bagging_fraction",
            "bagging_freq",
            "lambda_l1",
            "lambda_l2",
            "seed",
        }
        assert expected.issubset(DEFAULT_PARAMS.keys())

    def test_default_params_values(self):
        """Key default param values."""
        assert DEFAULT_PARAMS["objective"] == "binary"
        assert DEFAULT_PARAMS["metric"] == "binary_logloss"
        assert DEFAULT_PARAMS["boosting_type"] == "gbdt"
        assert DEFAULT_PARAMS["verbosity"] == -1
        assert DEFAULT_PARAMS["learning_rate"] == 0.05
        assert DEFAULT_PARAMS["num_leaves"] == 31
        assert DEFAULT_PARAMS["seed"] == 42


# =========================================================================
# Test: end-to-end light integration
# =========================================================================


class TestIntegration:
    def test_full_workflow(self, synthetic_X_y):
        """End-to-end: train → evaluate produces sane metrics.

        Uses early stopping on a held-out validation split so the model does
        not overfit when trained with the default 1000-boost-round budget.
        """
        X, y = synthetic_X_y
        n = len(X)
        split = int(n * 0.8)
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]

        # Use a sub-split of train as validation for early stopping
        val_split = int(len(X_train) * 0.8)
        X_tr, X_val = X_train.iloc[:val_split], X_train.iloc[val_split:]
        y_tr, y_val = y_train.iloc[:val_split], y_train.iloc[val_split:]

        trained = train(X_tr, y_tr, eval_set=(X_val, y_val))
        y_prob = trained.model.predict_proba(X_test)[:, 1]
        result = evaluate(y_test.values, y_prob)

        assert result.log_loss < 0.7, f"Log loss {result.log_loss:.4f} should be < 0.7"
        assert 0.0 <= result.brier_score <= 0.5
        assert 0.5 <= result.accuracy_at_0_5 <= 1.0
        assert result.n_samples == n - split

        # With early stopping, the model should have found a valid best iteration
        assert trained.best_iteration is not None and trained.best_iteration > 0

        # If we add hypothetical fair-market odds at 2.0 (implied 0.5),
        # expect some ROI computation
        market_odds = np.full(len(y_test), 2.0)
        result_with_market = evaluate(y_test.values, y_prob, market_over_odds=market_odds)
        assert result_with_market.roi_results is not None
