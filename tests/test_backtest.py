"""Tests for the backtest pipeline (``fussball.backtest``).

Test scenarios:
- Happy path: full backtest on synthetic data with sufficient samples
- Edge: league with fewer seasons than required is skipped
- Edge: ROI at 0 % vs 5 % thresholds differ
- Edge: ``save_artifacts=False`` produces same data without writing files
- Edge: empty record list
- Edge: all records belong to a single season
- Integration: determinism (same seed → identical aggregate_log_loss)
- Edge: JSON serialisation round-trip
- Edge: ``data/backtest_results.json`` is created when ``save_artifacts=True``
"""

from __future__ import annotations

import json
import logging
from datetime import datetime

import numpy as np
import pytest

from fussball.backtest import (
    BacktestConfig,
    BacktestResult,
    FoldResult,
    _records_to_dict,
    run_backtest,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_large_dataset(match_factory) -> list:
    """Build a deterministic dataset large enough for backtest folds.

    Layout: 2 leagues x 2 seasons x 6 teams (round-robin twice) = 60 matches
    per (league, season).  Use deterministic scores derived from match indices
    so the model can find some signal.

    Returns
    -------
    list[MatchRecord]
        240 match records total.
    """
    leagues = {
        "E0": [f"E_Team{i}" for i in range(6)],
        "SP1": [f"S_Team{i}" for i in range(6)],
    }
    seasons = ["2021-2022", "2022-2023"]
    matches: list = []
    match_count = 0

    for season in seasons:
        date_base = datetime(2020 + int(season[:2]), 8, 14)
        for league, teams in leagues.items():
            pairings: list[tuple[str, str]] = []
            for i, home in enumerate(teams):
                for j, away in enumerate(teams):
                    if i == j:
                        continue
                    pairings.append((home, away))
                    pairings.append((away, home))

            for k, (home, away) in enumerate(pairings):
                match_count += 1
                # Deterministic varied scores (enough signal to train)
                hg = (k * 7 + match_count * 3) % 6
                ag = (k * 11 + match_count * 5) % 4
                total = hg + ag
                over_odds = round(2.20 - total * 0.05, 2) if total < 5 else 1.50
                under_odds = round(1.05 / (1.0 - 1.0 / over_odds), 2) if over_odds > 1 else 1.90

                matches.append(
                    match_factory(
                        match_id=f"{season}-{league}-{match_count:04d}",
                        date=date_base,
                        league=league,
                        season=season,
                        home_team=home,
                        away_team=away,
                        home_goals=hg,
                        away_goals=ag,
                        closing_odds_over_2_5=over_odds,
                        closing_odds_under_2_5=under_odds,
                    )
                )

    return matches


def _assert_result_shape(result: BacktestResult) -> None:
    """Verify that a ``BacktestResult`` has all expected fields populated."""
    assert isinstance(result, BacktestResult)
    assert isinstance(result.folds, list)
    assert isinstance(result.config, BacktestConfig)
    assert isinstance(result.n_folds_skipped, int)
    assert isinstance(result.per_league_log_loss, dict)
    assert isinstance(result.per_league_roi, dict)
    assert isinstance(result.aggregate_roi_curve, list)
    assert isinstance(result.calibration_data, tuple)
    assert len(result.calibration_data) == 2


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_happy_path_full_backtest(match_factory, tmp_path):
    """Run a full backtest on synthetic data with sufficient samples per fold.

    Expects a ``BacktestResult`` with populated metrics and at least one fold.
    """
    records = _make_large_dataset(match_factory)
    config = BacktestConfig(
        min_train_seasons=1,
        min_samples_per_fold=10,  # well below 60 per season
        output_dir=tmp_path,
        save_artifacts=True,
        seed=42,
    )
    result = run_backtest(records, config)

    _assert_result_shape(result)
    assert len(result.folds) >= 1, "Should have at least one fold"
    assert result.n_folds_skipped == 0, "No folds should be skipped with large dataset"

    # Aggregate metrics should be finite
    assert np.isfinite(result.aggregate_log_loss), f"log_loss={result.aggregate_log_loss}"
    assert np.isfinite(result.aggregate_brier), f"brier={result.aggregate_brier}"
    assert np.isfinite(result.aggregate_accuracy), f"accuracy={result.aggregate_accuracy}"
    assert 0.0 <= result.aggregate_log_loss <= 2.0

    # Per-league metrics
    assert set(result.per_league_log_loss.keys()) == {"E0", "SP1"}
    assert set(result.per_league_roi.keys()) == {"E0", "SP1"}

    # ROI curve should have entries for each threshold
    assert len(result.aggregate_roi_curve) == len(config.edge_thresholds)

    # Calibration data should be non-empty
    mean_pred, frac_pos = result.calibration_data
    assert len(mean_pred) > 0
    assert len(frac_pos) > 0

    # Each fold should have the expected structure
    for fold in result.folds:
        assert isinstance(fold, FoldResult)
        assert isinstance(fold.league, str)
        assert isinstance(fold.train_seasons, list)
        assert isinstance(fold.test_season, str)
        assert fold.n_train > 0
        assert fold.n_test > 0
        assert isinstance(fold.roi_by_threshold, dict)
        for t in config.edge_thresholds:
            assert t in fold.roi_by_threshold


# ---------------------------------------------------------------------------
# Edge: league with fewer seasons than required
# ---------------------------------------------------------------------------


def test_league_skipped(match_factory, tmp_path, caplog):
    """A league with fewer seasons than ``min_train_seasons + 1`` is skipped.

    Verify a warning is logged and ``n_folds_skipped`` is incremented.
    """
    # Create data with only 1 season for one league
    records = _make_large_dataset(match_factory)
    # Filter to keep only "2022-2023" for SP1 (making it a 1-season league)
    filtered = [r for r in records if r.league == "E0" or r.season != "2022-2023"]

    config = BacktestConfig(
        min_train_seasons=1,
        min_samples_per_fold=10,
        output_dir=tmp_path,
        save_artifacts=False,
        seed=42,
    )

    with caplog.at_level(logging.WARNING):
        result = run_backtest(filtered, config)

    # SP1 should have only 1 season (2021-2022), which is < 2 needed
    assert result.n_folds_skipped >= 1
    assert any("Skipping league" in msg and "SP1" in msg for msg in caplog.messages)

    # E0 should still have its folds
    assert len(result.folds) >= 1
    _assert_result_shape(result)


# ---------------------------------------------------------------------------
# Edge: ROI at 0 % vs 5 % thresholds
# ---------------------------------------------------------------------------


def test_roi_threshold_differences(match_factory, tmp_path):
    """ROI metrics differ between 0 % and 5 % edge thresholds.

    At 0 % threshold more bets are placed than at 5 %, and the ROI values
    are reported correctly in the curve.
    """
    records = _make_large_dataset(match_factory)
    config = BacktestConfig(
        min_train_seasons=1,
        min_samples_per_fold=10,
        edge_thresholds=(0.0, 0.05),
        default_edge_threshold=0.05,
        output_dir=tmp_path,
        save_artifacts=False,
        seed=42,
    )
    result = run_backtest(records, config)

    # ROI curve has both thresholds
    curve = dict(result.aggregate_roi_curve)
    assert 0.0 in curve
    assert 0.05 in curve

    # Total bets at 0 % >= total bets at 5 %
    total_bets_0 = sum(f.roi_by_threshold[0.0].total_bets for f in result.folds)
    total_bets_5 = sum(f.roi_by_threshold[0.05].total_bets for f in result.folds)
    assert total_bets_0 >= total_bets_5

    # ROI values are reported (not NaN) and may differ
    roi_0 = curve[0.0]
    roi_5 = curve[0.05]
    assert isinstance(roi_0, float)
    assert isinstance(roi_5, float)
    # They should not both be zero unless model had no edge — but at least one
    # threshold should have placed bets if the model has any signal
    assert total_bets_0 > 0, "Model should place at least some bets at 0 % threshold"


# ---------------------------------------------------------------------------
# Edge: save_artifacts=False
# ---------------------------------------------------------------------------


def test_save_artifacts_false(match_factory, tmp_path):
    """When ``save_artifacts=False`` the same result data is returned but no
    files are written to disk.
    """
    records = _make_large_dataset(match_factory)
    config = BacktestConfig(
        min_train_seasons=1,
        min_samples_per_fold=10,
        output_dir=tmp_path / "data",
        save_artifacts=False,
        seed=42,
    )
    result = run_backtest(records, config)

    _assert_result_shape(result)
    assert len(result.folds) >= 1

    # Verify no artifacts were written
    results_json = tmp_path / "data" / "backtest_results.json"
    cal_png = tmp_path / "data" / "calibration_curve.png"
    roi_png = tmp_path / "data" / "roi_by_edge_threshold.png"
    assert not results_json.exists()
    assert not cal_png.exists()
    assert not roi_png.exists()

    # The result data should be valid regardless
    assert np.isfinite(result.aggregate_log_loss)


# ---------------------------------------------------------------------------
# Edge: empty records
# ---------------------------------------------------------------------------


def test_empty_records(tmp_path):
    """An empty record list produces a ``BacktestResult`` with zero folds,
    NaN aggregate metrics, and no per-league data.
    """
    config = BacktestConfig(
        output_dir=tmp_path,
        save_artifacts=False,
        seed=42,
    )
    result = run_backtest([], config)

    _assert_result_shape(result)
    assert len(result.folds) == 0
    assert result.n_folds_skipped == 0  # no leagues to skip
    assert np.isnan(result.aggregate_log_loss)
    assert np.isnan(result.aggregate_brier)
    assert np.isnan(result.aggregate_accuracy)
    assert result.per_league_log_loss == {}
    assert result.per_league_roi == {}

    # Calibration data should be empty arrays
    mean_pred, frac_pos = result.calibration_data
    assert len(mean_pred) == 0
    assert len(frac_pos) == 0


# ---------------------------------------------------------------------------
# Edge: single season
# ---------------------------------------------------------------------------


def test_single_season(match_factory, tmp_path):
    """All records belong to a single season.  With default config the league
    is skipped (need ≥ 2 seasons).  The backtest does not crash.
    """
    records = _make_large_dataset(match_factory)
    # Keep only the first season
    single_season = [r for r in records if r.season == "2021-2022"]

    config = BacktestConfig(
        min_train_seasons=1,
        min_samples_per_fold=10,
        output_dir=tmp_path,
        save_artifacts=False,
        seed=42,
    )
    result = run_backtest(single_season, config)

    _assert_result_shape(result)
    # With only 1 season, each league has 1 season < 2, so skip
    assert len(result.folds) == 0
    assert result.n_folds_skipped == len({r.league for r in single_season})


# ---------------------------------------------------------------------------
# Integration: determinism
# ---------------------------------------------------------------------------


def test_determinism(match_factory, tmp_path):
    """Running the backtest twice with the same data and seed produces
    identical aggregate log loss.
    """
    records = _make_large_dataset(match_factory)
    config = BacktestConfig(
        min_train_seasons=1,
        min_samples_per_fold=10,
        output_dir=tmp_path,
        save_artifacts=False,
        seed=42,
    )

    result_a = run_backtest(records, config)
    result_b = run_backtest(records, config)

    assert result_a.aggregate_log_loss == pytest.approx(result_b.aggregate_log_loss, abs=1e-10)
    assert result_a.aggregate_brier == pytest.approx(result_b.aggregate_brier, abs=1e-10)
    # Number of folds should match
    assert len(result_a.folds) == len(result_b.folds)


# ---------------------------------------------------------------------------
# Edge: JSON serialisation round-trip
# ---------------------------------------------------------------------------


def test_json_roundtrip(match_factory, tmp_path):
    """The JSON-serialised dict can be loaded back and key fields match the
    in-memory result.
    """
    records = _make_large_dataset(match_factory)
    config = BacktestConfig(
        min_train_seasons=1,
        min_samples_per_fold=10,
        output_dir=tmp_path,
        save_artifacts=False,
        seed=42,
    )
    result = run_backtest(records, config)

    # Serialise
    d = _records_to_dict(result)
    json_str = json.dumps(d, indent=2, default=str)

    # Round-trip
    loaded = json.loads(json_str)

    # Check top-level fields
    assert loaded["aggregate_log_loss"] == pytest.approx(result.aggregate_log_loss, abs=1e-10)
    assert loaded["aggregate_brier"] == pytest.approx(result.aggregate_brier, abs=1e-10)
    assert loaded["aggregate_accuracy"] == pytest.approx(result.aggregate_accuracy, abs=1e-10)
    assert len(loaded["folds"]) == len(result.folds)
    assert len(loaded["aggregate_roi_curve"]) == len(result.aggregate_roi_curve)

    # Check first fold
    if loaded["folds"]:
        f_loaded = loaded["folds"][0]
        f_orig = result.folds[0]
        assert f_loaded["league"] == f_orig.league
        assert f_loaded["test_season"] == f_orig.test_season
        assert f_loaded["n_train"] == f_orig.n_train
        assert f_loaded["n_test"] == f_orig.n_test
        assert len(f_loaded["roi_by_threshold"]) == len(config.edge_thresholds)

    # Per-league metrics match
    assert set(loaded["per_league_log_loss"].keys()) == set(result.per_league_log_loss.keys())
    for league in result.per_league_log_loss:
        assert loaded["per_league_log_loss"][league] == pytest.approx(
            result.per_league_log_loss[league], abs=1e-10
        )

    # Calibration data is stored as lists of floats
    assert isinstance(loaded["calibration_data"], list)
    assert len(loaded["calibration_data"]) == 2


# ---------------------------------------------------------------------------
# Edge: JSON file created when save_artifacts=True
# ---------------------------------------------------------------------------


def test_json_file_created(match_factory, tmp_path):
    """When ``save_artifacts=True``, ``backtest_results.json`` is created and
    contains the expected top-level keys.
    """
    records = _make_large_dataset(match_factory)
    config = BacktestConfig(
        min_train_seasons=1,
        min_samples_per_fold=10,
        output_dir=tmp_path,
        save_artifacts=True,
        seed=42,
    )
    result = run_backtest(records, config)

    json_path = tmp_path / "backtest_results.json"
    assert json_path.exists(), f"{json_path} should exist"

    with open(json_path) as f:
        data = json.load(f)

    # Expected top-level keys
    expected_keys = {
        "folds",
        "aggregate_log_loss",
        "aggregate_brier",
        "aggregate_accuracy",
        "per_league_log_loss",
        "per_league_roi",
        "aggregate_roi_curve",
        "calibration_data",
        "config",
        "n_folds_skipped",
    }
    assert expected_keys.issubset(data.keys()), f"Missing keys: {expected_keys - set(data.keys())}"

    # Values match the in-memory result
    assert data["aggregate_log_loss"] == pytest.approx(result.aggregate_log_loss, abs=1e-10)

    # PNG files should also exist (calibration + ROI plots)
    cal_png = tmp_path / "calibration_curve.png"
    roi_png = tmp_path / "roi_by_edge_threshold.png"
    assert cal_png.exists(), f"{cal_png} should exist"
    assert roi_png.exists(), f"{roi_png} should exist"
