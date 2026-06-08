"""Tests for the fussball CLI (U6: Prediction CLI)."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from fussball.cli import main
from fussball.features import build_feature_matrix, select_xy
from fussball.models import save_model, train

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_football_data_csv(
    path: Path,
    records: list[dict],
    date_fmt: str = "%Y-%m-%d",
) -> None:
    """Write synthetic match dicts to a Football-Data.co.uk-style CSV.

    Maps the synthetic record keys to the CSV column names expected by
    ``FreeCsvLoader``.
    """
    col_map = {
        "Date": "date",
        "HomeTeam": "home_team",
        "AwayTeam": "away_team",
        "FTHG": "home_goals",
        "FTAG": "away_goals",
        "HS": "home_shots",
        "AS": "away_shots",
        "HST": "home_shots_on_target",
        "AST": "away_shots_on_target",
        "HP": "home_possession",
        "AP": "away_possession",
        "HC": "home_corners",
        "AC": "away_corners",
        "HY": "home_cards",
        "AY": "away_cards",
        "HxG": "home_xg",
        "AxG": "away_xg",
        "BbAv>2.5": "odds_over_2_5",
        "BbAv<2.5": "odds_under_2_5",
        "AvgC>2.5": "closing_odds_over_2_5",
        "AvgC<2.5": "closing_odds_under_2_5",
    }

    rows: list[dict[str, str]] = []
    for rec in records:
        row = {}
        for csv_col, rec_key in col_map.items():
            val = rec.get(rec_key)
            if val is None:
                row[csv_col] = ""
            elif isinstance(val, datetime):
                row[csv_col] = val.strftime(date_fmt)
            elif isinstance(val, float):
                row[csv_col] = f"{val:.4f}"
            else:
                row[csv_col] = str(val)
        rows.append(row)

    fieldnames = list(col_map.keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_upcoming_csv(path: Path, rows: list[dict]) -> None:
    """Write an upcoming-fixtures CSV (Date, HomeTeam, AwayTeam, league, season)."""
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["Date", "HomeTeam", "AwayTeam", "Div", "league", "season"]
        )
        writer.writeheader()
        writer.writerows(rows)


def _train_and_save_model(
    synthetic_matches: list,
    output_path: Path,
) -> None:
    """Train a model on synthetic data and save to ``output_path``."""
    fm = build_feature_matrix(synthetic_matches)
    X, y = select_xy(fm)
    trained = train(X.values, y.values)
    save_model(trained, str(output_path))


@pytest.fixture
def runner() -> CliRunner:
    """Return a Click CliRunner."""
    return CliRunner()


# ---------------------------------------------------------------------------
# --help tests
# ---------------------------------------------------------------------------


class TestHelp:
    """Tests for CLI help output."""

    def test_main_help_shows_subcommands(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "download-data" in result.output
        assert "backtest" in result.output
        assert "predict" in result.output
        assert result.output.strip() != ""

    def test_backtest_help_shows_options(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["backtest", "--help"])
        assert result.exit_code == 0
        assert "--tune" in result.output
        assert "--no-tune" in result.output
        assert "--min-train-seasons" in result.output
        assert "--edge-threshold" in result.output

    def test_predict_help_shows_options(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["predict", "--help"])
        assert result.exit_code == 0
        assert "--fixtures" in result.output
        assert "--model" in result.output
        assert "--min-edge" in result.output
        assert "--json" in result.output

    def test_download_data_help_shows_options(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["download-data", "--help"])
        assert result.exit_code == 0
        assert "--leagues" in result.output
        assert "--seasons" in result.output
        assert "--force" in result.output


# ---------------------------------------------------------------------------
# Error / edge-case tests
# ---------------------------------------------------------------------------


class TestPredictErrors:
    """Tests for ``predict`` error handling."""

    def test_no_model_file(self, runner: CliRunner, tmp_path: Path) -> None:
        """Predict without a model file prints an error mentioning 'backtest'."""
        model_path = tmp_path / "nonexistent.joblib"
        result = runner.invoke(
            main,
            [
                "predict",
                "--model",
                str(model_path),
            ],
        )
        assert result.exit_code == 1
        assert "backtest" in result.output.lower()

    def test_missing_fixtures_file(self, runner: CliRunner, tmp_path: Path) -> None:
        """Predict with non-existent fixtures prints 'No upcoming matches'."""
        model_dir = tmp_path / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        fixtures_path = tmp_path / "no_such_file.csv"

        import joblib
        import lightgbm as lgb

        from fussball.models import TrainedModel

        dummy = lgb.LGBMClassifier(n_estimators=2, max_depth=1)
        dummy.fit([[0, 0], [1, 1]], [0, 1])
        tm = TrainedModel(model=dummy, params={})
        joblib.dump(tm, model_dir / "latest.joblib")

        result = runner.invoke(
            main,
            [
                "--data-dir",
                str(tmp_path),
                "predict",
                "--fixtures",
                str(fixtures_path),
                "--model",
                str(model_dir / "latest.joblib"),
            ],
        )
        assert result.exit_code == 0
        assert "No upcoming matches found" in result.output

    def test_empty_fixtures_csv(self, runner: CliRunner, tmp_path: Path) -> None:
        """Predict with an empty fixtures CSV prints 'No upcoming matches'."""
        model_dir = tmp_path / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        fixtures_path = tmp_path / "upcoming.csv"
        fixtures_path.write_text("Date,HomeTeam,AwayTeam,Div,league,season\n")

        import joblib
        import lightgbm as lgb

        from fussball.models import TrainedModel

        dummy = lgb.LGBMClassifier(n_estimators=2, max_depth=1)
        dummy.fit([[0, 0], [1, 1]], [0, 1])
        tm = TrainedModel(model=dummy, params={})
        joblib.dump(tm, model_dir / "latest.joblib")

        result = runner.invoke(
            main,
            [
                "--data-dir",
                str(tmp_path),
                "predict",
                "--fixtures",
                str(fixtures_path),
                "--model",
                str(model_dir / "latest.joblib"),
            ],
        )
        assert result.exit_code == 0
        assert "No upcoming matches found" in result.output

    def test_missing_history_dir(self, runner: CliRunner, tmp_path: Path) -> None:
        """Predict with missing history dir shows clear error."""
        model_dir = tmp_path / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        fixtures_path = tmp_path / "upcoming.csv"
        _write_upcoming_csv(
            fixtures_path,
            [
                {
                    "Date": "2024-01-15",
                    "HomeTeam": "Arsenal",
                    "AwayTeam": "Chelsea",
                    "Div": "E0",
                    "league": "E0",
                    "season": "2022-2023",
                }
            ],
        )

        import joblib
        import lightgbm as lgb

        from fussball.models import TrainedModel

        dummy = lgb.LGBMClassifier(n_estimators=2, max_depth=1)
        dummy.fit([[0, 0], [1, 1]], [0, 1])
        tm = TrainedModel(model=dummy, params={})
        joblib.dump(tm, model_dir / "latest.joblib")

        # Use a non-existent history dir
        missing_history = tmp_path / "no_data_here"
        result = runner.invoke(
            main,
            [
                "predict",
                "--fixtures",
                str(fixtures_path),
                "--model",
                str(model_dir / "latest.joblib"),
                "--history-dir",
                str(missing_history),
            ],
        )
        assert result.exit_code == 1
        assert "does not exist" in result.output.lower() or "download-data" in result.output.lower()


# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------


class TestPredictJson:
    """Test ``predict --json`` produces valid structured output."""

    @pytest.fixture(autouse=True)
    def _setup(
        self,
        synthetic_match_dicts: list[dict],
        synthetic_matches: list,
        tmp_path: Path,
    ) -> None:
        """Set up a model, upcoming fixtures, and historical CSV data."""
        # Write historical CSVs
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)

        # Group synthetic match dicts by (league, season) and write CSVs
        groups: dict[tuple[str, str], list[dict]] = {}
        for rec in synthetic_match_dicts:
            key = (rec["league"], rec["season"])
            groups.setdefault(key, []).append(rec)

        for (league, season), recs in groups.items():
            season_year = int(season.split("-")[0])
            compact = f"{season_year % 100:02d}{(season_year + 1) % 100:02d}"
            csv_path = raw_dir / f"{league}_{compact}.csv"
            _write_football_data_csv(csv_path, recs)

        # Train and save model
        model_dir = tmp_path / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / "latest.joblib"
        _train_and_save_model(synthetic_matches, model_path)

        # Write upcoming fixtures CSV (matches with future dates)
        fixtures_path = tmp_path / "upcoming.csv"
        _write_upcoming_csv(
            fixtures_path,
            [
                {
                    "Date": "2024-01-15",
                    "HomeTeam": "Arsenal",
                    "AwayTeam": "Chelsea",
                    "Div": "E0",
                    "league": "E0",
                    "season": "2022-2023",
                },
                {
                    "Date": "2024-01-15",
                    "HomeTeam": "Burnley",
                    "AwayTeam": "Sheffield",
                    "Div": "E0",
                    "league": "E0",
                    "season": "2022-2023",
                },
                {
                    "Date": "2024-01-16",
                    "HomeTeam": "RealMadrid",
                    "AwayTeam": "Sevilla",
                    "Div": "SP1",
                    "league": "SP1",
                    "season": "2022-2023",
                },
            ],
        )

        self.raw_dir = raw_dir
        self.model_path = model_path
        self.fixtures_path = fixtures_path
        self.tmp_path = tmp_path

    def test_json_output_is_parseable(self, runner: CliRunner) -> None:
        """``predict --json`` produces parseable JSON with expected keys."""
        result = runner.invoke(
            main,
            [
                "predict",
                "--json",
                "--fixtures",
                str(self.fixtures_path),
                "--model",
                str(self.model_path),
                "--history-dir",
                str(self.raw_dir),
            ],
        )
        assert result.exit_code == 0, f"OUT: {result.output}"

        data = json.loads(result.output)
        assert isinstance(data, list)
        assert len(data) > 0

        expected_keys = {
            "match_id",
            "home_team",
            "away_team",
            "over_prob",
            "edge",
        }
        for row in data:
            assert expected_keys.issubset(row.keys()), f"Missing keys in {row}"

        # Verify content
        home_teams = {r["home_team"] for r in data}
        assert "Arsenal" in home_teams
        assert "RealMadrid" in home_teams

    def test_json_output_valid_json(self, runner: CliRunner) -> None:
        """``--json`` output is valid JSON and each row is a dict."""
        result = runner.invoke(
            main,
            [
                "predict",
                "--json",
                "--fixtures",
                str(self.fixtures_path),
                "--model",
                str(self.model_path),
                "--history-dir",
                str(self.raw_dir),
            ],
        )
        assert result.exit_code == 0

        data = json.loads(result.output)
        assert isinstance(data, list)
        assert all(isinstance(r, dict) for r in data)

        # Check probability range
        for row in data:
            assert 0.0 <= row["over_prob"] <= 1.0, f"over_prob out of range: {row}"

    def test_table_output_contains_match_info(self, runner: CliRunner) -> None:
        """``predict`` without --json prints a Rich table with match info."""
        result = runner.invoke(
            main,
            [
                "predict",
                "--fixtures",
                str(self.fixtures_path),
                "--model",
                str(self.model_path),
                "--history-dir",
                str(self.raw_dir),
            ],
        )
        assert result.exit_code == 0, f"OUT: {result.output}"
        # Rich may truncate long names; check substrings
        assert "Arsen" in result.output
        assert "Chels" in result.output
        assert "RealM" in result.output
        assert "OVER" in result.output or "PASS" in result.output
        assert "Match Predictions" in result.output

    def test_min_edge_filters(self, runner: CliRunner) -> None:
        """``--min-edge`` filters out low-edge predictions."""
        # First get unfiltered output
        result_all = runner.invoke(
            main,
            [
                "predict",
                "--json",
                "--fixtures",
                str(self.fixtures_path),
                "--model",
                str(self.model_path),
                "--history-dir",
                str(self.raw_dir),
            ],
        )
        assert result_all.exit_code == 0
        _ = json.loads(result_all.output)

        # Now filter with a high edge threshold
        result_filtered = runner.invoke(
            main,
            [
                "predict",
                "--json",
                "--min-edge",
                "0.10",
                "--fixtures",
                str(self.fixtures_path),
                "--model",
                str(self.model_path),
                "--history-dir",
                str(self.raw_dir),
            ],
        )
        assert result_filtered.exit_code == 0

        try:
            filtered_rows = json.loads(result_filtered.output)
        except json.JSONDecodeError:
            # The output might say "No matches meet the edge threshold"
            assert "No matches meet the edge threshold" in result_filtered.output
            return

        # Verify filtered rows all have edge >= 0.10 (or edge is None)
        for row in filtered_rows:
            assert row["edge"] is None or row["edge"] >= 0.10


class TestBacktest:
    """Tests for the ``backtest`` subcommand."""

    def test_backtest_invokes_pipeline(
        self, runner: CliRunner, tmp_path: Path, synthetic_match_dicts: list[dict]
    ) -> None:
        """``backtest`` invokes ``run_backtest`` (mocked)."""
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)

        # Write a minimal CSV to the raw dir
        _write_football_data_csv(
            raw_dir / "E0_2122.csv",
            [
                r
                for r in synthetic_match_dicts
                if r["league"] == "E0" and r["season"] == "2021-2022"
            ],
        )

        with patch("fussball.cli.run_backtest") as mock_run:
            from fussball.backtest import BacktestResult

            mock_result = BacktestResult(
                aggregate_log_loss=0.62,
                aggregate_brier=0.25,
                aggregate_accuracy=0.65,
                folds=[],
                n_folds_skipped=0,
            )
            mock_run.return_value = mock_result

            result = runner.invoke(
                main,
                [
                    "--data-dir",
                    str(raw_dir),
                    "backtest",
                    "--no-tune",
                ],
            )

            assert result.exit_code == 0, f"OUT: {result.output}"
            mock_run.assert_called_once()
            args, _ = mock_run.call_args
            assert len(args) == 2  # (records, config)

    def test_backtest_empty_data_dir(self, runner: CliRunner, tmp_path: Path) -> None:
        """Backtest with no data exits with error."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir(parents=True, exist_ok=True)

        result = runner.invoke(
            main,
            [
                "--data-dir",
                str(empty_dir),
                "backtest",
                "--no-tune",
            ],
        )
        assert result.exit_code == 1
        assert "No match records found" in result.output


# ---------------------------------------------------------------------------
# download-data tests
# ---------------------------------------------------------------------------


class TestDownloadData:
    """Tests for the ``download-data`` subcommand."""

    @patch("requests.get")
    def test_download_data_makes_request(
        self,
        mock_get,
        runner: CliRunner,
        tmp_path: Path,
    ) -> None:
        """``download-data`` makes an HTTP request and writes a CSV."""
        mock_response = mock_get.return_value
        mock_response.status_code = 200
        mock_response.content = (
            b"Date,HomeTeam,AwayTeam,FTHG,FTAG\n2021-08-14,Arsenal,Chelsea,1,1\n"
        )

        data_dir = tmp_path / "raw"
        data_dir.mkdir(parents=True, exist_ok=True)

        result = runner.invoke(
            main,
            [
                "--data-dir",
                str(data_dir),
                "download-data",
                "--leagues",
                "E0",
                "--seasons",
                "2122",
            ],
        )

        assert result.exit_code == 0, f"OUT: {result.output}"
        # Should have downloaded at least 1 file
        downloaded_files = list(data_dir.glob("*.csv"))
        assert len(downloaded_files) >= 1

        # Verify the mock was called with the right URL
        mock_get.assert_called_once_with(
            "https://www.football-data.co.uk/mmz4281/2122/E0.csv",
            timeout=30,
        )

    @patch("requests.get")
    def test_download_data_handles_http_error(
        self,
        mock_get,
        runner: CliRunner,
        tmp_path: Path,
    ) -> None:
        """``download-data`` handles HTTP errors gracefully."""
        import requests

        mock_response = mock_get.return_value
        mock_response.raise_for_status.side_effect = requests.HTTPError("404 Not Found")
        mock_response.status_code = 404

        data_dir = tmp_path / "raw"
        data_dir.mkdir(parents=True, exist_ok=True)

        result = runner.invoke(
            main,
            [
                "--data-dir",
                str(data_dir),
                "download-data",
                "--leagues",
                "E0",
                "--seasons",
                "2122",
            ],
        )

        assert result.exit_code == 0  # Should handle gracefully
        assert "No files were downloaded" in result.output


# ---------------------------------------------------------------------------
# Global options tests
# ---------------------------------------------------------------------------


class TestGlobalOptions:
    """Tests for global CLI options."""

    def test_log_level_passthrough(self, runner: CliRunner) -> None:
        """``--log-level`` is accepted as a global option."""
        result = runner.invoke(main, ["--log-level", "DEBUG", "--help"])
        assert result.exit_code == 0

    def test_data_dir_and_models_dir(self, runner: CliRunner, tmp_path: Path) -> None:
        """``--data-dir`` and ``--models-dir`` are accepted."""
        result = runner.invoke(
            main,
            [
                "--data-dir",
                str(tmp_path / "data"),
                "--models-dir",
                str(tmp_path / "models"),
                "backtest",
                "--help",
            ],
        )
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# Integration: end-to-end predict
# ---------------------------------------------------------------------------


class TestPredictIntegration:
    """End-to-end integration tests for ``predict``."""

    @pytest.fixture(autouse=True)
    def _setup(
        self,
        synthetic_match_dicts: list[dict],
        synthetic_matches: list,
        tmp_path: Path,
    ) -> None:
        """Create a fully populated environment for predictions."""
        # Write historical CSVs grouped by (league, season)
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        groups: dict[tuple[str, str], list[dict]] = {}
        for rec in synthetic_match_dicts:
            groups.setdefault((rec["league"], rec["season"]), []).append(rec)
        for (league, season), recs in groups.items():
            season_year = int(season.split("-")[0])
            compact = f"{season_year % 100:02d}{(season_year + 1) % 100:02d}"
            _write_football_data_csv(raw_dir / f"{league}_{compact}.csv", recs)

        # Train and save model
        model_dir = tmp_path / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / "latest.joblib"
        _train_and_save_model(synthetic_matches, model_path)

        # Upcoming fixtures
        fixtures_path = tmp_path / "upcoming.csv"
        _write_upcoming_csv(
            fixtures_path,
            [
                {
                    "Date": "2024-01-15",
                    "HomeTeam": "Arsenal",
                    "AwayTeam": "Chelsea",
                    "Div": "E0",
                    "league": "E0",
                    "season": "2022-2023",
                },
                {
                    "Date": "2024-01-15",
                    "HomeTeam": "RealMadrid",
                    "AwayTeam": "Sevilla",
                    "Div": "SP1",
                    "league": "SP1",
                    "season": "2022-2023",
                },
            ],
        )

        self.raw_dir = raw_dir
        self.model_path = model_path
        self.fixtures_path = fixtures_path

    def test_predict_json_has_expected_keys(self, runner: CliRunner) -> None:
        """Full predict --json produces correct keys in every row."""
        result = runner.invoke(
            main,
            [
                "predict",
                "--json",
                "--fixtures",
                str(self.fixtures_path),
                "--model",
                str(self.model_path),
                "--history-dir",
                str(self.raw_dir),
            ],
        )
        assert result.exit_code == 0, f"OUT: {result.output}"

        data = json.loads(result.output)
        assert len(data) == 2  # 2 fixtures

        for row in data:
            assert "match_id" in row
            assert "home_team" in row
            assert "away_team" in row
            assert "over_prob" in row
            assert "edge" in row
            assert "recommendation" in row
            assert isinstance(row["over_prob"], float)
            assert 0.0 <= row["over_prob"] <= 1.0

    def test_predict_table_output(self, runner: CliRunner) -> None:
        """Full predict produces a table with match info."""
        result = runner.invoke(
            main,
            [
                "predict",
                "--fixtures",
                str(self.fixtures_path),
                "--model",
                str(self.model_path),
                "--history-dir",
                str(self.raw_dir),
            ],
        )
        assert result.exit_code == 0, f"OUT: {result.output}"
        # Rich may truncate long team names; check substrings
        assert "Arsen" in result.output
        assert "RealM" in result.output
        assert "Match Predictions" in result.output
