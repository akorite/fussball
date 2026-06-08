"""Fussball CLI — prediction and data management commands.

Usage
-----
    fussball --help
    fussball download-data
    fussball backtest
    fussball predict [--json]
"""

from __future__ import annotations

import json
import logging
import signal
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any

import click
import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

from fussball.backtest import BacktestConfig, BacktestResult, run_backtest
from fussball.data.loader import FreeCsvLoader
from fussball.data.schema import MatchRecord
from fussball.features import build_feature_matrix, select_xy
from fussball.models import load_model

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TOP5_LEAGUES = ["E0", "SP1", "D1", "I1", "F1"]

_DEFAULT_DATA_DIR = "data/raw"
_DEFAULT_MODELS_DIR = "data/models"
_DEFAULT_FIXTURES = "data/upcoming.csv"
_DEFAULT_MODEL = "latest.joblib"

# ---------------------------------------------------------------------------
# Signal handling
# ---------------------------------------------------------------------------


def _handle_sigint(signum: int, frame: Any) -> None:
    print("\nInterrupted.", file=sys.stderr)
    sys.exit(130)


signal.signal(signal.SIGINT, _handle_sigint)

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(levelname)s: %(message)s",
    )


# ---------------------------------------------------------------------------
# Season helpers
# ---------------------------------------------------------------------------


def _last_n_seasons(n: int = 5) -> list[str]:
    """Return the last N canonical season labels, e.g. ``["2021-2022", ...]``."""
    today = date.today()
    # Season starts around August; if we're past July use current year, else previous.
    end_year = today.year if today.month >= 8 else today.year - 1
    start_year = end_year - n + 1
    return [f"{y}-{y + 1}" for y in range(start_year, end_year + 1)]


def _season_to_compact(season: str) -> str:
    """Convert ``"2021-2022"`` → ``"2122"`` (2-digit compact form)."""
    start, end = season.split("-")
    return start[-2:] + end[-2:]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_upcoming_fixtures(path: Path) -> list[MatchRecord]:
    """Read a CSV of upcoming fixtures and return ``MatchRecord`` list.

    The CSV is expected to have at least the columns:
    ``Date, HomeTeam, AwayTeam, Div, league, season``

    Score fields are filled with ``0`` since results are unknown.
    """
    if not path.exists():
        return []

    df = pd.read_csv(path)
    if df.empty:
        return []

    # Normalise column names
    df = df.rename(columns={c: c.strip() for c in df.columns})

    required = {"Date", "HomeTeam", "AwayTeam", "league", "season"}
    if "Div" in df.columns and "league" not in df.columns:
        df["league"] = df["Div"]
    missing = required - set(df.columns)
    if missing:
        raise click.ClickException(
            f"Upcoming fixtures CSV is missing required columns: {sorted(missing)}"
        )

    records: list[MatchRecord] = []
    for idx, row in df.iterrows():
        league = str(row["league"]).strip().upper()
        season = str(row["season"]).strip()
        home = str(row["HomeTeam"]).strip()
        away = str(row["AwayTeam"]).strip()
        match_id = f"upcoming-{season}-{league}-{idx:04d}"

        try:
            parsed_date = datetime.strptime(str(row["Date"]).strip(), "%Y-%m-%d")
        except ValueError:
            try:
                parsed_date = datetime.strptime(str(row["Date"]).strip(), "%d/%m/%Y")
            except ValueError:
                raise click.ClickException(f"Row {idx}: unparseable date {row['Date']!r}") from None

        # Optional columns
        odds_over: float | None = None
        if "odds_over_2_5" in df.columns:
            v = row.get("odds_over_2_5")
            if pd.notna(v):
                odds_over = float(v)
        odds_under: float | None = None
        if "odds_under_2_5" in df.columns:
            v = row.get("odds_under_2_5")
            if pd.notna(v):
                odds_under = float(v)

        records.append(
            MatchRecord.model_validate(
                {
                    "match_id": match_id,
                    "date": parsed_date,
                    "league": league,
                    "season": season,
                    "source": "upcoming-fixtures",
                    "home_team": home,
                    "away_team": away,
                    "home_goals": 0,
                    "away_goals": 0,
                    "total_goals": 0,
                    "over_2_5": 0,
                    "odds_over_2_5": odds_over,
                    "odds_under_2_5": odds_under,
                }
            )
        )
    return records


def build_prediction_rows(
    records: list[MatchRecord],
    history: list[MatchRecord],
    model: Any,
) -> list[dict[str, Any]]:
    """Build prediction output rows for upcoming fixtures.

    Parameters
    ----------
    records:
        Upcoming fixture MatchRecords (placeholder scores).
    history:
        Historical MatchRecords used for feature computation.
    model:
        A ``TrainedModel`` instance with a ``.model`` attribute that exposes
        ``predict_proba``.

    Returns
    -------
    list[dict]
        Each dict has keys: ``match_id``, ``date``, ``league``, ``home_team``,
        ``away_team``, ``over_prob``, ``market_over_odds``, ``market_implied_prob``,
        ``edge``, ``recommendation``.
    """
    fixture_ids = {r.match_id for r in records}

    # Combine history + upcoming fixtures to compute features
    combined = history + records

    # Build the full feature matrix
    fm = build_feature_matrix(combined)
    if fm.X.empty:
        return []

    # Keep a copy with identifier columns before select_xy drops them
    X_full = fm.X.copy()

    # Get pure feature matrix for prediction
    X_feat, _ = select_xy(fm)

    # Filter to fixture rows
    fixture_mask = X_feat.index.isin(fixture_ids)
    X_fixtures = X_feat[fixture_mask]

    if X_fixtures.empty:
        return []

    # Predict
    probs = model.model.predict_proba(X_fixtures)[:, 1]

    # Build result rows
    rows: list[dict[str, Any]] = []
    for i, match_id in enumerate(X_fixtures.index):
        # Find the original match record
        match = _find_record(combined, match_id)
        if match is None:
            continue

        over_prob = float(probs[i])

        # Get market info
        market_over_odds = match.odds_over_2_5
        market_implied_prob: float | None = None
        edge: float | None = None
        if market_over_odds is not None:
            market_implied_prob = 1.0 / market_over_odds
            edge = over_prob - market_implied_prob

        recommendation = "OVER" if edge is not None and edge >= 0.05 else "PASS"

        # Get identifiers from the full feature matrix
        row_info = X_full.loc[match_id]
        home_team = str(row_info.get("home_team", match.home_team))
        away_team = str(row_info.get("away_team", match.away_team))
        league = str(row_info.get("league", match.league))
        match_date = match.date.strftime("%Y-%m-%d")

        rows.append(
            {
                "match_id": match_id,
                "date": match_date,
                "league": league,
                "home_team": home_team,
                "away_team": away_team,
                "over_prob": round(over_prob, 4),
                "market_over_odds": market_over_odds,
                "market_implied_prob": round(market_implied_prob, 4)
                if market_implied_prob is not None
                else None,
                "edge": round(edge, 4) if edge is not None else None,
                "recommendation": recommendation,
            }
        )
    return rows


def _find_record(records: list[MatchRecord], match_id: str) -> MatchRecord | None:
    """Find the first record with the given match_id."""
    for r in records:
        if r.match_id == match_id:
            return r
    return None


def _load_history(data_dir: str | Path) -> list[MatchRecord]:
    """Load all historical match records from a directory."""
    loader = FreeCsvLoader(Path(data_dir))
    if not loader.data_dir.exists():
        raise click.ClickException(
            f"Historical data directory does not exist: {loader.data_dir}. "
            "Run ``fussball download-data`` first."
        )
    records = loader.load_seasons([], [])
    if not records:
        raise click.ClickException(
            f"No match records found in {loader.data_dir}. Run ``fussball download-data`` first."
        )
    return records


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------


@click.group()
@click.option("--data-dir", default=_DEFAULT_DATA_DIR, show_default=True, help="Data directory")
@click.option(
    "--models-dir", default=_DEFAULT_MODELS_DIR, show_default=True, help="Models directory"
)
@click.option(
    "--log-level",
    default="INFO",
    show_default=True,
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], case_sensitive=False),
)
@click.pass_context
def main(ctx: click.Context, data_dir: str, models_dir: str, log_level: str) -> None:
    """Fussball — soccer over/under 2.5 goals ML prediction system."""
    _setup_logging(log_level)
    ctx.ensure_object(dict)
    ctx.obj["data_dir"] = data_dir
    ctx.obj["models_dir"] = models_dir


# ---------------------------------------------------------------------------
# download-data
# ---------------------------------------------------------------------------


@main.command()
@click.option(
    "--leagues",
    multiple=True,
    default=TOP5_LEAGUES,
    help="League codes to download (repeatable, default: top 5)",
)
@click.option(
    "--seasons",
    multiple=True,
    default=None,
    help="Season labels to download (repeatable, default: last 5)",
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Re-download even if file exists",
)
@click.pass_context
def download_data(
    ctx: click.Context,
    leagues: tuple[str, ...],
    seasons: tuple[str, ...] | None,
    force: bool,
) -> None:
    """Download historical match data from Football-Data.co.uk."""
    data_dir = Path(ctx.obj["data_dir"])
    data_dir.mkdir(parents=True, exist_ok=True)

    # Default seasons: last 5 from current date
    resolved_seasons: list[str] = list(seasons) if seasons else _last_n_seasons(5)

    loader = FreeCsvLoader(data_dir)
    paths = loader.download(leagues=list(leagues), seasons=resolved_seasons, force=force)

    if paths:
        console = Console()
        console.print(f"[green]Downloaded {len(paths)} file(s) to {data_dir}[/green]")
    else:
        console = Console()
        console.print("[yellow]No files were downloaded (all up-to-date?)[/yellow]")


# ---------------------------------------------------------------------------
# backtest
# ---------------------------------------------------------------------------


@main.command()
@click.option("--tune/--no-tune", default=False, help="Run hyperparameter tuning")
@click.option("--min-train-seasons", default=1, type=int, help="Minimum training seasons")
@click.option("--edge-threshold", default=0.05, type=float, help="Edge threshold for betting")
@click.option("--no-artifacts", is_flag=True, default=False, help="Skip writing artifacts")
@click.pass_context
def backtest(
    ctx: click.Context,
    tune: bool,
    min_train_seasons: int,
    edge_threshold: float,
    no_artifacts: bool,
) -> None:
    """Run the full backtest pipeline."""
    console = Console()

    # Load data
    data_dir = Path(ctx.obj["data_dir"])
    try:
        records = _load_history(data_dir)
    except click.ClickException:
        raise
    except Exception as e:
        raise click.ClickException(f"Failed to load data: {e}") from e

    config = BacktestConfig(
        tune=tune,
        min_train_seasons=min_train_seasons,
        default_edge_threshold=edge_threshold,
        save_artifacts=not no_artifacts,
    )

    console.print(f"[cyan]Running backtest on {len(records)} records...[/cyan]")
    try:
        results = run_backtest(records, config)
    except Exception as e:
        raise click.ClickException(f"Backtest failed: {e}") from e

    # Print summary table
    if results and results.folds:
        _print_backtest_summary(console, results)
    else:
        console.print("[yellow]Backtest returned no results (no folds completed).[/yellow]")


def _print_backtest_summary(console: Console, results: BacktestResult) -> None:
    """Print a Rich table with per-league and aggregate metrics."""
    table = Table(title="Backtest Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Aggregate Log Loss", f"{results.aggregate_log_loss:.4f}")
    if not np.isnan(results.aggregate_brier):
        table.add_row("Aggregate Brier", f"{results.aggregate_brier:.4f}")
    if not np.isnan(results.aggregate_accuracy):
        table.add_row("Aggregate Accuracy", f"{results.aggregate_accuracy:.4f}")
    table.add_row("Total Folds", str(len(results.folds)))
    if results.n_folds_skipped > 0:
        table.add_row("Skipped Folds", str(results.n_folds_skipped))

    for league, ll in results.per_league_log_loss.items():
        table.add_row(f"Log Loss ({league})", f"{ll:.4f}")

    for league, roi in results.per_league_roi.items():
        table.add_row(f"ROI ({league})", f"{roi:.2f}%")

    console.print(table)


# ---------------------------------------------------------------------------
# predict
# ---------------------------------------------------------------------------


@main.command()
@click.option(
    "--fixtures",
    default=_DEFAULT_FIXTURES,
    type=click.Path(path_type=Path),
    help="Path to upcoming fixtures CSV",
)
@click.option(
    "--model",
    default=None,
    type=click.Path(path_type=Path),
    help="Path to trained model file",
)
@click.option(
    "--history-dir",
    default=None,
    type=click.Path(path_type=Path),
    help="Directory with historical CSV data",
)
@click.option(
    "--min-edge",
    default=0.0,
    type=float,
    help="Minimum edge threshold for filtering predictions",
)
@click.option("--json/--no-json", default=False, help="Output JSON instead of table")
@click.pass_context
def predict(
    ctx: click.Context,
    fixtures: Path,
    model: Path | None,
    history_dir: Path | None,
    min_edge: float,
    json: bool,
) -> None:
    """Predict over/under probabilities for upcoming fixtures."""
    console = Console(file=sys.stdout)

    # Resolve model path
    models_dir = Path(ctx.obj["models_dir"])
    if model is None:
        model = models_dir / _DEFAULT_MODEL

    if not model.exists():
        raise click.ClickException(
            f"No trained model found at {model}. Run `fussball backtest` to train a model first."
        )

    # Load model
    try:
        trained = load_model(str(model))
    except Exception as e:
        raise click.ClickException(f"Failed to load model: {e}") from e

    # Resolve history dir
    if history_dir is None:
        history_dir = Path(ctx.obj["data_dir"])

    # Load fixtures
    fixture_records = load_upcoming_fixtures(fixtures)
    if not fixture_records:
        click.echo("No upcoming matches found.")
        return

    # Load historical data
    try:
        history = _load_history(history_dir)
    except click.ClickException:
        raise
    except Exception as e:
        raise click.ClickException(f"Failed to load historical data: {e}") from e

    # Build prediction rows
    logger.info(
        "Building predictions for %d fixtures on %d historical records...",
        len(fixture_records),
        len(history),
    )
    rows = build_prediction_rows(fixture_records, history, trained)

    if not rows:
        click.echo("No upcoming matches found.")
        return

    # Filter by min edge
    if min_edge > 0.0:
        rows = [r for r in rows if r["edge"] is not None and r["edge"] >= min_edge]

    if not rows:
        click.echo("No matches meet the edge threshold.")
        return

    # Output
    if json:
        _print_json(console, rows)
    else:
        _print_table(console, rows)


def _print_table(console: Console, rows: list[dict[str, Any]]) -> None:
    """Print prediction rows as a Rich table."""
    console = Console(file=sys.stdout, width=180)
    table = Table(title="Match Predictions")
    table.add_column("Date", style="cyan")
    table.add_column("League", style="magenta")
    table.add_column("Home", style="white")
    table.add_column("Away", style="white")
    table.add_column("Over Prob", justify="right")
    table.add_column("Market Odds", justify="right")
    table.add_column("Implied Prob", justify="right")
    table.add_column("Edge", justify="right")
    table.add_column("Rec", style="bold")

    for r in rows:
        edge_str = f"{r['edge']:.1%}" if r["edge"] is not None else "N/A"
        impl_str = (
            f"{r['market_implied_prob']:.1%}" if r["market_implied_prob"] is not None else "N/A"
        )
        odds_str = f"{r['market_over_odds']:.2f}" if r["market_over_odds"] is not None else "N/A"

        rec_style = "green" if r["recommendation"] == "OVER" else "white"
        table.add_row(
            r["date"],
            r["league"],
            r["home_team"],
            r["away_team"],
            f"{r['over_prob']:.1%}",
            odds_str,
            impl_str,
            edge_str,
            f"[{rec_style}]{r['recommendation']}[/{rec_style}]",
        )

    console.print(table)


def _print_json(console: Console, rows: list[dict[str, Any]]) -> None:
    """Print prediction rows as formatted JSON."""
    output = json.dumps(rows, indent=2, default=str)
    console.print(output)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
