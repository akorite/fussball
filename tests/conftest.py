"""Shared test fixtures for the fussball test suite.

This conftest is intentionally framework-light: it provides deterministic
synthetic match data and small utility helpers that the data/feature/model/
backtest/CLI test modules all consume. Tests can request fixtures by name
without importing this module directly.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

# -----------------------------------------------------------------------------
# Synthetic match data
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class _Team:
    """A team in the synthetic league."""

    name: str
    league: str
    attack: float  # avg goals scored per game
    defense: float  # avg goals conceded per game
    shots_on_target: float = 5.0
    shots: float = 13.0
    possession: float = 0.5
    corners: float = 5.5
    cards: float = 2.0
    xg_for: float = 1.4
    xg_against: float = 1.3


def _default_teams() -> dict[str, _Team]:
    """Construct a deterministic 4-team-per-league, 2-league roster.

    Teams are designed so the model can find some signal:
    - Strong attack vs weak defense produces over-2.5 matches
    - Weak attack vs strong defense produces under-2.5 matches
    """

    premier = [
        _Team("Arsenal", "E0", attack=1.9, defense=0.8, xg_for=1.9, xg_against=0.9),
        _Team("Chelsea", "E0", attack=1.4, defense=1.2, xg_for=1.5, xg_against=1.2),
        _Team("Burnley", "E0", attack=0.8, defense=1.9, xg_for=0.9, xg_against=1.8),
        _Team("Sheffield", "E0", attack=0.7, defense=2.0, xg_for=0.8, xg_against=2.0),
    ]
    laliga = [
        _Team("RealMadrid", "SP1", attack=2.1, defense=0.7, xg_for=2.1, xg_against=0.8),
        _Team("Sevilla", "SP1", attack=1.3, defense=1.1, xg_for=1.4, xg_against=1.1),
        _Team("Cadiz", "SP1", attack=0.9, defense=1.8, xg_for=1.0, xg_against=1.7),
        _Team("Granada", "SP1", attack=0.7, defense=2.0, xg_for=0.8, xg_against=2.0),
    ]
    return {t.name: t for t in premier + laliga}


def _simulate_score(home: _Team, away: _Team) -> tuple[int, int, float, float]:
    """Simulate a score from team strengths, plus over/under 2.5 odds (decimal).

    Returns (home_goals, away_goals, over_odds, under_odds). Odds are derived
    from a simple Poisson-style heuristic: higher expected total goals ->
    lower over odds, higher under odds. Bookmaker margin baked in (~5%).
    """

    import random

    # Use a deterministic RNG seeded by team names so test outputs are stable.
    rng = random.Random(f"{home.name}-vs-{away.name}")

    # Expected goals from team strengths (clamped to a sane range).
    home_xg = max(0.2, min(4.0, (home.attack + away.defense) / 2.0))
    away_xg = max(0.2, min(4.0, (away.attack + home.defense) / 2.0))

    # Sample Poisson-ish goals.
    home_goals = min(7, max(0, round(home_xg + rng.gauss(0, 0.6))))
    away_goals = min(7, max(0, round(away_xg + rng.gauss(0, 0.6))))

    total = home_xg + away_xg
    # Market over-probability: more total goals -> higher over-prob.
    over_prob = min(0.85, max(0.15, total / 4.0))
    margin = 1.05
    over_odds = round(margin / over_prob, 2)
    under_odds = round(margin / (1.0 - over_prob), 2)
    return home_goals, away_goals, over_odds, under_odds


def _build_match_dicts() -> list[dict[str, Any]]:
    """Generate a deterministic corpus of synthetic match records.

    Layout: 2 seasons, 2 leagues, 4 teams each, each team plays every other
    team in its league twice (home/away) per season.
    """

    teams = _default_teams()
    leagues: dict[str, list[_Team]] = {"E0": [], "SP1": []}
    for team in teams.values():
        leagues[team.league].append(team)

    matches: list[dict[str, Any]] = []
    season_starts = {
        "2021-2022": datetime(2021, 8, 14),
        "2022-2023": datetime(2022, 8, 13),
    }
    matchweek = 0
    for season_label, season_start in season_starts.items():
        for league, league_teams in leagues.items():
            # Schedule: round-robin twice (home/away), 1 match every ~3 days.
            cursor = season_start
            pairings: list[tuple[_Team, _Team, bool]] = []
            for i, home in enumerate(league_teams):
                for j, away in enumerate(league_teams):
                    if i == j:
                        continue
                    pairings.append((home, away, True))  # home fixture
                    pairings.append((away, home, False))  # away fixture
            for k, (home, away, _is_home) in enumerate(pairings):
                matchweek += 1
                kickoff = cursor + timedelta(days=k * 3 // max(1, len(pairings) // 4))
                hg, ag, over_odds, under_odds = _simulate_score(home, away)
                matches.append(
                    {
                        "match_id": f"{season_label}-{league}-{k:03d}",
                        "date": kickoff,
                        "league": league,
                        "season": season_label,
                        "home_team": home.name,
                        "away_team": away.name,
                        "home_goals": hg,
                        "away_goals": ag,
                        "total_goals": hg + ag,
                        "over_2_5": int(hg + ag > 2),
                        "home_shots": home.shots,
                        "away_shots": away.shots,
                        "home_shots_on_target": home.shots_on_target,
                        "away_shots_on_target": away.shots_on_target,
                        "home_possession": home.possession,
                        "away_possession": 1.0 - home.possession,
                        "home_corners": home.corners,
                        "away_corners": away.corners,
                        "home_cards": home.cards,
                        "away_cards": away.cards,
                        "home_xg": home.xg_for,
                        "away_xg": away.xg_against,
                        "odds_over_2_5": over_odds,
                        "odds_under_2_5": under_odds,
                        "closing_odds_over_2_5": over_odds,
                        "closing_odds_under_2_5": under_odds,
                        "source": "synthetic",
                    }
                )
    matches.sort(key=lambda m: (m["date"], m["league"], m["match_id"]))
    return matches


# Pre-build once at import time for speed; tests do not mutate this list.
_SYNTHETIC_MATCH_DICTS: list[dict[str, Any]] = _build_match_dicts()


@pytest.fixture(scope="session")
def synthetic_match_dicts() -> list[dict[str, Any]]:
    """Return the deterministic synthetic match corpus as plain dicts.

    Tests that exercise the data layer convert these to ``MatchRecord`` via
    :func:`synthetic_matches`. Tests that work on plain dataframes can use
    this directly.
    """

    return list(_SYNTHETIC_MATCH_DICTS)


@pytest.fixture(scope="session")
def synthetic_matches() -> list[Any]:
    """Return the synthetic match corpus as ``MatchRecord`` instances.

    The import is lazy so the conftest is collectable even before the data
    layer is implemented.
    """

    from fussball.data.schema import MatchRecord

    return [MatchRecord.model_validate(d) for d in _SYNTHETIC_MATCH_DICTS]


@pytest.fixture
def small_match_set(synthetic_matches: list[Any]) -> list[Any]:
    """A tiny slice (one league, one season) for fast unit tests."""

    return [m for m in synthetic_matches if m.league == "E0" and m.season == "2021-2022"]


@pytest.fixture
def single_team_matches(synthetic_matches: list[Any]) -> list[Any]:
    """All matches involving Arsenal — useful for rolling-feature tests."""

    return [m for m in synthetic_matches if m.home_team == "Arsenal" or m.away_team == "Arsenal"]


@pytest.fixture
def match_factory() -> Callable[..., Any]:
    """Factory that builds a ``MatchRecord`` with sensible defaults.

    Tests use this when they need bespoke data that doesn't fit a fixture:

    .. code-block:: python

        def test_x(match_factory):
            m = match_factory(home_team="Arsenal", away_team="Chelsea")
    """

    from fussball.data.schema import MatchRecord

    base: dict[str, Any] = {
        "match_id": "test-0001",
        "date": datetime(2023, 1, 1),
        "league": "E0",
        "season": "2022-2023",
        "home_team": "Arsenal",
        "away_team": "Chelsea",
        "home_goals": 1,
        "away_goals": 1,
        "total_goals": 2,
        "over_2_5": 0,
        "home_shots": 12.0,
        "away_shots": 10.0,
        "home_shots_on_target": 5.0,
        "away_shots_on_target": 4.0,
        "home_possession": 0.55,
        "away_possession": 0.45,
        "home_corners": 6.0,
        "away_corners": 4.0,
        "home_cards": 2.0,
        "away_cards": 3.0,
        "home_xg": 1.5,
        "away_xg": 1.0,
        "odds_over_2_5": 1.95,
        "odds_under_2_5": 1.90,
        "closing_odds_over_2_5": 1.95,
        "closing_odds_under_2_5": 1.90,
        "source": "synthetic",
    }

    def _make(**overrides: Any) -> Any:
        merged = {**base, **overrides}
        return MatchRecord.model_validate(merged)

    return _make


# -----------------------------------------------------------------------------
# Filesystem fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def tmp_csv_path(tmp_path: Path) -> Path:
    """A disposable path for writing a CSV file in a test."""

    return tmp_path / "matches.csv"


@pytest.fixture
def tmp_data_dir(tmp_path: Path) -> Iterator[Path]:
    """A disposable data/ directory with raw/ and models/ subdirs."""

    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "E0_2122.csv").write_text("")  # placeholder, tests overwrite
    yield tmp_path
