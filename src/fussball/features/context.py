"""Seasonal context features.

Produces matchweek-normalised, season-stage and league-phase indicators
that give the model information about *where* in a season a match falls.
"""

from __future__ import annotations

from collections import defaultdict

import pandas as pd

from fussball.data.schema import MatchRecord

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_SEASON_STAGE_BINS = [-0.001, 0.333, 0.667, 1.001]
_SEASON_STAGE_LABELS = [0, 1, 2]

_TITLE_RACE_CUTOFF = 4  # top N teams → title race (phase 0)
_RELEGATION_CUTOFF = 5  # bottom N teams → relegation battle (phase 2)

_PHASE_COLS = 3  # number of one-hot columns per phase (0, 1, 2)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_context_features(records: list[MatchRecord]) -> pd.DataFrame:
    """Compute seasonal context features for each match.

    Parameters
    ----------
    records:
        Match records, should already be sorted by ``(date, league, match_id)``
        for deterministic ordering.

    Returns
    -------
    pd.DataFrame
        Columns: ``match_id`` plus all context feature columns:
        ``matchweek_normalized``, ``season_stage_0/1/2``,
        ``league_phase_home_0/1/2``, ``league_phase_away_0/1/2``.
        Returns an empty DataFrame with just a ``match_id`` column when
        ``records`` is empty.
    """
    if not records:
        return pd.DataFrame({"match_id": pd.Series(dtype=str)})

    df = _records_to_df_simple(records)
    df.sort_values(["league", "season", "date", "match_id"], inplace=True)

    # ---- 1. Matchweek normalised -----------------------------------------
    df["matchweek_index"] = df.groupby(["league", "season"])["date"].transform(
        lambda dates: pd.factorize(dates)[0] + 1
    )
    df["total_matchweeks"] = df.groupby(["league", "season"])["date"].transform("nunique")
    df["matchweek_normalized"] = df["matchweek_index"] / df["total_matchweeks"]

    # ---- 2. Season stage (ordinal, one-hot encoded) ----------------------
    df["season_stage"] = pd.cut(
        df["matchweek_normalized"],
        bins=_SEASON_STAGE_BINS,
        labels=_SEASON_STAGE_LABELS,
        right=True,
    ).astype(int)
    # ---- 3. League phase -------------------------------------------------
    positions_df = _compute_team_positions(records)
    df = df.merge(positions_df, on="match_id", how="left")

    df["league_phase_home"] = _phase_from_position(df["home_position"], df["n_teams"])
    df["league_phase_away"] = _phase_from_position(df["away_position"], df["n_teams"])

    # Build one-hot columns manually using fillna to avoid NA in boolean
    # comparisons (pd.NA == 0 returns pd.NA, breaking .astype(int)).
    for prefix, col in (
        ("season_stage", "season_stage"),
        ("league_phase_home", "league_phase_home"),
        ("league_phase_away", "league_phase_away"),
    ):
        for i in _SEASON_STAGE_LABELS:
            df[f"{prefix}_{i}"] = (df[col].fillna(-1) == i).astype(int)

    # ---- Assemble result --------------------------------------------------
    result = df[["match_id", "matchweek_normalized"]].copy()
    result = pd.concat(
        [
            result,
            df[[f"season_stage_{i}" for i in _SEASON_STAGE_LABELS]],
            df[[f"league_phase_home_{i}" for i in _SEASON_STAGE_LABELS]],
            df[[f"league_phase_away_{i}" for i in _SEASON_STAGE_LABELS]],
        ],
        axis=1,
    )
    return result


# ---------------------------------------------------------------------------
# Internal: team positions at match time
# ---------------------------------------------------------------------------


def _compute_team_positions(
    records: list[MatchRecord],
) -> pd.DataFrame:
    """Compute 1-indexed league position for home/away team at match time.

    Returns a DataFrame with ``match_id``, ``home_position``, ``away_position``
    and ``n_teams`` (total teams in that league+season). Position is NaN
    for a team's first match of the season.
    """
    sorted_recs = sorted(records, key=lambda r: (r.date, r.league, r.match_id))
    standings: dict[tuple[str, str], dict[str, int]] = defaultdict(dict)
    team_counts: dict[tuple[str, str], int] = {}
    results: list[dict] = []

    for r in sorted_recs:
        key = (r.league, r.season)
        table = standings[key]

        # Count unique teams in this league+season on first sighting
        if key not in team_counts:
            all_teams = set()
            for rec in records:
                if rec.league == r.league and rec.season == r.season:
                    all_teams.add(rec.home_team)
                    all_teams.add(rec.away_team)
            team_counts[key] = len(all_teams)

        home_pts = table.get(r.home_team, 0)
        away_pts = table.get(r.away_team, 0)

        if r.home_team not in table or r.away_team not in table:
            home_pos = None
            away_pos = None
        else:
            ranked = sorted(table.items(), key=lambda x: (-x[1], x[0]))
            pos_map = {team: i + 1 for i, (team, _) in enumerate(ranked)}
            home_pos = pos_map.get(r.home_team)
            away_pos = pos_map.get(r.away_team)

        results.append(
            {
                "match_id": r.match_id,
                "home_position": home_pos,
                "away_position": away_pos,
                "n_teams": team_counts[key],
            }
        )

        # Update standings AFTER recording positions
        home_p = _points(r.home_goals, r.away_goals)
        away_p = _points(r.away_goals, r.home_goals)
        table[r.home_team] = home_pts + home_p
        table[r.away_team] = away_pts + away_p

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Internal: phase mapping
# ---------------------------------------------------------------------------


def _phase_from_position(
    position: pd.Series,
    n_teams: pd.Series,
) -> pd.Series:
    """Map 1-indexed positions to league-phase ordinals.

    * 0 = title race (top ``_TITLE_RACE_CUTOFF`` teams)
    * 2 = relegation battle (bottom ``_RELEGATION_CUTOFF`` teams)
    * 1 = mid-table (everything else)

    Returns NaN for null positions.
    """
    result = pd.Series(1, index=position.index, dtype="Int64")
    # Title race: position <= _TITLE_RACE_CUTOFF (top 4)
    result.loc[position <= _TITLE_RACE_CUTOFF] = 0
    # Relegation: position > n_teams - _RELEGATION_CUTOFF
    # Only meaningful when n_teams > _RELEGATION_CUTOFF.
    relegation_threshold: pd.Series | int
    if n_teams is not None:
        relegation_threshold = n_teams - _RELEGATION_CUTOFF
        result.loc[(position > relegation_threshold) & (position > _TITLE_RACE_CUTOFF)] = 2
    result[position.isna()] = pd.NA
    return result


def _points(goals_for: int, goals_against: int) -> int:
    """Standard 3 / 1 / 0 points."""
    if goals_for > goals_against:
        return 3
    if goals_for == goals_against:
        return 1
    return 0


def _records_to_df_simple(records: list[MatchRecord]) -> pd.DataFrame:
    """Minimal records → DataFrame for context computation."""
    return pd.DataFrame(
        [
            {
                "match_id": r.match_id,
                "date": r.date,
                "league": r.league,
                "season": r.season,
                "home_team": r.home_team,
                "away_team": r.away_team,
                "home_goals": r.home_goals,
                "away_goals": r.away_goals,
            }
            for r in records
        ]
    )
