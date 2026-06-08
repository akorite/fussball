"""Per-match features.

These features describe a single match — rest days, league position gap,
head-to-head history, and the trivial ``is_home`` flag.

MatchRecord instances are treated as read-only (frozen).  All computation
is done in pandas DataFrames.
"""

from __future__ import annotations

from collections import defaultdict

import pandas as pd

from fussball.data.schema import MatchRecord

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_H2H_WINDOW = 3
"""Number of past head-to-head meetings to consider for ``h2h_over_rate``."""

_RECENT_VS_OPP_WINDOW = 2
"""Number of past meetings with the same opponent for ``recent_vs_opp_over_rate``."""

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_match_features(records: list[MatchRecord]) -> pd.DataFrame:
    """Compute per-match features: rest days, is_home, league position gap, h2h.

    Parameters
    ----------
    records:
        Match records, should already be sorted by ``(date, league, match_id)``
        for deterministic ordering.

    Returns
    -------
    pd.DataFrame
        Columns: ``match_id`` + all match-level feature columns.
        Returns an empty DataFrame with just a ``match_id`` column when
        ``records`` is empty.
    """
    if not records:
        return pd.DataFrame({"match_id": pd.Series(dtype=str)})

    # ---- 1. Rest days ----------------------------------------------------
    rest_df = _compute_rest_days(records)

    # ---- 2. is_home (trivial) -------------------------------------------
    is_home_df = pd.DataFrame({"match_id": [r.match_id for r in records], "is_home": 1})

    # ---- 3. League position gap ------------------------------------------
    gap_df = _compute_league_position_gap(records)

    # ---- 4. Head-to-head features ---------------------------------------
    h2h_df = _compute_h2h_features(records)

    # ---- Merge -----------------------------------------------------------
    result = rest_df.merge(is_home_df, on="match_id", how="left")
    result = result.merge(gap_df, on="match_id", how="left")
    result = result.merge(h2h_df, on="match_id", how="left")
    return result


# ---------------------------------------------------------------------------
# Internal: rest days
# ---------------------------------------------------------------------------


def _compute_rest_days(records: list[MatchRecord]) -> pd.DataFrame:
    """Days since each team's last match (NaN for first match)."""
    rows = []
    for r in records:
        rows.append({"match_id": r.match_id, "team": r.home_team, "date": r.date, "is_home": True})
        rows.append({"match_id": r.match_id, "team": r.away_team, "date": r.date, "is_home": False})

    tg = pd.DataFrame(rows)
    tg.sort_values(["team", "date", "match_id"], inplace=True)
    tg["rest_days"] = tg.groupby("team")["date"].diff().dt.days

    home = tg.loc[tg["is_home"], ["match_id", "rest_days"]].copy()
    home.rename(columns={"rest_days": "home_rest_days"}, inplace=True)

    away = tg.loc[~tg["is_home"], ["match_id", "rest_days"]].copy()
    away.rename(columns={"rest_days": "away_rest_days"}, inplace=True)

    return home.merge(away, on="match_id", how="outer")


# ---------------------------------------------------------------------------
# Internal: league position gap
# ---------------------------------------------------------------------------


def _compute_league_position_gap(records: list[MatchRecord]) -> pd.DataFrame:
    """Home position - away position in the league table at match time.

    Position is determined by cumulative points **before** the current match
    (matches are processed in date order).  NaN for a team's first match of
    the season.
    """
    # Sort once - caller already ensures this, but double-check.
    sorted_recs = sorted(records, key=lambda r: (r.date, r.league, r.match_id))

    # Running standings: {(league, season): {team: cumulative_points}}
    standings: dict[tuple[str, str], dict[str, int]] = defaultdict(dict)
    results: list[dict] = []

    for r in sorted_recs:
        key = (r.league, r.season)
        table = standings[key]

        home_pts = table.get(r.home_team, 0)
        away_pts = table.get(r.away_team, 0)

        # Determine positions (NaN if either team hasn't played before)
        if r.home_team not in table or r.away_team not in table:
            gap = None
        else:
            ranked = _rank_teams(table)
            home_pos = ranked.get(r.home_team, len(table))
            away_pos = ranked.get(r.away_team, len(table))
            gap = home_pos - away_pos

        results.append({"match_id": r.match_id, "league_position_gap": gap})

        # Update standings AFTER computing positions
        home_p = _points(r.home_goals, r.away_goals, home=True)
        away_p = _points(r.away_goals, r.home_goals, home=False)
        table[r.home_team] = home_pts + home_p
        table[r.away_team] = away_pts + away_p

    return pd.DataFrame(results)


def _rank_teams(table: dict[str, int]) -> dict[str, int]:
    """Return {team_name: 1-indexed position} sorted by points desc, name asc."""
    ranked = sorted(table.items(), key=lambda x: (-x[1], x[0]))
    return {team: pos + 1 for pos, (team, _) in enumerate(ranked)}


# ---------------------------------------------------------------------------
# Internal: head-to-head features
# ---------------------------------------------------------------------------


def _compute_h2h_features(records: list[MatchRecord]) -> pd.DataFrame:
    """Compute head-to-head over-2.5 rates.

    * ``h2h_over_rate_3`` — over-2.5 rate in last 3 meetings with this opponent
      (across all seasons).
    * ``recent_vs_opp_over_rate`` — over-2.5 rate in last 2 meetings with this
      opponent.
    """
    df = pd.DataFrame(
        [
            {
                "match_id": r.match_id,
                "pair": tuple(sorted([r.home_team, r.away_team])),
                "over_2_5": r.over_2_5,
                "date": r.date,
            }
            for r in records
        ]
    )
    df.sort_values(["pair", "date", "match_id"], inplace=True)

    df["h2h_over_rate_3"] = df.groupby("pair")["over_2_5"].transform(
        lambda s: s.shift(1).rolling(_H2H_WINDOW, min_periods=1).mean()
    )
    df["recent_vs_opp_over_rate"] = df.groupby("pair")["over_2_5"].transform(
        lambda s: s.shift(1).rolling(_RECENT_VS_OPP_WINDOW, min_periods=1).mean()
    )

    return df[["match_id", "h2h_over_rate_3", "recent_vs_opp_over_rate"]]


# ---------------------------------------------------------------------------
# Internal: points helper
# ---------------------------------------------------------------------------


def _points(goals_for: int, goals_against: int, *, home: bool) -> int:
    """Standard 3 / 1 / 0 points."""
    if goals_for > goals_against:
        return 3
    if goals_for == goals_against:
        return 1
    return 0
