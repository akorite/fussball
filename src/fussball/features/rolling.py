"""Per-team rolling average features.

For each match, we compute rolling averages over the **previous** 5, 10 and 20
matches *for each team independently*. Both the home-team and away-team
perspectives are produced and joined back to the match-level row.

MatchRecord instances are treated as read-only (frozen).  All computation
is done in pandas DataFrames.
"""

from __future__ import annotations

import pandas as pd

from fussball.data.schema import MatchRecord

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_WINDOWS = [5, 10, 20]
"""Rolling window sizes."""

_STATS: dict[str, str] = {
    "goals_scored": "goals_scored_avg",
    "goals_conceded": "goals_conceded_avg",
    "shots_diff": "shots_diff_avg",
    "points": "ppg_avg",
    "xg_diff": "xg_diff_avg",
}
"""Mapping from raw stat column name → feature column prefix."""

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_rolling_features(records: list[MatchRecord]) -> pd.DataFrame:
    """Compute per-team rolling averages for each match.

    Parameters
    ----------
    records:
        Match records, should already be sorted by ``(date, league, match_id)``
        for deterministic ordering.

    Returns
    -------
    pd.DataFrame
        Columns: ``match_id`` + ``{home,away}_{stat_prefix}_{window}``.
        Returns an empty DataFrame with just a ``match_id`` column when
        ``records`` is empty.
    """
    if not records:
        return pd.DataFrame({"match_id": pd.Series(dtype=str)})

    # -- Build team-game perspective ---------------------------------------
    rows = []
    for r in records:
        home_pts = _points(r.home_goals, r.away_goals, home=True)
        away_pts = _points(r.away_goals, r.home_goals, home=False)

        rows.append(
            {
                "match_id": r.match_id,
                "team": r.home_team,
                "date": r.date,
                "goals_scored": r.home_goals,
                "goals_conceded": r.away_goals,
                "shots_for": r.home_shots,
                "shots_against": r.away_shots,
                "points": home_pts,
                "xg_for": r.home_xg,
                "xg_against": r.away_xg,
                "is_home": True,
            }
        )
        rows.append(
            {
                "match_id": r.match_id,
                "team": r.away_team,
                "date": r.date,
                "goals_scored": r.away_goals,
                "goals_conceded": r.home_goals,
                "shots_for": r.away_shots,
                "shots_against": r.home_shots,
                "points": away_pts,
                "xg_for": r.away_xg,
                "xg_against": r.home_xg,
                "is_home": False,
            }
        )

    tg = pd.DataFrame(rows)

    # Derived columns (may introduce NaN when inputs are missing — intentional)
    tg["shots_diff"] = tg["shots_for"] - tg["shots_against"]
    tg["xg_diff"] = tg["xg_for"] - tg["xg_against"]

    # Sort per team for stable rolling windows
    tg.sort_values(["team", "date", "match_id"], inplace=True)

    # -- Rolling averages ---------------------------------------------------
    # Use default argument binding to avoid closure issues in the loop.
    for w in _WINDOWS:
        for stat, col_prefix in _STATS.items():
            col_name = f"{col_prefix}_{w}"
            tg[col_name] = tg.groupby("team")[stat].transform(
                lambda s, win=w: s.shift(1).rolling(win, min_periods=1).mean()
            )

    # -- Pivot home / away ---------------------------------------------------
    all_feat_cols = [f"{p}_{w}" for w in _WINDOWS for p in _STATS.values()]

    home = tg.loc[tg["is_home"], ["match_id", *all_feat_cols]].copy()
    home.rename(columns={c: f"home_{c}" for c in all_feat_cols}, inplace=True)

    away = tg.loc[~tg["is_home"], ["match_id", *all_feat_cols]].copy()
    away.rename(columns={c: f"away_{c}" for c in all_feat_cols}, inplace=True)

    result = home.merge(away, on="match_id", how="outer")
    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _points(goals_for: int, goals_against: int, *, home: bool) -> int:
    """Standard 3 / 1 / 0 points."""
    if goals_for > goals_against:
        return 3
    if goals_for == goals_against:
        return 1
    return 0
