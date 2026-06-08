"""Head-to-head (H2H) features between paired teams.

For each match we look back at the last ``_WINDOW`` prior meetings between
the same two teams (direction-insensitive pairing) and compute aggregate
statistics.  Direction-aware fields (home wins, goals scored per side) are
resolved from the *current* match's home/away perspective.

Records are assumed to be pre-sorted by ``(date, league, match_id)`` by the
caller (``build_feature_matrix``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fussball.data.schema import MatchRecord

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_WINDOW = 10
"""Number of most-recent prior meetings to average over."""

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_h2h_features(records: list[MatchRecord]) -> pd.DataFrame:
    """Compute head-to-head features for each match.

    Returns DataFrame with ``match_id`` as index and columns:

    - ``h2h_total_goals_avg``: average total goals in last N meetings
    - ``h2h_over25_rate``: rate of over-2.5 goals in last N meetings
    - ``h2h_home_wins``: rate of current-home-team wins in last N meetings
    - ``h2h_draws``: rate of draws in last N meetings
    - ``h2h_goals_scored_home_avg``: avg goals scored by current home team
    - ``h2h_goals_scored_away_avg``: avg goals scored by current away team
    - ``h2h_meetings``: number of prior meetings (0 if first meeting)
    """
    if not records:
        return pd.DataFrame()

    # Build working frame --------------------------------------------------
    rows = []
    for r in records:
        pair = tuple(sorted([r.home_team, r.away_team]))
        rows.append(
            {
                "match_id": r.match_id,
                "date": r.date,
                "pair": pair,
                "home_team": r.home_team,
                "away_team": r.away_team,
                "home_goals": r.home_goals,
                "away_goals": r.away_goals,
                "total_goals": r.total_goals,
                "over_2_5": r.over_2_5,
            }
        )

    df = pd.DataFrame(rows)
    df.sort_values(["pair", "date", "match_id"], inplace=True)

    # home_is_first: True when the current match's home team is the
    # alphabetically-first member of the sorted pair.
    df["home_is_first"] = df["home_team"] == df["pair"].str[0]

    # ------------------------------------------------------------------
    # Pair-centric columns: goals from pair[0] and pair[1] perspective.
    # Computed BEFORE shift so every row is self-consistent.
    # ------------------------------------------------------------------
    _hf = df["home_is_first"]
    # pair[0]'s goals: home_goals when pair[0] was home, away_goals otherwise
    df["pair0_goals"] = np.where(_hf, df["home_goals"], df["away_goals"])
    df["pair1_goals"] = np.where(_hf, df["away_goals"], df["home_goals"])
    df["pair0_won"] = df["pair0_goals"] > df["pair1_goals"]
    grp = df.groupby("pair", sort=False)
    # Shift(1) to exclude the current match from lookback ------------------
    for col in (
        "home_goals", "away_goals", "total_goals", "over_2_5",
        "pair0_goals", "pair1_goals", "pair0_won",
    ):
        df[f"_p_{col}"] = grp[col].shift(1)

    _has_prior = df["_p_home_goals"].notna()

    # ------------------------------------------------------------------
    # Direction-neutral rolling stats
    # ------------------------------------------------------------------
    df["h2h_meetings"] = grp.cumcount()

    df["h2h_total_goals_avg"] = grp["_p_total_goals"].transform(
        lambda s: s.rolling(_WINDOW, min_periods=1).mean()
    )
    df["h2h_over25_rate"] = grp["_p_over_2_5"].transform(
        lambda s: s.rolling(_WINDOW, min_periods=1).mean()
    )

    df["_p_draw"] = np.where(
        _has_prior,
        (df["_p_home_goals"] == df["_p_away_goals"]).astype(float),
        np.nan,
    )
    df["h2h_draws"] = grp["_p_draw"].transform(
        lambda s: s.rolling(_WINDOW, min_periods=1).mean()
    )

    # ------------------------------------------------------------------
    # Direction-aware stats: compute rolling means of pair0 and pair1
    # metrics separately, then select based on current match's home team.
    # ------------------------------------------------------------------
    _roll_p0_goals = grp["_p_pair0_goals"].transform(
        lambda s: s.rolling(_WINDOW, min_periods=1).mean()
    )
    _roll_p1_goals = grp["_p_pair1_goals"].transform(
        lambda s: s.rolling(_WINDOW, min_periods=1).mean()
    )

    # pair0 won rate → pair1 win rate = 1 - pair0_won_rate - draw_rate
    # But draws are already computed direction-neutrally.  So we compute
    # pair0_wins and derive pair1_wins from it + draws.
    _roll_p0_wins = grp["_p_pair0_won"].transform(
        lambda s: s.rolling(_WINDOW, min_periods=1).mean()
    )
    _roll_draws = df["h2h_draws"]

    # pair1_wins = 1 - pair0_wins - draws (only where _has_prior)
    _roll_p1_wins = 1.0 - _roll_p0_wins - _roll_draws
    # Clamp negatives from floating point
    _roll_p1_wins = _roll_p1_wins.clip(lower=0.0)

    df["h2h_goals_scored_home_avg"] = np.where(_hf, _roll_p0_goals, _roll_p1_goals)
    df["h2h_goals_scored_away_avg"] = np.where(_hf, _roll_p1_goals, _roll_p0_goals)
    df["h2h_home_wins"] = np.where(_hf, _roll_p0_wins, _roll_p1_wins)
    # Mask first-row NaNs
    df.loc[~_has_prior, "h2h_home_wins"] = np.nan

    # Finalise: set match_id as index, keep only output columns ----------
    out_cols = [
        "match_id",
        "h2h_total_goals_avg",
        "h2h_over25_rate",
        "h2h_home_wins",
        "h2h_draws",
        "h2h_goals_scored_home_avg",
        "h2h_goals_scored_away_avg",
        "h2h_meetings",
    ]

    result = df[out_cols].copy()
    result.set_index("match_id", inplace=True)
    return result
