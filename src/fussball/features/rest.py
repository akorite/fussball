"""Rest day and schedule congestion features.

For each match we compute how many days have elapsed since each team's last
appearance and how many matches each team played in the preceding 30-day
window.  These features capture fatigue / freshness effects that influence
match outcomes.

MatchRecord instances are treated as read-only.  All computation is done
in pandas DataFrames.
"""

from __future__ import annotations

from collections import defaultdict

import pandas as pd

from fussball.data.schema import MatchRecord

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_rest_features(records: list[MatchRecord]) -> pd.DataFrame:
    """Compute rest day and schedule congestion features.

    Parameters
    ----------
    records:
        Match records **sorted by date** (ascending).  Each record must have
        at least ``match_id``, ``date``, ``home_team``, ``away_team``.
    Returns
    -------
    pd.DataFrame
        DataFrame indexed by ``match_id`` with columns:

        - ``home_games_last_30d`` – home team matches in the last 30 days
        - ``away_games_last_30d`` – away team matches in the last 30 days
        - ``rest_advantage`` – difference in rest days (home − away)
        - ``combined_congestion`` – ``home_games_last_30d + away_games_last_30d``

    .. note::

       ``home_rest_days`` and ``away_rest_days`` are already provided by
       :mod:`fussball.features.match` and are **not** duplicated here to
       avoid merge collisions.
    """
    if not records:
        return pd.DataFrame(
            columns=[
                "home_games_last_30d",
                "away_games_last_30d",
                "rest_advantage",
                "combined_congestion",
            ]
        ).rename_axis("match_id")

    # Track per-team match dates (in chronological order thanks to input sort).
    team_dates: dict[str, list[pd.Timestamp]] = defaultdict(list)

    rows: list[dict] = []

    for rec in records:
        match_date = pd.Timestamp(rec.date)

        # --- rest days (days since last match for each team) ---
        home_dates = team_dates[rec.home_team]
        away_dates = team_dates[rec.away_team]

        home_rest = (
            (match_date - home_dates[-1]).days
            if home_dates
            else pd.NA
        )
        away_rest = (
            (match_date - away_dates[-1]).days
            if away_dates
            else pd.NA
        )

        # --- games in last 30 days ---
        cutoff_30 = match_date - pd.Timedelta(days=30)

        home_last_30 = sum(
            1 for d in home_dates if cutoff_30 < d < match_date
        )
        away_last_30 = sum(
            1 for d in away_dates if cutoff_30 < d < match_date
        )

        rest_adv = home_rest - away_rest if pd.notna(home_rest) and pd.notna(away_rest) else float("nan")
        combined = home_last_30 + away_last_30

        rows.append(
            {
                "match_id": rec.match_id,
                "home_games_last_30d": home_last_30,
                "away_games_last_30d": away_last_30,
                "rest_advantage": rest_adv,
                "combined_congestion": combined,
            }
        )

        # Register this match for both teams (a team may appear as home or away).
        team_dates[rec.home_team].append(match_date)
        team_dates[rec.away_team].append(match_date)

    df = pd.DataFrame(rows).set_index("match_id")

    col_order = [
        "home_games_last_30d",
        "away_games_last_30d",
        "rest_advantage",
        "combined_congestion",
    ]
    return df[col_order]
