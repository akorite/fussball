"""Market odds features.

The closing market odds for the over-2.5 line are the strongest single
predictor of the true over-2.5 probability (a sharp bettor's pooled
estimate). Adding the implied probability as a feature gives the model
a strong prior to deviate from only when the data supports it.
"""

from __future__ import annotations

import math

import pandas as pd

from fussball.data.schema import MatchRecord


def compute_odds_features(records: list[MatchRecord]) -> pd.DataFrame:
    """Compute market-odds-based features for each match.

    Parameters
    ----------
    records:
        Match records (any order; will be sorted internally).

    Returns
    -------
    pd.DataFrame
        Columns: ``match_id`` plus:
        ``market_implied_prob`` (raw 1/odds),
        ``market_overround`` (1/over_odds + 1/under_odds, NaN if either is missing),
        ``market_log_odds`` (logit of implied prob, clipped),
        ``has_odds`` (0/1 flag for missing odds).
    """
    if not records:
        return pd.DataFrame({"match_id": pd.Series(dtype=str)})

    rows = []
    for r in records:
        over_odds = r.closing_odds_over_2_5
        under_odds = r.closing_odds_under_2_5
        has_odds = over_odds is not None and over_odds > 1.0

        if has_odds and over_odds is not None:
            market_p = 1.0 / over_odds
            if under_odds is not None and under_odds > 1.0:
                overround = 1.0 / over_odds + 1.0 / under_odds
            else:
                overround = float("nan")
            log_odds = _logit(market_p)
        else:
            market_p = float("nan")
            overround = float("nan")
            log_odds = float("nan")

        rows.append(
            {
                "match_id": r.match_id,
                "market_implied_prob": market_p,
                "market_overround": overround,
                "market_log_odds": log_odds,
                "has_odds": int(has_odds),
            }
        )
    return pd.DataFrame(rows)


def _logit(p: float) -> float:
    """Log-odds (logit), clipped to avoid inf."""
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))
