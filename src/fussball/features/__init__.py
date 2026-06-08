"""Feature engineering: build feature matrix from raw ``MatchRecord`` lists.

Central entry-point is :func:`build_feature_matrix`, which chains the three
feature extractors (rolling, match-level, seasonal context) into a single
:class:`FeatureMatrix` dataclass.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from fussball.data.schema import MatchRecord
from fussball.features.context import compute_context_features
from fussball.features.match import compute_match_features
from fussball.features.odds import compute_odds_features
from fussball.features.rolling import compute_rolling_features
from fussball.features.rest import compute_rest_features
from fussball.features.derby import compute_derby_features
from fussball.features.h2h import compute_h2h_features


# ---------------------------------------------------------------------------
# FeatureMatrix
# ---------------------------------------------------------------------------


@dataclass
class FeatureMatrix:
    """Container for the feature matrix produced by :func:`build_feature_matrix`.

    Attributes
    ----------
    X:
        Feature matrix with ``match_id`` as index. Contains both identifier
        columns (date, league, season, home_team, away_team) and feature
        columns.
    y:
        Target vector (``over_2_5`` as int 0/1), index aligned with ``X``.
    feature_names:
        Column names in ``X`` that are actual features (no identifiers, no
        target).
    identifier_columns:
        Names of the identifier columns present in ``X`` (date, league,
        season, home_team, away_team). ``match_id`` is the index.
    """

    X: pd.DataFrame
    y: pd.Series
    feature_names: list[str] = field(default_factory=list)
    identifier_columns: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_IDENTIFIER_COLS = ["date", "league", "season", "home_team", "away_team"]
"""Columns that identify a match but are not model features."""

_TARGET_COL = "over_2_5"
_DROP_COLS = {"home_goals", "away_goals", "total_goals"}


def _records_to_df(records: list[MatchRecord]) -> pd.DataFrame:
    """Convert a list of MatchRecord to a flat DataFrame (quick construction)."""
    rows = []
    for r in records:
        rows.append(
            {
                "match_id": r.match_id,
                "date": r.date,
                "league": r.league,
                "season": r.season,
                "home_team": r.home_team,
                "away_team": r.away_team,
                "home_goals": r.home_goals,
                "away_goals": r.away_goals,
                "total_goals": r.total_goals,
                _TARGET_COL: r.over_2_5,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_feature_matrix(
    records: list[MatchRecord],
    hmm_history: list[MatchRecord] | None = None,
    include_hmm: bool = False,
) -> FeatureMatrix:
    """Chain all feature extractors and return a complete :class:`FeatureMatrix`.

    Parameters
    ----------
    records:
        Match records to transform.  Sorted internally by
        ``(date, league, match_id)`` for deterministic ordering.
    hmm_history:
        Historical records used to fit HMMs.  Required when
        ``include_hmm=True``; the HMMs are fit on this data only
        (time-series safe).
    include_hmm:
        If True, compute Hidden-Markov-Model features (one HMM per
        team) using ``hmm_history`` as training data.

    Returns
    -------
    FeatureMatrix
        Container with ``X``, ``y``, ``feature_names`` and
        ``identifier_columns``.
    """
    if not records:
        empty_idx = pd.Index([], dtype=str, name="match_id")
        return FeatureMatrix(
            X=pd.DataFrame(index=empty_idx),
            y=pd.Series([], dtype=int, index=empty_idx, name=_TARGET_COL),
            feature_names=[],
            identifier_columns=list(_IDENTIFIER_COLS),
        )

    # Sort once - all downstream functions can rely on this ordering.
    records_sorted = sorted(records, key=lambda r: (r.date, r.league, r.match_id))

    rolling_df = compute_rolling_features(records_sorted)
    match_df = compute_match_features(records_sorted)
    context_df = compute_context_features(records_sorted)
    odds_df = compute_odds_features(records_sorted)
    rest_df = compute_rest_features(records_sorted)
    derby_df = compute_derby_features(records_sorted)
    h2h_df = compute_h2h_features(records_sorted)

    hmm_df = None
    if include_hmm:
        if hmm_history is None:
            hmm_history = records_sorted
        from fussball.hmm import HMMFeatureExtractor  # local import to avoid cycle

        extractor = HMMFeatureExtractor()
        hmm_df = extractor.transform_aggregate(records_sorted, hmm_history)

    # Build base frame (match_id + identifiers + target)
    base = _records_to_df(records_sorted)

    for feat_df in [rolling_df, match_df, context_df, odds_df, rest_df, derby_df, h2h_df, hmm_df]:
        if feat_df is not None and not feat_df.empty:
            base = base.merge(feat_df, on="match_id", how="left")

    # Identify feature columns (everything except identifiers, target, raw scores, match_id)
    exclude = set(_IDENTIFIER_COLS) | {_TARGET_COL} | _DROP_COLS | {"match_id"}
    feature_cols = [c for c in base.columns if c not in exclude]

    X = base.set_index("match_id")
    y = X.pop(_TARGET_COL).astype(int)

    # Drop raw score columns from X
    for col in _DROP_COLS:
        if col in X.columns:
            X.drop(columns=[col], inplace=True)

    # Only keep identifier columns that actually ended up in X
    id_cols_in_X = [c for c in _IDENTIFIER_COLS if c in X.columns]

    return FeatureMatrix(
        X=X,
        y=y,
        feature_names=feature_cols,
        identifier_columns=id_cols_in_X,
    )


def select_xy(matrix: FeatureMatrix) -> tuple[pd.DataFrame, pd.Series]:
    """Drop identifier columns from ``matrix.X``, return ``(X_features, y)``.

    This is a convenience for model training — identifiers are kept in the
    :class:`FeatureMatrix` for backtest traceability but should not be used
    as model features.
    """
    id_cols = [c for c in matrix.identifier_columns if c in matrix.X.columns]
    X_feat = matrix.X.drop(columns=id_cols)
    return X_feat, matrix.y


__all__ = [
    "FeatureMatrix",
    "build_feature_matrix",
    "select_xy",
]
