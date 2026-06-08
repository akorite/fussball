"""Data layer: schemas and loaders for match records."""

from __future__ import annotations

from fussball.data.loader import (
    DataSource,
    DataValidationError,
    FreeCsvLoader,
    make_synthetic_source,
)
from fussball.data.schema import (
    DEFAULT_LEAGUES,
    LEAGUE_CODES,
    MatchRecord,
    canonical_season_label,
    parse_football_data_season,
)

__all__ = [
    "DEFAULT_LEAGUES",
    "LEAGUE_CODES",
    "DataSource",
    "DataValidationError",
    "FreeCsvLoader",
    "MatchRecord",
    "canonical_season_label",
    "make_synthetic_source",
    "parse_football_data_season",
]
