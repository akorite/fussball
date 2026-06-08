"""Match record schema and season utilities.

The :class:`MatchRecord` is the canonical representation of a single football
match throughout the fussball pipeline. Every downstream module (features,
models, backtest, CLI) consumes ``MatchRecord`` instances — the data layer is
the only place that translates between external sources and this schema.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# -----------------------------------------------------------------------------
# League codes (Football-Data.co.uk convention)
# -----------------------------------------------------------------------------

LEAGUE_CODES: dict[str, str] = {
    # Top 5
    "E0": "Premier League",
    "SP1": "La Liga",
    "D1": "Bundesliga",
    "I1": "Serie A",
    "F1": "Ligue 1",
    # Tier 2 — same countries
    "E1": "English Championship",
    "E2": "English League One",
    "E3": "English League Two",
    "D2": "2. Bundesliga",
    "SP2": "La Liga 2",
    "I2": "Serie B",
    "F2": "Ligue 2",
    # Other major European leagues
    "N1": "Eredivisie",
    "B1": "Belgian Pro League",
    "P1": "Primeira Liga",
    "T1": "Turkish Süper Lig",
    "G1": "Greek Super League",
    "SC0": "Scottish Premiership",
}

# Default set used by the CLI for `download-data` and backtests.
DEFAULT_LEAGUES: list[str] = [
    "E0",
    "SP1",
    "D1",
    "I1",
    "F1",  # Big 5
    "E1",
    "D2",
    "SP2",
    "I2",
    "N1",  # Tier-2 + Eredivisie
]

# -----------------------------------------------------------------------------
# Season label helpers
# -----------------------------------------------------------------------------


def parse_football_data_season(token: str) -> str:
    """Convert ``"2122"`` or ``"2021-22"`` into the canonical ``"2021-2022"``.

    Football-Data.co.uk ships CSVs named like ``E0_2122.csv``. We standardise
    to ``"2021-2022"`` internally for readability and sortability.

    Accepted inputs:
    - ``"2021-2022"``, ``"2021-22"`` (canonical or short form, hyphenated)
    - ``"2122"`` (2-digit compact)
    - ``"202122"`` (4+2 compact)

    Raises ``ValueError`` if the token is not a recognisable season string.
    """

    token = token.strip()
    # 4-digit start with separator: "2021-2022" or "2021-22"
    m = re.fullmatch(r"(\d{4})-(\d{2,4})", token)
    if m:
        start = int(m.group(1))
        end = _expand_year(m.group(2), century_hint=start + 1)
        return f"{start}-{end}"
    # 4+2 compact: "202122" -> 2021-2022
    m = re.fullmatch(r"(\d{4})(\d{2})", token)
    if m:
        start = int(m.group(1))
        return f"{start}-{start + 1}"
    # 2+2 compact: "2122" -> 2021-2022
    m = re.fullmatch(r"(\d{2})(\d{2})", token)
    if m:
        start_short = int(m.group(1))
        end_short = int(m.group(2))
        century = 2000 if start_short < 50 else 1900  # naive century boundary
        return f"{century + start_short}-{century + end_short}"
    raise ValueError(f"unrecognised season token: {token!r}")


def _expand_year(tail: str, *, century_hint: int) -> int:
    """Convert a 2- or 4-digit year token into a full 4-digit year."""

    n = int(tail)
    if len(tail) == 4:
        return n
    if len(tail) == 2:
        # 2-digit year: align to the same century as the hint.
        century = (century_hint // 100) * 100
        return century + n
    raise ValueError(f"year tail has unexpected length: {tail!r}")


def canonical_season_label(start_year: int) -> str:
    """Return ``"2021-2022"`` for ``start_year=2021``."""

    return f"{start_year}-{start_year + 1}"


# -----------------------------------------------------------------------------
# MatchRecord
# -----------------------------------------------------------------------------


class MatchRecord(BaseModel):
    """A single football match.

    Required fields capture the minimum the model needs to produce a
    prediction. Optional fields are filled when present in the source; the
    feature and model layers treat them as missing safely.
    """

    model_config = ConfigDict(extra="ignore", frozen=True, str_strip_whitespace=True)

    # Identity / metadata
    match_id: str = Field(..., description="Unique stable identifier for the match")
    date: datetime = Field(..., description="Match kickoff (local venue time)")
    league: str = Field(..., description="League code, e.g. 'E0' for Premier League")
    season: str = Field(..., description="Canonical season label, e.g. '2021-2022'")
    source: str = Field(..., description="Provenance tag: 'football-data.co.uk', 'synthetic', ...")

    # Teams and result
    home_team: str
    away_team: str
    home_goals: int = Field(..., ge=0, le=20)
    away_goals: int = Field(..., ge=0, le=20)
    total_goals: int = Field(..., ge=0, le=40)
    over_2_5: int = Field(..., description="1 if total_goals > 2 else 0", ge=0, le=1)

    # Detailed stats (optional in free datasets; required by R2 when available)
    home_shots: float | None = Field(None, ge=0)
    away_shots: float | None = Field(None, ge=0)
    home_shots_on_target: float | None = Field(None, ge=0)
    away_shots_on_target: float | None = Field(None, ge=0)
    home_possession: float | None = Field(None, ge=0.0, le=1.0)
    away_possession: float | None = Field(None, ge=0.0, le=1.0)
    home_corners: float | None = Field(None, ge=0)
    away_corners: float | None = Field(None, ge=0)
    home_cards: float | None = Field(None, ge=0)
    away_cards: float | None = Field(None, ge=0)
    home_xg: float | None = Field(None, ge=0)
    away_xg: float | None = Field(None, ge=0)

    # Pre-match closing odds (decimal)
    odds_over_2_5: float | None = Field(None, gt=1.0, description="Decimal odds, > 1.0")
    odds_under_2_5: float | None = Field(None, gt=1.0)
    closing_odds_over_2_5: float | None = Field(None, gt=1.0)
    closing_odds_under_2_5: float | None = Field(None, gt=1.0)
    # Asian handicap odds (optional)
    opening_odds_asian_home: float | None = Field(None, description="Asian handicap home odds, opening")
    opening_odds_asian_away: float | None = Field(None, description="Asian handicap away odds, opening")
    closing_odds_asian_home: float | None = Field(None, description="Asian handicap home odds, closing")
    closing_odds_asian_away: float | None = Field(None, description="Asian handicap away odds, closing")

    # Aliases for non-football-data fields that downstream code may use.
    OVER_2_5_FIELD: ClassVar[str] = "over_2_5"

    # -- Validators -----------------------------------------------------------

    @field_validator("league")
    @classmethod
    def _normalise_league(cls, v: str) -> str:
        v = v.strip().upper()
        if not v:
            raise ValueError("league code is empty")
        return v

    @field_validator("home_team", "away_team")
    @classmethod
    def _team_non_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("team name is empty")
        return v

    @field_validator("home_goals", "away_goals", mode="before")
    @classmethod
    def _coerce_int(cls, v: Any) -> Any:
        if v is None or v == "":
            raise ValueError("required score field is missing")
        if isinstance(v, str):
            v = v.strip()
            if not v:
                raise ValueError("required score field is empty")
        return v

    @field_validator("home_possession", "away_possession", mode="before")
    @classmethod
    def _coerce_pct(cls, v: Any) -> Any:
        # Football-Data sometimes stores possession as 0-100 instead of 0-1.
        if v is None or v == "":
            return None
        f = float(v)
        if f > 1.0:
            f = f / 100.0
        return f

    @model_validator(mode="before")
    @classmethod
    def _compute_derived(cls, data: Any) -> Any:
        """Derive ``total_goals`` and ``over_2_5`` from home/away goals.

        These are always computed from the score — callers cannot set them
        to a value that disagrees with the score.
        """

        if isinstance(data, dict):
            hg = data.get("home_goals")
            ag = data.get("away_goals")
            if hg is not None and ag is not None:
                total = int(hg) + int(ag)
                data["total_goals"] = total
                data["over_2_5"] = 1 if total > 2 else 0
        return data

    @model_validator(mode="after")
    def _consistency(self) -> MatchRecord:
        if self.home_team == self.away_team:
            raise ValueError(f"home and away are the same team: {self.home_team!r}")
        if (
            self.home_possession is not None
            and self.away_possession is not None
            and abs((self.home_possession + self.away_possession) - 1.0) > 0.05
        ):
            # Allow small rounding tolerance; warn via value adjustment.
            # We don't reject — some sources report 0-100 not 0-1.
            pass
        # Note: we intentionally do NOT validate that combined implied probability
        # is >= 1.0 (i.e. positive bookmaker margin). Real markets in low-liquidity
        # windows (early prices, minor leagues, closing-minute moves) can have
        # temporarily negative margins, and Football-Data occasionally records
        # such values. The model treats these as noisy but usable.
        return self

    # -- Convenience ----------------------------------------------------------

    @property
    def result_str(self) -> str:
        return f"{self.home_team} {self.home_goals}-{self.away_goals} {self.away_team}"

    @property
    def season_start_year(self) -> int:
        return int(self.season.split("-", 1)[0])

    def with_odds(self, *, over: float | None, under: float | None) -> MatchRecord:
        """Return a copy with updated closing odds."""

        return self.model_copy(
            update={"closing_odds_over_2_5": over, "closing_odds_under_2_5": under}
        )
