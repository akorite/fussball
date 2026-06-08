"""Data loaders and the swappable :class:`DataSource` protocol.

The data layer is the only place that touches external sources. Downstream
modules speak :class:`~fussball.data.schema.MatchRecord` and don't care
whether the records came from a free CSV, a paid API, or a synthetic
generator. A future paid-API implementation only needs to implement the
:class:`DataSource` interface.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol, runtime_checkable

import pandas as pd

from fussball.data.schema import MatchRecord, parse_football_data_season

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Exceptions
# -----------------------------------------------------------------------------


class DataValidationError(ValueError):
    """Raised when match data fails schema or business-rule validation."""


# -----------------------------------------------------------------------------
# DataSource protocol
# -----------------------------------------------------------------------------


@runtime_checkable
class DataSource(Protocol):
    """Abstract data backend. The rest of the pipeline speaks this interface."""

    def load_seasons(
        self,
        leagues: Sequence[str],
        seasons: Sequence[str],
    ) -> list[MatchRecord]:
        """Return all match records for the given (leagues x seasons) grid.

        Parameters
        ----------
        leagues:
            League codes, e.g. ``["E0", "SP1"]``. Empty list means all known.
        seasons:
            Season labels in any format accepted by
            :func:`~fussball.data.schema.parse_football_data_season`.
            Empty list means all available.
        """

    def available_seasons(self, league: str) -> list[str]:
        """Return the canonical season labels available for ``league``."""


# -----------------------------------------------------------------------------
# Football-Data.co.uk column mapping
# -----------------------------------------------------------------------------


# Map our canonical field -> list of possible CSV column names (most recent first).
# Older CSVs used different bookmaker-specific column names; we look them up
# in order of preference.
_COL_MAP: dict[str, tuple[str, ...]] = {
    "home_goals": ("FTHG",),
    "away_goals": ("FTAG",),
    "home_shots": ("HS",),
    "away_shots": ("AS",),
    "home_shots_on_target": ("HST",),
    "away_shots_on_target": ("AST",),
    "home_possession": ("HP", "HomePoss"),
    "away_possession": ("AP", "AwayPoss"),
    "home_corners": ("HC",),
    "away_corners": ("AC",),
    "home_cards": ("HY", "HR"),  # yellow + red summed
    "away_cards": ("AY", "AR"),
    "home_xg": ("HxG", "xG_Home"),
    "away_xg": ("AxG", "xG_Away"),
    "opening_odds_asian_home": ("B365AHH", "MaxAHH"),
    "opening_odds_asian_away": ("B365AHA", "MaxAHA"),
    "closing_odds_asian_home": ("B365CAHH", "MaxCAHH", "B365AHH", "MaxAHH"),
    "closing_odds_asian_away": ("B365CAHA", "MaxCAHA", "B365AHA", "MaxAHA"),
    "odds_over_2_5": ("Max>2.5", "B365>2.5", "P>2.5", "BbAv>2.5", "Avg>2.5"),
    "odds_under_2_5": ("Max<2.5", "B365<2.5", "P<2.5", "BbAv<2.5", "Avg<2.5"),
    "closing_odds_over_2_5": ("MaxC>2.5", "B365C>2.5", "AvgC>2.5", "P>2.5", "BbAv>2.5", "Avg>2.5", "Max>2.5"),
    "closing_odds_under_2_5": ("MaxC<2.5", "B365C<2.5", "AvgC<2.5", "P<2.5", "BbAv<2.5", "Avg<2.5", "Max<2.5"),
}


def _resolve_col(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    """Return the first column from ``candidates`` that exists in ``df``."""

    for col in candidates:
        if col in df.columns:
            return col
    return None


# Fields that represent decimal bookmaker odds (must be >= 1.0).
_ODDS_FIELDS: frozenset[str] = frozenset(
    {
        "odds_over_2_5",
        "odds_under_2_5",
        "closing_odds_over_2_5",
        "closing_odds_under_2_5",
        "opening_odds_asian_home",
        "opening_odds_asian_away",
        "closing_odds_asian_home",
        "closing_odds_asian_away",
    }
)


# -----------------------------------------------------------------------------
# FreeCsvLoader
# -----------------------------------------------------------------------------


# Filename convention: <league>_<season>.csv  e.g. E0_2122.csv, SP1_2021-2022.csv
_FILENAME_RE = re.compile(
    r"^(?P<league>[A-Za-z0-9]+)_(?P<season>[A-Za-z0-9-]+)\.csv$", re.IGNORECASE
)


@dataclass
class FreeCsvLoader:
    """Loads match data from Football-Data.co.uk-style CSV files on disk.

    Parameters
    ----------
    data_dir:
        Directory containing CSVs named ``<league>_<season>.csv``. Defaults
        to ``data/raw`` relative to the project root.
    """

    data_dir: Path

    def __init__(self, data_dir: Path | str | None = None) -> None:
        if data_dir is None:
            # Default: <project_root>/data/raw
            data_dir = Path.cwd() / "data" / "raw"
        self.data_dir = Path(data_dir)

    # -- public API -----------------------------------------------------------

    def load_seasons(
        self,
        leagues: Sequence[str],
        seasons: Sequence[str],
    ) -> list[MatchRecord]:
        """Load all match records for the requested (leagues x seasons) grid.

        Unknown leagues/seasons are logged and skipped — they don't raise.
        Missing optional columns are logged at WARNING once per (league, season).
        """

        records: list[MatchRecord] = []
        canonical_seasons = [parse_football_data_season(s) for s in seasons] if seasons else None
        for path in self._iter_csvs(leagues, canonical_seasons):
            league, season = self._parse_filename(path)
            try:
                df = self._read_csv_with_encoding_fallback(path)
            except FileNotFoundError as e:
                raise FileNotFoundError(f"CSV file disappeared during read: {path}") from e
            except pd.errors.EmptyDataError:
                logger.warning("empty CSV at %s; skipping", path)
                continue
            except Exception as e:
                raise DataValidationError(f"failed to read CSV {path}: {e}") from e

            records.extend(self._df_to_records(df, league, season, path))
        records.sort(key=lambda r: (r.date, r.league, r.match_id))
        return records

    def available_seasons(self, league: str) -> list[str]:
        out: list[str] = []
        for path in self.data_dir.glob(f"{league}_*.csv"):
            m = _FILENAME_RE.match(path.name)
            if not m:
                continue
            try:
                out.append(parse_football_data_season(m.group("season")))
            except ValueError:
                continue
        return sorted(set(out))

    def download(
        self,
        leagues: list[str] | None = None,
        seasons: list[str] | None = None,
        force: bool = False,
    ) -> list[Path]:
        """Download CSV files from Football-Data.co.uk.

        For each (league, season) pair the method constructs a URL of the form
        ``https://www.football-data.co.uk/mmz4281/{season_compact}/{league}.csv``
        and saves the response to ``{self.data_dir}/{league}_{season_compact}.csv``.

        Parameters
        ----------
        leagues:
            League codes to download (e.g. ``["E0", "SP1"]``).  If ``None``,
            uses the top 5 leagues.
        seasons:
            Season labels in any format accepted by
            :func:`~fussball.data.schema.parse_football_data_season`.
            If ``None``, uses the last 5 seasons.
        force:
            Re-download even if the local file already exists.

        Returns
        -------
        list[Path]
            Paths of successfully downloaded files.
        """
        import requests

        if leagues is None:
            leagues = ["E0", "SP1", "D1", "I1", "F1"]
        if seasons is None:
            from datetime import date

            today = date.today()
            end_year = today.year if today.month >= 8 else today.year - 1
            seasons = [f"{y}-{y + 1}" for y in range(end_year - 4, end_year + 1)]

        self.data_dir.mkdir(parents=True, exist_ok=True)
        downloaded: list[Path] = []

        for season in seasons:
            canonical = parse_football_data_season(season)
            start_year, end_year = canonical.split("-")
            compact = start_year[2:] + end_year[2:]  # "2021-2022" -> "2122"
            for league in leagues:
                filename = f"{league}_{compact}.csv"
                dest = self.data_dir / filename
                if dest.exists() and not force:
                    logger.debug("Skipping %s (already exists)", filename)
                    continue
                url = f"https://www.football-data.co.uk/mmz4281/{compact}/{league}.csv"
                try:
                    resp = requests.get(url, timeout=30)
                    resp.raise_for_status()
                    dest.write_bytes(resp.content)
                    downloaded.append(dest)
                    logger.info("Downloaded %s -> %s", url, dest)
                except requests.RequestException as e:
                    logger.warning("Failed to download %s: %s", url, e)
                    continue

        return downloaded

    # -- internals ------------------------------------------------------------

    def _iter_csvs(
        self, leagues: Sequence[str], canonical_seasons: list[str] | None
    ) -> Iterable[Path]:
        if not self.data_dir.exists():
            raise FileNotFoundError(f"data directory does not exist: {self.data_dir}")
        league_filter = {lg.upper() for lg in leagues} if leagues else None
        season_filter = set(canonical_seasons) if canonical_seasons else None
        for path in sorted(self.data_dir.glob("*.csv")):
            m = _FILENAME_RE.match(path.name)
            if not m:
                logger.debug("skipping non-conforming file: %s", path.name)
                continue
            league = m.group("league").upper()
            try:
                season = parse_football_data_season(m.group("season"))
            except ValueError:
                logger.warning("cannot parse season from filename: %s", path.name)
                continue
            if league_filter and league not in league_filter:
                continue
            if season_filter and season not in season_filter:
                continue
            yield path

    @staticmethod
    def _parse_filename(path: Path) -> tuple[str, str]:
        m = _FILENAME_RE.match(path.name)
        if not m:
            raise DataValidationError(f"filename does not match <league>_<season>.csv: {path.name}")
        return m.group("league").upper(), parse_football_data_season(m.group("season"))

    def _df_to_records(
        self,
        df: pd.DataFrame,
        league: str,
        season: str,
        path: Path,
    ) -> list[MatchRecord]:
        if df.empty:
            return []
        # Football-Data sometimes ships with leading/trailing whitespace in column names
        df = df.rename(columns={c: c.strip() for c in df.columns})
        if "Date" not in df.columns or "HomeTeam" not in df.columns or "AwayTeam" not in df.columns:
            raise DataValidationError(
                f"CSV {path.name} missing required columns (need Date, HomeTeam, AwayTeam)"
            )

        date_fmt = self._detect_date_format(df["Date"])
        records: list[MatchRecord] = []
        missing_logged: set[str] = set()
        for idx, row in df.iterrows():
            try:
                date = self._parse_date(row["Date"], date_fmt)
            except ValueError as e:
                raise DataValidationError(
                    f"unparseable date {row['Date']!r} in {path.name} row {idx}: {e}"
                ) from e
            payload: dict[str, object] = {
                "match_id": f"{season}-{league}-{idx:04d}",
                "date": date,
                "league": league,
                "season": season,
                "source": "football-data.co.uk",
                "home_team": str(row["HomeTeam"]).strip(),
                "away_team": str(row["AwayTeam"]).strip(),
            }
            # Map each known field from the CSV.
            for field, candidates in _COL_MAP.items():
                col = _resolve_col(df, candidates)
                if col is None:
                    missing_logged.add(field)
                    continue
                value = row[col]
                if pd.isna(value) or value == "":
                    payload[field] = None
                elif field in _ODDS_FIELDS:
                    # Decimal odds must be >= 1.0; values < 1 are corrupt (Football-Data
                    # sometimes uses 0 / sentinel values for missing odds). Treat as None.
                    try:
                        f = float(value)
                    except (TypeError, ValueError):
                        payload[field] = None
                    else:
                        payload[field] = f if f >= 1.0 else None
                else:
                    payload[field] = value
            # Cards: sum yellow + red if both are present and the mapping gave us
            # separate columns. We collapsed (HY, HR) into one mapping key above;
            # handle the case where both are present by summing them.
            if "home_cards" in payload and "away_cards" in payload:
                hc = self._coerce_cards(df, row, "HY", "HR")
                ac = self._coerce_cards(df, row, "AY", "AR")
                if hc is not None:
                    payload["home_cards"] = hc
                if ac is not None:
                    payload["away_cards"] = ac
            # Possession default: split 50/50 if not present.
            if payload.get("home_possession") is None and payload.get("away_possession") is None:
                payload["home_possession"] = 0.5
                payload["away_possession"] = 0.5
            elif (
                payload.get("home_possession") is not None
                and payload.get("away_possession") is None
            ):
                payload["away_possession"] = 1.0 - float(payload["home_possession"])  # type: ignore[arg-type]
            elif (
                payload.get("away_possession") is not None
                and payload.get("home_possession") is None
            ):
                payload["home_possession"] = 1.0 - float(payload["away_possession"])  # type: ignore[arg-type]
            # Closing odds default to opening odds when not present.
            if (
                payload.get("closing_odds_over_2_5") is None
                and payload.get("odds_over_2_5") is not None
            ):
                payload["closing_odds_over_2_5"] = payload["odds_over_2_5"]
            if (
                payload.get("closing_odds_under_2_5") is None
                and payload.get("odds_under_2_5") is not None
            ):
                payload["closing_odds_under_2_5"] = payload["odds_under_2_5"]
            try:
                records.append(MatchRecord.model_validate(payload))
            except Exception as e:
                raise DataValidationError(
                    f"invalid record in {path.name} row {idx} ({payload.get('home_team')} vs "
                    f"{payload.get('away_team')}): {e}"
                ) from e
        if missing_logged:
            logger.warning(
                "%s: missing optional columns %s (features depending on them will be NaN)",
                path.name,
                sorted(missing_logged),
            )
        return records

    @staticmethod
    def _coerce_cards(df: pd.DataFrame, row: pd.Series, yellow: str, red: str) -> float | None:
        y = row.get(yellow) if yellow in df.columns else None
        r = row.get(red) if red in df.columns else None
        y_val = None if (y is None or pd.isna(y)) else float(y)
        r_val = None if (r is None or pd.isna(r)) else float(r)
        if y_val is None and r_val is None:
            return None
        return (y_val or 0.0) + (r_val or 0.0)

    @staticmethod
    def _detect_date_format(series: pd.Series) -> str:
        """Heuristic: pick a pandas strptime format that matches most rows."""

        sample = series.dropna().astype(str).head(50)
        if sample.empty:
            return "%d/%m/%Y"
        # Try common formats.
        for fmt in ("%d/%m/%Y", "%d/%m/%y", "%Y-%m-%d"):
            ok = 0
            for s in sample:
                try:
                    datetime.strptime(s, fmt)
                    ok += 1
                except ValueError:
                    pass
            if ok == len(sample):
                return fmt
        return "%d/%m/%Y"  # default; will raise on bad rows

    @staticmethod
    def _parse_date(value: object, fmt: str) -> datetime:
        s = str(value).strip()
        return datetime.strptime(s, fmt)

    @staticmethod
    def _read_csv_with_encoding_fallback(path: Path) -> pd.DataFrame:
        """Read a CSV, trying UTF-8 first then Latin-1.

        Some Football-Data.co.uk files (older seasons, some leagues) ship with
        Latin-1 / cp1252 team names that fail strict UTF-8 decoding.
        """

        for encoding in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
            try:
                return pd.read_csv(path, encoding=encoding)
            except UnicodeDecodeError:
                continue
        # Last resort: replace invalid bytes with the replacement char.
        return pd.read_csv(path, encoding="utf-8", encoding_errors="replace")


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def make_synthetic_source(records: list[MatchRecord]) -> DataSource:
    """Wrap an in-memory list as a :class:`DataSource` (useful for tests)."""

    class _InMemorySource:
        def load_seasons(self, leagues: Sequence[str], seasons: Sequence[str]) -> list[MatchRecord]:
            league_set = {lg.upper() for lg in leagues} if leagues else None
            season_set = {parse_football_data_season(s) for s in seasons} if seasons else None
            return [
                r
                for r in records
                if (league_set is None or r.league.upper() in league_set)
                and (season_set is None or r.season in season_set)
            ]

        def available_seasons(self, league: str) -> list[str]:
            league = league.upper()
            return sorted({r.season for r in records if r.league.upper() == league})

    return _InMemorySource()
