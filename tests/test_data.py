"""Tests for the data layer: schema and CSV loader."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from fussball.data import (
    LEAGUE_CODES,
    DataValidationError,
    FreeCsvLoader,
    MatchRecord,
    canonical_season_label,
    make_synthetic_source,
    parse_football_data_season,
)

# -----------------------------------------------------------------------------
# Season parser
# -----------------------------------------------------------------------------


class TestSeasonParser:
    @pytest.mark.parametrize(
        "token,expected",
        [
            ("2122", "2021-2022"),
            ("2021-22", "2021-2022"),
            ("2021-2022", "2021-2022"),
            ("202122", "2021-2022"),
            ("2122", "2021-2022"),
            ("2020-2021", "2020-2021"),
            ("2020-21", "2020-2021"),
        ],
    )
    def test_round_trip(self, token: str, expected: str) -> None:
        assert parse_football_data_season(token) == expected

    def test_canonical_helper(self) -> None:
        assert canonical_season_label(2021) == "2021-2022"

    @pytest.mark.parametrize("bad", ["", "abc", "212", "21-22-23", "2021/2022"])
    def test_invalid(self, bad: str) -> None:
        with pytest.raises(ValueError, match="unrecognised season"):
            parse_football_data_season(bad)


# -----------------------------------------------------------------------------
# MatchRecord
# -----------------------------------------------------------------------------


def _base_record_kwargs(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "match_id": "x",
        "date": datetime(2023, 1, 1),
        "league": "E0",
        "season": "2022-2023",
        "source": "synthetic",
        "home_team": "Arsenal",
        "away_team": "Chelsea",
        "home_goals": 2,
        "away_goals": 1,
        "total_goals": 3,
        "over_2_5": 1,
    }
    base.update(overrides)
    return base


class TestMatchRecord:
    def test_minimal_valid_record(self) -> None:
        m = MatchRecord.model_validate(_base_record_kwargs())
        assert m.home_team == "Arsenal"
        assert m.over_2_5 == 1

    def test_league_normalised_to_upper(self) -> None:
        m = MatchRecord.model_validate(_base_record_kwargs(league="e0"))
        assert m.league == "E0"

    def test_negative_goals_rejected(self) -> None:
        with pytest.raises(Exception, match=r"(greater than|greater than or equal|Validation)"):
            MatchRecord.model_validate(_base_record_kwargs(home_goals=-1))

    def test_total_derived_from_score(self) -> None:
        m = MatchRecord.model_validate(
            _base_record_kwargs(home_goals=2, away_goals=1, total_goals=999, over_2_5=9)
        )
        # Derived fields always reflect the actual score.
        assert m.total_goals == 3
        assert m.over_2_5 == 1

    def test_over_derived_from_total(self) -> None:
        m = MatchRecord.model_validate(_base_record_kwargs(home_goals=1, away_goals=1))
        assert m.total_goals == 2
        assert m.over_2_5 == 0

    def test_same_team_rejected(self) -> None:
        with pytest.raises(Exception, match=r"same team"):
            MatchRecord.model_validate(_base_record_kwargs(away_team="Arsenal"))

    def test_low_margin_odds_accepted_at_schema_level(self) -> None:
        # Real markets in low-liquidity windows can have negative margins.
        # We accept the values (the loader sanitizes corrupt ones; the schema
        # trusts whatever the loader passes through).
        m = MatchRecord.model_validate(
            _base_record_kwargs(
                odds_over_2_5=10.0,
                odds_under_2_5=10.0,
            )
        )
        assert m.odds_over_2_5 == 10.0

    def test_extra_fields_ignored(self) -> None:
        m = MatchRecord.model_validate(_base_record_kwargs(extra_unknown="x", other=42))
        assert not hasattr(m, "extra_unknown")

    def test_possession_normalised_from_pct(self) -> None:
        m = MatchRecord.model_validate(_base_record_kwargs(home_possession=55, away_possession=45))
        assert m.home_possession == 0.55
        assert m.away_possession == 0.45

    def test_frozen(self) -> None:
        m = MatchRecord.model_validate(_base_record_kwargs())
        with pytest.raises(Exception, match=r"(frozen|hash|setattr)"):
            m.home_team = "Other"  # type: ignore[misc]

    def test_league_codes_includes_top5(self) -> None:
        for code in ("E0", "SP1", "D1", "I1", "F1"):
            assert code in LEAGUE_CODES


# -----------------------------------------------------------------------------
# FreeCsvLoader
# -----------------------------------------------------------------------------


def _write_csv(
    path: Path, rows: list[dict[str, object]], *, league: str = "E0", season: str = "2122"
) -> Path:
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return path


def _sample_rows() -> list[dict[str, object]]:
    return [
        {
            "Div": "E0",
            "Date": "13/08/2021",
            "HomeTeam": "Arsenal",
            "AwayTeam": "Brentford",
            "FTHG": 2,
            "FTAG": 0,
            "HS": 15,
            "AS": 8,
            "HST": 6,
            "AST": 2,
            "HC": 7,
            "AC": 3,
            "HY": 2,
            "AY": 3,
            "Avg>2.5": 1.85,
            "Avg<2.5": 1.95,
        },
        {
            "Div": "E0",
            "Date": "14/08/2021",
            "HomeTeam": "Chelsea",
            "AwayTeam": "Crystal Palace",
            "FTHG": 3,
            "FTAG": 0,
            "HS": 20,
            "AS": 5,
            "HST": 8,
            "AST": 1,
            "HC": 9,
            "AC": 2,
            "HY": 1,
            "AY": 4,
            "Avg>2.5": 1.70,
            "Avg<2.5": 2.10,
        },
        {
            "Div": "E0",
            "Date": "21/08/2021",
            "HomeTeam": "Liverpool",
            "AwayTeam": "Burnley",
            "FTHG": 2,
            "FTAG": 0,
            "HS": 18,
            "AS": 4,
            "HST": 7,
            "AST": 1,
            "HC": 8,
            "AC": 2,
            "HY": 0,
            "AY": 3,
            "Avg>2.5": 1.60,
            "Avg<2.5": 2.30,
        },
    ]


class TestFreeCsvLoader:
    def test_happy_path_loads_minimal_csv(self, tmp_path: Path) -> None:
        _write_csv(tmp_path / "E0_2122.csv", _sample_rows())
        loader = FreeCsvLoader(data_dir=tmp_path)
        records = loader.load_seasons(["E0"], ["2122"])
        assert len(records) == 3
        assert all(isinstance(r, MatchRecord) for r in records)
        r0 = records[0]
        assert r0.home_team == "Arsenal"
        assert r0.away_team == "Brentford"
        assert r0.home_goals == 2
        assert r0.away_goals == 0
        assert r0.total_goals == 2
        assert r0.over_2_5 == 0
        assert r0.odds_over_2_5 == 1.85
        assert r0.closing_odds_over_2_5 == 1.85  # default
        assert r0.league == "E0"
        assert r0.season == "2021-2022"
        assert r0.source == "football-data.co.uk"

    def test_missing_optional_columns_yields_none(self, tmp_path: Path) -> None:
        rows = _sample_rows()
        # Drop xG and possession columns (none of the source rows had them anyway,
        # but let's also drop a stats column to ensure loader doesn't crash).
        for row in rows:
            row.pop("HS", None)
            row.pop("AS", None)
        _write_csv(tmp_path / "E0_2122.csv", rows)
        loader = FreeCsvLoader(data_dir=tmp_path)
        records = loader.load_seasons(["E0"], ["2122"])
        assert all(r.home_shots is None for r in records)
        # Possession should be defaulted to 0.5/0.5.
        assert all(r.home_possession == 0.5 for r in records)

    def test_extra_unknown_columns_ignored(self, tmp_path: Path) -> None:
        rows = _sample_rows()
        for row in rows:
            row["UnknownBookieH"] = 1.5
            row["CustomField"] = "ignore me"
        _write_csv(tmp_path / "E0_2122.csv", rows)
        loader = FreeCsvLoader(data_dir=tmp_path)
        records = loader.load_seasons(["E0"], ["2122"])
        assert len(records) == 3  # no crash, columns silently ignored

    def test_filter_by_league_and_season(self, tmp_path: Path) -> None:
        _write_csv(tmp_path / "E0_2122.csv", _sample_rows())
        sp = _sample_rows()
        for r in sp:
            r["Div"] = "SP1"
            r["HomeTeam"] = "Real Madrid"
            r["AwayTeam"] = "Valencia"
        _write_csv(tmp_path / "SP1_2122.csv", sp)

        loader = FreeCsvLoader(data_dir=tmp_path)
        epl = loader.load_seasons(["E0"], ["2122"])
        assert len(epl) == 3
        assert all(r.league == "E0" for r in epl)

        all_records = loader.load_seasons(["E0", "SP1"], ["2122"])
        assert len(all_records) == 6
        assert {r.league for r in all_records} == {"E0", "SP1"}

    def test_seasons_filter_excludes_other_years(self, tmp_path: Path) -> None:
        _write_csv(tmp_path / "E0_2122.csv", _sample_rows())
        rows2 = _sample_rows()
        for r in rows2:
            r["Date"] = "13/08/2022"
        _write_csv(tmp_path / "E0_2223.csv", rows2)
        loader = FreeCsvLoader(data_dir=tmp_path)
        only_2122 = loader.load_seasons(["E0"], ["2122"])
        assert len(only_2122) == 3
        assert all(r.season == "2021-2022" for r in only_2122)

    def test_negative_goals_raises_validation_error(self, tmp_path: Path) -> None:
        rows = _sample_rows()
        rows[0]["FTHG"] = -1
        _write_csv(tmp_path / "E0_2122.csv", rows)
        loader = FreeCsvLoader(data_dir=tmp_path)
        with pytest.raises(DataValidationError, match=r"row 0"):
            loader.load_seasons(["E0"], ["2122"])

    def test_unparseable_date_raises_validation_error(self, tmp_path: Path) -> None:
        rows = _sample_rows()
        rows[1]["Date"] = "not-a-date"
        _write_csv(tmp_path / "E0_2122.csv", rows)
        loader = FreeCsvLoader(data_dir=tmp_path)
        with pytest.raises(DataValidationError, match=r"date"):
            loader.load_seasons(["E0"], ["2122"])

    def test_missing_data_dir_raises_filenotfound(self, tmp_path: Path) -> None:
        loader = FreeCsvLoader(data_dir=tmp_path / "does-not-exist")
        with pytest.raises(FileNotFoundError, match=r"data directory"):
            loader.load_seasons(["E0"], ["2122"])

    def test_empty_dir_returns_empty_list(self, tmp_path: Path) -> None:
        loader = FreeCsvLoader(data_dir=tmp_path)
        assert loader.load_seasons(["E0"], ["2122"]) == []

    def test_non_conforming_filename_skipped(self, tmp_path: Path) -> None:
        _write_csv(tmp_path / "E0_2122.csv", _sample_rows())
        # Junk file should be ignored.
        (tmp_path / "README.txt").write_text("not a csv")
        (tmp_path / "garbage.csv").write_text("not a match csv")
        loader = FreeCsvLoader(data_dir=tmp_path)
        records = loader.load_seasons(["E0"], ["2122"])
        assert len(records) == 3

    def test_available_seasons(self, tmp_path: Path) -> None:
        _write_csv(tmp_path / "E0_2122.csv", _sample_rows())
        rows2 = _sample_rows()
        for r in rows2:
            r["Date"] = "13/08/2022"
        _write_csv(tmp_path / "E0_2223.csv", rows2)
        loader = FreeCsvLoader(data_dir=tmp_path)
        assert loader.available_seasons("E0") == ["2021-2022", "2022-2023"]
        assert loader.available_seasons("SP1") == []

    def test_records_sorted_by_date(self, tmp_path: Path) -> None:
        rows = _sample_rows()
        # Reorder to be out of order by date.
        rows = [rows[2], rows[0], rows[1]]
        _write_csv(tmp_path / "E0_2122.csv", rows)
        loader = FreeCsvLoader(data_dir=tmp_path)
        records = loader.load_seasons(["E0"], ["2122"])
        dates = [r.date for r in records]
        assert dates == sorted(dates)

    def test_first_matches_no_history_does_not_crash(self, tmp_path: Path) -> None:
        # First week of a season: feature engineering handles "no history" later,
        # but the data loader should not crash on early matches.
        rows = _sample_rows()
        # Make all matches on the same early-season date.
        for r in rows:
            r["Date"] = "13/08/2021"
        _write_csv(tmp_path / "E0_2122.csv", rows)
        loader = FreeCsvLoader(data_dir=tmp_path)
        records = loader.load_seasons(["E0"], ["2122"])
        assert len(records) == 3

    def test_invalid_odds_sanitized_to_none(self, tmp_path: Path) -> None:
        """Odds < 1.0 (corrupt values, sentinels) are set to None, not errors."""

        rows = _sample_rows()
        # 0.42 is the D1_1920 real-world bug: Football-Data sometimes writes
        # garbage for missing odds instead of leaving the cell empty.
        rows[0]["Avg>2.5"] = 0.42
        rows[0]["Avg<2.5"] = 0.0
        rows[1]["Avg>2.5"] = ""  # empty cell
        _write_csv(tmp_path / "E0_2122.csv", rows)
        loader = FreeCsvLoader(data_dir=tmp_path)
        records = loader.load_seasons(["E0"], ["2122"])
        assert records[0].odds_over_2_5 is None
        assert records[0].odds_under_2_5 is None
        assert records[1].odds_over_2_5 is None
        # Records still load — just without odds.
        assert len(records) == 3
        assert records[2].odds_over_2_5 == 1.60  # unaffected row

    def test_latin1_encoded_csv_loads(self, tmp_path: Path) -> None:
        """Older Football-Data files use Latin-1 for some team names."""

        rows = _sample_rows()
        # Add a team name with a non-ASCII char that breaks strict UTF-8.
        rows[0]["HomeTeam"] = "M\u00f6nchengladbach"  # 'ö' is 0xC3 0xB6 in UTF-8
        # Write as Latin-1 (which produces byte 0xF6 for 'ö').
        df = pd.DataFrame(rows)
        df.to_csv(tmp_path / "D1_2122.csv", index=False, encoding="latin-1")
        loader = FreeCsvLoader(data_dir=tmp_path)
        records = loader.load_seasons(["D1"], ["2122"])
        assert len(records) == 3
        # Team name round-trips (may be mojibake under replacement, but no crash).
        assert "M" in records[0].home_team  # accept either UTF-8 or Latin-1 decode

    def test_low_margin_odds_accepted(self, tmp_path: Path) -> None:
        """Odds that imply < 1.0 combined probability (early prices, low liquidity) load OK."""

        rows = _sample_rows()
        # Real-world example: 2.2 / 2.54 implies ~0.85 combined prob.
        rows[0]["Avg>2.5"] = 2.20
        rows[0]["Avg<2.5"] = 2.54
        _write_csv(tmp_path / "N1_1617.csv", rows)
        loader = FreeCsvLoader(data_dir=tmp_path)
        records = loader.load_seasons(["N1"], ["1617"])
        assert records[0].odds_over_2_5 == 2.20
        assert records[0].odds_under_2_5 == 2.54


# -----------------------------------------------------------------------------
# In-memory synthetic source
# -----------------------------------------------------------------------------


class TestSyntheticSource:
    def test_filters_by_league(self, synthetic_matches: list[MatchRecord]) -> None:
        src = make_synthetic_source(synthetic_matches)
        epl = src.load_seasons(["E0"], [])
        assert all(r.league == "E0" for r in epl)
        assert len(epl) > 0

    def test_filters_by_season(self, synthetic_matches: list[MatchRecord]) -> None:
        src = make_synthetic_source(synthetic_matches)
        rows = src.load_seasons([], ["2021-2022"])
        assert all(r.season == "2021-2022" for r in rows)

    def test_empty_filters_returns_all(self, synthetic_matches: list[MatchRecord]) -> None:
        src = make_synthetic_source(synthetic_matches)
        all_records = src.load_seasons([], [])
        assert all_records == synthetic_matches
