"""Tests for the feature-engineering module (U3).

Covers rolling features, match-level features, seasonal context features,
the ``FeatureMatrix`` dataclass, and the top-level ``build_feature_matrix``
entry-point.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import pytest

from fussball.features import build_feature_matrix, select_xy
from fussball.features.context import compute_context_features
from fussball.features.match import compute_match_features
from fussball.features.rolling import compute_rolling_features

# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture
def three_match_sequence(match_factory):
    """Three sequential matches for the same team (hand-calculable rolling)."""
    base_date = datetime(2023, 1, 1)
    return [
        match_factory(
            match_id="M001",
            date=base_date,
            home_team="Arsenal",
            away_team="Chelsea",
            home_goals=2,
            away_goals=1,
            home_shots=14.0,
            away_shots=8.0,
            home_xg=1.8,
            away_xg=0.9,
        ),
        match_factory(
            match_id="M002",
            date=base_date + timedelta(days=7),
            home_team="Arsenal",
            away_team="Burnley",
            home_goals=3,
            away_goals=0,
            home_shots=16.0,
            away_shots=4.0,
            home_xg=2.2,
            away_xg=0.3,
        ),
        match_factory(
            match_id="M003",
            date=base_date + timedelta(days=14),
            home_team="Arsenal",
            away_team="Sheffield",
            home_goals=1,
            away_goals=1,
            home_shots=10.0,
            away_shots=6.0,
            home_xg=1.2,
            away_xg=0.7,
        ),
    ]


@pytest.fixture
def first_match_of_season(match_factory):
    """A single match that is the first of the season for both teams."""
    return [
        match_factory(
            match_id="S001",
            date=datetime(2023, 8, 12),
            home_team="Arsenal",
            away_team="Chelsea",
            home_goals=1,
            away_goals=0,
        ),
    ]


@pytest.fixture
def back_to_back_matches(match_factory):
    """Matches with 1-day gap (back-to-back) and later a 14-day gap."""
    base = datetime(2023, 1, 1)
    return [
        match_factory(
            match_id="B001",
            date=base,
            home_team="Arsenal",
            away_team="Chelsea",
        ),
        match_factory(
            match_id="B002",
            date=base + timedelta(days=1),
            home_team="Arsenal",
            away_team="Burnley",
        ),
        match_factory(
            match_id="B003",
            date=base + timedelta(days=15),  # 14 days after B002
            home_team="Arsenal",
            away_team="Sheffield",
        ),
    ]


@pytest.fixture
def empty_records() -> list:
    return []


# ======================================================================
# Rolling feature tests
# ======================================================================


class TestRollingFeatures:
    """Tests for ``compute_rolling_features``."""

    def test_happy_path_hand_calculated(self, three_match_sequence):
        """Verify rolling averages for a 3-match sequence."""
        df = compute_rolling_features(three_match_sequence)
        # All three matches should be present
        assert len(df) == 3
        assert set(df["match_id"]) == {"M001", "M002", "M003"}

        # M001: Arsenal's first match → no prior data → NaN rolling stats
        m1 = df[df["match_id"] == "M001"].iloc[0]
        # Arsenal (home) stats
        assert pd.isna(m1["home_goals_scored_avg_5"])
        assert pd.isna(m1["home_goals_conceded_avg_5"])
        assert pd.isna(m1["home_shots_diff_avg_5"])
        assert pd.isna(m1["home_ppg_avg_5"])
        assert pd.isna(m1["home_xg_diff_avg_5"])

        # M002: 1 prior match → partial window average
        m2 = df[df["match_id"] == "M002"].iloc[0]
        assert m2["home_goals_scored_avg_5"] == pytest.approx(2.0)  # 2 goals in M001
        assert m2["home_goals_conceded_avg_5"] == pytest.approx(1.0)  # 1 conceded
        assert m2["home_shots_diff_avg_5"] == pytest.approx(6.0)  # 14-8
        assert m2["home_ppg_avg_5"] == pytest.approx(3.0)  # win = 3 pts
        assert m2["home_xg_diff_avg_5"] == pytest.approx(0.9)  # 1.8 - 0.9

        # M003: 2 prior matches → average of M001 and M002
        m3 = df[df["match_id"] == "M003"].iloc[0]
        assert m3["home_goals_scored_avg_5"] == pytest.approx(2.5)  # (2+3)/2
        assert m3["home_goals_conceded_avg_5"] == pytest.approx(0.5)  # (1+0)/2
        assert m3["home_shots_diff_avg_5"] == pytest.approx(9.0)  # (6+12)/2
        assert m3["home_ppg_avg_5"] == pytest.approx(3.0)  # (3+3)/2
        assert m3["home_xg_diff_avg_5"] == pytest.approx(1.4)  # (0.9+1.9)/2

    def test_first_match_of_season_no_nan(self, first_match_of_season):
        """First match of a season: rolling features use partial window.

        Since there are no prior matches, the first match will have NaN for
        rolling stats (no history to compute from).
        """
        df = compute_rolling_features(first_match_of_season)
        assert len(df) == 1

        m1 = df.iloc[0]
        # All rolling features should be NaN (no prior data)
        for side in ("home", "away"):
            for stat in (
                "goals_scored_avg",
                "goals_conceded_avg",
                "shots_diff_avg",
                "ppg_avg",
                "xg_diff_avg",
            ):
                for w in (5, 10, 20):
                    val = m1[f"{side}_{stat}_{w}"]
                    assert pd.isna(val), f"{side}_{stat}_{w} should be NaN, got {val}"

    def test_single_team_matches(self, single_team_matches):
        """All matches involving one team should compute without error."""
        df = compute_rolling_features(single_team_matches)
        assert len(df) == len(single_team_matches)
        # All rolling columns should exist
        for side in ("home", "away"):
            for stat in (
                "goals_scored_avg",
                "goals_conceded_avg",
                "shots_diff_avg",
                "ppg_avg",
                "xg_diff_avg",
            ):
                for w in (5, 10, 20):
                    assert f"{side}_{stat}_{w}" in df.columns

    def test_empty_records(self, empty_records):
        """Empty record list should produce an empty DataFrame (not crash)."""
        df = compute_rolling_features(empty_records)
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["match_id"]
        assert len(df) == 0

    def test_shots_diff_and_xg_diff_present(self, synthetic_matches):
        """Shots differential and xG differential rolling features exist."""
        df = compute_rolling_features(synthetic_matches[:20])
        assert "home_shots_diff_avg_5" in df.columns
        assert "home_xg_diff_avg_5" in df.columns
        assert "away_shots_diff_avg_5" in df.columns
        assert "away_xg_diff_avg_5" in df.columns

    def test_rolling_windows_all_sizes(self, three_match_sequence):
        """Each stat should have columns for windows 5, 10, and 20."""
        df = compute_rolling_features(three_match_sequence)
        for w in (5, 10, 20):
            assert f"home_goals_scored_avg_{w}" in df.columns
            assert f"home_ppg_avg_{w}" in df.columns


# ======================================================================
# Match feature tests
# ======================================================================


class TestMatchFeatures:
    """Tests for ``compute_match_features``."""

    def test_rest_days(self, back_to_back_matches):
        """Back-to-back (1 day) and long break (14+ days)."""
        df = compute_match_features(back_to_back_matches)
        assert len(df) == 3

        m1 = df[df["match_id"] == "B001"].iloc[0]
        m2 = df[df["match_id"] == "B002"].iloc[0]
        m3 = df[df["match_id"] == "B003"].iloc[0]

        # First match: no prior data → NaN
        assert pd.isna(m1["home_rest_days"])
        assert pd.isna(m1["away_rest_days"])

        # Second match: 1 day after first
        assert m2["home_rest_days"] == 1

        # Third match: 14 days after second
        assert m3["home_rest_days"] == 14

    def test_is_home_always_one(self, synthetic_matches):
        """is_home should be 1 for every match."""
        df = compute_match_features(synthetic_matches[:10])
        assert (df["is_home"] == 1).all()

    def test_league_position_gap(self, synthetic_matches):
        """League position gap should be computable, NaN for first matches."""
        df = compute_match_features(synthetic_matches)
        assert "league_position_gap" in df.columns

        # Some matches should have NaN (first matchweek per season per league)
        non_null = df["league_position_gap"].dropna()
        assert len(non_null) > 0, "Expected some matches to have a position gap"
        # Gaps should be integer values (positions are integers)
        for v in non_null:
            assert isinstance(v, (int, float))
            assert v == int(v)  # should be whole number

    def test_league_position_gap_first_match_nan(self, match_factory):
        """First match of a season has NaN position gap."""
        records = [
            match_factory(
                match_id="F001",
                date=datetime(2023, 8, 12),
                home_team="Arsenal",
                away_team="Chelsea",
                season="2023-2024",
            ),
        ]
        df = compute_match_features(records)
        assert pd.isna(df["league_position_gap"].iloc[0])

    def test_league_position_gap_positive(self, match_factory):
        """Home team higher in table → positive gap."""
        records = [
            match_factory(
                match_id="G001",
                date=datetime(2023, 8, 12),
                home_team="Arsenal",
                away_team="Chelsea",
                home_goals=3,
                away_goals=0,
                season="2023-2024",
            ),
            match_factory(
                match_id="G002",
                date=datetime(2023, 8, 19),
                home_team="Burnley",
                away_team="Sheffield",
                home_goals=1,
                away_goals=1,
                season="2023-2024",
            ),
            match_factory(
                match_id="G003",
                date=datetime(2023, 8, 26),
                home_team="Arsenal",
                away_team="Burnley",
                home_goals=2,
                away_goals=0,
                season="2023-2024",
            ),
        ]
        df = compute_match_features(records)
        # M001: first match, NaN
        assert pd.isna(df.loc[df["match_id"] == "G001", "league_position_gap"].iloc[0])
        # M002: first match for Burnley/Sheffield in season, NaN
        assert pd.isna(df.loc[df["match_id"] == "G002", "league_position_gap"].iloc[0])
        # M003: Arsenal has 3 pts, Burnley has 1 pt → Arsenal pos 1, Burnley pos 2 → gap = 1-2 = -1
        gap = df.loc[df["match_id"] == "G003", "league_position_gap"].iloc[0]
        assert gap == -1, f"Expected -1, got {gap}"

    def test_h2h_features(self, match_factory):
        """Head-to-head over rate computed across previous meetings."""
        records = [
            match_factory(
                match_id="H001",
                date=datetime(2021, 8, 12),
                home_team="Arsenal",
                away_team="Chelsea",
                home_goals=3,
                away_goals=0,  # over 2.5
                season="2021-2022",
            ),
            match_factory(
                match_id="H002",
                date=datetime(2021, 8, 19),
                home_team="Chelsea",
                away_team="Arsenal",
                home_goals=1,
                away_goals=0,  # under 2.5
                season="2021-2022",
            ),
            match_factory(
                match_id="H003",
                date=datetime(2022, 8, 13),
                home_team="Arsenal",
                away_team="Chelsea",
                home_goals=2,
                away_goals=1,  # over 2.5
                season="2022-2023",
            ),
        ]
        df = compute_match_features(records)
        # H001: no prior meetings → NaN
        assert pd.isna(df.loc[df["match_id"] == "H001", "h2h_over_rate_3"].iloc[0])
        assert pd.isna(df.loc[df["match_id"] == "H001", "recent_vs_opp_over_rate"].iloc[0])

        # H002: 1 prior meeting (H001 was over) → rate = 1.0
        assert df.loc[df["match_id"] == "H002", "h2h_over_rate_3"].iloc[0] == pytest.approx(1.0)
        assert df.loc[df["match_id"] == "H002", "recent_vs_opp_over_rate"].iloc[0] == pytest.approx(
            1.0
        )

        # H003: 2 prior meetings (H001 over, H002 under) → rate = 0.5
        assert df.loc[df["match_id"] == "H003", "h2h_over_rate_3"].iloc[0] == pytest.approx(0.5)
        assert df.loc[df["match_id"] == "H003", "recent_vs_opp_over_rate"].iloc[0] == pytest.approx(
            0.5
        )

    def test_empty_records(self, empty_records):
        """Empty record list should produce an empty DataFrame (not crash)."""
        df = compute_match_features(empty_records)
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["match_id"]
        assert len(df) == 0

    def test_all_feature_columns_present(self, synthetic_matches):
        """All expected match feature columns are present."""
        df = compute_match_features(synthetic_matches[:20])
        expected = {
            "match_id",
            "home_rest_days",
            "away_rest_days",
            "is_home",
            "league_position_gap",
            "h2h_over_rate_3",
            "recent_vs_opp_over_rate",
        }
        assert expected.issubset(set(df.columns))


# ======================================================================
# Context feature tests
# ======================================================================


class TestContextFeatures:
    """Tests for ``compute_context_features``."""

    def test_matchweek_normalized_in_range(self, synthetic_matches):
        """matchweek_normalized should always be in [0, 1]."""
        df = compute_context_features(synthetic_matches)
        mwn = df["matchweek_normalized"]
        assert mwn.min() >= 0.0
        assert mwn.max() <= 1.0

    def test_season_stage_one_hot(self, synthetic_matches):
        """season_stage should produce columns 0, 1, 2 that sum to 1 per row."""
        df = compute_context_features(synthetic_matches)
        stage_cols = ["season_stage_0", "season_stage_1", "season_stage_2"]
        for col in stage_cols:
            assert col in df.columns
        # Each row should have exactly one of the three = 1
        row_sums = df[stage_cols].sum(axis=1)
        assert (row_sums == 1).all(), "season_stage columns should be one-hot"

    def test_season_stage_distinct_values(self, synthetic_matches):
        """With season split into 3 stages, each stage should appear."""
        df = compute_context_features(synthetic_matches)
        for i in (0, 1, 2):
            col = f"season_stage_{i}"
            assert df[col].sum() > 0, f"season_stage_{i} has no positive rows"

    def test_league_phase_one_hot(self, synthetic_matches):
        """league_phase columns should exist and be one-hot where position known.

        Rows where a team has not yet played (first matchweek of a season)
        have NaN position → all one-hot columns are 0.  Once the team has
        played at least once, exactly one column should be 1.
        """
        df = compute_context_features(synthetic_matches)
        for prefix in ("league_phase_home", "league_phase_away"):
            cols = [f"{prefix}_{i}" for i in (0, 1, 2)]
            for col in cols:
                assert col in df.columns
            row_sums = df[cols].sum(axis=1)
            # Rows where position is known must be one-hot
            known = row_sums != 0
            if known.any():
                assert (row_sums[known] == 1).all(), (
                    f"{prefix} rows with known position should be one-hot"
                )
            # At least some rows have known position
            assert known.sum() > 0, f"No rows have known {prefix} position"

    def test_league_phase_small_league(self, match_factory):
        """With only 4 teams, all should be in title race (phase 0)."""
        records = [
            match_factory(
                match_id="L001",
                date=datetime(2023, 8, 12),
                home_team="Arsenal",
                away_team="Chelsea",
                season="2023-2024",
                league="E0",
                home_goals=2,
                away_goals=0,
            ),
            match_factory(
                match_id="L002",
                date=datetime(2023, 8, 12),
                home_team="Burnley",
                away_team="Sheffield",
                season="2023-2024",
                league="E0",
                home_goals=0,
                away_goals=1,
            ),
            match_factory(
                match_id="L003",
                date=datetime(2023, 8, 19),
                home_team="Arsenal",
                away_team="Burnley",
                season="2023-2024",
                league="E0",
                home_goals=3,
                away_goals=0,
            ),
        ]
        df = compute_context_features(records)
        # First match week: NaN positions → league_phase might be NaN
        # Second match: Arsenal has 3 pts ranked 1, Burnley 0 pts ranked tied-3rd
        # With 4 teams, top 4 = all teams → phase 0
        m3 = df[df["match_id"] == "L003"].iloc[0]
        assert m3["league_phase_home_0"] == 1

    def test_empty_records(self, empty_records):
        """Empty record list → empty DataFrame (not crash)."""
        df = compute_context_features(empty_records)
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["match_id"]
        assert len(df) == 0

    def test_has_all_context_columns(self, synthetic_matches):
        """All expected context columns are present."""
        df = compute_context_features(synthetic_matches)
        expected = {
            "match_id",
            "matchweek_normalized",
            "season_stage_0",
            "season_stage_1",
            "season_stage_2",
            "league_phase_home_0",
            "league_phase_home_1",
            "league_phase_home_2",
            "league_phase_away_0",
            "league_phase_away_1",
            "league_phase_away_2",
        }
        assert expected.issubset(set(df.columns))


# ======================================================================
# FeatureMatrix / build_feature_matrix tests
# ======================================================================


class TestFeatureMatrix:
    """Tests for ``build_feature_matrix`` and ``FeatureMatrix``."""

    def test_row_count_matches_input(self, synthetic_matches):
        """FeatureMatrix row count should match input count."""
        fm = build_feature_matrix(synthetic_matches)
        assert len(fm.X) == len(synthetic_matches)
        assert len(fm.y) == len(synthetic_matches)

    def test_x_and_y_indices_aligned(self, synthetic_matches):
        """X and y should have the same index (match_id)."""
        fm = build_feature_matrix(synthetic_matches)
        assert fm.X.index.name == "match_id"
        assert fm.y.index.name == "match_id"
        assert (fm.X.index == fm.y.index).all()

    def test_identifier_columns_present(self, synthetic_matches):
        """Identifier columns (date, league, season, home_team, away_team) exist."""
        fm = build_feature_matrix(synthetic_matches)
        for col in ["date", "league", "season", "home_team", "away_team"]:
            assert col in fm.X.columns, f"Missing identifier column: {col}"
            assert col in fm.identifier_columns

    def test_match_id_is_index(self, synthetic_matches):
        """match_id should be the DataFrame index."""
        fm = build_feature_matrix(synthetic_matches)
        assert fm.X.index.name == "match_id"

    def test_y_is_over_2_5_as_int(self, synthetic_matches):
        """y should be the over_2_5 target as int 0/1."""
        fm = build_feature_matrix(synthetic_matches)
        assert fm.y.dtype == int or fm.y.dtype == "int64" or fm.y.dtype == "Int64"
        assert set(fm.y.unique()).issubset({0, 1})

    def test_feature_names_no_identifiers(self, synthetic_matches):
        """feature_names should not include identifiers or target."""
        fm = build_feature_matrix(synthetic_matches)
        forbidden = {"match_id", "date", "league", "season", "home_team", "away_team", "over_2_5"}
        for name in fm.feature_names:
            assert name not in forbidden, f"feature_names includes forbidden: {name}"

    def test_select_xy_drops_identifiers(self, synthetic_matches):
        """select_xy should return (features_only, y)."""
        fm = build_feature_matrix(synthetic_matches)
        X_feat, y = select_xy(fm)
        for col in fm.identifier_columns:
            assert col not in X_feat.columns, f"select_xy leaked identifier: {col}"
        assert list(y.index) == list(fm.y.index)
        assert list(X_feat.columns) == fm.feature_names

    def test_empty_records(self, empty_records):
        """Empty record list should produce an empty FeatureMatrix (not crash)."""
        fm = build_feature_matrix(empty_records)
        assert fm.X.empty
        assert fm.y.empty
        assert fm.feature_names == []
        assert len(fm.identifier_columns) > 0
        # select_xy should also work on empty matrix
        X_feat, y = select_xy(fm)
        assert X_feat.empty
        assert y.empty

    def test_synthetic_matches_works(self, synthetic_matches):
        """The full pipeline works with the synthetic match fixture."""
        fm = build_feature_matrix(synthetic_matches)
        assert len(fm.X) == len(synthetic_matches)
        assert len(fm.feature_names) > 0
        # Verify no NaN in core rolling features for matches with history
        # (at least some rows should have non-NaN rolling features)
        has_vals = fm.X["home_goals_scored_avg_5"].dropna()
        assert len(has_vals) > 0

    def test_select_xy_returns_copy(self, synthetic_matches):
        """select_xy should not mutate the original FeatureMatrix."""
        fm = build_feature_matrix(synthetic_matches)
        original_id_cols = list(fm.identifier_columns)
        select_xy(fm)  # should not mutate
        # Original should be unchanged
        assert fm.identifier_columns == original_id_cols
        for col in original_id_cols:
            assert col in fm.X.columns

    def test_feature_names_count(self, synthetic_matches):
        """Count expected feature names."""
        fm = build_feature_matrix(synthetic_matches)
        # Rolling: 5 stats x 3 windows x 2 sides = 30
        # Match: home_rest_days, away_rest_days, is_home, league_position_gap,
        #        h2h_over_rate_3, recent_vs_opp_over_rate = 6
        # Context: matchweek_normalized, season_stage_0/1/2 (3),
        #          league_phase_home_0/1/2 (3), league_phase_away_0/1/2 (3) = 10
        # Odds: market_implied_prob, market_overround, market_log_odds, has_odds = 4
        # Rest: home_games_last_30d, away_games_last_30d, rest_advantage,
        #        combined_congestion = 4
        # Derby: same_city, same_region, historical_intensity = 3
        # H2H: h2h_total_goals_avg, h2h_over25_rate, h2h_home_wins, h2h_draws,
        #      h2h_goals_scored_home_avg, h2h_goals_scored_away_avg, h2h_meetings = 7
        # Total: 30 + 6 + 10 + 4 + 4 + 3 + 7 = 64
        assert len(fm.feature_names) == 64, (
            f"Expected 64 features, got {len(fm.feature_names)}: {fm.feature_names}"
        )

    def test_no_nan_in_mature_rolling(self, match_factory):
        """After enough matches, rolling features should have no NaN."""
        records = []
        base = datetime(2023, 1, 1)
        for i in range(25):
            records.append(
                match_factory(
                    match_id=f"N{i:03d}",
                    date=base + timedelta(days=i * 3),
                    home_team="Arsenal",
                    away_team="Chelsea",
                    home_goals=2,
                    away_goals=1,
                )
            )
        fm = build_feature_matrix(records)
        # After 20+ matches, all rolling stats should be non-NaN for home team
        # (but the last match uses data from match 1-24)
        assert not fm.X["home_goals_scored_avg_5"].iloc[20:].isna().any()
        assert not fm.X["home_ppg_avg_5"].iloc[20:].isna().any()
        assert not fm.X["home_goals_scored_avg_10"].iloc[20:].isna().any()
        assert not fm.X["home_goals_scored_avg_20"].iloc[20:].isna().any()


# ======================================================================
# Additional edge-case tests
# ======================================================================


class TestEdgeCases:
    """Additional edge-case scenarios from the U3 specification."""

    def test_team_promoted_no_prior_season(self, match_factory):
        """Team appearing in a new season (no prior history) — features populate.

        A team promoted to a league should get features from current season
        matches only.  There is no cross-season carry-over required.
        """
        records = [
            match_factory(
                match_id="P001",
                date=datetime(2023, 8, 12),
                home_team="Arsenal",
                away_team="Chelsea",
                season="2023-2024",
                home_goals=2,
                away_goals=0,
            ),
            match_factory(
                match_id="P002",
                date=datetime(2023, 8, 19),
                home_team="Arsenal",
                away_team="Burnley",
                season="2023-2024",
                home_goals=1,
                away_goals=1,
            ),
        ]
        fm = build_feature_matrix(records)
        assert len(fm.X) == 2
        # P001 has no prior history → NaN rolling
        assert pd.isna(fm.X.loc["P001", "home_goals_scored_avg_5"])
        # P002 has 1 prior match → partial average = 2.0
        assert fm.X.loc["P002", "home_goals_scored_avg_5"] == pytest.approx(2.0)

    def test_single_team_limited_history(self, match_factory):
        """Single team plays 3 matches — features compute with limited history."""
        records = [
            match_factory(
                match_id="S001",
                date=datetime(2023, 8, 12),
                home_team="Arsenal",
                away_team="Chelsea",
                home_goals=2,
                away_goals=2,
            ),
            match_factory(
                match_id="S002",
                date=datetime(2023, 8, 19),
                home_team="Arsenal",
                away_team="Burnley",
                home_goals=0,
                away_goals=0,
            ),
            match_factory(
                match_id="S003",
                date=datetime(2023, 8, 26),
                home_team="Arsenal",
                away_team="Sheffield",
                home_goals=3,
                away_goals=2,
            ),
        ]
        fm = build_feature_matrix(records)
        assert len(fm.X) == 3
        # No crash, rolling features exist with partial windows
        for match_id in ("S001", "S002", "S003"):
            val = fm.X.loc[match_id, "home_goals_scored_avg_5"]
            if match_id == "S001":
                assert pd.isna(val)  # no history
            else:
                assert not pd.isna(val)  # partial window

    def test_small_match_set(self, small_match_set):
        """Works with a single league + single season subset."""
        fm = build_feature_matrix(small_match_set)
        assert len(fm.X) == len(small_match_set)
        assert len(fm.feature_names) == 64

    def test_no_side_effects_on_records(self, three_match_sequence):
        """Original MatchRecord objects should not be mutated."""
        ids_before = [r.match_id for r in three_match_sequence]
        _ = build_feature_matrix(three_match_sequence)
        ids_after = [r.match_id for r in three_match_sequence]
        assert ids_before == ids_after

    def test_all_features_finite_after_warmup(self, match_factory):
        """After enough matches, all numeric features should be finite (not NaN)."""
        records = []
        base = datetime(2023, 1, 1)
        for i in range(30):
            records.append(
                match_factory(
                    match_id=f"W{i:03d}",
                    date=base + timedelta(days=i * 3),
                    home_team="Arsenal",
                    away_team="Chelsea" if i % 2 == 0 else "Burnley",
                    home_goals=2,
                    away_goals=1,
                )
            )
        fm = build_feature_matrix(records)
        # After 20+ matches, all rolling features for home should be finite
        # (match 20+ has windows 5, 10, 20 all available)
        rolling_home_cols = [c for c in fm.feature_names if c.startswith("home_")]
        last_10 = fm.X.iloc[-10:]
        for col in rolling_home_cols:
            if col in last_10.columns:
                assert last_10[col].notna().all(), f"Column {col} has NaN in last 10 rows"
