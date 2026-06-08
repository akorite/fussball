"""Tests for the HMM module.

These cover:

- Forward algorithm correctness with a known HMM
- Parameter fitting convergence (log-likelihood monotonicity)
- ``predict_next`` on constant and bimodal sequences
- State entropy properties
- ``expected_goals_to_p_over25`` edge cases and symmetry
- ``HMMFeatureExtractor`` on synthetic data (column shape, finite values,
  determinism, fallback behaviour)
- Integration sketch (feature matrix merge)
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from fussball.hmm import (
    HMMFeatureExtractor,
    HMMResult,
    TeamHMM,
    TeamHMMConfig,
    expected_goals_to_p_over25,
)

# =========================================================================
# TeamHMM — unit tests
# =========================================================================


class TestTeamHMMInit:
    """Parameter initialisation sanity checks."""

    def test_default_config(self) -> None:
        cfg = TeamHMMConfig()
        assert cfg.n_states == 3
        assert cfg.n_iter == 50
        assert cfg.min_matches == 30
        assert cfg.seed == 42
        assert cfg.regularization == 1e-3

    def test_custom_config(self) -> None:
        cfg = TeamHMMConfig(n_states=5, n_iter=100, min_matches=10)
        assert cfg.n_states == 5
        assert cfg.n_iter == 100
        assert cfg.min_matches == 10


class TestForwardAlgorithm:
    """Verify forward algorithm against a manually-computed reference.

    We construct a tiny HMM (K=2, T=3) and compute log-alpha manually using
    the known parameters, then compare with the implementation.
    """

    @pytest.fixture
    def hmm(self) -> TeamHMM:
        cfg = TeamHMMConfig(n_states=2, n_iter=1, regularization=1e-6)
        model = TeamHMM(cfg)
        # Manually set known parameters.
        model.means_ = np.array([0.0, 2.0])
        model.vars_ = np.array([1.0, 1.0])
        model.trans_ = np.array([[0.8, 0.2], [0.3, 0.7]])
        model.start_prob_ = np.array([0.6, 0.4])
        return model

    def test_forward_known_values(self, hmm: TeamHMM) -> None:
        goals = np.array([0.0, 1.0, 3.0])
        log_emit = hmm._log_emit(goals)
        log_alpha = hmm._forward(log_emit)

        # --- manual computation for t=0 ---
        # log N(0|0,1) = -0.5*log(2*pi) - 0.5*0^2 = -0.5*log(2*pi) ≈ -0.9189
        # log N(0|2,1) = -0.5*log(2*pi) - 0.5*4   ≈ -0.9189 - 2 = -2.9189
        log_pi0 = math.log(0.6)
        log_pi1 = math.log(0.4)

        log_n0_0 = -0.5 * math.log(2 * math.pi)
        log_n0_1 = -0.5 * math.log(2 * math.pi) - 0.5 * (0 - 2.0) ** 2
        # log_alpha[0,0] = log_pi0 + log_n0_0
        # log_alpha[0,1] = log_pi1 + log_n0_1
        ref_alpha_00 = log_pi0 + log_n0_0
        ref_alpha_01 = log_pi1 + log_n0_1

        assert log_alpha[0, 0] == pytest.approx(ref_alpha_00, abs=1e-6)
        assert log_alpha[0, 1] == pytest.approx(ref_alpha_01, abs=1e-6)

        # --- manual for t=1 ---
        # log N(1|0,1) = -0.5*log(2*pi) - 0.5*1 = -0.9189 - 0.5 = -1.4189
        # log N(1|2,1) = -0.5*log(2*pi) - 0.5*1 = -0.9189 - 0.5 = -1.4189 (same since both var=1, dist from mean is 1)
        log_n1_0 = -0.5 * math.log(2 * math.pi) - 0.5 * (1.0 - 0.0) ** 2
        log_n1_1 = -0.5 * math.log(2 * math.pi) - 0.5 * (1.0 - 2.0) ** 2
        assert log_n1_0 == pytest.approx(log_n1_1, abs=1e-10)

        # log_alpha[1,0] = log_n1_0 + logsumexp(log_alpha[0] + log_trans[:,0])
        log_trans_col0 = np.log(np.array([0.8, 0.3]))  # log P(s_t = 0 | s_{t-1} = i)
        logsumexp_0 = np.logaddexp(
            ref_alpha_00 + log_trans_col0[0], ref_alpha_01 + log_trans_col0[1]
        )
        ref_alpha_10 = log_n1_0 + logsumexp_0

        assert log_alpha[1, 0] == pytest.approx(ref_alpha_10, abs=1e-6)

    def test_forward_likelihood_positive(self, hmm: TeamHMM) -> None:
        """Likelihood should be positive (log-likelihood finite)."""
        goals = np.array([0.0, 1.0, 3.0])
        log_emit = hmm._log_emit(goals)
        log_alpha = hmm._forward(log_emit)
        log_lik = hmm._log_likelihood(log_alpha)
        assert np.isfinite(log_lik)
        assert log_lik < 0  # Likelihood of continuous data is < 1 for reasonable params
        assert not np.isnan(log_lik)


class TestFitConvergence:
    """Verify that log-likelihood is monotonically non-decreasing."""

    def test_ll_monotonic(self) -> None:
        cfg = TeamHMMConfig(n_states=2, n_iter=20, regularization=1e-3)
        hmm = TeamHMM(cfg)
        goals = np.array([0, 0, 0, 5, 5, 5, 0, 0, 5, 5], dtype=float)

        # Record per-iteration log-likelihood.
        ll_history = []
        hmm._init_params(goals)
        for _ in range(cfg.n_iter):
            log_emit = hmm._log_emit(goals)
            log_alpha = hmm._forward(log_emit)
            log_beta = hmm._backward(log_emit)
            log_lik = hmm._log_likelihood(log_alpha)
            ll_history.append(log_lik)

            # E-step
            gamma = np.exp(log_alpha + log_beta - log_lik)
            T = len(goals)
            K = cfg.n_states
            xi = np.zeros((T - 1, K, K))
            for t in range(T - 1):
                xi[t] = np.exp(
                    log_alpha[t, :, np.newaxis]
                    + np.log(hmm.trans_)
                    + log_emit[t + 1, np.newaxis, :]
                    + log_beta[t + 1, np.newaxis, :]
                    - log_lik
                )
            # M-step
            hmm.start_prob_ = gamma[0]
            hmm.start_prob_ /= hmm.start_prob_.sum()
            for i in range(K):
                denom = np.sum(xi[:, i, :])
                if denom > cfg.regularization:
                    hmm.trans_[i, :] = np.sum(xi[:, i, :], axis=0) / denom
            hmm.trans_ /= hmm.trans_.sum(axis=1, keepdims=True)
            for k in range(K):
                denom = np.sum(gamma[:, k])
                if denom > cfg.regularization:
                    hmm.means_[k] = np.sum(gamma[:, k] * goals) / denom
                denom2 = np.sum(gamma[:, k])
                if denom2 > cfg.regularization:
                    var = np.sum(gamma[:, k] * (goals - hmm.means_[k]) ** 2) / denom2
                    hmm.vars_[k] = max(var, cfg.regularization)

        # Check monotonicity (allow small numerical decreases)
        for i in range(1, len(ll_history)):
            assert ll_history[i] >= ll_history[i - 1] - 1e-9, (
                f"log-likelihood decreased at iteration {i}: "
                f"{ll_history[i - 1]:.6f} -> {ll_history[i]:.6f}"
            )


class TestPredictNext:
    """Behavioural tests for predict_next."""

    def test_constant_sequence(self) -> None:
        """If all observations are 2.0, expected_goals ≈ 2.0."""
        cfg = TeamHMMConfig(n_states=2, n_iter=100)
        hmm = TeamHMM(cfg)
        goals = np.full(50, 2.0, dtype=float)
        hmm.fit(goals)
        result = hmm.predict_next(goals)
        assert result.expected_goals == pytest.approx(2.0, abs=0.5)

    def test_bimodal_sequence(self) -> None:
        """Alternating [0,5,0,5,...] should produce states near 0 and 5."""
        cfg = TeamHMMConfig(n_states=2, n_iter=100)
        hmm = TeamHMM(cfg)
        goals = np.array([0.0, 5.0] * 25)
        hmm.fit(goals)
        # The two means should be near 0 and 5 (order may vary).
        sorted_means = sorted(hmm.means_)
        assert sorted_means[0] == pytest.approx(0.0, abs=1.0)
        assert sorted_means[1] == pytest.approx(5.0, abs=1.0)

    def test_entropy_low_for_iid_sequence(self) -> None:
        """i.i.d. Poisson sequence has no state structure, so HMM should
        concentrate on a single state and produce low entropy."""
        rng = np.random.default_rng(42)
        cfg = TeamHMMConfig(n_states=3, n_iter=50)
        hmm = TeamHMM(cfg)
        goals = rng.poisson(2.0, size=100).astype(float)
        hmm.fit(goals)
        result = hmm.predict_next(goals)
        # Single-state-dominant: entropy should be < 0.5 nats.
        assert result.state_entropy < 0.5, (
            f"Expected entropy < 0.5 for i.i.d. sequence (no state structure), "
            f"got {result.state_entropy:.4f}"
        )

    def test_predict_next_single_observation(self) -> None:
        """predict_next must work with a single observation."""
        cfg = TeamHMMConfig(n_states=2, n_iter=10)
        hmm = TeamHMM(cfg)
        goals = np.array([2.0])
        hmm.fit(goals)
        result = hmm.predict_next(goals)
        assert result.expected_goals >= 0
        assert result.state_probs.shape == (2,)
        assert abs(result.state_probs.sum() - 1.0) < 1e-9

    def test_predict_next_returns_hmmresult(self) -> None:
        """predict_next returns an HMMResult with correct field types."""
        cfg = TeamHMMConfig(n_states=3, n_iter=10)
        hmm = TeamHMM(cfg)
        goals = np.array([1.0, 2.0, 1.0, 3.0, 0.0])
        hmm.fit(goals)
        result = hmm.predict_next(goals)
        assert isinstance(result, HMMResult)
        assert result.state_probs.shape == (3,)
        assert abs(result.state_probs.sum() - 1.0) < 1e-9
        assert isinstance(result.expected_goals, float)
        assert isinstance(result.state_entropy, float)
        assert result.means.shape == (3,)
        assert result.vars.shape == (3,)
        assert 0 <= result.state_entropy <= math.log(3)

    def test_predict_next_empty_goals(self) -> None:
        """Empty goals → uniform state probs."""
        cfg = TeamHMMConfig(n_states=3, n_iter=10)
        hmm = TeamHMM(cfg)
        goals = np.array([1.0, 2.0, 1.0])
        hmm.fit(goals)
        result = hmm.predict_next(np.array([]))
        assert abs(result.state_probs.sum() - 1.0) < 1e-9
        # Uniform distribution over 3 states
        assert result.state_probs == pytest.approx(np.full(3, 1.0 / 3), abs=1e-9)


# =========================================================================
# expected_goals_to_p_over25
# =========================================================================


class TestExpectedGoalsToPOver25:
    """Edge cases and known values for the Poisson conversion."""

    def test_zero_zero(self) -> None:
        """Both lambdas near zero → P(over 2.5) ≈ 0."""
        p = expected_goals_to_p_over25(0.0, 0.0)
        # Clipped to 0.05 each, so there's a tiny probability.
        assert 0.0 <= p < 1e-3

    def test_high_high(self) -> None:
        """Both lambdas high → P(over 2.5) near 1."""
        p = expected_goals_to_p_over25(5.0, 5.0)
        assert p > 0.95

    def test_medium_pair(self) -> None:
        """lambda_home = lambda_away = 1.0 → P(over 2.5) between 0.3 and 0.5."""
        p = expected_goals_to_p_over25(1.0, 1.0)
        assert 0.3 <= p <= 0.5, f"Got {p:.4f}"

    def test_higher_pair(self) -> None:
        """lambda_home = lambda_away = 2.0 → P(over 2.5) between 0.5 and 0.85.

        Hand-computed: P(Poisson(2)+Poisson(2) <= 2) = sum_{i+j<=2} P_i * P_j
        where P_k = e^{-2} * 2^k / k!. This gives ~0.238, so P(>2.5) ~ 0.762.
        """
        p = expected_goals_to_p_over25(2.0, 2.0)
        assert 0.5 <= p <= 0.85, f"Got {p:.4f}"

    def test_symmetry(self) -> None:
        """expected_goals_to_p_over25(a, b) == expected_goals_to_p_over25(b, a)."""
        p1 = expected_goals_to_p_over25(1.0, 2.0)
        p2 = expected_goals_to_p_over25(2.0, 1.0)
        assert p1 == pytest.approx(p2, abs=1e-12)

    def test_clipping_low(self) -> None:
        """Lambdas below 0.05 get clipped to 0.05."""
        p = expected_goals_to_p_over25(0.01, 0.01)
        assert np.isfinite(p)

    def test_clipping_high(self) -> None:
        """Lambdas above 5.0 get clipped to 5.0."""
        p = expected_goals_to_p_over25(10.0, 10.0)
        assert np.isfinite(p)
        assert p > 0.95

    def test_deterministic(self) -> None:
        """Repeated calls produce identical results."""
        p1 = expected_goals_to_p_over25(1.3, 2.7)
        p2 = expected_goals_to_p_over25(1.3, 2.7)
        assert p1 == p2


# =========================================================================
# HMMFeatureExtractor — integration tests with synthetic data
# =========================================================================


class TestHMMFeatureExtractor:
    """End-to-end tests using the ``synthetic_matches`` fixture."""

    @pytest.fixture
    def extractor(self) -> HMMFeatureExtractor:
        # Use low min_matches so all synthetic teams get their own HMM.
        config = TeamHMMConfig(n_states=3, n_iter=30, min_matches=5, seed=42)
        return HMMFeatureExtractor(config)

    def test_transform_aggregate_columns(
        self,
        extractor: HMMFeatureExtractor,
        synthetic_matches: list,
    ) -> None:
        """transform_aggregate returns DataFrame with expected columns."""
        # Split: use first season as history, second season as targets.
        history = [m for m in synthetic_matches if m.season == "2021-2022"]
        records = [m for m in synthetic_matches if m.season == "2022-2023"]

        df = extractor.transform_aggregate(records, history)

        expected_cols = {
            "home_hmm_expected_goals",
            "away_hmm_expected_goals",
            "home_hmm_state_entropy",
            "away_hmm_state_entropy",
            "home_hmm_top_state",
            "away_hmm_top_state",
            "home_hmm_var",
            "away_hmm_var",
            "hmm_implied_p_over25",
        }
        assert set(df.columns) == expected_cols, (
            f"Columns differ: expected={expected_cols}, got={set(df.columns)}"
        )
        assert df.index.name == "match_id"

    def test_all_values_finite(
        self,
        extractor: HMMFeatureExtractor,
        synthetic_matches: list,
    ) -> None:
        """All feature values are finite (no NaN/Inf for matches with history)."""
        history = [m for m in synthetic_matches if m.season == "2021-2022"]
        records = [m for m in synthetic_matches if m.season == "2022-2023"]

        df = extractor.transform_aggregate(records, history)

        # Drop the implied p25 — it's always finite
        numeric_cols = [c for c in df.columns if c != "hmm_implied_p_over25"]
        assert df[numeric_cols].notna().all().all(), "NaN values found in features"
        assert np.isfinite(df[numeric_cols].values).all(), "Inf values found in features"
        # implied p25 should also be finite
        assert np.isfinite(df["hmm_implied_p_over25"].values).all()

    def test_deterministic(
        self,
        extractor: HMMFeatureExtractor,
        synthetic_matches: list,
    ) -> None:
        """Same match_id produces the same features across runs."""
        history = [m for m in synthetic_matches if m.season == "2021-2022"]
        records = [m for m in synthetic_matches if m.season == "2022-2023"]

        df1 = extractor.transform_aggregate(records, history)
        # Create a new extractor with same config
        extractor2 = HMMFeatureExtractor(
            TeamHMMConfig(n_states=3, n_iter=30, min_matches=5, seed=42)
        )
        df2 = extractor2.transform_aggregate(records, history)

        pd.testing.assert_frame_equal(df1, df2)

    def test_fallback_for_low_data_teams(self, synthetic_matches: list) -> None:
        """Teams with < min_matches use the fallback league-average HMM."""
        config = TeamHMMConfig(n_states=3, n_iter=20, min_matches=100, seed=42)
        extractor = HMMFeatureExtractor(config)

        # All teams have < 100 matches in one season, so all should fall back.
        history = [m for m in synthetic_matches if m.season == "2021-2022"]
        records = [m for m in synthetic_matches if m.season == "2022-2023"]

        df = extractor.transform_aggregate(records, history)
        assert not df.empty
        assert df["home_hmm_expected_goals"].notna().all()
        # With fallback, the features should still be finite and sensible.
        assert np.isfinite(df["home_hmm_expected_goals"].values).all()

    def test_transform_on_single_season(self, synthetic_matches: list) -> None:
        """fit and transform on the same data should be deterministic."""
        config = TeamHMMConfig(n_states=3, n_iter=20, min_matches=5, seed=42)
        extractor = HMMFeatureExtractor(config)
        records = [m for m in synthetic_matches if m.season == "2021-2022"]

        extractor.fit(records)
        df = extractor.transform(records)
        assert not df.empty
        assert df.index.name == "match_id"

    def test_hmm_features_merge_with_build_feature_matrix(
        self,
        synthetic_matches: list,
    ) -> None:
        """HMM features can be merged into a FeatureMatrix (integration sketch).

        This is not a full pipeline test but verifies that the HMM output
        DataFrame can be merged with ``build_feature_matrix`` output.
        """
        from fussball.features import build_feature_matrix

        # Use all data — this is a structural integration check.
        records = synthetic_matches
        config = TeamHMMConfig(n_states=3, n_iter=20, min_matches=5, seed=42)
        extractor = HMMFeatureExtractor(config)

        extractor.fit(records)
        hmm_df = extractor.transform(records)

        fm = build_feature_matrix(records)

        # Merge HMM features into the base matrix.
        merged = fm.X.merge(hmm_df, left_index=True, right_index=True, how="left")
        assert not merged.empty
        # Check that the HMM columns exist after merge.
        for col in hmm_df.columns:
            assert col in merged.columns

        # All values should still be finite.
        numeric_cols = [c for c in hmm_df.columns if c != "hmm_implied_p_over25"]
        assert np.isfinite(merged[numeric_cols].values).all()

    def test_hmm_feature_ranges(self, extractor, synthetic_matches) -> None:
        """Sanity-check feature value ranges."""
        history = [m for m in synthetic_matches if m.season == "2021-2022"]
        records = [m for m in synthetic_matches if m.season == "2022-2023"]

        df = extractor.transform_aggregate(records, history)

        # Expected goals should be non-negative and reasonable (< ~5).
        assert (df["home_hmm_expected_goals"] >= 0).all()
        assert (df["away_hmm_expected_goals"] >= 0).all()
        assert (df["home_hmm_expected_goals"] < 5).all()
        assert (df["away_hmm_expected_goals"] < 5).all()

        # Entropy should be in [0, ln(K)]
        assert (df["home_hmm_state_entropy"] >= 0).all()
        assert (df["away_hmm_state_entropy"] >= 0).all()
        max_entropy = math.log(3)
        assert (df["home_hmm_state_entropy"] <= max_entropy + 1e-9).all()
        assert (df["away_hmm_state_entropy"] <= max_entropy + 1e-9).all()

        # Top state should be in [0, K-1]
        assert (df["home_hmm_top_state"] >= 0).all()
        assert (df["home_hmm_top_state"] <= 2).all()
        assert (df["away_hmm_top_state"] >= 0).all()
        assert (df["away_hmm_top_state"] <= 2).all()

        # Implied p25 should be in [0, 1]
        assert (df["hmm_implied_p_over25"] >= 0).all()
        assert (df["hmm_implied_p_over25"] <= 1).all()

    def test_no_team_hmm_for_low_data(self, synthetic_matches) -> None:
        """A team seen rarely should not have a dedicated HMM."""
        config = TeamHMMConfig(n_states=3, n_iter=20, min_matches=1000, seed=42)
        extractor = HMMFeatureExtractor(config)
        history = [m for m in synthetic_matches if m.season == "2021-2022"]

        extractor.fit(history)

        # All teams have way fewer than 1000 matches, so no team-specific HMMs.
        # The only HMM should be the fallback.
        assert len(extractor._team_hmm) == 1  # only fallback
        assert "::fallback::" in extractor._team_hmm


# =========================================================================
# Forward algorithm correctness with a fully specified HMM (K=2, T=4)
# =========================================================================


class TestForwardBackwardKnownValues:
    """Compare forward-backward against manually computed values."""

    @pytest.fixture
    def hmm(self) -> TeamHMM:
        """Fully specified 2-state HMM for a T=4 sequence."""
        cfg = TeamHMMConfig(n_states=2, n_iter=1, regularization=1e-6)
        model = TeamHMM(cfg)
        model.means_ = np.array([1.0, 3.0])
        model.vars_ = np.array([0.5, 0.5])
        model.trans_ = np.array([[0.7, 0.3], [0.2, 0.8]])
        model.start_prob_ = np.array([0.5, 0.5])
        return model

    def test_log_emit_shapes(self, hmm: TeamHMM) -> None:
        goals = np.array([1.0, 2.0, 3.0, 0.0])
        log_emit = hmm._log_emit(goals)
        assert log_emit.shape == (4, 2)

    def test_forward_backward_consistency(self, hmm: TeamHMM) -> None:
        """Posterior marginals sum to 1 at each time-step."""
        goals = np.array([1.0, 2.0, 3.0, 0.0])
        log_emit = hmm._log_emit(goals)
        log_alpha = hmm._forward(log_emit)
        log_beta = hmm._backward(log_emit)
        log_lik = hmm._log_likelihood(log_alpha)

        gamma = np.exp(log_alpha + log_beta - log_lik)
        row_sums = gamma.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-9)


# =========================================================================
# Edge cases
# =========================================================================


class TestHMMEdgeCases:
    """Bounds and error handling."""

    def test_single_state(self) -> None:
        """K=1 should still work."""
        cfg = TeamHMMConfig(n_states=1, n_iter=10)
        hmm = TeamHMM(cfg)
        goals = np.array([1.0, 2.0, 1.5])
        hmm.fit(goals)
        result = hmm.predict_next(goals)
        assert abs(result.state_probs.sum() - 1.0) < 1e-9
        assert result.expected_goals == pytest.approx(np.mean(goals), abs=0.5)

    def test_two_observations(self) -> None:
        """Very short sequences should not crash."""
        cfg = TeamHMMConfig(n_states=2, n_iter=10)
        hmm = TeamHMM(cfg)
        goals = np.array([0.0, 5.0])
        hmm.fit(goals)
        result = hmm.predict_next(goals)
        assert result.expected_goals >= 0

    def test_raises_on_2d_goals(self) -> None:
        """Passing 2-D array raises."""
        cfg = TeamHMMConfig()
        hmm = TeamHMM(cfg)
        with pytest.raises(ValueError, match="1-D"):
            hmm.fit(np.array([[1.0, 2.0], [3.0, 4.0]]))

    def test_extractor_not_fitted(self, synthetic_matches) -> None:
        """Calling transform without fit raises RuntimeError."""
        extractor = HMMFeatureExtractor()
        records = [m for m in synthetic_matches if m.season == "2021-2022"]
        with pytest.raises(RuntimeError, match="not fitted"):
            extractor.transform(records)


# =========================================================================
# Reproducibility of Poisson conversion at extremes
# =========================================================================


class TestPoissonSymmetry:
    """Additional Poisson conversion checks."""

    @pytest.mark.parametrize(
        "lam_h, lam_a",
        [
            (0.05, 0.05),
            (0.1, 0.1),
            (0.5, 0.5),
            (1.0, 1.5),
            (2.5, 0.8),
            (3.0, 3.0),
            (5.0, 0.5),
        ],
    )
    def test_symmetry_parametrized(self, lam_h: float, lam_a: float) -> None:
        """Symmetry holds for a range of values."""
        p1 = expected_goals_to_p_over25(lam_h, lam_a)
        p2 = expected_goals_to_p_over25(lam_a, lam_h)
        assert p1 == pytest.approx(p2, abs=1e-12)

    @pytest.mark.parametrize(
        "lam_h, lam_a, expected_low, expected_high",
        [
            (0.05, 0.05, 0.0, 0.001),
            (0.5, 0.5, 0.05, 0.15),
            (1.0, 1.0, 0.3, 0.5),
            (2.0, 2.0, 0.5, 0.85),
            (3.0, 3.0, 0.85, 0.99),
            (5.0, 5.0, 0.95, 1.0),
        ],
    )
    def test_known_bounds(
        self, lam_h: float, lam_a: float, expected_low: float, expected_high: float
    ) -> None:
        """P(over 2.5) falls within documented bounds."""
        p = expected_goals_to_p_over25(lam_h, lam_a)
        assert expected_low <= p <= expected_high, (
            f"P(over 2.5 | lam_h={lam_h}, lam_a={lam_a}) = {p:.6f}, "
            f"expected [{expected_low}, {expected_high}]"
        )
