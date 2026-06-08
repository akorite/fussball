"""Feature extractor that applies :class:`TeamHMM` across all teams.

The :class:`HMMFeatureExtractor` trains one HMM per team (with a league-average
fallback for low-data teams) and produces match-level features such as the
expected goals from each team's latent state distribution, the entropy of that
distribution, and an implied over-2.5 probability.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from fussball.data.schema import MatchRecord
from fussball.hmm.team_hmm import (
    HMMResult,
    TeamHMM,
    TeamHMMConfig,
    expected_goals_to_p_over25,
)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_FALLBACK_LABEL = "::fallback::"
"""Special key used to store the league-average fallback HMM."""


class HMMFeatureExtractor:
    """Train and apply HMMs for all teams in a dataset.

    Typical usage (time-series safe)::

        extractor = HMMFeatureExtractor()
        df = extractor.transform_aggregate(test_records, train_records)
        # df contains HMM-derived features for each match in test_records

    Parameters
    ----------
    config:
        Hyper-parameters forwarded to each :class:`TeamHMM`.
    """

    def __init__(self, config: TeamHMMConfig | None = None) -> None:
        self.config = config or TeamHMMConfig()
        self._team_hmm: dict[str, TeamHMM] = {}
        self._team_goals: dict[str, np.ndarray] = {}
        self._team_predict_cache: dict[str, HMMResult] = {}
        self._is_fitted = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, records: list[MatchRecord]) -> None:
        """Train one HMM per team and a fallback league-average HMM.

        Teams with fewer than ``config.min_matches`` matches use the fallback
        HMM (trained on *all* teams' combined goal sequences).

        Parameters
        ----------
        records:
            Match records used for training.
        """
        self._team_hmm = {}
        self._team_goals = {}
        self._team_predict_cache = {}

        # Group records by team and sort within each group by date.
        team_goals_raw: dict[str, list[tuple]] = defaultdict(list)
        for r in records:
            team_goals_raw[r.home_team].append((r.date, r.home_goals, r.match_id))
            team_goals_raw[r.away_team].append((r.date, r.away_goals, r.match_id))

        # Build sorted goal arrays per team.
        for team, entries in team_goals_raw.items():
            entries.sort(key=lambda x: x[0])  # sort by date
            self._team_goals[team] = np.array([e[1] for e in entries], dtype=float)

        # Train per-team HMMs.
        all_goals_combined = []
        for team, goals in self._team_goals.items():
            all_goals_combined.append(goals)
            if len(goals) >= self.config.min_matches:
                hmm = TeamHMM(self.config)
                hmm.fit(goals)
                self._team_hmm[team] = hmm

        # Train fallback league-average HMM on ALL teams' combined goals.
        if all_goals_combined:
            all_goals = np.concatenate(all_goals_combined)
            fallback_hmm = TeamHMM(self.config)
            fallback_hmm.fit(all_goals)
            self._team_hmm[_FALLBACK_LABEL] = fallback_hmm

        self._is_fitted = True

    def transform(self, records: list[MatchRecord]) -> pd.DataFrame:
        """Compute HMM features for each match.

        Parameters
        ----------
        records:
            Match records to transform.  Sorted internally by
            ``(date, league, match_id)`` for deterministic ordering.

        Returns
        -------
        pd.DataFrame
            Columns:
            ``match_id`` (index),
            ``home_hmm_expected_goals``,
            ``away_hmm_expected_goals``,
            ``home_hmm_state_entropy``,
            ``away_hmm_state_entropy``,
            ``home_hmm_top_state``,
            ``away_hmm_top_state``,
            ``home_hmm_var``,
            ``away_hmm_var``,
            ``hmm_implied_p_over25``.
        """
        if not self._is_fitted:
            raise RuntimeError("HMMFeatureExtractor not fitted; call fit() first.")

        sorted_records = sorted(records, key=lambda r: (r.date, r.league, r.match_id))

        rows = []
        for r in sorted_records:
            home_result = self._predict_team(r.home_team, r.date)
            away_result = self._predict_team(r.away_team, r.date)

            row = {
                "match_id": r.match_id,
                "home_hmm_expected_goals": home_result.expected_goals,
                "away_hmm_expected_goals": away_result.expected_goals,
                "home_hmm_state_entropy": home_result.state_entropy,
                "away_hmm_state_entropy": away_result.state_entropy,
                "home_hmm_top_state": int(np.argmax(home_result.state_probs)),
                "away_hmm_top_state": int(np.argmax(away_result.state_probs)),
                "home_hmm_var": self._weighted_variance(
                    home_result.state_probs, home_result.means, home_result.vars
                ),
                "away_hmm_var": self._weighted_variance(
                    away_result.state_probs, away_result.means, away_result.vars
                ),
                "hmm_implied_p_over25": expected_goals_to_p_over25(
                    home_result.expected_goals, away_result.expected_goals
                ),
            }
            rows.append(row)

        df = pd.DataFrame(rows).set_index("match_id")
        return df

    def transform_aggregate(
        self,
        records: list[MatchRecord],
        history: list[MatchRecord],
    ) -> pd.DataFrame:
        """Convenience: ``fit(history)`` then ``transform(records)``.

        This is the recommended entry-point for time-series-safe evaluation.
        The HMM parameters are estimated on *history* alone, and features are
        computed for *records* without any leakage.

        Parameters
        ----------
        records:
            Target match records (the "test set").
        history:
            Historical match records used for training (the "training set").

        Returns
        -------
        pd.DataFrame
            HMM features (see :meth:`transform`).
        """
        self.fit(history)
        return self.transform(records)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_hmm(self, team: str) -> TeamHMM:
        """Return the HMM for *team*, falling back to the league average."""
        return self._team_hmm.get(team, self._team_hmm.get(_FALLBACK_LABEL))

    def _predict_team(self, team: str, before: object) -> HMMResult:
        """Run ``predict_next`` on *team*'s goals strictly before *before_date*.

        If the team has no history before the given date, or if no HMM
        exists for the team (including fallback), returns a zero-entropy
        result with expected_goals = 0.0.

        Results are cached per team since the HMM and goal history are
        fixed after ``fit()``.
        """
        if team in self._team_predict_cache:
            return self._team_predict_cache[team]

        hmm = self._get_hmm(team)
        if hmm is None:
            # No HMM at all (should not happen if fallback exists).
            K = self.config.n_states
            uniform = np.full(K, 1.0 / K)
            result = HMMResult(
                state_probs=uniform,
                expected_goals=0.0,
                state_entropy=float(-np.sum(uniform * np.log(uniform))),
                means=np.full(K, 0.0),
                vars=np.full(K, self.config.regularization),
            )
            self._team_predict_cache[team] = result
            return result

        # Get goals strictly before the match date.
        team_goals = self._team_goals.get(team)
        if team_goals is None:
            # Team not seen in training data — use fallback with uniform probs.
            K = self.config.n_states
            uniform = np.full(K, 1.0 / K)
            eg = float(uniform @ hmm.means_)
            ent = float(-np.sum(uniform * np.log(uniform)))
            result = HMMResult(
                state_probs=uniform,
                expected_goals=eg,
                state_entropy=ent,
                means=hmm.means_.copy(),
                vars=hmm.vars_.copy(),
            )
            self._team_predict_cache[team] = result
            return result

        result = hmm.predict_next(team_goals)
        self._team_predict_cache[team] = result
        return result

    @staticmethod
    def _weighted_variance(
        state_probs: np.ndarray,
        means: np.ndarray,
        vars: np.ndarray,
    ) -> float:
        """Compute the variance of the mixture distribution.

        ``Var[goals] = E[Var|state] + Var[E|state]``
                  ``= sum(p_k * var_k) + sum(p_k * mean_k^2) - (sum(p_k * mean_k))^2``
        """
        p = state_probs
        expected_mean = float(p @ means)
        var_expectation = float(p @ (means**2)) - expected_mean**2
        expected_var = float(p @ vars)
        return float(var_expectation + expected_var)
