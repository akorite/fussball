"""Hidden Markov Model module for latent team-form state discovery.

This module provides:

- :class:`TeamHMM` — a Gaussian-emission HMM trained via Baum-Welch
  (forward-backward in log-space).  Learns K latent "form" states from a
  team's historical goal-scored sequence.
- :func:`expected_goals_to_p_over25` — converts expected home/away goals to
  an over-2.5 probability under independent Poisson assumptions.
- :class:`HMMFeatureExtractor` — applies :class:`TeamHMM` across all teams
  in a dataset and produces match-level features (expected goals, state
  entropy, implied over-2.5 probability, etc.).
"""

from __future__ import annotations

from fussball.hmm.extractor import HMMFeatureExtractor
from fussball.hmm.team_hmm import (
    HMMResult,
    TeamHMM,
    TeamHMMConfig,
    expected_goals_to_p_over25,
)

__all__ = [
    "HMMFeatureExtractor",
    "HMMResult",
    "TeamHMM",
    "TeamHMMConfig",
    "expected_goals_to_p_over25",
]
