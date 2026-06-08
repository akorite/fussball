# Fussball

**Football over/under goals prediction pipeline with 64 features across 10 European leagues.**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-190-passing-brightgreen)](#testing)

---

## What is this?

Fussball is a complete ML pipeline for predicting football match outcomes — specifically over/under goal lines. It fetches historical data from [Football-Data.co.uk](https://www.football-data.co.uk/), engineers 64 features (rolling stats, head-to-head history, rest days, derby indicators, HMM states), and trains LightGBM models to find edges against bookmaker odds.

**The honest truth:** Closing market odds are brutally efficient. This project doesn't claim to beat them — it measures *how efficiently* they're priced and identifies the narrow windows where edges exist.

## Key Findings

| Metric | Result |
|---|---|
| **Market efficiency** | Closing odds log loss: 0.6762 |
| **Model log loss** | 0.6759 (best blend) — essentially identical |
| **Opening odds edge** | +4.2% ROI at 5% edge threshold |
| **Market movement signal** | When odds drift down, that side wins 55.4% |
| **Asian handicap accuracy** | 66.5% sign prediction (vs 50.2% market) |

### What works

- **Opening odds timing** — betting at opening prices before the market corrects shows a consistent edge
- **Market movement** — the *direction* of odds movement from opening to closing predicts outcomes
- **Rest advantage** — the #4 most important feature (above many rolling stats)

### What doesn't

- Feature engineering beyond what the market already knows
- League-specific models (they overfit)
- HMM-implied probabilities as standalone predictors

## Architecture

```
fussball/
├── src/fussball/
│   ├── data/
│   │   ├── schema.py        # MatchRecord (Pydantic)
│   │   └── loader.py        # Football-Data.co.uk CSV parser
│   ├── features/
│   │   ├── rolling.py       # Rolling averages (5/10/20 matches)
│   │   ├── match.py         # Per-match stats
│   │   ├── context.py       # Season context (matchweek, position)
│   │   ├── odds.py          # Market-implied probabilities
│   │   ├── h2h.py           # Head-to-head history
│   │   ├── rest.py          # Rest days & schedule congestion
│   │   ├── derby.py         # Derby/rivalry indicators
│   │   └── hmm/             # Hidden Markov Model states
│   ├── models/
│   │   └── trainer.py       # LGBM training & tuning
│   ├── backtest.py          # Walk-forward backtesting
│   └── cli.py               # Click CLI
├── scripts/                 # Evaluation scripts
├── tests/                   # 190 tests
└── data/raw/                # CSV downloads (gitignored)
```

## Setup

```bash
# Clone
git clone https://github.com/akorite/fussball.git
cd fussball

# Install (uses uv for fast installs)
pip install -e ".[dev]"

# Or with uv
uv sync --group dev

# Verify
python -c "import fussball; print(fussball.__version__)"
```

## Quick Start

```bash
# 1. Download historical data (10 leagues, 2016-2025)
python -m fussball.cli download-data

# 2. Run backtest
python -m fussball.cli backtest

# 3. Predict upcoming matches
python -m fussball.cli predict
```

## Evaluation Scripts

```bash
# Opening vs closing odds comparison
python scripts/opening_strategy.py

# Asian handicap market analysis
python scripts/ah_eval.py

# Multi-goal-line evaluation (1.5, 2.5, 3.5)
python scripts/multiline_eval.py

# League-specific models
python scripts/league_eval.py

# Full model evaluation with feature importance
python scripts/league_eval.py
```

## Features (64 total)

| Category | Count | Examples |
|---|---|---|
| **Rolling stats** | 30 | Goals scored/conceded, shots, PPG (5/10/20 windows) |
| **Market odds** | 3 | Implied probability, log odds, overround |
| **Match context** | 8 | League position, matchweek, home/away form |
| **Head-to-head** | 8 | H2H goal avg, over/under rate, win rate |
| **Rest & schedule** | 4 | Days since last match, games in 30 days |
| **Derby indicators** | 3 | Same city, same region, rivalry intensity |
| **HMM states** | 6 | Expected goals, variance, entropy from Hidden Markov Model |
| **Other** | 2 | Home advantage, goal difference trends |

## Data

Free historical CSVs from [Football-Data.co.uk](https://www.football-data.co.uk/):

- **Leagues:** E0 (Premier League), SP1 (La Liga), D1 (Bundesliga), I1 (Serie A), F1 (Ligue 1), E1 (Championship), D2 (2. Bundesliga), SP2 (La Liga 2), I2 (Serie B), N1 (Eredivisie)
- **Seasons:** 2016-2025 (10 seasons)
- **Matches:** ~38,000 total

The `DataSource` protocol abstracts the data source — swap in a paid API (Odds API, Sportmonks) without touching downstream code.

## Testing

```bash
# Full suite
pytest

# With coverage
pytest --cov=src/fussball

# Skip slow tests
pytest -m "not slow"
```

## How it works

1. **Data loading** — CSVs are parsed into typed `MatchRecord` objects with validation
2. **Feature engineering** — 64 features computed via rolling windows, groupby transforms, and lookup tables
3. **Model training** — LightGBM with tuned hyperparameters (Optuna-optimized)
4. **Evaluation** — Global split (train pre-2025, test 2025-2026) with calibration and blending
5. **Edge detection** — Model probabilities compared to market implied probabilities

## Contributing

Contributions welcome. Areas that would actually move the needle:

- **Better data sources** — xG, possession, lineup data for older seasons
- **In-play features** — live match statistics
- **Alternative markets** — correct score, half-time/full-time
- **Ensemble methods** — combining multiple model architectures

## License

MIT
