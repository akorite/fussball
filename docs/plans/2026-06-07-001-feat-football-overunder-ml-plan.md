---
title: Football Over/Under 2.5 — ML Prediction System
type: feat
status: active
date: 2026-06-07
origin: docs/brainstorms/football-overunder-ml.md
---

# Football Over/Under 2.5 — ML Prediction System

## Summary

Build a tabular ML system (LightGBM) for personal soccer over/under 2.5 prediction across the top 5 European leagues. Six implementation units deliver an end-to-end pipeline: data loading from free static datasets, feature engineering, model training with time-series cross-validation, backtest evaluation with log loss, and a CLI for matchweek predictions. Backtest-first — prove edge on historical data before any live pipeline work.

---

## Problem Frame

Greenfield project on an empty scaffold. The environment has most of the Python ML stack ready (LightGBM 4.6, scikit-learn 1.9, pandas 3.0, Optuna 4.9). No data, no code, no project structure exists yet. The plan establishes a modern Python project layout and builds the pipeline bottom-up so each unit is testable before the next depends on it.

---

## Requirements

- R1. Historical match data covering at least 5 seasons per league for the top 5 European leagues
- R2. Per-match record: date, teams, score, shots on target, possession, corners, cards, pre-match O/U 2.5 odds (closing line)
- R3. xG data included if available within budget
- R4. Data refreshable on weekly cadence
- R5. Rolling features: team form over 5, 10, 20 matches (goals, shots, points, xG differential)
- R6. Match-level features: rest days, home/away, league position gap, head-to-head, recent same-opponent performance
- R7. Seasonal context: matchweek, stage of season, league-phase position (title/mid-table/relegation)
- R8. Calibrated over 2.5 probability output
- R9. Time-series-aware cross-validation
- R10. Log loss as primary metric; ROI, Brier score, calibration curve as secondaries; comparison vs closing odds
- R11. First milestone: backtested model with per-league and aggregate metrics
- R12. Predictions for all upcoming matchweek fixtures across tracked leagues
- R13. Prediction output: match info, over probability, market odds when available, implied edge

---

## Scope Boundaries

- Live/in-play predictions — pre-match only
- Automated betting execution — displayed recommendations only
- Public platform or leaderboard — personal use
- Leagues outside top 5 — out of scope for v1
- Over/under lines other than 2.5 — deferred
- Web UI, API, or mobile app — CLI/notebook only
- Real-time data feeds or live odds streaming — out of scope
- GPU training — CPU-only; dataset size does not warrant it

### Deferred to Follow-Up Work

- Paid API integration for live data feeds — separate effort after backtest proves edge
- Goal-distribution modeling (Poisson/Dixon-Coles layer) — future upgrade after tabular baseline
- XGBoost model comparison — can be added as an experiment in the existing framework

---

## Context & Research

### Relevant Code and Patterns

No existing code in the repo — greenfield build. The environment provides:

| Tool | Version | Status |
|------|---------|--------|
| Python | 3.12.3 | Ready |
| LightGBM | 4.6.0 | Installed |
| scikit-learn | 1.9.0 | Installed |
| pandas | 3.0.2 | Installed |
| Optuna | 4.9.0 | Installed |
| uv | 0.11.17 | Installed |
| pytest | 9.0.3 | Installed |
| XGBoost | — | Not installed (install on demand if needed) |

The user's other Python projects (in `~/Projects/Math/`) show standard Python conventions without a consistent project structure.

### External References

- Football-Data.co.uk — free CSV archives with results and odds for European leagues (used as the v1 data source)
- LightGBM documentation — standard API for classification with `objective='binary'` and `metric='binary_logloss'`
- Optuna documentation — `optuna.create_study` with `TPESampler` for hyperparameter tuning

---

## Key Technical Decisions

- **Free static datasets first (Football-Data.co.uk CSVs):** fastest path to proof of concept. Data module abstracts behind a loader interface so a paid API (Odds API, Sportmonks) can swap in later
- **LightGBM as primary model:** already installed, comparable performance to XGBoost for tabular data, faster training. XGBoost can be added as an experimental comparison later
- **File-based storage (Parquet/CSV):** no database setup or ORM complexity for v1. Transformed feature matrices stored as Parquet for efficient columnar access
- **src-layout Python package with pyproject.toml:** standard modern Python project structure; editable install via `uv pip install -e .`
- **Data module abstracts source behind a `DataSource` interface:** the rest of the pipeline talks to an abstract loader, not CSV files directly. A paid API implementation can be added without changing feature engineering or model code

---

## Open Questions

### Resolved During Planning

- **Data source choice:** start with free static datasets (Football-Data.co.uk CSVs) to prototype; design data module for paid API swap later
- **Primary model engine:** LightGBM (already installed, tabular-appropriate)

### Deferred to Implementation

- **Optimal feature window sizes:** rolling windows at 5, 10, 20 matches as starting points; final selection via feature importance analysis during iteration
- **Minimum meaningful edge threshold:** to be determined empirically from backtest results; depends on market efficiency and variance across leagues
- **Handling of missing fields in free data (possession, detailed shots):** implementer decides whether to impute, flag, or omit per-league based on data coverage rates

---

## Output Structure

```
fussball/
├── pyproject.toml
├── .gitignore
├── README.md
├── data/
│   └── raw/                    # Downloaded CSVs (gitignored)
├── src/
│   └── fussball/
│       ├── __init__.py
│       ├── cli.py              # Matchweek prediction CLI
│       ├── backtest.py         # End-to-end backtest runner
│       ├── data/
│       │   ├── __init__.py
│       │   ├── schema.py       # MatchRecord model
│       │   └── loader.py       # Free-dataset CSV loader
│       ├── features/
│       │   ├── __init__.py
│       │   ├── rolling.py      # Rolling window features
│       │   ├── match.py        # Match-level features
│       │   └── context.py      # Seasonal context features
│       └── models/
│           ├── __init__.py
│           ├── trainer.py      # LightGBM training + Optuna tuning
│           └── evaluator.py    # Log loss, Brier, calibration, ROI
├── tests/
│   ├── __init__.py
│   ├── conftest.py             # Shared fixtures (synthetic match data)
│   ├── test_data.py
│   ├── test_features.py
│   ├── test_models.py
│   ├── test_backtest.py
│   └── test_cli.py
└── notebooks/                  # Exploration notebooks (gitignored)
```

---

## Implementation Units

### U1. Project scaffolding

**Goal:** Establish the Python project structure, dependency management, and tooling so all subsequent units have a working foundation.

**Requirements:** (infrastructure for all R1–R13)

**Dependencies:** None

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `README.md`
- Create: `src/fussball/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py` (with synthetic match data fixture shared across all test modules)

**Approach:**
- Use `uv` for project management (`uv init`, `uv add` for dependencies)
- pyproject.toml with `[project]` metadata, `[dependency-groups]` for dev/test deps, and `[tool.pytest.ini_options]`
- src-layout with `src/fussball/` as the package root
- EditorConfig-style `.gitignore` excluding `data/raw/`, `notebooks/`, `*.parquet`, `*.joblib`, `__pycache__/`
- Install core dependencies: lightgbm, pandas, numpy, scikit-learn, optuna, click, pydantic, pyarrow, python-dotenv
- Install dev dependencies: pytest, pytest-cov, rich, tqdm

**Test expectation:** none — scaffolding, no behavioral code. Verification: `uv run pytest` collects zero tests without error; `uv run python -c "import fussball"` succeeds.

**Verification:**
- `uv run python -c "import fussball"` — package imports
- `uv run pytest --collect-only` — test discovery works
- `uv run pytest` — exits zero (no tests yet)

---

### U2. Data acquisition & schema

**Goal:** Define a standardized match record schema and implement a loader for free static datasets (Football-Data.co.uk CSVs).

**Requirements:** R1, R2, R3, R4

**Dependencies:** U1

**Files:**
- Create: `src/fussball/data/__init__.py`
- Create: `src/fussball/data/schema.py`
- Create: `src/fussball/data/loader.py`
- Create: `data/raw/.gitkeep`
- Test: `tests/test_data.py`

**Approach:**
- Define `MatchRecord` as a Pydantic model with all fields from R2 plus league/season metadata and a `source` field for provenance
- Define `DataSource` protocol/ABC with `load_seasons(leagues, seasons) -> list[MatchRecord]` interface for future API-backed implementations
- Implement `FreeCsvLoader(DataSource)` that:
  - Downloads or reads local Football-Data.co.uk CSV files
  - Maps CSV columns to `MatchRecord` fields (column names vary slightly by league year)
  - Logs warnings for missing optional fields (xG, possession) instead of failing
  - Validates data integrity: no negative goals, dates parse, odds are positive floats
- Add a `download_data` CLI subcommand or script to fetch CSVs

**Patterns to follow:**
- Pydantic models for data validation (common in modern Python ML projects)
- ABC/protocol pattern for swappable backends

**Test scenarios:**
- Happy path: minimal CSV with required fields parses to correct `MatchRecord` instances
- Edge case: CSV missing optional fields (possession, xG) loads without error, missing fields are None
- Edge case: CSV with extra unknown columns loads without error, unknown columns ignored
- Edge case: first 5 rows of a season where the team has no recent history (feature module handles NaN — U3 territory, but the data layer should not crash)
- Error path: CSV with negative goals raises `DataValidationError`
- Error path: CSV with unparseable date raises `DataValidationError`
- Error path: CSV file not found raises `FileNotFoundError` with clear message

**Verification:**
- `pytest --cov=src/fussball/data tests/test_data.py` — all data tests pass
- Load one real Football-Data.co.uk CSV for Premier League and verify record count matches expected row count
- All edge case and error path scenarios pass

---

### U3. Feature engineering

**Goal:** Transform raw match records into a feature matrix suitable for model training.

**Requirements:** R5, R6, R7

**Dependencies:** U2

**Files:**
- Create: `src/fussball/features/__init__.py`
- Create: `src/fussball/features/rolling.py`
- Create: `src/fussball/features/match.py`
- Create: `src/fussball/features/context.py`
- Test: `tests/test_features.py`

**Approach:**
- Rolling features (`rolling.py`): compute per-team averages over sliding windows (last 5, 10, 20 matches) for goals scored, goals conceded, shots differential, points, and xG differential when available. Window edges handled with `min_periods=1` so early-season matches still get features (partial window averages)
- Match features (`match.py`): compute rest days since last match, home/away flag, league position gap at match time, head-to-head record over last 3 meetings, recent performance against same opponent (last 2 matches)
- Seasonal context (`context.py`): matchweek number (normalized 0–1), season stage (early/mid/late as ordinal or one-hot), league-phase position mapped from league table (top 4 = title race, mid-table = 5th–last relegation zone minus 5, relegation battle = bottom 5)
- Feature matrix builder in `features/__init__.py` that chains all feature extractors and returns a single DataFrame with row-per-match

**Patterns to follow:**
- Pandas groupby-apply pattern for per-team rolling computations
- Feature extractors as pure functions taking `list[MatchRecord]` and returning `pd.DataFrame`

**Test scenarios:**
- Happy path: known match sequence produces correct rolling average (verify with hand-calculated values for a 3-match window)
- Edge case: first match of a season (no history) — rolling features use partial window and do not produce NaN
- Edge case: team promoted to a league mid-cycle (no prior season data) — features still populate from current season only
- Edge case: rest days calculation — back-to-back matches (1 day rest) and after international break (14+ days)
- Edge case: all input records for a single team — features compute but with limited history
- Error path: empty record list returns empty feature DataFrame (not crash)

**Verification:**
- Feature matrix has row count matching input match count (one row per match)
- No NaN in core numeric features for matches with sufficient history
- All feature names follow convention and are documented
- `pytest --cov=src/fussball/features tests/test_features.py` — all tests pass

---

### U4. Model training & evaluation

**Goal:** Implement LightGBM training with hyperparameter tuning and comprehensive model evaluation.

**Requirements:** R8, R9, R10

**Dependencies:** U3

**Files:**
- Create: `src/fussball/models/__init__.py`
- Create: `src/fussball/models/trainer.py`
- Create: `src/fussball/models/evaluator.py`
- Test: `tests/test_models.py`

**Approach:**
- **Trainer (`trainer.py`):**
  - `train(X, y, params)` — trains LightGBM with early stopping on validation set
  - `tune(X, y, study_config)` — Optuna hyperparameter search over learning_rate, num_leaves, min_child_samples, subsample, colsample_bytree, lambda_l1/l2, max_depth
  - Default params: `objective='binary'`, `metric='binary_logloss'`, `boosting='gbdt'`, `verbosity=-1`
  - Returns trained Booster + best params + CV scores
  - Model serialization via `lightgbm.Booster.save_model()` to `.joblib` or `.txt`
- **Evaluator (`evaluator.py`):**
  - Log loss (primary metric) via `sklearn.metrics.log_loss`
  - Brier score via `sklearn.metrics.brier_score_loss`
  - Calibration curve via `sklearn.calibration.calibration_curve` (binned into 10 bins)
  - ROI simulation: for each prediction, compare model probability to market-implied probability (from odds). Bet when edge exceeds configurable threshold. Track cumulative ROI
  - Produces `EvaluationResult` dataclass with all metrics + calibration data

**Patterns to follow:**
- sklearn-compatible scoring interface for metrics
- LightGBM's built-in early stopping on validation set

**Test scenarios:**
- Happy path: training on synthetic data converges (log loss decreases from baseline ~0.693)
- Happy path: tuned model outperforms untuned baseline on held-out validation set
- Edge case: training on a single league only produces a valid model
- Edge case: all features are zero/near-zero — model trains and produces near-0.5 probabilities
- Error path: training with fewer samples than minimum required raises clear error
- Integration: training reproducibility — same data + same seed + same params produces identical model (check `model.feature_importance()`)

**Verification:**
- Model outputs calibrated probabilities in [0, 1] range
- Log loss on validation set is < 0.65 (beats naive 0.5-predictor baseline)
- `pytest --cov=src/fussball/models tests/test_models.py` — all tests pass

---

### U5. Backtest pipeline

**Goal:** Run end-to-end time-series cross-validation across all leagues and produce the performance report that defines v1 success.

**Requirements:** R9, R10, R11

**Dependencies:** U4

**Files:**
- Create: `src/fussball/backtest.py`
- Create: `data/` directory (gitignored, for downloaded CSVs and output artifacts)
- Test: `tests/test_backtest.py`

**Approach:**
- Time-series split: for each league, train on seasons 1..N-1, test on season N. Slide window forward season by season
- Pipeline per fold: load → feature engineer → train → evaluate → record metrics
- Aggregate metrics: weighted average log loss across all folds, per-league breakdown, overall ROI curve
- Output artifacts:
  - `data/backtest_results.json` — per-fold and aggregate metrics
  - `data/calibration_curve.png` — calibration plot
  - `data/roi_by_edge_threshold.png` — ROI vs edge threshold curve
- Reports both aggregate and per-league performance so underperforming leagues are visible

**Patterns to follow:**
- Iterate over time-series splits as an explicit generator (`TimeSeriesSplit` from sklearn or custom)

**Test scenarios:**
- Happy path: full backtest on synthetic data completes and produces metrics dict with all expected keys
- Edge case: one league has fewer than the required minimum seasons — skipped with logged warning, not crash
- Edge case: edge threshold of 0% vs 5% — verify ROI differences are reported correctly
- Integration: running backtest twice produces identical results (determinism with fixed seed)

**Verification:**
- Backtest runs end-to-end without errors on at least one league's real data
- Metric report contains: per-league log loss, aggregate log loss, ROI at default edge threshold, calibration curve data
- `pytest --cov=src/fussball/backtest tests/test_backtest.py` — all tests pass

---

### U6. Prediction CLI

**Goal:** Provide a command-line interface that loads a trained model and produces over/under predictions for upcoming matches.

**Requirements:** R8, R12, R13

**Dependencies:** U5

**Files:**
- Create: `src/fussball/cli.py`
- Test: `tests/test_cli.py`
- Modify: `pyproject.toml` (add `[project.scripts]` entry point)

**Approach:**
- Click-based CLI with subcommands:
  - `fussball backtest` — run the full backtest pipeline (invokes U5)
  - `fussball predict` — load latest model, process upcoming fixtures, print predictions
  - `fussball download-data` — fetch/update raw CSVs
- `predict` flow:
  1. Check for latest model file in `data/models/`
  2. If no model exists or data is newer, prompt to run backtest first
  3. Feature-engineer upcoming fixtures
  4. Load model and predict
  5. Output formatted table: match | over prob | market odds | edge | recommendation
- Output to stdout as a formatted table (rich tables); optional `--json` flag for machine-readable output

**Patterns to follow:**
- Click CLI pattern with `@click.group()` and `@click.command()`
- Rich `Table` for formatted terminal output

**Test scenarios:**
- Happy path: `fussball predict` with a trained model produces non-empty table output
- Happy path: `fussball predict --json` produces valid JSON
- Edge case: no model file found — CLI prints clear error message telling user to run `backtest` first
- Edge case: no upcoming fixtures available — CLI prints "No upcoming matches found" and exits zero
- Error path: data loading fails — CLI prints error and exits non-zero

**Verification:**
- `fussball --help` shows all subcommands
- `fussball predict --json` outputs parseable JSON with expected keys
- `pytest tests/test_cli.py` — all tests pass

---

## System-Wide Impact

- **Interaction graph:** linear data pipeline (load → feature → train → predict). No callbacks, middleware, or event-driven paths. Each stage consumes the previous output and produces the next input
- **Error propagation:** data-layer errors (missing files, corrupt CSVs, unparseable records) raise typed exceptions caught at the CLI layer for user-friendly messages. Model-layer errors (training failure, degenerate predictions) log diagnostics and abort with non-zero exit
- **State lifecycle risks:** model files on disk can become stale if data is updated without retraining. CLI warnings alert the user when the trained model is older than the latest data
- **Unchanged invariants:** no existing APIs, services, or shared surfaces to preserve — greenfield project. All code is self-contained within this repo

---

## Risks & Dependencies

| Risk | Mitigation |
|------|------------|
| Free dataset quality varies by league/season (missing columns, inconsistent formats) | Data loader logs warnings per-league for missing fields; graceful degradation (omit missing feature, continue with available data) |
| Free datasets may not include pre-match odds for all leagues/seasons | Model can still be trained on odds-available subsets for edge measurement; backtest is honest about coverage gaps |
| No edge found in backtest (model fails to beat market) | This is a valid research outcome. The pipeline still delivers value by quantifying the absence of edge and identifying which leagues approach efficiency |
| Overfitting to limited historical data | Time-series CV across separate seasons; feature count limited relative to sample size; early stopping on validation |
| Single data source has gaps in coverage period (< 5 seasons for some leagues) | Backtest reports coverage explicitly; model trains on available data only |
| `lightgbm` system dependency (libomp on Linux) | Verify installation during U1; install via apt if needed (`libomp-dev`) |

---

## Documentation / Operational Notes

- **Data directory**: `data/raw/` is gitignored. Download CSVs via `fussball download-data` or manually from Football-Data.co.uk
- **Model storage**: trained models saved to `data/models/` (gitignored) with timestamped filenames
- **First run**: the user needs to place or download Football-Data.co.uk CSVs before any backtest or prediction. The `download-data` subcommand automates this for supported sources
- **Evaluation reports**: backtest produces text and PNG outputs in `data/` for review. No auto-deployment or monitoring

---

## Sources & References

- **Origin document:** [docs/brainstorms/football-overunder-ml.md](docs/brainstorms/football-overunder-ml.md)
- **Environment:** Python 3.12.3, LightGBM 4.6.0, scikit-learn 1.9.0, pandas 3.0.2, uv 0.11.17
- **Data source:** [Football-Data.co.uk](https://www.football-data.co.uk/) — free CSV archives
- **External reference:** [LightGBM Binary Classification](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html)
