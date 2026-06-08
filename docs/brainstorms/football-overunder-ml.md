---
date: 2026-06-07
topic: football-overunder-ml
---

# Football Over/Under 2.5 — ML Prediction System

## Summary

A quantitative ML system for personal soccer over/under 2.5 goals prediction across the top 5 European leagues. Starting with a tabular feature-engineered model (gradient-boosted trees), with a goal-distribution modeling layer planned as a future upgrade. The first milestone is a backtested model proving edge against market odds — no deployment infrastructure, no UI, no automated betting.

---

## Problem Frame

The user bets casually on soccer over/under 2.5 markets today, relying on gut feel and basic stats — no systematic edge. Betting markets are information-dense; consistently beating them requires a data-driven approach that most casual bettors lack. A quantitative model trained on historical match data with rich feature engineering can surface actionable signal that intuition alone misses: which matches are structurally mispriced by the market.

Numer.ai demonstrates that a rigorous quantitative approach to prediction can produce consistent edge — but no comparable open system exists for football betting markets. The project applies the same ML rigor to a domain where the user already has domain intuition but no quantitative foundation.

---

## Requirements

### Data Acquisition

- R1. Historical match data must cover at least 5 seasons for each of the top 5 European leagues.
- R2. Each match record must include: date, teams, final score, shots on target, possession, corners, cards, and pre-match over/under 2.5 odds (closing line).
- R3. Expected goals (xG) data should be included if a reliable data source is available within budget.
- R4. Data must be refreshable on a weekly cadence during the season for live predictions.

### Feature Engineering

- R5. Rolling features must capture team form over multiple windows (last 5, 10, 20 matches): average goals scored/conceded, shots differential, points, xG differential.
- R6. Match-level features must include: days of rest, home/away, league position gap, head-to-head record, and recent same-opponent performance.
- R7. Seasonal context features must include: matchweek number, stage of season (early/mid/late), and team's league-phase position (title race, mid-table, relegation battle).

### Modeling & Evaluation

- R8. The model must output a calibrated probability of over 2.5 goals for each match.
- R9. The training pipeline must use time-series-aware cross-validation (no future data leaking into training).
- R10. Model evaluation must use log loss as the primary scoring rule, with ROI, Brier score, and calibration curve as secondary diagnostics. Predictions must be compared against closing market odds to quantify edge.
- R11. The first milestone is a backtest — a trained model evaluated on historical data, producing per-league and aggregate performance metrics.

### Prediction Output

- R12. Predictions must be produced for all upcoming matchweek fixtures across the tracked leagues.
- R13. Each prediction output must include: match info, over probability, market odds (if available), and implied edge (model probability vs market-implied probability).

---

## Success Criteria

- A backtested model demonstrates positive ROI against closing over/under 2.5 odds across at least 3 of the 5 tracked leagues over a minimum 2-season test window.
- Model probabilities are well-calibrated (calibration curve within 5% of diagonal on average).
- A reproducible pipeline exists that takes raw match data and produces matchweek predictions with a single command.

---

## Scope Boundaries

- Live/in-play predictions — pre-match prediction only
- Automated betting execution — recommendations shown to the user, nothing placed automatically
- Public platform or leaderboard — personal use, no sharing or competition infrastructure
- Leagues outside top 5 (Championship, Eredivisie, Primeira Liga, etc.) — out of scope for v1
- Over/under lines other than 2.5 (1.5, 3.5, 4.5 goals) — deferred
- American football or other sports — out of scope entirely
- Mobile app, web UI, or API — out of scope; CLI or notebook-based
- Real-time data feeds or live odds streaming — out of scope

---

## Key Decisions

- **Tabular model first (XGBoost/LightGBM), goal-distribution modeling later** — fastest path to a working system; statistical rigor deferred to a future layer
- **Top 5 leagues, single market (O/U 2.5) initial scope** — focused scope for faster iteration; expansion deferred
- **Personal CLI/notebook tool** — zero UI work, all engineering effort on model quality
- **First milestone is a backtest, not a live system** — prove edge exists before building data pipeline automation
- **Primary evaluation metric is log loss** — proper scoring rule that penalizes overconfidence; ROI and Brier score are diagnostic secondaries

---

## Dependencies / Assumptions

- Historical match data with pre-match odds is available from a paid API within budget ($50–200/mo for services like Football-Data.org, Odds API, or Sportmonks)
- xG data availability depends on the selected data provider and budget tier — may be omitted in v1
- Python ML stack (XGBoost, scikit-learn, pandas) is sufficient; no GPU training needed for the dataset size
- The user will source and pay for data subscriptions

---

## Outstanding Questions

### Resolve Before Planning

*None.*

### Deferred to Planning

- [Affects R1] **Data source choice** — which API provider to use, and whether free alternatives (scraped odds, public datasets) are viable as a starting point
- [Affects R5] **Feature window sizes** — optimal rolling windows for form features to be determined during feature exploration
- [Needs research] **Threshold for meaningful edge** — what ROI threshold constitutes sufficient edge to proceed to a live system vs requiring more model iteration
