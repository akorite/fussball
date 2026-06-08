"""Quick backtest driver: run on all 38k records and report metrics."""
from __future__ import annotations

import logging
import time
from pathlib import Path

from fussball.backtest import BacktestConfig, run_backtest
from fussball.data import DEFAULT_LEAGUES, FreeCsvLoader

logging.basicConfig(level=logging.WARNING)

t0 = time.time()
loader = FreeCsvLoader()
seasons = [f"{y}{str(y+1)[-2:]}" for y in range(2016, 2026)]
records = loader.load_seasons(DEFAULT_LEAGUES, seasons)
print(f"Loaded {len(records):,} records in {time.time()-t0:.1f}s")

cfg = BacktestConfig(
    min_train_seasons=3,
    default_edge_threshold=0.05,
    save_artifacts=True,
    output_dir=Path("data"),
)
t0 = time.time()
result = run_backtest(records, cfg)
elapsed = time.time() - t0
print(f"\nBacktest complete in {elapsed:.1f}s ({elapsed/60:.1f}min)")
print(f"Folds completed: {len(result.folds)}, skipped: {result.n_folds_skipped}")
print(f"\nAggregate log loss: {result.aggregate_log_loss:.4f}")
print(f"Aggregate Brier:    {result.aggregate_brier:.4f}")
print(f"Aggregate accuracy: {result.aggregate_accuracy:.4f}")
print(f"\nPer-league log loss:")
for league, ll in result.per_league_log_loss.items():
    print(f"  {league}: {ll:.4f}")
print(f"\nPer-league ROI @ {cfg.default_edge_threshold:.2f} edge:")
for league, roi in result.per_league_roi.items():
    print(f"  {league}: {roi:+.2f}%")
print(f"\nROI curve:")
for thr, roi in result.aggregate_roi_curve:
    print(f"  edge>={thr:.2f}: {roi:+.2f}%")

# Market baseline
import numpy as np

market_log_loss = 0.0
n = 0
for r in records:
    if r.closing_odds_over_2_5 is not None and r.closing_odds_over_2_5 > 1.0:
        market_p = 1.0 / r.closing_odds_over_2_5
        market_p = max(min(market_p, 1 - 1e-6), 1e-6)
        market_log_loss += -(r.over_2_5 * np.log(market_p) + (1 - r.over_2_5) * np.log(1 - market_p))
        n += 1
print(f"\nMarket baseline log loss (all 38k): {market_log_loss / n:.4f} (n={n:,})")
