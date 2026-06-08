"""Global train/test split: train on 2016-2024, test on 2025-2026.

This is faster than walk-forward and a more honest edge test (no time-decay
issues, single test period).
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve

from fussball.data import DEFAULT_LEAGUES, FreeCsvLoader
from fussball.features import build_feature_matrix

logging.basicConfig(level=logging.WARNING)


def market_log_loss(records, market_attr: str = "closing_odds_over_2_5") -> float:
    """Compute log loss if we use the market odds as our probability."""
    ll = 0.0
    n = 0
    for r in records:
        odds = getattr(r, market_attr)
        if odds is not None and odds > 1.0:
            p = max(min(1.0 / odds, 1 - 1e-6), 1e-6)
            ll += -(r.over_2_5 * np.log(p) + (1 - r.over_2_5) * np.log(1 - p))
            n += 1
    return ll / n if n > 0 else float("nan")


def main() -> None:
    t0 = time.time()
    loader = FreeCsvLoader()
    seasons = [f"{y}{str(y+1)[-2:]}" for y in range(2016, 2026)]
    records = loader.load_seasons(DEFAULT_LEAGUES, seasons)
    print(f"Loaded {len(records):,} records in {time.time()-t0:.1f}s")

    # Global split: train on 2016-2024, test on 2025-2026
    train_records = [r for r in records if int(r.season[:4]) < 2025]
    test_records = [r for r in records if int(r.season[:4]) >= 2025]
    print(f"Train: {len(train_records):,} records, Test: {len(test_records):,} records")

    print(f"\nMarket baseline log loss:")
    print(f"  Train: {market_log_loss(train_records):.4f}")
    print(f"  Test:  {market_log_loss(test_records):.4f}")

    # Build features
    t0 = time.time()
    all_records = train_records + test_records
    matrix = build_feature_matrix(all_records)
    print(f"\nFeature matrix: {matrix.X.shape} in {time.time()-t0:.1f}s")
    print(f"  Features: {len(matrix.feature_names)}")

    # Split by match_id
    train_ids = {r.match_id for r in train_records}
    test_ids = {r.match_id for r in test_records}
    train_mask = matrix.X.index.isin(train_ids)
    test_mask = matrix.X.index.isin(test_ids)

    X_train = matrix.X.loc[train_mask]
    y_train = matrix.y.loc[train_mask]
    X_test = matrix.X.loc[test_mask]
    y_test = matrix.y.loc[test_mask]

    # Drop identifier columns
    id_cols = [c for c in matrix.identifier_columns if c in X_train.columns]
    X_train_feat = X_train.drop(columns=id_cols)
    X_test_feat = X_test.drop(columns=id_cols)

    # Build market odds arrays
    train_odds_map = {r.match_id: r.closing_odds_over_2_5 for r in train_records}
    test_odds_map = {r.match_id: r.closing_odds_over_2_5 for r in test_records}
    train_odds = np.array([train_odds_map.get(mid) for mid in X_train_feat.index], dtype=float)
    test_odds = np.array([test_odds_map.get(mid) for mid in X_test_feat.index], dtype=float)

    # Train default LGBM
    from fussball.models import train, evaluate
    print("\nTraining default LGBM...")
    t0 = time.time()
    trained = train(X_train_feat, y_train)
    print(f"  Trained in {time.time()-t0:.1f}s")

    p_test = trained.model.predict_proba(X_test_feat)[:, 1]
    eval_test = evaluate(y_test.values, p_test, market_over_odds=test_odds, edge_threshold=0.05)
    print(f"\n=== Default LGBM (no tuning, no HMM) ===")
    print(f"  Test log loss: {eval_test.log_loss:.4f}")
    print(f"  Test Brier:    {eval_test.brier_score:.4f}")
    print(f"  Test accuracy: {eval_test.accuracy_at_0_5:.4f}")
    if eval_test.roi_results:
        print(f"  ROI @ 0.05:    {eval_test.roi_results.roi_percent:+.2f}% ({eval_test.roi_results.total_bets} bets)")

    # Save predictions for later analysis
    pred_df = pd.DataFrame({
        "match_id": X_test_feat.index,
        "p_lgbm": p_test,
        "y": y_test.values,
        "odds": test_odds,
    })
    pred_df.to_csv("data/baseline_predictions.csv", index=False)
    print(f"\nSaved {len(pred_df):,} test predictions to data/baseline_predictions.csv")

    # Also save market baseline + LGBM blended at various weights
    market_p_test = np.array([
        1.0 / o if o and o > 1.0 else 0.5
        for o in test_odds
    ])
    print(f"\nBlend exploration (LGBM w + market w):")
    for w in [0.0, 0.25, 0.5, 0.75, 1.0]:
        p_blend = w * p_test + (1 - w) * market_p_test
        ll = -(y_test.values * np.log(np.clip(p_blend, 1e-6, 1-1e-6))
               + (1 - y_test.values) * np.log(np.clip(1 - p_blend, 1e-6, 1-1e-6)))
        ll = ll.mean()
        # ROI @ 0.05
        bets = (p_blend - market_p_test) >= 0.05
        n_bets = int(bets.sum())
        if n_bets > 0:
            stake = 1.0
            win_mask = bets & (y_test.values == 1)
            profit = (win_mask * (test_odds - 1) - bets * stake).sum()
            roi = profit / n_bets * 100
        else:
            roi = 0.0
        print(f"  w_lgbm={w:.2f}: log_loss={ll:.4f}, ROI@0.05={roi:+.2f}% ({n_bets} bets)")


if __name__ == "__main__":
    main()
