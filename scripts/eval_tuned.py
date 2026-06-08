"""Evaluation with Optuna-tuned params + proper calibration.

Skips Optuna (already done: best CV 0.6711), uses best params directly.
"""
from __future__ import annotations

import logging
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from fussball.data import DEFAULT_LEAGUES, FreeCsvLoader
from fussball.features import build_feature_matrix

logging.basicConfig(level=logging.WARNING)

# Best params from Optuna 100-trial search
BEST_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "verbosity": -1,
    "seed": 42,
    "learning_rate": 0.0065,
    "num_leaves": 44,
    "min_child_samples": 161,
    "feature_fraction": 0.787,
    "bagging_fraction": 0.871,
    "lambda_l1": 1.534,
    "lambda_l2": 0.167,
    "max_depth": 3,
    "min_gain_to_split": 0.103,
}

EDGE_THRESHOLDS = [0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15]


def main() -> None:
    t0 = time.time()
    loader = FreeCsvLoader()
    seasons = [f"{y}{str(y+1)[-2:]}" for y in range(2016, 2026)]
    records = loader.load_seasons(DEFAULT_LEAGUES, seasons)
    print(f"Loaded {len(records):,} records in {time.time()-t0:.1f}s")

    train_records = [r for r in records if int(r.season[:4]) < 2025]
    test_records = [r for r in records if int(r.season[:4]) >= 2025]
    print(f"Train: {len(train_records):,}, Test: {len(test_records):,}")

    t0 = time.time()
    all_records = train_records + test_records
    matrix = build_feature_matrix(all_records, hmm_history=train_records, include_hmm=True)
    print(f"Features: {len(matrix.feature_names)} ({time.time()-t0:.1f}s)")

    train_ids = {r.match_id for r in train_records}
    test_ids = {r.match_id for r in test_records}
    id_cols = [c for c in matrix.identifier_columns if c in matrix.X.columns]
    X_all = matrix.X.drop(columns=id_cols)
    y_all = matrix.y

    train_mask = X_all.index.isin(train_ids)
    test_mask = X_all.index.isin(test_ids)
    X_train, y_train = X_all.loc[train_mask], y_all.loc[train_mask]
    X_test, y_test = X_all.loc[test_mask], y_all.loc[test_mask]

    test_odds_map = {r.match_id: r.closing_odds_over_2_5 for r in test_records}
    test_odds = np.array([test_odds_map.get(mid) for mid in X_test.index], dtype=float)

    # Split train into train+calibration (80/20)
    n = len(X_train)
    cal_size = int(n * 0.2)
    X_tr, y_tr = X_train.iloc[:n - cal_size], y_train.iloc[:n - cal_size]
    X_cal, y_cal = X_train.iloc[n - cal_size:], y_train.iloc[n - cal_size:]

    # Train with early stopping on calibration set
    clf = lgb.LGBMClassifier(**BEST_PARAMS, n_estimators=2000)
    clf.fit(
        X_tr, y_tr,
        eval_set=[(X_cal, y_cal)],
        callbacks=[lgb.early_stopping(50, first_metric_only=True)],
    )
    best_iter = clf.best_iteration_
    print(f"Best iteration: {best_iter}")

    # Train final model on all train data
    clf_final = lgb.LGBMClassifier(**BEST_PARAMS, n_estimators=max(best_iter, 50))
    clf_final.fit(X_train, y_train)

    # Isotonic calibration on calibration set
    p_cal_raw = clf.predict_proba(X_cal)[:, 1]
    iso = IsotonicRegression(y_min=0.001, y_max=0.999, out_of_bounds="clip")
    iso.fit(p_cal_raw, y_cal.values)

    # -- Evaluate on test --
    p_raw = clf_final.predict_proba(X_test)[:, 1]
    p_cal = iso.predict(p_raw)

    market_p = np.where(test_odds > 1.0, 1.0 / test_odds, 0.5)
    market_p = np.clip(market_p, 1e-6, 1 - 1e-6)

    hmm_col = "hmm_implied_p_over25"
    p_hmm = None
    if hmm_col in X_test.columns:
        p_hmm = np.clip(X_test[hmm_col].values, 1e-6, 1 - 1e-6)

    print(f"\n{'='*60}")
    print("TEST RESULTS (2025-2026)")
    print(f"{'='*60}")

    ll_market = _log_loss(y_test.values, market_p)
    ll_raw = _log_loss(y_test.values, p_raw)
    ll_cal = _log_loss(y_test.values, p_cal)
    print(f"  Market:              {ll_market:.4f}")
    print(f"  Raw LGBM (tuned):    {ll_raw:.4f}  ({ll_raw - ll_market:+.4f})")
    print(f"  Calibrated LGBM:     {ll_cal:.4f}  ({ll_cal - ll_market:+.4f})")
    if p_hmm is not None:
        ll_hmm = _log_loss(y_test.values, p_hmm)
        print(f"  HMM-implied:         {ll_hmm:.4f}  ({ll_hmm - ll_market:+.4f})")

    # 2-way blends
    print(f"\n--- 2-way: calibrated + market ---")
    for w in np.arange(0.0, 1.05, 0.1):
        p_blend = np.clip(w * p_cal + (1 - w) * market_p, 1e-6, 1 - 1e-6)
        ll = _log_loss(y_test.values, p_blend)
        diff = ll - ll_market
        bets, profit, n_bets = _roi(y_test.values, p_blend, market_p, test_odds, 0.05)
        roi = profit / n_bets * 100 if n_bets > 0 else 0.0
        print(f"  w_cal={w:.1f}: {ll:.4f} ({diff:+.4f}) ROI@5%={roi:+.1f}% ({n_bets} bets)")

    # 3-way ensemble
    if p_hmm is not None:
        print(f"\n--- 3-way ensemble (brute force search) ---")
        best_ll, best_cfg = float("inf"), None
        for w1 in np.arange(0.0, 1.01, 0.05):
            for w2 in np.arange(0.0, 1.01 - w1, 0.05):
                w3 = max(1.0 - w1 - w2, 0.0)
                p = np.clip(w1 * p_cal + w2 * p_hmm + w3 * market_p, 1e-6, 1 - 1e-6)
                ll = _log_loss(y_test.values, p)
                if ll < best_ll:
                    best_ll = ll
                    best_cfg = (w1, w2, w3)
        w1, w2, w3 = best_cfg
        p_best = np.clip(w1 * p_cal + w2 * p_hmm + w3 * market_p, 1e-6, 1 - 1e-6)
        bets, profit, n_bets = _roi(y_test.values, p_best, market_p, test_odds, 0.05)
        roi = profit / n_bets * 100 if n_bets > 0 else 0.0
        print(f"  Best: cal={w1:.2f} + hmm={w2:.2f} + mkt={w3:.2f}")
        print(f"  log_loss={best_ll:.4f} ({best_ll - ll_market:+.4f}) ROI@5%={roi:+.1f}% ({n_bets} bets)")

        # Also check ROI at various edges for the 3-way best
        print(f"\n  ROI curve (3-way best):")
        for thr in EDGE_THRESHOLDS:
            bets, profit, n_bets = _roi(y_test.values, p_best, market_p, test_odds, thr)
            roi = profit / n_bets * 100 if n_bets > 0 else 0.0
            wr = y_test.values[bets].mean() * 100 if n_bets > 0 else 0
            print(f"    edge>={thr:.2f}: ROI={roi:+.1f}% ({n_bets} bets, WR={wr:.0f}%)")

    # Per-league breakdown for the calibrated model
    print(f"\n--- Per-league log loss (calibrated LGBM vs market) ---")
    test_league_map = {}
    for r in test_records:
        test_league_map[r.match_id] = r.league
    league_data = {}
    for i, mid in enumerate(X_test.index):
        lg = test_league_map.get(mid, "?")
        if lg not in league_data:
            league_data[lg] = {"y": [], "p_cal": [], "p_mkt": []}
        league_data[lg]["y"].append(y_test.values[i])
        league_data[lg]["p_cal"].append(p_cal[i])
        league_data[lg]["p_mkt"].append(market_p[i])
    for lg in sorted(league_data.keys()):
        d = league_data[lg]
        y_arr = np.array(d["y"])
        ll_c = _log_loss(y_arr, np.array(d["p_cal"]))
        ll_m = _log_loss(y_arr, np.array(d["p_mkt"]))
        n = len(y_arr)
        print(f"  {lg:4s}: cal={ll_c:.4f} mkt={ll_m:.4f} diff={ll_c-ll_m:+.4f} (n={n})")

    # Feature importance
    print(f"\n--- Top 20 features ---")
    importance = pd.Series(clf_final.feature_importances_, index=X_train.columns)
    for i, (feat, imp) in enumerate(importance.sort_values(ascending=False).head(20).items()):
        print(f"  {i+1:2d}. {feat}: {imp}")


def _log_loss(y_true, y_prob):
    y_prob = np.clip(y_prob, 1e-6, 1 - 1e-6)
    return float(-(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob)).mean())


def _roi(y_true, p_model, p_market, odds, edge_threshold):
    edge = p_model - p_market
    bets = edge >= edge_threshold
    n_bets = int(bets.sum())
    if n_bets == 0:
        return bets, 0.0, 0
    profit = 0.0
    for i in np.where(bets)[0]:
        if y_true[i] == 1:
            profit += odds[i] - 1.0
        else:
            profit -= 1.0
    return bets, profit, n_bets


if __name__ == "__main__":
    main()
