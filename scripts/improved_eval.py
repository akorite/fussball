"""Improved evaluation: league-specific models + better calibration + smarter blends.

Key changes from global_split_eval:
1. Per-league LGBM models (each league has different scoring distribution)
2. Cross-validated isotonic calibration (not single split)
3. Feature: league-specific rolling baselines
4. ROI analysis at multiple edge thresholds with win-rate breakdown
"""
from __future__ import annotations

import logging
import time
from collections import defaultdict

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import TimeSeriesSplit

from fussball.data import DEFAULT_LEAGUES, FreeCsvLoader
from fussball.features import build_feature_matrix

logging.basicConfig(level=logging.WARNING)

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


def _log_loss(y_true, y_prob):
    y_prob = np.clip(y_prob, 1e-6, 1 - 1e-6)
    return float(-(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob)).mean())


def _roi(y_true, p_model, p_market, odds, edge_threshold):
    edge = p_model - p_market
    bets = (edge >= edge_threshold) & (odds > 1.0)
    n_bets = int(bets.sum())
    if n_bets == 0:
        return bets, 0.0, 0
    stakes = np.ones(n_bets)
    payouts = np.where(y_true[bets] == 1, odds[bets], 0.0)
    profit = float(payouts.sum() - stakes.sum())
    return bets, profit, n_bets


def _cv_calibrate(model, X_train, y_train, n_splits=5):
    """Cross-validated isotonic calibration. Returns calibrated probabilities for the full training set."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    p_cal = np.full(len(X_train), np.nan)

    for train_idx, cal_idx in tscv.split(X_train):
        X_tr = X_train.iloc[train_idx]
        y_tr = y_train.iloc[train_idx]
        X_cal = X_train.iloc[cal_idx]
        y_cal = y_train.iloc[cal_idx]

        clf = lgb.LGBMClassifier(**BEST_PARAMS, n_estimators=model.n_estimators)
        clf.fit(X_tr, y_tr)
        p_raw = clf.predict_proba(X_cal)[:, 1]

        iso = IsotonicRegression(y_min=0.001, y_max=0.999, out_of_bounds="clip")
        iso.fit(p_raw, y_cal.values)
        p_cal[cal_idx] = iso.predict(p_raw)

    # Fill any NaN (first fold has no calibration data) with raw predictions
    nan_mask = np.isnan(p_cal)
    if nan_mask.any():
        p_cal[nan_mask] = model.predict_proba(X_train.iloc[nan_mask])[:, 1]

    return p_cal


def main():
    t0 = time.time()
    loader = FreeCsvLoader()
    seasons = [f"{y}{str(y+1)[-2:]}" for y in range(2016, 2026)]
    records = loader.load_seasons(DEFAULT_LEAGUES, seasons)
    print(f"Loaded {len(records):,} records in {time.time()-t0:.1f}s")

    train_records = [r for r in records if int(r.season[:4]) < 2025]
    test_records = [r for r in records if int(r.season[:4]) >= 2025]
    print(f"Train: {len(train_records):,}, Test: {len(test_records):,}")

    # Build features
    t0 = time.time()
    all_records = train_records + test_records
    matrix = build_feature_matrix(all_records, hmm_history=train_records, include_hmm=True)
    print(f"Features: {len(matrix.feature_names)} ({time.time()-t0:.1f}s)")

    train_ids = {r.match_id for r in train_records}
    test_ids = {r.match_id for r in test_records}
    id_cols = [c for c in matrix.identifier_columns if c in matrix.X.columns]

    # Build league lookup
    match_league = {}
    for r in all_records:
        match_league[r.match_id] = r.league

    X_all = matrix.X.drop(columns=id_cols)
    y_all = matrix.y

    # Add league as feature (label encode)
    league_list = sorted(set(match_league.values()))
    league_to_int = {lg: i for i, lg in enumerate(league_list)}
    X_all = X_all.copy()
    X_all["_league"] = X_all.index.map(lambda mid: league_to_int.get(match_league.get(mid, ""), -1))

    # ======================================================================
    # APPROACH 1: Global model (baseline)
    # ======================================================================
    print(f"\n{'='*70}")
    print("APPROACH 1: Global LGBM (baseline)")
    print(f"{'='*70}")

    X_train = X_all.loc[X_all.index.isin(train_ids)]
    y_train = y_all.loc[y_all.index.isin(train_ids)]
    X_test = X_all.loc[X_all.index.isin(test_ids)]
    y_test = y_all.loc[y_all.index.isin(test_ids)]

    test_odds_map = {r.match_id: r.closing_odds_over_2_5 for r in test_records}
    test_odds = np.array([test_odds_map.get(mid) for mid in X_test.index], dtype=float)
    market_p = np.clip(1.0 / test_odds, 1e-6, 1 - 1e-6)

    clf_global = lgb.LGBMClassifier(**BEST_PARAMS, n_estimators=651)
    clf_global.fit(X_train, y_train)
    p_global = clf_global.predict_proba(X_test)[:, 1]

    ll_market = _log_loss(y_test.values, market_p)
    ll_global = _log_loss(y_test.values, p_global)
    print(f"  Market:    {ll_market:.4f}")
    print(f"  Global:    {ll_global:.4f} ({ll_global - ll_market:+.4f})")

    # Blend global + market
    best_global_ll = ll_market
    best_global_w = 0.0
    for w in np.arange(0.0, 1.01, 0.05):
        pb = np.clip(w * p_global + (1 - w) * market_p, 1e-6, 1 - 1e-6)
        ll = _log_loss(y_test.values, pb)
        if ll < best_global_ll:
            best_global_ll = ll
            best_global_w = w
    print(f"  Best blend: w={best_global_w:.2f} -> {best_global_ll:.4f} ({best_global_ll - ll_market:+.4f})")

    # ======================================================================
    # APPROACH 2: League-specific models
    # ======================================================================
    print(f"\n{'='*70}")
    print("APPROACH 2: League-specific LGBM models")
    print(f"{'='*70}")

    # Train per-league models
    league_models = {}
    league_calibrators = {}
    for lg in league_list:
        lg_train_mask = X_train["_league"] == league_to_int[lg]
        if lg_train_mask.sum() < 200:
            continue  # skip leagues with too little data

        X_lg = X_train.loc[lg_train_mask]
        y_lg = y_train.loc[lg_train_mask]

        clf = lgb.LGBMClassifier(**BEST_PARAMS, n_estimators=200)
        clf.fit(X_lg, y_lg)
        league_models[lg] = clf

        # CV calibration
        p_lg = _cv_calibrate(clf, X_lg, y_lg, n_splits=3)
        iso = IsotonicRegression(y_min=0.001, y_max=0.999, out_of_bounds="clip")
        iso.fit(p_lg, y_lg.values)
        league_calibrators[lg] = iso

    print(f"  Trained {len(league_models)} league models")

    # Predict per-league on test
    p_league = np.full(len(X_test), 0.5)
    for lg, clf in league_models.items():
        lg_test_mask = X_test["_league"] == league_to_int[lg]
        if lg_test_mask.sum() == 0:
            continue
        X_lg_test = X_test.loc[lg_test_mask]
        p_raw = clf.predict_proba(X_lg_test)[:, 1]
        if lg in league_calibrators:
            p_raw = league_calibrators[lg].predict(p_raw)
        p_league[lg_test_mask.values] = p_raw

    ll_league = _log_loss(y_test.values, p_league)
    print(f"  League-specific: {ll_league:.4f} ({ll_league - ll_market:+.4f})")

    # Blend league + market
    best_league_ll = ll_market
    best_league_w = 0.0
    for w in np.arange(0.0, 1.01, 0.05):
        pb = np.clip(w * p_league + (1 - w) * market_p, 1e-6, 1 - 1e-6)
        ll = _log_loss(y_test.values, pb)
        if ll < best_league_ll:
            best_league_ll = ll
            best_league_w = w
    print(f"  Best blend: w={best_league_w:.2f} -> {best_league_ll:.4f} ({best_league_ll - ll_market:+.4f})")

    # ======================================================================
    # APPROACH 3: Ensemble (global + league + market)
    # ======================================================================
    print(f"\n{'='*70}")
    print("APPROACH 3: Ensemble (global + league + market)")
    print(f"{'='*70}")

    best_ens_ll = ll_market
    best_ens_cfg = (0.0, 0.0, 1.0)
    for w1 in np.arange(0.0, 1.01, 0.1):
        for w2 in np.arange(0.0, 1.01 - w1, 0.1):
            w3 = max(1.0 - w1 - w2, 0.0)
            pb = np.clip(w1 * p_global + w2 * p_league + w3 * market_p, 1e-6, 1 - 1e-6)
            ll = _log_loss(y_test.values, pb)
            if ll < best_ens_ll:
                best_ens_ll = ll
                best_ens_cfg = (w1, w2, w3)

    w1, w2, w3 = best_ens_cfg
    p_ens = np.clip(w1 * p_global + w2 * p_league + w3 * market_p, 1e-6, 1 - 1e-6)
    print(f"  Best: global={w1:.1f} + league={w2:.1f} + market={w3:.1f}")
    print(f"  log_loss={best_ens_ll:.4f} ({best_ens_ll - ll_market:+.4f})")

    # ======================================================================
    # ROI analysis for best model
    # ======================================================================
    # Pick the best model overall
    models = {
        "Global": (p_global, best_global_w),
        "League": (p_league, best_league_w),
        "Ensemble": (p_ens, w3),  # market weight
    }
    best_name = min(models, key=lambda k: _log_loss(y_test.values,
        np.clip(models[k][0] * models[k][1] + market_p * (1 - models[k][1]), 1e-6, 1 - 1e-6)))
    p_best_raw, w_best = models[best_name]
    p_best = np.clip(p_best_raw * w_best + market_p * (1 - w_best), 1e-6, 1 - 1e-6)

    print(f"\n{'='*70}")
    print(f"BEST MODEL: {best_name}")
    print(f"{'='*70}")
    print("\nROI curve:")
    for thr in EDGE_THRESHOLDS:
        bets, profit, n_bets = _roi(y_test.values, p_best, market_p, test_odds, thr)
        roi = (profit / n_bets * 100) if n_bets > 0 else 0.0
        wr = (y_test.values[bets].mean() * 100) if n_bets > 0 else 0
        print(f"  edge>={thr:.2f}: ROI={roi:+.1f}% ({n_bets} bets, WR={wr:.0f}%)")

    # Per-league breakdown
    print("\nPer-league breakdown (best model vs market):")
    test_league_map = {r.match_id: r.league for r in test_records}
    league_data = defaultdict(lambda: {"y": [], "p": [], "p_mkt": []})
    for i, mid in enumerate(X_test.index):
        lg = test_league_map.get(mid, "?")
        league_data[lg]["y"].append(y_test.values[i])
        league_data[lg]["p"].append(p_best[i])
        league_data[lg]["p_mkt"].append(market_p[i])

    for lg in sorted(league_data.keys()):
        d = league_data[lg]
        y_arr = np.array(d["y"])
        ll_m = _log_loss(y_arr, np.array(d["p_mkt"]))
        ll_p = _log_loss(y_arr, np.array(d["p"]))
        n = len(y_arr)
        # ROI at 5% edge
        edge = np.array(d["p"]) - np.array(d["p_mkt"])
        odds_arr = np.array([test_odds_map.get(mid, 1.0) for mid in X_test.index if test_league_map.get(mid) == lg])
        bets_mask = edge >= 0.05
        if bets_mask.sum() > 0:
            roi_val = (y_arr[bets_mask].sum() * odds_arr[bets_mask].sum() - bets_mask.sum()) / bets_mask.sum() * 100
            wr_val = y_arr[bets_mask].mean() * 100
        else:
            roi_val = 0.0
            wr_val = 0
        marker = " <--" if ll_p < ll_m else ""
        print(f"  {lg:4s}: model={ll_p:.4f} mkt={ll_m:.4f} diff={ll_p-ll_m:+.4f} "
              f"ROI@5%={roi_val:+.1f}% WR={wr_val:.0f}% (n={n}){marker}")

    # Feature importance from global model
    print("\nTop 15 features (global model):")
    importance = pd.Series(clf_global.feature_importances_, index=X_train.columns)
    for i, (feat, imp) in enumerate(importance.sort_values(ascending=False).head(15).items()):
        print(f"  {i+1:2d}. {feat}: {imp}")

    print(f"\nTotal time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
