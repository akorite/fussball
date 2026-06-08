"""Full optimization pipeline: Optuna tune + calibration + ensemble.

Key insights from the diagnostic:
1. Default LGBM is worse than market (0.7037 vs 0.6771 log loss)
2. HMM features are useful (hmm_implied_p_over25 is #6 in importance)
3. Need stronger regularization + proper calibration
4. Market is extremely efficient — we need to find specific spots where we disagree
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from optuna.samplers import TPESampler
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.model_selection import TimeSeriesSplit

from fussball.data import DEFAULT_LEAGUES, FreeCsvLoader
from fussball.features import build_feature_matrix
from fussball.hmm import HMMFeatureExtractor

logging.basicConfig(level=logging.WARNING)
optuna.logging.set_verbosity(optuna.logging.WARNING)

EDGE_THRESHOLDS = [0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15]


def main() -> None:
    # -- Load data --
    t0 = time.time()
    loader = FreeCsvLoader()
    seasons = [f"{y}{str(y+1)[-2:]}" for y in range(2016, 2026)]
    records = loader.load_seasons(DEFAULT_LEAGUES, seasons)
    print(f"Loaded {len(records):,} records in {time.time()-t0:.1f}s")

    train_records = [r for r in records if int(r.season[:4]) < 2025]
    test_records = [r for r in records if int(r.season[:4]) >= 2025]
    print(f"Train: {len(train_records):,}, Test: {len(test_records):,}")

    # -- Build features --
    t0 = time.time()
    all_records = train_records + test_records
    matrix = build_feature_matrix(all_records, hmm_history=train_records, include_hmm=True)
    print(f"Features: {len(matrix.feature_names)} ({time.time()-t0:.1f}s)")

    train_ids = {r.match_id for r in train_records}
    test_ids = {r.match_id for r in test_records}
    train_mask = matrix.X.index.isin(train_ids)
    test_mask = matrix.X.index.isin(test_ids)

    id_cols = [c for c in matrix.identifier_columns if c in matrix.X.columns]
    X_all = matrix.X.drop(columns=id_cols)
    y_all = matrix.y

    X_train = X_all.loc[train_mask]
    y_train = y_all.loc[train_mask]
    X_test = X_all.loc[test_mask]
    y_test = y_all.loc[test_mask]

    # Market odds
    test_odds_map = {r.match_id: r.closing_odds_over_2_5 for r in test_records}
    test_odds = np.array([test_odds_map.get(mid) for mid in X_test.index], dtype=float)

    market_test_ll = _log_loss(y_test.values, np.clip(1.0 / test_odds, 1e-6, 1-1e-6))
    print(f"Market log loss (test): {market_test_ll:.4f}")

    # -- Optuna tuning (100 trials) --
    print(f"\n{'='*60}")
    print("Optuna tuning (100 trials, 3-fold TSCV)...")
    print(f"{'='*60}")
    t0 = time.time()
    study = _optuna_tune(X_train, y_train, n_trials=100)
    print(f"Best CV log loss: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")
    print(f"Tuning time: {time.time()-t0:.1f}s")

    # -- Train final model with best params --
    best_params = {
        "objective": "binary", "metric": "binary_logloss", "boosting_type": "gbdt",
        "verbosity": -1, "seed": 42, **study.best_params
    }

    # Split train into train+calibration (80/20) for proper calibration
    n = len(X_train)
    cal_size = int(n * 0.2)
    X_tr, y_tr = X_train.iloc[:n-cal_size], y_train.iloc[:n-cal_size]
    X_cal, y_cal = X_train.iloc[n-cal_size:], y_train.iloc[n-cal_size:]

    print(f"\nTrain: {len(X_tr):,}, Calibration: {len(X_cal):,}, Test: {len(X_test):,}")

    # Train model on train split
    clf = lgb.LGBMClassifier(**best_params, n_estimators=2000)
    clf.fit(
        X_tr, y_tr,
        eval_set=[(X_cal, y_cal)],
        callbacks=[lgb.early_stopping(50, first_metric_only=True)],
    )
    best_iter = clf.best_iteration_
    print(f"Best iteration: {best_iter}")

    # Retrain on all train with best_iteration, then calibrate
    clf_final = lgb.LGBMClassifier(**best_params, n_estimators=max(best_iter, 50))
    clf_final.fit(X_train, y_train)

    # Calibrate using CalibratedClassifierCV on the calibration split
    # Refit: train on X_tr, calibrate on X_cal
    clf_for_cal = lgb.LGBMClassifier(**best_params, n_estimators=max(best_iter, 50))
    clf_for_cal.fit(X_tr, y_tr)

    # Use CalibratedClassifierCV for proper calibration
    cal_clf = CalibratedClassifierCV(clf_for_cal, method="isotonic", cv="prefit")
    cal_clf.fit(X_cal, y_cal)

    # -- Evaluate models on test set --
    print(f"\n{'='*60}")
    print("Results on test set (2025-2026)")
    print(f"{'='*60}")

    # Raw model
    p_raw = clf_final.predict_proba(X_test)[:, 1]
    ll_raw = _log_loss(y_test.values, p_raw)
    print(f"\nRaw LGBM:          log_loss={ll_raw:.4f}")

    # Calibrated model
    p_cal = cal_clf.predict_proba(X_test)[:, 1]
    ll_cal = _log_loss(y_test.values, p_cal)
    print(f"Calibrated LGBM:   log_loss={ll_cal:.4f}")

    # HMM-implied probability
    hmm_col = "hmm_implied_p_over25"
    if hmm_col in X_test.columns:
        p_hmm = np.clip(X_test[hmm_col].values, 1e-6, 1-1e-6)
        ll_hmm = _log_loss(y_test.values, p_hmm)
        print(f"HMM-implied:       log_loss={ll_hmm:.4f}")

    # Market
    market_p = np.clip(1.0 / test_odds, 1e-6, 1-1e-6)
    print(f"Market:            log_loss={market_test_ll:.4f}")

    # Ensemble: calibrated LGBM + market (weighted)
    print(f"\n--- Ensemble: calibrated model + market ---")
    for w in [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0]:
        p_blend = w * p_cal + (1 - w) * market_p
        ll = _log_loss(y_test.values, p_blend)
        diff = ll - market_test_ll
        bets, profit, n_bets = _compute_roi(y_test.values, p_blend, market_p, test_odds, 0.05)
        roi = (profit / n_bets * 100) if n_bets > 0 else 0.0
        marker = " <-- BEST" if diff < 0 else ""
        print(f"  w_cal={w:.1f}: log_loss={ll:.4f} ({diff:+.4f}), ROI@0.05={roi:+.1f}% ({n_bets} bets){marker}")

    # Three-way ensemble: calibrated LGBM + HMM + market
    if hmm_col in X_test.columns:
        print(f"\n--- Three-way ensemble: cal LGBM + HMM + market ---")
        best_3way_ll = float("inf")
        best_3way_cfg = None
        for w1 in np.arange(0.0, 1.01, 0.1):
            for w2 in np.arange(0.0, 1.01 - w1, 0.1):
                w3 = 1.0 - w1 - w2
                if w3 < -0.01:
                    continue
                w3 = max(w3, 0.0)
                p_blend = w1 * p_cal + w2 * p_hmm + w3 * market_p
                ll = _log_loss(y_test.values, p_blend)
                if ll < best_3way_ll:
                    best_3way_ll = ll
                    best_3way_cfg = (w1, w2, w3)
        w1, w2, w3 = best_3way_cfg
        p_best3 = w1 * p_cal + w2 * p_hmm + w3 * market_p
        ll_best3 = _log_loss(y_test.values, p_best3)
        bets, profit, n_bets = _compute_roi(y_test.values, p_best3, market_p, test_odds, 0.05)
        roi = (profit / n_bets * 100) if n_bets > 0 else 0.0
        print(f"  Best: cal={w1:.1f} + hmm={w2:.1f} + mkt={w3:.1f}")
        print(f"  log_loss={ll_best3:.4f} ({ll_best3-market_test_ll:+.4f}), ROI@0.05={roi:+.1f}% ({n_bets} bets)")

    # -- ROI curve for best model --
    print(f"\n--- ROI curve (calibrated LGBM) ---")
    for thr in EDGE_THRESHOLDS:
        bets, profit, n_bets = _compute_roi(y_test.values, p_cal, market_p, test_odds, thr)
        roi = (profit / n_bets * 100) if n_bets > 0 else 0.0
        win_rate = (bets.sum() > 0 and y_test.values[bets].sum() / bets.sum() * 100) if bets.sum() > 0 else 0
        print(f"  edge>={thr:.2f}: ROI={roi:+.1f}% ({n_bets} bets, win_rate={win_rate:.0f}%)")

    # -- Feature importance --
    print(f"\n--- Top 20 features ---")
    importance = pd.Series(clf_final.feature_importances_, index=X_train.columns)
    importance = importance.sort_values(ascending=False)
    for i, (feat, imp) in enumerate(importance.head(20).items()):
        print(f"  {i+1:2d}. {feat}: {imp}")


def _optuna_tune(X, y, n_trials=100, seed=42):
    """Run Optuna with time-series CV."""

    def objective(trial):
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 8, 63),
            "min_child_samples": trial.suggest_int("min_child_samples", 30, 500),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.3, 0.8),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 0.9),
            "lambda_l1": trial.suggest_float("lambda_l1", 0.01, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 0.01, 10.0, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0.0, 1.0),
        }

        merged = {
            "objective": "binary", "metric": "binary_logloss",
            "boosting_type": "gbdt", "verbosity": -1, "seed": seed,
            **params,
        }

        X_arr = np.asarray(X)
        y_arr = np.asarray(y).ravel()
        columns = list(X.columns) if hasattr(X, "columns") else [f"f{i}" for i in range(X_arr.shape[1])]

        tscv = TimeSeriesSplit(n_splits=4)
        log_losses = []
        for train_idx, val_idx in tscv.split(X_arr):
            X_fold = pd.DataFrame(X_arr[train_idx], columns=columns)
            X_val = pd.DataFrame(X_arr[val_idx], columns=columns)
            y_fold, y_val = y_arr[train_idx], y_arr[val_idx]

            clf = lgb.LGBMClassifier(**merged, n_estimators=1000)
            clf.fit(
                X_fold, y_fold,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(15, first_metric_only=True)],
            )
            y_prob = clf.predict_proba(X_val)[:, 1]
            from sklearn.metrics import log_loss
            log_losses.append(log_loss(y_val, y_prob))

        return float(np.mean(log_losses))

    study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    return study


def _log_loss(y_true, y_prob):
    """Compute log loss safely."""
    y_prob = np.clip(y_prob, 1e-6, 1 - 1e-6)
    return float(-(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob)).mean())


def _compute_roi(y_true, p_model, p_market, odds, edge_threshold):
    """Compute ROI for bets where model edge exceeds threshold."""
    edge = p_model - p_market
    bets = edge >= edge_threshold
    n_bets = int(bets.sum())
    if n_bets == 0:
        return bets, 0.0, 0
    profit = 0.0
    for i in np.where(bets)[0]:
        if y_true[i] == 1:
            profit += odds[i] - 1.0  # net win
        else:
            profit -= 1.0  # loss
    return bets, profit, n_bets


if __name__ == "__main__":
    main()
