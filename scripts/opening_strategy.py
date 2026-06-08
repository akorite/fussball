"""Evaluate whether betting at opening odds beats closing odds.

Compares four timing strategies for over/under 2.5 goal betting:
  A) Bet at OPENING odds when model says over
  B) Bet at CLOSING odds when model says over
  C) Bet at OPENING odds when opening price > closing price (market moved against)
  D) Bet at OPENING odds when model AND opening price agree
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

P = {
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


def _roi(y_true: np.ndarray, odds: np.ndarray, mask: np.ndarray) -> tuple[int, float, int]:
    """Return (bet_indices, total_profit, n_bets)."""
    bets = np.where(mask)[0]
    n_bets = len(bets)
    if n_bets == 0:
        return bets, 0.0, 0
    wins = y_true[bets] == 1
    profit = float(np.sum(np.where(wins, odds[bets] - 1.0, -1.0)))
    return bets, profit, n_bets


def main() -> None:
    t0 = time.time()
    loader = FreeCsvLoader()
    seasons = [f"{y}{str(y + 1)[-2:]}" for y in range(2016, 2026)]
    records = loader.load_seasons(DEFAULT_LEAGUES, seasons)
    print(f"Loaded {len(records):,} records in {time.time() - t0:.1f}s")

    train_records = [r for r in records if int(r.season[:4]) < 2025]
    test_records = [r for r in records if int(r.season[:4]) >= 2025]
    print(f"Train: {len(train_records):,}, Test: {len(test_records):,}")

    # Build features (no HMM for speed)
    t0 = time.time()
    all_records = train_records + test_records
    matrix = build_feature_matrix(all_records, include_hmm=False)
    print(f"Features: {len(matrix.feature_names)} ({time.time() - t0:.1f}s)")

    # Split by match_id
    train_ids = {r.match_id for r in train_records}
    test_ids = {r.match_id for r in test_records}
    id_cols = [c for c in matrix.identifier_columns if c in matrix.X.columns]
    X_all = matrix.X.drop(columns=id_cols)
    y_all = matrix.y

    train_mask = X_all.index.isin(train_ids)
    test_mask = X_all.index.isin(test_ids)
    X_train, y_train = X_all.loc[train_mask], y_all.loc[train_mask]
    X_test, y_test = X_all.loc[test_mask], y_all.loc[test_mask]

    # Grab odds from test records
    test_map = {r.match_id: r for r in test_records}
    opening_odds = np.array(
        [test_map[mid].odds_over_2_5 for mid in X_test.index], dtype=float
    )
    closing_odds = np.array(
        [test_map[mid].closing_odds_over_2_5 for mid in X_test.index], dtype=float
    )

    # Replace NaN with 0 (will be masked later)
    has_opening = np.isfinite(opening_odds) & (opening_odds > 1.0)
    has_closing = np.isfinite(closing_odds) & (closing_odds > 1.0)
    has_both = has_opening & has_closing

    print(f"\nOdds availability: opening={has_opening.sum()}, closing={has_closing.sum()}, both={has_both.sum()} of {len(X_test)}")

    # Train LGBM on pre-2025 data
    n = len(X_train)
    cal_size = int(n * 0.2)
    X_tr, y_tr = X_train.iloc[: n - cal_size], y_train.iloc[: n - cal_size]
    X_cal, y_cal = X_train.iloc[n - cal_size :], y_train.iloc[n - cal_size :]

    clf = lgb.LGBMClassifier(**P, n_estimators=2000)
    clf.fit(
        X_tr,
        y_tr,
        eval_set=[(X_cal, y_cal)],
        callbacks=[lgb.early_stopping(50, first_metric_only=True)],
    )
    best_iter = clf.best_iteration_
    print(f"Best iteration: {best_iter}")

    # Retrain on full train set
    clf_final = lgb.LGBMClassifier(**P, n_estimators=max(best_iter, 50))
    clf_final.fit(X_train, y_train)

    # Calibrate on held-out portion
    p_cal_raw = clf.predict_proba(X_cal)[:, 1]
    iso = IsotonicRegression(y_min=0.001, y_max=0.999, out_of_bounds="clip")
    iso.fit(p_cal_raw, y_cal.values)

    # Test predictions
    p_raw = clf_final.predict_proba(X_test)[:, 1]
    p_model = np.clip(iso.predict(p_raw), 1e-6, 1 - 1e-6)

    # Implied probabilities from odds
    open_p = np.where(has_opening, 1.0 / opening_odds, 0.5)
    close_p = np.where(has_closing, 1.0 / closing_odds, 0.5)
    open_p = np.clip(open_p, 1e-6, 1 - 1e-6)
    close_p = np.clip(close_p, 1e-6, 1 - 1e-6)

    # Market movement: positive = closing implies more over (price dropped)
    market_movement = close_p - open_p

    # === Print summary stats ===
    print(f"\n{'=' * 70}")
    print("OPENING vs CLOSING ODDS ANALYSIS")
    print(f"{'=' * 70}")
    print(f"Test matches (with both odds): {has_both.sum()}")
    print(f"Over 2.5 hit rate: {y_test.values[has_both].mean():.1%}")
    print(f"Avg opening odds: {opening_odds[has_both].mean():.3f}")
    print(f"Avg closing odds: {closing_odds[has_both].mean():.3f}")
    print(f"Avg implied p (opening): {open_p[has_both].mean():.3f}")
    print(f"Avg implied p (closing): {close_p[has_both].mean():.3f}")
    print(f"Avg market movement (close-open): {market_movement[has_both].mean():+.4f}")
    print(f"  (positive = closing more confident in over)")

    # === Strategy comparison at multiple edge thresholds ===
    print(f"\n{'=' * 70}")
    print("STRATEGY COMPARISON — ROI CURVE")
    print(f"{'=' * 70}")
    print(f"{'Edge':>6s} | {'A: Open+Model':>14s} | {'B: Close+Model':>14s} | {'C: Open+Move':>14s} | {'D: Open+Both':>14s}")
    print(f"{'':>6s} | {'ROI%  Bets':>14s} | {'ROI%  Bets':>14s} | {'ROI%  Bets':>14s} | {'ROI%  Bets':>14s}")
    print("-" * 70)

    for thr in EDGE_THRESHOLDS:
        # Strategy A: bet at OPENING odds, model says over
        mask_a = has_both & ((p_model - open_p) > thr)
        _, profit_a, n_a = _roi(y_test.values, opening_odds, mask_a)
        roi_a = profit_a / n_a * 100 if n_a > 0 else 0.0

        # Strategy B: bet at CLOSING odds, model says over
        mask_b = has_both & ((p_model - close_p) > thr)
        _, profit_b, n_b = _roi(y_test.values, closing_odds, mask_b)
        roi_b = profit_b / n_b * 100 if n_b > 0 else 0.0

        # Strategy C: bet at OPENING odds, opening price > closing price
        # (market moved toward over = opening offered better value)
        mask_c = has_both & (open_p > close_p + thr)
        _, profit_c, n_c = _roi(y_test.values, opening_odds, mask_c)
        roi_c = profit_c / n_c * 100 if n_c > 0 else 0.0

        # Strategy D: bet at OPENING odds, model AND opening price agree
        # (both model and opening odds see value in over)
        mask_d = has_both & ((p_model - open_p) > thr) & (open_p > close_p)
        _, profit_d, n_d = _roi(y_test.values, opening_odds, mask_d)
        roi_d = profit_d / n_d * 100 if n_d > 0 else 0.0

        print(
            f"{thr:>6.2f} | {roi_a:>+6.1f}% {n_a:>5d} | "
            f"{roi_b:>+6.1f}% {n_b:>5d} | "
            f"{roi_c:>+6.1f}% {n_c:>5d} | "
            f"{roi_d:>+6.1f}% {n_d:>5d}"
        )

    # === Detailed view: A vs B at 5% edge ===
    print(f"\n{'=' * 70}")
    print("DETAILED: Opening vs Closing odds, model edge > 5%")
    print(f"{'=' * 70}")
    thr = 0.05
    mask_a = has_both & ((p_model - open_p) > thr)
    mask_b = has_both & ((p_model - close_p) > thr)

    n_a = mask_a.sum()
    n_b = mask_b.sum()
    bets_a = np.where(mask_a)[0]
    bets_b = np.where(mask_b)[0]

    if n_a > 0:
        wr_a = y_test.values[bets_a].mean()
        avg_odds_a = opening_odds[bets_a].mean()
        _, profit_a, _ = _roi(y_test.values, opening_odds, mask_a)
        print(f"  A (opening+model): {n_a} bets, WR={wr_a:.1%}, avg odds={avg_odds_a:.3f}, ROI={profit_a/n_a*100:+.1f}%")

    if n_b > 0:
        wr_b = y_test.values[bets_b].mean()
        avg_odds_b = closing_odds[bets_b].mean()
        _, profit_b, _ = _roi(y_test.values, closing_odds, mask_b)
        print(f"  B (closing+model): {n_b} bets, WR={wr_b:.1%}, avg odds={avg_odds_b:.3f}, ROI={profit_b/n_b*100:+.1f}%")

    # Overlap analysis
    both_ab = mask_a & mask_b
    only_a = mask_a & ~mask_b
    only_b = mask_b & ~mask_a
    print(f"\n  Overlap A∩B: {both_ab.sum()}")
    print(f"  Only A: {only_a.sum()} (unique to opening)")
    print(f"  Only B: {only_b.sum()} (unique to closing)")
    if only_a.sum() > 0:
        bets_only_a = np.where(only_a)[0]
        wr_only_a = y_test.values[bets_only_a].mean()
        _, profit_only_a, _ = _roi(y_test.values, opening_odds, only_a)
        print(f"  -> Opening-only bets: WR={wr_only_a:.1%}, ROI={profit_only_a/len(bets_only_a)*100:+.1f}%")
    if only_b.sum() > 0:
        bets_only_b = np.where(only_b)[0]
        wr_only_b = y_test.values[bets_only_b].mean()
        _, profit_only_b, _ = _roi(y_test.values, closing_odds, only_b)
        print(f"  -> Closing-only bets: WR={wr_only_b:.1%}, ROI={profit_only_b/len(bets_only_b)*100:+.1f}%")

    # === Strategy C detail: pure market movement bets ===
    print(f"\n{'=' * 70}")
    print("MARKET MOVEMENT ANALYSIS — Strategy C detail")
    print(f"{'=' * 70}")
    for move_thr in [0.0, 0.01, 0.02, 0.03, 0.05]:
        mask = has_both & (open_p > close_p + move_thr)
        n = mask.sum()
        if n == 0:
            continue
        bets_idx = np.where(mask)[0]
        wr = y_test.values[bets_idx].mean()
        avg_open = opening_odds[bets_idx].mean()
        _, profit, _ = _roi(y_test.values, opening_odds, mask)
        print(f"  move>{move_thr:.2f}: {n:>4d} bets, WR={wr:.1%}, avg odds={avg_open:.3f}, ROI={profit/n*100:+.1f}%")

    # === Per-league breakdown ===
    print(f"\n{'=' * 70}")
    print("PER-LEAGUE BREAKDOWN (edge > 5%)")
    print(f"{'=' * 70}")
    print(f"{'League':>6s} | {'A: Open':>12s} | {'B: Close':>12s} | {'C: Move':>12s} | {'D: Both':>12s} | {'#Test':>5s}")
    print("-" * 70)

    league_map = {}
    for r in test_records:
        league_map[r.match_id] = r.league

    league_data: dict[str, dict] = {}
    for i, mid in enumerate(X_test.index):
        lg = league_map.get(mid, "?")
        if lg not in league_data:
            league_data[lg] = {
                "y": [], "open_odds": [], "close_odds": [],
                "open_p": [], "close_p": [], "p_model": [],
                "has_both": [],
            }
        d = league_data[lg]
        d["y"].append(y_test.values[i])
        d["open_odds"].append(opening_odds[i])
        d["close_odds"].append(closing_odds[i])
        d["open_p"].append(open_p[i])
        d["close_p"].append(close_p[i])
        d["p_model"].append(p_model[i])
        d["has_both"].append(has_both[i])

    thr = 0.05
    for lg in sorted(league_data.keys()):
        d = league_data[lg]
        y_arr = np.array(d["y"])
        ho = np.array(d["open_odds"])
        hc = np.array(d["close_odds"])
        op = np.array(d["open_p"])
        cp = np.array(d["close_p"])
        pm = np.array(d["p_model"])
        hb = np.array(d["has_both"])
        n_test = len(y_arr)

        # Strategy A: opening + model
        mask_a = hb & ((pm - op) > thr)
        na = mask_a.sum()
        _, pa, _ = _roi(y_arr, ho, mask_a)
        roi_a = pa / na * 100 if na > 0 else 0.0

        # Strategy B: closing + model
        mask_b = hb & ((pm - cp) > thr)
        nb = mask_b.sum()
        _, pb, _ = _roi(y_arr, hc, mask_b)
        roi_b = pb / nb * 100 if nb > 0 else 0.0

        # Strategy C: opening + market moved against
        mask_c = hb & (op > cp + thr)
        nc = mask_c.sum()
        _, pc, _ = _roi(y_arr, ho, mask_c)
        roi_c = pc / nc * 100 if nc > 0 else 0.0

        # Strategy D: opening + model + market
        mask_d = hb & ((pm - op) > thr) & (op > cp)
        nd = mask_d.sum()
        _, pd_val, _ = _roi(y_arr, ho, mask_d)
        roi_d = pd_val / nd * 100 if nd > 0 else 0.0

        def _fmt(roi, n):
            return f"{roi:>+5.1f}% {n:>4d}" if n > 0 else f"{'--':>5s}   0"

        print(f"{lg:>6s} | {_fmt(roi_a, na)} | {_fmt(roi_b, nb)} | {_fmt(roi_c, nc)} | {_fmt(roi_d, nd)} | {n_test:>5d}")

    # === Final verdict ===
    print(f"\n{'=' * 70}")
    print("VERDICT")
    print(f"{'=' * 70}")
    # Compare A vs B across thresholds
    a_wins = 0
    b_wins = 0
    for thr in EDGE_THRESHOLDS:
        mask_a = has_both & ((p_model - open_p) > thr)
        mask_b = has_both & ((p_model - close_p) > thr)
        _, pa, na = _roi(y_test.values, opening_odds, mask_a)
        _, pb, nb = _roi(y_test.values, closing_odds, mask_b)
        roi_a = pa / na if na > 10 else -999
        roi_b = pb / nb if nb > 10 else -999
        if roi_a > roi_b:
            a_wins += 1
        elif roi_b > roi_a:
            b_wins += 1
    print(f"  Opening odds beat closing in {a_wins}/{len(EDGE_THRESHOLDS)} threshold settings")
    print(f"  Closing odds beat opening in {b_wins}/{len(EDGE_THRESHOLDS)} threshold settings")

    # Best strategy overall
    best_roi, best_name, best_thr = -999.0, "", 0.0
    for thr in EDGE_THRESHOLDS:
        for name, mask_fn, odds_fn in [
            ("A:Open+Model", lambda p, op, cp, pm: has_both & ((pm - op) > thr), lambda: opening_odds),
            ("B:Close+Model", lambda p, op, cp, pm: has_both & ((pm - cp) > thr), lambda: closing_odds),
            ("C:Open+Move", lambda p, op, cp, pm: has_both & (op > cp + thr), lambda: opening_odds),
            ("D:Open+Both", lambda p, op, cp, pm: has_both & ((pm - op) > thr) & (op > cp), lambda: opening_odds),
        ]:
            mask = mask_fn(p_model, open_p, close_p, p_model)
            odds = odds_fn()
            _, profit, n = _roi(y_test.values, odds, mask)
            if n >= 10:
                roi = profit / n * 100
                if roi > best_roi:
                    best_roi, best_name, best_thr = roi, name, thr

    if best_name:
        print(f"\n  Best strategy: {best_name} at edge>{best_thr:.2f} → ROI={best_roi:+.1f}%")
    else:
        print("\n  No strategy achieved positive ROI with >= 10 bets.")

    print(f"\nCompleted in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
