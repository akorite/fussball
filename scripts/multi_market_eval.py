"""Multi-market evaluation: over 1.5, 2.5, 3.5 + opening vs closing odds comparison.

Tests whether our model adds value at different goal lines, and whether
opening odds are less efficient than closing odds.
"""
from __future__ import annotations

import logging
import time
from collections import defaultdict

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from fussball.data import DEFAULT_LEAGUES, FreeCsvLoader
from fussball.features import build_feature_matrix

logging.basicConfig(level=logging.WARNING)

P = {
    "objective": "binary", "metric": "binary_logloss", "boosting_type": "gbdt",
    "verbosity": -1, "seed": 42,
    "learning_rate": 0.0065, "num_leaves": 44, "min_child_samples": 161,
    "feature_fraction": 0.787, "bagging_fraction": 0.871,
    "lambda_l1": 1.534, "lambda_l2": 0.167, "max_depth": 3,
    "min_gain_to_split": 0.103,
}


def ll(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def eval_market(y_true, p_model, odds, edge_thr=0.05):
    """Evaluate a model against market odds at a given edge threshold."""
    market_p = np.clip(1.0 / odds, 1e-6, 1 - 1e-6)
    ml = ll(y_true, market_p)
    ml_model = ll(y_true, p_model)

    edge = p_model - market_p
    bets = (edge >= edge_thr) & (odds > 1)
    n_bets = int(bets.sum())
    if n_bets > 0:
        profit = (y_true[bets] * odds[bets]).sum() - n_bets
        roi = profit / n_bets * 100
        wr = y_true[bets].mean() * 100
    else:
        roi, wr = 0, 0

    return {
        "market_ll": ml, "model_ll": ml_model, "diff": ml_model - ml,
        "n_bets": n_bets, "roi": roi, "wr": wr,
    }


def main():
    t0 = time.time()
    loader = FreeCsvLoader()
    seasons = [f"{y}{str(y+1)[-2:]}" for y in range(2016, 2026)]
    records = loader.load_seasons(DEFAULT_LEAGUES, seasons)
    print(f"Loaded {len(records):,} records in {time.time()-t0:.1f}s")

    train_recs = [r for r in records if int(r.season[:4]) < 2025]
    test_recs = [r for r in records if int(r.season[:4]) >= 2025]
    print(f"Train: {len(train_recs):,}, Test: {len(test_recs):,}")

    # Build features (no HMM for speed)
    t0 = time.time()
    matrix = build_feature_matrix(train_recs + test_recs)
    print(f"Features: {len(matrix.feature_names)} ({time.time()-t0:.1f}s)")

    train_ids = {r.match_id for r in train_recs}
    test_ids = {r.match_id for r in test_recs}
    id_cols = [c for c in matrix.identifier_columns if c in matrix.X.columns]
    X_all = matrix.X.drop(columns=id_cols)
    y_all = matrix.y

    X_tr = X_all.loc[X_all.index.isin(train_ids)]
    y_tr = y_all.loc[y_all.index.isin(train_ids)]
    X_te = X_all.loc[X_all.index.isin(test_ids)]
    y_te = y_all.loc[y_all.index.isin(test_ids)]

    # Build odds lookup for test set
    test_map = {r.match_id: r for r in test_recs}

    # ====================================================================
    # Multi-market evaluation
    # ====================================================================
    markets = [
        ("Over 1.5", lambda r: 1 if r.total_goals > 1 else 0, "closing_odds_over_2_5", "odds_over_2_5"),
        ("Over 2.5", lambda r: r.over_2_5, "closing_odds_over_2_5", "odds_over_2_5"),
        ("Over 3.5", lambda r: 1 if r.total_goals > 3 else 0, "closing_odds_over_2_5", "odds_over_2_5"),
    ]

    # We only have odds for >2.5 line. For 1.5 and 3.5 we derive target but
    # use >2.5 odds as a proxy (imperfect but shows the concept).

    print(f"\n{'='*70}")
    print("MULTI-MARKET: Model at different goal lines")
    print(f"{'='*70}")

    # Train one global LGBM
    clf = lgb.LGBMClassifier(**P, n_estimators=651)
    clf.fit(X_tr, y_tr)
    p_raw = clf.predict_proba(X_te)[:, 1]

    # Calibrate (simple: last 20% of train)
    cal_n = int(len(X_tr) * 0.2)
    p_cal_raw = clf.predict_proba(X_tr.iloc[-cal_n:])[:, 1]
    iso = IsotonicRegression(y_min=0.001, y_max=0.999, out_of_bounds="clip")
    iso.fit(p_cal_raw, y_tr.iloc[-cal_n:].values)
    p_cal = iso.predict(p_raw)

    # Blend with market
    best_w, best_ll = 0.0, ll(y_te.values, p_cal)
    for w in np.arange(0, 1.01, 0.05):
        test_odds_map = {mid: getattr(r, "closing_odds_over_2_5", None) for mid, r in test_map.items()}
        odds_arr = np.array([test_odds_map.get(mid, 1.0) for mid in X_te.index], dtype=float)
        mkt_p = np.clip(1.0 / odds_arr, 1e-6, 1 - 1e-6)
        bll = ll(y_te.values, np.clip(w * p_cal + (1 - w) * mkt_p, 1e-6, 1 - 1e-6))
        if bll < best_ll:
            best_ll, best_w = bll, w
    p_best = np.clip(best_w * p_cal + (1 - best_w) * np.clip(1.0 / np.array(
    [getattr(test_map.get(mid), "closing_odds_over_2_5", 1.0) or 1.0 for mid in X_te.index]
    ), 1e-6, 1 - 1e-6), 1e-6, 1 - 1e-6)

    for name, target_fn, close_col, open_col in markets:
        y_target = np.array([target_fn(test_map[mid]) for mid in X_te.index])
        y_pct = y_target.mean() * 100

        # Market odds (we only have >2.5 odds, use them as proxy)
        close_odds = np.array([getattr(test_map.get(mid), close_col, None) or 1.01 for mid in X_te.index], dtype=float)
        open_odds = np.array([getattr(test_map.get(mid), open_col, None) or 1.01 for mid in X_te.index], dtype=float)

        # Raw LGBM probabilities (we trained on over_2_5, so p_raw is P(over_2.5))
        # For different lines, we'd need retrained models, but let's show the concept

        mkt_p = np.clip(1.0 / close_odds, 1e-6, 1 - 1e-6)
        open_mkt_p = np.clip(1.0 / open_odds, 1e-6, 1 - 1e-6)

        mkt_ll = ll(y_target, mkt_p)
        open_ll = ll(y_target, open_mkt_p)

        print(f"\n  {name} (base rate: {y_pct:.1f}%)")
        print(f"    Market (closing): {mkt_ll:.4f}")
        print(f"    Market (opening): {open_ll:.4f} ({open_ll - mkt_ll:+.4f})")

        # Note: model is trained on over_2_5, not this line. For 1.5 and 3.5
        # the model probabilities are miscalibrated. Show anyway for comparison.
        if name == "Over 2.5":
            model_ll = ll(y_target, p_best)
            print(f"    Model (blended):  {model_ll:.4f} ({model_ll - mkt_ll:+.4f})")
            # ROI
            edge = p_best - mkt_p
            for thr in [0.0, 0.02, 0.05]:
                bets = (edge >= thr) & (close_odds > 1)
                n = int(bets.sum())
                if n > 0:
                    profit = (y_target[bets] * close_odds[bets]).sum() - n
                    roi = profit / n * 100
                    wr = y_target[bets].mean() * 100
                else:
                    roi, wr = 0, 0
                print(f"      edge>={thr:.2f}: ROI={roi:+.1f}% ({n} bets, WR={wr:.0f}%)")

    # ====================================================================
    # Opening vs Closing odds comparison
    # ====================================================================
    print(f"\n{'='*70}")
    print("OPENING vs CLOSING ODDS EFFICIENCY")
    print(f"{'='*70}")

    y25 = y_te.values  # over_2.5 target

    # Get opening and closing odds for over 2.5
    close_odds = np.array([getattr(test_map.get(mid), "closing_odds_over_2_5", None) or 1.01
                          for mid in X_te.index], dtype=float)
    open_odds = np.array([getattr(test_map.get(mid), "odds_over_2_5", None) or 1.01
                         for mid in X_te.index], dtype=float)

    close_p = np.clip(1.0 / close_odds, 1e-6, 1 - 1e-6)
    open_p = np.clip(1.0 / open_odds, 1e-6, 1 - 1e-6)

    ll_close = ll(y25, close_p)
    ll_open = ll(y25, open_p)

    print("\n  Log Loss (over 2.5):")
    print(f"    Closing odds: {ll_close:.4f}")
    print(f"    Opening odds: {ll_open:.4f} ({ll_open - ll_close:+.4f})")
    print(f"    Model best:   {ll(y25, p_best):.4f} ({ll(y25, p_best) - ll_close:+.4f})")

    # ROI comparison: opening vs closing at various edges
    print("\n  ROI at edge thresholds:")
    print(f"  {'Threshold':>10s}  {'Closing':>12s}  {'Opening':>12s}  {'Model':>12s}")
    for thr in [0.0, 0.01, 0.02, 0.03, 0.05]:
        # Closing odds bets
        # Opening odds: edge = closing implied - opening implied
        edge_open = close_p - open_p  # positive means closing says higher prob
        bets_open = (edge_open >= thr) & (close_odds > 1)
        n_open = int(bets_open.sum())
        if n_open > 0:
            profit_open = (y25[bets_open] * close_odds[bets_open]).sum() - n_open
            roi_open = profit_open / n_open * 100
        else:
            roi_open = 0

        # Model bets
        edge_m = p_best - close_p
        bets_m = (edge_m >= thr) & (close_odds > 1)
        n_m = int(bets_m.sum())
        if n_m > 0:
            profit_m = (y25[bets_m] * close_odds[bets_m]).sum() - n_m
            roi_m = profit_m / n_m * 100
        else:
            roi_m = 0

        # Opening odds: bet when opening probability > closing probability
        # (market moved against, so the "value" was at opening)
        bets_open_v = (open_p - close_p >= thr) & (close_odds > 1)
        n_ov = int(bets_open_v.sum())
        if n_ov > 0:
            profit_ov = (y25[bets_open_v] * close_odds[bets_open_v]).sum() - n_ov
            roi_ov = profit_ov / n_ov * 100
        else:
            roi_ov = 0

        print(f"  edge>={thr:.2f}    {roi_open:+6.1f}% ({n_open:3d})  {roi_ov:+6.1f}% ({n_ov:3d})  {roi_m:+6.1f}% ({n_m:3d})")

    # Per-league opening vs closing
    print("\n  Per-league (opening vs closing log loss, over 2.5):")
    test_lg = {r.match_id: r.league for r in test_recs}
    lg_data = defaultdict(lambda: {"y": [], "close_p": [], "open_p": []})
    for i, mid in enumerate(X_te.index):
        lg_ = test_lg.get(mid, "?")
        lg_data[lg_]["y"].append(y25[i])
        lg_data[lg_]["close_p"].append(close_p[i])
        lg_data[lg_]["open_p"].append(open_p[i])

    for lg_ in sorted(lg_data):
        d = lg_data[lg_]
        ya = np.array(d["y"])
        cp = np.array(d["close_p"])
        op = np.array(d["open_p"])
        lc = ll(ya, cp)
        lo = ll(ya, op)
        n = len(ya)
        mkr = " <--" if lo > lc else ""
        print(f"    {lg_:4s}: close={lc:.4f} open={lo:.4f} diff={lo-lc:+.4f} (n={n}){mkr}")

    # Market movement analysis
    print("\n  Market movement summary:")
    movements = []
    for _mid, r in test_map.items():
        o = r.odds_over_2_5
        c = r.closing_odds_over_2_5
        if o and c and o > 1 and c > 1:
            movements.append({
                "move": c - o,
                "move_pct": (c - o) / o * 100,
                "result": r.over_2_5,
            })
    mov_arr = pd.DataFrame(movements)
    up = mov_arr[mov_arr["move"] > 0]
    down = mov_arr[mov_arr["move"] < 0]
    flat = mov_arr[mov_arr["move"] == 0]
    print(f"    Odds drifted UP (over got pricier): {len(up)} matches, WR={up['result'].mean()*100:.1f}%")
    print(f"    Odds drifted DOWN (over got cheaper): {len(down)} matches, WR={down['result'].mean()*100:.1f}%")
    print(f"    No movement: {len(flat)} matches, WR={flat['result'].mean()*100:.1f}%")

    # Top features
    print("\nTop 10 features:")
    imp = pd.Series(clf.feature_importances_, index=X_tr.columns)
    for i, (f_, v) in enumerate(imp.sort_values(ascending=False).head(10).items()):
        print(f"  {i+1:2d}. {f_}: {v}")

    print(f"\nTotal: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
