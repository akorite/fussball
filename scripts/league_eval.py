"""Fast league-specific evaluation (no HMM to keep it under 2 min)."""
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
    return float(-(y * np.log(np.clip(p, 1e-6, 1-1e-6)) + (1-y) * np.log(np.clip(1-p, 1e-6, 1-1e-6))).mean())


def main():
    t0 = time.time()
    loader = FreeCsvLoader()
    seasons = [f"{y}{str(y+1)[-2:]}" for y in range(2016, 2026)]
    records = loader.load_seasons(DEFAULT_LEAGUES, seasons)
    print(f"Loaded {len(records):,} records in {time.time()-t0:.1f}s")

    train_recs = [r for r in records if int(r.season[:4]) < 2025]
    test_recs = [r for r in records if int(r.season[:4]) >= 2025]
    print(f"Train: {len(train_recs):,}, Test: {len(test_recs):,}")

    # Build features (NO HMM - keeps it fast)
    t0 = time.time()
    matrix = build_feature_matrix(train_recs + test_recs)
    print(f"Features: {len(matrix.feature_names)} ({time.time()-t0:.1f}s)")

    train_ids = {r.match_id for r in train_recs}
    test_ids = {r.match_id for r in test_recs}
    id_cols = [c for c in matrix.identifier_columns if c in matrix.X.columns]

    X_all = matrix.X.drop(columns=id_cols)
    y_all = matrix.y

    # League column
    match_lg = {r.match_id: r.league for r in records}
    lg_list = sorted(set(match_lg.values()))
    lg2i = {lg: i for i, lg in enumerate(lg_list)}
    X_all = X_all.copy()
    X_all["_lg"] = X_all.index.map(lambda mid: lg2i.get(match_lg.get(mid, ""), -1))

    X_tr = X_all.loc[X_all.index.isin(train_ids)]
    y_tr = y_all.loc[y_all.index.isin(train_ids)]
    X_te = X_all.loc[X_all.index.isin(test_ids)]
    y_te = y_all.loc[y_all.index.isin(test_ids)]

    odds_map = {r.match_id: r.closing_odds_over_2_5 for r in test_recs}
    odds = np.array([odds_map.get(mid) for mid in X_te.index], dtype=float)
    mkt = np.clip(1.0 / odds, 1e-6, 1 - 1e-6)
    y = y_te.values

    mkt_ll = ll(y, mkt)
    print(f"\nMarket baseline: {mkt_ll:.4f}")

    # --- Global model ---
    clf_g = lgb.LGBMClassifier(**P, n_estimators=651)
    clf_g.fit(X_tr, y_tr)
    p_g = clf_g.predict_proba(X_te)[:, 1]

    g_ll = ll(y, p_g)
    print(f"Global LGBM:     {g_ll:.4f} ({g_ll-mkt_ll:+.4f})")

    # Global + market blend
    best_gw, best_gll = 0.0, mkt_ll
    for w in np.arange(0, 1.01, 0.05):
        bll = ll(y, np.clip(w * p_g + (1-w) * mkt, 1e-6, 1-1e-6))
        if bll < best_gll:
            best_gll, best_gw = bll, w
    print(f"Global blend:    {best_gll:.4f} ({best_gll-mkt_ll:+.4f}) w={best_gw:.2f}")

    # --- League-specific models ---
    lg_models = {}
    for lg in lg_list:
        mask = X_tr["_lg"] == lg2i[lg]
        if mask.sum() < 200:
            continue
        X_lg, y_lg = X_tr.loc[mask], y_tr.loc[mask]
        clf = lgb.LGBMClassifier(**P, n_estimators=200)
        clf.fit(X_lg, y_lg)
        # Simple calibrator: last 20% of training data
        n = len(X_lg)
        cal_n = max(int(n * 0.2), 50)
        p_cal = clf.predict_proba(X_lg.iloc[-cal_n:])[:, 1]
        iso = IsotonicRegression(y_min=0.001, y_max=0.999, out_of_bounds="clip")
        iso.fit(p_cal, y_lg.iloc[-cal_n:].values)
        lg_models[lg] = (clf, iso)
    print(f"League models:   {len(lg_models)}")

    # Predict league-specific
    p_lg = np.full(len(X_te), 0.5)
    for lg, (clf, iso) in lg_models.items():
        mask = X_te["_lg"] == lg2i[lg]
        if mask.sum() == 0:
            continue
        p_raw = clf.predict_proba(X_te.loc[mask])[:, 1]
        p_lg[mask.values] = iso.predict(p_raw)

    lg_ll = ll(y, p_lg)
    print(f"League-specific: {lg_ll:.4f} ({lg_ll-mkt_ll:+.4f})")

    # League + market blend
    best_lw, best_ll_ = 0.0, mkt_ll
    for w in np.arange(0, 1.01, 0.05):
        bll = ll(y, np.clip(w * p_lg + (1-w) * mkt, 1e-6, 1-1e-6))
        if bll < best_ll_:
            best_ll_, best_lw = bll, w
    print(f"League blend:    {best_ll_:.4f} ({best_ll_-mkt_ll:+.4f}) w={best_lw:.2f}")

    # --- 3-way ensemble ---
    best_e_ll, best_e = mkt_ll, (0, 0, 1)
    for w1 in np.arange(0, 1.01, 0.1):
        for w2 in np.arange(0, 1.01 - w1, 0.1):
            w3 = max(1 - w1 - w2, 0)
            bll = ll(y, np.clip(w1*p_g + w2*p_lg + w3*mkt, 1e-6, 1-1e-6))
            if bll < best_e_ll:
                best_e_ll, best_e = bll, (w1, w2, w3)
    w1, w2, w3 = best_e
    p_ens = np.clip(w1*p_g + w2*p_lg + w3*mkt, 1e-6, 1-1e-6)
    print(f"Ensemble:        {best_e_ll:.4f} ({best_e_ll-mkt_ll:+.4f}) "
          f"gw={w1:.1f} lw={w2:.1f} mw={w3:.1f}")

    # Pick best overall
    candidates = {
        "Global": (p_g, best_gw),
        "League": (p_lg, best_lw),
        "Ensemble": (p_ens, 0),  # already blended
    }
    best_name = min(candidates, key=lambda k: ll(y,
        np.clip(candidates[k][0] * candidates[k][1] + mkt * (1 - candidates[k][1]), 1e-6, 1-1e-6)
        if candidates[k][1] > 0 else candidates[k][0]))
    if best_name == "Ensemble":
        p_best = p_ens
    else:
        p_best = np.clip(candidates[best_name][0] * candidates[best_name][1]
                         + mkt * (1 - candidates[best_name][1]), 1e-6, 1-1e-6)

    print(f"\n{'='*60}")
    print(f"BEST: {best_name} (LL={ll(y, p_best):.4f})")
    print(f"{'='*60}")

    # ROI curve
    print(f"\nROI curve:")
    for thr in [0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15]:
        edge = p_best - mkt
        bets = (edge >= thr) & (odds > 1)
        n_bets = int(bets.sum())
        if n_bets > 0:
            profit = (y[bets] * odds[bets]).sum() - n_bets
            roi = profit / n_bets * 100
            wr = y[bets].mean() * 100
        else:
            roi, wr = 0, 0
        print(f"  edge>={thr:.2f}: ROI={roi:+.1f}% ({n_bets} bets, WR={wr:.0f}%)")

    # Per-league
    test_lg = {r.match_id: r.league for r in test_recs}
    lg_data = defaultdict(lambda: {"y": [], "p": [], "m": []})
    for i, mid in enumerate(X_te.index):
        lg_ = test_lg.get(mid, "?")
        lg_data[lg_]["y"].append(y[i])
        lg_data[lg_]["p"].append(p_best[i])
        lg_data[lg_]["m"].append(mkt[i])

    print(f"\nPer-league (model vs market):")
    for lg_ in sorted(lg_data):
        d = lg_data[lg_]
        ya = np.array(d["y"])
        pa = np.array(d["p"])
        ma = np.array(d["m"])
        lm = ll(ya, ma)
        lp = ll(ya, pa)
        n = len(ya)
        edge = pa - ma
        bets = edge >= 0.05
        if bets.sum() > 0:
            o = np.array([odds_map.get(mid, 1.0) for mid in X_te.index if test_lg.get(mid) == lg_])
            roi = ((ya[bets] * o[bets]).sum() - bets.sum()) / bets.sum() * 100
            wr = ya[bets].mean() * 100
        else:
            roi, wr = 0, 0
        mkr = " <--" if lp < lm else ""
        print(f"  {lg_:4s}: model={lp:.4f} mkt={lm:.4f} diff={lp-lm:+.4f} "
              f"ROI@5%={roi:+.1f}% WR={wr:.0f}% (n={n}){mkr}")

    # Top features
    print(f"\nTop 15 features:")
    imp = pd.Series(clf_g.feature_importances_, index=X_tr.columns)
    for i, (f_, v) in enumerate(imp.sort_values(ascending=False).head(15).items()):
        print(f"  {i+1:2d}. {f_}: {v}")

    print(f"\nTotal: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
