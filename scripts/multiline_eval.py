"""Multi-line evaluation: train separate models for different goal lines.

We only have odds for over/under 2.5, but we can derive targets for other
lines from total_goals. Training separate models per line lets each learn
the specific threshold dynamics at that line.
"""
from __future__ import annotations

import logging
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from fussball.data import DEFAULT_LEAGUES, FreeCsvLoader
from fussball.features import build_feature_matrix, select_xy

logging.basicConfig(level=logging.WARNING)

P = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "verbosity": -1,
    "seed": 42,
    "learning_rate": 0.01,
    "num_leaves": 31,
    "min_child_samples": 50,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "lambda_l1": 1.0,
    "lambda_l2": 1.0,
    "max_depth": 5,
}

GOAL_LINES = [0.5, 1.5, 2.5, 3.5, 4.5]

EDGE_THRESHOLD = 0.05


def _log_loss(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_prob = np.clip(y_prob, 1e-6, 1 - 1e-6)
    return float(
        -(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob)).mean()
    )


def main() -> None:
    t0 = time.time()

    # ---- Load data ----
    loader = FreeCsvLoader()
    seasons = [f"{y}{str(y + 1)[-2:]}" for y in range(2016, 2026)]
    records = loader.load_seasons(DEFAULT_LEAGUES, seasons)
    print(f"Loaded {len(records):,} records in {time.time() - t0:.1f}s")

    # ---- Train / test split ----
    train_records = [r for r in records if r.season_start_year < 2025]
    test_records = [r for r in records if r.season_start_year >= 2025]
    print(f"Train: {len(train_records):,}, Test: {len(test_records):,}")

    # ---- Build features (no HMM for speed) ----
    t_feat = time.time()
    all_records = train_records + test_records
    matrix = build_feature_matrix(all_records)
    print(f"Features: {len(matrix.feature_names)} ({time.time() - t_feat:.1f}s)")

    # ---- Extract feature matrix ----
    train_ids = {r.match_id for r in train_records}
    test_ids = {r.match_id for r in test_records}
    X_feat, y_default = select_xy(matrix)

    train_mask = X_feat.index.isin(train_ids)
    test_mask = X_feat.index.isin(test_ids)

    X_train = X_feat.loc[train_mask]
    X_test = X_feat.loc[test_mask]

    # ---- Goal-line target lookup from records ----
    goals_map: dict[str, int] = {r.match_id: r.total_goals for r in all_records}
    total_goals_train = np.array(
        [goals_map[mid] for mid in X_train.index], dtype=int
    )
    total_goals_test = np.array(
        [goals_map[mid] for mid in X_test.index], dtype=int
    )

    # ---- Market odds (closing over 2.5 — proxy for all lines) ----
    odds_map = {r.match_id: r.closing_odds_over_2_5 for r in test_records}
    test_odds_raw = np.array(
        [odds_map.get(mid) for mid in X_test.index], dtype=float
    )
    has_odds = ~np.isnan(test_odds_raw) & (test_odds_raw > 1.0)

    # ---- Evaluate each goal line ----
    results: list[dict] = []

    for line in GOAL_LINES:
        t_line = time.time()

        # Target: 1 if total_goals > line else 0
        y_train_line = (total_goals_train > line).astype(int)
        y_test_line = (total_goals_test > line).astype(int)

        # Skip lines with no positive or no negative examples
        if y_train_line.sum() == 0 or y_train_line.sum() == len(y_train_line):
            print(f"\n  Line {line}: skipped (degenerate target)")
            continue

        # Train classifier
        clf = lgb.LGBMClassifier(**P, n_estimators=300)
        clf.fit(X_train, y_train_line)

        # Predict
        p_model = clf.predict_proba(X_test)[:, 1]

        # Market implied probability (proxy: uses 2.5 odds for all lines)
        market_p = np.clip(1.0 / np.where(has_odds, test_odds_raw, 2.0), 1e-6, 1 - 1e-6)

        # Log loss
        ll_market = _log_loss(y_test_line, market_p)
        ll_model = _log_loss(y_test_line, p_model)

        # ROI at edge >= EDGE_THRESHOLD (only on matches with valid odds)
        edge = p_model - market_p
        bets = (edge >= EDGE_THRESHOLD) & has_odds
        n_bets = int(bets.sum())
        if n_bets > 0:
            valid_odds = test_odds_raw[bets]
            profit = float(y_test_line[bets].sum() * valid_odds.sum() - n_bets)
            roi = profit / n_bets * 100
            win_rate = float(y_test_line[bets].mean()) * 100
        else:
            profit = 0.0
            roi = 0.0
            win_rate = 0.0

        results.append(
            {
                "line": line,
                "train_pos": int(y_train_line.sum()),
                "train_neg": int(len(y_train_line) - y_train_line.sum()),
                "test_pos": int(y_test_line.sum()),
                "test_neg": int(len(y_test_line) - y_test_line.sum()),
                "ll_market": ll_market,
                "ll_model": ll_model,
                "ll_diff": ll_model - ll_market,
                "n_bets": n_bets,
                "roi": roi,
                "win_rate": win_rate,
                "elapsed": time.time() - t_line,
            }
        )
        print(
            f"  Line {line}: model={ll_model:.4f} mkt={ll_market:.4f} "
            f"diff={ll_model - ll_market:+.4f} ({time.time() - t_line:.1f}s)"
        )

    # ---- Comparison table ----
    print(f"\n{'=' * 72}")
    print("MULTI-LINE MODEL vs MARKET (closing over 2.5 as proxy)")
    print(f"{'=' * 72}")

    header = (
        f"{'Line':>5}  {'LL Model':>9}  {'LL Mkt':>9}  {'Diff':>8}  "
        f"{'Bets':>5}  {'ROI%':>7}  {'WR%':>6}  {'Test+':>6}  {'Test-':>6}"
    )
    print(header)
    print("-" * len(header))

    for r in results:
        line_label = f"{r['line']:.1f}"
        ll_diff = r["ll_diff"]
        marker = " *" if ll_diff < -0.005 else (" ." if ll_diff < 0 else "")
        print(
            f"{line_label:>5}  {r['ll_model']:>9.4f}  {r['ll_market']:>9.4f}  "
            f"{ll_diff:>+8.4f}  {r['n_bets']:>5}  {r['roi']:>+7.1f}  "
            f"{r['win_rate']:>6.1f}  {r['test_pos']:>6}  {r['test_neg']:>6}{marker}"
        )

    print()
    print("  * = model beats market by >0.005 log loss")
    print("  . = model beats market (margin)")

    # ---- Competitive analysis ----
    if results:
        print(f"\n{'=' * 72}")
        print("COMPETITIVE ANALYSIS")
        print(f"{'=' * 72}")

        best_line = min(results, key=lambda r: r["ll_diff"])
        worst_line = max(results, key=lambda r: r["ll_diff"])

        print(
            f"\n  Most competitive line: {best_line['line']:.1f} "
            f"(log loss diff {best_line['ll_diff']:+.4f})"
        )
        print(
            f"  Least competitive line: {worst_line['line']:.1f} "
            f"(log loss diff {worst_line['ll_diff']:+.4f})"
        )

        profitable = [r for r in results if r["roi"] > 0 and r["n_bets"] > 0]
        if profitable:
            best_roi = max(profitable, key=lambda r: r["roi"])
            print(
                f"\n  Best ROI line: {best_roi['line']:.1f} "
                f"(ROI {best_roi['roi']:+.1f}%, {best_roi['n_bets']} bets, "
                f"WR {best_roi['win_rate']:.1f}%)"
            )
        else:
            print("\n  No profitable lines at edge >= 0.05")

        # Where does each model rank?
        ranked = sorted(results, key=lambda r: r["ll_diff"])
        print("\n  Model advantage ranking (best to worst):")
        for i, r in enumerate(ranked, 1):
            status = "BEATS" if r["ll_diff"] < 0 else "LOSES"
            print(
                f"    {i}. Line {r['line']:.1f}: {r['ll_diff']:+.4f} "
                f"({status} market)  ROI={r['roi']:+.1f}%"
            )

    print(f"\nTotal time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
