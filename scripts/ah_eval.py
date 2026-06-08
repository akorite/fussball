"""Asian handicap evaluation: predict goal margin, compare to market AH odds.

Key insight: Asian handicap is about goal DIFFERENCE, not total goals.
The model predicts home_goals - away_goals via LGBMRegressor, then
evaluates sign prediction accuracy and ROI vs market-implied probabilities.
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
    "objective": "regression",
    "metric": "rmse",
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

EDGE_THRESHOLDS = [0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10]


def _margin_from_records(records, match_ids):
    """Extract goal margin (home - away) for given match IDs from record list."""
    lookup = {r.match_id: r.home_goals - r.away_goals for r in records}
    return np.array([lookup[mid] for mid in match_ids], dtype=float)


def _ah_odds_from_records(records, match_ids):
    """Extract Asian handicap odds (opening, closing) for given match IDs."""
    rec_lookup = {r.match_id: r for r in records}
    open_home, open_away, close_home, close_away = [], [], [], []
    for mid in match_ids:
        r = rec_lookup.get(mid)
        open_home.append(r.opening_odds_asian_home if r and r.opening_odds_asian_home else np.nan)
        open_away.append(r.opening_odds_asian_away if r and r.opening_odds_asian_away else np.nan)
        close_home.append(r.closing_odds_asian_home if r and r.closing_odds_asian_home else np.nan)
        close_away.append(r.closing_odds_asian_away if r and r.closing_odds_asian_away else np.nan)
    return (
        np.array(open_home, dtype=float),
        np.array(open_away, dtype=float),
        np.array(close_home, dtype=float),
        np.array(close_away, dtype=float),
    )


def _implied_prob(odds):
    """Convert decimal odds to implied probability, NaN-safe."""
    p = np.where(odds > 1.0, 1.0 / odds, np.nan)
    return p


def _remove_overround(p_home, p_away):
    """Normalize AH implied probs to remove bookmaker overround."""
    total = p_home + p_away
    valid = total > 0
    p_home_fair = np.where(valid, p_home / total, np.nan)
    p_away_fair = np.where(valid, p_away / total, np.nan)
    return p_home_fair, p_away_fair


def _sign_accuracy(y_true, y_pred):
    """Fraction of correctly predicted signs."""
    correct = np.sign(y_pred) == np.sign(y_true)
    # Ignore exact zeros (draws are hard)
    nonzero = y_true != 0
    if nonzero.sum() == 0:
        return 0.0, 0
    return float(correct[nonzero].mean()), int(nonzero.sum())


def _fit_margin_to_win_prob(margins, outcomes):
    """Fit isotonic regression mapping margin -> P(home wins).

    Parameters
    ----------
    margins : array of predicted margins
    outcomes : binary array (1 if actual margin > 0, else 0)

    Returns
    -------
    callable : margin -> probability
    """
    iso = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds="clip")
    iso.fit(margins, outcomes)
    return iso


def _roi_curve(y_true, p_model, p_market, edge_thresholds):
    """Compute ROI at various edge thresholds.

    When model probability of home winning exceeds market probability by >= threshold,
    bet 1 unit on home at fair odds implied by model probability.

    Returns list of (threshold, roi%, n_bets, win_rate%).
    """
    results = []
    for thr in edge_thresholds:
        edge = p_model - p_market
        bets = edge >= thr
        n_bets = int(bets.sum())
        if n_bets == 0:
            results.append((thr, 0.0, 0, 0.0))
            continue
        # Fair odds = 1/p_model for the bet
        bet_odds = 1.0 / np.clip(p_model[bets], 0.01, 0.99)
        wins = y_true[bets]
        payouts = wins * bet_odds
        profit = float(payouts.sum() - n_bets)
        roi = profit / n_bets * 100
        wr = float(wins.mean()) * 100
        results.append((thr, roi, n_bets, wr))
    return results


def _market_roi(y_true, p_market, edge_thresholds):
    """Baseline ROI: always bet at market-implied fair odds."""
    results = []
    for thr in edge_thresholds:
        # Market always has edge=0 relative to itself
        bets = p_market > 0.5  # bet home when market says home is favorite
        n_bets = int(bets.sum())
        if n_bets == 0:
            results.append((thr, 0.0, 0, 0.0))
            continue
        bet_odds = 1.0 / np.clip(p_market[bets], 0.01, 0.99)
        wins = y_true[bets]
        payouts = wins * bet_odds
        profit = float(payouts.sum() - n_bets)
        roi = profit / n_bets * 100
        wr = float(wins.mean()) * 100
        results.append((thr, roi, n_bets, wr))
    return results


def main():
    t0 = time.time()

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    loader = FreeCsvLoader()
    seasons = [f"{y}{str(y + 1)[-2:]}" for y in range(2016, 2026)]
    records = loader.load_seasons(DEFAULT_LEAGUES, seasons)
    print(f"Loaded {len(records):,} records in {time.time() - t0:.1f}s")

    train_records = [r for r in records if int(r.season[:4]) < 2025]
    test_records = [r for r in records if int(r.season[:4]) >= 2025]
    print(f"Train: {len(train_records):,}  Test: {len(test_records):,}")

    # ------------------------------------------------------------------
    # 2. Build feature matrix (no HMM for speed)
    # ------------------------------------------------------------------
    t_feat = time.time()
    all_records = train_records + test_records
    matrix = build_feature_matrix(all_records)
    print(f"Features: {len(matrix.feature_names)} ({time.time() - t_feat:.1f}s)")

    # ------------------------------------------------------------------
    # 3. Build train/test splits
    # ------------------------------------------------------------------
    train_ids = {r.match_id for r in train_records}
    test_ids = {r.match_id for r in test_records}
    id_cols = [c for c in matrix.identifier_columns if c in matrix.X.columns]

    X_all = matrix.X.drop(columns=id_cols)
    train_mask = X_all.index.isin(train_ids)
    test_mask = X_all.index.isin(test_ids)

    X_train = X_all.loc[train_mask]
    X_test = X_all.loc[test_mask]

    # Build margin target from records
    y_train = _margin_from_records(all_records, X_train.index)
    y_test = _margin_from_records(all_records, X_test.index)

    # Binary outcomes: 1 if home wins, 0 otherwise (excluding draws for accuracy)
    y_train_win = (y_train > 0).astype(float)
    y_test_win = (y_test > 0).astype(float)

    # Asian handicap odds
    open_home, open_away, close_home, close_away = _ah_odds_from_records(all_records, X_test.index)
    n_with_ah = (~np.isnan(close_home)).sum()
    print(f"Test matches with closing AH odds: {n_with_ah}/{len(X_test)}")

    # Market implied probabilities from AH odds
    mkt_p_home_close = _implied_prob(close_home)
    mkt_p_away_close = _implied_prob(close_away)
    mkt_p_home_fair, mkt_p_away_fair = _remove_overround(mkt_p_home_close, mkt_p_away_close)

    # Opening AH odds comparison
    mkt_p_home_open = _implied_prob(open_home)
    mkt_p_away_open = _implied_prob(open_away)
    mkt_p_home_open_fair, mkt_p_away_open_fair = _remove_overround(mkt_p_home_open, mkt_p_away_open)

    # ------------------------------------------------------------------
    # 4. Train LGBMRegressor for margin prediction
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("TRAINING: LGBMRegressor for goal margin (home - away)")
    print("=" * 70)

    reg = lgb.LGBMRegressor(**P, n_estimators=500)
    reg.fit(X_train, y_train)
    y_pred_train = reg.predict(X_train)
    y_pred_test = reg.predict(X_test)

    # Training RMSE
    train_rmse = float(np.sqrt(np.mean((y_pred_train - y_train) ** 2)))
    test_rmse = float(np.sqrt(np.mean((y_pred_test - y_test) ** 2)))
    print(f"  Train RMSE: {train_rmse:.4f}")
    print(f"  Test  RMSE: {test_rmse:.4f}")

    # Correlation between predicted and actual margin
    corr = float(np.corrcoef(y_pred_test, y_test)[0, 1])
    print(f"  Test correlation (pred vs actual): {corr:.4f}")

    # ------------------------------------------------------------------
    # 5. Sign prediction accuracy
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SIGN PREDICTION ACCURACY (margin > 0 = home wins)")
    print("=" * 70)

    # Model sign accuracy
    model_acc, n_nonzero = _sign_accuracy(y_test, y_pred_test)
    print(f"  Model sign accuracy: {model_acc:.1%} ({n_nonzero} non-draw matches)")

    # Baseline: always predict home wins
    home_adv = float((y_test > 0).mean())
    print(f"  Baseline (always home): {home_adv:.1%}")

    # Market implied accuracy
    valid_mkt = ~np.isnan(mkt_p_home_fair)
    if valid_mkt.sum() > 0:
        # Market predicts home wins when fair prob > 0.5
        mkt_pred_home = mkt_p_home_fair > 0.5
        actual_home_win = y_test > 0
        mkt_correct = np.logical_and(mkt_pred_home == actual_home_win, valid_mkt)
        mkt_acc = float(mkt_correct.sum() / valid_mkt.sum())
        print(f"  Market sign accuracy:  {mkt_acc:.1%} ({valid_mkt.sum()} matches)")

        # What if market is just using closing odds as a stronger signal?
        mkt_confidence = np.abs(mkt_p_home_fair - 0.5)
        high_conf = mkt_confidence > 0.1
        if high_conf.sum() > 0:
            hc_correct = np.logical_and(
                (mkt_p_home_fair[high_conf] > 0.5) == actual_home_win[high_conf],
                valid_mkt[high_conf],
            )
            hc_acc = float(hc_correct.sum() / valid_mkt[high_conf].sum())
            print(f"  Market (high conf >10%): {hc_acc:.1%} ({valid_mkt[high_conf].sum()} matches)")

    # ------------------------------------------------------------------
    # 6. Model vs Market: probability calibration
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("MODEL vs MARKET: implied home-win probability")
    print("=" * 70)

    # Fit isotonic regression to convert predicted margin -> win probability
    iso = _fit_margin_to_win_prob(y_pred_test, y_test_win)
    model_win_prob = iso.predict(y_pred_test)

    # Compare model and market on available AH matches
    if valid_mkt.sum() > 0:
        m_model = model_win_prob[valid_mkt]
        m_market = mkt_p_home_fair[valid_mkt]
        m_actual = y_test_win[valid_mkt]

        print(f"  Model avg P(home): {m_model.mean():.3f}")
        print(f"  Market avg P(home): {m_market.mean():.3f}")
        print(f"  Actual home win rate: {m_actual.mean():.3f}")

        # Brier score
        brier_model = float(np.mean((m_model - m_actual) ** 2))
        brier_market = float(np.mean((m_market - m_actual) ** 2))
        print(f"  Brier score - Model: {brier_model:.4f}")
        print(f"  Brier score - Market: {brier_market:.4f}")
        print(f"  Brier advantage (lower=better): {brier_market - brier_model:+.4f}")

    # ------------------------------------------------------------------
    # 7. Opening vs Closing AH efficiency
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("OPENING vs CLOSING ASIAN HANDICAP EFFICIENCY")
    print("=" * 70)

    valid_both = (~np.isnan(mkt_p_home_open_fair)) & valid_mkt
    if valid_both.sum() > 0:
        open_p = mkt_p_home_open_fair[valid_both]
        close_p = mkt_p_home_fair[valid_both]
        actual = y_test_win[valid_both]

        # Brier scores
        brier_open = float(np.mean((open_p - actual) ** 2))
        brier_close = float(np.mean((close_p - actual) ** 2))
        print(f"  Opening AH Brier:  {brier_open:.4f}")
        print(f"  Closing AH Brier:  {brier_close:.4f}")
        print(f"  Closing advantage: {brier_open - brier_close:+.4f} (positive = closing better)")

        # How much did odds move?
        line_move = close_p - open_p
        print(f"  Avg probability shift (close - open): {line_move.mean():+.4f}")
        print(f"  Avg |probability shift|: {np.abs(line_move).mean():.4f}")

        # Calibration: accuracy at different confidence buckets
        print("\n  Calibration (closing AH):")
        buckets = [(0, 0.45), (0.45, 0.50), (0.50, 0.55), (0.55, 0.60),
                   (0.60, 0.70), (0.70, 1.01)]
        print(f"    {'P(home)':>10s}  {'Actual':>8s}  {'Count':>6s}  {'Brier':>8s}")
        for lo, hi in buckets:
            mask = (close_p >= lo) & (close_p < hi)
            n = int(mask.sum())
            if n == 0:
                continue
            act_rate = float(actual[mask].mean())
            avg_prob = float(close_p[mask].mean())
            brier = float(np.mean((close_p[mask] - actual[mask]) ** 2))
            print(f"    {lo:.2f}-{hi:.2f}  {act_rate:8.3f}  {n:6d}  {brier:8.4f}")

    # ------------------------------------------------------------------
    # 8. ROI analysis
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("ROI ANALYSIS: Model betting on home-win sign")
    print("=" * 70)

    if valid_mkt.sum() > 0:
        # Use model probabilities and actual outcomes on AH-available matches
        model_p = model_win_prob[valid_mkt]
        market_p = mkt_p_home_fair[valid_mkt]
        actuals = y_test_win[valid_mkt]

        print(f"\n  Model-based betting (edge = model_p - market_p):")
        print(f"    {'Edge':>6s}  {'ROI':>8s}  {'Bets':>6s}  {'Win%':>8s}")
        for thr, roi, n_bets, wr in _roi_curve(actuals, model_p, market_p, EDGE_THRESHOLDS):
            if n_bets > 0:
                print(f"    >= {thr:.2f}  {roi:+7.1f}%  {n_bets:6d}  {wr:7.1f}%")

        print(f"\n  Market baseline (bet home when market_p > 0.5):")
        print(f"    {'Edge':>6s}  {'ROI':>8s}  {'Bets':>6s}  {'Win%':>8s}")
        for thr, roi, n_bets, wr in _market_roi(actuals, market_p, EDGE_THRESHOLDS):
            if n_bets > 0:
                print(f"    >= {thr:.2f}  {roi:+7.1f}%  {n_bets:6d}  {wr:7.1f}%")

        # Edge distribution
        edge = model_p - market_p
        print(f"\n  Edge distribution:")
        print(f"    Model > Market: {(edge > 0).sum()} matches ({(edge > 0).mean():.1%})")
        print(f"    Model < Market: {(edge < 0).sum()} matches ({(edge < 0).mean():.1%})")
        print(f"    Mean edge: {edge.mean():+.4f}")
        print(f"    Std  edge: {edge.std():.4f}")

        # Best edge threshold
        best_roi = max(
            ((thr, roi, n, wr) for thr, roi, n, wr in _roi_curve(actuals, model_p, market_p, EDGE_THRESHOLDS) if n >= 20),
            key=lambda x: x[1],
            default=None,
        )
        if best_roi:
            print(f"\n  Best ROI (min 20 bets): edge>={best_roi[0]:.2f} -> {best_roi[1]:+.1f}% "
                  f"({best_roi[2]} bets, WR={best_roi[3]:.1f}%)")

    # ------------------------------------------------------------------
    # 9. Per-league breakdown
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PER-LEAGUE BREAKDOWN")
    print("=" * 70)

    # Build league lookup from test records
    test_league = {}
    for r in test_records:
        test_league[r.match_id] = r.league

    league_data = defaultdict(lambda: {
        "y_test": [], "y_pred": [], "model_p": [], "market_p": [], "actual_win": [],
    })

    for i, mid in enumerate(X_test.index):
        lg = test_league.get(mid, "?")
        d = league_data[lg]
        d["y_test"].append(y_test[i])
        d["y_pred"].append(y_pred_test[i])
        d["model_p"].append(model_win_prob[i])
        d["actual_win"].append(y_test_win[i])
        if valid_mkt[i]:
            d["market_p"].append(mkt_p_home_fair[i])

    print(f"\n  {'League':>6s}  {'N':>5s}  {'RMSE':>6s}  {'Corr':>6s}  {'SignAcc':>8s}"
          f"  {'MktBrier':>9s}  {'ModelBrier':>11s}  {'HomeWin%':>9s}")
    print(f"  {'-'*6:>6s}  {'-'*5:>5s}  {'-'*6:>6s}  {'-'*6:>6s}  {'-'*8:>8s}"
          f"  {'-'*9:>9s}  {'-'*11:>11s}  {'-'*9:>9s}")

    for lg in sorted(league_data.keys()):
        d = league_data[lg]
        y_t = np.array(d["y_test"])
        y_p = np.array(d["y_pred"])
        m_p = np.array(d["model_p"])
        a_w = np.array(d["actual_win"])
        mk_p = np.array(d["market_p"])
        n = len(y_t)

        rmse = float(np.sqrt(np.mean((y_p - y_t) ** 2)))
        corr = float(np.corrcoef(y_p, y_t)[0, 1]) if n > 1 else 0.0

        # Sign accuracy (non-draw only)
        nonzero = y_t != 0
        sign_acc = float((np.sign(y_p[nonzero]) == np.sign(y_t[nonzero])).mean()) if nonzero.sum() > 0 else 0.0

        # Brier scores
        brier_m = float(np.mean((m_p - a_w) ** 2))
        if len(mk_p) > 0:
            mk_brier = float(np.mean((mk_p - a_w[: len(mk_p)]) ** 2))
            mk_str = f"{mk_brier:.4f}"
        else:
            mk_str = "    N/A"

        home_wr = float(a_w.mean()) * 100

        marker = ""
        if len(mk_p) > 0 and brier_m < mk_brier:
            marker = " *"

        print(f"  {lg:>6s}  {n:5d}  {rmse:6.3f}  {corr:6.3f}  {sign_acc:7.1%}"
              f"  {mk_str:>9s}  {brier_m:11.4f}  {home_wr:8.1f}%{marker}")

    # ------------------------------------------------------------------
    # 10. Feature importance
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("TOP 15 FEATURES (margin predictor)")
    print("=" * 70)

    importance = pd.Series(reg.feature_importances_, index=X_train.columns)
    for i, (feat, imp) in enumerate(importance.sort_values(ascending=False).head(15).items()):
        print(f"  {i + 1:2d}. {feat}: {imp}")

    # ------------------------------------------------------------------
    # 11. Margin distribution
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("MARGIN DISTRIBUTION (test set)")
    print("=" * 70)

    margins = y_test
    print(f"  Mean margin: {margins.mean():+.3f}")
    print(f"  Std margin:  {margins.std():.3f}")
    print(f"  Home wins:   {(margins > 0).sum()} ({(margins > 0).mean():.1%})")
    print(f"  Draws:       {(margins == 0).sum()} ({(margins == 0).mean():.1%})")
    print(f"  Away wins:   {(margins < 0).sum()} ({(margins < 0).mean():.1%})")

    # Prediction vs actual scatter
    print(f"\n  Prediction accuracy by actual margin:")
    for m in [-3, -2, -1, 0, 1, 2, 3]:
        mask = y_test == m
        if mask.sum() == 0:
            continue
        avg_pred = y_pred_test[mask].mean()
        print(f"    Actual={m:+d}: avg_pred={avg_pred:+.2f} (n={mask.sum()})")

    print(f"\nTotal time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
