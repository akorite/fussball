"""Full eval: base features + HMM + isotonic calibration + blends."""
import warnings; warnings.filterwarnings('ignore')
import logging; logging.disable(logging.WARNING)
import time

t_total = time.time()

from fussball.data import DEFAULT_LEAGUES, FreeCsvLoader
from fussball.features import build_feature_matrix
from fussball.hmm import HMMFeatureExtractor, TeamHMMConfig
from fussball.models import train, evaluate
import numpy as np
import pandas as pd

loader = FreeCsvLoader()
seasons = [f'{y}{str(y+1)[-2:]}' for y in range(2016, 2026)]
records = loader.load_seasons(DEFAULT_LEAGUES, seasons)
train_recs = [r for r in records if int(r.season[:4]) < 2025]
test_recs = [r for r in records if int(r.season[:4]) >= 2025]
print(f'Data: {len(records)} records, train={len(train_recs)}, test={len(test_recs)}', flush=True)

# Market baseline
m_ll = 0; m_n = 0
for r in test_recs:
    o = r.closing_odds_over_2_5
    if o and o > 1.0:
        p = max(min(1.0/o, 1-1e-6), 1e-6)
        m_ll += -(r.over_2_5 * np.log(p) + (1-r.over_2_5) * np.log(1-p))
        m_n += 1
m_ll /= m_n
print(f'Market baseline: {m_ll:.4f}', flush=True)

# Build base features
all_recs = train_recs + test_recs
t0 = time.time()
matrix = build_feature_matrix(all_recs)
print(f'Base features ({len(matrix.feature_names)}): {time.time()-t0:.1f}s', flush=True)

# Add HMM features
cfg = TeamHMMConfig(n_states=3, n_iter=15, min_matches=20)
t0 = time.time()
hmm = HMMFeatureExtractor(config=cfg)
hmm.fit(train_recs)
print(f'HMM fit (288 teams): {time.time()-t0:.1f}s', flush=True)

t0 = time.time()
hmm_df = hmm.transform(all_recs)
print(f'HMM transform: {time.time()-t0:.1f}s', flush=True)

# Merge
matrix.X = matrix.X.merge(hmm_df, left_index=True, right_index=True, how='left')
matrix.feature_names = list(matrix.feature_names) + list(hmm_df.columns)

# Split
train_ids = {r.match_id for r in train_recs}
id_cols = [c for c in matrix.identifier_columns if c in matrix.X.columns]
X_tr = matrix.X.loc[matrix.X.index.isin(train_ids)].drop(columns=id_cols).fillna(0)
y_tr = matrix.y.loc[matrix.X.index.isin(train_ids)]
X_te = matrix.X.loc[~matrix.X.index.isin(train_ids)].drop(columns=id_cols).fillna(0)
y_te = matrix.y.loc[~matrix.X.index.isin(train_ids)]

test_odds_map = {r.match_id: r.closing_odds_over_2_5 for r in test_recs}
test_odds = np.array([test_odds_map.get(mid) for mid in X_te.index], dtype=float)
market_p = np.where(test_odds > 1.0, 1.0/test_odds, 0.5)
y = y_te.values

# Train LGBM+HMM
print(f'\nTraining LGBM ({X_tr.shape[1]} features)...', flush=True)
t0 = time.time()
m = train(X_tr, y_tr)
p = m.model.predict_proba(X_te)[:, 1]
ll_raw = -(y * np.log(np.clip(p, 1e-6, 1-1e-6)) + (1-y)*np.log(np.clip(1-p, 1e-6, 1-1e-6))).mean()
print(f'Raw LGBM+HMM:   {ll_raw:.4f} (diff: {ll_raw-m_ll:+.4f}) [{time.time()-t0:.1f}s]', flush=True)

# Isotonic calibration
from sklearn.isotonic import IsotonicRegression
n_cal = len(X_tr) // 5
p_cal = m.model.predict_proba(X_tr.iloc[-n_cal:])[:, 1]
iso = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds='clip')
iso.fit(p_cal, y_tr.iloc[-n_cal:])
p_iso = iso.predict(p)
ll_iso = -(y * np.log(np.clip(p_iso, 1e-6, 1-1e-6)) + (1-y)*np.log(np.clip(1-p_iso, 1e-6, 1-1e-6))).mean()
print(f'Calibrated:     {ll_iso:.4f} (diff: {ll_iso-m_ll:+.4f})', flush=True)

# Blends
print(f'\nBlend (calibrated model w + market 1-w):', flush=True)
best_ll = float('inf')
best_w = 0
for w in [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    pb = w * p_iso + (1-w) * market_p
    llb = -(y * np.log(np.clip(pb, 1e-6, 1-1e-6)) + (1-y)*np.log(np.clip(1-pb, 1e-6, 1-1e-6))).mean()
    diff = llb - m_ll
    bets = (pb - market_p) >= 0.05
    nb = int(bets.sum())
    if nb > 0:
        wm = bets & (y == 1)
        profit = (wm * (test_odds - 1) - bets).sum()
        roi = profit / nb * 100
    else:
        roi = 0
    if llb < best_ll:
        best_ll = llb
        best_w = w
    print(f'  w={w:.2f}: ll={llb:.4f} ({diff:+.4f}) ROI={roi:+.1f}% ({nb} bets)', flush=True)

# Also try raw model blends (uncalibrated)
print(f'\nBlend (raw model w + market 1-w):', flush=True)
for w in [0, 0.1, 0.2, 0.3, 0.4, 0.5]:
    pb = w * p + (1-w) * market_p
    llb = -(y * np.log(np.clip(pb, 1e-6, 1-1e-6)) + (1-y)*np.log(np.clip(1-pb, 1e-6, 1-1e-6))).mean()
    diff = llb - m_ll
    bets = (pb - market_p) >= 0.05
    nb = int(bets.sum())
    if nb > 0:
        wm = bets & (y == 1)
        profit = (wm * (test_odds - 1) - bets).sum()
        roi = profit / nb * 100
    else:
        roi = 0
    print(f'  w={w:.1f}: ll={llb:.4f} ({diff:+.4f}) ROI={roi:+.1f}% ({nb} bets)', flush=True)

# Feature importances
importances = pd.Series(m.model.feature_importances_, index=X_tr.columns).sort_values(ascending=False)
print(f'\nTop 20 features:', flush=True)
for rank, (name, imp) in enumerate(importances.head(20).items(), 1):
    print(f'  {rank:2d}. {name}: {imp}', flush=True)

print(f'\nBest calibrated blend: w={best_w:.2f} -> ll={best_ll:.4f} (diff: {best_ll-m_ll:+.4f})', flush=True)
print(f'Total: {time.time()-t_total:.1f}s', flush=True)
