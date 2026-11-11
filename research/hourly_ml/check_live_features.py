"""Check that the shipped bot code (kalshi_btc_bot/bandml_paper.py + data/band_ml_correction.json) reproduces the
research features and probabilities exactly on Sep-Oct 2026 band snapshots. Run from a folder holding work/brti.npy,
work/bands_live_scored.parquet (train_live.py output). Result 2026-11-11: max |p diff| 0.0 on 3,000 rows."""
import sys
import numpy as np, pandas as pd
from kalshi_btc_bot.bandml_paper import features
from kalshi_btc_bot.model_gap_paper import DirectModel
T0 = 1764460800
B = pd.Series(np.load('work/brti.npy')).ffill().bfill().to_numpy()
S = pd.read_parquet('work/bands_live_scored.parquet')
S = S[S.t >= pd.Timestamp('2026-09-01', tz='UTC').timestamp()].sample(3000, random_state=1)
dm = DirectModel.load('band_ml_correction.json')
diff = []
for r in S.itertuples():
    e = int(r.t) - T0 - 1
    f = features(B[e - 3600:e + 1], int(r.t), int(r.close), r.floor_strike, r.cap_strike, r.yes_bid, r.yes_ask)
    diff.append(abs(dm.predict(f) - r.p_live))
print('max |p diff|', max(diff), 'mean', np.mean(diff))
