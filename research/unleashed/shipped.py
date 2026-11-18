import numpy as np, pandas as pd, json, sys
sys.path.insert(0, '/home/user/kalshi-btc-bot')
from kalshi_btc_bot.model_gap_paper import DirectModel
S = pd.read_parquet('work/unleashed_pred.parquet')
dm = DirectModel(json.load(open('/home/user/kalshi-btc-bot/kalshi_btc_bot/data/model_gap_correction.json')))
cut = pd.Timestamp('2026-09-01', tz='UTC').timestamp()
m = S.snap_ts >= cut
X = S.loc[m, dm.features].to_dict('records')
ps = np.array([dm.predict(r) for r in X])
d = ps - S.loc[m, 'p'].to_numpy()
print('n', m.sum(), 'mean|diff|', np.abs(d).mean(), 'mean diff', d.mean(), 'p>0.2', (np.abs(d) > .02).mean())
S['p_retrained'] = S.p; S.loc[m, 'p'] = ps
S.to_parquet('work/unleashed_pred.parquet')
