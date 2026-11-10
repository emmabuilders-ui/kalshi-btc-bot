# Export the frozen live Correction ML (corr_live.py, fit Dec-Aug, no quote-age inputs) to JSON and check the pure-Python evaluator.
import json, pickle, numpy as np, pandas as pd, sys
sys.path.insert(0, '/home/user/kalshi-btc-bot')
from kalshi_btc_bot.model_gap_paper import DirectModel
m, CL = pickle.load(open('work/corr_live_model.pkl', 'rb'))
trees = []
for (p,) in m._predictors:
    n = p.nodes
    trees.append({'f': n['feature_idx'].astype(int).tolist(), 'th': [float(x) for x in n['num_threshold']], 'l': n['left'].astype(int).tolist(),
                  'r': n['right'].astype(int).tolist(), 'leaf': n['is_leaf'].astype(int).tolist(), 'v': [float(x) for x in n['value']],
                  'ml': n['missing_go_to_left'].astype(int).tolist()})
out = {'name': 'Correction ML live (BRTI + Kalshi mid/spread/volume, no quote ages), fit Dec 10 2025 - Aug 31 2026', 'features': CL,
       'baseline': float(np.ravel(m._baseline_prediction)[0]), 'trees': trees}
json.dump(out, open('/home/user/kalshi-btc-bot/kalshi_btc_bot/data/model_gap_correction.json', 'w'), separators=(',', ':'))
S = pd.read_parquet('work/models_scored.parquet'); S = S[S.snap_ts >= pd.Timestamp('2026-09-01', tz='UTC').timestamp()].sample(3000, random_state=0)
dm = DirectModel(out); pp = np.array([dm.predict(r) for r in S[CL].to_dict('records')])
print('max |python - sklearn|', np.abs(pp - m.predict_proba(S[CL])[:, 1]).max(), 'trees', len(trees))
