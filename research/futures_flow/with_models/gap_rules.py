"""Regenerate the Direct ML and Correction ML NO gap-rule trades on the 15-min replay snapshots (work/replay.parquet),
using the frozen models in kalshi_btc_bot/data and the 1-s BRTI array (work/brti.npy). Mirrors kalshi_btc_bot/model_gap_paper.py:
checks at k = 1..9 minutes after the open, first side whose ask is 70-99c with model - ask - fee >= 3 pts is the contract's one look,
NO only, Direct needs cushion z > 1.0427 or breakout below the 60-min pre-open low; Correction has no condition.
Fill = one contract at the ask (no book depth in the replay). Output: work/gap_snaps.parquet (every k<=9 snapshot with p_direct,
p_corr) and work/gap_trades.parquet (the trades)."""
import sys, json, math, numpy as np, pandas as pd
sys.path.insert(0, '/home/user/kalshi-btc-bot')
from kalshi_btc_bot.model_gap_paper import features, evaluate, PX_MIN, PX_MAX, MIN_GAP, CUSHION_MIN
T0 = 1764460800

def vec_predict(spec, X):
    """Vectorised version of DirectModel.predict. X: (n, nfeat) float array (nan = missing)."""
    raw = np.full(len(X), spec['baseline'])
    for t in spec['trees']:
        leaf = np.asarray(t['leaf'], bool); f = np.asarray(t['f']); th = np.asarray(t['th'], float)
        ml = np.asarray(t['ml'], bool); l = np.asarray(t['l']); r = np.asarray(t['r']); v = np.asarray(t['v'], float)
        i = np.zeros(len(X), int)
        for _ in range(64):
            act = ~leaf[i]
            if not act.any(): break
            ia = i[act]; x = X[act, f[ia]]
            left = np.where(np.isnan(x), ml[ia], x <= th[ia])
            i[act] = np.where(left, l[ia], r[ia])
        raw += v[i]
    return 1 / (1 + np.exp(-raw))

def main():
    brti = np.load('work/brti.npy'); brti = pd.Series(brti).ffill().bfill().to_numpy()
    r = pd.read_parquet('work/replay.parquet'); r = r[r.k <= 9].sort_values(['ticker', 'k']).reset_index(drop=True)
    print('snapshots', len(r), flush=True)
    direct = json.load(open('/home/user/kalshi-btc-bot/kalshi_btc_bot/data/model_gap_direct.json'))
    corr = json.load(open('/home/user/kalshi-btc-bot/kalshi_btc_bot/data/model_gap_correction.json'))
    feats = []; pre = {}
    for row in r.itertuples(index=False):
        t = int(row.snap_ts); a = t - T0
        sec = brti[a - 3600:a + 1]
        f = features(sec, t, int(row.open_ts), int(row.close_ts), float(row.strike))
        mid = min(max(float(row.mid), 1e-4), 1 - 1e-4)
        f['x'] = math.log(mid / (1 - mid)); f['spread'] = float(row.spread); f['log_cumvol'] = math.log1p(max(float(row.cumvol), 0.0))
        if row.ticker not in pre:
            o = int(row.open_ts) - T0; seg = brti[o - 3600:o + 1]; pre[row.ticker] = (float(seg.max()), float(seg.min()))
        f['pre_hi'], f['pre_lo'] = pre[row.ticker]
        feats.append(f)
    F = pd.DataFrame(feats)
    print('features done', flush=True)
    r['p_direct'] = vec_predict(direct, F[direct['features']].to_numpy(float))
    r['p_corr'] = vec_predict(corr, F[corr['features']].to_numpy(float))
    for c in ['z', 'pre_hi', 'pre_lo', 'brti']: r['f_' + c] = F[c].to_numpy()
    r['yes_ask'] = r['ask']; r['no_ask'] = 1 - r['bid']
    r.to_parquet('work/gap_snaps.parquet')
    print('predict done', flush=True)
    for name, pcol, cond in [('Direct NO gap', 'p_direct', True), ('Correction NO gap', 'p_corr', False)]:
        rows = []
        for tk, g in r.groupby('ticker', sort=False):
            for s in g.itertuples(index=False):
                p_yes = getattr(s, pcol)
                asks = {'yes': [[s.yes_ask, 1]] if s.yes_ask == s.yes_ask else [], 'no': [[s.no_ask, 1]] if s.no_ask == s.no_ask else []}
                f = {'z': s.f_z, 'brti': s.f_brti}
                ev = evaluate(f, p_yes, asks, s.f_pre_hi, s.f_pre_lo, MIN_GAP)
                if ev['look'] is None: continue
                if ev['look'] == 'no' and (ev['condition'] or not cond):
                    rows.append({'bot': name, 'ticker': tk, 't': int(s.snap_ts), 'close': int(s.close_ts), 'k': int(s.k), 'min_left': int(s.min_left),
                                 'side': 'NO', 'price': 100 * s.no_ask, 'won': int(s.res == 0), 'p_model': 1 - p_yes, 'period': s.period, 'month': s.month,
                                 'level': 'cushion' if ev['cushion'] > CUSHION_MIN else ('breakout' if ev['breakout'] else 'gap only')})
                break  # one look per contract
        d = pd.DataFrame(rows); d['fee'] = np.ceil(7 * d.price / 100 * (1 - d.price / 100)); d['edge'] = 100 * d.won - d.price - d.fee
        print(name, len(d)); print(d.groupby('period').agg(n=('edge', 'size'), price=('price', 'mean'), win=('won', 'mean'), edge=('edge', 'mean')).round(3))
        d.to_parquet(f'work/gap_trades_{"direct" if cond else "corr"}.parquet')

if __name__ == '__main__':
    main()
