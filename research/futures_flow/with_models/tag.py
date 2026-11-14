"""Build one trade table for every bot rule with the futures flow at each trade's entry time. Output work/wm_trades.parquet."""
import sys, re, numpy as np, pandas as pd
sys.path.insert(0, '.')
from flow import Flow, T0
from flow_sec import FlowSec
FEE = lambda p: np.ceil(7 * p * (1 - p))  # p in dollars -> cents

def main():
    brti = pd.Series(np.load('work/brti.npy')).ffill().bfill().to_numpy()
    mk = pd.read_parquet('work/markets_audited.parquet').set_index('close_ts')
    out = []
    # --- 7 rules (hourly bands + Break 15-min) ---
    d = pd.read_parquet('/mnt/project-files/strategy/sizing/all_trades_7_rules.parquet')
    nb = len(d); d = d[d.t - T0 < len(brti) - 1]; print('dropped (after BRTI file end)', nb - len(d))
    d['product'] = np.where(d.bot == 'Break', '15m', 'hourly')
    d['secs_left'] = np.where(d.bot == 'Break', d.left, d.left * 60).astype(int)
    d['min_left'] = np.where(d.bot == 'Break', np.ceil(d.left / 60), d.left).astype(int)
    b = d[d['product'] == 'hourly'].copy()
    b['center'] = b.ticker.str.extract(r'-B(\d+)$')[0].astype(float)
    b['brti'] = brti[b.t.to_numpy() - T0]
    b['side'] = 'YES'  # all seven hourly rules buy YES on the band BTC is in (stays inside)
    b['dir'] = np.sign(b.center - b.brti).astype(int)  # flow toward the band centre = agrees
    b['kind'] = 'band'
    k = d[d['product'] == '15m'].copy()
    rs = pd.read_parquet('work/replay.parquet', columns=['close_ts', 'strike']).dropna().groupby('close_ts').strike.first()
    k['strike'] = k.close.map(rs).fillna(k.close.map(mk['strike']))
    nb = len(k); k = k[k.strike.notna()]; print('break trades without strike dropped', nb - len(k))
    k['brti'] = brti[k.t.to_numpy() - T0]
    k['dir'] = np.sign(k.brti - k.strike).astype(int)   # break direction
    k['side'] = np.where(k.dir > 0, 'YES', 'NO'); k['kind'] = 'directional'
    for x in (b, k):
        x['price'] = x.price; x['won'] = x.won.astype(int)
        out.append(x[['bot', 'product', 'kind', 'period', 't', 'close', 'ticker', 'side', 'dir', 'price', 'won', 'min_left', 'secs_left']])
    # --- lock-in ---
    l = pd.read_csv('/mnt/project-files/strategy/lockin/lockin_bot_replay_aug_oct_2026.csv')
    l['t'] = (pd.to_datetime(l.time_utc) - pd.Timestamp('1970-01-01')).dt.total_seconds().astype(int)
    l['close'] = l.t + l.secs_left; l['price'] = 100 * l.price; l['won'] = l.won.astype(int)
    l['dir'] = np.where(l.side == 'YES', 1, -1); l['bot'] = 'Lock-in'; l['product'] = '15m'; l['kind'] = 'directional'
    l['period'] = np.where(l.t < 1788220800, 'Dec-Aug', 'Sep-Oct'); l['min_left'] = 1
    l['p_model'] = l.model_prob
    out.append(l[['bot', 'product', 'kind', 'period', 't', 'close', 'ticker', 'side', 'dir', 'price', 'won', 'min_left', 'secs_left', 'p_model']])
    # --- gap rules ---
    for f in ['direct', 'corr']:
        g = pd.read_parquet(f'work/gap_trades_{f}.parquet'); g['product'] = '15m'; g['kind'] = 'directional'; g['dir'] = -1
        g['secs_left'] = g.close - g.t
        out.append(g[['bot', 'product', 'kind', 'period', 't', 'close', 'ticker', 'side', 'dir', 'price', 'won', 'min_left', 'secs_left', 'p_model']])
    T = pd.concat(out, ignore_index=True)
    T['fee'] = FEE(T.price / 100); T['edge'] = 100 * T.won - T.price - T.fee
    T['day'] = pd.to_datetime(T.t, unit='s').dt.floor('D'); T['month'] = T.day.dt.strftime('%Y-%m')
    T['band10'] = (np.clip(T.price, 0, 99) // 10 * 10).astype(int)
    # flow at entry (minute grid: bars closed before the minute containing t)
    fl = Flow().at((T.t // 60 * 60).to_numpy())
    for c in ['nvz_comb_1', 'nvz_comb_3', 'nvz_comb_10', 'vi_comb_1', 'vi_comb_3', 'vi_comb_10', 'ti_comb_1', 'ti_comb_3', 'ti_comb_10', 'tr_comb_3']:
        T[c] = fl[c].to_numpy()
    fs = FlowSec()
    for X in (30, 60):
        s = fs.at(T.t.to_numpy(), X)
        for c in s.columns: T[c] = s[c].to_numpy()
    T.to_parquet('work/wm_trades.parquet')
    print(T.groupby(['bot', 'period']).agg(n=('edge', 'size'), price=('price', 'mean'), win=('won', 'mean'), edge=('edge', 'mean'),
                                           flow_cov=('nvz_comb_3', lambda s: s.notna().mean())).round(3).to_string())

if __name__ == '__main__':
    main()
