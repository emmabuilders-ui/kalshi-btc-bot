"""Inside the settlement minute: Kalshi quotes at 60/45/30/15 s left (15-min, May 14+), combined futures flow over the last 15/30 s.
Baseline = lock-in (buy favoured side, price >= 50c). Split favoured-side trades by flow agreeing / disagreeing / none. Price bands too."""
import numpy as np, pandas as pd
from flow_sec import FlowSec
from analyze import fee, auc
T0 = 1764460800; B = pd.Series(np.load('work/brti.npy')).ffill().to_numpy()
M = pd.read_parquet('work/markets_audited.parquet'); M = M[(M.n_prints == 60) & M.strike.notna() & (M.strike > 1000)]
S = pd.read_parquet('work/subminute_raw.parquet').merge(M[['ticker', 'close_ts', 'strike', 'res', 'brti_avg', 'month']], on='ticker')
S = S[(S.ask_age < 20) & (S.bid_age < 20) & S.ask.notna() & S.bid.notna() & (S.ask >= S.bid - 1e-9)].copy()
e = S.snap_ts.to_numpy() - T0; S['brti'] = B[e]
S['period'] = np.where(S.snap_ts < pd.Timestamp('2026-09-01', tz='UTC').timestamp(), 'May-Aug', 'Sep-Oct')
S['up'] = (S.brti_avg > S.brti).astype(float); S.loc[S.brti_avg == S.brti, 'up'] = np.nan
FS = FlowSec(); rows = []; trows = []
for X in [15, 30]:
    G = FS.at(S.snap_ts.to_numpy(), X); d = pd.concat([S.reset_index(drop=True), G.reset_index(drop=True)], axis=1)
    d['imb'] = d[f'svi_{X}']; d['has'] = d[f'str_{X}'] >= 3
    for sl in [60, 45, 30, 15]:
        for per in ['May-Aug', 'Sep-Oct']:
            g = d[(d.secs_left == sl) & (d.period == per)]
            h = g[g.has]
            rows.append(dict(window_s=X, secs_left=sl, period=per, snapshots=len(g), with_flow=len(h), pct_with_flow=100 * len(h) / max(len(g), 1),
                             auc_imb_vs_up=auc(h.imb.to_numpy(), h.up.to_numpy()), up_pct_flow_pos=100 * h.loc[h.imb > 0.2, 'up'].mean(), up_pct_flow_neg=100 * h.loc[h.imb < -0.2, 'up'].mean()))
            # favoured side (lock-in) baseline and flow splits
            fav_yes = g.mid if 'mid' in g else (g.ask + g.bid) / 2
            yes = ((g.ask + g.bid) / 2 >= 0.5); price = np.where(yes, g.ask, 1 - g.bid); win = np.where(yes, g.res == 1, g.res == 0).astype(float); pnl = win - price - fee(price)
            dirn = np.where(yes, 1, -1); fl = np.where(g.has, np.sign(np.where(g.imb.abs() > 0.2, g.imb, 0)), 0)
            state = np.where(fl == 0, 'no/flat flow', np.where(fl == dirn, 'flow agrees', 'flow against'))
            band = np.where(price < 0.7, '50-69c', np.where(price < 0.9, '70-89c', '90-99c'))
            T = pd.DataFrame(dict(state=state, band=band, price=price, win=win, pnl=pnl, month=g.month.to_numpy()))
            for (st, bd), q in list(T.groupby(['state', 'band'])) + [((st, 'all'), q) for st, q in T.groupby('state')] + [(('all (lock-in baseline)', 'all'), T)] + [(('all (lock-in baseline)', bd), q) for bd, q in T.groupby('band')]:
                if len(q) < 20: continue
                m = q.groupby('month').pnl.mean()
                trows.append(dict(window_s=X, secs_left=sl, period=per, flow=st, band=bd, trades=len(q), avg_price=100 * q.price.mean(), win_pct=100 * q.win.mean(), win_minus_price=100 * (q.win.mean() - q.price.mean()),
                                  edge_after_fee_c=100 * q.pnl.mean(), t=q.pnl.mean() / (q.pnl.std() / np.sqrt(len(q))) if q.pnl.std() > 0 else np.nan, months_pos=f'{(m > 0).sum()}/{len(m)}'))
R = pd.DataFrame(rows); T = pd.DataFrame(trows)
R.to_parquet('work/subminute_screen.parquet'); T.to_parquet('work/subminute_trades.parquet')
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 500)
print(R.round(3).to_string()); print(T[(T.window_s == 30)].round(2).to_string())
with pd.ExcelWriter('/mnt/project-files/strategy/futures_flow/futures_flow_test.xlsx', engine='openpyxl', mode='a', if_sheet_exists='replace') as xw:
    R.round(3).to_excel(xw, sheet_name='settle_minute_screen', index=False); T.round(2).to_excel(xw, sheet_name='settle_minute_trades', index=False)
