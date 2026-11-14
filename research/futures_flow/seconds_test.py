"""Last-minute test: flow over the 30/60/120 s before the snapshot at 60 s left (15-min k=14; hourly min_left=1), May 14+.
Outcome: settlement (60-s avg) vs BRTI at the snapshot; Kalshi side bought with/against the flow."""
import numpy as np, pandas as pd
from scipy import stats
from flow_sec import FlowSec, S0
from analyze import auc, tau, fee
FS = FlowSec(); rows = []; trows = []
for prod in ['15m', 'hourly']:
    S = pd.read_parquet(f'work/merged_{prod}.parquet')
    S = S[(S.secs_left == 60) & (S.snap_ts >= S0 + 86400)].copy()
    for X in [30, 60, 120]:
        G = FS.at(S.snap_ts.to_numpy(), X); d = pd.concat([S.reset_index(drop=True), G], axis=1)
        d = d[(d[f'str_{X}'] >= 5) & d[f'svi_{X}'].notna()]
        for per in ['Dec-Aug', 'Sep-Oct']:
            e = d[d.period == per]
            t_, p_ = tau(e[f'snv_{X}'].to_numpy(), e.move.to_numpy())
            rows.append(dict(product=prod, window_s=X, period=per, n=len(e), tau_netvol_vs_move=t_, auc_netvol_vs_up=auc(e[f'snv_{X}'].to_numpy(), e.up.to_numpy()),
                             auc_imbalance_vs_up=auc(e[f'svi_{X}'].to_numpy(), e.up.to_numpy()), up_pct_when_flow_pos=100 * e.loc[e[f'snv_{X}'] > 0, 'up'].mean(), up_pct_when_flow_neg=100 * e.loc[e[f'snv_{X}'] < 0, 'up'].mean()))
            for thr in [0.3, 0.6]:
                for wf in [True, False]:
                    f = e[e[f'svi_{X}'].abs() > thr].copy()
                    yes = (f[f'svi_{X}'] > 0) if wf else (f[f'svi_{X}'] < 0)
                    f['price'] = np.where(yes, f.ask, 1 - f.bid); f['win'] = np.where(yes, f.res == 1, f.res == 0).astype(float); f['pnl'] = f.win - f.price - fee(f.price)
                    # favoured side only (price >= 0.5) as the lock-in bot does, and all
                    for sub, g in [('all', f), ('favourite >=50c', f[f.price >= 0.5]), ('favourite 80-99c', f[(f.price >= 0.8) & (f.price < 1.0)])]:
                        if len(g) < 30: continue
                        m = g.groupby('month').pnl.mean()
                        trows.append(dict(product=prod, window_s=X, period=per, rule='with flow' if wf else 'fade flow', imbalance_thr=thr, subset=sub, trades=len(g), avg_price=100 * g.price.mean(), win_pct=100 * g.win.mean(),
                                          win_minus_price=100 * (g.win.mean() - g.price.mean()), edge_after_fee_c=100 * g.pnl.mean(), t=g.pnl.mean() / (g.pnl.std() / np.sqrt(len(g))), months_pos=f'{(m > 0).sum()}/{len(m)}'))
R = pd.DataFrame(rows); T = pd.DataFrame(trows)
R.to_parquet('work/seconds_screen.parquet'); T.to_parquet('work/seconds_trades.parquet')
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 300)
print(R.round(3).to_string()); print(T[T.subset != 'all'].round(2).to_string())
