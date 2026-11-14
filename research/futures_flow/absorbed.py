"""Absorbed flow: primary signal split by the concurrent 3-min BRTI move (did the flow move the price?).
Also 'flow vs price move disagreement': heavy flow one way, BRTI moved the other way."""
import numpy as np, pandas as pd, sys
from analyze import auc, fee
T0 = 1764460800; B = pd.Series(np.load('work/brti.npy')).ffill().to_numpy()
rows = []
for prod in sys.argv[1:] or ['15m', 'hourly']:
    S = pd.read_parquet(f'work/merged_{prod}.parquet', columns=['ticker', 'snap_ts', 'secs_left', 'ask', 'bid', 'mid', 'brti', 'res', 'move', 'up', 'period', 'month', 'nvz_comb_3', 'valid_3'])
    S = S[S.nvz_comb_3.notna()].copy()
    e = S.snap_ts.to_numpy() - T0; S['ret3'] = np.log(S.brti.to_numpy() / B[e - 180]) * 1e4  # bp over the same 3 minutes
    cut = S.loc[S.period == 'Dec-Aug', 'ret3'].abs().quantile([1/3, 2/3]).to_numpy()
    S['move_grp'] = pd.cut(S.ret3.abs(), [-1, cut[0], cut[1], 1e9], labels=['price flat', 'price mid', 'price moved'])
    S['agree'] = np.where(np.sign(S.ret3) == np.sign(S.nvz_comb_3), 'flow & price same way', 'flow against price move')
    for per in ['Dec-Aug', 'Sep-Oct']:
        for key, g in S[S.period == per].groupby(['move_grp', 'agree'], observed=True):
            h = g[g.nvz_comb_3.abs() > 1.5]
            yes = h.nvz_comb_3 > 0; price = np.where(yes, h.ask, 1 - h.bid); win = np.where(yes, h.res == 1, h.res == 0).astype(float); pnl = win - price - fee(price)
            rows.append(dict(product=prod, period=per, price_move=key[0], flow_vs_price=key[1], snapshots=len(g), auc_flow_vs_up=auc(g.nvz_comb_3.to_numpy(), g.up.to_numpy()),
                             up_pct_flow_pos=100 * g.loc[g.nvz_comb_3 > 0, 'up'].mean(), up_pct_flow_neg=100 * g.loc[g.nvz_comb_3 < 0, 'up'].mean(),
                             trades_k1_5=len(h), avg_price=100 * price.mean() if len(h) else np.nan, win_pct=100 * win.mean() if len(h) else np.nan, edge_after_fee_c=100 * pnl.mean() if len(h) else np.nan,
                             t=pnl.mean() / (pnl.std() / np.sqrt(len(h))) if len(h) > 2 else np.nan))
R = pd.DataFrame(rows); R.to_parquet(f'work/absorbed_{"_".join(sys.argv[1:]) or "all"}.parquet')
pd.set_option('display.width', 250); print(R.round(2).to_string())
