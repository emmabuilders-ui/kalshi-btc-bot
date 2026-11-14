"""Minute x 10c price-band grid, 15-min contracts, primary signal, every snapshot a separate look.
Rule: |flow z| > 1.5, buy the side the flow points to (YES at ask / NO at 1-bid). Edge = win% - price - fee."""
import numpy as np, pandas as pd, sys
from analyze import fee
PROD = sys.argv[2] if len(sys.argv) > 2 else '15m'; S = pd.read_parquet(f'work/merged_{PROD}.parquet', columns=['ticker', 'secs_left', 'ask', 'bid', 'res', 'period', 'month', 'nvz_comb_3'])
THR = float(sys.argv[1]) if len(sys.argv) > 1 else 1.5; S = S[S.nvz_comb_3.abs() > THR].copy(); S['ml'] = (S.secs_left // 60).astype(int)
yes = S.nvz_comb_3 > 0; S['price'] = np.where(yes, S.ask, 1 - S.bid); S['win'] = np.where(yes, S.res == 1, S.res == 0).astype(float); S['pnl'] = S.win - S.price - fee(S.price)
S['band'] = (np.floor(S.price * 10).clip(0, 9) * 10).astype(int)
S['side'] = np.where(yes, 'YES', 'NO')
def agg(g):
    m = g.groupby('month').pnl.mean()
    return pd.Series(dict(n=len(g), avg_price=100 * g.price.mean(), win_pct=100 * g.win.mean(), win_minus_price=100 * (g.win.mean() - g.price.mean()), edge_c=100 * g.pnl.mean(),
                          t=g.pnl.mean() / (g.pnl.std() / np.sqrt(len(g))) if len(g) > 2 and g.pnl.std() > 0 else np.nan, months_pos=f'{(m > 0).sum()}/{len(m)}'))
G = S.groupby(['period', 'ml', 'band']).apply(agg, include_groups=False).reset_index()
Gs = S.groupby(['period', 'side', 'ml', 'band']).apply(agg, include_groups=False).reset_index()
A = G[G.period == 'Dec-Aug'].set_index(['ml', 'band']); B = G[G.period == 'Sep-Oct'].set_index(['ml', 'band'])
J = A.join(B, lsuffix='_DecAug', rsuffix='_SepOct', how='outer').reset_index()
J.to_parquet(f'work/grid_minute_band_{PROD}_{THR}.parquet'); Gs.to_parquet(f'work/grid_minute_band_side_{PROD}_{THR}.parquet')
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 400)
piv_e = J.pivot(index='ml', columns='band', values='edge_c_DecAug').round(1); piv_e2 = J.pivot(index='ml', columns='band', values='edge_c_SepOct').round(1)
piv_n = J.pivot(index='ml', columns='band', values='n_DecAug'); piv_n2 = J.pivot(index='ml', columns='band', values='n_SepOct')
print('EDGE after fee, Dec-Aug (rows = min left, cols = price band)'); print(piv_e.to_string()); print('n'); print(piv_n.to_string())
print('EDGE after fee, Sep-Oct'); print(piv_e2.to_string()); print('n'); print(piv_n2.to_string())
both = J[(J.edge_c_DecAug > 0) & (J.edge_c_SepOct > 0) & (J.n_DecAug >= 100) & (J.n_SepOct >= 50)]
print('cells positive both periods (nA>=100, nB>=50):', len(both), 'of', len(J[(J.n_DecAug >= 100) & (J.n_SepOct >= 50)]))
print(both[['ml', 'band', 'n_DecAug', 'avg_price_DecAug', 'win_pct_DecAug', 'edge_c_DecAug', 't_DecAug', 'months_pos_DecAug', 'n_SepOct', 'avg_price_SepOct', 'win_pct_SepOct', 'edge_c_SepOct', 't_SepOct', 'months_pos_SepOct']].round(1).to_string())
with pd.ExcelWriter('/mnt/project-files/strategy/futures_flow/futures_flow_test.xlsx', engine='openpyxl', mode='a', if_sheet_exists='replace') as xw:
    J.round(2).to_excel(xw, sheet_name=f"grid_{PROD}_z{THR}", index=False); Gs.round(2).to_excel(xw, sheet_name=f"grid_{PROD}_side_z{THR}", index=False)
    piv_e.to_excel(xw, sheet_name=f"grid_{PROD}_edge_DecAug_z{THR}"); piv_e2.to_excel(xw, sheet_name=f"grid_{PROD}_edge_SepOct_z{THR}")
