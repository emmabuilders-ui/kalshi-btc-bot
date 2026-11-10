# Rolling vs Kalshi trades: gap x price band x time left, both periods side by side.
import numpy as np, pandas as pd
exec(open('rolltrade.py').read().split("def summ")[0])
T['gap'] = 100*(T.prob - T.px - T.fee)
T['gap3'] = pd.cut(T.gap, [0, 1, 2, 100], labels=['0-1', '1-2', '2+'])
T['gap6'] = pd.cut(T.gap, [0, 0.5, 1, 1.5, 2, 3, 100], labels=['0-0.5', '0.5-1', '1-1.5', '1.5-2', '2-3', '3+'])
def agg(keys):
    g = T.groupby(keys + ['per'], observed=True)
    s = g.agg(trades=('win', 'size'), wins=('win', 'sum'), win=('win', 'mean'), px=('px', 'mean'), pnl=('pnl', 'mean'))
    s['months_pos'] = T.groupby(keys + ['per', 'month'], observed=True).pnl.mean().gt(0).groupby(level=list(range(len(keys) + 1))).agg(lambda x: f'{x.sum()}/{len(x)}')
    s['cell'] = (100 * s.win).round(1).astype(str) + ' vs ' + (100 * s.px).round(1).astype(str)
    s['win_minus_px'] = (100 * (s.win - s.px)).round(2); s['after_fee_c'] = (100 * s.pnl).round(2)
    W = s[['trades', 'cell', 'win_minus_px', 'after_fee_c', 'months_pos']].unstack('per')
    W.columns = [f'{b} {a}' for a, b in W.columns]
    order = [f'{p} {c}' for p in ['Mar-Aug', 'Sep-Oct'] for c in ['trades', 'cell', 'win_minus_px', 'after_fee_c', 'months_pos']]
    return W[[c for c in order if c in W.columns]].reset_index()
pd.set_option('display.width', 260); pd.set_option('display.max_rows', 500)
G3 = agg(['side', 'gap3', 'band', 'time']); G6 = agg(['side', 'gap6', 'band', 'time']); A3 = agg(['gap3', 'band', 'time'])
ok = lambda G: G[(G['Mar-Aug after_fee_c'] > 0) & (G['Sep-Oct after_fee_c'] > 0) & (G['Mar-Aug trades'] >= 100) & (G['Sep-Oct trades'] >= 50)]
elig = lambda G: G[(G['Mar-Aug trades'] >= 100) & (G['Sep-Oct trades'] >= 50)]
for nm, G in [('side x gap3', G3), ('both sides x gap3', A3)]:
    print(nm, 'eligible cells', len(elig(G)), 'positive both', len(ok(G)))
    print(ok(G).to_string(index=False)); print()
# gap effect within price x time: compare after-fee of gap 2+ vs 0-1 in the same cell
with pd.ExcelWriter('work/rolling_trades_gap_price_time.xlsx') as w:
    A3.to_excel(w, sheet_name='Gap3_price_time', index=False); G3.to_excel(w, sheet_name='Side_gap3_price_time', index=False)
    G6.to_excel(w, sheet_name='Side_gap6_price_time', index=False); ok(G3).to_excel(w, sheet_name='Positive_both_periods', index=False)
