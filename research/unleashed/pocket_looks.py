"""Direct pocket (gap >= 5 pts, BRTI >= 0.10% past the strike on the side bought) by look number and consecutive run.
Run from research/ml_replay after vol3_15_scan.py (needs work/vol315_all.parquet: rsi_all.parquet + the vol buckets).
Prints trades, win % vs avg price, avg c / median c, Mar-Aug and Sep-Oct. Minutes left 4-14 unless MINUTES_ALL = True."""
import pandas as pd
MINUTES_ALL = False
A = pd.read_parquet('work/vol315_all.parquet'); A['pn'] = A.pnl * 100
def s(g):
    n = len(g); return f"{n}, {100*g.win.mean():.0f}% vs {100*g.ask_s.mean():.0f}, {g.pn.mean():+.1f}/{g.pn.median():+.1f}" if n else '-'
for m in ('Direct', 'Correction'):
    P = A[(A.model == m) & (A.gg == '5+') & A.db.isin(['.1..+.2', '>+.2'])]
    if not MINUTES_ALL: P = P[P.mg != '1-3']
    P = P.sort_values(['ticker', 'side', 'k']).copy()
    P['run'] = (P.k.diff().fillna(99) != 1) | (P.ticker != P.ticker.shift()) | (P.side != P.side.shift())
    P['rl'] = P.groupby(P.run.cumsum()).cumcount() + 1                       # consecutive minutes, same side
    P['n'] = P.sort_values(['ticker', 'k']).groupby('ticker').cumcount() + 1  # look number in the contract
    print(m)
    for lab, col in (('look', 'n'), ('consecutive', 'rl')):
        for n in (1, 2, 3):
            D = P[P[col] == n]
            print(f'  {lab} {n}:', ' | '.join(s(D[D.per == p]) for p in ('Mar-Aug', 'Sep-Oct')))
