"""Test 1: futures flow as a filter on each bot's trades. Reads work/wm_trades.parquet; writes work/wm_t1_*.parquet."""
import numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
rng = np.random.default_rng(7)
T = pd.read_parquet('work/wm_trades.parquet')
PER = ['Dec-Aug', 'Sep-Oct']
# thresholds: nvz +-0.5 (setup); vi/ti: median |value| over Dec-Aug 15-min snapshots (half of valid minutes count as "strong")
m15 = pd.read_parquet('work/merged_15m.parquet', columns=['period', 'vi_comb_1', 'vi_comb_3', 'vi_comb_10', 'ti_comb_1', 'ti_comb_3', 'ti_comb_10'])
m15 = m15[m15.period == 'Dec-Aug']
THR = {f'nvz_comb_{w}': 0.5 for w in (1, 3, 10)}
for c in ['vi_comb_1', 'vi_comb_3', 'vi_comb_10', 'ti_comb_1', 'ti_comb_3', 'ti_comb_10']: THR[c] = float(m15[c].abs().median())
THR['svi_30'] = THR['vi_comb_1']; THR['svi_60'] = THR['vi_comb_1']
def lab_of(B, v): return label(B[MAG[v]], B.dir, THR[v], True) if v in MAG else label(B[v], B.dir, THR[v])
print('thresholds', {k: round(v, 3) for k, v in THR.items()})

def label(sig, dr, thr, mag=False):
    s = np.asarray(sig, float)
    if mag:  # band bots: quiet flow = agrees with "BTC stays inside the band", strong flow either way = against
        lab = np.where(np.isnan(s), 0, np.where(np.abs(s) <= thr, 1, -1))
    else:
        lab = np.where(np.isnan(s) | (np.abs(s) <= thr), 0, np.sign(s) * np.asarray(dr))
    return lab.astype(int)
MAG = {f'quiet_{v}': v for v in ['nvz_comb_1', 'nvz_comb_3', 'nvz_comb_10']}
for k, v in MAG.items(): THR[k] = THR[v]
VARS = list(THR)

def stats(g):
    if len(g) == 0: return pd.Series({'n': 0})
    byday = g.groupby('day').edge.mean(); bym = g.groupby('month').edge.mean()
    return pd.Series({'n': len(g), 'avg_price': g.price.mean(), 'win_pct': 100 * g.won.mean(), 'win_minus_price': 100 * g.won.mean() - g.price.mean(),
                      'edge_avg': g.edge.mean(), 'edge_day_median': byday.median(), 'days': len(byday),
                      'months_pos': f'{int((bym > 0).sum())}/{len(bym)}', 't_stat': g.edge.mean() / (g.edge.std(ddof=1) / np.sqrt(len(g))) if len(g) > 1 else np.nan})

SPL = {1: 'agrees', -1: 'against', 0: 'neutral'}
rows, grids, settle, shuf = [], [], [], []
for bot, B in T.groupby('bot'):
    B = B.copy()
    for v in VARS:
        if v.startswith('svi') and B['product'].iloc[0] != '15m': continue
        if v in MAG and B.kind.iloc[0] != 'band': continue
        B['lab'] = lab_of(B, v)
        for per in PER:
            P = B[B.period == per]
            if len(P) == 0: continue
            st_all = stats(P); st_all['split'] = 'all'
            out = [st_all]
            for lv, nm in SPL.items():
                s = stats(P[P.lab == lv]); s['split'] = nm; out.append(s)
            df = pd.DataFrame(out); df['bot'] = bot; df['variant'] = v; df['period'] = per; rows.append(df)
        # grids for agrees / against (primary + all variants; filtered later for the workbook)
        for lv in (1, -1):
            S = B[B.lab == lv]
            if len(S) == 0: continue
            g = S.groupby(['period', 'side', 'band10', 'min_left']).apply(stats, include_groups=False).reset_index()
            g['bot'] = bot; g['variant'] = v; g['split'] = SPL[lv]; grids.append(g)
            if B['product'].iloc[0] == '15m' and bot in ('Break', 'Lock-in'):
                S = S.copy(); S['secs_bucket'] = pd.cut(S.secs_left, [0, 15, 30, 45, 60, 90, 150], labels=['<=15s', '15-30s', '30-45s', '45-60s', '60-90s', '90-150s'])
                g = S.groupby(['period', 'side', 'band10', 'secs_bucket'], observed=True).apply(stats, include_groups=False).reset_index()
                g['bot'] = bot; g['variant'] = v; g['split'] = SPL[lv]; settle.append(g)
    # permutation test: statistic per variant = min over periods of edge(agrees) - edge(against); joint shuffle of the label matrix rows within period
    vs = [v for v in VARS if not (v.startswith('svi') and B['product'].iloc[0] != '15m') and not (v in MAG and B.kind.iloc[0] != 'band')]
    obs = {}; perm = np.zeros((10000, len(vs))); permf = perm.copy()
    ok = True
    for per in PER:
        P = B[B.period == per]
        if len(P) < 30: ok = False; break
        L = np.stack([lab_of(P, v) for v in vs], 1); e = P.edge.to_numpy()
        A = (L == 1).astype(float); G = (L == -1).astype(float); na = A.sum(0); ng = G.sum(0)
        with np.errstate(invalid='ignore', divide='ignore'):
            o = e @ A / na - e @ G / ng
            idx = np.argsort(rng.random((10000, len(e))), 1); E = e[idx]
            pm = (E @ A) / na - (E @ G) / ng
        obs[per] = o; perm = np.minimum(perm, pm) if per != PER[0] else pm; permf = np.minimum(permf, -pm) if per != PER[0] else -pm
    if ok:
        o = np.minimum(obs[PER[0]], obs[PER[1]]); of = np.minimum(-obs[PER[0]], -obs[PER[1]])
        for j, v in enumerate(vs):
            shuf.append({'bot': bot, 'variant': v, 'stat_min_period_agree_minus_against': o[j], 'p_single': float(np.nanmean(perm[:, j] >= o[j])),
                         'p_bot_family': float(np.nanmean(np.nanmax(perm, 1) >= o[j])),
                         'fade_stat_min_period_against_minus_agree': of[j], 'fade_p_single': float(np.nanmean(permf[:, j] >= of[j])),
                         'fade_p_bot_family': float(np.nanmean(np.nanmax(permf, 1) >= of[j]))})
        np.save(f'work/wm_perm_{bot.replace(" ", "_")}.npy', np.nanmax(perm, 1)); np.save(f'work/wm_permf_{bot.replace(" ", "_")}.npy', np.nanmax(permf, 1))
R = pd.concat(rows, ignore_index=True); R.to_parquet('work/wm_t1_splits.parquet')
pd.concat(grids, ignore_index=True).to_parquet('work/wm_t1_grids.parquet')
pd.concat(settle, ignore_index=True).to_parquet('work/wm_t1_settle.parquet')
S = pd.DataFrame(shuf)
# family-wise over all bots x variants: max over everything in each shuffle
mx = None
for bot in T.bot.unique():
    try: a = np.load(f'work/wm_perm_{bot.replace(" ", "_")}.npy')
    except FileNotFoundError: continue
    mx = a if mx is None else np.fmax(mx, a)
S['p_family_all'] = [float(np.nanmean(mx >= o)) for o in S.stat_min_period_agree_minus_against]
mxf = None
for bot in T.bot.unique():
    try: a = np.load(f'work/wm_permf_{bot.replace(" ", "_")}.npy')
    except FileNotFoundError: continue
    mxf = a if mxf is None else np.fmax(mxf, a)
S['fade_p_family_all'] = [float(np.nanmean(mxf >= o)) for o in S.fade_stat_min_period_against_minus_agree]
S.to_parquet('work/wm_t1_shuffle.parquet')
# pass bar
pb = []
for (bot, v), g in R.groupby(['bot', 'variant']):
    rec = {'bot': bot, 'variant': v}; passes = True
    for per in PER:
        p = g[g.period == per].set_index('split')
        if 'agrees' not in p.index or p.loc['agrees', 'n'] == 0 or 'against' not in p.index or p.loc['against', 'n'] == 0: passes = False; rec[f'{per}_agree_edge'] = np.nan; continue
        ea, eg, el = p.loc['agrees', 'edge_avg'], p.loc['against', 'edge_avg'], p.loc['all', 'edge_avg']
        rec[f'{per}_n_agree'] = p.loc['agrees', 'n']; rec[f'{per}_agree_edge'] = ea; rec[f'{per}_against_edge'] = eg; rec[f'{per}_all_edge'] = el
        rec[f'{per}_months_pos'] = p.loc['agrees', 'months_pos']
        mp = p.loc['agrees', 'months_pos']; a, b = map(int, mp.split('/'))
        need_n = 50 if bot in ('Break', 'Lock-in') else 100
        cond = (ea >= eg + 1.5) and (ea >= el + 1.0) and (per == 'Dec-Aug' or p.loc['agrees', 'n'] >= need_n) and (a >= (7 if per == 'Dec-Aug' else 2) or (per == 'Dec-Aug' and b < 9 and a == b))
        passes &= bool(cond)
    rec['passes'] = passes
    fade = True
    for per in PER:
        p = g[g.period == per].set_index('split')
        if 'against' not in p.index or p.loc['against', 'n'] == 0 or 'agrees' not in p.index or p.loc['agrees', 'n'] == 0: fade = False; continue
        ea, eg, el = p.loc['agrees', 'edge_avg'], p.loc['against', 'edge_avg'], p.loc['all', 'edge_avg']
        a, b = map(int, p.loc['against', 'months_pos'].split('/'))
        need_n = 50 if bot in ('Break', 'Lock-in') else 100
        fade &= bool((eg >= ea + 1.5) and (eg >= el + 1.0) and (per == 'Dec-Aug' or p.loc['against', 'n'] >= need_n) and (a >= (7 if per == 'Dec-Aug' else 2) or (per == 'Dec-Aug' and b < 9 and a == b)))
        rec[f'{per}_n_against'] = p.loc['against', 'n']; rec[f'{per}_against_months_pos'] = p.loc['against', 'months_pos']
    rec['fade_passes'] = fade; pb.append(rec)
PB = pd.DataFrame(pb).merge(S.drop(columns=['stat_min_period_agree_minus_against', 'fade_stat_min_period_against_minus_agree']), on=['bot', 'variant'], how='left')
PB.to_parquet('work/wm_t1_passbar.parquet')
print(PB[PB.passes | PB.fade_passes].to_string()); print('passing looks:', int(PB.passes.sum()), 'fade passing:', int(PB.fade_passes.sum()), 'of', len(PB))
print(S.sort_values('p_family_all').head(8).round(3).to_string()); print(S.sort_values('fade_p_family_all').head(8).round(3).to_string())
print(R[(R.variant == 'nvz_comb_3')].pivot_table(index=['bot', 'period'], columns='split', values=['n', 'edge_avg'], aggfunc='first').round(2).to_string())
