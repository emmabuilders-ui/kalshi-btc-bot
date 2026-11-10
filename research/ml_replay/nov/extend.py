# Extend the replay inputs with Oct 26 - Nov 9 2026 (new ZIPs + 1-s BRTI) on top of the old work/ files.
import glob, gzip, json, sys, numpy as np, pandas as pd
sys.path.insert(0, '.')
from build_snaps import from_zip
from build_flow import work as flow_work
ML = sys.argv[1]; NEW = sys.argv[2]
T0 = 1764460800; end = int(pd.Timestamp('2026-11-10', tz='UTC').timestamp())
old = np.load(f'{ML}/work/brti.npy'); arr = np.full(end - T0, np.nan); arr[:len(old)] = old
for f in glob.glob(f'{NEW}/brti/*.json.gz'):
    for k, v in json.load(gzip.open(f)).items():
        i = int(k) - T0
        if 0 <= i < len(arr) and (i >= len(old) or np.isnan(arr[i])): arr[i] = v
np.save('work/brti.npy', arr); print('brti missing after Oct 26:', int(np.isnan(arr[len(old):]).sum()))
zips = sorted(glob.glob(f'{NEW}/kalshi/**/*.zip', recursive=True)); print('zips', zips[:2], len(zips))
M, S, A, F = [], [], [], []
for z in zips:
    m, s, a = from_zip(z); M.append(m); S.append(s); A.append(a); F.append(flow_work(z))
M = pd.concat(M); S = pd.concat(S); M['src'] = 'trades'; S['src'] = 'trades'
for c in ['strike', 'expiration_value']: M[c] = pd.to_numeric(M[c], errors='coerce')
M['result'] = M['result'].astype(str)
Mo = pd.read_parquet(f'{ML}/work/markets.parquet'); So = pd.read_parquet(f'{ML}/work/snaps_raw.parquet')
pd.concat([Mo, M]).drop_duplicates('ticker').to_parquet('work/markets.parquet')
pd.concat([So, S]).drop_duplicates(['ticker', 'k']).to_parquet('work/snaps_raw.parquet')
pd.concat([pd.read_csv(f'{ML}/work/zip_audit.csv'), pd.DataFrame(A)]).to_csv('work/zip_audit.csv', index=False)
pd.concat([pd.read_parquet(f'{ML}/work/flow.parquet'), pd.concat(F)]).drop_duplicates(['ticker', 'k']).to_parquet('work/flow.parquet')
print('new markets', len(M), 'new snaps', len(S))
