# Oct 26 - Nov 9 2026: KXBTC15M markets + every trade (daily ZIPs like kalshi-daily/) and 1-s BRTI, into this folder.
import os, sys, json, gzip, time
from datetime import date
import pandas as pd
pk = os.environ.pop('KALSHI_PRIVATE_KEY'); sys.path.insert(0, '/mnt/project-files/data-download')
import fetch_30day_data as f
os.environ['KALSHI_PRIVATE_KEY'] = pk
import kget
from pathlib import Path
f.kalshi = lambda path, **p: kget.get(path, **p)
part = sys.argv[1]
if part == 'brti':
    out = Path('brti'); out.mkdir(exist_ok=True)
    for h in pd.date_range('2026-10-25T20:00Z', '2026-11-10T00:00Z', freq='h', inclusive='left'):
        fn = out / f'{int(h.timestamp())}.json.gz'
        if fn.exists(): continue
        j = kget.get('/cfbenchmarks/history/values', min_interval=2.0, tries=40, id='BRTI', timespan='HOUR', timestamp=h.strftime('%Y-%m-%dT%H:%M:%SZ'))
        sec = {p['time'] // 1000: float(p['value']) for p in j['data']['payload'] if p['time'] % 1000 == 0}
        gzip.open(fn, 'wt').write(json.dumps(sec)); print(h, len(sec), flush=True)
else:
    f.fetch_kalshi(date(2026, 10, 26), date(2026, 11, 9), Path('kalshi'), Path('cache'), workers=4)
