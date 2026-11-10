import os, sys, json, gzip, zipfile
from datetime import datetime, timezone
from pathlib import Path
pk = os.environ.pop('KALSHI_PRIVATE_KEY', None); sys.path.insert(0, '/mnt/project-files/data-download')
import fetch_30day_data as f
C = Path('cache/kalshi'); markets = json.loads((C / 'markets.json').read_text()); out = Path('kalshi'); out.mkdir(exist_ok=True)
by = {}
for m in markets: by.setdefault(datetime.fromtimestamp(f.iso_to_unix(m['open_time']), timezone.utc).date().isoformat(), []).append(m)
for day, ms in sorted(by.items()):
    with zipfile.ZipFile(out / f'Kalshi_BTC15M_{day}.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        z.writestr('markets.csv', f._csv(f.MARKET_COLS, [f.market_row(m) for m in ms]))
        with z.open(f'trades/{day}.csv', 'w', force_zip64=True) as dst:
            dst.write((','.join(f.TRADE_COLS) + '\n').encode())
            for m in ms:
                p = C / 'trades' / f"{m['ticker']}.csv.gz"
                if not p.exists(): print('missing', m['ticker']); continue
                with gzip.open(p, 'rb') as g:
                    g.readline(); [dst.write(l) for l in g]
    print(day, len(ms), flush=True)
