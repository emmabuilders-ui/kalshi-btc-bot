"""Download CF Benchmarks 1-s index history (whole-second ticks) via Kalshi passthrough, one gz CSV per UTC day."""
import os, sys, time, base64, gzip, random, threading, queue
import requests
from datetime import datetime, timezone, timedelta
from cryptography.hazmat.primitives.serialization import load_der_private_key
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ed25519, padding, rsa
KALSHI = "https://api.elections.kalshi.com"; API = "/trade-api/v2"
KEY_ID = os.environ["KALSHI_API_KEY_ID"]; KEY = load_der_private_key(base64.b64decode(os.environ["KALSHI_PRIVATE_KEY"]), None)
def sign(path):
    ts = str(int(time.time() * 1000)); msg = (ts + "GET" + path).encode()
    if isinstance(KEY, ed25519.Ed25519PrivateKey): sig = KEY.sign(msg)
    else: sig = KEY.sign(msg, padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH), hashes.SHA256())
    return {"KALSHI-ACCESS-KEY": KEY_ID, "KALSHI-ACCESS-TIMESTAMP": ts, "KALSHI-ACCESS-SIGNATURE": base64.b64encode(sig).decode()}
S = requests.Session(); S.headers["User-Agent"] = "research-client"
def hour(idx, ts):
    stamp = datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    for k in range(12):
        try:
            r = S.get(KALSHI + API + "/cfbenchmarks/history/values", params=dict(id=idx, timespan="HOUR", timestamp=stamp),
                      headers=sign(API + "/cfbenchmarks/history/values"), timeout=20)
        except Exception as e:
            time.sleep(3); continue
        if r.status_code == 429: time.sleep(min(60, 2 * (k + 1)) + random.uniform(0, 1)); continue
        if r.status_code >= 500: time.sleep(5); continue
        r.raise_for_status()
        return {p["time"] // 1000: float(p["value"]) for p in r.json()["data"]["payload"] if p["time"] % 1000 == 0}
    raise RuntimeError(f"gave up {idx} {stamp}")
def run(coin, idx, start, end, out, nthreads=2):
    os.makedirs(out, exist_ok=True)
    d = start
    while d < end:
        fn = f"{out}/{coin}_1s_{d:%Y-%m-%d}.csv.gz"
        if not os.path.exists(fn):
            res = {}; q = queue.Queue()
            for h in range(24): q.put(int((d + timedelta(hours=h)).timestamp()))
            def worker():
                while True:
                    try: ts = q.get_nowait()
                    except queue.Empty: return
                    res.update(hour(idx, ts))
            th = [threading.Thread(target=worker) for _ in range(nthreads)]
            [t.start() for t in th]; [t.join() for t in th]
            with gzip.open(fn + ".tmp", "wt") as f:
                f.write("unix_s,value\n")
                for t in sorted(res): f.write(f"{t},{res[t]}\n")
            os.replace(fn + ".tmp", fn)
            print(coin, d.date(), len(res), time.strftime("%H:%M:%S"), flush=True)
        d += timedelta(days=1)
if __name__ == "__main__":
    coin, idx = sys.argv[1], sys.argv[2]
    start = datetime(2025, 11, 30, tzinfo=timezone.utc); end = datetime(2026, 10, 26, tzinfo=timezone.utc)
    if len(sys.argv) > 3: start = datetime.fromisoformat(sys.argv[3]).replace(tzinfo=timezone.utc)
    run(coin, idx, start, end, f"{os.path.dirname(os.path.abspath(__file__))}/idx/{coin}", nthreads=int(os.environ.get("NT", "2")))
