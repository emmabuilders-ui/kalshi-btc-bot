import time, random, requests, kidx
S = requests.Session(); S.headers["User-Agent"] = "research-client"
N429 = [0]
def get(path, retries=15, **params):
    for k in range(retries):
        try:
            r = S.get(kidx.KALSHI + kidx.API + path, params=params or None, headers=kidx.sign(kidx.API + path), timeout=30)
        except Exception:
            time.sleep(3); continue
        if r.status_code == 429:
            N429[0] += 1
            try: pause = float(r.headers.get("Retry-After", ""))
            except ValueError: pause = 1.5 * (k + 1)
            time.sleep(min(60, pause) + random.uniform(0, 0.5)); continue
        if r.status_code >= 500: time.sleep(3); continue
        r.raise_for_status(); return r.json()
    raise RuntimeError(f"gave up {path}")
