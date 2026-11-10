import os, time, base64, random, requests
from cryptography.hazmat.primitives.serialization import load_der_private_key, load_pem_private_key
from cryptography.hazmat.primitives.asymmetric import ed25519, rsa, padding
from cryptography.hazmat.primitives import hashes
API = 'https://api.elections.kalshi.com/trade-api/v2'
raw = os.environ['KALSHI_PRIVATE_KEY'].strip()
try: KEY = load_der_private_key(base64.b64decode(raw), None)
except Exception: KEY = load_pem_private_key(raw.replace('\\n', '\n').encode(), None)
KID = os.environ['KALSHI_API_KEY_ID'].strip()
S = requests.Session(); S.verify = '/root/.ccr/ca-bundle.crt'
import threading
_last = [0.0]; _lock = threading.Lock()
def get(path, min_interval=0.07, tries=10, **params):
    for k in range(tries):
        with _lock:
            dt = time.time() - _last[0]
            if dt < min_interval: time.sleep(min_interval - dt)
            _last[0] = time.time()
        ts = str(int(time.time() * 1000)); msg = (ts + 'GET' + '/trade-api/v2' + path).encode()
        sig = KEY.sign(msg) if isinstance(KEY, ed25519.Ed25519PrivateKey) else KEY.sign(msg, padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH), hashes.SHA256())
        h = {'KALSHI-ACCESS-KEY': KID, 'KALSHI-ACCESS-TIMESTAMP': ts, 'KALSHI-ACCESS-SIGNATURE': base64.b64encode(sig).decode()}
        try:
            r = S.get(API + path, params=params, headers=h, timeout=60)
        except requests.RequestException:
            time.sleep(2 + k); continue
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(min(30, 2 * (k + 1)) + random.random()); continue
        r.raise_for_status(); return r.json()
    raise RuntimeError(f'giving up {path} {params}')
