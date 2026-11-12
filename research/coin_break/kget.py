import os,time,base64,threading,json,urllib.request,urllib.parse
from cryptography.hazmat.primitives import hashes,serialization
from cryptography.hazmat.primitives.asymmetric import padding,ed25519
BASE="https://api.elections.kalshi.com"
_kid=os.environ["KALSHI_API_KEY_ID"]; _raw=os.environ["KALSHI_PRIVATE_KEY"].strip()
def _load():
    if "BEGIN" in _raw: return serialization.load_pem_private_key(_raw.encode(),None)
    return serialization.load_der_private_key(base64.b64decode(_raw),None)
_key=_load()
_lock=threading.Lock(); _last=[0.0]
def get(path,min_interval=0.0,tries=12,**params):
    q=urllib.parse.urlencode({k:v for k,v in params.items() if v is not None})
    url=BASE+"/trade-api/v2"+path+("?"+q if q else "")
    for a in range(tries):
        if min_interval:
            with _lock:
                w=_last[0]+min_interval-time.time()
                if w>0: time.sleep(w)
                _last[0]=time.time()
        ts=str(int(time.time()*1000)); msg=(ts+"GET"+"/trade-api/v2"+path).encode()
        if isinstance(_key,ed25519.Ed25519PrivateKey): sig=_key.sign(msg)
        else: sig=_key.sign(msg,padding.PSS(padding.MGF1(hashes.SHA256()),padding.PSS.DIGEST_LENGTH),hashes.SHA256())
        req=urllib.request.Request(url,headers={"KALSHI-ACCESS-KEY":_kid,"KALSHI-ACCESS-SIGNATURE":base64.b64encode(sig).decode(),"KALSHI-ACCESS-TIMESTAMP":ts})
        try: return json.load(urllib.request.urlopen(req,timeout=40))
        except urllib.error.HTTPError as e:
            if e.code in (429,503,502,500): time.sleep(min(30,3+3*a)); continue
            if e.code==404: return None
            raise
        except Exception: time.sleep(3)
    raise RuntimeError("failed "+url)
