"""Live chart of the paper bots, served from the server itself (read-only, no orders).

Open http://<server-ip>:8080/<token>/ in any browser. The page refreshes every 2 seconds:
the live BRTI price (Kalshi's cfbenchmarks feed, 1-second values), the hourly range bot's open
bands, the 15-minute strike, and the buys of the two 15-minute bots, with settled P/L so far.

The token (a random path, made by setup.sh) keeps the page private to whoever has the link.
"""

from __future__ import annotations

import csv
import gzip
import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .late_breakdown import KalshiClient

log = logging.getLogger("kalshi_btc_bot")

KEEP = 3900          # seconds of price history to serve (a bit over one hour)


def _rows(path: Path) -> list[dict]:
    try:
        with path.open() as f:
            return list(csv.DictReader(f))
    except (OSError, csv.Error):
        return []


def _json(path: Path, default):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def range_positions(range_dir: Path) -> list[dict]:
    """Open range positions with their band edges (from the signal that opened them)."""
    held = _json(range_dir / "open_positions.json", [])
    edges = {r["ticker"]: (_num(r.get("floor")), _num(r.get("cap"))) for r in _rows(range_dir / "signals.csv")}
    out = []
    for p in held:
        lo, hi = edges.get(p["ticker"], (None, None))
        out.append({**p, "lo": lo, "hi": hi})
    return out


def settled_summary(rows: list[dict]) -> dict:
    pnl = [_num(r.get("pnl")) for r in rows]
    pnl = [x for x in pnl if x is not None]
    won = sum(1 for r in rows if str(r.get("won")) == "1")
    return {"trades": len(pnl), "wins": won, "pnl": round(sum(pnl), 2)}


def pnl_points(rows: list[dict], bot: str) -> list[dict]:
    """Settled trades as (time, bot, pnl) points for the cumulative P/L chart."""
    out = []
    for r in rows:
        x = _num(r.get("pnl"))
        if x is not None and r.get("time_utc"):
            out.append({"bot": bot, "time_utc": r["time_utc"], "pnl": x, "won": str(r.get("won")) == "1"})
    return out


def session_buys(rows: list[dict], ticker: str, bot: str) -> list[dict]:
    out = []
    for r in rows:
        if r.get("ticker") != ticker or not str(r.get("decision", "")).startswith("BUY"):
            continue
        t = r.get("break_utc") or r.get("time_utc")
        out.append({"bot": bot, "time_utc": t, "side": r.get("side"), "level": r.get("level"),
                    "level_price": _num(r.get("level_price")), "ask": _num(r.get("ask"))})
    return out


def _ts(s: str) -> int:
    return int(datetime.strptime(s[:19], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp())


def read_price_hour(path: Path, res: int) -> list[list[float]]:
    """One hour of the breakdown bot's 1-second price log, cut to one point per `res` seconds."""
    out: dict[int, float] = {}
    try:
        with gzip.open(path, "rt") as f:
            for line in f:
                a = line.split(",")
                if not a[0].isdigit() or not a[1]:
                    continue
                out[int(a[0]) // res * res] = float(a[1])
    except (OSError, EOFError, ValueError):
        pass  # the current hour is still being written; the live feed covers it
    return [[t, out[t]] for t in sorted(out)]


def history_events(L: dict[str, Path]) -> dict:
    """Everything the bots did, for the full-history chart."""
    edges = {r["ticker"]: (_num(r.get("floor")), _num(r.get("cap"))) for r in _rows(L["range"] / "signals.csv")}
    rng = []
    for r in _rows(L["range"] / "trades.csv"):
        lo, hi = edges.get(r["ticker"], (None, None))
        t = _ts(r["time_utc"])
        rng.append({"t": t, "close": (t // 3600 + 1) * 3600, "side": r["side"], "lo": lo, "hi": hi,
                    "n": r.get("contracts"), "price": _num(r.get("avg_price")), "model": _num(r.get("model")),
                    "status": "won" if str(r.get("won")) == "1" else "lost", "pnl": _num(r.get("pnl"))})
    for p in range_positions(L["range"]):
        t = _ts(p["time_utc"])
        rng.append({"t": t, "close": p["close_ts"], "side": p["side"], "lo": p["lo"], "hi": p["hi"],
                    "n": p["contracts"], "price": p["avg_price"], "model": p.get("model"), "status": "open", "pnl": None})
    sessions = []
    for r in _rows(L["breakdown"] / "sessions.csv"):
        k = _num(r.get("strike"))
        if k and r.get("open_utc"):
            sessions.append([_ts(r["open_utc"]), k])
    buys = []
    for bot, key in (("Breakdown", "breakdown"), ("Fast push", "fastpush")):
        res = {(r.get("ticker"), r.get("side")): r for r in _rows(L[key] / "trades.csv")}
        for r in _rows(L[key] / "signals.csv"):
            if not str(r.get("decision", "")).startswith("BUY"):
                continue
            tr = res.get((r.get("ticker"), r.get("side")))
            st = "open" if tr is None else ("won" if str(tr.get("won")) == "1" else "lost")
            buys.append({"bot": bot, "t": _ts(r.get("break_utc") or r["time_utc"]), "side": r.get("side"),
                         "level": r.get("level"), "ask": _num(r.get("ask")), "status": st,
                         "pnl": _num(tr.get("pnl")) if tr else None})
    return {"range": rng, "sessions": sessions, "buys": buys}


class Dashboard:
    def __init__(self, logs: dict[str, Path], key_id=None, key_path=None):
        self.k = KalshiClient(key_id, key_path)
        self.logs = logs
        self.prices: dict[int, float] = {}
        self.session: dict | None = None
        self.lock = threading.Lock()
        self.err = ""
        self._hours: dict[str, list] = {}

    def history(self, res: int = 5) -> dict:
        """Price history from the breakdown bot's hourly logs (closed hours are cached)."""
        cur = datetime.now(timezone.utc).strftime("%Y-%m-%d/%H")
        pts = []
        for f in sorted((self.logs["breakdown"] / "prices").glob("*/*.csv.gz")):
            key = f"{f.parent.name}/{f.name[:2]}"
            if key not in self._hours or key >= cur:
                self._hours[key] = read_price_hour(f, res)
            pts += self._hours[key]
        with self.lock:
            live = sorted(self.prices)
            pts += [[t, self.prices[t]] for t in live[::res] if not pts or t > pts[-1][0]]
        return {"prices": pts, **history_events(self.logs)}

    def poll(self):
        now = time.time()
        try:
            got = self.k.brti_live()
            with self.lock:
                self.prices.update(got)
                for t in [t for t in self.prices if t < now - KEEP]:
                    del self.prices[t]
            self.err = ""
        except Exception as e:  # keep serving what we have
            self.err = f"price feed: {e}"
        s = self.session
        if s is None or now >= s["close_ts"]:
            try:
                self.session = self.k.open_market(now)
            except Exception as e:
                self.err = f"market list: {e}"

    def loop(self, every: float = 2.0):
        while True:
            t0 = time.time()
            self.poll()
            time.sleep(max(0.2, every - (time.time() - t0)))

    def data(self) -> dict:
        with self.lock:
            ts = sorted(self.prices)
            pts = [[t, self.prices[t]] for t in ts]
        L = self.logs
        s = self.session or {}
        bd = _rows(L["breakdown"] / "signals.csv")
        fp = _rows(L["fastpush"] / "signals.csv")
        hour_close = (int(time.time()) // 3600 + 1) * 3600
        return {
            "now": time.time(), "prices": pts, "error": self.err, "hour_close": hour_close,
            "range": {"open": range_positions(L["range"]),
                      "settled": settled_summary(_rows(L["range"] / "trades.csv"))},
            "session": s,
            "buys": session_buys(bd, s.get("ticker"), "Breakdown") + session_buys(fp, s.get("ticker"), "Fast push"),
            "pnl": sorted(pnl_points(_rows(L["range"] / "trades.csv"), "Range")
                          + pnl_points(_rows(L["breakdown"] / "trades.csv"), "Breakdown")
                          + pnl_points(_rows(L["fastpush"] / "trades.csv"), "Fast push"),
                          key=lambda r: r["time_utc"]),
            "breakdown": settled_summary(_rows(L["breakdown"] / "trades.csv")),
            "fastpush": settled_summary(_rows(L["fastpush"] / "trades.csv")),
        }


def make_handler(dash: Dashboard, token: str):
    page = PAGE.encode()
    root = f"/{token}/"

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body: bytes, ctype: str):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == root:
                self._send(200, page, "text/html; charset=utf-8")
            elif self.path == root + "history.json":
                self._send(200, json.dumps(dash.history()).encode(), "application/json")
            elif self.path == root + "data.json":
                self._send(200, json.dumps(dash.data()).encode(), "application/json")
            elif self.path == root[:-1]:
                self.send_response(301)
                self.send_header("Location", root)
                self.end_headers()
            else:
                self._send(404, b"not found", "text/plain")

    return H


def serve(port: int, token: str, logs: dict[str, Path]):
    dash = Dashboard(logs, os.environ.get("KALSHI_KEY_ID"), os.environ.get("KALSHI_KEY_FILE"))
    dash.poll()
    threading.Thread(target=dash.loop, daemon=True).start()
    srv = ThreadingHTTPServer(("0.0.0.0", port), make_handler(dash, token))
    log.info("live chart on port %d", port)
    srv.serve_forever()


PAGE = r"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Bot Live Chart</title>
<style>
:root{--bg:#f6f7f9;--card:#fff;--fg:#1b1f24;--mut:#66707a;--line:#1b1f24;--grid:#e6e8ec;--grid2:#c3c8cf;
--no:#c0392b;--yes:#1f8a4c;--strike:#2f6fb2;--target:#b8875a;--fp:#8e44ad;--win:#1f8a4c;--lose:#c0392b;--open:#d4a106;--win-bg:#e6f4ec;--lose-bg:#fbe9e7}
@media (prefers-color-scheme:dark){:root{--bg:#121417;--card:#1b1e22;--fg:#e8eaed;--mut:#9aa3ad;
--line:#e8eaed;--grid:#262a30;--grid2:#3d434b;--no:#ef6b5b;--yes:#4cc27e;--strike:#6ea8e8;--target:#d2a87a;--fp:#c08be0;--win:#4cc27e;--lose:#ef6b5b;--open:#f2c94c;
--win-bg:#17301f;--lose-bg:#3a1c19;color-scheme:dark}}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.4 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
.wrap{max-width:1300px;margin:0 auto;padding:16px}
h1{font-size:18px;margin:0}
.top{display:flex;flex-wrap:wrap;align-items:baseline;gap:8px 20px;margin-bottom:10px}
.px{font-size:26px;font-weight:700;font-variant-numeric:tabular-nums}
.mut{color:var(--mut)}
.bar{display:flex;flex-wrap:wrap;gap:6px;align-items:center;margin:6px 0}
.bar button{border:1px solid var(--grid2);background:var(--card);color:var(--fg);padding:5px 11px;border-radius:6px;cursor:pointer;font:inherit}
.bar button:focus-visible{outline:2px solid var(--strike)}
.card{background:var(--card);border:1px solid var(--grid);border-radius:8px;padding:12px}
#sc{overflow-x:auto;overflow-y:hidden}
#stick{position:sticky;left:0}
canvas{display:block}
.lbl{font-weight:600;margin:4px 0 2px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:10px;margin-top:12px}
table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}
td,th{padding:6px 8px;text-align:left;border-bottom:1px solid var(--grid);white-space:nowrap}
th{font-size:12px;color:var(--mut);font-weight:600}
.pill{padding:2px 8px;border-radius:10px;font-weight:600;font-size:12px}
.w{background:var(--win-bg);color:var(--win)}.l{background:var(--lose-bg);color:var(--lose)}
.scroll{overflow-x:auto}
.big{font-size:20px;font-weight:700;font-variant-numeric:tabular-nums}
.key{display:flex;flex-wrap:wrap;gap:4px 14px;font-size:12px;color:var(--mut)}
.key i{display:inline-block;width:12px;height:10px;margin-right:4px;vertical-align:middle;border-radius:2px}
</style></head><body><div class="wrap">
<div class="top"><h1>Paper bots, live</h1><span class="px" id="px">…</span>
<span class="mut" id="clock"></span><span class="mut" id="err"></span></div>
<div class="bar"><button id="zin">Zoom in</button><button id="zout">Zoom out</button><button id="back">Back 1 hour</button><button id="live">Jump to now</button>
<span class="mut" id="zinfo"></span></div>
<div class="card">
<div class="key"><span><i style="background:var(--no);opacity:.35"></i>range NO band</span><span><i style="background:var(--yes);opacity:.35"></i>range YES band</span>
<span>band outline: <b style="color:var(--win)">won</b> / <b style="color:var(--lose)">lost</b> / <b style="color:var(--open)">open</b></span><span>dots and triangles: <b style="color:var(--win)">won</b> / <b style="color:var(--lose)">lost</b> / <b style="color:var(--open)">open</b></span>
<span><i style="background:var(--target)"></i>15-min target (strike) for each contract: above at the close = YES wins</span><span>▲▼ 15-min bot buys</span><span>thin vertical lines = 15-min contract closes</span></div>
<div id="sc"><div id="inner"><div id="stick">
<div class="lbl">Bitcoin (BRTI) and every trade</div><canvas id="c"></canvas>
<div class="lbl">Profit and loss, after fees <span class="mut" id="pl-legend" style="font-weight:400"></span></div><canvas id="pl"></canvas>
</div></div></div>
<div class="mut" style="font-size:12px;margin-top:4px">Scroll sideways (or drag the scrollbar) to go back through the full history. Both graphs move together.</div>
</div>
<div class="card scroll" style="margin-top:12px"><b>Open range positions</b><table id="pos"></table></div>
<div class="card scroll" style="margin-top:12px"><b>15-minute bots, this session</b><table id="pos15"></table></div>
<div class="grid">
<div class="card"><div class="mut">Range bot, settled</div><div class="big" id="sr"></div></div>
<div class="card"><div class="mut">Breakdown bot, settled</div><div class="big" id="sb"></div></div>
<div class="card"><div class="mut">Fast-push bot, settled</div><div class="big" id="sf"></div></div>
</div>
<p class="mut">BRTI is the index Kalshi settles on; settlement is its average over the last 60 seconds. Paper trades only. Updates every 2 seconds.</p>
</div>
<script>
let D=null,H=null,P=[];          // live data, history, merged prices [[t,p]]
const ZOOMS=[16,32,64,128,256,512];let zi=3;  // pixels per 15 minutes
let follow=true;
const $=id=>document.getElementById(id);
const css=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const hm=t=>new Date(t*1000).toISOString().slice(11,16);
const hms=t=>new Date(t*1000).toISOString().slice(11,19);
const money=x=>(x<0?'-$':'+$')+Math.abs(x).toFixed(2);
const usd=x=>'$'+x.toLocaleString('en-US',{maximumFractionDigits:0});
const tsOf=s=>Date.parse(s.replace(' ','T')+'Z')/1000;
const sc=$('sc');
function pps(){return ZOOMS[zi]/900}
function range(){const first=Math.min(P.length?P[0][0]:D.now,...(H?H.range.map(r=>r.close-3600):[]));
 return [Math.floor(first/900)*900,Math.ceil(D.now/900)*900+300]}
function sum(id,s){$(id).textContent=s.trades?`${s.wins}/${s.trades} won, ${money(s.pnl)}`:'no trades yet'}
function merge(pts){if(!pts.length)return;const last=P.length?P[P.length-1][0]:0;for(const p of pts)if(p[0]>last)P.push(p)}
async function loadHistory(){try{const r=await fetch('history.json',{cache:'no-store'});H=await r.json();
 const live=P.filter(p=>p[0]>(H.prices.length?H.prices[H.prices.length-1][0]:0));P=H.prices.concat(live);layout()}catch(e){}}
async function tick(){try{const r=await fetch('data.json',{cache:'no-store'});D=await r.json();merge(D.prices);
 if(!H)await loadHistory();layout()}catch(e){$('err').textContent='connection lost, retrying'}}
function layout(){
 if(!D)return;const [T0,T1]=range();const w=sc.clientWidth;
 const total=Math.max(w,(T1-T0)*pps()+90);$('inner').style.width=total+'px';$('stick').style.width=w+'px';
 if(follow)sc.scrollLeft=total;draw()}
sc.addEventListener('scroll',()=>{follow=sc.scrollLeft+sc.clientWidth>=sc.scrollWidth-30;draw()});
function zoom(d){if(!D)return;const [T0]=range();const mid=T0+(sc.scrollLeft+sc.clientWidth/2)/pps();
 zi=Math.max(0,Math.min(ZOOMS.length-1,zi+d));layout();if(!follow)sc.scrollLeft=(mid-T0)*pps()-sc.clientWidth/2;draw()}
$('zin').onclick=()=>zoom(1);$('zout').onclick=()=>zoom(-1);
$('back').onclick=()=>{follow=false;sc.scrollLeft-=3600*pps()};
$('live').onclick=()=>{follow=true;layout()};
addEventListener('resize',layout);
function setup(cv,h){const dpr=devicePixelRatio||1,w=$('stick').clientWidth;cv.style.width=w+'px';cv.style.height=h+'px';
 cv.width=w*dpr;cv.height=h*dpr;const g=cv.getContext('2d');g.setTransform(dpr,0,0,dpr,0,0);g.clearRect(0,0,w,h);return [g,w]}
function timegrid(g,X,tL,tR,T,B,h,labels){
 for(let t=Math.floor(tL/900)*900;t<=tR;t+=900){const hr=t%3600===0,day=t%86400===0;
  g.strokeStyle=css(hr?'--grid2':'--grid');g.lineWidth=1;g.beginPath();g.moveTo(X(t),T);g.lineTo(X(t),h-B);g.stroke();
  if(labels){const every=ZOOMS[zi]>=64?900:ZOOMS[zi]>=32?3600:10800;
   if(t%every===0||day){g.fillStyle=css(day?'--fg':'--mut');g.font=(day||hr?'600 ':'')+'11px system-ui';
    const d=new Date(t*1000);g.fillText(day?d.toUTCString().slice(5,11):hm(t),X(t)-14,h-6)}}}}
function draw(){
 if(!D)return;
 const last=P.length?P[P.length-1]:null;if(last)$('px').textContent=usd(last[1]);
 $('err').textContent=D.error||'';
 $('clock').textContent=`${last?hms(last[0]):''} UTC · 15-min closes ${hm(D.session&&D.session.close_ts||0)} · hour closes ${hm(D.hour_close)}`;
 $('zinfo').textContent=`${Math.round(sc.clientWidth/pps()/3600*10)/10} hours on screen`;
 sum('sr',D.range.settled);sum('sb',D.breakdown);sum('sf',D.fastpush);
 const [T0]=range(),tL=T0+sc.scrollLeft/pps(),tR=tL+sc.clientWidth/pps();
 // ---------------- price chart
 const [g,w]=setup($('c'),430),h=430,L=66,R=8,T=8,B=22;
 const X=t=>L+(t-tL)/(tR-tL)*(w-L-R);
 const vis=P.filter(p=>p[0]>=tL-60&&p[0]<=tR+60);
 const ses=(H?H.sessions:[]).concat(D.session&&D.session.strike?[[D.session.open_ts,D.session.strike]]:[]).filter(s=>s[0]+900>=tL&&s[0]<=tR);
 const rng=(H?H.range:[]).filter(r=>r.close>=tL&&r.close-3600<=tR&&r.lo!=null);
 let ys=vis.map(p=>p[1]);if(!ys.length&&last)ys=[last[1]];
 for(const s of ses)ys.push(s[1]);for(const r of rng)ys.push(r.lo,r.hi);
 let lo=Math.min(...ys),hi=Math.max(...ys);const pad=(hi-lo)*0.06+5;lo-=pad;hi+=pad;
 const Y=v=>T+(hi-v)/(hi-lo)*(h-T-B);
 timegrid(g,X,tL,tR,T,B,h,true);
 g.font='11px system-ui';
 const step=[5,10,20,25,50,100,200,250,500,1000,2000,5000].find(s=>(hi-lo)/s<=9)||10000;
 for(let v=Math.ceil(lo/step)*step;v<=hi;v+=step){g.strokeStyle=css('--grid');g.beginPath();g.moveTo(L,Y(v));g.lineTo(w-R,Y(v));g.stroke();g.fillStyle=css('--mut');g.fillText(usd(v),4,Y(v)+4)}
 g.save();g.beginPath();g.rect(L,0,w-L-R,h);g.clip();
 const st={won:'--win',lost:'--lose',open:'--open'};
 for(const r of rng){const x0=X(r.close-3600),x1=X(r.close),c=css(r.side==='NO'?'--no':'--yes');
  g.fillStyle=c;g.globalAlpha=.10;g.fillRect(x0,Y(r.hi),x1-x0,Y(r.lo)-Y(r.hi));g.globalAlpha=1;
  g.strokeStyle=css(st[r.status]);g.lineWidth=1.5;g.strokeRect(x0,Y(r.hi),x1-x0,Y(r.lo)-Y(r.hi));
}
 g.strokeStyle=css('--target');g.lineWidth=2.2;
 for(const s of ses){g.beginPath();g.moveTo(X(s[0]),Y(s[1]));g.lineTo(X(s[0]+900),Y(s[1]));g.stroke()}
 
 if(ZOOMS[zi]>=64){g.fillStyle=css('--target');g.font='600 10.5px system-ui';for(const s of ses)g.fillText((ZOOMS[zi]>=128?'target ':'')+usd(s[1]),X(s[0])+3,Y(s[1])-4)}
 g.strokeStyle=css('--line');g.lineWidth=1.3;g.beginPath();vis.forEach((p,i)=>{const x=X(p[0]),y=Y(p[1]);i?g.lineTo(x,y):g.moveTo(x,y)});g.stroke();
 const at=t=>{let a=0,b=P.length-1;while(a<b){const m=(a+b)>>1;P[m][0]<t?a=m+1:b=m}return P[a]?P[a][1]:null};
 const placed=[];
 const label=(x,y,txt,col)=>{g.font='600 10.5px system-ui';const tw=g.measureText(txt).width+8,th=15;
  if(x+tw>w-R)x=Math.max(L,x-tw-28);
  let yy=y-th/2;for(let k=0;k<30;k++){const hit=placed.find(p=>x<p[0]+p[2]&&x+tw>p[0]&&yy<p[1]+p[3]&&yy+th>p[1]);if(!hit)break;yy=hit[1]+hit[3]+1}
  placed.push([x,yy,tw,th]);g.fillStyle=css('--card');g.globalAlpha=.9;g.fillRect(x,yy,tw,th);g.globalAlpha=1;
  g.strokeStyle=col;g.lineWidth=1;g.strokeRect(x,yy,tw,th);g.fillStyle=col;g.fillText(txt,x+4,yy+11);return [x,yy+th/2,tw]};
 const full=ZOOMS[zi]>=128,short=ZOOMS[zi]>=32;
 for(const r of rng){const y=at(r.t);if(y==null)continue;const x=X(r.t),yd=Y(y),col=css(st[r.status]),mid=Y((r.lo+r.hi)/2);
  const res=r.status==='open'?'OPEN':r.status.toUpperCase()+' '+money(r.pnl);
  const txt=full?`${hm(r.t)} ${r.side} ${usd(r.lo)}–${usd(r.hi+0.01)} x${r.n} @${Math.round(r.price*100)}¢, model ${r.model?(r.model*100).toFixed(1):'?'}¢ → ${res}`:`${r.side} @${Math.round(r.price*100)}¢ ${res}`;
  g.strokeStyle=col;g.lineWidth=1.2;g.setLineDash([3,3]);g.beginPath();g.moveTo(x,yd);g.lineTo(x,mid);
  g.stroke();if(short){g.setLineDash([]);const [lx,ly,tw]=label(x+14,mid,txt,col);const ex=lx>x?lx:lx+tw;g.strokeStyle=col;g.setLineDash([3,3]);g.beginPath();g.moveTo(x,mid);g.lineTo(x,ly);g.lineTo(ex,ly);g.stroke()}
  g.setLineDash([]);g.fillStyle=col;g.strokeStyle=css('--card');g.lineWidth=1.5;g.beginPath();g.arc(x,yd,5,0,7);g.fill();g.stroke()}
 for(const b of (H?H.buys:[]).filter(b=>b.t>=tL&&b.t<=tR)){const y=at(b.t);if(y==null)continue;const x=X(b.t),yy=Y(y),up=b.side==='YES';
  g.fillStyle=css(st[b.status]);g.beginPath();g.moveTo(x,yy+(up?-9:9));g.lineTo(x-6,yy+(up?2:-2));g.lineTo(x+6,yy+(up?2:-2));g.fill();
  if(short){const col=css(st[b.status]),ty=yy+(up?-28:28);
   const [lx,ly,tw]=label(x+12,ty,`${hm(b.t)} ${b.bot} ${b.side} @${b.ask?Math.round(b.ask*100)+'¢':''} ${b.level||''} → ${b.status==='open'?'OPEN':b.status.toUpperCase()+(b.pnl!=null?' '+money(b.pnl):'')}`,col);g.strokeStyle=col;g.setLineDash([3,3]);g.beginPath();g.moveTo(x,yy);g.lineTo(lx>x?lx:lx+tw,ly);g.stroke();g.setLineDash([])}}
 if(last&&last[0]>=tL&&last[0]<=tR){g.fillStyle=css('--fg');g.beginPath();g.arc(X(last[0]),Y(last[1]),4,0,7);g.fill()}
 g.restore();
 plchart(tL,tR);tables(last?last[1]:null);
}
const BOTS=[['Total','--line'],['Range','--yes'],['Breakdown','--no'],['Fast push','--fp']];
function plchart(tL,tR){
 const [g,w]=setup($('pl'),240),h=240,L=66,R=8,T=8,B=22;
 const X=t=>L+(t-tL)/(tR-tL)*(w-L-R);
 const pts=D.pnl.map(r=>({...r,t:tsOf(r.time_utc)}));
 const [T0]=range();const series={},run={};for(const [b] of BOTS){series[b]=[[T0,0]];run[b]=0}
 for(const p of pts){run[p.bot]+=p.pnl;run.Total+=p.pnl;series[p.bot].push([p.t,run[p.bot]]);series.Total.push([p.t,run.Total])}
 for(const [b] of BOTS){const s=series[b];s.push([D.now,s[s.length-1][1]])}
 const valAt=(s,t)=>{let v=0;for(const p of s){if(p[0]>t)break;v=p[1]}return v};
 let vals=[0];for(const [b] of BOTS){const s=series[b];vals.push(valAt(s,tL),valAt(s,tR));for(const p of s)if(p[0]>=tL&&p[0]<=tR)vals.push(p[1])}
 let lo=Math.min(...vals),hi=Math.max(...vals);const pad=(hi-lo)*0.12+0.5;lo-=pad;hi+=pad;
 const Y=v=>T+(hi-v)/(hi-lo)*(h-T-B);
 timegrid(g,X,tL,tR,T,B,h,true);
 g.font='11px system-ui';
 const step=[0.5,1,2,5,10,20,50,100,200,500,1000].find(s=>(hi-lo)/s<=6)||2000;
 for(let v=Math.ceil(lo/step)*step;v<=hi;v+=step){g.strokeStyle=css('--grid');g.beginPath();g.moveTo(L,Y(v));g.lineTo(w-R,Y(v));g.stroke();g.fillStyle=css('--mut');g.fillText(money(v).replace('+',''),4,Y(v)+4)}
 g.save();g.beginPath();g.rect(L,0,w-L-R,h);g.clip();
 g.strokeStyle=css('--mut');g.beginPath();g.moveTo(L,Y(0));g.lineTo(w-R,Y(0));g.stroke();
 const leg=[];
 for(const [b,col] of BOTS){const s=series[b];g.strokeStyle=css(col);g.lineWidth=b==='Total'?2.5:1.8;g.setLineDash(b==='Total'?[7,5]:[]);g.beginPath();
  s.forEach((p,i)=>{if(!i){g.moveTo(X(p[0]),Y(p[1]));return}g.lineTo(X(p[0]),Y(s[i-1][1]));g.lineTo(X(p[0]),Y(p[1]))});g.stroke();g.setLineDash([]);
  leg.push(`<span style="color:var(${col})">■</span> ${b==='Total'?'Cumulative, all bots (dashed)':b} ${money(s[s.length-1][1])}`)}
 for(const p of pts){if(p.won)continue;g.fillStyle=css('--lose');g.beginPath();g.arc(X(p.t),Y(valAt(series.Total,p.t)),4,0,7);g.fill()}
 g.restore();
 $('pl-legend').innerHTML=leg.join(' &nbsp; ')+' &nbsp; red dot = losing trade';
}
function tables(px){
 let rows='<tr><th>Bought (UTC)</th><th>Side</th><th>Band</th><th>Contracts</th><th>Paid</th><th>Model</th><th>Now</th><th>Closes</th></tr>';
 const ps=D.range.open.filter(p=>p.lo!=null);
 for(const p of ps){if(px==null)break;const inside=px>=p.lo&&px<=p.hi,win=(p.side==='NO')!==inside;
  const dist=inside?Math.min(px-p.lo,p.hi-px):(px<p.lo?p.lo-px:px-p.hi);
  rows+=`<tr><td>${p.time_utc.slice(11,16)}</td><td>${p.side}</td><td>${usd(p.lo)}–${usd(p.hi+0.01)}</td><td>${p.contracts}</td><td>${Math.round(p.avg_price*100)}¢</td><td>${(p.model*100).toFixed(1)}¢</td>
  <td><span class="pill ${win?'w':'l'}">${win?'winning':'losing'} ${win?money(p.contracts*(1-p.avg_price)-p.fee):money(-p.cost)}</span> <span class="mut">${usd(dist)} ${inside?'inside':'outside'}</span></td><td>${hm(p.close_ts)}</td></tr>`}
 if(!ps.length)rows+='<tr><td colspan="8" class="mut">None open. The range bot starts checking at 30 minutes left in each hour.</td></tr>';
 $('pos').innerHTML=rows;
 const s=D.session||{};rows='<tr><th>Break (UTC)</th><th>Bot</th><th>Side</th><th>Level</th><th>Ask</th><th>Now</th></tr>';
 for(const b of D.buys){const win=(b.side==='YES')===(px>s.strike);
  rows+=`<tr><td>${b.time_utc.slice(11,19)}</td><td>${b.bot}</td><td>${b.side}</td><td>${b.level} ${b.level_price?usd(b.level_price):''}</td><td>${b.ask?Math.round(b.ask*100)+'¢':''}</td><td><span class="pill ${win?'w':'l'}">${win?'winning':'losing'}</span></td></tr>`}
 if(!D.buys.length)rows+=`<tr><td colspan="6" class="mut">No buys this session${s.strike?' (strike '+usd(s.strike)+')':''}. Both bots only buy in the last 1–3 minutes.</td></tr>`;
 $('pos15').innerHTML=rows;
}
tick();setInterval(tick,2000);setInterval(loadHistory,60000);
</script></body></html>
"""
