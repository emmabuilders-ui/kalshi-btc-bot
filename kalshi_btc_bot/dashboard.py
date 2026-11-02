"""Live chart of the paper bots, served from the server itself (read-only, no orders).

Open http://<server-ip>:8080/<token>/ in any browser. The page refreshes every 2 seconds:
the live BRTI price (Kalshi's cfbenchmarks feed, 1-second values), the hourly range bot's open
bands, the 15-minute strike, and the buys of the two 15-minute bots, with settled P/L so far.

The token (a random path, made by setup.sh) keeps the page private to whoever has the link.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import threading
import time
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


class Dashboard:
    def __init__(self, logs: dict[str, Path], key_id=None, key_path=None):
        self.k = KalshiClient(key_id, key_path)
        self.logs = logs
        self.prices: dict[int, float] = {}
        self.session: dict | None = None
        self.lock = threading.Lock()
        self.err = ""

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
:root{--bg:#f6f7f9;--card:#fff;--fg:#1b1f24;--mut:#66707a;--line:#1b1f24;--grid:#e3e6ea;
--no:#c0392b;--yes:#1f8a4c;--strike:#2f6fb2;--win:#1f8a4c;--lose:#c0392b;--win-bg:#e6f4ec;--lose-bg:#fbe9e7}
@media (prefers-color-scheme:dark){:root{--bg:#121417;--card:#1b1e22;--fg:#e8eaed;--mut:#9aa3ad;
--line:#e8eaed;--grid:#2a2e34;--no:#ef6b5b;--yes:#4cc27e;--strike:#6ea8e8;--win:#4cc27e;--lose:#ef6b5b;
--win-bg:#17301f;--lose-bg:#3a1c19;color-scheme:dark}}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.4 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
.wrap{max-width:1200px;margin:0 auto;padding:16px}
h1{font-size:18px;margin:0}
.top{display:flex;flex-wrap:wrap;align-items:baseline;gap:8px 20px;margin-bottom:12px}
.px{font-size:26px;font-weight:700;font-variant-numeric:tabular-nums}
.mut{color:var(--mut)}
.tabs{display:flex;gap:6px;margin:8px 0}
.tabs button{border:1px solid var(--grid);background:var(--card);color:var(--fg);padding:6px 12px;border-radius:6px;cursor:pointer;font:inherit}
.tabs button.on{border-color:var(--strike);color:var(--strike);font-weight:600}
.card{background:var(--card);border:1px solid var(--grid);border-radius:8px;padding:12px}
canvas{width:100%;height:440px;display:block}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:10px;margin-top:12px}
table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}
td,th{padding:6px 8px;text-align:left;border-bottom:1px solid var(--grid);white-space:nowrap}
th{font-size:12px;color:var(--mut);font-weight:600}
.pill{padding:2px 8px;border-radius:10px;font-weight:600;font-size:12px}
.w{background:var(--win-bg);color:var(--win)}.l{background:var(--lose-bg);color:var(--lose)}
.scroll{overflow-x:auto}
.big{font-size:20px;font-weight:700;font-variant-numeric:tabular-nums}
</style></head><body><div class="wrap">
<div class="top"><h1>Paper bots, live</h1><span class="px" id="px">…</span>
<span class="mut" id="clock"></span><span class="mut" id="err"></span></div>
<div class="tabs"><button id="t1" class="on">Hourly range bot</button><button id="t2">15-minute bots</button></div>
<div class="card"><canvas id="c"></canvas></div>
<div class="card scroll" style="margin-top:12px"><table id="pos"></table></div>
<div class="grid">
<div class="card"><div class="mut">Range bot, settled</div><div class="big" id="sr"></div></div>
<div class="card"><div class="mut">Breakdown bot, settled</div><div class="big" id="sb"></div></div>
<div class="card"><div class="mut">Fast-push bot, settled</div><div class="big" id="sf"></div></div>
</div>
<div class="card" style="margin-top:12px"><div style="display:flex;flex-wrap:wrap;gap:6px 16px;align-items:baseline">
<b>Profit and loss, all bots (after fees)</b><span class="mut" id="pl-legend"></span></div><canvas id="pl" style="height:260px"></canvas></div>
<p class="mut">BRTI is the index Kalshi settles on; settlement is its average over the last 60 seconds. Paper trades only. Updates every 2 seconds.</p>
</div>
<script>
let D=null,view='hour';
const $=id=>document.getElementById(id);
$('t1').onclick=()=>{view='hour';$('t1').className='on';$('t2').className='';draw()};
$('t2').onclick=()=>{view='15m';$('t2').className='on';$('t1').className='';draw()};
const css=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const hm=t=>new Date(t*1000).toISOString().slice(11,16);
const hms=t=>new Date(t*1000).toISOString().slice(11,19);
const money=x=>(x<0?'-$':'+$')+Math.abs(x).toFixed(2);
const usd=x=>'$'+x.toLocaleString('en-US',{maximumFractionDigits:0});
function sum(id,s){$(id).textContent=s.trades?`${s.wins}/${s.trades} won, ${money(s.pnl)}`:'no trades yet'}
async function tick(){try{const r=await fetch('data.json',{cache:'no-store'});D=await r.json();draw()}catch(e){$('err').textContent='connection lost, retrying'}}
function draw(){
 if(!D)return;const P=D.prices;if(!P.length)return;
 const last=P[P.length-1];$('px').textContent=usd(last[1]);
 $('err').textContent=D.error||'';
 sum('sr',D.range.settled);sum('sb',D.breakdown);sum('sf',D.fastpush);
 let t0,t1,bands=[],lines=[],marks=[];
 if(view==='hour'){t1=D.hour_close;t0=t1-3600;
  $('clock').textContent=`${hms(last[0])} UTC, hour closes ${hm(t1)} UTC (${Math.max(0,Math.round((t1-last[0])/60))} min left)`;
  for(const p of D.range.open){if(p.close_ts!==t1||p.lo==null)continue;
   bands.push({lo:p.lo,hi:p.hi,col:p.side==='NO'?css('--no'):css('--yes'),lab:`${p.side} ${usd(p.lo)}–${usd(p.hi+0.01)} x${p.contracts} @ ${Math.round(p.avg_price*100)}¢`});
   marks.push({t:Date.parse(p.time_utc.replace(' ','T')+'Z')/1000,col:p.side==='NO'?css('--no'):css('--yes')});}
 }else{const s=D.session||{};t1=s.close_ts||Math.ceil(last[0]/900)*900;t0=t1-900;
  $('clock').textContent=`${hms(last[0])} UTC, 15-min closes ${hm(t1)} UTC (${Math.max(0,t1-last[0])} s left)`;
  if(s.strike)lines.push({y:s.strike,col:css('--strike'),lab:`strike ${usd(s.strike)} (above = YES wins)`});
  for(const b of D.buys){marks.push({t:Date.parse(b.time_utc.replace(' ','T')+'Z')/1000,col:b.side==='NO'?css('--no'):css('--yes'),lab:`${b.bot} ${b.side}`});
   if(b.level_price)lines.push({y:b.level_price,col:css('--mut'),lab:`${b.level} ${usd(b.level_price)}`,dash:true});}
 }
 const W=P.filter(p=>p[0]>=t0-5);
 const c=$('c'),dpr=devicePixelRatio||1,w=c.clientWidth,h=c.clientHeight;c.width=w*dpr;c.height=h*dpr;
 const g=c.getContext('2d');g.setTransform(dpr,0,0,dpr,0,0);g.clearRect(0,0,w,h);
 const L=64,R=12,T=10,B=26;
 let ys=W.map(p=>p[1]);for(const b of bands){ys.push(b.lo,b.hi)}for(const l of lines)ys.push(l.y);
 let lo=Math.min(...ys),hi=Math.max(...ys);const pad=(hi-lo)*0.08+5;lo-=pad;hi+=pad;
 const X=t=>L+(t-t0)/(t1-t0)*(w-L-R),Y=v=>T+(hi-v)/(hi-lo)*(h-T-B);
 g.font='11px system-ui';g.fillStyle=css('--mut');g.strokeStyle=css('--grid');g.lineWidth=1;
 const step=[5,10,20,25,50,100,200,250,500,1000].find(s=>(hi-lo)/s<=8)||1000;
 for(let v=Math.ceil(lo/step)*step;v<=hi;v+=step){g.beginPath();g.moveTo(L,Y(v));g.lineTo(w-R,Y(v));g.stroke();g.fillText(usd(v),4,Y(v)+4)}
 const tstep=view==='hour'?600:120;
 for(let t=t0;t<=t1;t+=tstep){g.beginPath();g.moveTo(X(t),T);g.lineTo(X(t),h-B);g.stroke();g.fillText(hm(t),X(t)-14,h-8)}
 g.fillStyle=css('--grid');g.globalAlpha=.5;g.fillRect(X(t1-60),T,X(t1)-X(t1-60),h-T-B);g.globalAlpha=1;
 for(const b of bands){g.fillStyle=b.col;g.globalAlpha=.14;g.fillRect(L,Y(b.hi),w-L-R,Y(b.lo)-Y(b.hi));g.globalAlpha=1;
  g.strokeStyle=b.col;g.beginPath();g.moveTo(L,Y(b.hi));g.lineTo(w-R,Y(b.hi));g.moveTo(L,Y(b.lo));g.lineTo(w-R,Y(b.lo));g.stroke();
  g.fillStyle=b.col;g.font='600 11px system-ui';g.fillText(b.lab,L+6,Y(b.hi)+13)}
 for(const l of lines){g.strokeStyle=l.col;g.setLineDash(l.dash?[4,4]:[]);g.beginPath();g.moveTo(L,Y(l.y));g.lineTo(w-R,Y(l.y));g.stroke();g.setLineDash([]);
  g.fillStyle=l.col;g.font='600 11px system-ui';g.fillText(l.lab,L+6,Y(l.y)-4)}
 g.strokeStyle=css('--line');g.lineWidth=1.4;g.beginPath();W.forEach((p,i)=>{const x=X(p[0]),y=Y(p[1]);i?g.lineTo(x,y):g.moveTo(x,y)});g.stroke();
 for(const m of marks){const p=W.find(q=>q[0]>=m.t);if(!p)continue;g.fillStyle=m.col;g.beginPath();g.arc(X(p[0]),Y(p[1]),5,0,7);g.fill();
  if(m.lab){g.font='11px system-ui';g.fillText(m.lab,X(p[0])+7,Y(p[1])-7)}}
 g.fillStyle=css('--fg');g.beginPath();g.arc(X(last[0]),Y(last[1]),4,0,7);g.fill();
 table(last[1]);
}
function table(px){
 let rows='';
 if(view==='hour'){rows='<tr><th>Bought (UTC)</th><th>Side</th><th>Band</th><th>Contracts</th><th>Paid</th><th>Model</th><th>Now</th><th>Closes</th></tr>';
  const ps=D.range.open.filter(p=>p.lo!=null);
  for(const p of ps){const inside=px>=p.lo&&px<=p.hi,win=(p.side==='NO')!==inside;
   const dist=inside?Math.min(px-p.lo,p.hi-px):(px<p.lo?p.lo-px:px-p.hi);
   rows+=`<tr><td>${p.time_utc.slice(11,16)}</td><td>${p.side}</td><td>${usd(p.lo)}–${usd(p.hi+0.01)}</td><td>${p.contracts}</td><td>${Math.round(p.avg_price*100)}¢</td><td>${(p.model*100).toFixed(1)}¢</td>
   <td><span class="pill ${win?'w':'l'}">${win?'winning':'losing'} ${win?money(p.contracts*(1-p.avg_price)-p.fee):money(-p.cost)}</span> <span class="mut">${usd(dist)} ${inside?'inside':'outside'}</span></td><td>${hm(p.close_ts)}</td></tr>`}
  if(!ps.length)rows+='<tr><td colspan="8" class="mut">No open range positions. The bot starts checking at 30 minutes left in each hour.</td></tr>';
 }else{rows='<tr><th>Break (UTC)</th><th>Bot</th><th>Side</th><th>Level</th><th>Ask</th><th>Now</th></tr>';
  const s=D.session||{};
  for(const b of D.buys){const win=(b.side==='YES')===(px>s.strike);
   rows+=`<tr><td>${b.time_utc.slice(11,19)}</td><td>${b.bot}</td><td>${b.side}</td><td>${b.level} ${b.level_price?usd(b.level_price):''}</td><td>${b.ask?Math.round(b.ask*100)+'¢':''}</td><td><span class="pill ${win?'w':'l'}">${win?'winning':'losing'}</span></td></tr>`}
  if(!D.buys.length)rows+='<tr><td colspan="6" class="mut">No 15-minute trades this session. Both bots only buy in the last 1–3 minutes.</td></tr>';
 }
 $('pos').innerHTML=rows;
}
const BOTS=[['Total','--line'],['Range','--yes'],['Breakdown','--no'],['Fast push','--strike']];
function plchart(){
 const c=$('pl'),dpr=devicePixelRatio||1,w=c.clientWidth,h=c.clientHeight;c.width=w*dpr;c.height=h*dpr;
 const g=c.getContext('2d');g.setTransform(dpr,0,0,dpr,0,0);g.clearRect(0,0,w,h);
 const pts=D.pnl.map(r=>({...r,t:Date.parse(r.time_utc.replace(' ','T')+'Z')/1000}));
 g.font='12px system-ui';g.fillStyle=css('--mut');
 if(!pts.length){g.fillText('No settled trades yet.',12,24);$('pl-legend').textContent='';return}
 const series={};for(const [b] of BOTS)series[b]=[[pts[0].t-60,0]];
 const run={Total:0,Range:0,Breakdown:0,'Fast push':0};
 for(const p of pts){run[p.bot]+=p.pnl;run.Total+=p.pnl;series[p.bot].push([p.t,run[p.bot]]);series.Total.push([p.t,run.Total])}
 const now=D.now;for(const [b] of BOTS){const s=series[b];s.push([now,s[s.length-1][1]])}
 const t0=pts[0].t-60,t1=now;let vals=Object.values(series).flat().map(p=>p[1]);
 let lo=Math.min(0,...vals),hi=Math.max(0,...vals);const pad=(hi-lo)*0.12+0.5;lo-=pad;hi+=pad;
 const L=56,R=70,T=10,B=24,X=t=>L+(t-t0)/(t1-t0)*(w-L-R),Y=v=>T+(hi-v)/(hi-lo)*(h-T-B);
 g.strokeStyle=css('--grid');g.lineWidth=1;g.font='11px system-ui';
 const step=[0.5,1,2,5,10,20,50,100,200,500].find(s=>(hi-lo)/s<=6)||1000;
 for(let v=Math.ceil(lo/step)*step;v<=hi;v+=step){g.beginPath();g.moveTo(L,Y(v));g.lineTo(w-R,Y(v));g.stroke();g.fillStyle=css('--mut');g.fillText(money(v).replace('+',''),4,Y(v)+4)}
 const span=t1-t0,ts=span>4*86400?86400:span>86400?21600:span>21600?3600*3:span>7200?3600:900;
 for(let t=Math.ceil(t0/ts)*ts;t<=t1;t+=ts){g.fillStyle=css('--mut');const d=new Date(t*1000).toISOString();g.fillText(ts>=86400?d.slice(5,10):d.slice(11,16),X(t)-14,h-6)}
 g.strokeStyle=css('--mut');g.beginPath();g.moveTo(L,Y(0));g.lineTo(w-R,Y(0));g.stroke();
 let leg=[];
 for(const [b,col] of BOTS){const s=series[b];
  g.strokeStyle=css(col);g.lineWidth=b==='Total'?2.5:1.8;g.setLineDash(b==='Total'?[7,5]:[]);g.beginPath();
  s.forEach((p,i)=>{if(!i){g.moveTo(X(p[0]),Y(p[1]));return}g.lineTo(X(p[0]),Y(s[i-1][1]));g.lineTo(X(p[0]),Y(p[1]))});g.stroke();g.setLineDash([]);
  const e=s[s.length-1];g.fillStyle=css(col);g.font=(b==='Total'?'600 ':'')+'11px system-ui';g.fillText(money(e[1]),X(e[0])+4,Y(e[1])+4);
  leg.push(`<span style="color:var(${col})">\u25A0</span> ${b==='Total'?'Cumulative, all bots (dashed)':b} ${money(e[1])}`)}
 for(const p of pts){if(p.won)continue;g.fillStyle=css('--lose');g.beginPath();g.arc(X(p.t),Y(series.Total.find(q=>q[0]===p.t)[1]),4,0,7);g.fill()}
 $('pl-legend').innerHTML=leg.join(' &nbsp; ')+' &nbsp; <span class="mut">red dot = losing trade</span>';
}
const _draw=draw;draw=function(){_draw();if(D)plchart()};
tick();setInterval(tick,2000);addEventListener('resize',draw);
</script></body></html>"""
