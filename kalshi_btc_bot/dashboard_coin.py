"""Chart pages for the ETH, SOL and XRP break-with-gate paper bot (kalshi-coinbreak), served by the dashboard at
http://<server-ip>:8080/<token>/coinbreak/<COIN>/ . Read-only: it only reads the bot's log files, so it adds no
API calls and almost no work to the server (a few small files, answered from a 5-second cache)."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

from .dashboard import FILTER_CSS, FILTER_JS, _json, _num, _rows, _ts, gz_lines, perf_row

COINS = ("ETH", "SOL", "XRP")
PRICE_HOURS = 3                 # hours of 1-second index prices sent to the page


def coin_prices(d: Path, now: float, hours: int = PRICE_HOURS) -> list[list[float]]:
    """1-second index prices of the last `hours` hours from the bot's hourly gzip files (damage-tolerant)."""
    out: dict[int, float] = {}
    h0 = int(now) // 3600 * 3600 - (hours - 1) * 3600
    for h in range(h0, int(now) // 3600 * 3600 + 1, 3600):
        day = datetime.fromtimestamp(h, timezone.utc).strftime("%Y-%m-%d/%H")
        for line in gz_lines(d / "prices" / f"{day}.csv.gz"):
            a, _, b = line.partition(",")
            try:
                out[int(a)] = float(b.split(",")[0])
            except ValueError:
                continue            # the header line, or half a line from a cut-off write
    return [[t, out[t]] for t in sorted(out) if t >= now - hours * 3600]


def coin_data(d: Path, coin: str, now: float | None = None) -> dict:
    now = now or time.time()
    prices = coin_prices(d, now)
    seen = {}
    for r in _rows(d / "sessions.csv"):
        if r.get("ticker") and r.get("strike"):
            seen[r["ticker"]] = {"ticker": r["ticker"], "open": _ts(r["open_utc"]), "close": _ts(r["close_utc"]),
                                 "strike": _num(r["strike"])}
    t0 = prices[0][0] if prices else now - PRICE_HOURS * 3600
    sessions = sorted((s for s in seen.values() if s["close"] >= t0), key=lambda s: s["open"])
    trades = []
    for r in _rows(d / "trades.csv"):
        trades.append({"time_utc": r.get("time_utc"), "ticker": r.get("ticker"), "side": r.get("side"),
                       "contracts": _num(r.get("contracts")), "price": _num(r.get("avg_price")),
                       "cost": _num(r.get("cost")), "model": _num(r.get("model")), "secs_left": _num(r.get("secs_left")),
                       "won": str(r.get("won")) == "1", "pnl": _num(r.get("pnl")), "result": r.get("result")})
    settled = {(t["ticker"], t["side"]) for t in trades}
    open_ = [{"time_utc": o.get("time_utc"), "ticker": o.get("ticker"), "side": o.get("side"),
              "contracts": o.get("contracts"), "price": o.get("avg_price"), "cost": o.get("cost"),
              "model": o.get("model"), "secs_left": o.get("secs_left")}
             for o in _json(d / "open_trades.json", []) if (o.get("ticker"), o.get("side")) not in settled]
    signals = []
    for r in _rows(d / "signals.csv")[-300:]:
        signals.append({"time_utc": r.get("time_utc"), "ticker": r.get("ticker"), "side": r.get("side"),
                        "secs_left": _num(r.get("secs_left")), "index": _num(r.get("index")),
                        "strike": _num(r.get("strike")), "threshold": _num(r.get("break_threshold")),
                        "model": _num(r.get("model")), "ask": _num(r.get("ask")), "decision": r.get("decision"),
                        "contracts": _num(r.get("contracts")), "price": _num(r.get("avg_price")),
                        "lag": _num(r.get("lag_s"))})
    return {"now": now, "coin": coin, "prices": prices, "sessions": sessions, "trades": trades, "open": open_,
            "signals": signals, "status": _json(d / "status.json", {}),
            "perf": perf_row(f"Break + gate {coin}", _rows(d / "trades.csv"))}


PAGEC = r"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>__COIN__ break bot</title>
<style>
:root{--bg:#f9f9f7;--card:#fcfcfb;--fg:#0b0b0b;--mut:#52514e;--grid:#e6e6e3;--grid2:#c9c8c2;--ladder-t:#1d4f91;
--price:#0b0b0b;--strike:#2f6fb2;--zone:#b8875a;--yes:#1f8a4c;--no:#c0392b;--win:#0ca30c;--lose:#d03b3b;--open:#e0a100;--skip:#8a8984;--pl:#3b5bdb}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#0d0d0d;--card:#1a1a19;--fg:#f0efec;--mut:#c3c2b7;--grid:#2a2a28;--grid2:#45443f;--ladder-t:#7aa7e0;
--price:#f0efec;--strike:#6ea8e8;--zone:#d2a87a;--yes:#4cc27e;--no:#ef6b5b;--win:#2fbf2f;--lose:#e66767;--open:#fab219;--skip:#9a9992;--pl:#8da2ff;color-scheme:dark}}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.4 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
.wrap{max-width:1300px;margin:0 auto;padding:16px}
h1{font-size:18px;margin:0}h2{font-size:15px;margin:0 0 8px}
.mut{color:var(--mut)}.top{display:flex;flex-wrap:wrap;gap:6px 14px;align-items:center;margin-bottom:12px}
.big{font-size:22px;font-weight:700;font-variant-numeric:tabular-nums}
.card{background:var(--card);border:1px solid var(--grid);border-radius:8px;padding:12px;margin-bottom:12px;min-width:0}
.cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:10px;margin-bottom:12px}
.cards .card{margin:0}
canvas{display:block;width:100%}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
th,td{padding:5px 8px;border-bottom:1px solid var(--grid);text-align:right;white-space:nowrap}
th:first-child,td:first-child,td.l,th.l{text-align:left}
th{color:var(--mut);font-weight:600;position:sticky;top:0;background:var(--card)}
.scroll{max-height:420px;overflow:auto}
.won{color:var(--win);font-weight:600}.lost{color:var(--lose);font-weight:600}.op{color:var(--open);font-weight:600}
.navbtn,.bar button{padding:5px 12px;border:1px solid var(--grid2);border-radius:6px;background:var(--card);color:var(--fg);text-decoration:none;font-weight:600;font-size:13px;cursor:pointer}
.navbtn:hover,.bar button:hover{border-color:var(--ladder-t)}.navbtn.cur{border-color:var(--ladder-t);box-shadow:0 0 0 1px var(--ladder-t) inset}
.bar{display:flex;flex-wrap:wrap;gap:6px;margin:0 0 8px;align-items:center}
.bar button.on{border-color:var(--ladder-t);box-shadow:0 0 0 1px var(--ladder-t) inset}
.key{display:flex;flex-wrap:wrap;gap:4px 14px;font-size:12px;color:var(--mut);margin-top:6px}
.key i{display:inline-block;width:12px;height:10px;margin-right:4px;vertical-align:middle;border-radius:2px}
#tip{position:fixed;pointer-events:none;background:var(--card);border:1px solid var(--grid2);border-radius:6px;padding:6px 8px;font-size:12px;display:none;z-index:5;max-width:320px}
/*FILTER_CSS*/</style></head><body><div class="wrap">
<div class="top"><a class="navbtn" href="../../">← Main chart</a>
<a class="navbtn" id="n-ETH" href="../ETH/">ETH</a><a class="navbtn" id="n-SOL" href="../SOL/">SOL</a><a class="navbtn" id="n-XRP" href="../XRP/">XRP</a>
<h1>__COIN__ 15-min · break with gate</h1><span class="mut">Paper only, no real orders · times ET · updates every 10 s</span><span id="err" class="lost"></span></div>
<div class="top"><span>Total profit <span class="big" id="tot">–</span></span><span id="sub" class="mut"></span></div>
<div class="cards" id="cards"></div>
<div class="card"><div class="bar"><span class="mut">Window</span><button data-w="1800">30 min</button><button data-w="3600" class="on">1 hour</button><button data-w="7200">2 hours</button><button data-w="10800">3 hours</button><span class="mut" id="pxnote"></span></div>
<canvas id="px" height="380"></canvas>
<div class="key"><span><i style="background:var(--price)"></i>__COIN__ index (what Kalshi settles on, 1 s)</span><span><i style="background:var(--strike)"></i>strike of each 15-minute market</span><span><i style="background:var(--zone);opacity:.35"></i>strike ± break distance while 60–150 s are left</span><span><b style="color:var(--win)">▲</b> bought, won · <b style="color:var(--lose)">▲</b> lost · <b style="color:var(--open)">▲</b> open · <b style="color:var(--skip)">○</b> break seen, no buy (hover for why)</span></div></div>
<div class="card"><h2>Profit, settled trades</h2><canvas id="pl" height="150"></canvas></div>
<div class="card"><h2>Results</h2><div class="scroll"><table id="perf"></table></div></div>
<div class="card"><h2>Open positions</h2><div class="scroll"><table id="opn"></table></div></div>
<div class="card"><h2>Every break the bot saw (bought or skipped)</h2><div class="scroll"><table id="sig"></table></div></div>
<div class="card"><h2>All settled trades</h2><div class="scroll"><table id="trd"></table></div></div>
</div><div id="tip"></div>
<script>
/*FILTER_JS*/
const COIN='__COIN__',BF=5/75000,TZ='America/New_York';
document.getElementById('n-'+COIN).classList.add('cur');
const fDT=new Intl.DateTimeFormat('en-US',{timeZone:TZ,month:'short',day:'numeric',hour:'numeric',minute:'2-digit',hour12:true});
const fHM=new Intl.DateTimeFormat('en-US',{timeZone:TZ,hour:'numeric',minute:'2-digit',hour12:true});
const fHMS=new Intl.DateTimeFormat('en-US',{timeZone:TZ,hour:'numeric',minute:'2-digit',second:'2-digit',hour12:true});
const css=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const ts=s=>Date.parse(s.replace(' ','T')+'Z')/1000;
const money=x=>(x<0?'−$':'$')+Math.abs(x).toFixed(2);
const sgn=x=>(x>0?'+':x<0?'−':'')+'$'+Math.abs(x).toFixed(2);
const cents=x=>x==null?'–':(x*100).toFixed(1)+'¢';
const pxf=v=>v>=100?v.toFixed(2):v>=10?v.toFixed(3):v.toFixed(4);
function niceStep(r){const p=Math.pow(10,Math.floor(Math.log10(r||1)));for(const m of [1,2,2.5,5,10])if(r<=m*p)return m*p;return 10*p}
function setup(c){const r=c.getBoundingClientRect(),d=window.devicePixelRatio||1;if(!c.dataset.h)c.dataset.h=c.getAttribute('height');const h=+c.dataset.h;c.width=r.width*d;c.height=h*d;c.style.height=h+'px';const x=c.getContext('2d');x.setTransform(d,0,0,d,0,0);return [x,r.width,h]}
let D=null,WIN=3600,MARK=[],GEO=null;
const outcome=(tk,side)=>{const t=(D.trades||[]).find(t=>t.ticker===tk&&t.side===side);if(t)return t.won?'won':'lost';return (D.open||[]).some(o=>o.ticker===tk&&o.side===side)?'open':'open'};
function drawPX(){
  const c=document.getElementById('px'),[x,W,H]=setup(c),L=70,R=12,T=10,B=26;const P=D.prices||[];
  MARK=[];
  if(!P.length){x.fillStyle=css('--mut');x.font='13px system-ui';x.fillText('Waiting for the bot to log index prices…',L,40);return}
  const t1=P[P.length-1][0],t0=t1-WIN,V=P.filter(p=>p[0]>=t0);
  const ses=(D.sessions||[]).filter(s=>s.close>=t0&&s.open<=t1);
  let lo=Infinity,hi=-Infinity;V.forEach(p=>{lo=Math.min(lo,p[1]);hi=Math.max(hi,p[1])});
  ses.forEach(s=>{if(s.strike!=null&&s.strike>lo-(hi-lo)*0.6&&s.strike<hi+(hi-lo)*0.6){lo=Math.min(lo,s.strike);hi=Math.max(hi,s.strike)}});
  const pad=Math.max((hi-lo)*0.08,hi*0.0004);lo-=pad;hi+=pad;
  const X=t=>L+(t-t0)/(t1-t0)*(W-L-R),Y=v=>T+(hi-v)/(hi-lo)*(H-T-B);
  x.font='12px system-ui';x.lineWidth=1;x.strokeStyle=css('--grid');x.fillStyle=css('--mut');
  const st=niceStep((hi-lo)/5);for(let v=Math.ceil(lo/st)*st;v<=hi;v+=st){x.beginPath();x.moveTo(L,Y(v));x.lineTo(W-R,Y(v));x.stroke();x.textAlign='right';x.fillText(pxf(v),L-6,Y(v)+4)}
  x.textAlign='center';const n=Math.max(2,Math.floor((W-L-R)/110));for(let i=0;i<=n;i++){const t=t0+(t1-t0)*i/n;x.fillText(fHM.format(new Date(t*1000)),X(t),H-8)}
  // each market: close line, entry window, strike and the break zone
  for(const s of ses){
    const a=Math.max(s.open,t0),b=Math.min(s.close,t1);
    x.strokeStyle=css('--grid2');x.setLineDash([2,4]);x.beginPath();x.moveTo(X(s.close),T);x.lineTo(X(s.close),H-B);x.stroke();x.setLineDash([]);
    if(s.strike==null||s.strike<lo||s.strike>hi)continue;
    const za=Math.max(s.close-150,t0),zb=Math.min(s.close-60,t1);
    if(zb>za){const th=s.strike*BF;x.globalAlpha=0.22;x.fillStyle=css('--zone');x.fillRect(X(za),Y(s.strike+th),X(zb)-X(za),Y(s.strike-th)-Y(s.strike+th));x.globalAlpha=1}
    x.strokeStyle=css('--strike');x.lineWidth=1.5;x.setLineDash([6,3]);x.beginPath();x.moveTo(X(a),Y(s.strike));x.lineTo(X(b),Y(s.strike));x.stroke();x.setLineDash([]);
    x.fillStyle=css('--strike');x.textAlign='left';x.fillText(pxf(s.strike),Math.min(X(a)+3,W-R-60),Y(s.strike)-4);x.lineWidth=1}
  x.strokeStyle=css('--price');x.lineWidth=1.6;x.beginPath();let first=true;V.forEach(p=>{x[first?'moveTo':'lineTo'](X(p[0]),Y(p[1]));first=false});x.stroke();
  // markers: every break the bot judged
  for(const g of D.signals||[]){const t=ts(g.time_utc);if(t<t0||t>t1)continue;
    const v=V.find(p=>p[0]>=t);const yv=Y(v?v[1]:g.index);const px=X(t);const buy=(g.decision||'').startsWith('BUY');
    if(buy){const o=outcome(g.ticker,g.side);x.fillStyle=css(o==='won'?'--win':o==='lost'?'--lose':'--open');x.strokeStyle=css('--card');x.lineWidth=1.5;
      x.beginPath();if(g.side==='YES'){x.moveTo(px,yv-9);x.lineTo(px-7,yv+4);x.lineTo(px+7,yv+4)}else{x.moveTo(px,yv+9);x.lineTo(px-7,yv-4);x.lineTo(px+7,yv-4)}x.closePath();x.fill();x.stroke()}
    else{x.strokeStyle=css('--skip');x.lineWidth=2;x.beginPath();x.arc(px,yv,5,0,7);x.stroke()}
    MARK.push({x:px,y:yv,g,buy})}
  const last=P[P.length-1][1];document.getElementById('pxnote').textContent=COIN+' now '+pxf(last)+' · logged '+fHMS.format(new Date(t1*1000));
  GEO={L,R,W,t0,t1,H,T,B,X,Y,V,lo,hi};
}
document.getElementById('px').addEventListener('mousemove',e=>{const tip=document.getElementById('tip');if(!GEO){return}
  const r=e.target.getBoundingClientRect(),mx=e.clientX-r.left,my=e.clientY-r.top;
  let best=null,bd=14;for(const m of MARK){const d=Math.hypot(m.x-mx,m.y-my);if(d<bd){bd=d;best=m}}
  if(best){const g=best.g;tip.innerHTML=`<b>${best.buy?'Bought '+g.side:'Break '+g.side+', no buy'}</b> · ${fHMS.format(new Date(ts(g.time_utc)*1000))}<br>${g.secs_left}s left · index ${pxf(g.index)} vs strike ${pxf(g.strike)}<br>settlement chance ${g.model!=null?(g.model*100).toFixed(1)+'%':'–'} · ask ${g.ask!=null?cents(g.ask):'–'}<br>${best.buy?`${g.contracts} at ${cents(g.price)}`:(g.decision||'').replace('skip: ','')}`}
  else{const t=GEO.t0+(mx-GEO.L)/(GEO.W-GEO.L-GEO.R)*(GEO.t1-GEO.t0);if(t<GEO.t0||t>GEO.t1){tip.style.display='none';return}
    const p=GEO.V.find(p=>p[0]>=t);if(!p){tip.style.display='none';return}tip.innerHTML=`${fHMS.format(new Date(p[0]*1000))}<br><b>${pxf(p[1])}</b>`}
  tip.style.display='block';tip.style.left=(e.clientX+14)+'px';tip.style.top=(e.clientY+10)+'px'});
document.getElementById('px').addEventListener('mouseleave',()=>document.getElementById('tip').style.display='none');
document.querySelectorAll('[data-w]').forEach(b=>b.addEventListener('click',()=>{WIN=+b.dataset.w;document.querySelectorAll('[data-w]').forEach(o=>o.classList.toggle('on',o===b));if(D)drawPX()}));
function drawPL(){
  const c=document.getElementById('pl'),[x,W,H]=setup(c),L=64,R=14,T=10,B=26;
  const tr=(D.trades||[]).slice().sort((a,b)=>ts(a.time_utc)-ts(b.time_utc));
  x.font='12px system-ui';
  if(!tr.length){x.fillStyle=css('--mut');x.fillText('No settled trades yet.',L,40);return}
  const pts=[];let cum=0;tr.forEach(t=>{cum+=t.pnl||0;pts.push([ts(t.time_utc),cum])});
  const t0=pts[0][0]-600,t1=Math.max(D.now,pts[pts.length-1][0]+600);let lo=Math.min(0,...pts.map(p=>p[1])),hi=Math.max(0,...pts.map(p=>p[1]));
  if(hi-lo<5){hi+=2.5;lo-=2.5}const pad=(hi-lo)*0.08;lo-=pad;hi+=pad;
  const X=t=>L+(t-t0)/(t1-t0)*(W-L-R),Y=v=>T+(hi-v)/(hi-lo)*(H-T-B);
  x.strokeStyle=css('--grid');x.fillStyle=css('--mut');x.lineWidth=1;const st=niceStep((hi-lo)/5);
  for(let v=Math.ceil(lo/st)*st;v<=hi;v+=st){x.beginPath();x.moveTo(L,Y(v));x.lineTo(W-R,Y(v));x.stroke();x.textAlign='right';x.fillText(money(v).replace('.00',''),L-6,Y(v)+4)}
  x.strokeStyle=css('--grid2');x.beginPath();x.moveTo(L,Y(0));x.lineTo(W-R,Y(0));x.stroke();
  x.textAlign='center';const n=Math.max(2,Math.floor((W-L-R)/140));for(let i=0;i<=n;i++){const t=t0+(t1-t0)*i/n;x.fillText(fDT.format(new Date(t*1000)),X(t),H-8)}
  x.strokeStyle=css('--pl');x.lineWidth=2.5;x.beginPath();x.moveTo(X(t0),Y(0));let prev=0;pts.forEach(p=>{x.lineTo(X(p[0]),Y(prev));x.lineTo(X(p[0]),Y(p[1]));prev=p[1]});x.lineTo(X(t1),Y(prev));x.stroke();
}
function tables(){
  const P=D.perf||{},tr=(D.trades||[]).slice().sort((a,b)=>ts(a.time_utc)-ts(b.time_utc)),op=D.open||[];
  const tot=P.pnl!=null?P.pnl:0;const T=document.getElementById('tot');T.textContent=sgn(tot);T.className='big '+(tot>0?'won':tot<0?'lost':'');
  const n=P.trades||0;
  document.getElementById('sub').textContent=`${n} settled · ${op.length} open · ${(D.signals||[]).length} breaks seen`+(D.status&&D.status.updated_utc?` · bot updated ${fDT.format(new Date(D.status.updated_utc))}`:'');
  const f1=v=>v==null?'–':(v>=0?'+':'')+v.toFixed(1);
  document.getElementById('cards').innerHTML=[['Settled trades',n],['Won',n?P.wins+' ('+P.win_pct+'%)':'–'],['Avg price paid',n?P.avg_price+'¢':'–'],['Win % − price (pts)',n?f1(P.gap):'–'],['Break-even win %',n?P.break_even+'%':'–'],['EV per contract',n?f1(P.ev)+'¢':'–']]
    .map(([a,b])=>`<div class="card"><div class="mut">${a}</div><div class="big">${b}</div></div>`).join('');
  document.getElementById('perf').innerHTML=n?'<tr><th class="l">Bot</th><th>Trades</th><th>Won</th><th>Win %</th><th>Avg price</th><th>Win % − price</th><th>Break-even %</th><th>EV per contract</th><th>Total P/L</th></tr>'+
    `<tr><td class="l">${P.bot}</td><td>${P.trades}</td><td>${P.wins}</td><td>${P.win_pct}%</td><td>${P.avg_price}¢</td><td>${f1(P.gap)}</td><td>${P.break_even}%</td><td>${f1(P.ev)}¢</td><td class="${P.pnl>=0?'won':'lost'}">${sgn(P.pnl)}</td></tr>`:'<tr><td class="mut">No settled trades yet. The rule retires a coin if its EV is under +1¢ per contract at 300 trades.</td></tr>';
  document.getElementById('opn').innerHTML=op.length?'<tr><th class="l">Bought (ET)</th><th class="l">Market</th><th>Side</th><th>Ct</th><th>Price</th><th>Model</th><th>Secs left</th></tr>'+op.map(o=>`<tr><td class="l">${fDT.format(new Date(ts(o.time_utc)*1000))}</td><td class="l">${o.ticker}</td><td class="op">${o.side}</td><td>${o.contracts}</td><td>${cents(o.price)}</td><td>${cents(o.model)}</td><td>${o.secs_left}</td></tr>`).join(''):'<tr><td class="mut">No open positions right now.</td></tr>';
  const sg=(D.signals||[]).slice().reverse();
  document.getElementById('sig').innerHTML='<tr><th class="l">Break seen (ET)</th><th class="l">Market</th><th>Side</th><th>Secs left</th><th>Index</th><th>Strike</th><th>Chance</th><th>Ask</th><th class="l">Decision</th></tr>'+(sg.length?sg.map(g=>{const b=(g.decision||'').startsWith('BUY');return `<tr><td class="l">${fDT.format(new Date(ts(g.time_utc)*1000))}</td><td class="l">${g.ticker}</td><td>${g.side}</td><td>${g.secs_left}</td><td>${pxf(g.index)}</td><td>${pxf(g.strike)}</td><td>${g.model!=null?(g.model*100).toFixed(1)+'%':'–'}</td><td>${g.ask!=null?cents(g.ask):'–'}</td><td class="l ${b?'won':'mut'}">${g.decision}</td></tr>`}).join(''):'<tr><td class="mut" colspan="9">No breaks yet.</td></tr>');
  let cum=0,w=0;const out=tr.map((t,i)=>{cum+=t.pnl||0;w+=t.won?1:0;return {...t,cum,wp:100*w/(i+1)}}).reverse();
  document.getElementById('trd').innerHTML='<tr><th class="l">Bought (ET)</th><th class="l">Market</th><th>Side</th><th>Ct</th><th>Price</th><th>Model</th><th>Spent</th><th>Result</th><th>P/L</th><th>Cum P/L</th><th>Win % so far</th></tr>'+(out.length?out.map(t=>`<tr><td class="l">${fDT.format(new Date(ts(t.time_utc)*1000))}</td><td class="l">${t.ticker}</td><td>${t.side}</td><td>${t.contracts}</td><td>${cents(t.price)}</td><td>${cents(t.model)}</td><td>${money(t.cost||0)}</td><td class="${t.won?'won':'lost'}">${t.won?'Won':'Lost'}</td><td class="${t.pnl>=0?'won':'lost'}">${sgn(t.pnl||0)}</td><td>${sgn(t.cum)}</td><td>${t.wp.toFixed(1)}%</td></tr>`).join(''):'<tr><td class="mut" colspan="11">No settled trades yet.</td></tr>');
}
async function tick(){try{const r=await fetch('data.json',{cache:'no-store'});D=await r.json();document.getElementById('err').textContent='';tables();drawPX();drawPL()}catch(e){document.getElementById('err').textContent='connection lost, retrying…'}}
tick();setInterval(tick,10000);addEventListener('resize',()=>{if(D){drawPX();drawPL()}});
</script></body></html>"""
PAGEC = PAGEC.replace("/*FILTER_CSS*/", FILTER_CSS).replace("/*FILTER_JS*/", FILTER_JS + "\nfor (const id of ['perf','opn','sig','trd']) ftWatch(id);")
