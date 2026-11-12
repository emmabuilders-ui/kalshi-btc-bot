"""Page for the ML model-gap paper bots (Direct ML and Correction ML), served by the dashboard at
http://<server-ip>:8080/<token>/models/ . Read-only: it only reads the bots' log files."""

from __future__ import annotations

import time
from pathlib import Path

from .dashboard import FILTER_CSS, FILTER_JS, _json, _num, _rows, perf_row

# key in the dashboard's log map -> (name on the page, one-line rule)
MODEL_BOTS = {
    "model_gap": ("Direct ML", "NO 70–99¢, model ≥ 3 pts over ask + fee, 6–14 min left, cushion or breakout"),
    "corr_gap": ("Correction ML", "NO 70–99¢, model ≥ 3 pts over ask + fee, 6–14 min left, no other condition"),
}
CHECK_ROWS = 60


def _trade(r: dict, bot: str) -> dict:
    return {"bot": bot, "time_utc": r.get("time_utc"), "ticker": r.get("ticker"), "side": r.get("side"),
            "secs_left": _num(r.get("secs_left")), "model": _num(r.get("model_prob")), "gap": _num(r.get("gap_pts")),
            "contracts": _num(r.get("contracts")), "price": _num(r.get("avg_price")), "cost": _num(r.get("cost")),
            "won": str(r.get("won")) == "1", "pnl": _num(r.get("pnl")), "level": r.get("level")}


def models_data(logs: dict[str, Path]) -> dict:
    bots, trades, opened, checks, perf = [], [], [], [], []
    for key, (name, rule) in MODEL_BOTS.items():
        d = logs.get(key)
        if not d:
            continue
        tr = [_trade(r, name) for r in _rows(d / "trades.csv")]
        op = [_trade(r, name) for r in _json(d / "open_trades.json", [])]
        sig = _rows(d / "signals.csv")
        n = len(tr)
        bots.append({"key": key, "name": name, "rule": rule, "trades": n, "wins": sum(t["won"] for t in tr),
                     "avg_price": (sum(t["price"] or 0 for t in tr) / n) if n else None,
                     "pnl": sum(t["pnl"] or 0 for t in tr), "open": len(op), "checks": len(sig),
                     "looks": sum(1 for r in sig if str(r.get("side") or "")),
                     "updated_utc": _json(d / "status.json", {}).get("updated_utc")})
        perf.append(perf_row(name, _rows(d / "trades.csv")))
        trades += tr
        opened += op
        for r in sig[-CHECK_ROWS:]:
            checks.append({"bot": name, "time_utc": r.get("time_utc"), "min_left": _num(r.get("min_left")),
                           "p_yes": _num(r.get("model_p_yes")), "yes_ask": _num(r.get("yes_ask")), "no_ask": _num(r.get("no_ask")),
                           "yes_gap": _num(r.get("yes_gap_pts")), "no_gap": _num(r.get("no_gap_pts")),
                           "decision": r.get("decision")})
    return {"now": time.time(), "bots": bots, "trades": trades, "open": opened, "checks": checks, "perf": perf}


PAGEM = r"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ML Model Bots</title>
<style>
:root{--bg:#f9f9f7;--card:#fcfcfb;--fg:#0b0b0b;--mut:#52514e;--grid:#e6e6e3;--grid2:#c9c8c2;
--b0:#9c2f6e;--b1:#2a78d6;--all:#0b0b0b;--win:#0ca30c;--lose:#d03b3b;--open:#c98500}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#0d0d0d;--card:#1a1a19;--fg:#f0efec;--mut:#c3c2b7;--grid:#2a2a28;--grid2:#45443f;
--b0:#e88ac0;--b1:#3987e5;--all:#f0efec;--win:#2fbf2f;--lose:#e66767;--open:#eda100;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#0d0d0d;--card:#1a1a19;--fg:#f0efec;--mut:#c3c2b7;--grid:#2a2a28;--grid2:#45443f;
--b0:#e88ac0;--b1:#3987e5;--all:#f0efec;--win:#2fbf2f;--lose:#e66767;--open:#eda100;color-scheme:dark}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.4 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
.wrap{max-width:1100px;margin:0 auto;padding:16px}
h1{font-size:18px;margin:0}h2{font-size:15px;margin:0 0 8px}
.mut{color:var(--mut)}.top{display:flex;flex-wrap:wrap;gap:6px 16px;align-items:baseline;margin-bottom:12px}
.card{background:var(--card);border:1px solid var(--grid);border-radius:8px;padding:12px;margin-bottom:12px;min-width:0}
.tiles{display:grid;grid-template-columns:1fr;gap:12px}@media(min-width:700px){.tiles{grid-template-columns:1fr 1fr}}
.tile .big{font-size:22px;font-weight:700;font-variant-numeric:tabular-nums}
.tile .rule{font-size:12px;color:var(--mut);margin:2px 0 8px}
.kv{display:grid;grid-template-columns:auto 1fr;gap:2px 12px;font-variant-numeric:tabular-nums}.kv b{text-align:right}
canvas{display:block;width:100%}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
th,td{padding:4px 6px;border-bottom:1px solid var(--grid);text-align:right;white-space:nowrap;font-size:12px}
th:first-child,td:first-child,td.l,th.l{text-align:left}
th{color:var(--mut);font-weight:600;position:sticky;top:0;background:var(--card)}
.scroll{max-height:380px;overflow:auto}
.sw{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px;vertical-align:middle}
.won{color:var(--win);font-weight:600}.lost{color:var(--lose);font-weight:600}.op{color:var(--open);font-weight:600}
.legend{display:flex;flex-wrap:wrap;gap:4px 16px;margin:6px 0;color:var(--mut)}
.navbtn{padding:5px 12px;border:1px solid var(--grid2);border-radius:6px;background:var(--card);color:var(--fg);text-decoration:none;font-weight:600;font-size:13px}
#tip{position:fixed;pointer-events:none;background:var(--card);border:1px solid var(--grid2);border-radius:6px;padding:6px 8px;font-size:12px;display:none;z-index:5}
/*FILTER_CSS*/</style></head><body><div class="wrap">
<div class="top"><a class="navbtn" href="../">← Main chart</a><h1>ML model bots</h1>
<span class="mut">Paper only, no real orders · times ET · updates every 10 s</span><span id="err" class="lost"></span></div>
<div class="tiles" id="tiles"></div>
<div class="card" style="padding:8px"><h2 style="margin:4px 4px 6px">Bitcoin and the ML bots' trades</h2>
<iframe src="../?v=models" title="Bitcoin chart with the ML bots' trades" style="width:100%;height:600px;border:0;display:block"></iframe></div>
<div class="card"><h2>Results</h2><div class="scroll"><table id="perf"></table></div></div>
<div class="card"><h2>Profit, settled trades</h2><div class="legend" id="leg"></div><canvas id="pl" height="170"></canvas></div>
<div class="card"><h2>Open positions</h2><div class="scroll"><table id="opn"></table></div></div>
<div class="card"><h2>All trades</h2><div class="scroll"><table id="trd"></table></div></div>
<div class="card"><h2>Latest checks</h2><p class="mut" style="margin:0 0 6px;font-size:12px">Every minute each bot checks the model against both asks. Gap = model probability − ask − fee, in points.</p><div class="scroll"><table id="chk"></table></div></div>
</div><div id="tip"></div>
<script>
/*FILTER_JS*/
const TZ='America/New_York';
const fDT=new Intl.DateTimeFormat('en-US',{timeZone:TZ,month:'short',day:'numeric',hour:'numeric',minute:'2-digit',hour12:true});
const css=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const ts=s=>Date.parse(String(s).replace(' ','T').replace(/Z?$/,'Z'))/1000;
const sgn=x=>(x>0?'+':x<0?'−':'')+'$'+Math.abs(x||0).toFixed(2);
const c=v=>v==null?'–':Math.round(v*100)+'¢';
const pc=v=>v==null?'–':(v*100).toFixed(1)+'%';
const pt=v=>v==null||v===''?'–':(v>0?'+':'')+(+v).toFixed(1);
let D=null,COL={},geom=null,SER={};
function setup(cv){const r=cv.getBoundingClientRect(),d=window.devicePixelRatio||1;if(!cv.dataset.h)cv.dataset.h=cv.getAttribute('height');const h=+cv.dataset.h;cv.width=r.width*d;cv.height=h*d;cv.style.height=h+'px';const x=cv.getContext('2d');x.setTransform(d,0,0,d,0,0);return [x,r.width,h]}
function niceStep(r){const p=Math.pow(10,Math.floor(Math.log10(r||1)));for(const m of [1,2,2.5,5,10])if(r<=m*p)return m*p;return 10*p}
function tiles(){
  D.bots.forEach((b,i)=>COL[b.name]=css('--b'+i));
  document.getElementById('tiles').innerHTML=D.bots.map(b=>{const wp=b.trades?b.wins/b.trades:null,edge=wp!=null&&b.avg_price!=null?100*(wp-b.avg_price):null;
   return `<div class="card tile"><h2><span class="sw" style="background:${COL[b.name]}"></span>${b.name}</h2><div class="rule">${b.rule}</div>
   <div class="big ${b.pnl>0?'won':b.pnl<0?'lost':''}">${sgn(b.pnl)}</div>
   <div class="kv"><span>Settled trades</span><b>${b.trades}</b><span>Win % vs avg price</span><b>${pc(wp)} vs ${c(b.avg_price)}</b>
   <span>Win % − price</span><b class="${edge>0?'won':edge<0?'lost':''}">${edge==null?'–':pt(edge)+' pts'}</b><span>Open now</span><b>${b.open}</b>
   <span>Minute checks / looks</span><b>${b.checks} / ${b.looks}</b><span>Bot updated</span><b>${b.updated_utc?fDT.format(new Date(b.updated_utc)):'–'}</b></div></div>`}).join('')||'<div class="card mut">No ML bot logs found yet.</div>';
}
function drawPL(){
  const cv=document.getElementById('pl'),[x,W,H]=setup(cv);const L=56,R=12,T=10,B=26;
  const tr=D.trades.map(t=>({...t,x:ts(t.time_utc)})).sort((a,b)=>a.x-b.x);
  const names=D.bots.map(b=>b.name),cum={},S={};names.forEach(n=>{cum[n]=0;S[n]=[]});
  const t0=tr.length?tr[0].x-900:D.now-3600,t1=Math.max(D.now,t0+3600);names.forEach(n=>S[n].push([t0,0]));
  tr.forEach(t=>{cum[t.bot]+=t.pnl||0;S[t.bot].push([t.x,cum[t.bot]])});names.forEach(n=>S[n].push([t1,cum[n]]));
  let lo=0,hi=0;Object.values(S).forEach(s=>s.forEach(p=>{lo=Math.min(lo,p[1]);hi=Math.max(hi,p[1])}));if(hi-lo<5){hi+=2.5;lo-=2.5}const pad=(hi-lo)*.08;lo-=pad;hi+=pad;
  const X=t=>L+(t-t0)/(t1-t0)*(W-L-R),Y=v=>T+(hi-v)/(hi-lo)*(H-T-B);
  x.font='12px system-ui';x.fillStyle=css('--mut');x.strokeStyle=css('--grid');x.lineWidth=1;
  const st=niceStep((hi-lo)/5);for(let v=Math.ceil(lo/st)*st;v<=hi;v+=st){x.beginPath();x.moveTo(L,Y(v));x.lineTo(W-R,Y(v));x.stroke();x.textAlign='right';x.fillText(sgn(v).replace('.00',''),L-6,Y(v)+4)}
  x.textAlign='center';const n=Math.max(2,Math.floor((W-L-R)/140));for(let i=0;i<=n;i++){const t=t0+(t1-t0)*i/n;x.fillText(fDT.format(new Date(t*1000)),X(t),H-8)}
  names.forEach(nm=>{x.strokeStyle=COL[nm];x.lineWidth=2.5;x.beginPath();S[nm].forEach((p,i)=>{if(i)x.lineTo(X(p[0]),Y(S[nm][i-1][1]));x[i?'lineTo':'moveTo'](X(p[0]),Y(p[1]))});x.stroke()});
  SER=S;geom={t0,t1,L,R,W};
  document.getElementById('leg').innerHTML=names.map(n=>`<span><span class="sw" style="background:${COL[n]}"></span>${n}: ${sgn(cum[n])}</span>`).join('');
}
function valAt(s,t){let v=0;for(const p of s){if(p[0]<=t)v=p[1];else break}return v}
const tip=document.getElementById('tip'),pl=document.getElementById('pl');
pl.addEventListener('mousemove',e=>{if(!geom)return;const r=pl.getBoundingClientRect(),t=geom.t0+(e.clientX-r.left-geom.L)/(geom.W-geom.L-geom.R)*(geom.t1-geom.t0);
  if(t<geom.t0||t>geom.t1){tip.style.display='none';return}
  tip.innerHTML='<b>'+fDT.format(new Date(t*1000))+'</b><br>'+Object.keys(SER).map(n=>`<span class="sw" style="background:${COL[n]}"></span>${n}: ${sgn(valAt(SER[n],t))}`).join('<br>');
  tip.style.display='block';tip.style.left=(e.clientX+14)+'px';tip.style.top=(e.clientY+10)+'px'});
pl.addEventListener('mouseleave',()=>tip.style.display='none');
const bcell=n=>`<td class="l"><span class="sw" style="background:${COL[n]}"></span>${n}</td>`;
function tables(){
  const f1=v=>v==null?'–':(v>=0?'+':'')+v.toFixed(1);
  document.getElementById('perf').innerHTML='<tr><th class="l">Bot</th><th>Trades</th><th>Won</th><th>Win %</th><th>Avg price</th><th>Win % − price</th><th>Break-even %</th><th>EV per contract</th><th>Total P/L</th></tr>'+
   D.perf.map(P=>P.trades?`<tr>${bcell(P.bot)}<td>${P.trades}</td><td>${P.wins}</td><td>${P.win_pct}%</td><td>${P.avg_price}¢</td><td class="${P.gap>0?'won':'lost'}">${f1(P.gap)}</td><td>${P.break_even}%</td><td class="${P.ev>0?'won':'lost'}">${f1(P.ev)}¢</td><td class="${P.pnl>=0?'won':'lost'}">${sgn(P.pnl)}</td></tr>`:`<tr>${bcell(P.bot)}<td colspan="8" class="mut">No settled trades yet. Retire rule: under +1¢ per contract at 300 trades.</td></tr>`).join('');
  const op=D.open;document.getElementById('opn').innerHTML=op.length?'<tr><th class="l">Bot</th><th class="l">Bought</th><th>Side</th><th>Min left</th><th>Ct</th><th>Price</th><th>Model</th><th>Gap</th></tr>'+
   op.map(o=>`<tr>${bcell(o.bot)}<td class="l">${fDT.format(new Date(ts(o.time_utc)*1000))}</td><td class="op">${o.side}</td><td>${o.secs_left==null?'–':(o.secs_left/60).toFixed(1)}</td><td>${o.contracts}</td><td>${c(o.price)}</td><td>${pc(o.model)}</td><td>${pt(o.gap)}</td></tr>`).join(''):'<tr><td class="mut">No open positions right now.</td></tr>';
  const tr=D.trades.slice().sort((a,b)=>ts(b.time_utc)-ts(a.time_utc));
  document.getElementById('trd').innerHTML='<tr><th class="l">Bot</th><th class="l">Bought (ET)</th><th>Side</th><th>Min left</th><th>Price</th><th>Model</th><th>Gap</th><th>Result</th><th>P/L</th></tr>'+
   (tr.length?tr.map(t=>`<tr>${bcell(t.bot)}<td class="l">${fDT.format(new Date(ts(t.time_utc)*1000))}</td><td>${t.side}</td><td>${t.secs_left==null?'–':(t.secs_left/60).toFixed(1)}</td><td>${c(t.price)}</td><td>${pc(t.model)}</td><td>${pt(t.gap)}</td><td class="${t.won?'won':'lost'}">${t.won?'Won':'Lost'}</td><td class="${t.pnl>=0?'won':'lost'}">${sgn(t.pnl)}</td></tr>`).join(''):'<tr><td class="mut" colspan="9">No settled trades yet.</td></tr>');
  const ck=D.checks.slice().sort((a,b)=>ts(b.time_utc)-ts(a.time_utc));
  document.getElementById('chk').innerHTML='<tr><th class="l">Bot</th><th class="l">Time</th><th>Min left</th><th>Model YES</th><th>YES ask</th><th>YES gap</th><th>NO ask</th><th>NO gap</th><th class="l">Decision</th></tr>'+
   (ck.length?ck.map(r=>`<tr>${bcell(r.bot)}<td class="l">${fDT.format(new Date(ts(r.time_utc)*1000))}</td><td>${r.min_left??'–'}</td><td>${pc(r.p_yes)}</td><td>${c(r.yes_ask)}</td><td>${pt(r.yes_gap)}</td><td>${c(r.no_ask)}</td><td>${pt(r.no_gap)}</td><td class="l">${r.decision||''}</td></tr>`).join(''):'<tr><td class="mut" colspan="9">No checks yet.</td></tr>');
}
async function tick(){try{const r=await fetch('data.json',{cache:'no-store'});D=await r.json();document.getElementById('err').textContent='';tiles();drawPL();tables()}catch(e){document.getElementById('err').textContent='connection lost, retrying…'}}
tick();setInterval(tick,10000);addEventListener('resize',()=>{if(D)drawPL()});
</script></body></html>"""
PAGEM = PAGEM.replace("/*FILTER_CSS*/", FILTER_CSS).replace(
    "/*FILTER_JS*/", FILTER_JS + "\nfor (const id of ['perf','opn','trd','chk']) ftWatch(id);")
