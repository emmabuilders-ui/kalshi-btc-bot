"""Chart page for the 5-account hourly range paper bot (kalshi-range5), served by the dashboard
at http://<server-ip>:8080/<token>/range5/ . Read-only: it only reads the bot's log files."""

from __future__ import annotations

import time
from pathlib import Path

from .dashboard import FILTER_CSS, FILTER_JS, _json, _num, _rows


def range5_data(d: Path, prices: list) -> dict:
    status = _json(d / "status.json", {})
    sig = _rows(d / "signals.csv")
    edges = {r["ticker"]: (_num(r.get("floor")), _num(r.get("cap"))) for r in sig}
    held = []
    for p in _json(d / "open_positions.json", []):
        lo, hi = edges.get(p["ticker"], (None, None))
        held.append({**p, "lo": lo, "hi": hi})
    trades = []
    for r in _rows(d / "trades.csv"):
        lo, hi = edges.get(r.get("ticker"), (None, None))
        trades.append({"time_utc": r.get("time_utc"), "account": int(r.get("account") or 0), "ticker": r.get("ticker"),
                       "side": r.get("side"), "contracts": _num(r.get("contracts")), "price": _num(r.get("avg_price")),
                       "cost": _num(r.get("cost")), "model": _num(r.get("model")), "won": str(r.get("won")) == "1",
                       "pnl": _num(r.get("pnl")), "mins_left": _num(r.get("mins_left")), "lo": lo, "hi": hi})
    misses = sum(1 for r in sig if str(r.get("filled")) in ("0", ""))
    return {"now": time.time(), "prices": prices, "status": status, "open": held, "trades": trades,
            "signals": len(sig), "unfilled": misses}


PAGE5 = r"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Range 5-Account Chart</title>
<style>
:root{--bg:#f9f9f7;--card:#fcfcfb;--fg:#0b0b0b;--mut:#52514e;--grid:#e6e6e3;--grid2:#c9c8c2;
--a1:#2a78d6;--a2:#eb6834;--a3:#1baf7a;--a4:#c98500;--a5:#6250d6;--all:#0b0b0b;--win:#0ca30c;--lose:#d03b3b;--open:#fab219;--price:#52514e}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#0d0d0d;--card:#1a1a19;--fg:#f0efec;--mut:#c3c2b7;--grid:#2a2a28;--grid2:#45443f;
--a1:#3987e5;--a2:#d95926;--a3:#199e70;--a4:#eda100;--a5:#9085e9;--all:#f0efec;--win:#2fbf2f;--lose:#e66767;--open:#fab219;--price:#c3c2b7;color-scheme:dark}}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.4 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
.wrap{max-width:1300px;margin:0 auto;padding:16px}
h1{font-size:18px;margin:0}h2{font-size:15px;margin:0 0 8px}
.mut{color:var(--mut)}.top{display:flex;flex-wrap:wrap;gap:6px 20px;align-items:baseline;margin-bottom:12px}
.big{font-size:24px;font-weight:700;font-variant-numeric:tabular-nums}
.card{background:var(--card);border:1px solid var(--grid);border-radius:8px;padding:12px;margin-bottom:12px}
canvas{display:block;width:100%}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
th,td{padding:5px 8px;border-bottom:1px solid var(--grid);text-align:right;white-space:nowrap}
th:first-child,td:first-child,td.l,th.l{text-align:left}
th{color:var(--mut);font-weight:600;position:sticky;top:0;background:var(--card)}
.scroll{max-height:420px;overflow:auto}
.sw{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px;vertical-align:middle}
.won{color:var(--win);font-weight:600}.lost{color:var(--lose);font-weight:600}.op{color:var(--open);font-weight:600}
.legend{display:flex;flex-wrap:wrap;gap:4px 16px;margin:6px 0;color:var(--mut)}
#tip{position:fixed;pointer-events:none;background:var(--card);border:1px solid var(--grid2);border-radius:6px;padding:6px 8px;font-size:12px;display:none;z-index:5}
.grid2{display:grid;grid-template-columns:1fr;gap:12px}
@media(min-width:1000px){.grid2{grid-template-columns:1fr 1fr}}
.navbtn{padding:5px 12px;border:1px solid var(--grid2);border-radius:6px;background:var(--card);color:var(--fg);text-decoration:none;font-weight:600;font-size:13px}.navbtn:hover{border-color:var(--a1)}
.grid2>*{min-width:0}.card{min-width:0}
/*FILTER_CSS*/</style></head><body><div class="wrap">
<div class="top"><a class="navbtn" href="../">← Main chart (3 bots)</a><h1>Hourly range bot · 5 paper accounts</h1>
<span class="mut">Paper only, no real orders · times ET · updates every 10 s</span></div>
<div class="top"><span>Total profit <span class="big" id="tot">–</span></span>
<span id="sub" class="mut"></span><span id="err" class="lost"></span></div>
<div class="card" style="padding:8px"><h2 style="margin:4px 4px 6px">Bitcoin and this bot's trades</h2>
<iframe src="../?v=range5" title="Bitcoin chart" style="width:100%;height:790px;border:0;display:block"></iframe></div>
<div class="card"><h2>Profit by account (settled trades)</h2><div class="legend" id="leg"></div>
<canvas id="pl" height="160"></canvas></div>
<div class="card"><h2>Accounts</h2><div class="scroll"><table id="acc"></table></div></div>
<div class="card"><h2>Open positions</h2><div class="scroll"><table id="opn"></table></div></div>
<div class="card"><h2>All trades</h2><div class="scroll"><table id="trd"></table></div></div>
</div><div id="tip"></div>
<script>
/*FILTER_JS*/
const TZ='America/New_York';
const fDT=new Intl.DateTimeFormat('en-US',{timeZone:TZ,month:'short',day:'numeric',hour:'numeric',minute:'2-digit',hour12:true});
const fHM=new Intl.DateTimeFormat('en-US',{timeZone:TZ,hour:'numeric',minute:'2-digit',hour12:true});
const NAMES={1:'YES 90–99¢ rebuilt +2¢',2:'NO current bot rule',3:'YES 75–89¢ rebuilt ≥ ask',4:'NO 90–99¢ rebuilt +2¢',5:'YES 90–99¢ doubling'};
const css=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const col=a=>css(a==='all'?'--all':'--a'+a);
const ts=s=>Date.parse(s.replace(' ','T')+'Z')/1000;
const money=x=>(x<0?'−$':'$')+Math.abs(x).toFixed(2);
const sgn=x=>(x>0?'+':x<0?'−':'')+'$'+Math.abs(x).toFixed(2);
function band(lo,hi){return lo==null?('below $'+Math.round(hi).toLocaleString()):hi==null?('above $'+Math.round(lo).toLocaleString()):('$'+Math.round(lo).toLocaleString()+'–'+Math.round(hi+0.01).toLocaleString())}
function setup(c){const r=c.getBoundingClientRect(),d=window.devicePixelRatio||1;if(!c.dataset.h)c.dataset.h=c.getAttribute('height');const h=+c.dataset.h;c.width=r.width*d;c.height=h*d;c.style.height=h+'px';const x=c.getContext('2d');x.setTransform(d,0,0,d,0,0);return [x,r.width,h]}
let D=null, plSeries=[], plGeom=null;
function drawPL(){
  const c=document.getElementById('pl'),[x,W,H]=setup(c);const L=56,R=16,T=10,B=26;
  const tr=(D.trades||[]).map(t=>({...t,x:Math.ceil(ts(t.time_utc)/3600)*3600})).sort((a,b)=>a.x-b.x);
  const acc=[1,2,3,4,5],cum={},S={all:[]};acc.forEach(a=>{cum[a]=0;S[a]=[]});let all=0;
  const t0=tr.length?tr[0].x-3600:D.now-3600,t1=Math.max(D.now,t0+3600);
  acc.forEach(a=>S[a].push([t0,0]));S.all.push([t0,0]);
  tr.forEach(t=>{cum[t.account]+=t.pnl;all+=t.pnl;S[t.account].push([t.x,cum[t.account]]);S.all.push([t.x,all])});
  acc.forEach(a=>S[a].push([t1,cum[a]]));S.all.push([t1,all]);
  let lo=0,hi=0;Object.values(S).forEach(s=>s.forEach(p=>{lo=Math.min(lo,p[1]);hi=Math.max(hi,p[1])}));
  if(hi-lo<5){hi+=2.5;lo-=2.5}const pad=(hi-lo)*0.08;lo-=pad;hi+=pad;
  const X=t=>L+(t-t0)/(t1-t0)*(W-L-R),Y=v=>T+(hi-v)/(hi-lo)*(H-T-B);
  x.font='12px system-ui';x.fillStyle=css('--mut');x.strokeStyle=css('--grid');x.lineWidth=1;
  const step=niceStep((hi-lo)/5);for(let v=Math.ceil(lo/step)*step;v<=hi;v+=step){x.beginPath();x.moveTo(L,Y(v));x.lineTo(W-R,Y(v));x.stroke();x.textAlign='right';x.fillText(money(v).replace('.00',''),L-6,Y(v)+4)}
  x.strokeStyle=css('--grid2');x.beginPath();x.moveTo(L,Y(0));x.lineTo(W-R,Y(0));x.stroke();
  x.textAlign='center';const n=Math.max(2,Math.floor((W-L-R)/140));for(let i=0;i<=n;i++){const t=t0+(t1-t0)*i/n;x.fillText(fDT.format(new Date(t*1000)),X(t),H-8)}
  const draw=(s,c,w)=>{x.strokeStyle=c;x.lineWidth=w;x.beginPath();s.forEach((p,i)=>{if(i){x.lineTo(X(p[0]),Y(s[i-1][1]))}x[i?'lineTo':'moveTo'](X(p[0]),Y(p[1]))});x.stroke()};
  acc.forEach(a=>draw(S[a],col(a),2));draw(S.all,col('all'),3);
  plSeries=S;plGeom={X,Y,t0,t1,L,R,W};
  document.getElementById('leg').innerHTML=acc.map(a=>`<span><span class="sw" style="background:${col(a)}"></span>${a}. ${NAMES[a]}: ${sgn(cum[a])}</span>`).join('')+`<span><span class="sw" style="background:${col('all')}"></span><b>All five: ${sgn(all)}</b></span>`;
}
function niceStep(r){const p=Math.pow(10,Math.floor(Math.log10(r||1)));for(const m of [1,2,2.5,5,10])if(r<=m*p)return m*p;return 10*p}
function valAt(s,t){let v=0;for(const p of s){if(p[0]<=t)v=p[1];else break}return v}
document.getElementById('pl').addEventListener('mousemove',e=>{if(!plGeom)return;const r=e.target.getBoundingClientRect();const px=e.clientX-r.left;
  const t=plGeom.t0+(px-plGeom.L)/(plGeom.W-plGeom.L-plGeom.R)*(plGeom.t1-plGeom.t0);if(t<plGeom.t0||t>plGeom.t1){tip.style.display='none';return}
  tip.innerHTML='<b>'+fDT.format(new Date(t*1000))+'</b><br>'+[1,2,3,4,5].map(a=>`<span class="sw" style="background:${col(a)}"></span>Acct ${a}: ${sgn(valAt(plSeries[a],t))}`).join('<br>')+`<br><b>All: ${sgn(valAt(plSeries.all,t))}</b>`;
  tip.style.display='block';tip.style.left=(e.clientX+14)+'px';tip.style.top=(e.clientY+10)+'px'});
document.getElementById('pl').addEventListener('mouseleave',()=>tip.style.display='none');
function drawPX(){
  if(!document.getElementById('px'))return;  // the full chart now sits at the top of the page
  const c=document.getElementById('px'),[x,W,H]=setup(c);const L=64,R=10,T=10,B=24;const P=D.prices||[];
  if(!P.length){x.fillStyle=css('--mut');x.fillText('waiting for BRTI…',L,40);return}
  const t0=P[0][0],t1=Math.max(P[P.length-1][0],t0+60);let lo=Infinity,hi=-Infinity;P.forEach(p=>{lo=Math.min(lo,p[1]);hi=Math.max(hi,p[1])});
  const op=D.open||[];op.forEach(o=>{if(o.lo!=null)lo=Math.min(lo,o.lo);if(o.hi!=null)hi=Math.max(hi,o.hi)});
  const pad=Math.max(20,(hi-lo)*0.06);lo-=pad;hi+=pad;const X=t=>L+(t-t0)/(t1-t0)*(W-L-R),Y=v=>T+(hi-v)/(hi-lo)*(H-T-B);
  x.font='12px system-ui';x.strokeStyle=css('--grid');x.fillStyle=css('--mut');x.lineWidth=1;
  const st=niceStep((hi-lo)/5);for(let v=Math.ceil(lo/st)*st;v<=hi;v+=st){x.beginPath();x.moveTo(L,Y(v));x.lineTo(W-R,Y(v));x.stroke();x.textAlign='right';x.fillText('$'+Math.round(v).toLocaleString(),L-6,Y(v)+4)}
  x.textAlign='center';for(let i=0;i<=4;i++){const t=t0+(t1-t0)*i/4;x.fillText(fHM.format(new Date(t*1000)),X(t),H-6)}
  op.forEach(o=>{const yl=Y(o.lo==null?lo:o.lo),yh=Y(o.hi==null?hi:o.hi+0.01);x.globalAlpha=0.16;x.fillStyle=col(o.account);x.fillRect(L,yh,W-L-R,yl-yh);x.globalAlpha=1;
    x.strokeStyle=col(o.account);x.setLineDash([4,3]);x.strokeRect(L,yh,W-L-R,yl-yh);x.setLineDash([]);x.fillStyle=col(o.account);x.textAlign='left';x.fillText(`Acct ${o.account} ${o.side} ${band(o.lo,o.hi)}`,L+4,Math.max(yh,T)+13)});
  x.strokeStyle=css('--price');x.lineWidth=2;x.beginPath();P.forEach((p,i)=>x[i?'lineTo':'moveTo'](X(p[0]),Y(p[1])));x.stroke();
  op.forEach(o=>{const t=ts(o.time_utc);if(t<t0)return;const py=P.find(p=>p[0]>=t);if(!py)return;x.fillStyle=css('--open');x.strokeStyle=css('--card');x.lineWidth=2;x.beginPath();x.arc(X(t),Y(py[1]),5,0,7);x.fill();x.stroke()});
  const last=P[P.length-1][1];document.getElementById('pxnote').textContent='BRTI now $'+last.toLocaleString(undefined,{maximumFractionDigits:2})+' · shaded = bands this bot holds · yellow dots = entries';
}
function tables(){
  const st=(D.status&&D.status.accounts)||{};const rows=Object.entries(st);
  document.getElementById('acc').innerHTML='<tr><th class="l">Account</th><th>Size</th><th>Trades</th><th>Won</th><th>Win %</th><th>P/L</th><th>Open</th></tr>'+
   rows.map(([k,v])=>{const a=+k.split(' ')[0];const wp=v.trades?(100*v.wins/v.trades).toFixed(1)+'%':'–';
    const size=typeof v.size==='number'?v.size+' ct':(v.size+(v.next_stake_dollars!=null?` · next $${(+v.next_stake_dollars).toFixed(2)}`:''));
    return `<tr><td class="l"><span class="sw" style="background:${col(a)}"></span>${a}. ${NAMES[a]||k}</td><td>${size}</td><td>${v.trades}</td><td>${v.wins}</td><td>${wp}</td><td class="${v.pnl_dollars>0?'won':v.pnl_dollars<0?'lost':''}">${sgn(v.pnl_dollars)}</td><td>${v.open}</td></tr>`}).join('');
  const op=D.open||[];document.getElementById('opn').innerHTML=op.length?'<tr><th class="l">Bought</th><th class="l">Acct</th><th class="l">Band</th><th>Side</th><th>Ct</th><th>Price</th><th>Model</th></tr>'+
   op.map(o=>`<tr><td class="l">${fDT.format(new Date(ts(o.time_utc)*1000))}</td><td class="l"><span class="sw" style="background:${col(o.account)}"></span>${o.account}</td><td class="l">${band(o.lo,o.hi)}</td><td class="op">${o.side}</td><td>${o.contracts}</td><td>${Math.round(o.avg_price*100)}¢</td><td>${(o.model*100).toFixed(1)}¢</td></tr>`).join(''):'<tr><td class="mut">No open positions right now.</td></tr>';
  const tr=(D.trades||[]).slice().sort((a,b)=>ts(a.time_utc)-ts(b.time_utc));let cum=0,w=0;const out=tr.map((t,i)=>{cum+=t.pnl;w+=t.won?1:0;return {...t,cum,wp:100*w/(i+1)}}).reverse();
  document.getElementById('trd').innerHTML='<tr><th class="l">Bought (ET)</th><th class="l">Acct</th><th class="l">Band</th><th>Side</th><th>Contracts</th><th>Price</th><th>Model</th><th>Spent</th><th>Result</th><th>P/L</th><th>Cum P/L</th><th>Win % so far</th></tr>'+
   (out.length?out.map(t=>`<tr><td class="l">${fDT.format(new Date(ts(t.time_utc)*1000))}</td><td class="l"><span class="sw" style="background:${col(t.account)}"></span>${t.account}</td><td class="l">${band(t.lo,t.hi)}</td><td>${t.side}</td><td>${t.contracts}</td><td>${Math.round(t.price*100)}¢</td><td>${(t.model*100).toFixed(1)}¢</td><td>${money(t.cost)}</td><td class="${t.won?'won':'lost'}">${t.won?'Won':'Lost'}</td><td class="${t.pnl>=0?'won':'lost'}">${sgn(t.pnl)}</td><td>${sgn(t.cum)}</td><td>${t.wp.toFixed(1)}%</td></tr>`).join(''):'<tr><td class="mut" colspan="12">No settled trades yet.</td></tr>');
  const tot=(D.status&&D.status.total_pnl_dollars)||0;const T=document.getElementById('tot');T.textContent=sgn(tot);T.className='big '+(tot>0?'won':tot<0?'lost':'');
  const n=tr.length,wn=tr.filter(t=>t.won).length;
  document.getElementById('sub').textContent=`${n} settled trades · ${n?(100*wn/n).toFixed(1):'–'}% won · ${op.length} open · ${D.signals} signals, ${D.unfilled} not filled`+(D.status&&D.status.updated_utc?` · bot updated ${fDT.format(new Date(D.status.updated_utc))}`:'');
}
const tip=document.getElementById('tip');
async function tick(){try{const r=await fetch('data.json',{cache:'no-store'});D=await r.json();document.getElementById('err').textContent='';tables();drawPL();drawPX()}catch(e){document.getElementById('err').textContent='connection lost, retrying…'}}
tick();setInterval(tick,10000);addEventListener('resize',()=>{if(D){drawPL();drawPX()}});
</script></body></html>"""
PAGE5 = PAGE5.replace("/*FILTER_CSS*/", FILTER_CSS).replace("/*FILTER_JS*/", FILTER_JS + "\nfor (const id of ['acc','opn','trd']) ftWatch(id);")

