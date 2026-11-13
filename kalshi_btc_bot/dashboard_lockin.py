"""Page for the last-minute lock-in paper bot (kalshi_btc_bot/lockin_paper.py), served by the dashboard at
http://<server-ip>:8080/<token>/lockin/ . Read-only: it only reads the bot's log files.

Stats shown the way Michael reads them: win % always next to the average price paid, win % minus price,
break-even (price plus fee), EV per contract after fees, and the same split by seconds left and by side."""

from __future__ import annotations

import time
from pathlib import Path

from .dashboard import _json, _num, _rows

RULE = ("15-min BTC market, 50 / 40 / 30 / 20 / 10 s before the close: buy the side the settlement math favours when its chance "
        "(locked-in BRTI prints + the remaining variance) beats the ask + fee by 5 pts; first qualifying check, "
        "one trade per contract, 20 contracts, held to settlement; spread over 2c skipped")
CHECK_ROWS = 80
RETIRE_AT, RETIRE_EDGE = 300, 1.0      # docs/paper_test_decision_rule.md: under +1c at 300 trades retires the bot


def _trade(r: dict) -> dict:
    return {"time_utc": r.get("time_utc"), "ticker": r.get("ticker"), "side": r.get("side"),
            "secs_left": _num(r.get("secs_left")), "model": _num(r.get("model_prob")), "gap": _num(r.get("gap_pts")),
            "brti": _num(r.get("brti")), "locked_avg": _num(r.get("locked_avg")), "strike": _num(r.get("strike")),
            "contracts": _num(r.get("contracts")), "price": _num(r.get("avg_price")), "fee": _num(r.get("fee")),
            "cost": _num(r.get("cost")), "won": str(r.get("won")) == "1", "pnl": _num(r.get("pnl")),
            "level": r.get("level"), "settled": r.get("result") not in (None, "")}


def stats(tr: list[dict], worse: float = 0.0) -> dict:
    """Settled trades: win %, avg price, win % minus price, break-even, EV per contract with its standard error;
    `worse` adds that many dollars to every fill (the decision rule asks for the edge at fills 1c worse)."""
    evs, wins, price, be, pnl = [], 0, 0.0, 0.0, 0.0
    for t in tr:
        c, p = t["contracts"], t["price"]
        if not c or p is None or t["pnl"] is None:
            continue
        p2 = min(p + worse, 0.99)
        fee = 0.07 * p2 * (1 - p2)
        wins += t["won"]
        price += p2
        be += p2 + fee
        pnl += t["pnl"]
        evs.append((1.0 if t["won"] else 0.0) - p2 - fee)
    n = len(evs)
    if not n:
        return {"trades": 0}
    ev = sum(evs) / n
    sd = (sum((e - ev) ** 2 for e in evs) / (n - 1)) ** 0.5 if n > 1 else None
    return {"trades": n, "wins": wins, "win_pct": 100 * wins / n, "avg_price": 100 * price / n,
            "gap": 100 * (wins / n - price / n), "break_even": 100 * be / n, "ev": 100 * ev, "pnl": pnl,
            "se": 100 * sd / n ** 0.5 if sd is not None else None}


def lockin_data(logs: dict[str, Path]) -> dict:
    d = logs.get("lockin")
    if not d:
        return {"now": time.time(), "found": False}
    tr = [_trade(r) for r in _rows(d / "trades.csv")]
    tr = [t for t in tr if t["settled"]]
    op = [_trade(r) for r in _json(d / "open_trades.json", [])]
    sig = _rows(d / "signals.csv")
    skips = _rows(d / "skips.csv")
    by_sec = {}
    for s in (10, 20, 30, 40, 50):
        by_sec[s] = stats([t for t in tr if t["secs_left"] == s])
    by_side = {sd: stats([t for t in tr if t["side"] == sd]) for sd in ("YES", "NO")}
    checks = []
    for r in sig[-CHECK_ROWS:]:
        checks.append({"time_utc": r.get("time_utc"), "secs_left": _num(r.get("secs_left")), "brti": _num(r.get("brti")),
                       "strike": _num(r.get("strike")), "locked_avg": _num(r.get("locked_avg")),
                       "p_yes": _num(r.get("p_yes_pct")), "yes_ask": _num(r.get("yes_ask")), "no_ask": _num(r.get("no_ask")),
                       "yes_edge": _num(r.get("yes_edge_pts")), "no_edge": _num(r.get("no_edge_pts")),
                       "lag": _num(r.get("lag_s")), "decision": r.get("decision")})
    status = _json(d / "status.json", {})
    return {"now": time.time(), "found": True, "rule": RULE, "all": stats(tr), "worse1c": stats(tr, 0.01),
            "by_sec": by_sec, "by_side": by_side, "trades": tr, "open": op, "checks": checks,
            "n_checks": len(sig), "n_skips_spread": len(skips),
            "n_skips_late": sum(1 for r in sig if "late" in str(r.get("decision", "")) or "proxy" in str(r.get("decision", ""))),
            "retire_at": RETIRE_AT, "retire_edge": RETIRE_EDGE, "updated_utc": status.get("updated_utc")}


PAGEL = r"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Lock-in Bot</title>
<style>
:root{--bg:#f9f9f7;--card:#fcfcfb;--fg:#0b0b0b;--mut:#52514e;--grid:#e6e6e3;--grid2:#c9c8c2;
--lk:#c2185b;--win:#0ca30c;--lose:#d03b3b;--open:#c98500}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#0d0d0d;--card:#1a1a19;--fg:#f0efec;--mut:#c3c2b7;--grid:#2a2a28;--grid2:#45443f;
--lk:#f06292;--win:#2fbf2f;--lose:#e66767;--open:#eda100;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#0d0d0d;--card:#1a1a19;--fg:#f0efec;--mut:#c3c2b7;--grid:#2a2a28;--grid2:#45443f;
--lk:#f06292;--win:#2fbf2f;--lose:#e66767;--open:#eda100;color-scheme:dark}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.4 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
.wrap{max-width:1100px;margin:0 auto;padding:16px}
h1{font-size:18px;margin:0}h2{font-size:15px;margin:0 0 8px}
.mut{color:var(--mut)}.top{display:flex;flex-wrap:wrap;gap:6px 16px;align-items:baseline;margin-bottom:12px}
.card{background:var(--card);border:1px solid var(--grid);border-radius:8px;padding:12px;margin-bottom:12px;min-width:0}
.tiles{display:grid;grid-template-columns:1fr 1fr;gap:12px}@media(min-width:700px){.tiles{grid-template-columns:repeat(4,1fr)}}
.tile .big{font-size:22px;font-weight:700;font-variant-numeric:tabular-nums}.tile .lbl{font-size:12px;color:var(--mut)}
.tile .sub{font-size:12px;color:var(--mut);margin-top:2px}
.rule{font-size:13px;color:var(--mut);margin:4px 0 10px}
canvas{display:block;width:100%}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
th,td{padding:4px 6px;border-bottom:1px solid var(--grid);text-align:right;white-space:nowrap;font-size:12px}
th:first-child,td:first-child,td.l,th.l{text-align:left}
th{color:var(--mut);font-weight:600;position:sticky;top:0;background:var(--card)}
.scroll{max-height:380px;overflow:auto}
.won{color:var(--win);font-weight:600}.lost{color:var(--lose);font-weight:600}.op{color:var(--open);font-weight:600}
.navbtn{padding:5px 12px;border:1px solid var(--grid2);border-radius:6px;background:var(--card);color:var(--fg);text-decoration:none;font-weight:600;font-size:13px}
.bar{height:8px;background:var(--grid);border-radius:4px;overflow:hidden;margin-top:6px}.bar i{display:block;height:100%;background:var(--lk)}
#tip{position:fixed;pointer-events:none;background:var(--card);border:1px solid var(--grid2);border-radius:6px;padding:6px 8px;font-size:12px;display:none;z-index:5}
</style></head><body><div class="wrap">
<div class="top"><a class="navbtn" href="../">← Main chart</a><h1>Lock-in bot (15-min, last 60 seconds)</h1>
<span class="mut">Paper only, no real orders · times ET · updates every 10 s</span><span id="err" class="lost"></span></div>
<div class="card"><div class="rule" id="rule"></div>
<div class="tiles" id="tiles"></div>
<div class="mut" style="font-size:12px;margin-top:10px" id="foot"></div></div>
<div class="card"><h2>By seconds left and by side</h2><p class="mut" style="margin:0 0 6px;font-size:12px">Settled trades. Win % is read next to the average price paid; win % − price is the edge before fees, EV is profit per contract after fees. Backtest for comparison (trade prints, Dec–Aug / Sep–Oct): first fire at 10 s +20.0 / +24.9c (39 / 9 trades) · 20 s +17.6 / +4.0c (61 / 12) · 30 s +17.2 / +6.6c (89 / 22) · 40 s +10.9 / +5.8c (121 / 41) · 50 s +18.8 / +6.6c (188 / 76); YES +15.3 / +6.9c (456 / 153), NO +30.1 / +13.8c (42 / 7).</p>
<table id="split"></table></div>
<div class="card" style="padding:8px"><h2 style="margin:4px 4px 6px">Bitcoin and the lock-in bot's trades</h2>
<iframe src="../?v=lockin" title="Bitcoin chart with the lock-in bot's trades" style="width:100%;height:600px;border:0;display:block"></iframe></div>
<div class="card"><h2>Profit, settled trades</h2><canvas id="pl" height="170"></canvas></div>
<div class="card"><h2>Open positions</h2><div class="scroll"><table id="opn"></table></div></div>
<div class="card"><h2>All trades</h2><div class="scroll"><table id="trd"></table></div></div>
<div class="card"><h2>Latest checks</h2><p class="mut" style="margin:0 0 6px;font-size:12px">Each row is one check at 50, 40, 30, 20 or 10 s left. Locked avg = mean of the BRTI prints already inside the 60-second settlement average. Edge = settlement-aware chance − ask − fee, in points; 5 or more buys. Lag = seconds between the check second and the decision.</p><div class="scroll"><table id="chk"></table></div></div>
</div><div id="tip"></div>
<script>
const TZ='America/New_York';
const fDT=new Intl.DateTimeFormat('en-US',{timeZone:TZ,month:'short',day:'numeric',hour:'numeric',minute:'2-digit',second:'2-digit',hour12:true});
const css=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const ts=s=>Date.parse(String(s).replace(' ','T').replace(/Z?$/,'Z'))/1000;
const sgn=x=>(x>0?'+':x<0?'−':'')+'$'+Math.abs(x||0).toFixed(2);
const c=v=>v==null?'–':Math.round(v*100)+'¢';
const c1=v=>v==null?'–':v.toFixed(1)+'¢';
const pc=v=>v==null?'–':(+v).toFixed(1)+'%';
const pt=v=>v==null||v===''?'–':(v>0?'+':'')+(+v).toFixed(1);
const usd=x=>x==null?'–':'$'+x.toLocaleString('en-US',{maximumFractionDigits:2});
let D=null,geom=null,SER=[];
function setup(cv){const r=cv.getBoundingClientRect(),d=window.devicePixelRatio||1;if(!cv.dataset.h)cv.dataset.h=cv.getAttribute('height');const h=+cv.dataset.h;cv.width=r.width*d;cv.height=h*d;cv.style.height=h+'px';const x=cv.getContext('2d');x.setTransform(d,0,0,d,0,0);return [x,r.width,h]}
function niceStep(r){const p=Math.pow(10,Math.floor(Math.log10(r||1)));for(const m of [1,2,2.5,5,10])if(r<=m*p)return m*p;return 10*p}
const cls=v=>v>0?'won':v<0?'lost':'';
function tiles(){
  document.getElementById('rule').textContent=D.rule;
  const a=D.all,w=D.worse1c,n=a.trades||0;
  const tile=(l,v,s,k)=>`<div class="tile"><div class="lbl">${l}</div><div class="big ${k||''}">${v}</div>${s?`<div class="sub">${s}</div>`:''}</div>`;
  const pct=Math.min(100,100*n/D.retire_at);
  document.getElementById('tiles').innerHTML=
   tile('Settled trades',n,`${D.open.length} open · check at ${D.retire_at} trades<div class="bar"><i style="width:${pct}%"></i></div>`)+
   tile('Win % vs avg price',n?`${pc(a.win_pct)} vs ${c1(a.avg_price)}`:'–','win % next to the price paid')+
   tile('Win % − price',n?pt(a.gap)+' pts':'–',n?'break-even '+pc(a.break_even):'',n?cls(a.gap):'')+
   tile('EV per contract',n?pt(a.ev)+'¢':'–',n?`after fees · ${a.se!=null?'± '+a.se.toFixed(1)+'¢ SE':''} · with fills 1¢ worse ${pt(w.ev)}¢`:'',n?cls(a.ev):'')+
   tile('Total P/L',sgn(a.pnl||0),'settled, after fees',cls(a.pnl||0))+
   tile('Retire rule',n?(n>=D.retire_at?(a.ev<D.retire_edge?'RETIRE':'passes'):`${D.retire_at-n} to go`):'–',`under +${D.retire_edge.toFixed(0)}¢ EV at ${D.retire_at} trades retires it`,n>=D.retire_at?(a.ev<D.retire_edge?'lost':'won'):'')+
   tile('Checks / spread skips',`${D.n_checks} / ${D.n_skips_spread}`,`${D.n_skips_late} late or proxy-price checks skipped`)+
   tile('Bot updated',D.updated_utc?fDT.format(new Date(D.updated_utc)):'–','last status write');
  document.getElementById('foot').textContent='Backtest (trade prints, not live asks): Dec–Aug 498 trades, 80.5% vs 62.8¢, +16.5¢ per contract, 9 of 9 months; Sep–Oct 160 trades, 76.2% vs 67.8¢, +7.2¢, 2 of 2. The grid and margin were picked on this data, so only these paper fills count.';
}
function split(){
  const row=(name,s)=>s.trades?`<tr><td class="l">${name}</td><td>${s.trades}</td><td>${s.wins}</td><td>${pc(s.win_pct)}</td><td>${c1(s.avg_price)}</td><td class="${cls(s.gap)}">${pt(s.gap)} pts</td><td>${pc(s.break_even)}</td><td class="${cls(s.ev)}">${pt(s.ev)}¢</td><td class="${cls(s.pnl)}">${sgn(s.pnl)}</td></tr>`
    :`<tr><td class="l">${name}</td><td colspan="8" class="mut">no settled trades yet</td></tr>`;
  document.getElementById('split').innerHTML='<tr><th class="l">Group</th><th>Trades</th><th>Won</th><th>Win %</th><th>Avg price</th><th>Win % − price</th><th>Break-even</th><th>EV / contract</th><th>P/L</th></tr>'+
   row('All',D.all)+[10,20,30,40,50].map(s=>row(s+' s left',D.by_sec[s])).join('')+row('YES',D.by_side.YES)+row('NO',D.by_side.NO);
}
function drawPL(){
  const cv=document.getElementById('pl'),[x,W,H]=setup(cv);const L=56,R=12,T=10,B=26;
  const tr=D.trades.map(t=>({...t,x:ts(t.time_utc)})).sort((a,b)=>a.x-b.x);
  const t0=tr.length?tr[0].x-900:D.now-3600,t1=Math.max(D.now,t0+3600);let cum=0;const S=[[t0,0]];
  tr.forEach(t=>{cum+=t.pnl||0;S.push([t.x,cum])});S.push([t1,cum]);
  let lo=0,hi=0;S.forEach(p=>{lo=Math.min(lo,p[1]);hi=Math.max(hi,p[1])});if(hi-lo<5){hi+=2.5;lo-=2.5}const pad=(hi-lo)*.08;lo-=pad;hi+=pad;
  const X=t=>L+(t-t0)/(t1-t0)*(W-L-R),Y=v=>T+(hi-v)/(hi-lo)*(H-T-B);
  x.font='12px system-ui';x.fillStyle=css('--mut');x.strokeStyle=css('--grid');x.lineWidth=1;
  const st=niceStep((hi-lo)/5);for(let v=Math.ceil(lo/st)*st;v<=hi;v+=st){x.beginPath();x.moveTo(L,Y(v));x.lineTo(W-R,Y(v));x.stroke();x.textAlign='right';x.fillText(sgn(v).replace('.00',''),L-6,Y(v)+4)}
  x.textAlign='center';const n=Math.max(2,Math.floor((W-L-R)/140));for(let i=0;i<=n;i++){const t=t0+(t1-t0)*i/n;x.fillText(fDT.format(new Date(t*1000)).replace(/:\d\d (AM|PM)/,' $1'),X(t),H-8)}
  x.strokeStyle=css('--lk');x.lineWidth=2.5;x.beginPath();S.forEach((p,i)=>{if(i)x.lineTo(X(p[0]),Y(S[i-1][1]));x[i?'lineTo':'moveTo'](X(p[0]),Y(p[1]))});x.stroke();
  tr.forEach(t=>{x.fillStyle=css(t.won?'--win':'--lose');x.beginPath();x.arc(X(t.x),Y(S.find(p=>p[0]===t.x)?.[1]??0),3,0,7);x.fill()});
  SER=S;geom={t0,t1,L,R,W};
}
function valAt(s,t){let v=0;for(const p of s){if(p[0]<=t)v=p[1];else break}return v}
const tip=document.getElementById('tip'),pl=document.getElementById('pl');
pl.addEventListener('mousemove',e=>{if(!geom)return;const r=pl.getBoundingClientRect(),t=geom.t0+(e.clientX-r.left-geom.L)/(geom.W-geom.L-geom.R)*(geom.t1-geom.t0);
  if(t<geom.t0||t>geom.t1){tip.style.display='none';return}
  tip.innerHTML='<b>'+fDT.format(new Date(t*1000))+'</b><br>Lock-in: '+sgn(valAt(SER,t));
  tip.style.display='block';tip.style.left=(e.clientX+14)+'px';tip.style.top=(e.clientY+10)+'px'});
pl.addEventListener('mouseleave',()=>tip.style.display='none');
function tables(){
  const op=D.open;document.getElementById('opn').innerHTML=op.length?'<tr><th class="l">Bought (ET)</th><th>Side</th><th>Secs left</th><th>Ct</th><th>Price</th><th>Chance</th><th>Edge</th><th>BRTI</th><th>Locked avg</th><th>Strike</th></tr>'+
   op.map(o=>`<tr><td class="l">${fDT.format(new Date(ts(o.time_utc)*1000))}</td><td class="op">${o.side}</td><td>${o.secs_left??'–'}</td><td>${o.contracts}</td><td>${c(o.price)}</td><td>${o.model==null?'–':(o.model*100).toFixed(1)+'%'}</td><td>${pt(o.gap)}</td><td>${usd(o.brti)}</td><td>${usd(o.locked_avg)}</td><td>${usd(o.strike)}</td></tr>`).join(''):'<tr><td class="mut">No open positions right now.</td></tr>';
  const tr=D.trades.slice().sort((a,b)=>ts(b.time_utc)-ts(a.time_utc));
  document.getElementById('trd').innerHTML='<tr><th class="l">Bought (ET)</th><th>Side</th><th>Secs left</th><th>Ct</th><th>Price</th><th>Chance</th><th>Edge</th><th>BRTI</th><th>Locked avg</th><th>Strike</th><th>Result</th><th>P/L</th></tr>'+
   (tr.length?tr.map(t=>`<tr><td class="l">${fDT.format(new Date(ts(t.time_utc)*1000))}</td><td>${t.side}</td><td>${t.secs_left??'–'}</td><td>${t.contracts}</td><td>${c(t.price)}</td><td>${t.model==null?'–':(t.model*100).toFixed(1)+'%'}</td><td>${pt(t.gap)}</td><td>${usd(t.brti)}</td><td>${usd(t.locked_avg)}</td><td>${usd(t.strike)}</td><td class="${t.won?'won':'lost'}">${t.won?'Won':'Lost'}</td><td class="${t.pnl>=0?'won':'lost'}">${sgn(t.pnl)}</td></tr>`).join(''):'<tr><td class="mut" colspan="12">No settled trades yet.</td></tr>');
  const ck=D.checks.slice().sort((a,b)=>ts(b.time_utc)-ts(a.time_utc));
  document.getElementById('chk').innerHTML='<tr><th class="l">Time (ET)</th><th>Secs left</th><th>BRTI</th><th>Locked avg</th><th>Strike</th><th>Chance YES</th><th>YES ask</th><th>YES edge</th><th>NO ask</th><th>NO edge</th><th>Lag</th><th class="l">Decision</th></tr>'+
   (ck.length?ck.map(r=>`<tr><td class="l">${fDT.format(new Date(ts(r.time_utc)*1000))}</td><td>${r.secs_left??'–'}</td><td>${usd(r.brti)}</td><td>${usd(r.locked_avg)}</td><td>${usd(r.strike)}</td><td>${pc(r.p_yes)}</td><td>${c(r.yes_ask)}</td><td>${pt(r.yes_edge)}</td><td>${c(r.no_ask)}</td><td>${pt(r.no_edge)}</td><td>${r.lag==null?'–':r.lag+' s'}</td><td class="l">${r.decision||''}</td></tr>`).join(''):'<tr><td class="mut" colspan="12">No checks yet. The bot only looks in the last 50 seconds of each 15-minute market.</td></tr>');
}
async function tick(){try{const r=await fetch('data.json',{cache:'no-store'});D=await r.json();document.getElementById('err').textContent='';
  if(!D.found){document.getElementById('tiles').innerHTML='<div class="mut">No lock-in bot logs found yet.</div>';return}
  tiles();split();drawPL();tables()}catch(e){document.getElementById('err').textContent='connection lost, retrying…'}}
tick();setInterval(tick,10000);addEventListener('resize',()=>{if(D&&D.found)drawPL()});
</script></body></html>"""
