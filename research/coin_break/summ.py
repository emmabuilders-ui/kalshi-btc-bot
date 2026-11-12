import pandas as pd,numpy as np,sys
def per(d): return 'Dec-Aug' if d<'2026-09-01' else ('Sep-Oct' if d<'2026-11-01' else 'Nov')
def tab(df,label):
    if df.empty: print(label,'no trades'); return
    df=df.copy(); df['per']=df.day.map(per); df['mon']=df.day.str[:7]
    df['ev']=(df.win-df.price-df.fee)*100
    out=[]
    for p,g in df.groupby('per'):
        mon=g.groupby('mon').ev.mean(); 
        out.append((p,len(g),round(g.price.mean()*100,1),round(g.win.mean()*100,1),round(g.fee.mean()*100,1),round((g.win.mean()-g.price.mean())*100,1),round(g.ev.mean(),2),f'{(mon>0).sum()}/{len(mon)}'))
    print(label); print(pd.DataFrame(out,columns=['period','trades','avg price c','win %','fee c','win-price pts','EV c','months +']).to_string(index=False))
if __name__=='__main__':
    for c in sys.argv[1:]:
        for mg in (5,2): tab(pd.read_csv(f'out_{c}_lockin.csv').query('margin==@mg'),f'{c} lock-in margin {mg}c')
        b=pd.read_csv(f'out_{c}_break.csv'); tab(b[b.gate==1],f'{c} break 60-150s with gate'); tab(b,f'{c} break 60-150s no gate')
        br=pd.read_csv(f'out_{c}_brier.csv'); br['m']=(br.p_sa-br.y)**2; br['k']=(br.kal-br.y)**2
        print(c,'Brier by tau (model, kalshi):',br.groupby('tau')[['m','k']].mean().round(4).to_dict('index'))
