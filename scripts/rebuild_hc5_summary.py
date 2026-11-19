"""Recalculate SSDs and paired species-bootstrap intervals from existing standardized records.

No toxicity model is fitted here. All-right-censored SSDs have unidentified location
and are excluded; no chemical is selected by the magnitude of its hazard factor.
"""
import sys, json, concurrent.futures
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats, optimize
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import R
Z=stats.norm.ppf(.05)
S=pd.read_parquet(R/'E07_standardized_records.parquet')
IDENTITY=S.log_dur.sub(np.log10(96)).abs()<1e-10
OFFSET=S.y96-S.log10_mgL
for col in ['y96','y28d','y96_p2','y96_p4']:
    S.loc[IDENTITY,col]=S.loc[IDENTITY,col]-OFFSET[IDENTITY]
def species(col):
    out=[]
    for (c,sp),d in S.groupby(['test_cas','species_number']):
        ex=d[d.cens=='exact']
        if len(ex): out.append((c,sp,ex[col].mean(),False))
        else:
            rc=d[d.cens=='right']
            if len(rc):out.append((c,sp,rc[col].max(),True))
    return pd.DataFrame(out,columns=['test_cas','species','y','rc'])
def hc5(y,rc):
    y=np.asarray(y);rc=np.asarray(rc,bool)
    if rc.all() or not np.isfinite(y).all():return np.nan
    if not rc.any():
        sd=y.std(ddof=0)
        return y.mean()+Z*sd if 0<sd<3 else np.nan
    def f(p):
        sd=np.exp(p[1])
        return -(stats.norm.logpdf(y[~rc],p[0],sd).sum()+stats.norm.logsf(y[rc],p[0],sd).sum())
    v=optimize.minimize(f,[y.mean(),np.log(max(y.std(),.2))],method='Nelder-Mead',
                       options={'maxiter':800,'xatol':1e-5,'fatol':1e-5})
    return v.x[0]+Z*np.exp(v.x[1]) if v.success and 0<np.exp(v.x[1])<3 else np.nan
def one(arg):
    idx,c,a,b,t2,t4,tau,sigma=arg
    a=a.sort_values('species'); ids=a.species
    b=b.set_index('species').reindex(ids)
    n=len(a); y=a.y.to_numpy();rc=a.rc.to_numpy(); y28=b.y.to_numpy();r28=b.rc.to_numpy()
    h0=hc5(y,rc);h28=hc5(y28,r28)
    if not np.isfinite(h0+h28):return {'test_cas':c,'status':'invalid fit','n_species':n,'n_cens_species':int(rc.sum())}
    rng=np.random.default_rng(20261030+idx);draws=[]
    for _ in range(200):
        ii=rng.integers(0,n,n);shift=rng.normal(0,np.sqrt(tau+sigma/n))
        v=hc5(y[ii],rc[ii])-hc5(y28[ii]+shift,r28[ii])
        if np.isfinite(v):draws.append(v)
    d=np.asarray(draws)
    out={'test_cas':c,'status':'included','n_species':n,'n_cens_species':int(rc.sum()),
         'HC5_96_cens':h0,'HC5_y28d':h28,'hidden_factor_28d':10**(h0-h28),'factor_point':10**(h0-h28),
         'n_valid_draws':len(d),'factor_med':10**np.median(d),'factor_lo':10**np.percentile(d,5),
         'factor_hi':10**np.percentile(d,95)}
    for lab,v in [('p2',t2),('p4',t4)]:
        v=v.set_index('species').reindex(ids)
        h=hc5(v.y,v.rc)
        out['hidden_factor_'+lab]=10**(h0-h)
    return out
if __name__=='__main__':
    S.to_parquet(R/'standardized_records_identity.parquet',index=False)
    o=pd.read_parquet(R/'E05_pairs.parquet');o=o[(o.fold>=0)&(o.type=='duration')&(o.dur_j>96)].copy()
    o['e']=o.dy-o.p_cal
    g=o.groupby('test_cas').agg(m=('e','mean'),n=('e','size'),v=('e','var')).dropna()
    sigma=float(np.average(g.v,weights=g.n-1));tau=max(0,float(np.average(g.m**2-sigma/g.n,weights=g.n)))
    tabs={c:species(c) for c in ['y96','y28d','y96_p2','y96_p4']}
    grp={k:{c:d for c,d in v.groupby('test_cas')} for k,v in tabs.items()};args=[];ex=[]
    for i,(c,a) in enumerate(grp['y96'].items()):
        if len(a)<8:continue
        if a.rc.all():
            ex.append({'test_cas':c,'status':'all right censored','n_species':len(a),'n_cens_species':len(a)})
            continue
        args.append((i,c,a,grp['y28d'][c],grp['y96_p2'][c],grp['y96_p4'][c],tau,sigma))
    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as pool: rows=list(pool.map(one,args))
    H=pd.DataFrame(rows+ex)
    H.to_csv(R/'hc5_duration_analysis.csv',index=False)
    ok=H[H.status=='included'].copy()
    ok['percentile96']=ok.HC5_96_cens.rank(pct=True)*100;ok['percentile28']=ok.HC5_y28d.rank(pct=True)*100
    ok['percentile_move']=(ok.percentile28-ok.percentile96).abs()
    ok.to_csv(R/'hc5_rank_analysis.csv',index=False)
    summary={'n':len(ok),'excluded':H[H.status!='included'][['test_cas','status']].to_dict('records'),
      'factor_quantiles':ok.hidden_factor_28d.quantile([.1,.25,.5,.75,.9]).to_dict(),
      'share_gt2':float((ok.hidden_factor_28d>2).mean()),'share_gt3':float((ok.hidden_factor_28d>3).mean()),
      'share_gt10':float((ok.hidden_factor_28d>10).mean()),'lower_gt1':float((ok.factor_lo>1).mean()),
      'upper_gt10':float((ok.factor_hi>10).mean()),'median_interval_width':float((ok.factor_hi/ok.factor_lo).median()),
      'spearman':float(ok[['HC5_96_cens','HC5_y28d']].corr(method='spearman').iloc[0,1]),
      'kendall':float(ok[['HC5_96_cens','HC5_y28d']].corr(method='kendall').iloc[0,1]),
      'median_rank_move':float(ok.percentile_move.median()),'rank_move_gt10':int((ok.percentile_move>10).sum()),
      'tau2':tau,'sigma2':sigma,'warm2_median':float(ok.hidden_factor_p2.median()),'warm4_median':float(ok.hidden_factor_p4.median())}
    (R/'hc5_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))
