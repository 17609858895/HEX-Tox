"""Compile directly traceable counts and matched held-out comparisons."""
import sys,json
from pathlib import Path
import numpy as np,pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import load_lc50,R,D
x=load_lc50();z=load_lc50(min_h=24,max_h=96)
ex=load_lc50(include_censored=False)
sk=['reference_number','test_cas','species_number','organism_lifestage','exposure_type','media_type','conc1_type','chem_analysis_method']
ex['sid']=ex[sk].astype(str).agg('|'.join,axis=1)
ex['ckey']=ex.sid+'|'+ex.dur_h.round(2).astype(str)+'|'+ex.temp_C.round(1).astype(str)
u=ex.drop_duplicates('ckey');ns=u.groupby('sid').size();n_multi=ns[ns>=2]
p=pd.read_parquet(R/'E05_pairs.parquet');rows=[];rng=np.random.default_rng(20261030)
for typ,v in [('duration','dlogt'),('temperature','dT')]:
    d=p[(p.type==typ)&(p.fold>=0)].copy()
    d['vsq']=d[v]**2
    for col in ['dy','p_pooled','p_cal','p_cal_nomono']:
        d[col+'_cross']=d[col]*d[v]
        if col!='dy':d[col+'_ss']=(d[col]-d.dy)**2
    g=d.groupby('ref').sum(numeric_only=True);g['nn']=d.groupby('ref').size()
    arr=g[[v+'_cross' for v in ['dy','p_pooled','p_cal','p_cal_nomono']]+['vsq','p_pooled_ss','p_cal_ss','nn']].to_numpy()
    draws=[]
    for _ in range(1200):
        a=arr[rng.integers(0,len(g),len(g))].sum(0)
        draws.append([a[1]/a[0],a[2]/a[0],np.sqrt(a[6]/a[7])-np.sqrt(a[5]/a[7])])
    b=np.array(draws)
    rows.append({'type':typ,'n':len(d),'refs':d.ref.nunique(),'obs_slope':float(np.sum(d.dy*d[v])/np.sum(d[v]**2)),
       'pooled_slope':float(np.sum(d.p_pooled*d[v])/np.sum(d[v]**2)),
       'cal_slope':float(np.sum(d.p_cal*d[v])/np.sum(d[v]**2)),
       'pooled_recovery_ci':np.quantile(b[:,0],[.025,.975]).tolist(),'cal_recovery_ci':np.quantile(b[:,1],[.025,.975]).tolist(),
       'rmse_difference_ci':np.quantile(b[:,2],[.025,.975]).tolist(),
       'missing_dT':float(d.dT.isna().mean())})
out={'records_1h_60d':len(x),'exact_1h_60d':len(ex),'chemicals':x.test_cas.nunique(),'species':x.species_number.nunique(),
     'references':x.reference_number.nunique(),'years':[float(x.pub_year.min()),float(x.pub_year.max())],
     'censoring':x.cens.value_counts().to_dict(),'records_24_96h':len(z),'exact_24_96h':int((z.cens=='exact').sum()),
     'multi_series':len(n_multi),'multi_condition_rows':int(n_multi.sum()),'candidate_pairs':int(((n_multi*(n_multi-1))/2).sum()),
     'evaluated_pairs':len(p),'evaluated_references':p.ref.nunique(),'temperature_missing_records':float(x.temp_C.isna().mean()),
     'contrasts':rows}
(R/'manuscript_metrics.json').write_text(json.dumps(out,indent=2),encoding='utf8');print(json.dumps(out,indent=2))
