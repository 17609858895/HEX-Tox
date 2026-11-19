"""E03: temperature effect on acute LC50 — within-study anchor vs observational ML / DML estimates.

Estimands: d log10 LC50 / d degC (negative = warming increases toxicity).
 (a) within-study fixed-effects slope (same reference x chemical x species x lifestage x exposure x media x duration)
 (b) pair fixed effects across studies (chemical x species x duration bin)
 (c) naive pooled OLS
 (d) LightGBM partial-dependence slope (10-30 degC)
 (e) DML partialling-out (LightGBM nuisances, reference-grouped cross-fitting), controls incl. chemical/species IDs
"""
import sys
import numpy as np, pandas as pd, lightgbm as lgb, statsmodels.formula.api as smf
from sklearn.model_selection import GroupKFold
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from common import load_lc50, to_codes, CAT, R

x = load_lc50(min_h=24, max_h=96, include_censored=False).dropna(subset=["temp_C"]).reset_index(drop=True)
x["dr"] = x.dur_h.round()
out = []
cl = lambda d: pd.factorize(d.reference_number)[0]

# (a) within-study
x["gid"] = x[["reference_number", "test_cas", "species_number", "organism_lifestage", "exposure_type", "media_type"]] \
    .astype(str).agg("|".join, axis=1) + "|" + x.dr.astype(str)
g = x.groupby("gid").temp_C.agg(["nunique", "min", "max"])
w = x[x.gid.isin(g.index[(g["nunique"] >= 2) & ((g["max"] - g["min"]) >= 3)])].copy()
w["tc"] = w.temp_C - w.groupby("gid").temp_C.transform("mean")
w["yc"] = w.log10_mgL - w.groupby("gid").log10_mgL.transform("mean")
f = smf.ols("yc~tc-1", w).fit(cov_type="cluster", cov_kwds={"groups": cl(w)})
out.append(dict(estimator="within-study FE (anchor)", slope=f.params["tc"], se=f.bse["tc"], n=len(w),
                refs=w.reference_number.nunique()))
# (b) pair FE across studies (exclude within-study variation by using study-mean temperature)
x["pid"] = x.pair + "|" + x.dr.astype(str)
p = x.groupby(["pid", "reference_number"]).agg(T=("temp_C", "mean"), y=("log10_mgL", "mean")).reset_index()
p = p[p.groupby("pid").T.transform("nunique") >= 2]
p["tc"] = p["T"] - p.groupby("pid")["T"].transform("mean"); p["yc"] = p.y - p.groupby("pid").y.transform("mean")
f = smf.ols("yc~tc-1", p).fit(cov_type="cluster", cov_kwds={"groups": cl(p)})
out.append(dict(estimator="between-study pair FE", slope=f.params["tc"], se=f.bse["tc"], n=len(p),
                refs=p.reference_number.nunique()))
# (c) naive
f = smf.ols("log10_mgL~temp_C", x).fit(cov_type="cluster", cov_kwds={"groups": cl(x)})
out.append(dict(estimator="naive pooled OLS", slope=f.params["temp_C"], se=f.bse["temp_C"], n=len(x),
                refs=x.reference_number.nunique()))
# (d) LightGBM PD slope, (e) DML
NUMS = ["log_dur", "pH", "log_hardness", "pub_year", "active_ingredient"]
FEAT = CAT + NUMS
P = dict(objective="regression", learning_rate=0.05, num_leaves=63, min_data_in_leaf=30, feature_fraction=0.8,
         bagging_fraction=0.8, bagging_freq=1, verbose=-1, num_threads=12, cat_smooth=10)
gkf = GroupKFold(5)
ry, rt = np.zeros(len(x)), np.zeros(len(x)); pd_slopes = []
for itr, ite in gkf.split(x, groups=x.reference_number):
    a, b = to_codes(x.iloc[itr], x.iloc[ite])
    my = lgb.train(P, lgb.Dataset(a[FEAT], a.log10_mgL, categorical_feature=CAT), 600)
    mt = lgb.train(P, lgb.Dataset(a[FEAT], a.temp_C, categorical_feature=CAT), 600)
    ry[ite] = b.log10_mgL.values - my.predict(b[FEAT]); rt[ite] = b.temp_C.values - mt.predict(b[FEAT])
    mf = lgb.train(P, lgb.Dataset(a[FEAT + ["temp_C"]], a.log10_mgL, categorical_feature=CAT), 600)
    bb = b[FEAT + ["temp_C"]].copy()
    grid = np.arange(10, 31, 2.0); pdv = []
    for T in grid:
        bb["temp_C"] = T; pdv.append(mf.predict(bb).mean())
    pd_slopes.append(np.polyfit(grid, pdv, 1)[0])
theta = np.sum(rt * ry) / np.sum(rt * rt)
psi = (ry - theta * rt) * rt
# cluster-robust SE by reference
cdf = pd.DataFrame(dict(psi=psi, g=x.reference_number.values)).groupby("g").psi.sum()
se = np.sqrt(np.sum(cdf ** 2)) / np.sum(rt * rt)
out.append(dict(estimator="LightGBM partial dependence", slope=np.mean(pd_slopes), se=np.std(pd_slopes) / np.sqrt(5),
                n=len(x), refs=x.reference_number.nunique()))
out.append(dict(estimator="DML partialling-out (IDs as controls)", slope=theta, se=se, n=len(x),
                refs=x.reference_number.nunique()))
res = pd.DataFrame(out)
res["pct_per_C"] = (10 ** res.slope - 1) * 100
res.to_csv(R / "E03_temperature_anchor.csv", index=False)
print(res.round(4).to_string())
