"""E04: reference-grouped benchmark over all durations (1 h - 60 d): TC-Net heads vs LightGBM (L2 / Tobit),
plus the within-study (Mundlak) temperature anchor. Evaluation on exact rows of held-out references.
"""
import sys, time
import numpy as np, pandas as pd, lightgbm as lgb, torch
from sklearn.model_selection import GroupKFold
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from common import load_lc50, to_codes, metrics, CAT, R
from tcnet import fit_tcnet, predict_tcnet
from tobit_lgb import fit_tobit_lgb, censored_loglik

torch.set_num_threads(int(sys.argv[1]) if len(sys.argv) > 1 else 16)
x = load_lc50(min_h=1, max_h=24 * 60, include_censored=True)
g = x[["reference_number", "test_cas", "species_number", "organism_lifestage", "exposure_type", "media_type"]] \
    .astype(str).agg("|".join, axis=1)
x["Tbar"] = x.groupby(g).temp_C.transform("mean")
x["dT"] = (x.temp_C - x.Tbar).fillna(0.0)
NUMS = ["temp_C", "pH", "log_hardness", "pub_year", "active_ingredient"]
NUMS_A = ["Tbar", "pH", "log_hardness", "pub_year", "active_ingredient"]
FEAT = CAT + ["log_dur"] + NUMS
P = dict(learning_rate=0.06, num_leaves=127, min_data_in_leaf=20, feature_fraction=0.8, bagging_fraction=0.8,
         bagging_freq=1, lambda_l2=1.0, cat_smooth=10, max_cat_to_onehot=8, verbose=-1, num_threads=16)
cz = x.cens.map({"exact": 0, "right": 1, "left": 2}).values
names = ["lgb_exact", "lgb_tobit", "tc_free", "tc_haber", "tc_curve", "tc_curve_nocens", "tc_curve_anchor"]
pred = {k: np.full(len(x), np.nan) for k in names}
beta = np.full(len(x), np.nan)
for k, (itr, ite) in enumerate(GroupKFold(5).split(x, groups=x.reference_number)):
    t0 = time.time()
    tr, te = x.iloc[itr], x.iloc[ite]
    a, b = to_codes(tr, te); ex = cz[itr] == 0
    m = lgb.train(dict(P, objective="regression"), lgb.Dataset(a[FEAT][ex], a.log10_mgL[ex], categorical_feature=CAT), 800)
    pred["lgb_exact"][ite] = m.predict(b[FEAT])
    m, init = fit_tobit_lgb(a[FEAT], a.log10_mgL.values, cz[itr], P, 800, CAT, sigma=0.7)
    pred["lgb_tobit"][ite] = m.predict(b[FEAT]) + init
    for head, nm, uc, anc in [("free", "tc_free", True, None), ("haber", "tc_haber", True, None),
                              ("curve", "tc_curve", True, None), ("curve", "tc_curve_nocens", False, None),
                              ("curve", "tc_curve_anchor", True, "dT")]:
        mod, enc = fit_tcnet(tr, CAT, NUMS_A if anc else NUMS, head=head, epochs=40, use_cens=uc, seed=k, anchor_col=anc)
        mu, parts = predict_tcnet(mod, enc, te, parts=True)
        pred[nm][ite] = mu
        if anc:
            beta[ite] = parts["beta"]
    print("fold", k, f"{time.time()-t0:.0f}s", flush=True)

y = x.log10_mgL.values; ex = cz == 0
share = x.groupby("test_cas").cens.transform(lambda s: (s == "right").mean()).values
hi = ex & (share >= 0.3)
dbin = pd.cut(x.dur_h, [0, 23.9, 96.1, 1e9], labels=["<24h", "24-96h", ">96h"]).astype(str).values
rows = []
for nm, p in pred.items():
    r = dict(model=nm, **metrics(y[ex], p[ex]), bias_highcens=float(np.mean(p[hi] - y[hi])),
             right_cons=float(np.mean(p[cz == 1] >= y[cz == 1])),
             cens_ll=float(np.mean(censored_loglik(y, p, cz, 0.7))))
    for d in ["<24h", "24-96h", ">96h"]:
        mk = ex & (dbin == d); r[f"rmse_{d}"] = metrics(y[mk], p[mk])["rmse"]
    rows.append(r)
res = pd.DataFrame(rows)
res.to_csv(R / "E04_tcnet_benchmark.csv", index=False)
out = pd.DataFrame(pred).assign(y=y, cens=x.cens.values, dur_h=x.dur_h.values, beta=beta, dT=x.dT.values,
                                test_cas=x.test_cas.values, sp=x.sp_group1.values, chem_group=x.chem_group.values,
                                ref=x.reference_number.values)
out.to_parquet(R / "E04_pred.parquet")
print(res.round(3).to_string())
print("anchor beta (log10/degC): mean", np.nanmean(beta), "by taxon:")
print(out.groupby("sp").beta.mean().round(4))
