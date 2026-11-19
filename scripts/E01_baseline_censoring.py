"""E01: pairwise LC50 baseline under several split designs; censoring-aware Tobit-LGBM vs exact-only vs naive substitution.

Target: log10 LC50 (mg/L), aquatic, mortality, 24-96 h.
"""
import sys, time
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import GroupKFold, KFold
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from common import load_lc50, to_codes, metrics, CAT, NUM, R
from tobit_lgb import fit_tobit_lgb, censored_loglik

x = load_lc50(min_h=24, max_h=96)
FEAT = CAT + NUM
SIG = 0.7
splits = {
    "random": (KFold(5, shuffle=True, random_state=0), None),
    "reference": (GroupKFold(5), x.reference_number.values),
    "pair": (GroupKFold(5), x.pair.values),
    "species": (GroupKFold(5), x.species_number.values),
}
P = dict(learning_rate=0.06, num_leaves=127, min_data_in_leaf=20, feature_fraction=0.8, bagging_fraction=0.8,
         bagging_freq=1, lambda_l2=1.0, cat_smooth=10, max_cat_to_onehot=8, verbose=-1, num_threads=16)
NR = 800
cz_all = x.cens.map({"exact": 0, "right": 1, "left": 2}).values

rows = []; oof = {}
for sname, (cv, groups) in splits.items():
    pred = {k: np.full(len(x), np.nan) for k in ["exact_only", "naive_sub", "tobit"]}
    for k, (itr, ite) in enumerate(cv.split(x, groups=groups)):
        t0 = time.time()
        tr, te = to_codes(x.iloc[itr], x.iloc[ite])
        cz = cz_all[itr]
        ex = cz == 0
        m = lgb.train(dict(P, objective="regression"),
                      lgb.Dataset(tr[FEAT][ex], tr.log10_mgL[ex], categorical_feature=CAT), NR)
        pred["exact_only"][ite] = m.predict(te[FEAT])
        m = lgb.train(dict(P, objective="regression"), lgb.Dataset(tr[FEAT], tr.log10_mgL, categorical_feature=CAT), NR)
        pred["naive_sub"][ite] = m.predict(te[FEAT])
        m, init = fit_tobit_lgb(tr[FEAT], tr.log10_mgL.values, cz, P, NR, CAT, sigma=SIG)
        pred["tobit"][ite] = m.predict(te[FEAT]) + init
        print(sname, k, f"{time.time()-t0:.0f}s", flush=True)
    ex = cz_all == 0; rc = cz_all == 1
    share = x.groupby("test_cas").cens.transform(lambda s: (s == "right").mean()).values
    hi = ex & (share >= 0.3)
    y = x.log10_mgL.values
    for k, p in pred.items():
        mm = metrics(y[ex], p[ex]); mh = metrics(y[hi], p[hi])
        rows.append(dict(split=sname, model=k, rmse=mm["rmse"], r2=mm["r2"], mae=mm["mae"], n_exact=mm["n"],
                         rmse_highcens=mh["rmse"], bias_highcens=float(np.mean(p[hi] - y[hi])), n_highcens=mh["n"],
                         right_cens_consistency=float(np.mean(p[rc] >= y[rc])),
                         cens_loglik_all=float(np.mean(censored_loglik(y, p, cz_all, SIG)))))
        oof[f"{sname}__{k}"] = p
    print(pd.DataFrame(rows).tail(3).round(3).to_string(), flush=True)

res = pd.DataFrame(rows)
res.to_csv(R / "E01_baseline_censoring.csv", index=False)
pd.DataFrame(oof).assign(log10_mgL=x.log10_mgL.values, cens=x.cens.values, test_cas=x.test_cas.values,
                         species=x.species_number.values, ref=x.reference_number.values,
                         sp_group=x.sp_group1.values, chem_group=x.chem_group.values).to_parquet(R / "E01_oof.parquet")
print(res.round(3).to_string())
