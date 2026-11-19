"""E05: within-study contrast learning (CAL) vs pooled ML for predicting condition-induced toxicity changes.

Pairs (i, j) are exact LC50s from the same study series (reference x chemical x species x lifestage x exposure x
media x conc type x analysis method) that differ in duration and/or temperature. Target: dy = y_j - y_i (log10).
 pooled : LightGBM level model on rows of training references; dy_hat = f(cond_j) - f(cond_i)
 global : -0.43 * dlogt - 0.0192 * dT (within-study pooled slopes)
 cal    : LightGBM trained on training-reference pairs (antisymmetric augmentation, monotone in dlogt)
Evaluation: GroupKFold by reference; RMSE of dy by pair type, and anchored prediction of y_j = y_i + dy_hat.
"""
import sys, time
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import GroupKFold
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from common import load_lc50, to_codes, metrics, CAT, NUM, R

x = load_lc50(min_h=1, max_h=24 * 60, include_censored=False)
x["tr_"] = x.temp_C.round(1)
skey = ["reference_number", "test_cas", "species_number", "organism_lifestage", "exposure_type", "media_type",
        "conc1_type", "chem_analysis_method"]
x["sid"] = x[skey].astype(str).agg("|".join, axis=1)
# collapse replicate rows at identical (series, duration, temperature)
x["ckey"] = x.sid + "|" + x.dur_h.round(2).astype(str) + "|" + x.tr_.astype(str)
agg = {c: "first" for c in x.columns if c not in ("log10_mgL", "ckey")}
agg["log10_mgL"] = "mean"
u = x.groupby("ckey").agg(agg).reset_index(drop=True)
u = u[u.sid.isin(u.sid.value_counts().index[u.sid.value_counts() >= 2])].reset_index(drop=True)
print("condition-level rows in multi-condition series:", len(u), "series:", u.sid.nunique())

# build ordered pairs within series
pairs = []
for sid, d in u.groupby("sid"):
    idx = d.index.values
    for a in range(len(idx)):
        for b in range(a + 1, len(idx)):
            pairs.append((idx[a], idx[b]))
pairs = np.array(pairs)
i, j = pairs[:, 0], pairs[:, 1]
P_ = pd.DataFrame({"i": i, "j": j})
P_["dy"] = u.log10_mgL.values[j] - u.log10_mgL.values[i]
P_["dlogt"] = u.log_dur.values[j] - u.log_dur.values[i]
P_["dT"] = u.temp_C.values[j] - u.temp_C.values[i]
P_["ref"] = u.reference_number.values[i]
P_["type"] = np.select([(P_.dT.abs().fillna(0) < 0.5) & (P_.dlogt.abs() > 0.01),
                        (P_.dlogt.abs() <= 0.01) & (P_.dT.abs() >= 2)], ["duration", "temperature"], "other")
P_ = P_[P_.type != "other"].reset_index(drop=True)
print(P_.type.value_counts().to_dict(), "refs", P_.ref.nunique())

PF = ["log_dur", "temp_C", "pH", "log_hardness"]
BASE = [c for c in CAT if c not in ("organism_lifestage",)] + ["organism_lifestage", "pub_year", "active_ingredient"]

def pair_frame(Pp, flip=False):
    a, b = (Pp.j.values, Pp.i.values) if flip else (Pp.i.values, Pp.j.values)
    F = u.loc[a, BASE].reset_index(drop=True).copy()
    for c in PF:
        F[c + "_i"] = u[c].values[a]
        F["d_" + c] = u[c].values[b] - u[c].values[a]
    return F

FEATS = BASE + [c + "_i" for c in PF] + ["d_" + c for c in PF]
CATS = [c for c in CAT]
LP = dict(learning_rate=0.05, num_leaves=63, min_data_in_leaf=20, feature_fraction=0.8, bagging_fraction=0.8,
          bagging_freq=1, lambda_l2=1.0, cat_smooth=10, max_cat_to_onehot=8, verbose=-1, num_threads=24)
mono = [(-1 if f == "d_log_dur" else 0) for f in FEATS]

pred = {k: np.full(len(P_), np.nan) for k in ["pooled", "global", "cal", "cal_nomono"]}
P_["fold"] = -1
refs_all = x.reference_number.values
for k, (itr, ite) in enumerate(GroupKFold(5).split(x, groups=refs_all)):
    t0 = time.time()
    test_refs = set(np.unique(refs_all[ite]))
    te = P_.ref.isin(test_refs).values; trp = ~te
    P_.loc[te, "fold"] = k
    # pooled level model on rows of training references
    xr = x.iloc[itr]
    a_, b_ = to_codes(xr, u)
    m = lgb.train(dict(LP, objective="regression"), lgb.Dataset(a_[CAT + NUM], a_.log10_mgL, categorical_feature=CAT), 800)
    lev = m.predict(b_[CAT + NUM])
    pred["pooled"][te] = lev[P_.j.values[te]] - lev[P_.i.values[te]]
    # contrast learner on training-reference pairs (antisymmetric augmentation)
    Ptr = P_[trp]
    Ftr = pd.concat([pair_frame(Ptr), pair_frame(Ptr, flip=True)], ignore_index=True)
    ytr = np.concatenate([Ptr.dy.values, -Ptr.dy.values])
    Fte = pair_frame(P_[te])
    A, B = to_codes(Ftr, Fte, cats=CATS)
    for nm, mc in [("cal", mono), ("cal_nomono", None)]:
        pp = dict(LP, objective="regression")
        if mc: pp["monotone_constraints"] = mc; pp["monotone_constraints_method"] = "advanced"
        mm = lgb.train(pp, lgb.Dataset(A[FEATS], ytr, categorical_feature=CATS), 800)
        pred[nm][te] = mm.predict(B[FEATS])
    pred["global"][te] = -0.43 * P_.dlogt.values[te] - 0.0192 * P_.dT.fillna(0).values[te]
    print("fold", k, f"{time.time()-t0:.0f}s", flush=True)

ev = P_.fold.values >= 0
rows = []
yj = u.log10_mgL.values[P_.j.values]; yi = u.log10_mgL.values[P_.i.values]
longj = u.dur_h.values[P_.j.values] > 97
for nm, p in pred.items():
    for typ in ["duration", "temperature"]:
        mk = ev & (P_.type.values == typ)
        e = p[mk] - P_.dy.values[mk]
        rows.append(dict(model=nm, type=typ, n=int(mk.sum()), rmse_dy=float(np.sqrt(np.mean(e ** 2))),
                         mae_dy=float(np.mean(np.abs(e))), bias_dy=float(np.mean(e)),
                         corr=float(np.corrcoef(p[mk], P_.dy.values[mk])[0, 1])))
    mk = ev & (P_.type.values == "duration") & longj
    e = p[mk] - P_.dy.values[mk]
    rows.append(dict(model=nm, type="duration_to_>96h", n=int(mk.sum()), rmse_dy=float(np.sqrt(np.mean(e ** 2))),
                     mae_dy=float(np.mean(np.abs(e))), bias_dy=float(np.mean(e)),
                     corr=float(np.corrcoef(p[mk], P_.dy.values[mk])[0, 1])))
res = pd.DataFrame(rows)
res.to_csv(R / "E05_contrast_learning.csv", index=False)
P_.assign(**{f"p_{k}": v for k, v in pred.items()}, yi=yi, yj=yj,
          dur_i=u.dur_h.values[P_.i.values], dur_j=u.dur_h.values[P_.j.values],
          T_i=u.temp_C.values[P_.i.values], sp=u.sp_group1.values[P_.i.values],
          chem_group=u.chem_group.values[P_.i.values], test_cas=u.test_cas.values[P_.i.values]).to_parquet(R / "E05_pairs.parquet")
print(res.round(4).to_string())
