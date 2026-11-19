"""E10: heterogeneity of warming effects by chemical class (structure-derived) and taxon.

 (a) observed within-study fixed-effect slopes (log10 LC50 per degC) by chemical class x taxon, clustered by reference
 (b) structure-aware contrast learner (CAL + class/descriptor features) evaluated on temperature contrasts of
     unseen chemicals (GroupKFold by parent group): does it reproduce class-level slopes (e.g. pyrethroid > 0)?
     Compared with CAL without structure and with the pooled level model (differencing).
"""
import sys, time
import numpy as np, pandas as pd, lightgbm as lgb, statsmodels.formula.api as smf
from sklearn.model_selection import GroupKFold
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from common import load_lc50, to_codes, CAT, NUM, R, D

x = load_lc50(min_h=1, max_h=24 * 60, include_censored=False)
F = pd.read_parquet(D / "chem_features.parquet")
DESC = ["MolLogP", "MolWt", "TPSA", "NumHDonors", "NumHAcceptors", "NumAromaticRings", "FractionCSP3", "has_metal",
        "xlogp_pubchem"] + [c for c in F.columns if c.startswith("el_")]
x = x.merge(F[["test_cas", "parent_group", "chem_class"] + DESC], on="test_cas", how="inner")
for c in DESC:
    x[c] = pd.to_numeric(x[c], errors="coerce")
skey = ["reference_number", "test_cas", "species_number", "organism_lifestage", "exposure_type", "media_type",
        "conc1_type", "chem_analysis_method"]
x["sid"] = x[skey].astype(str).agg("|".join, axis=1)
x["ckey"] = x.sid + "|" + x.dur_h.round(2).astype(str) + "|" + x.temp_C.round(1).astype(str)
agg = {c: "first" for c in x.columns if c not in ("log10_mgL", "ckey")}
agg["log10_mgL"] = "mean"
u = x.groupby("ckey").agg(agg).reset_index(drop=True)
pairs = []
for sid, d in u.groupby("sid"):
    idx = d.index.values
    for a in range(len(idx)):
        for b in range(a + 1, len(idx)):
            pairs.append((idx[a], idx[b]))
P = pd.DataFrame(pairs, columns=["i", "j"])
P["dy"] = u.log10_mgL.values[P.j] - u.log10_mgL.values[P.i]
P["dlogt"] = u.log_dur.values[P.j] - u.log_dur.values[P.i]
P["dT"] = u.temp_C.values[P.j] - u.temp_C.values[P.i]
P["ref"] = u.reference_number.values[P.i]
P["pg"] = u.parent_group.values[P.i]
P["cls"] = u.chem_class.values[P.i]
P["sp"] = u.sp_group1.values[P.i]
P["type"] = np.select([(P.dT.abs().fillna(0) < 0.5) & (P.dlogt.abs() > 0.01),
                       (P.dlogt.abs() <= 0.01) & (P.dT.abs() >= 2)], ["duration", "temperature"], "other")
P = P[P.type != "other"].reset_index(drop=True)
T = P[P.type == "temperature"].copy()

# (a) observed within-study slopes by class and by class x taxon
rows = []
for keys, d in list(T.groupby("cls")) + list(T.groupby(["cls", "sp"])):
    if d.ref.nunique() < 4 or len(d) < 30:
        continue
    f = smf.ols("dy ~ dT - 1", d).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d.ref)[0]})
    k = keys if isinstance(keys, tuple) else (keys, "all taxa")
    rows.append(dict(chem_class=k[0], taxon=k[1], n_pairs=len(d), refs=d.ref.nunique(), chems=d.pg.nunique(),
                     slope=f.params["dT"], se=f.bse["dT"], pct_per_C=(10 ** f.params["dT"] - 1) * 100))
obs = pd.DataFrame(rows)
obs.to_csv(R / "E10_observed_class_slopes.csv", index=False)
print(obs[obs.taxon == "all taxa"].sort_values("slope").round(4).to_string())

# (b) structure-aware CAL on unseen chemicals
BASE = [c for c in CAT if c != "test_cas"] + ["pub_year", "active_ingredient"]
STRUCT = ["chem_class"] + DESC
PF = ["log_dur", "temp_C", "pH", "log_hardness"]


def pf(Pp, extra, flip=False):
    a, b = (Pp.j.values, Pp.i.values) if flip else (Pp.i.values, Pp.j.values)
    Fm = u.loc[a, BASE + extra].reset_index(drop=True).copy()
    for c in PF:
        Fm[c + "_i"] = u[c].values[a]
        Fm["d_" + c] = u[c].values[b] - u[c].values[a]
    return Fm


LP = dict(objective="regression", learning_rate=0.05, num_leaves=63, min_data_in_leaf=20, feature_fraction=0.8,
          bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, cat_smooth=10, max_cat_to_onehot=8, verbose=-1,
          num_threads=28)
CATS = [c for c in CAT if c != "test_cas"]
pred = {k: np.full(len(P), np.nan) for k in ["cal_struct", "cal_nostruct", "pooled_struct"]}
for k, (itr, ite) in enumerate(GroupKFold(5).split(u, groups=u.parent_group)):
    t0 = time.time()
    test_pg = set(u.parent_group.values[ite])
    te = P.pg.isin(test_pg).values
    Ptr = P[~te]
    for nm, extra in [("cal_struct", STRUCT), ("cal_nostruct", [])]:
        Ftr = pd.concat([pf(Ptr, extra), pf(Ptr, extra, True)], ignore_index=True)
        ytr = np.concatenate([Ptr.dy.values, -Ptr.dy.values])
        cats = CATS + (["chem_class"] if extra else [])
        feats = BASE + extra + [c + "_i" for c in PF] + ["d_" + c for c in PF]
        A, B = to_codes(Ftr, pf(P[te], extra), cats=cats)
        m = lgb.train(LP, lgb.Dataset(A[feats], ytr, categorical_feature=cats), 800)
        pred[nm][te] = m.predict(B[feats])
    # pooled level model with structure, trained on rows of training chemicals
    trr = ~u.parent_group.isin(test_pg).values
    cats = CATS + ["chem_class"]
    feats = CATS + NUM + STRUCT
    A, B = to_codes(u[trr], u, cats=cats)
    m = lgb.train(LP, lgb.Dataset(A[feats], A.log10_mgL, categorical_feature=cats), 800)
    lev = m.predict(B[feats])
    pred["pooled_struct"][te] = lev[P.j.values[te]] - lev[P.i.values[te]]
    print("fold", k, f"{time.time()-t0:.0f}s", flush=True)

out = []
for typ in ["temperature", "duration"]:
    mk = (P.type.values == typ)
    v = "dT" if typ == "temperature" else "dlogt"
    for nm, p in pred.items():
        e = p[mk] - P.dy.values[mk]
        out.append(dict(type=typ, model=nm, rmse=np.sqrt(np.mean(e ** 2)),
                        slope=np.sum(p[mk] * P[v].values[mk]) / np.sum(P[v].values[mk] ** 2),
                        obs_slope=np.sum(P.dy.values[mk] * P[v].values[mk]) / np.sum(P[v].values[mk] ** 2)))
res = pd.DataFrame(out)
res.to_csv(R / "E10_unseen_chemical_contrasts.csv", index=False)
print(res.round(4).to_string())
# class-level slopes: observed vs predicted on unseen chemicals
cl = []
for c, d in T.assign(**{k: pred[k][P.type.values == "temperature"] for k in pred}).groupby("cls"):
    if d.ref.nunique() < 4 or len(d) < 30:
        continue
    sl = lambda col: np.sum(d[col] * d.dT) / np.sum(d.dT ** 2)
    cl.append(dict(chem_class=c, n=len(d), refs=d.ref.nunique(), obs=sl("dy"), cal_struct=sl("cal_struct"),
                   cal_nostruct=sl("cal_nostruct"), pooled_struct=sl("pooled_struct")))
cl = pd.DataFrame(cl)
cl.to_csv(R / "E10_class_slopes_unseen.csv", index=False)
print(cl.round(4).to_string())
for k in ["cal_struct", "cal_nostruct", "pooled_struct"]:
    print(k, "class-level corr with observed:", round(np.corrcoef(cl.obs, cl[k])[0, 1], 3))
P.assign(**{f"p_{k}": v for k, v in pred.items()}).to_parquet(R / "E10_pairs.parquet")
