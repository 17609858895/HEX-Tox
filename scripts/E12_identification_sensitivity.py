"""E12: identification sensitivity and negative controls for the within-study contrast design,
plus a structure-aware re-test of the temporal (pre/post-2000) transfer that failed without structure.

Parts
 1 placebo    : replicate pairs inside one series with identical duration and temperature. The observed contrast and
                the predicted contrast should both be ~0; the residual spread is the replicate noise of a contrast.
 2 same test  : duration pairs inside a single test_id (same exposure, same organism batch, repeated observation) vs
                duration pairs from different test_ids of the same study. If the slope is similar, batch differences
                within a study are not driving it.
 3 reported   : pairs where lifestage, organism source, exposure type, concentration type and analysis method are all
                reported (not NR/NC) and identical, vs all pairs.
 4 leave-taxon: leave-one-taxon-out slope estimates (is the pooled slope driven by one group?).
 5 temporal   : CAL with and without structure features, trained on studies published <=2000 and tested after 2000.
"""
import sys, time
import numpy as np, pandas as pd, lightgbm as lgb, statsmodels.formula.api as smf
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from common import load_lc50, to_codes, CAT, R, D

F = pd.read_parquet(D / "chem_features.parquet")
SDESC = ["MolLogP", "MolWt", "TPSA", "NumAromaticRings", "FractionCSP3", "has_metal", "xlogp_pubchem"]
x = load_lc50(min_h=1, max_h=24 * 60, include_censored=False)
x = x.merge(F[["test_cas", "parent_group", "chem_class"] + SDESC], on="test_cas", how="left")
for c in SDESC:
    x[c] = pd.to_numeric(x[c], errors="coerce")
x["chem_class"] = x.chem_class.fillna("unknown")
x["parent_group"] = x.parent_group.fillna(x.test_cas)
skey = ["reference_number", "test_cas", "species_number", "organism_lifestage", "exposure_type", "media_type",
        "conc1_type", "chem_analysis_method"]
x["sid"] = x[skey].astype(str).agg("|".join, axis=1)
x["Tr"] = x.temp_C.round(1)
x = x.reset_index(drop=True)
rows = []

# ---------- 1 placebo: identical conditions inside one series
P0 = []
for sid, d in x.groupby("sid"):
    for (t_, T_), z in d.groupby([d.dur_h.round(2), d.Tr], dropna=False):
        idx = z.index.values
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                P0.append((idx[a], idx[b]))
P0 = pd.DataFrame(P0, columns=["i", "j"])
if len(P0):
    dy = x.log10_mgL.values[P0.j] - x.log10_mgL.values[P0.i]
    rows.append(dict(part="placebo (identical conditions)", n=len(P0), refs=x.reference_number.values[P0.i].size,
                     stat="mean |dy|", value=float(np.mean(np.abs(dy))), extra=f"sd(dy)={np.std(dy):.3f}"))
    print("placebo pairs:", len(P0), "mean|dy|", round(float(np.mean(np.abs(dy))), 3),
          "sd", round(float(np.std(dy)), 3), flush=True)

# ---------- pair construction helper (condition-level, as in E05)
x["ckey"] = x.sid + "|" + x.dur_h.round(2).astype(str) + "|" + x.Tr.astype(str)
agg = {c: "first" for c in x.columns if c not in ("log10_mgL", "ckey")}
agg["log10_mgL"] = "mean"
u = x.groupby("ckey").agg(agg).reset_index(drop=True)
u["tid"] = u.test_id if "test_id" in u.columns else np.nan
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
P["sp"] = u.sp_group1.values[P.i]
P["same_test"] = (u.test_id.values[P.i] == u.test_id.values[P.j]) if "test_id" in u.columns else False
rep = lambda col: (~u[col].astype(str).isin(["NR", "NC", "--", "nan", ""])).values
ok = rep("organism_lifestage") & rep("organism_source") & rep("conc1_type") & rep("chem_analysis_method")
P["fully_reported"] = ok[P.i] & ok[P.j]
P["type"] = np.select([(P.dT.abs().fillna(0) < 0.5) & (P.dlogt.abs() > 0.01),
                       (P.dlogt.abs() <= 0.01) & (P.dT.abs() >= 2)], ["duration", "temperature"], "other")
P = P[P.type != "other"].reset_index(drop=True)


def slope(d, typ):
    v = "dlogt" if typ == "duration" else "dT"
    f = smf.ols(f"dy ~ {v} - 1", d).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d.ref)[0]})
    return f.params[v], f.bse[v], len(d), d.ref.nunique()


for typ in ["duration", "temperature"]:
    q = P[P.type == typ]
    for lab, sel in [("all pairs", q), ("same test_id", q[q.same_test]), ("different test_id", q[~q.same_test]),
                     ("fully reported conditions", q[q.fully_reported])]:
        if len(sel) < 100 or sel.ref.nunique() < 5:
            continue
        s, se, n, r = slope(sel, typ)
        rows.append(dict(part=f"design: {typ}", n=n, refs=r, stat=lab, value=s, extra=f"se={se:.4f}"))
    for tax in q.sp.dropna().unique():
        sel = q[q.sp != tax]
        if sel.ref.nunique() < 10:
            continue
        s, se, n, r = slope(sel, typ)
        rows.append(dict(part=f"leave-one-taxon-out: {typ}", n=n, refs=r, stat=f"excluding {tax}", value=s,
                         extra=f"se={se:.4f}"))
res = pd.DataFrame(rows)
print(res.round(4).to_string(), flush=True)

# ---------- 5 temporal transfer with and without structure
BASE = [c for c in CAT if c != "test_cas"] + ["pub_year", "active_ingredient"]
PF = ["log_dur", "temp_C", "pH", "log_hardness"]
LP = dict(objective="regression", learning_rate=0.05, num_leaves=63, min_data_in_leaf=20, feature_fraction=0.8,
          bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, cat_smooth=10, max_cat_to_onehot=8, verbose=-1,
          num_threads=28)
CATS = [c for c in CAT if c != "test_cas"]


def pf(Pp, extra, flip=False):
    a, b = (Pp.j.values, Pp.i.values) if flip else (Pp.i.values, Pp.j.values)
    Fm = u.loc[a, BASE + extra].reset_index(drop=True).copy()
    for c in PF:
        Fm[c + "_i"] = u[c].values[a]
        Fm["d_" + c] = u[c].values[b] - u[c].values[a]
    return Fm


yr = pd.to_numeric(u.pub_year, errors="coerce").fillna(1990).values
P["yr_i"] = yr[P.i]
te = (P.yr_i > 2000).values
trm = ~te
out = []
for nm, extra in [("CAL + structure", ["chem_class"] + SDESC), ("CAL no structure", [])]:
    cats = CATS + (["chem_class"] if extra else [])
    feats = BASE + extra + [c + "_i" for c in PF] + ["d_" + c for c in PF]
    Ptr = P[trm]
    Ftr = pd.concat([pf(Ptr, extra), pf(Ptr, extra, True)], ignore_index=True)
    ytr = np.concatenate([Ptr.dy.values, -Ptr.dy.values])
    A, B = to_codes(Ftr, pf(P[te], extra), cats=cats)
    m = lgb.train(LP, lgb.Dataset(A[feats], ytr, categorical_feature=cats), 800)
    p = m.predict(B[feats])
    Pt = P[te].copy(); Pt["p"] = p
    for typ in ["duration", "temperature"]:
        d = Pt[Pt.type == typ]
        v = "dlogt" if typ == "duration" else "dT"
        out.append(dict(model=nm, type=typ, n=len(d), rmse=float(np.sqrt(np.mean((d.p - d.dy) ** 2))),
                        slope=float(np.sum(d.p * d[v]) / np.sum(d[v] ** 2)),
                        obs_slope=float(np.sum(d.dy * d[v]) / np.sum(d[v] ** 2)),
                        corr=float(np.corrcoef(d.p, d.dy)[0, 1])))
    # class-level slope recovery on post-2000 temperature pairs
    dT_ = Pt[Pt.type == "temperature"].copy()
    dT_["cls"] = u.chem_class.values[dT_.i]
    cl = []
    for c, d in dT_.groupby("cls"):
        if len(d) < 30 or d.ref.nunique() < 4:
            continue
        sl = lambda col: np.sum(d[col] * d.dT) / np.sum(d.dT ** 2)
        cl.append((c, len(d), sl("dy"), sl("p")))
    if len(cl) >= 3:
        cl = pd.DataFrame(cl, columns=["cls", "n", "obs", "pred"])
        print(nm, "post-2000 class slope corr:", round(float(np.corrcoef(cl.obs, cl.pred)[0, 1]), 3), flush=True)
        print(cl.round(4).to_string(), flush=True)
tmp = pd.DataFrame(out)
print(tmp.round(4).to_string())
res = pd.concat([res, tmp.assign(part="temporal transfer", stat=tmp.model + " | " + tmp.type,
                                 value=tmp.rmse, extra="slope=" + tmp.slope.round(4).astype(str))[
    ["part", "n", "stat", "value", "extra"]]], ignore_index=True)
res.to_csv(R / "E12_identification_sensitivity.csv", index=False)
tmp.to_csv(R / "E12_temporal_structure.csv", index=False)
