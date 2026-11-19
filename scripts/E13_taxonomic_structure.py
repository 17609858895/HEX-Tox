"""E13: taxonomic structure of condition sensitivity (data for the taxonomy panel).

 1 per-species observed within-study slopes (duration, temperature) where enough pairs exist, with cluster SEs
 2 variance components of the species slopes across class / order / family (nested, method of moments)
 3 CAL-predicted sensitivity for every species in the LC50 set, including species with no within-study pairs,
   evaluated at a common reference condition (96 h -> 672 h; 20 -> 24 degC)
 4 held-out-taxon check: are predicted family-level slopes correlated with observed ones when the family's own
   pairs are excluded from training?
Outputs a species table with full taxonomy for drawing the tree, plus the variance-component and validation tables.
"""
import sys, time
import numpy as np, pandas as pd, lightgbm as lgb, statsmodels.formula.api as smf
from sklearn.model_selection import GroupKFold
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from common import load_lc50, to_codes, CAT, R, D

L96, L672 = np.log10(96.0), np.log10(672.0)
F = pd.read_parquet(D / "chem_features.parquet")
SDESC = ["MolLogP", "MolWt", "TPSA", "NumAromaticRings", "FractionCSP3", "has_metal", "xlogp_pubchem"]
sp = pd.read_parquet(D / "species.parquet")
x = load_lc50(min_h=1, max_h=24 * 60, include_censored=False)
x = x.merge(F[["test_cas", "parent_group", "chem_class"] + SDESC], on="test_cas", how="left")
for c in SDESC:
    x[c] = pd.to_numeric(x[c], errors="coerce")
x["chem_class"] = x.chem_class.fillna("unknown")
x["parent_group"] = x.parent_group.fillna(x.test_cas)
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
P["species"] = u.species_number.values[P.i]
P["type"] = np.select([(P.dT.abs().fillna(0) < 0.5) & (P.dlogt.abs() > 0.01),
                       (P.dlogt.abs() <= 0.01) & (P.dT.abs() >= 2)], ["duration", "temperature"], "other")
P = P[P.type != "other"].reset_index(drop=True)

# ---------- 1 per-species observed slopes
rows = []
for (s_, typ), d in P.groupby(["species", "type"]):
    v = "dlogt" if typ == "duration" else "dT"
    if len(d) < 8 or d.ref.nunique() < 2:
        continue
    f = smf.ols(f"dy ~ {v} - 1", d).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d.ref)[0]})
    rows.append(dict(species_number=s_, type=typ, slope=f.params[v], se=f.bse[v], n_pairs=len(d),
                     refs=d.ref.nunique(), chems=u.test_cas.values[d.i.values].size))
obs = pd.DataFrame(rows)
obs = obs.merge(sp[["species_number", "latin_name", "common_name", "kingdom", "phylum_division", "class", "tax_order",
                    "family", "genus", "ecotox_group"]], on="species_number", how="left")
obs.to_csv(R / "E13_species_observed_slopes.csv", index=False)
print("species with observed slopes:", obs.groupby("type").size().to_dict(), flush=True)

# ---------- 2 variance components across taxonomy (method of moments on species slopes, weighted by 1/se^2 floor)
vc = []
for typ, d in obs.groupby("type"):
    d = d.dropna(subset=["class", "tax_order", "family"]).copy()
    d["nest_order"] = d["class"].astype(str) + "|" + d.tax_order.astype(str)
    d["nest_family"] = d.nest_order + "|" + d.family.astype(str)
    w = 1.0 / np.clip(d.se.values, 0.02, None) ** 2
    grand = np.average(d.slope, weights=w)
    tot = float(np.average((d.slope - grand) ** 2, weights=w))
    err = float(np.average(np.clip(d.se.values, 0.02, None) ** 2, weights=w))
    # sequential variance explained by nested taxonomic grouping (weighted group means as predictions)
    share, prev = {}, 0.0
    for lev, col in [("class", "class"), ("order", "nest_order"), ("family", "nest_family")]:
        fitted = d.groupby(col).apply(
            lambda z: np.average(z.slope, weights=1 / np.clip(z.se, 0.02, None) ** 2)).reindex(d[col]).values
        r2 = float(1 - np.average((d.slope.values - fitted) ** 2, weights=w) / tot)
        share[lev] = dict(cumulative_r2=r2, incremental=r2 - prev)
        prev = r2
    vc.append(dict(type=typ, n_species=len(d), total_var=tot, sampling_var_share=err / tot,
                   r2_class=share["class"]["cumulative_r2"], r2_class_order=share["order"]["cumulative_r2"],
                   r2_class_order_family=share["family"]["cumulative_r2"],
                   incr_order=share["order"]["incremental"], incr_family=share["family"]["incremental"],
                   n_classes=d["class"].nunique(), n_orders=d.nest_order.nunique(),
                   n_families=d.nest_family.nunique()))
VC = pd.DataFrame(vc)
VC.to_csv(R / "E13_variance_components.csv", index=False)
print(VC.round(5).to_string(), flush=True)

# ---------- 3/4 CAL sensitivity for every species, with held-out-family validation
BASE = [c for c in CAT if c != "test_cas"] + ["pub_year", "active_ingredient"]
PF = ["log_dur", "temp_C", "pH", "log_hardness"]
EXTRA = ["chem_class"] + SDESC
CATS = [c for c in CAT if c != "test_cas"] + ["chem_class"]
FEATS = BASE + EXTRA + [c + "_i" for c in PF] + ["d_" + c for c in PF]
LP = dict(objective="regression", learning_rate=0.05, num_leaves=63, min_data_in_leaf=20, feature_fraction=0.8,
          bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, cat_smooth=10, max_cat_to_onehot=8, verbose=-1,
          num_threads=28)
u["family"] = u.family.fillna("unknown")  # taxonomy already present from the derived LC50 table


def pf(Pp, flip=False):
    a, b = (Pp.j.values, Pp.i.values) if flip else (Pp.i.values, Pp.j.values)
    Fm = u.loc[a, BASE + EXTRA].reset_index(drop=True).copy()
    for c in PF:
        Fm[c + "_i"] = u[c].values[a]
        Fm["d_" + c] = u[c].values[b] - u[c].values[a]
    return Fm


P["family"] = u.family.values[P.i]
pred = np.full(len(P), np.nan)
for k, (itr, ite) in enumerate(GroupKFold(5).split(u, groups=u.family)):
    t0 = time.time()
    fam_te = set(u.family.values[ite])
    te = P.family.isin(fam_te).values
    Ptr = P[~te]
    Ftr = pd.concat([pf(Ptr), pf(Ptr, True)], ignore_index=True)
    ytr = np.concatenate([Ptr.dy.values, -Ptr.dy.values])
    A, B = to_codes(Ftr, pf(P[te]), cats=CATS)
    m = lgb.train(LP, lgb.Dataset(A[FEATS], ytr, categorical_feature=CATS), 800)
    pred[te] = m.predict(B[FEATS])
    print("family fold", k, f"{time.time()-t0:.0f}s", flush=True)
P["pred_heldout_family"] = pred
fam = []
for (f_, typ), d in P.dropna(subset=["pred_heldout_family"]).groupby(["family", "type"]):
    v = "dlogt" if typ == "duration" else "dT"
    if len(d) < 20 or d.ref.nunique() < 3:
        continue
    sl = lambda col: np.sum(d[col] * d[v]) / np.sum(d[v] ** 2)
    fam.append(dict(family=f_, type=typ, n=len(d), refs=d.ref.nunique(), obs=sl("dy"),
                    pred=sl("pred_heldout_family")))
FAM = pd.DataFrame(fam)
FAM.to_csv(R / "E13_family_heldout_slopes.csv", index=False)
for typ, d in FAM.groupby("type"):
    print(typ, "families:", len(d), "obs-vs-predicted family slope corr:", round(float(np.corrcoef(d.obs, d.pred)[0, 1]), 3),
          "| rank corr:", round(float(d[["obs", "pred"]].corr(method="spearman").iloc[0, 1]), 3), flush=True)

# species-level reference sensitivity for the tree panel (trained on all pairs)
Ftr = pd.concat([pf(P), pf(P, True)], ignore_index=True)
ytr = np.concatenate([P.dy.values, -P.dy.values])
ref = u.drop_duplicates("species_number").reset_index(drop=True).copy()
ANCHOR = dict(log_dur=L96, temp_C=20.0, pH=float(u.pH.median()), log_hardness=float(u.log_hardness.median()))
grid = []
for lab, tgt in [("dur_96_to_672", dict(log_dur=L672)), ("temp_20_to_24", dict(temp_C=24.0))]:
    G = ref[BASE + EXTRA].reset_index(drop=True).copy()
    for c in PF:
        G[c + "_i"] = ANCHOR[c]
        G["d_" + c] = tgt.get(c, ANCHOR[c]) - ANCHOR[c]
    grid.append((lab, G))
A, _ = to_codes(Ftr, Ftr, cats=CATS)
m = lgb.train(LP, lgb.Dataset(A[FEATS], ytr, categorical_feature=CATS), 800)
out = ref[["species_number", "latin_name", "class", "tax_order", "family"]].reset_index(drop=True).copy()
for lab, G in grid:
    _, B = to_codes(Ftr, G, cats=CATS)
    out[lab] = m.predict(B[FEATS])
out = out.merge(sp[["species_number", "kingdom", "phylum_division", "genus", "ecotox_group"]], on="species_number",
                how="left")
nrec = x.groupby("species_number").size().rename("n_records")
out = out.merge(nrec, on="species_number", how="left")
out.to_csv(R / "E13_species_sensitivity_tree.csv", index=False)
print("species in tree table:", len(out), "| median 96->672 h shift (log10):", round(float(out.dur_96_to_672.median()), 3),
      "| median +4 degC shift:", round(float(out.temp_20_to_24.median()), 3))
