"""E11: matched comparison with published ECOTOX benchmarks.

Published chemical-split RMSE values come from curated subsets, so the full-library number is not comparable.
Nested subsets, each evaluated with GroupKFold by parent chemical group (structure features, exact values only):
 A full      : all taxa, 24-96 h (the paper's main scope)
 B adore3    : fish + crustaceans + algae, 24-96 h          (ADORE scope; Schuer 2023)
 C adore_std : B restricted to standard durations (24/48/72/96 h) and freshwater
 D fish_f2f  : fish only, 96 h, freshwater, standard methods (closest to ADORE t-F2F; Gasser 2024)
 E one_per   : D collapsed to one geometric-mean value per chemical x species (removes replicate-driven variance)
Reported: RMSE, R2, within-10x, median fold error; plus the irreducible within-chemical-species-duration replicate
spread of the same subset (an empirical noise floor for that subset).
"""
import sys, time
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import GroupKFold
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from common import load_lc50, to_codes, metrics, CAT, NUM, R, D

F = pd.read_parquet(D / "chem_features.parquet")
DESC = [c for c in F.columns if c in ("MolWt", "MolLogP", "TPSA", "NumHDonors", "NumHAcceptors", "NumRotatableBonds",
        "RingCount", "NumAromaticRings", "FractionCSP3", "HeavyAtomCount", "NumHeteroatoms", "BertzCT", "LabuteASA",
        "NumAliphaticRings", "NumSaturatedRings", "MaxPartialCharge", "MinPartialCharge", "NumValenceElectrons",
        "frag_logp", "xlogp_pubchem", "n_fragments", "formal_charge_total", "has_metal", "organic", "pubchem_mw")] \
       + [c for c in F.columns if c.startswith("el_")]
FP = [c for c in F.columns if c.startswith("maccs_") or c.startswith("morgan_")]
x0 = load_lc50(min_h=24, max_h=96, include_censored=False)
x0 = x0.merge(F[["test_cas", "parent_group", "chem_class"] + DESC + FP], on="test_cas", how="inner")
for c in DESC + FP:
    x0[c] = pd.to_numeric(x0[c], errors="coerce")
CATS = [c for c in CAT if c != "test_cas"]
FEATS = CATS + NUM + DESC + FP
P = dict(objective="regression", learning_rate=0.05, num_leaves=127, min_data_in_leaf=30, feature_fraction=0.5,
         bagging_fraction=0.8, bagging_freq=1, lambda_l2=2.0, cat_smooth=10, max_cat_to_onehot=8, verbose=-1,
         num_threads=28)

A3 = ["Fish", "Crustaceans", "Algae"]
subsets = {}
subsets["A_full_alltaxa"] = x0
b = x0[x0.sp_group1.isin(A3)]
subsets["B_adore3_taxa"] = b
c = b[(b.dur_h.round().isin([24, 48, 72, 96])) & (b.media_type == "FW")]
subsets["C_adore_standard"] = c
d = x0[(x0.sp_group1 == "Fish") & (x0.dur_h.round() == 96) & (x0.media_type == "FW")
       & (x0.exposure_type.isin(["S", "R", "F"]))]
subsets["D_fish_96h_FW"] = d
subsets["D2_fish_96h_guideline"] = d[d.test_method.isin(["USEPA", "OECD", "ASTM", "OM"])]
e = d.copy()
e["gm"] = e.groupby(["test_cas", "species_number"]).log10_mgL.transform("mean")
e = e.drop_duplicates(["test_cas", "species_number"]).assign(log10_mgL=lambda q: q.gm)
subsets["E_fish_96h_one_per_pair"] = e

rows = []
for name, s in subsets.items():
    s = s.reset_index(drop=True)
    if len(s) < 500:
        print("skip", name, len(s)); continue
    # empirical replicate spread within chemical x species x duration (same subset)
    g = s.groupby(["test_cas", "species_number", s.dur_h.round()]).log10_mgL
    rep = g.transform("size") >= 2
    noise = float(np.sqrt(np.mean((s.log10_mgL[rep] - g.transform("mean")[rep]) ** 2))) if rep.any() else np.nan
    pred = np.full(len(s), np.nan)
    t0 = time.time()
    for itr, ite in GroupKFold(5).split(s, groups=s.parent_group.values):
        tr, te = to_codes(s.iloc[itr], s.iloc[ite], cats=CATS)
        m = lgb.train(P, lgb.Dataset(tr[FEATS], tr.log10_mgL, categorical_feature=CATS), 1000)
        pred[ite] = m.predict(te[FEATS])
    y = s.log10_mgL.values
    mm = metrics(y, pred)
    rows.append(dict(subset=name, n=len(s), chemicals=s.test_cas.nunique(), species=s.species_number.nunique(),
                     groups=s.parent_group.nunique(), rmse=mm["rmse"], r2=mm["r2"], mae=mm["mae"],
                     within_10x=float(np.mean(np.abs(pred - y) <= 1)),
                     median_fold_error=float(10 ** np.median(np.abs(pred - y))),
                     replicate_noise_rmse=noise, seconds=time.time() - t0))
    print(pd.DataFrame(rows).tail(1).round(3).to_string(), flush=True)
res = pd.DataFrame(rows)
res.to_csv(R / "E11_matched_benchmark.csv", index=False)
print(res.round(3).to_string())
