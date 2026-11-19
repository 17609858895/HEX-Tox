"""E11b: how much of a published "new chemical" score comes from the split definition?

Same model, same features, same data; only the grouping unit of the 5-fold split changes:
 row       : random rows (replicates of one test can straddle folds)
 cas       : CAS number (the usual "unseen chemical" split; salts, hydrates and stereoisomers of one parent are
             still different CAS numbers, so they can land on both sides)
 parent    : parent group - the metal element for metal substances, otherwise the InChIKey skeleton of the largest
             organic fragment (counterions, salts and hydrates of one parent stay together)
 scaffold  : Murcko scaffold of the largest fragment (metals keep their element group)
Reported together with the empirical replicate-noise floor of the same data.
"""
import sys, time
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import GroupKFold, KFold
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from common import load_lc50, to_codes, metrics, CAT, NUM, R, D

F = pd.read_parquet(D / "chem_features.parquet")
DESC = [c for c in F.columns if c in ("MolWt", "MolLogP", "TPSA", "NumHDonors", "NumHAcceptors", "NumRotatableBonds",
        "RingCount", "NumAromaticRings", "FractionCSP3", "HeavyAtomCount", "NumHeteroatoms", "BertzCT", "LabuteASA",
        "NumAliphaticRings", "NumSaturatedRings", "MaxPartialCharge", "MinPartialCharge", "NumValenceElectrons",
        "frag_logp", "xlogp_pubchem", "n_fragments", "formal_charge_total", "has_metal", "organic", "pubchem_mw")] \
       + [c for c in F.columns if c.startswith("el_")]
FP = [c for c in F.columns if c.startswith("maccs_") or c.startswith("morgan_")]
x = load_lc50(min_h=24, max_h=96, include_censored=False)
x = x.merge(F[["test_cas", "parent_group", "scaffold"] + DESC + FP], on="test_cas", how="inner").reset_index(drop=True)
for c in DESC + FP:
    x[c] = pd.to_numeric(x[c], errors="coerce")
CATS = [c for c in CAT if c != "test_cas"]
FEATS = CATS + NUM + DESC + FP
P = dict(objective="regression", learning_rate=0.05, num_leaves=127, min_data_in_leaf=30, feature_fraction=0.5,
         bagging_fraction=0.8, bagging_freq=1, lambda_l2=2.0, cat_smooth=10, max_cat_to_onehot=8, verbose=-1,
         num_threads=28)
g = x.groupby(["test_cas", "species_number", x.dur_h.round()]).log10_mgL
rep = g.transform("size") >= 2
noise = float(np.sqrt(np.mean((x.log10_mgL[rep] - g.transform("mean")[rep]) ** 2)))
print("replicate-noise RMSE:", round(noise, 3), flush=True)
rows = []
for name, groups in [("row (random)", None), ("cas", x.test_cas.values), ("parent", x.parent_group.values),
                     ("scaffold", x.scaffold.values)]:
    cv = KFold(5, shuffle=True, random_state=0) if groups is None else GroupKFold(5)
    pred = np.full(len(x), np.nan)
    t0 = time.time()
    for itr, ite in cv.split(x, groups=groups):
        tr, te = to_codes(x.iloc[itr], x.iloc[ite], cats=CATS)
        m = lgb.train(P, lgb.Dataset(tr[FEATS], tr.log10_mgL, categorical_feature=CATS), 1000)
        pred[ite] = m.predict(te[FEATS])
    y = x.log10_mgL.values
    mm = metrics(y, pred)
    rows.append(dict(split=name, n=len(x), units=(len(x) if groups is None else pd.Series(groups).nunique()),
                     rmse=mm["rmse"], r2=mm["r2"], mae=mm["mae"],
                     within_10x=float(np.mean(np.abs(pred - y) <= 1)),
                     median_fold_error=float(10 ** np.median(np.abs(pred - y))),
                     replicate_noise_rmse=noise, seconds=time.time() - t0))
    print(pd.DataFrame(rows).tail(1).round(3).to_string(), flush=True)
res = pd.DataFrame(rows)
res.to_csv(R / "E11b_split_strictness.csv", index=False)
print(res.round(3).to_string())
