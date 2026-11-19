"""E08: extrapolation to unseen chemicals with RDKit structure features (PubChem SMILES).

Data: aquatic LC50 (mortality, 24-96 h), chemicals with parsed structures.
Splits: GroupKFold by parent group (metal element for metal substances, else InChIKey-14 of largest fragment);
        GroupKFold by Murcko scaffold.
Models: no-structure (ECOTOX chemical group only) / descriptors / descriptors + MACCS + Morgan / same with Tobit loss.
"""
import sys, time
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import GroupKFold
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from common import load_lc50, to_codes, metrics, CAT, NUM, R, D
from tobit_lgb import fit_tobit_lgb, censored_loglik

x = load_lc50(min_h=24, max_h=96)
F = pd.read_parquet(D / "chem_features.parquet")
n0, c0 = len(x), x.test_cas.nunique()
x = x.merge(F, on="test_cas", how="inner").reset_index(drop=True)
print(f"rows with structure: {len(x)}/{n0}; chemicals {x.test_cas.nunique()}/{c0}; parent groups {x.parent_group.nunique()}")
CATS = [c for c in CAT if c != "test_cas"]
DESC = [c for c in F.columns if c in ("MolWt", "MolLogP", "TPSA", "NumHDonors", "NumHAcceptors", "NumRotatableBonds",
        "RingCount", "NumAromaticRings", "FractionCSP3", "HeavyAtomCount", "NumHeteroatoms", "BertzCT", "LabuteASA",
        "NumAliphaticRings", "NumSaturatedRings", "MaxPartialCharge", "MinPartialCharge", "NumValenceElectrons",
        "frag_logp", "xlogp_pubchem", "n_fragments", "formal_charge_total", "has_metal", "organic", "pubchem_mw")] \
       + [c for c in F.columns if c.startswith("el_")]
FP = [c for c in F.columns if c.startswith("maccs_") or c.startswith("morgan_")]
for c in DESC + FP:
    x[c] = pd.to_numeric(x[c], errors="coerce")
SETS = {"nostruct": CATS + NUM, "desc": CATS + NUM + DESC, "full": CATS + NUM + DESC + FP}
P = dict(learning_rate=0.05, num_leaves=127, min_data_in_leaf=30, feature_fraction=0.5, bagging_fraction=0.8,
         bagging_freq=1, lambda_l2=2.0, cat_smooth=10, max_cat_to_onehot=8, verbose=-1, num_threads=28)
NR = 1000
cz = x.cens.map({"exact": 0, "right": 1, "left": 2}).values
jobs = [("chemical", x.parent_group.values, ["nostruct", "desc", "full", "full_tobit"]),
        ("scaffold", x.scaffold.values, ["nostruct", "full", "full_tobit"])]
rows, oof = [], {}
for sname, groups, models in jobs:
    pred = {m: np.full(len(x), np.nan) for m in models}
    for k, (itr, ite) in enumerate(GroupKFold(5).split(x, groups=groups)):
        t0 = time.time()
        tr, te = to_codes(x.iloc[itr], x.iloc[ite], cats=CATS)
        ex = cz[itr] == 0
        for m in models:
            feats = SETS[m.replace("_tobit", "")]
            if m.endswith("_tobit"):
                mod, init = fit_tobit_lgb(tr[feats], tr.log10_mgL.values, cz[itr], P, NR, CATS, sigma=0.8)
                pred[m][ite] = mod.predict(te[feats]) + init
            else:
                mod = lgb.train(dict(P, objective="regression"),
                                lgb.Dataset(tr[feats][ex], tr.log10_mgL[ex], categorical_feature=CATS), NR)
                pred[m][ite] = mod.predict(te[feats])
        print(sname, k, f"{time.time()-t0:.0f}s", flush=True)
    y = x.log10_mgL.values; ex = cz == 0
    fish = ex & (x.sp_group1.values == "Fish")
    for m, p in pred.items():
        mm = metrics(y[ex], p[ex]); mf = metrics(y[fish], p[fish])
        rows.append(dict(split=sname, model=m, rmse=mm["rmse"], r2=mm["r2"], mae=mm["mae"], n=mm["n"],
                         within_10x=float(np.mean(np.abs(p[ex] - y[ex]) <= 1)),
                         median_fold_error=float(10 ** np.median(np.abs(p[ex] - y[ex]))),
                         rmse_fish=mf["rmse"], r2_fish=mf["r2"],
                         right_cons=float(np.mean(p[cz == 1] >= y[cz == 1])),
                         bias=float(np.mean(p[ex] - y[ex]))))
        oof[f"{sname}__{m}"] = p
    print(pd.DataFrame(rows).round(3).to_string(), flush=True)
res = pd.DataFrame(rows); res.to_csv(R / "E08_new_chemical_structure.csv", index=False)
pd.DataFrame(oof).assign(y=x.log10_mgL.values, cens=x.cens.values, test_cas=x.test_cas.values,
                         parent_group=x.parent_group.values, scaffold=x.scaffold.values, sp=x.sp_group1.values,
                         species=x.species_number.values, ref=x.reference_number.values,
                         log_dur=x.log_dur.values).to_parquet(R / "E08_oof.parquet")
