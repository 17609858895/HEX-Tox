"""E05b: discriminating ablation — which pairs should the contrast learner see?
 within  : pairs inside the same study series (CAL)
 between : pairs of the same chemical x species from different studies (pairwise-difference learning, PDL-style)
 all     : union
Evaluation: held-out-reference within-study pairs (duration / temperature contrasts); slopes and RMSE.
Also reports the same-anchor persistence baseline (dy_hat = 0).
"""
import sys, time
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import GroupKFold
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from common import load_lc50, to_codes, CAT, R

x = load_lc50(min_h=1, max_h=24 * 60, include_censored=False)
skey = ["reference_number", "test_cas", "species_number", "organism_lifestage", "exposure_type", "media_type",
        "conc1_type", "chem_analysis_method"]
x["sid"] = x[skey].astype(str).agg("|".join, axis=1)
x["ckey"] = x.sid + "|" + x.dur_h.round(2).astype(str) + "|" + x.temp_C.round(1).astype(str)
agg = {c: "first" for c in x.columns if c not in ("log10_mgL", "ckey")}; agg["log10_mgL"] = "mean"
u = x.groupby("ckey").agg(agg).reset_index(drop=True)
rng = np.random.default_rng(0)

def within_pairs():
    out = []
    for sid, d in u.groupby("sid"):
        idx = d.index.values
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                out.append((idx[a], idx[b]))
    return np.array(out)

def between_pairs(max_per=30):
    out = []
    for key, d in u.groupby(["test_cas", "species_number"]):
        if d.sid.nunique() < 2: continue
        idx = d.index.values; sids = d.sid.values
        cand = [(idx[a], idx[b]) for a in range(len(idx)) for b in range(a + 1, len(idx)) if sids[a] != sids[b]]
        if len(cand) > max_per:
            cand = [cand[k] for k in rng.choice(len(cand), max_per, replace=False)]
        out += cand
    return np.array(out)

def to_df(arr, src):
    P = pd.DataFrame(arr, columns=["i", "j"])
    P["dy"] = u.log10_mgL.values[P.j] - u.log10_mgL.values[P.i]
    P["dlogt"] = u.log_dur.values[P.j] - u.log_dur.values[P.i]
    P["dT"] = u.temp_C.values[P.j] - u.temp_C.values[P.i]
    P["ref_i"] = u.reference_number.values[P.i]; P["ref_j"] = u.reference_number.values[P.j]
    P["src"] = src
    return P

W = to_df(within_pairs(), "within"); B = to_df(between_pairs(), "between")
W["type"] = np.select([(W.dT.abs().fillna(0) < 0.5) & (W.dlogt.abs() > 0.01),
                       (W.dlogt.abs() <= 0.01) & (W.dT.abs() >= 2)], ["duration", "temperature"], "other")
print("within pairs", len(W), "between pairs", len(B))
BASE = CAT + ["pub_year", "active_ingredient"]
PF = ["log_dur", "temp_C", "pH", "log_hardness"]
FEATS = BASE + [c + "_i" for c in PF] + ["d_" + c for c in PF]
def pf(P, flip=False):
    a, b = (P.j.values, P.i.values) if flip else (P.i.values, P.j.values)
    F = u.loc[a, BASE].reset_index(drop=True).copy()
    for c in PF:
        F[c + "_i"] = u[c].values[a]; F["d_" + c] = u[c].values[b] - u[c].values[a]
    return F
LP = dict(objective="regression", learning_rate=0.05, num_leaves=63, min_data_in_leaf=20, feature_fraction=0.8,
          bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, cat_smooth=10, max_cat_to_onehot=8, verbose=-1,
          num_threads=28)
pred = {k: np.full(len(W), np.nan) for k in ["within", "between", "all", "persistence"]}
W["fold"] = -1
refs_all = x.reference_number.values
for k, (itr, ite) in enumerate(GroupKFold(5).split(x, groups=refs_all)):
    t0 = time.time()
    test_refs = set(np.unique(refs_all[ite]))
    te = W.ref_i.isin(test_refs).values; W.loc[te, "fold"] = k
    trW = W[~te & (W.type != "other")]
    trB = B[~B.ref_i.isin(test_refs) & ~B.ref_j.isin(test_refs)]
    Fte = pf(W[te])
    for nm, Ptr in [("within", trW), ("between", trB), ("all", pd.concat([trW, trB]))]:
        Ftr = pd.concat([pf(Ptr), pf(Ptr, True)], ignore_index=True)
        ytr = np.concatenate([Ptr.dy.values, -Ptr.dy.values])
        A, Bt = to_codes(Ftr, Fte)
        m = lgb.train(LP, lgb.Dataset(A[FEATS], ytr, categorical_feature=CAT), 800)
        pred[nm][te] = m.predict(Bt[FEATS])
    pred["persistence"][te] = 0.0
    print("fold", k, f"{time.time()-t0:.0f}s", flush=True)
rows = []
ev = W.fold.values >= 0
for nm, p in pred.items():
    for typ in ["duration", "temperature"]:
        mk = ev & (W.type.values == typ)
        d = W[mk]; pp = p[mk]
        slope_var = "dlogt" if typ == "duration" else "dT"
        rows.append(dict(model=nm, type=typ, n=int(mk.sum()), rmse=float(np.sqrt(np.mean((pp - d.dy) ** 2))),
                         slope=float(np.sum(pp * d[slope_var]) / np.sum(d[slope_var] ** 2)),
                         obs_slope=float(np.sum(d.dy * d[slope_var]) / np.sum(d[slope_var] ** 2))))
res = pd.DataFrame(rows); res.to_csv(R / "E05b_pair_source_ablation.csv", index=False)
print(res.round(4).to_string())
