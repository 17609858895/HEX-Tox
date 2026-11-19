"""E06: semi-synthetic test on the real ECOTOX design (chemicals, species, studies, durations, temperatures).

Truth: y = a_chem + s_species + u_study + beta_g*(T-20) + n_g*(logt - log48) + eps
       u_study = gT*(Tbar_study-20) + gt*(logtbar_study - log48) + N(0, 0.45)   (between-study confounding)
       right-censoring at study-level limit concentrations (limit tests) mimicking the observed censoring share.
Estimators: pooled LightGBM (exact only) -> implied slopes via within-pair differencing on held-out studies;
            contrast learner (CAL) trained on within-study pairs; level bias for censored chemicals: exact-only vs Tobit.
"""
import sys, time
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import GroupKFold
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from common import load_lc50, to_codes, CAT, NUM, R
from tobit_lgb import fit_tobit_lgb

LP = dict(learning_rate=0.06, num_leaves=63, min_data_in_leaf=20, feature_fraction=0.8, bagging_fraction=0.8,
          bagging_freq=1, lambda_l2=1.0, cat_smooth=10, max_cat_to_onehot=8, verbose=-1, num_threads=28)
x0 = load_lc50(min_h=1, max_h=24 * 60, include_censored=True)
x0 = x0.dropna(subset=["temp_C"]).reset_index(drop=True)
skey = ["reference_number", "test_cas", "species_number", "organism_lifestage", "exposure_type", "media_type",
        "conc1_type", "chem_analysis_method"]
x0["sid"] = x0[skey].astype(str).agg("|".join, axis=1)
L48 = np.log10(48)

def simulate(gT, gt, seed):
    rng = np.random.default_rng(seed)
    x = x0.copy()
    chems, sps, refs = x.test_cas.unique(), x.species_number.unique(), x.reference_number.unique()
    a = dict(zip(chems, rng.normal(0.3, 1.2, len(chems))))
    s = dict(zip(sps, rng.normal(0, 0.6, len(sps))))
    grp_beta = {g: rng.normal(-0.019, 0.010) for g in x.chem_group.unique()}
    tax_n = {g: -abs(rng.normal(0.5, 0.2)) for g in x.sp_group1.unique()}
    Tbar = x.groupby("reference_number").temp_C.transform("mean")
    tbar = x.groupby("reference_number").log_dur.transform("mean")
    ustudy = dict(zip(refs, rng.normal(0, 0.45, len(refs))))
    u = x.reference_number.map(ustudy) + gT * (Tbar - 20) + gt * (tbar - L48)
    beta = x.chem_group.map(grp_beta); n = x.sp_group1.map(tax_n)
    x["beta_true"], x["n_true"] = beta, n
    y = x.test_cas.map(a) + x.species_number.map(s) + u + beta * (x.temp_C - 20) + n * (x.log_dur - L48) \
        + rng.normal(0, 0.25, len(x))
    # limit tests: study-level upper limit = chemical median + offset; censor y above limit for 'limit' studies
    lim_study = dict(zip(refs, rng.random(len(refs)) < 0.35))
    limit = x.test_cas.map(a) + 0.9 + rng.normal(0, 0.3, len(x))
    cens = x.reference_number.map(lim_study).values & (y.values > limit.values)
    x["y_true"] = y.values
    x["log10_mgL"] = np.where(cens, limit.values, y.values)
    x["cens"] = np.where(cens, "right", "exact")
    return x

def pairs_of(x):
    u = x[x.cens == "exact"]
    out = []
    for sid, d in u.groupby("sid"):
        if len(d) < 2: continue
        idx = d.index.values
        for i in range(len(idx)):
            for j in range(i + 1, len(idx)):
                out.append((idx[i], idx[j]))
    P = pd.DataFrame(out, columns=["i", "j"])
    P["dlogt"] = x.log_dur.values[P.j] - x.log_dur.values[P.i]
    P["dT"] = x.temp_C.values[P.j] - x.temp_C.values[P.i]
    P["dy"] = x.log10_mgL.values[P.j] - x.log10_mgL.values[P.i]
    P["ref"] = x.reference_number.values[P.i]
    return P

BASE = CAT + ["pub_year", "active_ingredient"]
PF = ["log_dur", "temp_C"]
def pair_frame(x, P, flip=False):
    a, b = (P.j.values, P.i.values) if flip else (P.i.values, P.j.values)
    F = x.loc[a, BASE].reset_index(drop=True).copy()
    for c in PF:
        F[c + "_i"] = x[c].values[a]; F["d_" + c] = x[c].values[b] - x[c].values[a]
    return F
FEATS = BASE + [c + "_i" for c in PF] + ["d_" + c for c in PF]

rows = []
for gT, gt in [(0.0, 0.0), (0.02, 0.15), (0.04, 0.30)]:
    for seed in range(3):
        t0 = time.time()
        x = simulate(gT, gt, seed)
        P = pairs_of(x)
        tr_ref = set(pd.Series(x.reference_number.unique()).sample(frac=0.8, random_state=seed))
        trm = x.reference_number.isin(tr_ref).values
        te_p = ~P.ref.isin(tr_ref).values
        xt, xs = to_codes(x[trm], x)
        ex = (x.cens.values[trm] == "exact")
        m = lgb.train(dict(LP, objective="regression"),
                      lgb.Dataset(xt[CAT + NUM][ex], xt.log10_mgL[ex], categorical_feature=CAT), 600)
        lev = m.predict(xs[CAT + NUM])
        pp = lev[P.j.values] - lev[P.i.values]
        Ptr = P[~te_p]
        Ftr = pd.concat([pair_frame(x, Ptr), pair_frame(x, Ptr, True)], ignore_index=True)
        ytr = np.concatenate([Ptr.dy.values, -Ptr.dy.values])
        A, B = to_codes(Ftr, pair_frame(x, P[te_p]))
        mc = lgb.train(dict(LP, objective="regression"), lgb.Dataset(A[FEATS], ytr, categorical_feature=CAT), 600)
        pc = mc.predict(B[FEATS])
        Pt = P[te_p].copy(); Pt["pooled"] = pp[te_p]; Pt["cal"] = pc
        Pt["true"] = (x.y_true.values[Pt.j] - x.y_true.values[Pt.i])
        dur = Pt[(Pt.dT.abs() < 0.5) & (Pt.dlogt.abs() > 0.01)]
        tem = Pt[(Pt.dlogt.abs() <= 0.01) & (Pt.dT.abs() >= 2)]
        sl = lambda d, c, v: float(np.sum(d[c] * d[v]) / np.sum(d[v] ** 2))
        # level bias on test-study exact rows of high-censoring chemicals: exact-only vs Tobit
        mt, init = fit_tobit_lgb(xt[CAT + NUM], xt.log10_mgL.values,
                                 (x.cens.values[trm] == "right").astype(int), LP, 600, CAT, sigma=0.6)
        levt = mt.predict(xs[CAT + NUM]) + init
        share = x.groupby("test_cas").cens.transform(lambda q: (q == "right").mean()).values
        hi = (~trm) & (share >= 0.3)
        rows.append(dict(gT=gT, gt=gt, seed=seed,
                         true_time_slope=sl(dur, "true", "dlogt"), pooled_time_slope=sl(dur, "pooled", "dlogt"),
                         cal_time_slope=sl(dur, "cal", "dlogt"),
                         true_temp_slope=sl(tem, "true", "dT"), pooled_temp_slope=sl(tem, "pooled", "dT"),
                         cal_temp_slope=sl(tem, "cal", "dT"),
                         lev_bias_exactonly_hicens=float(np.mean(lev[hi] - x.y_true.values[hi])),
                         lev_bias_tobit_hicens=float(np.mean(levt[hi] - x.y_true.values[hi])),
                         lev_bias_exactonly_all=float(np.mean(lev[~trm] - x.y_true.values[~trm])),
                         lev_bias_tobit_all=float(np.mean(levt[~trm] - x.y_true.values[~trm])),
                         cens_share=float((x.cens == "right").mean())))
        print(pd.DataFrame(rows).tail(1).round(4).to_string(), f"{time.time()-t0:.0f}s", flush=True)
res = pd.DataFrame(rows); res.to_csv(R / "E06_semisynthetic.csv", index=False)
print(res.groupby(["gT", "gt"]).mean().round(4).T.to_string())
