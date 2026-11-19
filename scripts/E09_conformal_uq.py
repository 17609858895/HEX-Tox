"""E09: group-conformal prediction intervals under shift, and propagation to the 28-d hidden hazard factor.

Inputs: out-of-fold predictions from grouped CV (E01 reference/species splits, E08 chemical split, E05 contrasts).
Method: normalized split-conformal at the group level. A difficulty model sigma(x) (LightGBM on |residual|) is fitted
with grouped cross-fitting; scores s = |y - p| / sigma. Calibration and test sets are disjoint sets of groups
(50/50, repeated 20 times). Reported: marginal coverage, mean width, subgroup coverage. For right-censored rows the
necessary condition "upper limit >= bound" is reported (the true value above the bound is unknown).
HC5 propagation: contrast errors decomposed into a chemical-shared and an idiosyncratic component (method of moments
on held-out-study residuals); draws plus species bootstrap give per-chemical intervals for HC5_96h / HC5_28d.
"""
import sys
import numpy as np, pandas as pd, lightgbm as lgb
from scipy import stats, optimize
from sklearn.model_selection import GroupKFold
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from common import R, D

LP = dict(objective="regression", learning_rate=0.05, num_leaves=31, min_data_in_leaf=50, feature_fraction=0.8,
          verbose=-1, num_threads=24)
rng = np.random.default_rng(0)


def sigma_model(df, resid, groups, feats, cats):
    s = np.full(len(df), np.nan)
    X = df[feats].copy()
    for c in cats:
        X[c] = X[c].astype("category")
    for itr, ite in GroupKFold(5).split(X, groups=groups):
        m = lgb.train(LP, lgb.Dataset(X.iloc[itr], np.abs(resid[itr]), categorical_feature=cats), 300)
        s[ite] = m.predict(X.iloc[ite])
    return np.clip(s, 0.05, None)


def group_conformal(y, p, sig, groups, cens, alpha_list=(0.2, 0.1), reps=20, sub=None):
    out = []
    ug = np.unique(groups)
    ex = cens == 0
    rc = cens == 1
    for alpha in alpha_list:
        for mode, sc in [("constant", np.ones_like(sig)), ("normalized", sig)]:
            cov, wid, rcons, subcov = [], [], [], {}
            for r in range(reps):
                cal_g = rng.choice(ug, len(ug) // 2, replace=False)
                cal = np.isin(groups, cal_g)
                te = ~cal
                s = np.abs(y - p) / sc
                n = (cal & ex).sum()
                q = np.quantile(s[cal & ex], min(1, np.ceil((n + 1) * (1 - alpha)) / n))
                lo, hi = p - q * sc, p + q * sc
                m = te & ex
                cov.append(np.mean((y[m] >= lo[m]) & (y[m] <= hi[m])))
                wid.append(np.mean(hi[m] - lo[m]))
                rcons.append(np.mean(hi[te & rc] >= y[te & rc]) if (te & rc).any() else np.nan)
                if sub is not None:
                    for g in np.unique(sub):
                        mg = m & (sub == g)
                        if mg.sum() > 200:
                            subcov.setdefault(g, []).append(np.mean((y[mg] >= lo[mg]) & (y[mg] <= hi[mg])))
            rec = dict(nominal=1 - alpha, mode=mode, coverage=np.mean(cov), coverage_sd=np.std(cov), width=np.mean(wid),
                       right_cens_upper_ok=np.nanmean(rcons))
            for g, v in subcov.items():
                rec[f"cov[{g}]"] = np.mean(v)
            out.append(rec)
    return out


rows = []
# ---- level predictions: unseen studies / species (E01), unseen chemicals (E08)
o1 = pd.read_parquet(R / "E01_oof.parquet")
o1["cz"] = o1.cens.map({"exact": 0, "right": 1, "left": 2}).values
feat1 = ["sp_group", "chem_group"]
for split, grp, model in [("reference", "ref", "tobit"), ("species", "species", "tobit")]:
    p = o1[f"{split}__{model}"].values
    y = o1.log10_mgL.values
    sig = sigma_model(o1, np.where(o1.cz == 0, y - p, 0.0), o1[grp].values, feat1, feat1)
    for rec in group_conformal(y, p, sig, o1[grp].values, o1.cz.values, sub=o1.sp_group.values):
        rows.append(dict(target="level", shift=f"unseen {split}", model=model, **rec))
o8 = pd.read_parquet(R / "E08_oof.parquet")
o8["cz"] = o8.cens.map({"exact": 0, "right": 1, "left": 2}).values
F = pd.read_parquet(D / "chem_features.parquet")[["test_cas", "MolLogP", "MolWt", "has_metal", "TPSA"]]
o8 = o8.merge(F, on="test_cas", how="left")
o8["sp"] = o8.sp.astype(str)
for model in ["full", "full_tobit"]:
    p = o8[f"chemical__{model}"].values
    y = o8.y.values
    sig = sigma_model(o8, np.where(o8.cz == 0, y - p, 0.0), o8.parent_group.values,
                      ["sp", "MolLogP", "MolWt", "has_metal", "TPSA", "log_dur"], ["sp"])
    for rec in group_conformal(y, p, sig, o8.parent_group.values, o8.cz.values, sub=o8.sp.values):
        rows.append(dict(target="level", shift="unseen chemical", model=model, **rec))
# ---- contrasts (E05, held-out studies)
o5 = pd.read_parquet(R / "E05_pairs.parquet")
o5 = o5[o5.fold >= 0].reset_index(drop=True)
o5["adt"] = o5.dlogt.abs()
o5["adT"] = o5.dT.abs().fillna(0)
for typ in ["duration", "temperature"]:
    q = o5[o5.type == typ].reset_index(drop=True)
    sig = sigma_model(q, q.dy.values - q.p_cal.values, q.ref.values, ["sp", "adt", "adT", "dur_i"], ["sp"])
    for rec in group_conformal(q.dy.values, q.p_cal.values, sig, q.ref.values, np.zeros(len(q), int), sub=q.sp.values):
        rows.append(dict(target=f"contrast:{typ}", shift="unseen study", model="CAL", **rec))
res = pd.DataFrame(rows)
res.to_csv(R / "E09_conformal_coverage.csv", index=False)
pd.set_option("display.width", 250)
print(res[["target", "shift", "model", "nominal", "mode", "coverage", "coverage_sd", "width",
           "right_cens_upper_ok"]].round(3).to_string())
sub = res.filter(like="cov[")
print(pd.concat([res[["target", "shift", "model", "mode", "nominal"]], sub], axis=1).round(3).to_string())

# ---- HC5 propagation for the 28-d hidden hazard factor
d = o5[(o5.type == "duration") & (o5.dur_j > 96)].copy()
d["e"] = d.dy - d.p_cal
g = d.groupby("test_cas").agg(m=("e", "mean"), n=("e", "size"), v=("e", "var")).dropna()
sigma2 = np.average(g.v, weights=g.n - 1)
tau2 = max(0.0, np.average(g.m ** 2 - sigma2 / g.n, weights=g.n))
print(f"28-d contrast error: total var {d.e.var():.3f}, within-chemical {sigma2:.3f}, chemical-shared tau2 {tau2:.3f}")
s = pd.read_parquet(R / "E07_standardized_records.parquet")


def species_vals(df, col):
    out = []
    for (c, sp), z in df.groupby(["test_cas", "species_number"]):
        e = z[z.cens == "exact"]
        if len(e):
            out.append((c, sp, e[col].mean(), 0))
        else:
            r = z[z.cens == "right"]
            if len(r):
                out.append((c, sp, r[col].max(), 1))
    return pd.DataFrame(out, columns=["test_cas", "species", "y", "rc"])


def hc5(y, rc):
    y, rc = np.asarray(y), np.asarray(rc).astype(bool)
    f = lambda p: -(stats.norm.logpdf(y[~rc], p[0], np.exp(p[1])).sum()
                    + stats.norm.logsf(y[rc], p[0], np.exp(p[1])).sum())
    r = optimize.minimize(f, [y.mean(), np.log(max(y.std(), 0.2))], method="Nelder-Mead")
    sd = np.exp(r.x[1])
    return r.x[0] + stats.norm.ppf(0.05) * sd if (r.success and sd < 3) else np.nan


S96, S28 = species_vals(s, "y96"), species_vals(s, "y28d")
H = []
eff = np.sqrt(sigma2)
for c, a in S96.groupby("test_cas"):
    if len(a) < 8:
        continue
    b = S28[S28.test_cas == c].set_index("species").loc[a.species]
    base = hc5(a.y, a.rc)
    draws = []
    for _ in range(200):
        idx = rng.integers(0, len(a), len(a))
        e = rng.normal(0, np.sqrt(tau2)) + rng.normal(0, eff, len(a))
        h96 = hc5(a.y.values[idx], a.rc.values[idx])
        h28 = hc5(b.y.values[idx] + e[idx], b.rc.values[idx])
        draws.append(h96 - h28)
    draws = np.array(draws)
    draws = draws[np.isfinite(draws)]
    if len(draws) < 100 or not np.isfinite(base):
        continue
    H.append(dict(test_cas=c, n_species=len(a), factor_med=10 ** np.median(draws),
                  factor_lo=10 ** np.percentile(draws, 5), factor_hi=10 ** np.percentile(draws, 95)))
H = pd.DataFrame(H)
H.to_csv(R / "E09_hc5_factor_intervals.csv", index=False)
print("chemicals:", len(H))
print("median factor:", round(H.factor_med.median(), 2),
      "| share with 90% lower bound > 1:", round((H.factor_lo > 1).mean(), 3),
      "| > 1.5:", round((H.factor_lo > 1.5).mean(), 3), "| > 2:", round((H.factor_lo > 2).mean(), 3))
print("median interval width (fold):", round((H.factor_hi / H.factor_lo).median(), 2))
