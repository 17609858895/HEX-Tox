"""E07: consequences for hazard assessment.
 (1) censored vs exact-only species sensitivity distributions (SSD, log-normal MLE) -> HC5 at 96 h
 (2) contrast-learner (CAL) adjustment of each species' 96-h LC50 to prolonged exposure (28 d) and warming (+2/+4 degC)
     -> 'hidden hazard factor' = HC5_standard / HC5_realistic.
CAL is trained on all within-study pairs (exact values); predictions are pair-specific contrasts from the study anchor.
"""
import sys
import numpy as np, pandas as pd, lightgbm as lgb
from scipy import stats, optimize
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from common import load_lc50, to_codes, CAT, R

LP = dict(learning_rate=0.05, num_leaves=63, min_data_in_leaf=20, feature_fraction=0.8, bagging_fraction=0.8,
          bagging_freq=1, lambda_l2=1.0, cat_smooth=10, max_cat_to_onehot=8, verbose=-1, num_threads=24)
x = load_lc50(min_h=1, max_h=24 * 60, include_censored=True)
skey = ["reference_number", "test_cas", "species_number", "organism_lifestage", "exposure_type", "media_type",
        "conc1_type", "chem_analysis_method"]
x["sid"] = x[skey].astype(str).agg("|".join, axis=1)
BASE = CAT + ["pub_year", "active_ingredient"]
PF = ["log_dur", "temp_C", "pH", "log_hardness"]
FEATS = BASE + [c + "_i" for c in PF] + ["d_" + c for c in PF]

# ---- train CAL on all within-study exact pairs
ex = x[x.cens == "exact"].copy()
ex["ckey"] = ex.sid + "|" + ex.dur_h.round(2).astype(str) + "|" + ex.temp_C.round(1).astype(str)
agg = {c: "first" for c in ex.columns if c not in ("log10_mgL", "ckey")}; agg["log10_mgL"] = "mean"
u = ex.groupby("ckey").agg(agg).reset_index(drop=True)
pi, pj = [], []
for sid, d in u.groupby("sid"):
    idx = d.index.values
    for a in range(len(idx)):
        for b in range(a + 1, len(idx)):
            pi.append(idx[a]); pj.append(idx[b])
pi, pj = np.array(pi), np.array(pj)

def frame(a_idx, b_vals):
    F = u.loc[a_idx, BASE].reset_index(drop=True).copy() if b_vals is None else b_vals
    return F

def pf(src, a, cond_b):
    F = src.loc[a, BASE].reset_index(drop=True).copy()
    for c in PF:
        F[c + "_i"] = src[c].values[a]
        F["d_" + c] = cond_b[c] - src[c].values[a]
    return F

Ftr = pd.concat([pf(u, pi, {c: u[c].values[pj] for c in PF}), pf(u, pj, {c: u[c].values[pi] for c in PF})],
                ignore_index=True)
ytr = np.concatenate([u.log10_mgL.values[pj] - u.log10_mgL.values[pi], u.log10_mgL.values[pi] - u.log10_mgL.values[pj]])
cats_tr = {c: pd.Index(Ftr[c].astype(str).unique()) for c in CAT}
for c in CAT: Ftr[c] = pd.Categorical(Ftr[c].astype(str), categories=cats_tr[c])
mono = [(-1 if f == "d_log_dur" else 0) for f in FEATS]
cal = lgb.train(dict(LP, objective="regression", monotone_constraints=mono, monotone_constraints_method="advanced"),
                lgb.Dataset(Ftr[FEATS], ytr, categorical_feature=CAT), 800)
print("CAL trained on", len(ytr) // 2, "pairs")

def cal_delta(src, target):
    """predict contrast from each row's own condition to target conditions (dict of arrays/scalars)"""
    F = src[BASE].reset_index(drop=True).copy()
    for c in PF:
        F[c + "_i"] = src[c].values
        F["d_" + c] = (target.get(c, src[c].values) - src[c].values) if c in target else 0.0
    for c in CAT: F[c] = pd.Categorical(F[c].astype(str), categories=cats_tr[c])
    return cal.predict(F[FEATS])

# ---- standardize all 24-96 h records to 96 h (anchor + contrast), then to realistic scenarios
s = x[(x.dur_h >= 24) & (x.dur_h <= 96)].copy().reset_index(drop=True)
# unknown test temperatures: impute the taxon x medium median so every species enters every scenario SSD
s["temp_imputed"] = s.temp_C.isna()
med_T = s.groupby(["sp_group1", "media_type"]).temp_C.transform("median").fillna(s.temp_C.median())
s["temp_C"] = s.temp_C.fillna(med_T)
L96, L672 = np.log10(96), np.log10(672)
s["y96"] = s.log10_mgL + cal_delta(s, {"log_dur": L96})
s96 = s.assign(log_dur=L96)
s["y28d"] = s.y96 + cal_delta(s96, {"log_dur": L672})
s96 = s.assign(log_dur=L96)
for dT in (2, 4):
    s[f"y96_p{dT}"] = s.y96 + cal_delta(s96, {"temp_C": s.temp_C.values + dT})
s["y96_glob"] = s.log10_mgL - 0.43 * (L96 - s.log_dur)

# ---- species-level values per chemical (geometric mean of exact; censored species: max bound as right-censored)
def species_table(col):
    out = []
    for (c, sp), d in s.groupby(["test_cas", "species_number"]):
        e = d[d.cens == "exact"]
        if len(e):
            out.append((c, sp, e[col].mean(), 0))
        else:
            r = d[d.cens == "right"]
            if len(r): out.append((c, sp, r[col].max(), 1))
    return pd.DataFrame(out, columns=["test_cas", "species", "y", "rc"])

def fit_lognormal(y, rc):
    y, rc = np.asarray(y), np.asarray(rc).astype(bool)
    def nll(p):
        mu, ls = p; sd = np.exp(ls)
        return -(stats.norm.logpdf(y[~rc], mu, sd).sum() + stats.norm.logsf(y[rc], mu, sd).sum())
    p0 = [np.mean(y), np.log(max(np.std(y), 0.2))]
    r = optimize.minimize(nll, p0, method="Nelder-Mead")
    mu, sd = r.x[0], np.exp(r.x[1])
    if not r.success or sd > 3 or not np.isfinite(mu):
        return np.nan
    return mu + stats.norm.ppf(0.05) * sd

rows = []
T = {k: species_table(k) for k in ["y96", "y28d", "y96_p2", "y96_p4", "y96_glob", "log10_mgL"]}
base = T["y96"]
for c, d in base.groupby("test_cas"):
    if len(d) < 8: continue
    r = dict(test_cas=c, n_species=len(d), n_cens_species=int(d.rc.sum()))
    r["HC5_96_cens"] = fit_lognormal(d.y, d.rc)
    de = d[d.rc == 0]
    r["HC5_96_exactonly"] = fit_lognormal(de.y, de.rc) if len(de) >= 5 else np.nan
    for k in ["y28d", "y96_p2", "y96_p4"]:
        dk = T[k][T[k].test_cas == c].dropna()
        r[f"HC5_{k}"] = fit_lognormal(dk.y, dk.rc) if len(dk) >= 8 else np.nan
    rows.append(r)
H = pd.DataFrame(rows)
chem = pd.read_parquet(R.parent / "data_derived" / "chemicals.parquet").rename(columns={"cas_number": "test_cas"})
H = H.merge(chem[["test_cas", "chemical_name", "ecotox_group"]], on="test_cas", how="left")
H["log_ratio_cens_vs_exact"] = H.HC5_96_cens - H.HC5_96_exactonly
H["hidden_factor_28d"] = 10 ** (H.HC5_96_cens - H.HC5_y28d)
H["hidden_factor_p2"] = 10 ** (H.HC5_96_cens - H.HC5_y96_p2)
H["hidden_factor_p4"] = 10 ** (H.HC5_96_cens - H.HC5_y96_p4)
H.to_csv(R / "E07_hc5_hidden_hazard.csv", index=False)
s[["test_cas", "species_number", "sp_group1", "chem_group", "log_dur", "temp_C", "temp_imputed", "cens", "log10_mgL",
   "y96", "y28d", "y96_p2", "y96_p4"]].to_parquet(R / "E07_standardized_records.parquet")
print("chemicals with SSD (>=8 species):", len(H), " with censored species:", int((H.n_cens_species > 0).sum()))
print("HC5 censored - exact-only (log10), chemicals with censored species:")
print(H.loc[H.n_cens_species > 0, "log_ratio_cens_vs_exact"].describe().round(3))
for k in ["hidden_factor_28d", "hidden_factor_p2", "hidden_factor_p4"]:
    print(k, "by chemical group (median):"); print(H.groupby(H.ecotox_group.fillna("NA").str.split(";").str[0])[k].agg(["median", "size"]).sort_values("size", ascending=False).head(8).round(2))
print("hidden hazard factor (HC5_96h / HC5_28d):"); print(H.hidden_factor_28d.describe(percentiles=[.1, .25, .5, .75, .9]).round(2))
print("share > 2x:", (H.hidden_factor_28d > 2).mean().round(3), " >3x:", (H.hidden_factor_28d > 3).mean().round(3))
print("warming +2C factor:"); print(H.hidden_factor_p2.describe(percentiles=[.1, .5, .9]).round(3))
print("warming +4C factor:"); print(H.hidden_factor_p4.describe(percentiles=[.1, .5, .9]).round(3))
