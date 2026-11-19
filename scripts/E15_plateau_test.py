"""E15: has the lethality-time curve flattened by the end of the test?

The classic concept of an incipient (asymptotic) lethal level implies that the LC50-time curve eventually becomes
parallel to the time axis. If that plateau is normally reached inside the standard window, a fixed 96-h endpoint is
adequate; if not, the acute endpoint itself is not time-stationary.

Per within-study series with >= 4 distinct durations: fit the series' own LC50 against log10 duration, split the
durations at the median, and compare the late-half slope with the early-half slope. Pooled tests cluster by study.
Also: the share of series with a strictly positive upper limit on the late slope,
by the last observed duration, and the pooled late-half slope for series that extend beyond 96 h and beyond 14 d.
"""
import sys
import numpy as np, pandas as pd, statsmodels.formula.api as smf
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from common import load_lc50, R

# A slope in log10 concentration per log10 hour. This tolerance removes round-off
# when comparing a fitted slope or an interval endpoint with mathematical zero.
ZERO_TOL = 1e-12

x = load_lc50(min_h=1, max_h=24 * 60, include_censored=False)
skey = ["reference_number", "test_cas", "species_number", "organism_lifestage", "exposure_type", "media_type",
        "conc1_type", "chem_analysis_method"]
x["sid"] = x[skey].astype(str).agg("|".join, axis=1) + "|" + x.temp_C.round().astype(str)
a = x.groupby(["sid", "log_dur"], as_index=False).agg(y=("log10_mgL", "mean"), dur_h=("dur_h", "first"),
                                                      ref=("reference_number", "first"), sp=("sp_group1", "first"),
                                                      cas=("test_cas", "first"))
n = a.groupby("sid").size()
rows = []
for sid in n.index[n >= 4]:
    d = a[a.sid == sid].sort_values("log_dur")
    t, y = d.log_dur.values, d.y.values
    h = len(d) // 2
    e_sl = np.polyfit(t[:h + 1], y[:h + 1], 1)[0]
    l_sl = np.polyfit(t[h - 1:], y[h - 1:], 1)[0]
    # residual spread of the late-half fit gives a crude per-series uncertainty for the late slope
    tt, yy = t[h - 1:], y[h - 1:]
    fit = np.polyval(np.polyfit(tt, yy, 1), tt)
    dof = max(len(tt) - 2, 1)
    s2 = float(np.sum((yy - fit) ** 2) / dof)
    sxx = float(np.sum((tt - tt.mean()) ** 2))
    se_l = float(np.sqrt(s2 / sxx)) if sxx > 0 else np.nan
    if np.ptp(y[:h + 1]) == 0 or abs(e_sl) <= ZERO_TOL:
        e_sl = 0.0
    if np.ptp(yy) == 0:
        l_sl, se_l = 0.0, 0.0
    elif abs(l_sl) <= ZERO_TOL:
        l_sl = 0.0
    rows.append(dict(sid=sid, ref=d.ref.iloc[0], sp=d.sp.iloc[0], cas=d.cas.iloc[0], n_dur=len(d),
                     t_last_h=float(d.dur_h.max()), t_first_h=float(d.dur_h.min()),
                     early_slope=e_sl, late_slope=l_sl, late_minus_early=l_sl - e_sl, se_late=se_l))
S = pd.DataFrame(rows)
S["plateau"] = (S.late_slope + 1.96 * S.se_late > ZERO_TOL) & S.se_late.notna()
S.to_csv(R / "E15_plateau_series.csv", index=False)
cl = lambda d: dict(cov_type="cluster", cov_kwds={"groups": pd.factorize(d.ref)[0]})
out = []
for lab, d in [("all series (>=4 durations)", S), ("last duration > 96 h", S[S.t_last_h > 97]),
               ("last duration > 14 d", S[S.t_last_h > 336]), ("last duration <= 96 h", S[S.t_last_h <= 97])]:
    if len(d) < 30:
        continue
    f1 = smf.ols("late_slope ~ 1", d).fit(**cl(d))
    f2 = smf.ols("late_minus_early ~ 1", d).fit(**cl(d))
    out.append(dict(subset=lab, n_series=len(d), refs=d.ref.nunique(),
                    late_slope=f1.params["Intercept"], late_slope_se=f1.bse["Intercept"],
                    late_minus_early=f2.params["Intercept"], late_minus_early_se=f2.bse["Intercept"],
                    share_plateau=float(d.plateau.mean()),
                    share_late_slope_negative=float((d.late_slope < 0).mean())))
O = pd.DataFrame(out)
O.to_csv(R / "E15_plateau_summary.csv", index=False)
print(O.round(4).to_string())
by = S.groupby("sp").agg(n=("late_slope", "size"), late=("late_slope", "median"),
                         late_minus_early=("late_minus_early", "median"), plateau=("plateau", "mean"))
by[by.n >= 30].to_csv(R / "E15_plateau_by_taxon.csv")
print(by[by.n >= 30].round(3).to_string())
# how far past the last observed duration would the curve still be falling? pooled late slope by window
b = pd.cut(S.t_last_h / 24, [0, 2, 4, 7, 14, 28, 60])
print(S.groupby(b, observed=True).agg(n=("late_slope", "size"), late_slope=("late_slope", "median"),
                                      plateau=("plateau", "mean")).round(3).to_string())
