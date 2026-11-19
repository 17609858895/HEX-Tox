"""E02: acute-to-prolonged lethality extrapolation for chemicals whose prolonged (>96 h) LC50s are hidden.

Design: GroupKFold over chemicals that have >96 h LC50s. Test = >96 h exact rows of held-out chemicals.
Training keeps the held-out chemicals' <=96 h rows (their acute information) and all rows of other chemicals.
"""
import sys, time, math
import numpy as np, pandas as pd, lightgbm as lgb, torch
from sklearn.model_selection import GroupKFold
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from common import load_lc50, to_codes, metrics, CAT, R
from tcnet import fit_tcnet, predict_tcnet

torch.set_num_threads(int(sys.argv[1]) if len(sys.argv) > 1 else 12)
x = load_lc50(min_h=1, max_h=24 * 60, include_censored=True)
x["sid"] = x[["reference_number", "test_cas", "species_number", "organism_lifestage", "exposure_type",
              "media_type"]].astype(str).agg("|".join, axis=1) + "|" + x.temp_C.round().astype(str)
NUMS = ["temp_C", "pH", "log_hardness", "pub_year", "active_ingredient"]
FEAT = CAT + ["log_dur"] + NUMS
LGB_P = dict(objective="regression", learning_rate=0.05, num_leaves=127, min_data_in_leaf=20, feature_fraction=0.8,
             bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, cat_smooth=10, max_cat_to_onehot=8, verbose=-1,
             num_threads=8)
long = (x.dur_h > 97).values
chem_long = x.loc[long, "test_cas"].unique()
gkf = GroupKFold(5)
chem_fold = {}
for k, (_, ite) in enumerate(gkf.split(chem_long, groups=chem_long)):
    for c in chem_long[ite]:
        chem_fold[c] = k
x["fold"] = x.test_cas.map(chem_fold).fillna(-1).astype(int)

models = ["lgb_free", "tc_free", "tc_haber", "tc_curve", "tc_curve_nocens"]
pred = {m: np.full(len(x), np.nan) for m in models}
for k in range(5):
    t0 = time.time()
    te_mask = long & (x.fold.values == k)
    tr = x[~te_mask]; te = x[te_mask]
    a, b = to_codes(tr[tr.cens == "exact"], te)
    m = lgb.train(LGB_P, lgb.Dataset(a[FEAT], a.log10_mgL, categorical_feature=CAT), 1200)
    pred["lgb_free"][te_mask] = m.predict(b[FEAT])
    for head, mname, uc in [("free", "tc_free", True), ("haber", "tc_haber", True), ("curve", "tc_curve", True),
                            ("curve", "tc_curve_nocens", False)]:
        mod, enc = fit_tcnet(tr, CAT, NUMS, head=head, epochs=40, use_cens=uc, seed=k)
        pred[mname][te_mask] = predict_tcnet(mod, enc, te)
    print("fold", k, f"{time.time()-t0:.0f}s", flush=True)

# same-series baselines on test rows: last <=96h value in series; Haber extrapolation with global within-series slope
short = x[(x.dur_h <= 96) & (x.cens == "exact")]
last = short.sort_values("dur_h").groupby("sid").agg(y_last=("log10_mgL", "last"), t_last=("log_dur", "last"))
x = x.join(last, on="sid")
pred["persist_series"] = np.where(long, x.y_last.values, np.nan)
pred["haber_series_-0.43"] = np.where(long, x.y_last.values - 0.43 * (x.log_dur.values - x.t_last.values), np.nan)

ev = long & (x.cens == "exact").values & (x.fold.values >= 0)
has_series = ev & x.y_last.notna().values
bins = pd.cut(x.dur_h / 24, [4, 7, 14, 28, 60])
rows = []
for name, p in pred.items():
    for subset, msk in [("all_long", ev), ("with_series_acute", has_series)]:
        mk = msk & ~np.isnan(p)
        mm = metrics(x.log10_mgL.values[mk], p[mk])
        rows.append(dict(model=name, subset=subset, **mm, bias=float(np.mean(p[mk] - x.log10_mgL.values[mk]))))
        for bn in bins.cat.categories:
            mb = mk & (bins == bn).values
            if mb.sum() > 20:
                rows.append(dict(model=name, subset=f"{subset}|{bn}", **metrics(x.log10_mgL.values[mb], p[mb]),
                                 bias=float(np.mean(p[mb] - x.log10_mgL.values[mb]))))
res = pd.DataFrame(rows)
res.to_csv(R / "E02_duration_extrapolation.csv", index=False)
pd.DataFrame(pred).assign(y=x.log10_mgL.values, dur_h=x.dur_h.values, ev=ev, has_series=has_series,
                          test_cas=x.test_cas.values, sp=x.sp_group1.values).to_parquet(R / "E02_pred.parquet")
print(res[res.subset.isin(["all_long", "with_series_acute"])].round(3).to_string())
print(res[res.subset.str.startswith("with_series_acute|")].round(3).to_string())
