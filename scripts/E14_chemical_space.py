"""E14: chemical-space map, applicability domain, and hazard-ranking consequences.

 1 t-SNE embedding of Morgan fingerprints (Jaccard) for the LC50 chemicals; per-chemical attributes:
   chemical class, record count, censoring share, CAL-predicted 96 h -> 28 d shift and +4 degC shift,
   the E07 hidden hazard factor, and the E08 held-out-chemical level error.
 2 applicability domain: nearest-neighbour Tanimoto similarity to the training chemicals of each held-out fold
   vs absolute level error and conformal interval width (binned curve).
 3 hazard-ranking consequences: Spearman/Kendall between 96-h and 28-d HC5 rankings, rank movement, and how the
   hidden hazard factor compares with a fixed default assessment factor (reported as the ratio needed per chemical).
"""
import sys
import numpy as np, pandas as pd
from scipy import stats
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from common import R, D

F = pd.read_parquet(D / "chem_features.parquet")
MORG = [c for c in F.columns if c.startswith("morgan_")]
o8 = pd.read_parquet(R / "E08_oof.parquet")
rec = pd.read_parquet(R / "E07_standardized_records.parquet")
H = pd.read_csv(R / "E07_hc5_hidden_hazard.csv", dtype={"test_cas": str})

keep = F[F.test_cas.isin(o8.test_cas.unique())].reset_index(drop=True)
X = keep[MORG].values.astype(bool)
print("chemicals embedded:", len(keep))
emb = TSNE(n_components=2, metric="jaccard", init="random", perplexity=30, learning_rate="auto",
           random_state=0).fit_transform(X)
keep["tsne1"], keep["tsne2"] = emb[:, 0], emb[:, 1]

# per-chemical attributes
err = o8.assign(ae=(o8["chemical__full"] - o8.y).abs()).groupby("test_cas").agg(
    n_records=("y", "size"), mae_heldout=("ae", "mean"), bias_heldout=("ae", "mean"))
shift = rec.assign(d28=rec.y96 - rec.y28d, d4=rec.y96 - rec.y96_p4).groupby("test_cas").agg(
    med_shift_28d=("d28", "median"), med_shift_p4=("d4", "median"),
    cens_share=("cens", lambda s: float((s == "right").mean())))
keep = keep.merge(err, on="test_cas", how="left").merge(shift, on="test_cas", how="left") \
           .merge(H[["test_cas", "hidden_factor_28d", "hidden_factor_p4", "n_species", "HC5_96_cens", "HC5_y28d"]],
                  on="test_cas", how="left")
cols = ["test_cas", "chemical_name"] if "chemical_name" in keep.columns else ["test_cas"]
keep[cols + ["chem_class", "parent_group", "scaffold", "tsne1", "tsne2", "MolLogP", "MolWt", "has_metal",
             "n_records", "mae_heldout", "cens_share", "med_shift_28d", "med_shift_p4", "hidden_factor_28d",
             "hidden_factor_p4", "n_species", "HC5_96_cens", "HC5_y28d"]].to_csv(R / "E14_chemical_space.csv",
                                                                                 index=False)

# ---------- 2 applicability domain
rows = []
pg = o8.drop_duplicates("test_cas")[["test_cas", "parent_group"]]
fp = keep.set_index("test_cas")[MORG]
groups = pg.set_index("test_cas").parent_group
ug = groups.unique()
rng = np.random.default_rng(0)
fold_of = {g: k for k, g in enumerate(ug[rng.permutation(len(ug))] if False else ug)}
# reproduce the 5 chemical folds of E08 (GroupKFold order is deterministic on the group array)
from sklearn.model_selection import GroupKFold
o8i = o8.reset_index(drop=True)
sim = np.full(len(o8i), np.nan)
for itr, ite in GroupKFold(5).split(o8i, groups=o8i.parent_group.values):
    tr_cas = o8i.test_cas.values[itr]
    te_cas = np.unique(o8i.test_cas.values[ite])
    Xtr = fp.reindex(np.unique(tr_cas)).dropna()
    Xte = fp.reindex(te_cas).dropna()
    if not len(Xtr) or not len(Xte):
        continue
    nn = NearestNeighbors(n_neighbors=1, metric="jaccard").fit(Xtr.values.astype(bool))
    dist, _ = nn.kneighbors(Xte.values.astype(bool))
    s = pd.Series(1 - dist[:, 0], index=Xte.index)
    m = np.isin(o8i.test_cas.values, s.index) & np.isin(np.arange(len(o8i)), ite)
    sim[m] = s.reindex(o8i.test_cas.values[m]).values
o8i["nn_tanimoto"] = sim
ex = o8i.cens.values == "exact"
o8i["ae"] = (o8i["chemical__full"] - o8i.y).abs()
b = pd.cut(o8i.nn_tanimoto, [0, .2, .3, .4, .5, .6, .8, 1.0])
ad = o8i[ex].groupby(b, observed=True).agg(n=("ae", "size"), chemicals=("test_cas", "nunique"),
                                           mae=("ae", "mean"),
                                           rmse=("ae", lambda s: float(np.sqrt(np.mean(s ** 2)))),
                                           within_10x=("ae", lambda s: float(np.mean(s <= 1))))
ad.to_csv(R / "E14_applicability_domain.csv")
print(ad.round(3).to_string())
o8i[["test_cas", "parent_group", "nn_tanimoto", "ae", "cens", "sp"]].to_parquet(R / "E14_ad_records.parquet")

# ---------- 3 hazard-ranking consequences
h = H.dropna(subset=["HC5_96_cens", "HC5_y28d"]).copy()
h = h[(h.hidden_factor_28d > 0.02) & (h.hidden_factor_28d < 50)]
r96 = h.HC5_96_cens.rank(); r28 = h.HC5_y28d.rank()
sp_ = stats.spearmanr(h.HC5_96_cens, h.HC5_y28d)
kd = stats.kendalltau(h.HC5_96_cens, h.HC5_y28d)
pct_move = (r28 / len(h) - r96 / len(h)) * 100
summary = dict(n_chemicals=len(h), spearman=float(sp_.statistic), kendall=float(kd.statistic),
               median_abs_percentile_move=float(np.median(np.abs(pct_move))),
               share_moving_gt10pct=float(np.mean(np.abs(pct_move) > 10)),
               median_factor=float(h.hidden_factor_28d.median()),
               share_factor_gt2=float((h.hidden_factor_28d > 2).mean()),
               share_factor_gt3=float((h.hidden_factor_28d > 3).mean()),
               share_factor_gt10=float((h.hidden_factor_28d > 10).mean()))
# how many chemicals are protected by a fixed default factor of 10 applied to the 96-h HC5
summary["share_exceeding_default_factor_10"] = float((h.hidden_factor_28d > 10).mean())
pd.DataFrame([summary]).to_csv(R / "E14_ranking_consequences.csv", index=False)
print(summary)
h[["test_cas", "chemical_name", "n_species", "HC5_96_cens", "HC5_y28d", "hidden_factor_28d",
   "hidden_factor_p4"]].assign(pct_move=pct_move.values).to_csv(R / "E14_hc5_rank_table.csv", index=False)
