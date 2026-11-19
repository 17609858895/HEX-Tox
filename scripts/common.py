"""Shared loading / feature engineering for ECOTOX aquatic LC50 modelling."""
import numpy as np, pandas as pd
from pathlib import Path

PROJ = Path(__file__).resolve().parents[1]
D = PROJ / "data_derived"
R = PROJ / "results"

CAT = ["test_cas", "chem_group", "species_number", "genus", "family", "tax_order", "class", "phylum_division",
       "sp_group1", "organism_lifestage", "exposure_type", "media_type", "test_location", "conc1_type",
       "chem_analysis_method", "organism_source", "test_formulation"]
NUM = ["log_dur", "temp_C", "pH", "log_hardness", "pub_year", "active_ingredient"]


def load_lc50(min_h=1, max_h=24 * 60, effects=("MOR",), endpoints=("LC50",), include_censored=True):
    x = pd.read_parquet(D / "aquatic_acute_all.parquet")
    x = x[x.ep.isin(endpoints) & x.effect.fillna("").str.contains("|".join(effects))]
    x = x[(x.dur_h >= min_h) & (x.dur_h <= max_h)].copy()
    x = x[x.cens.isin(["exact", "right", "left"] if include_censored else ["exact"])]
    x["log_dur"] = np.log10(x.dur_h)
    x["log_hardness"] = np.log10(x.hardness.clip(lower=1))
    x["pub_year"] = pd.to_numeric(x.publication_year, errors="coerce")
    x["active_ingredient"] = x.active_ingredient.astype(float)
    x["sp_group1"] = x.sp_group.fillna("NA").str.split(";").str[0]
    x["media_type"] = x.media_type.fillna("NR").str.replace("/", "", regex=False)
    x["exposure_type"] = x.exposure_type.fillna("NR").str.replace("/", "", regex=False)
    x["chem_group"] = x.chem_group.fillna("NA").str.split(";").str[0]
    x["pair"] = x.test_cas + "_" + x.species_number
    for c in CAT:
        x[c] = x[c].fillna("NR").astype(str)
    x = x[np.isfinite(x.log10_mgL)]
    return x.reset_index(drop=True)


def to_codes(train, test, cats=CAT):
    """Integer-code categoricals using train categories only (unseen -> -1 / NaN)."""
    tr, te = train.copy(), test.copy()
    for c in cats:
        cats_ = pd.Index(tr[c].unique())
        tr[c] = pd.Categorical(tr[c], categories=cats_)
        te[c] = pd.Categorical(te[c], categories=cats_)
    return tr, te


def metrics(y, p):
    y, p = np.asarray(y), np.asarray(p)
    rmse = float(np.sqrt(np.mean((y - p) ** 2)))
    r2 = float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))
    mae = float(np.mean(np.abs(y - p)))
    return dict(rmse=rmse, r2=r2, mae=mae, n=len(y))
