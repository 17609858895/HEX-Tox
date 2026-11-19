"""Derive a standardized aquatic acute LC50/EC50 table (mass-based, mg/L) with censoring flags."""
import numpy as np, pandas as pd
from pathlib import Path
D = Path(__file__).resolve().parents[1] / "data_derived"

r = pd.read_parquet(D/"results_core.parquet")
t = pd.read_parquet(D/"tests_core.parquet")
med = pd.read_parquet(D/"media_core.parquet")
chem = pd.read_parquet(D/"chemicals.parquet")
sp = pd.read_parquet(D/"species.parquet")
ref = pd.read_parquet(D/"references.parquet")

r["ep"] = r.endpoint.str.replace("/", "", regex=False).str.replace("*", "", regex=False).str.strip()
r = r[r.ep.isin(["LC50", "EC50"])].copy()

# concentration value: prefer mean, else geometric mean of min/max if both present
def num(s):
    return pd.to_numeric(s.str.replace(",", "", regex=False).str.replace("*","",regex=False).str.strip(), errors="coerce")
r["v_mean"], r["v_min"], r["v_max"] = num(r.conc1_mean), num(r.conc1_min), num(r.conc1_max)
op = r.conc1_mean_op.fillna("").str.strip()
r["value"] = r.v_mean
rng = r.value.isna() & r.v_min.notna() & r.v_max.notna() & (r.v_min > 0) & (r.v_max > 0)
r.loc[rng, "value"] = np.sqrt(r.v_min[rng] * r.v_max[rng])
r["is_range"] = rng
r["op"] = op.where(~rng, "range")
# censoring: '>' / '>=' right-censored (true LC50 above value); '<','<=' left-censored
r["cens"] = np.select([op.isin([">", ">="]), op.isin(["<", "<="])], ["right", "left"], "exact")
r.loc[op.eq("~"), "cens"] = "approx"

unit = r.conc1_unit.fillna("").str.strip()
ai = unit.str.startswith("AI ") | unit.str.startswith("ae ")
base = unit.str.replace(r"^(AI|ae) ", "", regex=True)
fac = {"mg/L": 1, "ug/L": 1e-3, "ng/L": 1e-6, "g/L": 1e3, "ppm": 1, "ppb": 1e-3, "ppt": 1e-6,
       "ug/ml": 1, "mg/ml": 1e3, "ng/ml": 1e-3, "mg/dm3": 1, "ug/dm3": 1e-3, "pg/L": 1e-9, "ug/l": 1e-3,
       "mg/l": 1, "g/ml": 1e6}
r["to_mgL"] = base.map(fac)
r["active_ingredient"] = ai

# duration in hours (observation duration of the result)
dur = num(r.obs_duration_mean)
dmin, dmax = num(r.obs_duration_min), num(r.obs_duration_max)
dur = dur.fillna(np.sqrt(dmin*dmax))
du = r.obs_duration_unit.fillna("").str.strip()
dfac = {"h": 1, "d": 24, "wk": 168, "mi": 1/60, "mo": 730, "ht": 1, "dph": 24, "hph": 1, "dpf": 24, "hpf": 1}
r["dur_h"] = dur * du.map({"h": 1, "d": 24, "wk": 168, "mi": 1/60, "mo": 730})

t2 = t[["test_id", "reference_number", "test_cas", "species_number", "organism_habitat", "organism_lifestage",
        "media_type", "test_location", "exposure_type", "test_method", "control_type", "test_formulation",
        "test_grade", "test_radiolabel", "num_doses_mean", "organism_source"]]
x = r.merge(t2, on="test_id", how="left")
x = x[x.organism_habitat.eq("Water")]
m2 = med[["result_id", "media_ph_mean", "media_ph_min", "media_ph_max", "media_temperature_mean",
          "media_temperature_min", "media_temperature_max", "media_temperature_unit",
          "media_hardness_mean", "media_hardness_unit", "dissolved_oxygen_mean", "media_salinity_mean",
          "media_salinity_unit", "media_org_carbon_mean", "media_alkalinity_mean"]]
x = x.merge(m2, on="result_id", how="left")

def mid(a, lo, hi):
    a, lo, hi = num(x[a]), num(x[lo]), num(x[hi])
    return a.fillna((lo+hi)/2)
x["temp_C"] = mid("media_temperature_mean", "media_temperature_min", "media_temperature_max")
tu = x.media_temperature_unit.fillna("").str.strip()
x.loc[tu.eq("F"), "temp_C"] = (x.temp_C[tu.eq("F")] - 32) * 5/9
x.loc[~tu.isin(["C", "F", ""]), "temp_C"] = np.nan
x.loc[(x.temp_C < -2) | (x.temp_C > 45), "temp_C"] = np.nan
x["pH"] = mid("media_ph_mean", "media_ph_min", "media_ph_max")
x.loc[(x.pH < 2) | (x.pH > 12), "pH"] = np.nan
x["hardness"] = num(x.media_hardness_mean)
x.loc[~x.media_hardness_unit.fillna("").str.strip().isin(["mg/L CaCO3", "mg/L", "ppm CaCO3", "ppm"]), "hardness"] = np.nan

x = x.merge(chem.rename(columns={"cas_number": "test_cas", "ecotox_group": "chem_group"}), on="test_cas", how="left")
x = x.merge(sp[["species_number", "latin_name", "kingdom", "phylum_division", "subphylum_div", "superclass", "class",
                "tax_order", "family", "genus", "ecotox_group", "ncbi_taxid"]].rename(columns={"ecotox_group": "sp_group"}),
            on="species_number", how="left")
x = x.merge(ref[["reference_number", "publication_year", "reference_type"]] if "publication_year" in ref.columns else ref,
            on="reference_number", how="left")

keep = x.to_mgL.notna() & x.value.notna() & (x.value > 0) & x.dur_h.notna()
x = x[keep].copy()
x["conc_mgL"] = x.value * x.to_mgL
x["log10_mgL"] = np.log10(x.conc_mgL)
print("aquatic LC50/EC50 mass-unit rows:", len(x))
print(x.cens.value_counts())
x.to_parquet(D/"aquatic_acute_all.parquet", index=False)
