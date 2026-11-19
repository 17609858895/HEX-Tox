"""Build a derived (read-only source) core table from the ECOTOX ASCII zip.

Reads members directly from the zip; writes parquet files to data_derived/.
No modification of the original archive.
"""
import zipfile, io, time, os
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ZIP = Path(os.environ.get("ECOTOX_ARCHIVE", ROOT / "ecotox_ascii_09_15_2026.zip"))
OUT = Path(__file__).resolve().parents[1] / "data_derived"
PFX = "ecotox_ascii_09_15_2026/"

def read(member, usecols=None):
    t = time.time()
    with zipfile.ZipFile(ZIP) as z, z.open(PFX + member) as f:
        df = pd.read_csv(f, sep="|", dtype=str, usecols=usecols, encoding="latin1",
                         keep_default_na=False, na_values=[""], quoting=3,
                         low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    print(f"{member}: {df.shape} in {time.time()-t:.1f}s")
    return df

test_cols = ["test_id", "reference_number", "test_cas", "species_number", "organism_habitat",
             "organism_lifestage", "organism_age_mean", "organism_age_unit",
             "organism_init_wt_mean", "organism_init_wt_unit",
             "exposure_duration_mean", "exposure_duration_unit", "study_duration_mean", "study_duration_unit",
             "test_type", "test_location", "test_method", "exposure_type", "control_type",
             "media_type", "num_doses_mean", "test_formulation", "test_grade", "test_radiolabel",
             "test_purity_mean", "experimental_design", "organism_source", "geographic_code"]
res_cols = ["result_id", "test_id", "sample_size_mean", "obs_duration_mean_op", "obs_duration_mean",
            "obs_duration_min", "obs_duration_max", "obs_duration_unit",
            "endpoint", "trend", "effect", "measurement", "response_site",
            "effect_pct_mean", "conc1_type", "ion1",
            "conc1_mean_op", "conc1_mean", "conc1_min_op", "conc1_min", "conc1_max_op", "conc1_max",
            "conc1_unit", "conc1_comments", "significance_code", "chem_analysis_method",
            "endpoint_assigned", "dry_wet", "steady_state", "additional_comments"]
media_cols = ["result_id", "media_ph_mean", "media_ph_min", "media_ph_max",
              "media_temperature_mean", "media_temperature_min", "media_temperature_max", "media_temperature_unit",
              "media_hardness_mean", "media_hardness_unit", "dissolved_oxygen_mean", "dissolved_oxygen_unit",
              "media_salinity_mean", "media_salinity_unit", "media_alkalinity_mean", "media_alkalinity_unit",
              "media_org_carbon_mean", "media_org_carbon_unit", "media_conductivity_mean", "media_conductivity_unit",
              "media_org_matter_mean", "media_org_matter_unit", "media_cec_mean", "media_cec_unit",
              "media_texture_clay_mean", "media_moisture_pct_mean"]

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT.parent / "results").mkdir(exist_ok=True)
    tests = read("tests.txt", test_cols)
    tests.to_parquet(OUT / "tests_core.parquet", index=False)
    res = read("results.txt", res_cols)
    res.to_parquet(OUT / "results_core.parquet", index=False)
    media = read("media_characteristics.txt", media_cols)
    media.to_parquet(OUT / "media_core.parquet", index=False)
    for m in ["validation/chemicals.txt", "validation/species.txt", "validation/references.txt",
              "chemical_carriers.txt", "doses.txt", "dose_responses.txt",
              "dose_response_details.txt", "dose_response_links.txt"]:
        d = read(m)
        d.to_parquet(OUT / (Path(m).stem + ".parquet"), index=False)
    for code in ["endpoint_codes", "effect_codes", "concentration_unit_codes", "duration_unit_codes",
                 "lifestage_codes", "media_type_codes", "exposure_type_codes", "test_method_codes",
                 "measurement_codes", "concentration_type_codes", "chemical_analysis_codes", "test_location_codes",
                 "media_char_unit_codes", "habitat_codes"]:
        d = read(f"validation/{code}.txt")
        d.to_parquet(OUT / f"code_{code}.parquet", index=False)
