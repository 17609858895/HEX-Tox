"""Audit the frozen E07 projection against reference-disjoint measured long endpoints.

Run after E07 and rebuild_hc5_summary.py. E05 folds are not used to select
the independent evaluation set. Concentrations are log10 mg/L throughout.
"""
import json
import re
import unicodedata

import numpy as np
import pandas as pd

from common import D, R, load_lc50


SERIES = ["reference_number", "test_cas", "species_number", "organism_lifestage",
          "exposure_type", "media_type", "conc1_type", "chem_analysis_method"]
PAIR = ["test_cas", "species_number"]
REF = "reference_number"


def normalized(value):
    if pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKD", str(value)).lower()
    return re.sub("[^a-z0-9]", "", text)


def reconstruct_sources(records, references):
    exact = records.loc[records.cens.eq("exact")].copy()
    exact["series"] = exact[SERIES].astype(str).agg("|".join, axis=1)
    exact["condition"] = exact.dur_h.round(2).astype(str) + "|" + exact.temp_C.round(1).astype(str)
    sizes = exact.groupby("series").condition.nunique()
    fit_refs = set(exact.loc[exact.series.isin(sizes[sizes > 1].index), REF])
    acute = records.loc[records.dur_h.between(24, 96)].reset_index(drop=True)
    contributors = set(acute[REF]) | fit_refs
    ref = references.copy()
    ref["title_key"] = ref.title.map(normalized)
    ref["doi_key"] = ref.doi.fillna("").str.lower().str.strip().str.replace(
        r"^https?://(?:dx\.)?doi\.org/", "", regex=True)
    excluded = set(contributors)
    while True:
        used = ref[ref[REF].isin(excluded)]
        collisions = ref.title_key.isin(set(used.title_key) - {""}) | ref.doi_key.isin(set(used.doi_key) - {""})
        expanded = excluded | set(ref.loc[collisions, REF])
        if expanded == excluded:
            break
        excluded = expanded
    counts = dict(training_pairs=int((sizes * (sizes - 1) // 2).sum()),
                  training_references=len(fit_refs), anchor_references=acute[REF].nunique(),
                  contributing_references=len(contributors))
    return acute, contributors, excluded, counts


def point_endpoint(df):
    return (df.cens.eq("exact") & ~df.is_range & df.op.fillna("").isin(["", "="])
            & df.obs_duration_mean.notna() & df.obs_duration_mean_op.fillna("").isin(["", "="]))


def score(cases):
    observed = cases.observed_log.to_numpy()
    base = cases.acute_log.to_numpy()
    projected = cases.projected_log.to_numpy()
    residual0, residual1 = base - observed, projected - observed
    metrics = dict(pairs=len(cases), chemicals=cases.test_cas.nunique(),
                   lower_than_acute=int((observed < base).sum()),
                   median_observed_factor=float(np.median(10 ** (base - observed))),
                   median_predicted_factor=float(np.median(10 ** (base - projected))))
    common_residual = residual0 - np.log10(2.52)
    for tag, residual in [("persistence", residual0), ("projection", residual1),
                          ("common_2_52", common_residual)]:
        metrics[tag] = dict(rmse=float(np.sqrt(np.mean(residual**2))),
                            mae=float(np.mean(abs(residual))), bias=float(np.mean(residual)))
    # Resample chemicals, keeping all of a chemical's species together.
    stats = pd.DataFrame({"chem": cases.test_cas.to_numpy(), "n": 1,
                          "a": residual0**2, "b": residual1**2, "c": common_residual**2}).groupby("chem").sum().to_numpy()
    rng = np.random.default_rng(20261119)
    samples = stats[rng.integers(len(stats), size=(5000, len(stats)))].sum(axis=1)
    differences = np.sqrt(samples[:, 2] / samples[:, 0]) - np.sqrt(samples[:, 1] / samples[:, 0])
    metrics["delta_rmse_ci95"] = np.quantile(differences, [.025, .975]).tolist()
    vs_common = np.sqrt(samples[:, 2] / samples[:, 0]) - np.sqrt(samples[:, 3] / samples[:, 0])
    metrics["vs_common_rmse_ci95"] = np.quantile(vs_common, [.025, .975]).tolist()
    return metrics


def main():
    records = load_lc50()
    refs = pd.read_parquet(D / "references.parquet")
    acute, contributing, blocked, report = reconstruct_sources(records, refs)
    frozen = pd.read_parquet(R / "E07_standardized_records.parquet").reset_index(drop=True)
    for field in [*PAIR, "log_dur", "cens", "log10_mgL"]:
        assert np.array_equal(acute[field].to_numpy(), frozen[field].to_numpy()), field
    acute["projected_log"] = acute.log10_mgL + frozen.y28d - frozen.y96
    acute = acute[point_endpoint(acute) & acute.dur_h.eq(96)].copy()
    candidates = records[records.dur_h.between(504, 672) & records.cens.eq("exact")]
    candidates = candidates[~candidates[REF].isin(blocked)].copy()
    candidates["point_endpoint"] = point_endpoint(candidates)
    candidates = candidates.merge(refs[[REF, "title", "doi", "author"]], on=REF, validate="many_to_one")
    candidates.to_csv(R / "E16_independent_long_records.csv", index=False, columns=[
        "result_id", "test_id", REF, "test_cas", "chemical_name", "species_number", "latin_name",
        "dur_h", "conc_mgL", "point_endpoint", "op", "is_range", "obs_duration_mean_op",
        "conc1_type", "media_type", "organism_lifestage", "exposure_type", "title", "doi", "author"])
    report["source_checks"] = {ref:ref in blocked for ref in ["160944"]}
    report["long_records"] = len(candidates)
    report["long_refs"] = candidates[REF].nunique()
    clean = candidates[candidates.point_endpoint].copy()
    def coverage(df):
        return dict(rows=len(df), references=df[REF].nunique(), chemicals=df.test_cas.nunique(),
                    species=df.species_number.nunique(), max_species=int(df.groupby("test_cas").species_number.nunique().max()))
    report["clean_21_28"] = coverage(clean)
    obs = clean[clean.dur_h.eq(672)]
    report["clean_28"] = coverage(obs)
    projected_set = pd.read_csv(R / "hc5_duration_analysis.csv", dtype={"test_cas":str})
    target = set(projected_set.loc[projected_set.status.eq("included"), "test_cas"])
    # Equal study weights on each side of the comparison; each pair scores once.
    def summarise(df, keys, cols):
        by_ref = df.groupby(keys + [REF], as_index=False)[cols].mean()
        return by_ref.groupby(keys)[cols].mean(), by_ref.groupby(keys)[REF].agg(lambda v:"|".join(sorted(v)))
    def compare(keys):
        observed, orefs = summarise(obs, keys, ["log10_mgL"])
        anchors, arefs = summarise(acute, keys, ["log10_mgL", "projected_log"])
        ans = observed.rename(columns={"log10_mgL":"observed_log"}).join(
            anchors.rename(columns={"log10_mgL":"acute_log"}), how="inner")
        ans = ans.join(orefs.rename("long_refs")).join(arefs.rename("acute_refs")).reset_index()
        ans["in_projected_658"] = ans.test_cas.isin(target)
        ans["observed_factor"] = 10 ** (ans.acute_log - ans.observed_log)
        ans["predicted_factor"] = 10 ** (ans.acute_log - ans.projected_log)
        return ans
    all_pairs = compare(PAIR)
    # Distinct titles can reprint a laboratory result. Flag the entire pair
    # when a day-28 point matches any day-28 point in a contributing source.
    contributed_long = records[records[REF].isin(blocked) & records.dur_h.eq(672) & point_endpoint(records)]
    match_keys = PAIR + ["log_key"]
    overlaps = obs.assign(log_key=obs.log10_mgL.round(5)).merge(
        contributed_long.assign(log_key=contributed_long.log10_mgL.round(5))[match_keys+[REF]],
        on=match_keys, suffixes=("_long","_contributing"))
    overlaps[PAIR+["result_id","reference_number_long","reference_number_contributing","conc_mgL"]].to_csv(
        R/"E16_possible_reprints.csv",index=False)
    duplicate_pairs = pd.MultiIndex.from_frame(overlaps[PAIR].drop_duplicates())
    def no_reprints(df):
        return ~pd.MultiIndex.from_frame(df[PAIR]).isin(duplicate_pairs)
    all_pairs["possible_reprint"] = ~no_reprints(all_pairs)
    unfiltered = all_pairs[all_pairs.in_projected_658]
    report["reference_disjoint_only"] = score(unfiltered)
    cases = unfiltered[~unfiltered.possible_reprint]
    name = records[PAIR + ["chemical_name","latin_name"]].drop_duplicates(PAIR)
    all_pairs.merge(name, on=PAIR, validate="one_to_one").to_csv(R / "E16_matched_28d.csv", index=False)
    report["all_pairs"] = score(all_pairs[~all_pairs.possible_reprint])
    report["target_pairs"] = score(cases)
    report["target_refs"] = len(set("|".join(cases.long_refs).split("|")))
    report["max_target_species"] = int(cases.groupby("test_cas").size().max())
    trout = set(name.loc[name.latin_name.eq("Oncorhynchus mykiss"),"species_number"])
    report["without_rainbow_trout"] = score(cases[~cases.species_number.isin(trout)])
    # A different article can report work from the same laboratory. Drop a
    # long record if its lead surname also appears as lead surname on any
    # contributing reference for that same chemical and species.
    surname = refs.set_index(REF).author.fillna("").str.split(",").str[0].map(normalized)
    used_sources = records.loc[records[REF].isin(contributing), PAIR + [REF]].drop_duplicates()
    used_sources["lead"] = used_sources[REF].map(surname)
    used_keys = pd.MultiIndex.from_frame(used_sources[PAIR + ["lead"]].drop_duplicates())
    original_obs = obs
    lead_keys = obs[PAIR].assign(lead=obs[REF].map(surname))
    obs = obs.loc[~pd.MultiIndex.from_frame(lead_keys).isin(used_keys) & lead_keys.lead.ne("")]
    different_lead = compare(PAIR)
    report["different_lead"] = score(different_lead[different_lead.in_projected_658 & no_reprints(different_lead)])
    obs = original_obs[~original_obs.author.fillna("").str.contains(r"\b(?:birge|black)\b", case=False)]
    without_team = compare(PAIR)
    report["without_birge_black"] = score(without_team[without_team.in_projected_658 & no_reprints(without_team)])
    obs = original_obs
    strict = compare(PAIR + ["media_type","conc1_type","active_ingredient"])
    strict = strict[strict.in_projected_658 & no_reprints(strict)]
    # Average matched strata before scoring so each chemical/species has equal weight.
    strata = strict.groupby(PAIR, as_index=False)[["observed_log","acute_log","projected_log"]].mean()
    if len(strata):
        report["matched_basis_medium"] = score(strata)
    summary_rows = []
    for label in ["target_pairs","reference_disjoint_only","without_rainbow_trout","different_lead","without_birge_black","matched_basis_medium","all_pairs"]:
        row = report[label]
        for method in ["persistence","common_2_52","projection"]:
            summary_rows.append({"subset":label, "method":method,"pairs":row["pairs"],"chemicals":row["chemicals"],
                                 "lower":row["lower_than_acute"],"observed_median_factor":row["median_observed_factor"],
                                 "model_median_factor":row["median_predicted_factor"],**row[method]})
    pd.DataFrame(summary_rows).to_csv(R / "E16_validation_metrics.csv", index=False)
    (R / "E16_validation_summary.json").write_text(json.dumps(report, indent=2), encoding="utf8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
