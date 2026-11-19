# HEX-Tox

Analysis code for *Within-study difference learning recovers exposure-duration responses in aquatic toxicity archives*. The study tests how well aquatic toxicity models recover changes in mortality LC50 with exposure duration within matched experimental series.

## Inputs and environment

Use the EPA ECOTOX ASCII archive with the 15 September 2026 package label. Obtain the source archive from the [EPA ECOTOX Knowledgebase](https://cfpub.epa.gov/ecotox/). Keep the original archive intact. Set the environment variable `ECOTOX_ARCHIVE` to its full path, or place `ecotox_ascii_09_15_2026.zip` immediately above this checkout. The scripts expect the original internal member prefix `ecotox_ascii_09_15_2026/`. The optional chemical-structure benchmarks retrieve public records using PubChem PUG REST. Live PubChem responses can change.

Install the packages in `requirements.txt` in a Python environment. Run the following from the repository root. The first script creates `data_derived/` and `results/`. Processing the full archive and model folds requires substantially more memory and time than reading the tabulated results.

~~~sh
python scripts/01_build_core.py
python scripts/02_build_aquatic_acute.py
python scripts/E05_contrast_learning.py
python scripts/E05b_pair_source_ablation.py
python scripts/E05c_temporal_split.py
python scripts/E15_plateau_test.py
~~~

## Analysis routes

- **Duration response and transfer.** E05 fits pooled level and within-series difference models and creates held-out contrast predictions. E05b compares within-series, cross-series and combined pairing. E05c evaluates chronological transfer. E06 tests confounding in semi-synthetic data. E10, E12, E13 and E15 examine temperature heterogeneity, identification, taxonomy and late-duration slopes.
- **Level prediction.** E01–E04 cover censored levels, duration extrapolation, temperature anchoring and curve models. Run 03 and 04 before chemical-structure benchmarks E08, E11, E11b and E14. E09 uses E05 predictions to evaluate empirical interval coverage. For all secondary analyses, run the numbered preparation scripts followed by the E scripts in numerical order, including lettered variants after the base number.
- **Multispecies projections.** E07 generates the standardized species records. After E05 and E07, run `python scripts/rebuild_hc5_summary.py` for the final paired species resampling, censoring rules and projected 96 h to 28 d ratios. Use its `hc5_duration_analysis.csv`, `hc5_rank_analysis.csv` and `hc5_summary.json` for the reported factors. These projections concern mortality LC50.
- **Manuscript summaries.** After running the experiments, run `python scripts/manuscript_metrics.py` to summarize record counts and held-out comparisons. The manuscript Supporting Information defines the estimands, grouping rules and subset-dependent denominators.

The repository distributes analysis scripts. The source archive, derived toxicity records and submission documents are obtained separately.
