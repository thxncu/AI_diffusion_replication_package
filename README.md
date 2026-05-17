# Replication package for Q1 2026 generative AI diffusion analysis

This repository contains the data and code used to reproduce the country-level analyses for the manuscript **Conversion Profiles in the Global Diffusion of Generative AI: Governance Readiness, Access, and Platform Observability**.

The analysis evaluates how governance-readiness-access profiles are associated with three observable patterns:

1. **Broad mass diffusion**, measured with Microsoft Artificial Intelligence (AI) diffusion data for Q1 2026.
2. **Claude-visible platform use**, measured with Anthropic Claude per-capita usage indicators.
3. **Access-adjusted broad-diffusion residuals**, used as diagnostic indicators rather than welfare or productivity measures.

The package also reproduces the robustness checks reported in the Supplementary Material, including alternative access-baseline models, logit-transformed broad diffusion models, outlier and leverage checks, Gaussian mixture model Bayesian information criterion checks, and World Bank region and income-group descriptive checks.

## Repository structure

```text
.
├── code/
│   ├── 00_run_all.py
│   ├── reproduce_q1_2026_analysis.py
│   └── reproduce_extended_robustness_analysis.py
├── data/
│   ├── final_analysis_data_with_bti_and_selection.csv
│   ├── AI_Diffusion_Q12026_Update.csv
│   ├── final_analysis_data_q1_2026.csv
│   ├── final_analysis_data_q1_2026_with_alternative_gaps.csv
│   ├── final_analysis_data_q1_2026_with_wb_metadata.csv
│   ├── world_bank_country_region_income_metadata.csv
│   └── world_bank_population_indicator_download_with_country_metadata.xlsx
├── tables/
├── figures/
├── reports/
├── docs/
├── requirements.txt
├── environment.yml
├── DATA_SOURCES.md
├── CODEBOOK.md
├── LICENSE
└── CITATION.cff
```

## Quick start

Create a clean Python environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Run the full replication workflow:

```bash
python code/00_run_all.py
```

The workflow regenerates the harmonized Q1 2026 data, main tables, robustness tables, figures, and summary reports. On the test environment used to assemble the package, the full workflow ran in about 30 seconds.

## Main input files

The full workflow starts from two files:

- `data/final_analysis_data_with_bti_and_selection.csv`: harmonized pre-Q1 country-level dataset with Microsoft H1/H2 2025, Anthropic, International Monetary Fund, World Bank, OECD.AI, and Bertelsmann Stiftung Transformation Index variables.
- `data/AI_Diffusion_Q12026_Update.csv`: Microsoft Q1 2026 update file.

The extended robustness workflow also uses:

- `data/world_bank_population_indicator_download_with_country_metadata.xlsx`: World Bank metadata source used to derive region and income group checks.

## Key outputs

- `data/final_analysis_data_q1_2026.csv`: harmonized Q1 2026 analysis dataset.
- `tables/k3_configuration_profiles_q1.csv`: descriptive profiles for the three-profile solution.
- `tables/outcome_separation_k2_k3_q1.csv`: k = 2 versus k = 3 outcome validation.
- `tables/main_channel_regressions_q1.csv`: channel-specific regression models.
- `tables/alternative_access_baseline_robustness_q1.csv`: robustness of the access-adjusted residual under alternative access baselines.
- `tables/logit_transformed_broad_diffusion_robustness_q1.csv`: logit-transformed broad diffusion robustness.
- `tables/outlier_and_leverage_robustness_q1.csv`: outlier and leverage robustness.
- `tables/gmm_bic_cluster_robustness_q1.csv`: Gaussian mixture model Bayesian information criterion diagnostics.
- `figures/`: reproduced figures used in the manuscript and Supplementary Material.

## Reproducibility notes

- The random seed is fixed at `123` in both analysis scripts. Bootstrap stability uses 80 resamples for a balance between reproducibility and runtime.
- Diffusion outcomes are not used to construct the k-means profiles. They are held out for post-cluster comparison.
- The three-profile solution is interpreted as a theory-guided diagnostic partition, not as a uniquely optimal statistical partition or a causal regime classification.
- All regressions reported by the scripts use heteroskedasticity-consistent type 3 (HC3) standard errors where applicable.
- The included country-level data are public or derived from public source datasets and contain no personal data.

## Anonymous peer review

For double-anonymized review, upload this repository to an anonymized repository service or a GitHub repository that does not reveal author identity. After acceptance, the repository can be transferred to a permanent public archive, such as Zenodo, Open Science Framework, or Mendeley Data.

## License

The code and package documentation are released under the MIT License. Data files and derived variables are provided for replication purposes and remain subject to the terms and licenses of the original data providers listed in `DATA_SOURCES.md`.
