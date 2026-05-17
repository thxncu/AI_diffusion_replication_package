# Codebook summary

Key analytical variables are listed below. The file `data/data_dictionary.csv` provides a broader column-level inventory.

| Variable | Meaning |
|---|---|
| `iso3` | ISO 3166-1 alpha-3 country/economy code. |
| `country_name` | Country or economy name. |
| `ms_ai_diffusion_q1_2026_pct` | Microsoft Q1 2026 broad AI diffusion percentage. |
| `q1_diffusion_change_from_h2_pct_points` | Q1 2026 broad diffusion minus H2 2025 broad diffusion, in percentage points. |
| `log_anthropic_per_capita_index` | Log-transformed Anthropic Claude per-capita usage index. Interpreted as Claude-visible platform use. |
| `institutional_capacity_z` | Standardized institutional-capacity composite based on Worldwide Governance Indicators. |
| `imf_aipi_index_2023_z` | Standardized International Monetary Fund Artificial Intelligence Preparedness Index. |
| `access_assets_z` | Standardized access-assets composite. |
| `reg_ethics_z` | Standardized regulatory and ethics readiness indicator. |
| `policy_intensity_z` | Standardized documented AI-policy attention indicator based on OECD.AI records. |
| `ai_capability_conversion_gap_q1_2026_pct` | Access-adjusted broad-diffusion residual for Q1 2026, in percentage points. |
| `k2_cluster` | Two-profile k-means assignment. |
| `k3_cluster` | Three-profile k-means assignment. |
| `k3_label` | Human-readable label for the three-profile solution. |
| `capacity_bundle_pc1_z` | Standardized principal-component score for the overall capacity-readiness bundle. |
| `capacity_bundle_pc2_z` | Standardized principal-component score for the access-capacity imbalance axis. |

Interpretation cautions:

- `log_anthropic_per_capita_index` is not a comprehensive measure of specialist or enterprise adoption.
- `ai_capability_conversion_gap_q1_2026_pct` is a diagnostic residual, not a welfare or productivity measure.
- Profile assignments are descriptive and diagnostic, not causal regime classifications.
