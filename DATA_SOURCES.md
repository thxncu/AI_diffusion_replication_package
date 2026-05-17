# Data sources

The analysis uses public country-level indicators. The harmonized datasets in `data/` combine variables from the sources below.

| Source | Role in the analysis | Included package file or derived field |
|---|---|---|
| Microsoft AI Economy Institute, AI Diffusion Report Q1 2026 update | Broad mass diffusion outcome and H2 2025 to Q1 2026 growth | `AI_Diffusion_Q12026_Update.csv`; `microsoft_ai_diffusion_q1_2026_source_normalized.csv` |
| Anthropic Economic Index | Claude-visible platform-use indicator | `log_anthropic_per_capita_index`; Anthropic fields in the harmonized data |
| International Monetary Fund Artificial Intelligence Preparedness Index | AI preparedness and subdimensions | IMF AIPI fields in the harmonized data |
| World Bank Worldwide Governance Indicators | Institutional capacity indicators | WGI fields in the harmonized data |
| World Bank World Development Indicators | GDP per capita, internet use, population, education, research and development, region and income metadata | WDI fields and World Bank metadata files |
| Organisation for Economic Co-operation and Development Artificial Intelligence Policy Observatory (OECD.AI) | Documented AI-policy attention | OECD.AI fields in the harmonized data |
| Bertelsmann Stiftung Transformation Index | Independent governance-source robustness | BTI fields in the harmonized data |

## Data-use note

The code and package documentation are released under the MIT License. The datasets and derived indicators remain subject to the terms of the original providers. Users should consult each provider before redistributing the data outside replication and scholarly review contexts.
