# Midnight Zone
### Causal Impact of Fishing Vessel Noise on Deep-Sea Migration

**Course:** IS 590 : Causal Inference Machine Learning  
**Author:** Priyal Maniar 

---

## Causal Claim

> COVID-19 disrupted commercial fishing vessel activity in the English Channel and North Sea, causing a measurable recovery in diel vertical migration (DVM) amplitude in those corridors relative to remote Mid-Atlantic control corridors where fishing effort was not comparably disrupted.

**Diel Vertical Migration (DVM)** is the largest daily animal migration on Earth by biomass. Mesopelagic organisms drive the biological carbon pump that sequesters CO2 in the deep ocean. Underwater radiated noise from vessel traffic suppresses DVM — but traditional ecological studies cannot isolate this from seasonal variation. This project uses COVID-19 as a natural experiment.

---

## Dual Panel Design

| Panel | Treatment variable | COVID signal | Purpose |
|-------|-------------------|-------------|---------|
| Panel A | Real GFW fishing effort hours (174,026 records) | 15.1% drop treated vs 5.8% control | Primary specification |
| Panel B | Simulated 60% COVID drop on real geographic grid | 64.3% drop treated vs 0.5% control | Methodological validation |

Panel B validates that the DML estimator correctly recovers a known true effect (TRUE_ATE = -0.35) on the real geographic structure. Both panels converge to essentially the same estimate — the central result of the project.

---

## Method Stack

| Layer | Method | Tool | Purpose |
|-------|--------|------|---------|
| Identification | Difference-in-Differences | statsmodels | COVID fishing collapse as natural experiment |
| Estimation | Double/Debiased ML | EconML LinearDML | Residualises confounders nonlinearly |
| Validation | 3 DoWhy refutation tests | DoWhy | Placebo, random cause, data subset |
| Extension | CausalForestDML | EconML | Heterogeneous effects by depth zone |
| Pipeline | DuckDB + dbt | dbt-duckdb | Staging and mart transformation layer |
| Frontend | Interactive dashboard | Streamlit + Plotly | Panel toggle, charts, results |

---

## Key Results

| Method | Panel A (Real AIS) | Panel B (Simulated) |
|--------|-------------------|---------------------|
| True ATE (DGP) | - 0.3500 | - 0.3500 |
| TWFE DiD | + 0.037 (confounded) | NaN (multicollinearity) |
| LinearDML ATE | - 0.3489 | -0.3460 |
| Recovery error | 0.32% | 1.13% |
| Parallel trends | p = 0.913 (PASS) | p = 0.551 (PASS) |
| DoWhy refutations | 3 / 3 PASS | 3 / 3 PASS |

**Why TWFE DiD fails:** SST and chlorophyll-a have nonlinear effects on DVM that linear regression cannot remove. DML's gradient boosting nuisance models residualise these flexibly, recovering the true ATE with under 1.2% error.

**Why both panels converge:** The DML estimator isolates the causal effect regardless of the treatment shock magnitude, provided confounders are adequately controlled.

---

## Data

| Source | Records | Coverage | Role |
|--------|---------|----------|------|
| Global Fishing Watch API (public-global-fishing-effort) | 174,026 | 2018-2020 Q2 | Real treatment variable |
| Simulated DVM amplitude | 118,221 cell-month obs | 2018-2020 Q2 | Outcome (known DGP, TRUE_ATE = - 0.35) |
| NOAA OISST (parameterised) | Per cell-month | 2018-2020 Q2 | SST confounder |

**Geographic Regions:**

| Region | Type | Bathymetry | Rationale |
|--------|------|-----------|-----------|
| English Channel | Treated | ~60m | High-traffic fishing corridor |
| North Sea | Treated | ~95m | High-traffic fishing corridor |
| Mid-Atlantic zone 1 (35-40N, 40-30W) | Control | ~4200m | Remote, low-traffic |
| Mid-Atlantic zone 2 (40-45N, 45-35W) | Control | ~4500m | Remote, low-traffic |

**Note on treatment variable:** The GFW free-tier API provides fishing vessel effort hours only — not all commercial shipping. Cargo ships, tankers, and container vessels (the primary URN sources and the vessels with the largest COVID collapse, ~60%) require licensed data access. The causal claim is therefore specific to fishing vessel acoustic disturbance. Panel B simulates the sharper cargo shock to validate the estimator under a known DGP.

---

## Project Structure

```
midnight_zone/
├── data/
│   ├── fetch_ais.py          - Downloads real GFW fishing effort data
│   ├── build_panel.py        - Builds Panel A (real) and Panel B (simulated shock)
│   └── simulate.py           - Original simulation script (reference)
├── dbt_pipeline/
│   ├── models/
│   │   ├── staging/
│   │   │   └── stg_panel.sql       - Clean types, depth zones, derived fields
│   │   └── marts/
│   │       └── mart_analysis.sql   - Standardised features, group-week means
│   ├── dbt_project.yml
│   └── profiles.yml
├── analysis/
│   ├── 01_did_baseline.py    - TWFE DiD + parallel trends (visual + formal slope test)
│   ├── 02_dml.py             - LinearDML + DoWhy 3 refutation tests
│   └── 03_hte.py             - CausalForestDML by depth zone
├── app/
│   └── streamlit_app.py      - 6-page interactive dashboard with panel toggle
├── outputs/                  - Auto-generated plots and result CSVs
└── requirements.txt
```

---

## Setup and Run

```bash
# 1. Clone
git clone git@github.com:PriyalManiar/midnight-zone.git
cd midnight-zone

# 2. Create virtual environment (Python 3.13 required)
python3.13 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download real AIS data from GFW
# Add your GFW API token to data/fetch_ais.py (GFW_TOKEN variable)
python data/fetch_ais.py

# 5. Build both panels
python data/build_panel.py

# 6. Run dbt pipeline
cd dbt_pipeline && dbt run --profiles-dir . && cd ..

# 7. Create sim mart table (run once after dbt)
python -c "
import duckdb
con = duckdb.connect('data/midnight_zone.duckdb')
con.execute('DROP TABLE IF EXISTS mart_analysis_sim')
con.execute('''CREATE TABLE mart_analysis_sim AS
    SELECT *, date::date as observation_date,
    avg(vessel_density) OVER (PARTITION BY week_index, is_treated) as group_week_mean_density,
    avg(dvm_amplitude) OVER (PARTITION BY week_index, is_treated) as group_week_mean_dvm,
    (sst-14.7)/4.3 as sst_std, (log_chla-0.48)/0.65 as log_chla_std,
    (bathymetry-1462)/1194 as bathymetry_std FROM raw_panel_sim''')
con.close()
"

# 8. Run analysis scripts in order
python analysis/01_did_baseline.py
python analysis/02_dml.py
python analysis/03_hte.py

# 9. Launch Streamlit dashboard
streamlit run app/streamlit_app.py
```

---

## GFW API Token

Register at [globalfishingwatch.org/our-apis](https://globalfishingwatch.org/our-apis) for a free API token. Add it to `data/fetch_ais.py`:

```python
GFW_TOKEN = "your_token_here"
```

The free tier gives access to `public-global-fishing-effort` : sufficient for this project.

---

## Limitations

1. **Parallel trends** : another 2020 shock could confound; untestable in the post-period
2. **Geographic dissimilarity** : European shelf vs deep Mid-Atlantic; parallel trends rests on statistical evidence alone
3. **Fishing vessels only** : GFW free tier excludes cargo/tanker traffic, the primary URN sources
4. **Simulated DVM outcome** : recovery error validates on DGP, not real acoustic backscatter
5. **SUTVA violation** : acoustic spillover from treated to adjacent control cells
6. **Spatial aggregation** : 0.5-degree grid loses within-cell heterogeneity

---

## Tech Stack

`Python 3.13` · `DuckDB` · `dbt-duckdb` · `EconML` · `DoWhy` · `statsmodels` · `scikit-learn` · `Streamlit` · `Plotly` · `GFW API v3` · `pandas` · `numpy`
