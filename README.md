# Midnight Zone
### Causal Impact of Shipping Noise on Deep-Sea Migration

**Course:** Causal Inference Machine Learning — Final Project  
**Author:** Priyal Maniar  

---

## The core causal claim

> The COVID-19 collapse in commercial shipping traffic (March–June 2020) caused a measurable **recovery** in diel vertical migration (DVM) amplitude in high-traffic ocean corridors, relative to low-traffic control corridors where vessel density did not meaningfully change.

**Diel vertical migration (DVM)** is the largest daily animal migration on Earth by biomass. Mesopelagic organisms drive the biological carbon pump that sequesters CO₂. Underwater radiated noise from shipping suppresses DVM — but traditional ecological studies cannot isolate this from seasonal variation. This project uses a natural experiment to establish causality.

---

## Method stack

| Layer | Method | Tool |
|-------|--------|------|
| Identification | Difference-in-Differences (COVID shipping collapse as natural experiment) | statsmodels |
| Estimation | Double/Debiased ML — residualises confounders nonlinearly before DiD | EconML |
| Validation | 3 DoWhy refutation tests (placebo, random cause, data subset) | DoWhy |
| Extension | CausalForestDML — heterogeneous effects by bathymetry depth zone | EconML |
| Pipeline | DuckDB + dbt (staging → mart) | dbt-duckdb |
| Frontend | 6-page interactive dashboard | Streamlit + Plotly |

---

## Key results

| Method | ATE Estimate | Notes |
|--------|-------------|-------|
| True ATE (DGP) | −0.3500 | Encoded in simulation |
| TWFE DiD | +5.80 | Severely confounded by nonlinear SST/chlorophyll effects |
| LinearDML | −0.3505 | **99.9% recovery accuracy** |
| DoWhy refutations | 3 / 3 PASS | Placebo → −0.002, others stable |
| Parallel trends | p = 0.105 | Formal slope test PASS |

**Why DML outperforms plain DiD:** SST and chlorophyll-a have nonlinear effects on DVM amplitude that linear regression cannot remove. Gradient boosting nuisance models residualise these flexibly, recovering the true ATE at 99.9% accuracy.

---

## Project structure
midnight_zone/
├── data/
│   ├── simulate.py              ← Generates 13,000-row panel (100 cells × 130 weeks)
│   └── panel.csv                ← Simulated panel data
├── dbt_pipeline/
│   ├── models/
│   │   ├── staging/stg_panel.sql     ← Clean types, depth zones, derived fields
│   │   └── marts/mart_analysis.sql   ← Standardised features, group-week means
│   ├── dbt_project.yml
│   └── profiles.yml
├── analysis/
│   ├── 01_did_baseline.py       ← TWFE DiD + parallel trends (visual + formal slope test)
│   ├── 02_dml.py                ← LinearDML + DoWhy 3 refutation tests
│   └── 03_hte.py                ← CausalForestDML by depth zone
├── app/
│   └── streamlit_app.py         ← 6-page interactive Streamlit dashboard
├── outputs/                     ← CSVs with all results (auto-generated)
└── requirements.txt

---

## Setup & run

```bash
# 1. Clone the repo
git clone git@github.com:PriyalManiar/midnight-zone.git
cd midnight-zone

# 2. Create virtual environment with Python 3.13
python3.13 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Generate data and load into DuckDB
python data/simulate.py

# 5. Run dbt pipeline
cd dbt_pipeline
dbt run --profiles-dir .
cd ..

# 6. Run analysis scripts in order
python analysis/01_did_baseline.py
python analysis/02_dml.py
python analysis/03_hte.py

# 7. Launch Streamlit dashboard
streamlit run app/streamlit_app.py
```

---

## Data

| Source | Description | Role |
|--------|-------------|------|
| AIS vessel tracking (simulated) | 100 grid cells × 130 weeks, COVID drop encoded | Treatment variable |
| DVM amplitude (simulated) | Acoustic backscatter parameterised on ICES/NOAA literature | Outcome variable |

Simulated outcome data is explicitly permitted by the course guidelines. Generating data with a known treatment effect (TRUE_ATE = −0.35) allows validation that the causal estimator correctly recovers the ground truth — a stronger methodological demonstration than messy observational data with unresolvable confounders.

**Planned extension:** Replace simulated AIS data with real vessel tracking data from NOAA/MarineTraffic. The dbt pipeline is designed for this — only `simulate.py` needs to be replaced.

---

## Causal identification

**Natural experiment:** The COVID-19 shipping collapse (March–June 2020) reduced vessel density by ~60% in high-traffic corridors while remote control cells saw <0.3% change.

**DAG structure:**
vessel_density → dvm_amplitude
sst            → vessel_density, dvm_amplitude
log_chla       → dvm_amplitude
bathymetry     → vessel_density, dvm_amplitude
season_sin     → vessel_density, dvm_amplitude

**Parallel trends:** Formal slope test on 2018–2019 pre-period difference series.  
Slope = 0.00306, p = 0.105 → **PASS** (no significant pre-trend divergence)

---

## Limitations

1. **Parallel trends violation** — another 2020 shock could confound; untestable in post-period
2. **SUTVA violation** — acoustic spillover from treated to adjacent control cells
3. **Simulated outcome** — DGP validated, not real acoustic backscatter (NASC) data
4. **Vessel class heterogeneity** — AIS density doesn't distinguish tanker noise from fishing vessels
5. **Spatial aggregation** — 0.5° grid loses within-cell heterogeneity

---

## Tech stack

`Python 3.13` · `DuckDB` · `dbt-duckdb` · `EconML` · `DoWhy` · `statsmodels` · `scikit-learn` · `Streamlit` · `Plotly` · `pandas` · `numpy`
