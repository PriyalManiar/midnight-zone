"""
Generates the ~13,000 grid-cell × week panel.

Design:
- 100 grid cells: 50 treated (high-traffic corridors), 50 control (remote ocean)
- 130 weeks: 2018 W1 → 2020 W22 (end of Q2)
- COVID shock: treated cells drop 60% in vessel density from week 104 (2020 W10)
"""

import numpy as np
import pandas as pd
import duckdb
import os

np.random.seed(42)

N_CELLS     = 100
N_WEEKS     = 130
TRUE_ATE    = -0.35
COVID_START = 104
COVID_DROP  = 0.60

cell_ids   = [f"CELL_{i:03d}" for i in range(N_CELLS)]
treated    = np.array([1]*50 + [0]*50)

bathymetry = np.where(treated,
    np.random.uniform(200, 800, N_CELLS),
    np.random.uniform(800, 4000, N_CELLS)
)

baseline_density = np.where(treated,
    np.random.uniform(15, 40, N_CELLS),
    np.random.uniform(1, 6, N_CELLS)
)

cell_fe = np.random.normal(0, 0.15, N_CELLS)

records = []

for w in range(N_WEEKS):
    year  = 2018 + w // 52
    week  = (w % 52) + 1
    date  = pd.Timestamp("2018-01-01") + pd.Timedelta(weeks=w)
    season_sin = np.sin(2 * np.pi * week / 52)
    season_cos = np.cos(2 * np.pi * week / 52)

    for i, cell in enumerate(cell_ids):
        covid_active   = int(w >= COVID_START and treated[i] == 1)
        vessel_density = max(0, baseline_density[i] * (1 - COVID_DROP * covid_active) + np.random.normal(0, 2))
        sst            = 14 + 6 * season_sin + np.random.normal(0, 0.8)
        chla_raw       = np.exp(0.4 + 0.6 * season_sin + np.random.normal(0, 0.5))
        log_chla       = np.log(chla_raw)
        season_cat     = (week - 1) // 13

        dvm_amplitude = (
            2.5
            + TRUE_ATE * vessel_density
            - 0.08  * sst
            + 0.25  * log_chla
            - 0.0002 * bathymetry[i]
            + 0.30  * season_sin
            + cell_fe[i]
            + np.random.normal(0, 0.20)
        )

        records.append({
            "cell_id":        cell,
            "week_index":     w,
            "year":           year,
            "week":           week,
            "date":           date.strftime("%Y-%m-%d"),
            "treated":        int(treated[i]),
            "post":           int(w >= COVID_START),
            "covid_active":   covid_active,
            "vessel_density": round(vessel_density, 3),
            "dvm_amplitude":  round(dvm_amplitude, 4),
            "sst":            round(sst, 3),
            "log_chla":       round(log_chla, 4),
            "bathymetry":     round(bathymetry[i], 1),
            "season_sin":     round(season_sin, 4),
            "season_cos":     round(season_cos, 4),
            "season_q":       season_cat,
        })

panel = pd.DataFrame(records)

print(f"Panel shape: {panel.shape}")
print(f"\nDescriptive statistics:")
print(panel[["vessel_density","dvm_amplitude","sst","log_chla","bathymetry"]].describe().round(3))

treated_pre  = panel[(panel.treated==1) & (panel.post==0)]["vessel_density"].mean()
treated_post = panel[(panel.treated==1) & (panel.post==1)]["vessel_density"].mean()
control_pre  = panel[(panel.treated==0) & (panel.post==0)]["vessel_density"].mean()
control_post = panel[(panel.treated==0) & (panel.post==1)]["vessel_density"].mean()

print(f"\n── COVID signal check ──")
print(f"Treated pre:  {treated_pre:.2f}")
print(f"Treated post: {treated_post:.2f}  (drop: {(1-treated_post/treated_pre)*100:.1f}%)")
print(f"Control pre:  {control_pre:.2f}")
print(f"Control post: {control_post:.2f}  (change: {abs(1-control_post/control_pre)*100:.1f}%)")

os.makedirs("data", exist_ok=True)
panel.to_csv("data/panel.csv", index=False)

con = duckdb.connect("data/midnight_zone.duckdb")
con.execute("DROP TABLE IF EXISTS raw_panel")
con.execute("CREATE TABLE raw_panel AS SELECT * FROM panel")
count = con.execute("SELECT COUNT(*) FROM raw_panel").fetchone()[0]
con.close()

print(f"\n Saved data/panel.csv")
print(f" Loaded into DuckDB: {count:,} rows")