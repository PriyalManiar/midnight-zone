"""
fetch_ais.py
------------
Downloads real fishing effort data from Global Fishing Watch API v3.
Builds the treatment variable (vessel activity hours) for our causal panel.

Regions:
  Treated  — English Channel + North Sea (high-traffic shipping corridors)
  Control  — Mid-Atlantic remote ocean (low-traffic)

Output: data/ais_raw.csv + loaded into DuckDB raw_ais table
"""

import requests
import pandas as pd
import numpy as np
import duckdb
import os, time

# ── CONFIG ────────────────────────────────────────────────────────────────────
GFW_TOKEN = "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCIsImtpZCI6ImtpZEtleSJ9.eyJkYXRhIjp7Im5hbWUiOiJtaWRuaWdodF96b25lIiwidXNlcklkIjo2MTY1NywiYXBwbGljYXRpb25OYW1lIjoibWlkbmlnaHRfem9uZSIsImlkIjoxMDY1MCwidHlwZSI6InVzZXItYXBwbGljYXRpb24ifSwiaWF0IjoxNzc4MDExMjM1LCJleHAiOjIwOTMzNzEyMzUsImF1ZCI6ImdmdyIsImlzcyI6ImdmdyJ9.e9S09Nw8TH8L6pzJ4Pk1N11GEnRR3-wHDCwv8jbcilP7_u1Hf61GsAQwB7de4Ek97xVoSJGPO9FucwPDX5Dqw4cgsUSOL1tT03LmOzlLyha_SfKW79p_HwPRBzhxNjErIp21TFamQ8kqion8i1IW6qeUNRZ1PlvTtr4Iq91N_To69-KseWxu9suz3uY1f-NRDkykc2zP86pSXHnV9HCD-BWRPhPqYLseRlZKUFJXT9kjt5zbj5wUf55U233mxEAA3f9AJU2GjWv5GTzLbl2GxUMZFqopimUcX43OZ_dELc_j-VIcR4K9iAFU2_p01qLP1YebHpQRc7wy_LTXPIsLAHjz_iE7RbipdjFQyMZEkb7Il16sjWKv1ngFQzFlgW4ZlplGbVlCztsetfhruDPV3AAYCzUziEQleaQH8hNUzadHtMoCR0jrEqmZvHQAZOLqP_ouV-h0mILCxm_Fx_DswHbk2hLk69xJw1cw6nMdU_R9Oo4x-lBdw6oQ1j1sIiuR"
HEADERS   = {"Authorization": f"Bearer {GFW_TOKEN}"}
BASE_URL  = "https://gateway.api.globalfishingwatch.org/v3"

# ── Regions ───────────────────────────────────────────────────────────────────
REGIONS = [
    {"name": "english_channel", "treated": 1,
     "lon_min": -5, "lat_min": 48, "lon_max": 2,  "lat_max": 52},
    {"name": "north_sea",       "treated": 1,
     "lon_min": 2,  "lat_min": 52, "lon_max": 8,  "lat_max": 58},
    {"name": "mid_atlantic_1",  "treated": 0,
     "lon_min": -40,"lat_min": 35, "lon_max": -30,"lat_max": 40},
    {"name": "mid_atlantic_2",  "treated": 0,
     "lon_min": -45,"lat_min": 40, "lon_max": -35,"lat_max": 45},
]

# Date chunks — API limit is 366 days per request
DATE_CHUNKS = [
    ("2018-01-01", "2018-12-31"),
    ("2019-01-01", "2019-12-31"),
    ("2020-01-01", "2020-05-31"),
]

def fetch_effort(region, start, end):
    """Fetch monthly fishing effort for a region."""
    url = f"{BASE_URL}/4wings/report"
    params = {
        "datasets[0]":         "public-global-fishing-effort:latest",
        "date-range":          f"{start},{end}",
        "spatial-resolution":  "LOW",
        "temporal-resolution": "MONTHLY",
        "group-by":            "FLAG",
        "format":              "JSON",
    }
    body = {
        "geojson": {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [region["lon_min"], region["lat_min"]],
                    [region["lon_max"], region["lat_min"]],
                    [region["lon_max"], region["lat_max"]],
                    [region["lon_min"], region["lat_max"]],
                    [region["lon_min"], region["lat_min"]],
                ]]
            }
        }
    }

    for attempt in range(3):
        try:
            r = requests.post(url, params=params, json=body,
                              headers=HEADERS, timeout=60)
            if r.status_code == 200:
                return r.json()
            elif r.status_code == 429:
                print(f"    Rate limited, waiting 30s...")
                time.sleep(30)
            else:
                print(f"    Error {r.status_code}: {r.text[:200]}")
                return None
        except Exception as e:
            print(f"    Exception: {e}")
            time.sleep(5)
    return None

def parse_response(result, region, start, end):
    """Parse GFW response into flat records."""
    records = []
    if not result or "entries" not in result:
        return records

    entries = result["entries"]
    if not entries:
        return records

    # Entries is a list with one item containing the dataset key
    dataset_key = list(entries[0].keys())[0] if entries else None
    if not dataset_key:
        return records

    data_points = entries[0][dataset_key]

    for point in data_points:
        records.append({
            "region":    region["name"],
            "treated":   region["treated"],
            "lat":       point.get("lat"),
            "lon":       point.get("lon"),
            "date":      point.get("date"),       # YYYY-MM format
            "flag":      point.get("flag", "UNK"),
            "hours":     point.get("hours", 0),
            "vessel_ids": point.get("vesselIDs", 0),
            "period_start": start,
            "period_end":   end,
        })
    return records

def aggregate_to_weekly_panel(df):
    """
    Convert monthly GFW data to weekly panel matching our simulation structure.
    Aggregates hours per region per month, then distributes to weeks.
    """
    # Parse date
    df["year"]  = df["date"].str[:4].astype(int)
    df["month"] = df["date"].str[5:7].astype(int)

    # Total hours per region per month (sum across flags/cells)
    monthly = (df.groupby(["region","treated","year","month"])
                 .agg(total_hours=("hours","sum"),
                      n_cells=("lat","count"),
                      n_vessels=("vessel_ids","sum"))
                 .reset_index())

    # Expand to weekly — 4 weeks per month (approximate)
    weekly_records = []
    week_counter = {}

    for _, row in monthly.iterrows():
        region_key = row["region"]
        if region_key not in week_counter:
            week_counter[region_key] = 0

        for w in range(4):
            week_counter[region_key] += 1
            # Distribute monthly hours evenly across 4 weeks
            weekly_records.append({
                "region":         row["region"],
                "treated":        row["treated"],
                "year":           row["year"],
                "month":          row["month"],
                "week_of_month":  w + 1,
                "week_index":     week_counter[region_key],
                "vessel_hours":   row["total_hours"] / 4,
                "n_cells":        row["n_cells"],
                "n_vessels":      row["n_vessels"] / 4,
            })

    return pd.DataFrame(weekly_records)

if __name__ == "__main__":
    print("── GFW Real AIS Data Download ───────────────")
    print(f"Regions: {len(REGIONS)} | Date chunks: {len(DATE_CHUNKS)}\n")

    all_records = []

    for region in REGIONS:
        print(f"Region: {region['name']} (treated={region['treated']})")
        for start, end in DATE_CHUNKS:
            print(f"  Period: {start} → {end}")
            result = fetch_effort(region, start, end)
            if result:
                records = parse_response(result, region, start, end)
                all_records.extend(records)
                print(f"  ✓ {len(records)} data points")
            else:
                print(f"  ✗ No data")
            time.sleep(1)  # rate limit buffer
        print()

    if not all_records:
        print("✗ No data downloaded. Check token and permissions.")
        exit(1)

    # Save raw
    raw_df = pd.DataFrame(all_records)
    # Save raw records WITH lat/lon to DuckDB
    con = duckdb.connect("data/midnight_zone.duckdb")
    con.execute("DROP TABLE IF EXISTS raw_ais_full")
    con.execute("CREATE TABLE raw_ais_full AS SELECT * FROM raw_df")
    con.close()
    print(f" Saved raw_ais_full to DuckDB: {len(raw_df):,} rows")
    os.makedirs("data", exist_ok=True)
    raw_df.to_csv("data/ais_raw.csv", index=False)
    print(f" Raw data saved: {len(raw_df):,} records")
    print(f"  Columns: {list(raw_df.columns)}")
    print(f"\nSample:")
    print(raw_df.head(3).to_string())

    # Aggregate to weekly panel
    weekly = aggregate_to_weekly_panel(raw_df)
    weekly.to_csv("data/ais_weekly.csv", index=False)
    print(f"\n✓ Weekly panel: {len(weekly):,} rows")
    print(f"  Date range: {weekly['year'].min()} - {weekly['year'].max()}")
    print(f"  Regions: {weekly['region'].unique()}")

    # Load into DuckDB
    con = duckdb.connect("data/midnight_zone.duckdb")
    con.execute("DROP TABLE IF EXISTS raw_ais")
    con.execute("CREATE TABLE raw_ais AS SELECT * FROM weekly")
    count = con.execute("SELECT COUNT(*) FROM raw_ais").fetchone()[0]
    con.close()
    print(f"\n Loaded into DuckDB: raw_ais ({count:,} rows)")
    print("\n AIS download complete.")