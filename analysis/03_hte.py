"""
03_hte.py
---------
CausalForestDML HTE for BOTH panels.
"""

import duckdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings, os
warnings.filterwarnings("ignore")

from econml.dml import CausalForestDML
from sklearn.ensemble import GradientBoostingRegressor

os.makedirs("outputs", exist_ok=True)

CONFOUNDERS = ["sst","log_chla","bathymetry","season_sin","season_cos"]
EFFECT_MODS = ["bathymetry","sst"]

def run_hte(panel_name, df, label, color):
    print(f"\n{'='*50}")
    print(f"Running HTE — {label}")
    print(f"{'='*50}")

    df = df.dropna(subset=["dvm_amplitude","vessel_density"] + CONFOUNDERS)

    if len(df) > 30000:
        df = df.sample(30000, random_state=42)
        print(f"Sampled to 30,000 rows")

    Y = df["dvm_amplitude"].values
    T = df["vessel_density"].values
    X = df[EFFECT_MODS].values
    W = df[CONFOUNDERS].values

    print("Fitting CausalForestDML...")
    cf = CausalForestDML(
        model_y=GradientBoostingRegressor(n_estimators=100, max_depth=3,
                                          learning_rate=0.1, random_state=42),
        model_t=GradientBoostingRegressor(n_estimators=100, max_depth=3,
                                          learning_rate=0.1, random_state=42),
        discrete_treatment=False,
        n_estimators=200,
        min_samples_leaf=20,
        max_depth=4,
        random_state=42,
        cv=3
    )
    cf.fit(Y, T, X=X, W=W)

    ite            = cf.effect(X)
    ite_inf        = cf.effect_inference(X)
    ite_lo, ite_hi = ite_inf.conf_int()

    df = df.copy()
    df["ite"]    = ite
    df["ite_lo"] = ite_lo
    df["ite_hi"] = ite_hi

    print(f"Mean ITE:  {ite.mean():.4f}")
    print(f"ITE std:   {ite.std():.4f}")
    print(f"ITE range: [{ite.min():.4f}, {ite.max():.4f}]")

    depth_stats = df.groupby("depth_zone").agg(
        mean_ite=("ite","mean"),
        std_ite=("ite","std"),
        n=("ite","count")
    ).reset_index()
    print(f"\nHTE by depth zone:")
    print(depth_stats.to_string(index=False))

    # ── Plot ──────────────────────────────────────────────────────────────────
    zone_colors = {"shallow":"#d97706","mid":"#0f6e56","deep":"#1a56a0"}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#f8fafc")
    fig.suptitle(f"HTE by Depth Zone — {label}", fontsize=13, fontweight="bold")

    ax1 = axes[0]
    ax1.set_facecolor("#f8fafc")
    for zone, col in zone_colors.items():
        sub = df[df.depth_zone == zone]["ite"]
        if len(sub) > 0:
            ax1.hist(sub, bins=40, alpha=0.65, color=col,
                     label=f"{zone} (n={len(sub):,})",
                     edgecolor="white", linewidth=0.5)
            ax1.axvline(sub.mean(), color=col, lw=2, linestyle="--")
    ax1.axvline(0, color="#9ca3af", lw=1, linestyle=":")
    ax1.set_xlabel("Individual Treatment Effect"); ax1.set_ylabel("Count")
    ax1.set_title("ITE Distribution by Depth Zone", fontsize=11, fontweight="bold")
    ax1.legend(fontsize=8)
    ax1.spines[["top","right"]].set_visible(False)
    ax1.grid(axis="y", alpha=0.3)

    ax2 = axes[1]
    ax2.set_facecolor("#f8fafc")
    zones = ["shallow","mid","deep"]
    y_pos = [2,1,0]
    for i, zone in enumerate(zones):
        sub = df[df.depth_zone == zone]
        if len(sub) == 0:
            continue
        m  = sub["ite"].mean()
        lo = sub["ite_lo"].mean()
        hi = sub["ite_hi"].mean()
        col = zone_colors[zone]
        ax2.scatter(m, y_pos[i], color=col, s=140, zorder=5)
        ax2.plot([lo,hi],[y_pos[i],y_pos[i]], color=col, lw=3, alpha=0.6)
        ax2.text(m, y_pos[i]+0.2, f"{m:.4f}", ha="center",
                 fontsize=9, fontweight="bold", color=col)

    ax2.axvline(0, color="#9ca3af", lw=1, linestyle=":")
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(["Deep (>1500m)","Mid (500-1500m)","Shallow (<500m)"],
                        fontsize=9)
    ax2.set_xlabel("Mean ITE (95% CI)")
    ax2.set_title("Mean Effect by Depth Zone", fontsize=11, fontweight="bold")
    ax2.spines[["top","right"]].set_visible(False)
    ax2.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"outputs/03_hte_{panel_name}.png", dpi=150,
                bbox_inches="tight", facecolor="#f8fafc")
    plt.close()
    print(f"✓ Saved: outputs/03_hte_{panel_name}.png")

    cell_level = df.groupby("cell_id").agg(
        mean_ite=("ite","mean"),
        bathymetry=("bathymetry","first"),
        depth_zone=("depth_zone","first"),
    ).reset_index()
    cell_level.to_csv(f"outputs/hte_results_{panel_name}.csv", index=False)
    df[["cell_id","ite","ite_lo","ite_hi","depth_zone",
        "bathymetry","vessel_density","dvm_amplitude",
        "is_treated","is_post_covid"]].to_csv(
        f"outputs/hte_panel_{panel_name}.csv", index=False)
    print(f"✓ Saved: outputs/hte_results_{panel_name}.csv")

# ── Load and run ──────────────────────────────────────────────────────────────
con = duckdb.connect("data/midnight_zone.duckdb")
df_real = con.execute("SELECT * FROM mart_analysis").df()
df_sim  = con.execute("SELECT * FROM mart_analysis_sim").df()
con.close()

run_hte("real", df_real, "Panel A — Real AIS Fishing Effort", "#0f6e56")
run_hte("sim",  df_sim,  "Panel B — Simulated COVID Shock",   "#7c3aed")

print("\n✓ Block 3 complete. All analysis done.")