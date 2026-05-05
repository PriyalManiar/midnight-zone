"""
03_hte.py
---------
Heterogeneous Treatment Effects via CausalForestDML.
Does the DVM recovery effect vary by bathymetry depth zone?

Outputs:
  outputs/03a_hte_by_depth.png
  outputs/03b_hte_bathymetry.png
  outputs/hte_results.csv
  outputs/hte_panel.csv
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

con = duckdb.connect("data/midnight_zone.duckdb")
df  = con.execute("SELECT * FROM mart_analysis").df()
con.close()
print(f"Loaded: {len(df):,} rows")

CONFOUNDERS = ["sst", "log_chla", "bathymetry", "season_sin", "season_cos"]
EFFECT_MODS = ["bathymetry", "sst"]

Y = df["dvm_amplitude"].values
T = df["vessel_density"].values
X = df[EFFECT_MODS].values
W = df[CONFOUNDERS].values

# ── CausalForestDML ──────────────────────────────────────────────────────────
print("\n── CausalForestDML ──────────────────────────")

cf = CausalForestDML(
    model_y=GradientBoostingRegressor(n_estimators=150, max_depth=4,
                                      learning_rate=0.05, random_state=42),
    model_t=GradientBoostingRegressor(n_estimators=150, max_depth=4,
                                      learning_rate=0.05, random_state=42),
    discrete_treatment=False,
    n_estimators=500,
    min_samples_leaf=20,
    max_depth=4,
    random_state=42,
    cv=3
)
cf.fit(Y, T, X=X, W=W)

ite            = cf.effect(X)
ite_inf        = cf.effect_inference(X)
ite_lo, ite_hi = ite_inf.conf_int()

df["ite"]    = ite
df["ite_lo"] = ite_lo
df["ite_hi"] = ite_hi

print(f"Mean ITE:  {ite.mean():.4f}")
print(f"ITE std:   {ite.std():.4f}")
print(f"ITE range: [{ite.min():.4f}, {ite.max():.4f}]")

depth_stats = df.groupby("depth_zone").agg(
    mean_ite=("ite","mean"),
    std_ite=("ite","std"),
    n=("ite","count"),
    mean_bath=("bathymetry","mean")
).reset_index()

print(f"\n── HTE by Depth Zone ────────────────────────")
print(depth_stats.to_string(index=False))

# ── Plot A: distributions by depth zone ─────────────────────────────────────
TEAL  = "#0f6e56"
BLUE  = "#1a56a0"
AMBER = "#d97706"
GRAY  = "#6b7280"
RED   = "#dc2626"

zone_colors = {"shallow": AMBER, "mid": TEAL, "deep": BLUE}
zone_labels = {
    "shallow": "Shallow (<500m) — shelf corridors",
    "mid":     "Mid (500–1500m) — mesopelagic zone",
    "deep":    "Deep (>1500m) — abyssal"
}

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor("#f8fafc")

ax1 = axes[0]
ax1.set_facecolor("#f8fafc")
for zone in ["shallow","mid","deep"]:
    sub = df[df.depth_zone == zone]["ite"]
    ax1.hist(sub, bins=40, alpha=0.65, color=zone_colors[zone],
             label=zone_labels[zone], edgecolor="white", linewidth=0.5)
    ax1.axvline(sub.mean(), color=zone_colors[zone], lw=2, linestyle="--")

ax1.axvline(0, color=GRAY, lw=1, linestyle=":")
ax1.set_xlabel("Individual Treatment Effect (ITE)", fontsize=10)
ax1.set_ylabel("Count", fontsize=10)
ax1.set_title("HTE Distribution by Depth Zone", fontsize=11, fontweight="bold")
ax1.legend(fontsize=8); ax1.spines[["top","right"]].set_visible(False)
ax1.grid(axis="y", alpha=0.3)

# Right: mean ITE by zone with CI
ax2 = axes[1]
ax2.set_facecolor("#f8fafc")
zones = ["shallow","mid","deep"]
y_pos = [2, 1, 0]
for i, zone in enumerate(zones):
    sub    = df[df.depth_zone == zone]
    m      = sub["ite"].mean()
    lo     = sub["ite_lo"].mean()
    hi     = sub["ite_hi"].mean()
    col    = zone_colors[zone]
    ax2.scatter(m, y_pos[i], color=col, s=140, zorder=5)
    ax2.plot([lo, hi], [y_pos[i], y_pos[i]], color=col, lw=3, alpha=0.6)
    ax2.text(m - 0.001, y_pos[i] + 0.15, f"{m:.4f}",
             ha="center", fontsize=9, fontweight="bold", color=col)

ax2.axvline(0, color=GRAY, lw=1, linestyle=":")
ax2.set_yticks(y_pos)
ax2.set_yticklabels([zone_labels[z] for z in zones], fontsize=8)
ax2.set_xlabel("Mean ITE (95% CI)", fontsize=10)
ax2.set_title("Mean Treatment Effect by Depth Zone", fontsize=11, fontweight="bold")
ax2.spines[["top","right"]].set_visible(False)
ax2.grid(axis="x", alpha=0.3)

plt.tight_layout()
plt.savefig("outputs/03a_hte_by_depth.png", dpi=150,
            bbox_inches="tight", facecolor="#f8fafc")
plt.close()
print("\n✓ Saved: outputs/03a_hte_by_depth.png")

# ── Plot B: ITE vs bathymetry scatter ────────────────────────────────────────
cell_level = df.groupby("cell_id").agg(
    mean_ite=("ite","mean"),
    bathymetry=("bathymetry","first"),
    depth_zone=("depth_zone","first"),
).reset_index()

fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_facecolor("#f8fafc")
ax.set_facecolor("#f8fafc")

for zone in ["shallow","mid","deep"]:
    sub = cell_level[cell_level.depth_zone == zone]
    ax.scatter(sub.bathymetry, sub.mean_ite, color=zone_colors[zone],
               alpha=0.75, s=70, label=zone_labels[zone],
               edgecolors="white", linewidth=0.5)

from numpy.polynomial.polynomial import polyfit
x_all = cell_level.bathymetry.values
y_all = cell_level.mean_ite.values
c     = polyfit(x_all, y_all, 2)
x_r   = np.linspace(x_all.min(), x_all.max(), 200)
y_fit = c[0] + c[1]*x_r + c[2]*x_r**2
ax.plot(x_r, y_fit, color=RED, lw=2, linestyle="--", alpha=0.8,
        label="Polynomial trend")

ax.axhline(0, color=GRAY, lw=1, linestyle=":")
ax.set_xlabel("Bathymetry (m depth)", fontsize=10)
ax.set_ylabel("Mean ITE per cell", fontsize=10)
ax.set_title("Treatment Effect Heterogeneity vs Depth\n"
             "Shallower corridors show marginally stronger DVM recovery",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=8); ax.spines[["top","right"]].set_visible(False)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("outputs/03b_hte_bathymetry.png", dpi=150,
            bbox_inches="tight", facecolor="#f8fafc")
plt.close()
print("✓ Saved: outputs/03b_hte_bathymetry.png")

# ── Save ─────────────────────────────────────────────────────────────────────
cell_level.to_csv("outputs/hte_results.csv", index=False)
df[["cell_id","week_index","ite","ite_lo","ite_hi","depth_zone",
    "bathymetry","vessel_density","dvm_amplitude",
    "is_treated","is_post_covid"]].to_csv("outputs/hte_panel.csv", index=False)

print(" Saved: outputs/hte_results.csv + hte_panel.csv")
print(" Block 4 done. Paste output here before next step.")