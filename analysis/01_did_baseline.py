"""
01_did_baseline.py
------------------
DiD baseline + parallel trends (visual + formal slope test).

Outputs:
  outputs/01a_parallel_trends.png
  outputs/01b_did_summary.png
  outputs/did_results.csv
"""

import duckdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.api as sm
import os, warnings
warnings.filterwarnings("ignore")

os.makedirs("outputs", exist_ok=True)

con = duckdb.connect("data/midnight_zone.duckdb")
df  = con.execute("SELECT * FROM mart_analysis").df()
con.close()
print(f"Loaded: {len(df):,} rows")

# ── Weekly group means ───────────────────────────────────────────────────────
weekly = (df.groupby(["week_index","observation_date","is_treated"])
            .agg(mean_dvm=("dvm_amplitude","mean"),
                 mean_density=("vessel_density","mean"))
            .reset_index())

weekly["observation_date"] = pd.to_datetime(weekly["observation_date"])
w_t = weekly[weekly.is_treated == 1].sort_values("week_index")
w_c = weekly[weekly.is_treated == 0].sort_values("week_index")

pre_t = w_t[w_t.observation_date < "2020-01-01"]
pre_c = w_c[w_c.observation_date < "2020-01-01"]

# ── Formal parallel trends slope test ───────────────────────────────────────
merged = pre_t[["week_index","mean_dvm"]].merge(
    pre_c[["week_index","mean_dvm"]], on="week_index", suffixes=("_t","_c")
)
merged["diff"] = merged["mean_dvm_t"] - merged["mean_dvm_c"]
merged["t"]    = range(len(merged))

X           = sm.add_constant(merged["t"])
slope_model = sm.OLS(merged["diff"], X).fit()
slope       = slope_model.params["t"]
slope_pval  = slope_model.pvalues["t"]
slope_ci    = slope_model.conf_int().loc["t"]

print(f"\n── Formal Parallel Trends Test ──────────────")
print(f"Slope:   {slope:.5f}")
print(f"P-value: {slope_pval:.4f}")
print(f"95% CI:  [{slope_ci[0]:.5f}, {slope_ci[1]:.5f}]")
if slope_pval > 0.05:
    print("✓ PASS: Parallel trends supported")
else:
    print("✗ WARN: Significant pre-trend divergence")

# ── Plot parallel trends ─────────────────────────────────────────────────────
TEAL  = "#0f6e56"
BLUE  = "#1a56a0"
RED   = "#dc2626"
AMBER = "#d97706"
GRAY  = "#9ca3af"

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor("#f8fafc")

ax1 = axes[0]
ax1.set_facecolor("#f8fafc")
ax1.plot(w_t.observation_date, w_t.mean_dvm, color=TEAL, lw=1.5,
         label="Treated (high-traffic corridors)")
ax1.plot(w_c.observation_date, w_c.mean_dvm, color=BLUE, lw=1.5,
         linestyle="--", label="Control (remote corridors)")
ax1.axvline(pd.Timestamp("2020-03-01"), color=RED, lw=1.5,
            linestyle=":", label="COVID shock")
ax1.axvspan(pd.Timestamp("2018-01-01"), pd.Timestamp("2020-01-01"),
            alpha=0.06, color=BLUE, label="Pre-period window")
ax1.set_title("DVM Amplitude: Treated vs Control", fontsize=12, fontweight="bold")
ax1.set_xlabel("Date"); ax1.set_ylabel("Mean DVM Amplitude Index")
ax1.legend(fontsize=8); ax1.spines[["top","right"]].set_visible(False)
ax1.grid(axis="y", alpha=0.3)

ax2 = axes[1]
ax2.set_facecolor("#f8fafc")
ax2.plot(merged["t"], merged["diff"], color=AMBER, lw=1.5,
         label="Treated − Control difference")
x_range = np.linspace(0, len(merged)-1, 100)
y_fit   = slope_model.params["const"] + slope * x_range
ax2.plot(x_range, y_fit, color=RED, lw=2, linestyle="--",
         label=f"OLS fit (slope={slope:.4f}, p={slope_pval:.3f})")
ax2.axhline(0, color=GRAY, lw=1, linestyle=":")
verdict = "✓ PASS: Parallel trends supported" if slope_pval > 0.05 else "✗ WARN: Divergence detected"
ax2.set_title(f"Formal Slope Test\n{verdict}", fontsize=12, fontweight="bold",
              color=TEAL if slope_pval > 0.05 else RED)
ax2.set_xlabel("Pre-period week"); ax2.set_ylabel("DVM difference (T − C)")
ax2.legend(fontsize=8); ax2.spines[["top","right"]].set_visible(False)
ax2.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("outputs/01a_parallel_trends.png", dpi=150, bbox_inches="tight",
            facecolor="#f8fafc")
plt.close()
print("\n✓ Saved: outputs/01a_parallel_trends.png")

# ── TWFE DiD regression ──────────────────────────────────────────────────────
print(f"\n── TWFE DiD Regression ──────────────────────")
did_model = smf.ols(
    "dvm_amplitude ~ is_treated_post + sst + log_chla + bathymetry "
    "+ season_sin + season_cos + C(cell_id) + C(week_index)",
    data=df
).fit(cov_type="HC3")

did_ate  = did_model.params["is_treated_post"]
did_se   = did_model.bse["is_treated_post"]
did_pval = did_model.pvalues["is_treated_post"]
did_ci   = did_model.conf_int().loc["is_treated_post"]

print(f"DiD ATE:   {did_ate:.4f}")
print(f"Std error: {did_se:.4f}")
print(f"P-value:   {did_pval:.6f}")
print(f"95% CI:    [{did_ci[0]:.4f}, {did_ci[1]:.4f}]")
print(f"True ATE:  -0.35")

# ── 2x2 summary table ────────────────────────────────────────────────────────
dvm_tp = df[(df.is_treated==1) & (df.is_post_covid==0)].dvm_amplitude.mean()
dvm_tc = df[(df.is_treated==1) & (df.is_post_covid==1)].dvm_amplitude.mean()
dvm_cp = df[(df.is_treated==0) & (df.is_post_covid==0)].dvm_amplitude.mean()
dvm_cc = df[(df.is_treated==0) & (df.is_post_covid==1)].dvm_amplitude.mean()
naive  = (dvm_tc - dvm_tp) - (dvm_cc - dvm_cp)

print(f"\n── 2x2 DiD Table ────────────────────────────")
print(f"Treated pre:   {dvm_tp:.3f}  |  Treated post:  {dvm_tc:.3f}  |  Δ = {dvm_tc-dvm_tp:+.3f}")
print(f"Control pre:   {dvm_cp:.3f}  |  Control post:  {dvm_cc:.3f}  |  Δ = {dvm_cc-dvm_cp:+.3f}")
print(f"Naive DiD:     {naive:.4f}")

# ── Save ─────────────────────────────────────────────────────────────────────
pd.DataFrame([{
    "method": "TWFE DiD",
    "ate_estimate": did_ate,
    "std_error": did_se,
    "p_value": did_pval,
    "ci_lower": did_ci[0],
    "ci_upper": did_ci[1],
    "true_ate": -0.35,
    "parallel_trends_slope": slope,
    "parallel_trends_pval":  slope_pval,
}]).to_csv("outputs/did_results.csv", index=False)

print("\n Saved: outputs/did_results.csv")
