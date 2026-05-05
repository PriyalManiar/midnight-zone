"""
02_dml.py
---------
Double/Debiased ML + DoWhy refutation tests.

Outputs:
  outputs/02a_dml_refutations.png
  outputs/dml_results.csv
"""

import duckdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings, os
warnings.filterwarnings("ignore")

from econml.dml import LinearDML
from sklearn.ensemble import GradientBoostingRegressor
import dowhy
from dowhy import CausalModel

os.makedirs("outputs", exist_ok=True)

con = duckdb.connect("data/midnight_zone.duckdb")
df  = con.execute("SELECT * FROM mart_analysis").df()
con.close()
print(f"Loaded: {len(df):,} rows")

# ── Feature setup ────────────────────────────────────────────────────────────
CONFOUNDERS = ["sst", "log_chla", "bathymetry", "season_sin", "season_cos"]
Y = df["dvm_amplitude"].values
T = df["vessel_density"].values
W = df[CONFOUNDERS].values

# ── LinearDML ────────────────────────────────────────────────────────────────
print("\n── Double/Debiased ML (EconML LinearDML) ────")

dml = LinearDML(
    model_y=GradientBoostingRegressor(n_estimators=200, max_depth=4,
                                      learning_rate=0.05, random_state=42),
    model_t=GradientBoostingRegressor(n_estimators=200, max_depth=4,
                                      learning_rate=0.05, random_state=42),
    discrete_treatment=False,
    cv=5,
    random_state=42
)
dml.fit(Y, T, X=None, W=W)

inf       = dml.ate_inference()
ate_point = inf.mean_point
ate_se    = inf.stderr_mean
ate_ci    = inf.conf_int_mean()

print(f"DML ATE:          {ate_point:.4f}")
print(f"Std error:        {ate_se:.4f}")
print(f"95% CI:           [{ate_ci[0]:.4f}, {ate_ci[1]:.4f}]")
print(f"True ATE (DGP):   -0.35")
print(f"Recovery error:   {abs(ate_point - (-0.35)) / 0.35 * 100:.2f}%")

# ── DoWhy ────────────────────────────────────────────────────────────────────
print("\n── DoWhy Refutation Tests ───────────────────")

dowhy_df = df.groupby("cell_id").agg(
    vessel_density=("vessel_density","mean"),
    dvm_amplitude=("dvm_amplitude","mean"),
    sst=("sst","mean"),
    log_chla=("log_chla","mean"),
    bathymetry=("bathymetry","mean"),
    season_sin=("season_sin","mean"),
).reset_index()

model = CausalModel(
    data=dowhy_df,
    treatment="vessel_density",
    outcome="dvm_amplitude",
    common_causes=["sst","log_chla","bathymetry","season_sin"],
    graph="""digraph {
        vessel_density -> dvm_amplitude;
        sst -> vessel_density; sst -> dvm_amplitude;
        log_chla -> dvm_amplitude;
        bathymetry -> vessel_density; bathymetry -> dvm_amplitude;
        season_sin -> vessel_density; season_sin -> dvm_amplitude;
    }"""
)

estimand = model.identify_effect(proceed_when_unidentifiable=True)
estimate = model.estimate_effect(estimand,
               method_name="backdoor.linear_regression")
base = estimate.value
print(f"DoWhy base estimate: {base:.4f}")

print("Running refutation 1: random common cause...")
ref1 = model.refute_estimate(estimand, estimate,
           method_name="random_common_cause", random_seed=42)

print("Running refutation 2: placebo treatment...")
ref2 = model.refute_estimate(estimand, estimate,
           method_name="placebo_treatment_refuter",
           placebo_type="permute", random_seed=42, num_simulations=100)

print("Running refutation 3: data subset (80%)...")
ref3 = model.refute_estimate(estimand, estimate,
           method_name="data_subset_refuter",
           subset_fraction=0.8, random_seed=42, num_simulations=100)

r1, r2, r3 = ref1.new_effect, ref2.new_effect, ref3.new_effect

print(f"\n── Results ──────────────────────────────────")
print(f"Base estimate:           {base:.4f}")
print(f"[1] Random common cause: {r1:.4f}  {'✓ stable' if abs(r1-base) < abs(base)*0.1 else '✗ changed'}")
print(f"[2] Placebo treatment:   {r2:.4f}  {'✓ collapsed' if abs(r2) < abs(base)*0.3 else '✗ did not collapse'}")
print(f"[3] Data subset 80%:     {r3:.4f}  {'✓ stable' if abs(r3-base) < abs(base)*0.1 else '✗ changed'}")

# ── Plot ─────────────────────────────────────────────────────────────────────
did_res   = pd.read_csv("outputs/did_results.csv")
did_ate   = did_res["ate_estimate"].iloc[0]
did_ci_lo = did_res["ci_lower"].iloc[0]
did_ci_hi = did_res["ci_upper"].iloc[0]

TEAL  = "#0f6e56"
BLUE  = "#1a56a0"
AMBER = "#d97706"
RED   = "#dc2626"
GRAY  = "#6b7280"

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor("#f8fafc")

# Left: method comparison
ax1 = axes[0]
ax1.set_facecolor("#f8fafc")
methods   = ["True ATE\n(DGP)", "TWFE DiD\n(baseline)", "DML\n(EconML)"]
estimates = [-0.35, did_ate, ate_point]
ci_lo     = [None, did_ci_lo, ate_ci[0]]
ci_hi     = [None, did_ci_hi, ate_ci[1]]
colors    = [GRAY, BLUE, TEAL]

for i, (e, lo, hi, col) in enumerate(zip(estimates, ci_lo, ci_hi, colors)):
    ax1.scatter(e, i, color=col, s=120, zorder=5)
    if lo is not None:
        ax1.plot([lo, hi], [i, i], color=col, lw=2.5, alpha=0.7)
    ax1.text(e + 0.1, i, f"{e:.3f}", va="center", fontsize=9,
             color=col, fontweight="bold")

ax1.axvline(0, color=GRAY, lw=1, linestyle=":")
ax1.set_yticks([0,1,2])
ax1.set_yticklabels(methods, fontsize=10)
ax1.set_xlabel("ATE Estimate", fontsize=10)
ax1.set_title("Method Comparison: DiD vs DML\n(DML corrects nonlinear confounding)",
              fontsize=11, fontweight="bold")
ax1.spines[["top","right"]].set_visible(False)
ax1.grid(axis="x", alpha=0.3)

# Right: refutation bars
ax2 = axes[1]
ax2.set_facecolor("#f8fafc")
labels = ["Base\nestimate", "[1] Random\ncommon cause",
          "[2] Placebo\ntreatment", "[3] Data\nsubset 80%"]
values = [base, r1, r2, r3]
cols   = [BLUE,
          TEAL if abs(r1-base) < abs(base)*0.1 else RED,
          TEAL if abs(r2)      < abs(base)*0.3  else RED,
          TEAL if abs(r3-base) < abs(base)*0.1  else RED]

bars = ax2.bar(labels, values, color=cols, width=0.55,
               alpha=0.85, edgecolor="white", linewidth=1.5)
ax2.axhline(base, color=BLUE, lw=1.5, linestyle="--", alpha=0.5,
            label=f"Base: {base:.3f}")
ax2.axhline(0, color=GRAY, lw=1, linestyle=":")
for bar, val in zip(bars, values):
    ax2.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.01 if val >= 0 else bar.get_height() - 0.04,
             f"{val:.3f}", ha="center", fontsize=9, fontweight="bold")

ax2.set_title("DoWhy Refutation Tests\n(placebo → 0, others stable = robust)",
              fontsize=11, fontweight="bold")
ax2.set_ylabel("Effect estimate", fontsize=10)
ax2.legend(fontsize=9)
ax2.spines[["top","right"]].set_visible(False)
ax2.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("outputs/02a_dml_refutations.png", dpi=150,
            bbox_inches="tight", facecolor="#f8fafc")
plt.close()
print("\n✓ Saved: outputs/02a_dml_refutations.png")

# ── Save results ─────────────────────────────────────────────────────────────
pd.DataFrame([{
    "method":             "LinearDML",
    "ate_estimate":       ate_point,
    "std_error":          ate_se,
    "ci_lower":           ate_ci[0],
    "ci_upper":           ate_ci[1],
    "true_ate":           -0.35,
    "dowhy_base":         base,
    "ref1_random_cause":  r1,
    "ref2_placebo":       r2,
    "ref3_subset":        r3,
}]).to_csv("outputs/dml_results.csv", index=False)

print(" Saved: outputs/dml_results.csv")
print(" Block 3 done. Paste output here before next step.")