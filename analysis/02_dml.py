"""
02_dml.py
---------
Double/Debiased ML + DoWhy refutations for BOTH panels.
"""

import duckdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings, os
warnings.filterwarnings("ignore")

from econml.dml import LinearDML
from sklearn.ensemble import GradientBoostingRegressor
from dowhy import CausalModel

os.makedirs("outputs", exist_ok=True)

CONFOUNDERS = ["sst", "log_chla", "bathymetry", "season_sin", "season_cos"]

def run_dml(panel_name, df, label, color, true_ate=-0.35):
    print(f"\n{'='*50}")
    print(f"Running DML — {label}")
    print(f"{'='*50}")
    print(f"Loaded: {len(df):,} rows")

    # Sample for speed if too large
    if len(df) > 50000:
        df = df.sample(50000, random_state=42)
        print(f"Sampled to 50,000 rows for DML speed")

    df = df.dropna(subset=["dvm_amplitude","vessel_density"] + CONFOUNDERS)
    Y = df["dvm_amplitude"].values
    T = df["vessel_density"].values
    W = df[CONFOUNDERS].values

    # ── LinearDML ─────────────────────────────────────────────────────────────
    print("Fitting LinearDML...")
    dml = LinearDML(
        model_y=GradientBoostingRegressor(n_estimators=100, max_depth=3,
                                          learning_rate=0.1, random_state=42),
        model_t=GradientBoostingRegressor(n_estimators=100, max_depth=3,
                                          learning_rate=0.1, random_state=42),
        discrete_treatment=False,
        cv=3,
        random_state=42
    )
    dml.fit(Y, T, X=None, W=W)

    inf       = dml.ate_inference()
    ate_point = inf.mean_point
    ate_se    = inf.stderr_mean
    ate_ci    = inf.conf_int_mean()

    print(f"DML ATE:         {ate_point:.4f}")
    print(f"Std error:       {ate_se:.4f}")
    print(f"95% CI:          [{ate_ci[0]:.4f}, {ate_ci[1]:.4f}]")
    print(f"True ATE (DGP):  {true_ate}")
    print(f"Recovery error:  {abs(ate_point - true_ate) / abs(true_ate) * 100:.2f}%")

    # ── DoWhy refutations ─────────────────────────────────────────────────────
    print("\nRunning DoWhy refutations...")
    sample = df.sample(min(2000, len(df)), random_state=42)
    dowhy_df = sample.groupby("cell_id").agg(
        vessel_density=("vessel_density","mean"),
        dvm_amplitude=("dvm_amplitude","mean"),
        sst=("sst","mean"),
        log_chla=("log_chla","mean"),
        bathymetry=("bathymetry","mean"),
        season_sin=("season_sin","mean"),
    ).reset_index()

    causal_model = CausalModel(
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

    estimand = causal_model.identify_effect(proceed_when_unidentifiable=True)
    estimate = causal_model.estimate_effect(estimand,
                   method_name="backdoor.linear_regression")
    base = estimate.value
    print(f"DoWhy base: {base:.4f}")

    print("Refutation 1: random common cause...")
    ref1 = causal_model.refute_estimate(estimand, estimate,
               method_name="random_common_cause", random_seed=42)
    print("Refutation 2: placebo treatment...")
    ref2 = causal_model.refute_estimate(estimand, estimate,
               method_name="placebo_treatment_refuter",
               placebo_type="permute", random_seed=42, num_simulations=50)
    print("Refutation 3: data subset...")
    ref3 = causal_model.refute_estimate(estimand, estimate,
               method_name="data_subset_refuter",
               subset_fraction=0.8, random_seed=42, num_simulations=50)

    r1, r2, r3 = ref1.new_effect, ref2.new_effect, ref3.new_effect
    print(f"\nBase:              {base:.4f}")
    print(f"[1] Random cause:  {r1:.4f}  {'✓' if abs(r1-base)<abs(base)*0.1 else '✗'}")
    print(f"[2] Placebo:       {r2:.4f}  {'✓' if abs(r2)<abs(base)*0.3 else '✗'}")
    print(f"[3] Subset 80%:    {r3:.4f}  {'✓' if abs(r3-base)<abs(base)*0.1 else '✗'}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    did_res  = pd.read_csv(f"outputs/did_results_{panel_name}.csv")
    did_ate  = did_res["ate_estimate"].iloc[0]
    did_ci_lo = did_res["ci_lower"].iloc[0]
    did_ci_hi = did_res["ci_upper"].iloc[0]

    TEAL = "#0f6e56"; BLUE = "#1a56a0"; GRAY = "#6b7280"; RED = "#dc2626"

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#f8fafc")
    fig.suptitle(f"DML Results — {label}", fontsize=13, fontweight="bold")

    ax1 = axes[0]
    ax1.set_facecolor("#f8fafc")
    methods   = ["True ATE\n(DGP)", "TWFE DiD\n(baseline)", "DML\n(EconML)"]
    estimates = [true_ate, did_ate, ate_point]
    ci_lo     = [None, did_ci_lo if not np.isnan(did_ci_lo) else ate_point, ate_ci[0]]
    ci_hi     = [None, did_ci_hi if not np.isnan(did_ci_hi) else ate_point, ate_ci[1]]
    colors    = [GRAY, BLUE, color]

    for i, (e, lo, hi, col) in enumerate(zip(estimates, ci_lo, ci_hi, colors)):
        ax1.scatter(e, i, color=col, s=120, zorder=5)
        if lo is not None:
            ax1.plot([lo, hi], [i, i], color=col, lw=2.5, alpha=0.7)
        ax1.text(e + 0.01, i + 0.15, f"{e:.4f}", fontsize=9,
                 color=col, fontweight="bold")

    ax1.axvline(0, color=GRAY, lw=1, linestyle=":")
    ax1.set_yticks([0,1,2]); ax1.set_yticklabels(methods, fontsize=9)
    ax1.set_xlabel("ATE Estimate")
    ax1.set_title("Method Comparison", fontsize=11, fontweight="bold")
    ax1.spines[["top","right"]].set_visible(False)
    ax1.grid(axis="x", alpha=0.3)

    ax2 = axes[1]
    ax2.set_facecolor("#f8fafc")
    ref_labels = ["Base", "[1] Random\ncause", "[2] Placebo", "[3] Subset\n80%"]
    ref_values = [base, r1, r2, r3]
    ref_colors = [BLUE,
                  TEAL if abs(r1-base)<abs(base)*0.1 else RED,
                  TEAL if abs(r2)<abs(base)*0.3 else RED,
                  TEAL if abs(r3-base)<abs(base)*0.1 else RED]

    bars = ax2.bar(ref_labels, ref_values, color=ref_colors,
                   width=0.5, alpha=0.85, edgecolor="white")
    ax2.axhline(base, color=BLUE, lw=1.5, linestyle="--", alpha=0.5)
    ax2.axhline(0, color=GRAY, lw=1, linestyle=":")
    for bar, val in zip(bars, ref_values):
        ax2.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.005,
                 f"{val:.3f}", ha="center", fontsize=9, fontweight="bold")
    ax2.set_title("DoWhy Refutations", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Effect estimate")
    ax2.spines[["top","right"]].set_visible(False)
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"outputs/02_dml_{panel_name}.png", dpi=150,
                bbox_inches="tight", facecolor="#f8fafc")
    plt.close()
    print(f"✓ Saved: outputs/02_dml_{panel_name}.png")

    pd.DataFrame([{
        "panel": panel_name, "method": "LinearDML",
        "ate_estimate": ate_point, "std_error": ate_se,
        "ci_lower": ate_ci[0], "ci_upper": ate_ci[1],
        "true_ate": true_ate,
        "dowhy_base": base,
        "ref1_random_cause": r1,
        "ref2_placebo": r2,
        "ref3_subset": r3,
    }]).to_csv(f"outputs/dml_results_{panel_name}.csv", index=False)
    print(f"✓ Saved: outputs/dml_results_{panel_name}.csv")

# ── Load and run ──────────────────────────────────────────────────────────────
con = duckdb.connect("data/midnight_zone.duckdb")
df_real = con.execute("SELECT * FROM mart_analysis").df()
df_sim  = con.execute("SELECT * FROM mart_analysis_sim").df()
con.close()

run_dml("real", df_real, "Panel A — Real AIS Fishing Effort", "#0f6e56")
run_dml("sim",  df_sim,  "Panel B — Simulated COVID Shock",   "#7c3aed")

print("\n✓ Block 2 complete. Run analysis/03_hte.py next.")