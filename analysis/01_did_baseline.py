"""
01_did_baseline.py
------------------
DiD baseline + parallel trends for BOTH panels.
Uses within-transformation (demeaning) instead of dummy variables
for cell fixed effects — much faster with 13k cells.
"""

import duckdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import os, warnings
warnings.filterwarnings("ignore")

os.makedirs("outputs", exist_ok=True)

def within_transform(df, outcome, covariates, group_col):
    """Demean all variables by group (absorbs fixed effects)."""
    cols = [outcome] + covariates
    group_means = df.groupby(group_col)[cols].transform("mean")
    df_demean   = df[cols] - group_means
    return df_demean

def run_did(panel_name, df, label, color):
    print(f"\n{'='*50}")
    print(f"Running DiD — {label}")
    print(f"{'='*50}")
    print(f"Loaded: {len(df):,} rows, {df['cell_id'].nunique():,} cells")

    df["observation_date"] = pd.to_datetime(df["observation_date"])

    # ── Weekly group means ────────────────────────────────────────────────────
    weekly = (df.groupby(["week_index","observation_date","is_treated"])
                .agg(mean_dvm=("dvm_amplitude","mean"),
                     mean_density=("vessel_density","mean"))
                .reset_index())

    w_t   = weekly[weekly.is_treated==1].sort_values("week_index")
    w_c   = weekly[weekly.is_treated==0].sort_values("week_index")
    pre_t = w_t[w_t.observation_date < "2020-01-01"]
    pre_c = w_c[w_c.observation_date < "2020-01-01"]

    # ── Formal parallel trends test ───────────────────────────────────────────
    merged = pre_t[["week_index","mean_dvm"]].merge(
        pre_c[["week_index","mean_dvm"]], on="week_index", suffixes=("_t","_c"))
    merged["diff"] = merged["mean_dvm_t"] - merged["mean_dvm_c"]
    merged["t"]    = range(len(merged))

    X           = sm.add_constant(merged["t"])
    slope_model = sm.OLS(merged["diff"], X).fit()
    slope       = slope_model.params["t"]
    slope_pval  = slope_model.pvalues["t"]

    print(f"Parallel trends slope: {slope:.5f}, p={slope_pval:.4f}")
    print(f"{'✓ PASS' if slope_pval > 0.05 else '✗ WARN'}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    GRAY  = "#9ca3af"
    RED   = "#dc2626"
    AMBER = "#d97706"

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#f8fafc")
    fig.suptitle(f"Parallel Trends — {label}", fontsize=13, fontweight="bold")

    ax1 = axes[0]
    ax1.set_facecolor("#f8fafc")
    ax1.plot(w_t.observation_date, w_t.mean_dvm,
             color=color, lw=1.5, label="Treated")
    ax1.plot(w_c.observation_date, w_c.mean_dvm,
             color="#1a56a0", lw=1.5, linestyle="--", label="Control")
    ax1.axvspan(pd.Timestamp("2018-01-01"), pd.Timestamp("2020-01-01"),
                alpha=0.06, color="#1a56a0")
    ax1.axvline(pd.Timestamp("2020-01-01"), color=RED, lw=1.5, linestyle=":")
    ax1.set_title("DVM Amplitude Over Time", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Date"); ax1.set_ylabel("Mean DVM Amplitude")
    ax1.legend(fontsize=8)
    ax1.spines[["top","right"]].set_visible(False)
    ax1.grid(axis="y", alpha=0.3)

    ax2 = axes[1]
    ax2.set_facecolor("#f8fafc")
    ax2.plot(merged["t"], merged["diff"], color=AMBER, lw=1.5)
    x_r = np.linspace(0, len(merged)-1, 100)
    y_f = slope_model.params["const"] + slope * x_r
    ax2.plot(x_r, y_f, color=RED, lw=2, linestyle="--",
             label=f"slope={slope:.4f}, p={slope_pval:.3f}")
    ax2.axhline(0, color=GRAY, lw=1, linestyle=":")
    verdict = "✓ PASS" if slope_pval > 0.05 else "✗ WARN"
    ax2.set_title(f"Formal Slope Test — {verdict}", fontsize=11,
                  fontweight="bold",
                  color="#0f6e56" if slope_pval > 0.05 else RED)
    ax2.set_xlabel("Pre-period week")
    ax2.set_ylabel("DVM diff (T-C)")
    ax2.legend(fontsize=8)
    ax2.spines[["top","right"]].set_visible(False)
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"outputs/01_parallel_trends_{panel_name}.png", dpi=150,
                bbox_inches="tight", facecolor="#f8fafc")
    plt.close()
    print(f"✓ Saved: outputs/01_parallel_trends_{panel_name}.png")

    # ── TWFE DiD via within-transformation ───────────────────────────────────
    print("Running TWFE DiD (within-transformation)...")
    COVARS = ["is_treated_post","sst","log_chla","bathymetry",
              "season_sin","season_cos"]

    # Demean by cell (absorbs cell FE)
    dm = within_transform(df, "dvm_amplitude", COVARS, "cell_id")

    # Also demean by week (absorbs time FE)
    week_means = dm.groupby(df["week_index"]).transform("mean")
    dm2 = dm - week_means

    Y = dm2["dvm_amplitude"]
    X = sm.add_constant(dm2[COVARS])
    model   = sm.OLS(Y, X).fit(cov_type="HC3")

    did_ate  = model.params["is_treated_post"]
    did_se   = model.bse["is_treated_post"]
    did_pval = model.pvalues["is_treated_post"]
    did_ci   = model.conf_int().loc["is_treated_post"]

    print(f"TWFE DiD ATE:  {did_ate:.4f}")
    print(f"Std error:     {did_se:.4f}")
    print(f"P-value:       {did_pval:.6f}")
    print(f"95% CI:        [{did_ci[0]:.4f}, {did_ci[1]:.4f}]")

    # ── 2x2 table ─────────────────────────────────────────────────────────────
    dvm_tp = df[(df.is_treated==1)&(df.is_post_covid==0)].dvm_amplitude.mean()
    dvm_tc = df[(df.is_treated==1)&(df.is_post_covid==1)].dvm_amplitude.mean()
    dvm_cp = df[(df.is_treated==0)&(df.is_post_covid==0)].dvm_amplitude.mean()
    dvm_cc = df[(df.is_treated==0)&(df.is_post_covid==1)].dvm_amplitude.mean()
    naive  = (dvm_tc - dvm_tp) - (dvm_cc - dvm_cp)

    print(f"\n2x2 DiD:")
    print(f"Treated: {dvm_tp:.3f} → {dvm_tc:.3f} (Δ={dvm_tc-dvm_tp:+.3f})")
    print(f"Control: {dvm_cp:.3f} → {dvm_cc:.3f} (Δ={dvm_cc-dvm_cp:+.3f})")
    print(f"Naive DiD: {naive:.4f}")

    pd.DataFrame([{
        "panel": panel_name, "method": "TWFE DiD",
        "ate_estimate": did_ate, "std_error": did_se,
        "p_value": did_pval, "ci_lower": did_ci[0], "ci_upper": did_ci[1],
        "true_ate": -0.35,
        "parallel_trends_slope": slope,
        "parallel_trends_pval":  slope_pval,
    }]).to_csv(f"outputs/did_results_{panel_name}.csv", index=False)
    print(f"✓ Saved: outputs/did_results_{panel_name}.csv")
    return did_ate, did_se, did_pval

# ── Load both panels ──────────────────────────────────────────────────────────
con = duckdb.connect("data/midnight_zone.duckdb")
df_real = con.execute("SELECT * FROM mart_analysis").df()
df_sim  = con.execute("SELECT * FROM mart_analysis_sim").df()
con.close()

run_did("real", df_real, "Panel A — Real AIS Fishing Effort", "#0f6e56")
run_did("sim",  df_sim,  "Panel B — Simulated COVID Shock",   "#7c3aed")

print("\n✓ Block 1 complete. Run analysis/02_dml.py next.")