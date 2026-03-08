# phase6_temporal.py
# Goal: analyze how the bias between Santa Ana and the reference mills
# behaves across the 15 weeks of the zafra. Establishes whether the
# horizontal sampler bias is stable over time or drifts seasonally.

import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

from utils import (
    load_paired_data,
    MILL_SANTA_ANA, MILL_LA_UNION, MILL_PANTALEON,
    ALL_MILLS, QUALITY_VARS, PRIMARY_VAR,
    MILL_COLORS, FIGURES, TABLES,
)

FIG_DIR = FIGURES / "phase6"
FIG_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", font_scale=1.1)


# =============================================================================
# LOAD
# =============================================================================

df = load_paired_data()

print("=" * 60)
print("PHASE 6 — Temporal & Stability Analysis")
print("=" * 60)
print(f"Paired triplets loaded: {len(df)}")


# =============================================================================
# 1. WEEKLY BIAS SUMMARY
# =============================================================================
# For each week, compute the mean bias (Santa Ana minus Reference)
# and its standard deviation across the samples in that week.

print("\n--- Weekly bias summary ---")

weekly_rows = []
for week, group in df.groupby("Semana"):
    for var in QUALITY_VARS:
        col_diff = f"{var}_Diff"
        diffs = group[col_diff].dropna()
        weekly_rows.append({
            "Semana"    : week,
            "Variable"  : var,
            "n"         : len(diffs),
            "Mean diff" : round(diffs.mean(), 4),
            "SD diff"   : round(diffs.std(), 4),
        })

weekly_df = pd.DataFrame(weekly_rows)

primary_weekly = weekly_df[weekly_df["Variable"] == PRIMARY_VAR]
print(f"\nWeekly bias for {PRIMARY_VAR}:")
print(primary_weekly.to_string(index=False))

weekly_path = TABLES / "phase6_weekly_bias.csv"
weekly_df.to_csv(weekly_path, index=False, encoding="utf-8")
print(f"\nSaved: {weekly_path}")


# =============================================================================
# 2. IS THE BIAS STABLE ACROSS WEEKS?
# =============================================================================
# One-way ANOVA on the differences grouped by week.
# If significant, the bias is not constant — it changes over the zafra.
# Then Pearson correlation between week number and bias to detect linear drift.

print("\n--- Stability test: does bias change across weeks? ---")

stability_results = []

for var in QUALITY_VARS:
    col_diff = f"{var}_Diff"

    # Group differences by week
    groups = [group[col_diff].dropna().values
              for _, group in df.groupby("Semana")]

    # One-way ANOVA across weeks
    f_stat, p_val = stats.f_oneway(*groups)

    # Pearson correlation: week number vs mean weekly bias
    week_means = df.groupby("Semana")[col_diff].mean()
    r_corr, p_corr = stats.pearsonr(week_means.index, week_means.values)

    stability_results.append({
        "Variable"          : var,
        "ANOVA F"           : round(f_stat, 4),
        "ANOVA p-value"     : round(p_val, 4),
        "Bias drifts (ANOVA)": "Yes" if p_val < 0.05 else "No",
        "Pearson r (week)"  : round(r_corr, 4),
        "Pearson p (week)"  : round(p_corr, 4),
        "Linear drift"      : "Yes" if p_corr < 0.05 else "No",
    })

stability_df = pd.DataFrame(stability_results)
print(stability_df.to_string(index=False))

stability_path = TABLES / "phase6_stability.csv"
stability_df.to_csv(stability_path, index=False, encoding="utf-8")
print(f"\nSaved: {stability_path}")


# =============================================================================
# 3. VISUALIZATIONS
# =============================================================================

# --- 3a. Weekly bias plot for primary variable ---
# Each dot is the mean bias for that week. Error bars show SD across samples.
# The red dashed line is the overall mean bias across the full zafra.
# A flat scatter around the dashed line = stable bias.
# A trend upward or downward = drifting bias.

print("\n--- Generating weekly bias plot ---")

primary_weekly = weekly_df[weekly_df["Variable"] == PRIMARY_VAR].copy()
overall_bias   = df[f"{PRIMARY_VAR}_Diff"].mean()

fig, ax = plt.subplots(figsize=(10, 5))

ax.errorbar(
    primary_weekly["Semana"],
    primary_weekly["Mean diff"],
    yerr=primary_weekly["SD diff"],
    fmt="o-", color="#e74c3c", linewidth=1.8,
    markersize=7, capsize=4,
    label="Weekly mean bias ± SD"
)

# Overall mean bias line
ax.axhline(overall_bias, color="black", linestyle="--", linewidth=1.5,
           label=f"Overall mean bias = {overall_bias:.2f} kg/t")

# Zero line
ax.axhline(0, color="gray", linestyle=":", linewidth=1, label="Zero (no bias)")

ax.set_xlabel("Week (Semana)")
ax.set_ylabel(f"Mean diff — {PRIMARY_VAR} (Santa Ana − Reference)")
ax.set_title(f"Weekly Bias Stability — {PRIMARY_VAR}", fontweight="bold")
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.legend(fontsize=9)

plt.tight_layout()
fpath = FIG_DIR / "weekly_bias.png"
plt.savefig(fpath, dpi=150)
plt.close()
print(f"  Saved: {fpath.name}")


# --- 3b. Weekly trend lines per mill for primary variable ---
# Shows the seasonal cane quality curve for each mill.
# All three should follow the same seasonal pattern if sampling is consistent.
# Divergence at specific weeks indicates where the horizontal sampler fails most.

print("\n--- Generating weekly trend lines per mill ---")

# Compute weekly mean per mill
weekly_mill_rows = []
for week, group in df.groupby("Semana"):
    for mill in ALL_MILLS:
        col = f"{PRIMARY_VAR}_{mill}"
        weekly_mill_rows.append({
            "Semana": week,
            "Mill"  : mill,
            "Mean"  : group[col].mean(),
        })

weekly_mill_df = pd.DataFrame(weekly_mill_rows)

fig, ax = plt.subplots(figsize=(11, 5))

for mill in ALL_MILLS:
    mill_data = weekly_mill_df[weekly_mill_df["Mill"] == mill]
    ax.plot(
        mill_data["Semana"], mill_data["Mean"],
        marker="o", color=MILL_COLORS[mill],
        linewidth=2, markersize=6, label=mill
    )

ax.set_xlabel("Week (Semana)")
ax.set_ylabel(PRIMARY_VAR)
ax.set_title(f"Weekly {PRIMARY_VAR} by Mill — Seasonal Trend", fontweight="bold")
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.legend()

plt.tight_layout()
fpath = FIG_DIR / "weekly_trend_by_mill.png"
plt.savefig(fpath, dpi=150)
plt.close()
print(f"  Saved: {fpath.name}")


# --- 3c. Bias heatmap across all variables and weeks ---
# Rows = variables, columns = weeks, color = mean bias that week.
# Gives a single visual overview of where and when bias is largest.

print("\n--- Generating bias heatmap ---")

heatmap_data = weekly_df.pivot(index="Variable", columns="Semana", values="Mean diff")

fig, ax = plt.subplots(figsize=(13, 5))

sns.heatmap(
    heatmap_data,
    annot=True, fmt=".1f",
    cmap="RdBu_r", center=0,
    linewidths=0.5, linecolor="white",
    ax=ax
)

ax.set_title("Weekly Mean Bias (Santa Ana − Reference) by Variable", fontweight="bold")
ax.set_xlabel("Week (Semana)")
ax.set_ylabel("")

plt.tight_layout()
fpath = FIG_DIR / "bias_heatmap.png"
plt.savefig(fpath, dpi=150)
plt.close()
print(f"  Saved: {fpath.name}")

print("\nPhase 6 complete.")
