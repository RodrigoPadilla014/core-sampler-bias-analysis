# phase2_descriptive.py
# Goal: characterize each mill's readings through summary statistics,
# difference tables, normality tests, and visualizations.

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

from utils import (
    load_paired_data,
    ALL_MILLS, MILL_SANTA_ANA, MILL_LA_UNION, MILL_PANTALEON,
    QUALITY_VARS, PRIMARY_VAR,
    MILL_COLORS, FIGURES, TABLES,
)

# Output folder for this phase
FIG_DIR = FIGURES / "phase2"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# General plot style
sns.set_theme(style="whitegrid", font_scale=1.1)


# =============================================================================
# LOAD
# =============================================================================

df = load_paired_data()

print("=" * 60)
print("PHASE 2 — Descriptive Statistics")
print("=" * 60)
print(f"Paired triplets loaded: {len(df)}")


# =============================================================================
# 1. SUMMARY STATISTICS PER MILL
# =============================================================================
# For each quality variable and each mill, compute the basic descriptive stats.
# CV (coefficient of variation) = SD / mean — useful to compare variability
# across variables that are measured in different units.

print("\n--- Summary statistics per mill ---")

rows = []
for var in QUALITY_VARS:
    for mill in ALL_MILLS:
        col = f"{var}_{mill}"
        values = df[col].dropna()
        rows.append({
            "Variable" : var,
            "Mill"     : mill,
            "n"        : len(values),
            "Mean"     : round(values.mean(), 3),
            "SD"       : round(values.std(), 3),
            "CV (%)"   : round(values.std() / values.mean() * 100, 2),
            "Min"      : round(values.min(), 3),
            "Max"      : round(values.max(), 3),
        })

summary_df = pd.DataFrame(rows)
print(summary_df.to_string(index=False))

summary_path = TABLES / "phase2_summary_stats.csv"
summary_df.to_csv(summary_path, index=False, encoding="utf-8")
print(f"\nSaved: {summary_path}")


# =============================================================================
# 2. MEAN DIFFERENCE TABLE
# =============================================================================
# Shows how much Santa Ana deviates from each reference mill on average.

print("\n--- Mean differences (Santa Ana minus reference) ---")

diff_rows = []
for var in QUALITY_VARS:
    col_sa  = f"{var}_{MILL_SANTA_ANA}"
    col_lu  = f"{var}_{MILL_LA_UNION}"
    col_pan = f"{var}_{MILL_PANTALEON}"
    col_ref = f"{var}_Reference"

    diff_rows.append({
        "Variable"             : var,
        "SA minus La Union"    : round((df[col_sa] - df[col_lu]).mean(), 3),
        "SA minus Pantaleon"   : round((df[col_sa] - df[col_pan]).mean(), 3),
        "SA minus Reference"   : round((df[col_sa] - df[col_ref]).mean(), 3),
    })

diff_df = pd.DataFrame(diff_rows)
print(diff_df.to_string(index=False))

diff_path = TABLES / "phase2_mean_differences.csv"
diff_df.to_csv(diff_path, index=False, encoding="utf-8")
print(f"\nSaved: {diff_path}")


# =============================================================================
# 3. NORMALITY TEST ON THE DIFFERENCES
# =============================================================================
# We test normality on the differences (not the raw values) because that is
# what the paired tests in Phase 3 will use.
# Shapiro-Wilk is reliable for small samples (n < 50) and reasonably so
# for n = 63 as we have here.
# If p > 0.05 we cannot reject normality — paired t-test is appropriate.
# If p <= 0.05 the differences are not normal — use Wilcoxon signed-rank.

print("\n--- Shapiro-Wilk normality test on differences ---")

norm_rows = []
for var in QUALITY_VARS:
    col_diff = f"{var}_Diff"
    diffs = df[col_diff].dropna()
    stat, p = stats.shapiro(diffs)
    normal = "Yes" if p > 0.05 else "No"
    norm_rows.append({
        "Variable"       : var,
        "W statistic"    : round(stat, 4),
        "p-value"        : round(p, 4),
        "Normal (p>0.05)": normal,
    })

norm_df = pd.DataFrame(norm_rows)
print(norm_df.to_string(index=False))

norm_path = TABLES / "phase2_normality_tests.csv"
norm_df.to_csv(norm_path, index=False, encoding="utf-8")
print(f"\nSaved: {norm_path}")


# =============================================================================
# 4. BOX PLOTS + STRIP PLOTS PER VARIABLE
# =============================================================================
# One figure per quality variable, showing the distribution of each mill
# side by side. Individual data points are overlaid so nothing is hidden
# behind the box summary.

print("\n--- Generating box plots ---")

for var in QUALITY_VARS:
    # Gather data in long format for seaborn
    plot_data = pd.DataFrame({
        mill: df[f"{var}_{mill}"] for mill in ALL_MILLS
    }).melt(var_name="Mill", value_name=var)

    fig, ax = plt.subplots(figsize=(8, 5))

    # Box plot
    sns.boxplot(
        data=plot_data, x="Mill", y=var,
        hue="Mill", palette=MILL_COLORS, legend=False,
        width=0.5, linewidth=1.2,
        flierprops=dict(marker="o", markersize=4, alpha=0.5),
        ax=ax,
    )

    # Individual points overlaid
    sns.stripplot(
        data=plot_data, x="Mill", y=var,
        hue="Mill", palette=MILL_COLORS, legend=False,
        size=4, alpha=0.4, jitter=True,
        ax=ax,
    )

    ax.set_title(f"{var} by Mill", fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel(var)

    # Add mean as a horizontal dashed line per mill
    means = plot_data.groupby("Mill")[var].mean()
    for i, mill in enumerate(ALL_MILLS):
        ax.hlines(
            means[mill], i - 0.3, i + 0.3,
            colors="black", linewidths=1.5, linestyles="--", label="Mean" if i == 0 else ""
        )

    ax.legend(["Mean"], loc="upper right", fontsize=9)
    plt.tight_layout()

    # Save — clean filename
    fname = var.replace("%", "pct").replace("/", "_").replace(" ", "_").replace("(", "").replace(")", "")
    fpath = FIG_DIR / f"boxplot_{fname}.png"
    plt.savefig(fpath, dpi=150)
    plt.close()
    print(f"  Saved: {fpath.name}")


# =============================================================================
# 5. HISTOGRAM OF DIFFERENCES FOR EACH VARIABLE
# =============================================================================
# Visualizes how the Santa Ana minus Reference differences are distributed.
# A bell-shaped histogram supports the normality assumption.
# A skewed or bimodal histogram would raise concerns.

print("\n--- Generating histograms of differences ---")

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()

for i, var in enumerate(QUALITY_VARS):
    col_diff = f"{var}_Diff"
    diffs = df[col_diff].dropna()
    ax = axes[i]

    ax.hist(diffs, bins=12, color="#e74c3c", edgecolor="white", alpha=0.8)

    # Mark the mean difference
    ax.axvline(diffs.mean(), color="black", linestyle="--", linewidth=1.5, label=f"Mean = {diffs.mean():.2f}")
    ax.axvline(0, color="gray", linestyle=":", linewidth=1, label="Zero (no bias)")

    ax.set_title(var, fontweight="bold", fontsize=10)
    ax.set_xlabel("Santa Ana − Reference")
    ax.set_ylabel("Count")
    ax.legend(fontsize=8)

plt.suptitle("Distribution of Differences (Santa Ana − Reference)", fontweight="bold", fontsize=13)
plt.tight_layout()

fpath = FIG_DIR / "histograms_differences.png"
plt.savefig(fpath, dpi=150)
plt.close()
print(f"  Saved: {fpath.name}")

print("\nPhase 2 complete.")
