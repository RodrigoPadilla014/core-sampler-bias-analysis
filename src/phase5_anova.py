# phase5_anova.py
# Goal: formally test the mill effect using a blocked ANOVA (RCBD),
# decompose the variance into mill, block, and residual components,
# run Tukey HSD post-hoc comparisons, and test whether the bias
# differs between cut types (Mecanizado vs Manual).

import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from utils import (
    load_paired_data,
    MILL_SANTA_ANA, MILL_LA_UNION, MILL_PANTALEON,
    ALL_MILLS, QUALITY_VARS, PRIMARY_VAR,
    MILL_COLORS, FIGURES, TABLES,
)

FIG_DIR = FIGURES / "phase5"
FIG_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", font_scale=1.1)


# =============================================================================
# LOAD AND RESHAPE
# =============================================================================
# The paired dataset is in wide format (one row per cane load, columns per mill).
# For ANOVA we need long format: one row per observation, with columns for
# mill, block (cane load), and the measurement value.

df = load_paired_data()

print("=" * 60)
print("PHASE 5 — Blocked ANOVA")
print("=" * 60)
print(f"Paired triplets loaded: {len(df)}")

# Create a unique block identifier per cane load
df["Block"] = df["Semana"].astype(str) + "_" + df["# Muestra"].astype(str)

# Reshape to long format
long_rows = []
for _, row in df.iterrows():
    for mill in ALL_MILLS:
        for var in QUALITY_VARS:
            long_rows.append({
                "Block"  : row["Block"],
                "Semana" : row["Semana"],
                "Muestra": row["# Muestra"],
                "Corte"  : row["Corte"],
                "Mill"   : mill,
                "Variable": var,
                "Value"  : row[f"{var}_{mill}"],
            })

long_df = pd.DataFrame(long_rows)
print(f"Long format shape: {long_df.shape}")


# =============================================================================
# 1. BLOCKED ANOVA PER VARIABLE
# =============================================================================
# Model: Value ~ C(Mill) + C(Block)
# C() tells statsmodels to treat these as categorical variables.
# The block term absorbs cane load variation, leaving the mill effect clean.
#
# We report the ANOVA table showing:
#   - Mill effect: is there a significant difference between mills?
#   - Block effect: how much variation is explained by cane load?
#   - Residual: unexplained noise

print("\n--- Blocked ANOVA results ---")

anova_results = []

for var in QUALITY_VARS:
    data = long_df[long_df["Variable"] == var].copy()

    # Fit the blocked ANOVA model
    model  = ols("Value ~ C(Mill) + C(Block)", data=data).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)

    mill_f   = round(anova_table.loc["C(Mill)",  "F"], 4)
    mill_p   = round(anova_table.loc["C(Mill)",  "PR(>F)"], 4)
    block_f  = round(anova_table.loc["C(Block)", "F"], 4)
    block_p  = round(anova_table.loc["C(Block)", "PR(>F)"], 4)

    # Eta squared for mill — proportion of total variance explained by mill
    ss_mill  = anova_table.loc["C(Mill)",  "sum_sq"]
    ss_total = anova_table["sum_sq"].sum()
    eta_sq   = round(ss_mill / ss_total, 4)

    anova_results.append({
        "Variable"       : var,
        "Mill F"         : mill_f,
        "Mill p-value"   : mill_p,
        "Mill significant": "Yes" if mill_p < 0.05 else "No",
        "Eta squared"    : eta_sq,
        "Block F"        : block_f,
        "Block p-value"  : block_p,
    })

    print(f"\n{var}:")
    print(f"  Mill effect  — F={mill_f}, p={mill_p}, eta²={eta_sq}")
    print(f"  Block effect — F={block_f}, p={block_p}")

anova_df = pd.DataFrame(anova_results)
anova_path = TABLES / "phase5_anova.csv"
anova_df.to_csv(anova_path, index=False, encoding="utf-8")
print(f"\nSaved: {anova_path}")


# =============================================================================
# 2. TUKEY HSD POST-HOC
# =============================================================================
# The ANOVA tells us IF there is a mill difference.
# Tukey HSD tells us WHICH specific mill pairs are different.
# It controls the family-wise error rate — adjusting for the fact that
# we are making multiple comparisons simultaneously.

print("\n--- Tukey HSD post-hoc comparisons ---")

tukey_results = []

for var in QUALITY_VARS:
    data = long_df[long_df["Variable"] == var].copy()

    tukey = pairwise_tukeyhsd(
        endog=data["Value"],
        groups=data["Mill"],
        alpha=0.05,
    )

    for row in tukey.summary().data[1:]:   # skip header row
        tukey_results.append({
            "Variable"   : var,
            "Group 1"    : row[0],
            "Group 2"    : row[1],
            "Mean diff"  : round(row[2], 4),
            "Lower CI"   : round(row[4], 4),
            "Upper CI"   : round(row[5], 4),
            "Significant": "Yes" if row[6] else "No",
        })

tukey_df = pd.DataFrame(tukey_results)

# Show only primary variable
primary_tukey = tukey_df[tukey_df["Variable"] == PRIMARY_VAR]
print(f"\nTukey HSD for {PRIMARY_VAR}:")
print(primary_tukey.to_string(index=False))

tukey_path = TABLES / "phase5_tukey.csv"
tukey_df.to_csv(tukey_path, index=False, encoding="utf-8")
print(f"\nSaved: {tukey_path}")


# =============================================================================
# 3. CUT TYPE INTERACTION (Mill x Corte)
# =============================================================================
# Does the bias between Santa Ana and the reference mills differ between
# Mecanizado and Manual harvesting?
# We test this by looking at the Diff column grouped by cut type.

print("\n--- Cut type interaction ---")

cut_results = []

for var in QUALITY_VARS:
    col_diff = f"{var}_Diff"
    for corte, group in df.groupby("Corte"):
        diffs = group[col_diff].dropna()
        t_stat, p_val = stats.ttest_1samp(diffs, 0)
        cut_results.append({
            "Variable"   : var,
            "Corte"      : corte,
            "n"          : len(diffs),
            "Mean diff"  : round(diffs.mean(), 4),
            "SD diff"    : round(diffs.std(), 4),
            "t statistic": round(t_stat, 4),
            "p-value"    : round(p_val, 4),
            "Significant": "Yes" if p_val < 0.05 else "No",
        })

cut_df = pd.DataFrame(cut_results)

primary_cut = cut_df[cut_df["Variable"] == PRIMARY_VAR]
print(f"\nCut type breakdown for {PRIMARY_VAR}:")
print(primary_cut.to_string(index=False))

cut_path = TABLES / "phase5_cut_type.csv"
cut_df.to_csv(cut_path, index=False, encoding="utf-8")
print(f"\nSaved: {cut_path}")


# =============================================================================
# 3b. PROPER INTERACTION TEST — IS THE BIAS DIFFERENT BETWEEN CUT TYPES?
# =============================================================================
# The one-sample t-tests above told us whether bias exists within each cut type.
# This test directly asks: is the bias significantly different BETWEEN cut types?
#
# We compare the two independent groups of differences:
#   - Mecanizado differences (43 values)
#   - Manual differences (20 values)
#
# First check normality of each group, then choose the appropriate test:
#   - Both normal → two-sample t-test (Welch, unequal variances assumed)
#   - Either non-normal → Mann-Whitney U test

print("\n--- Interaction test: is bias different between cut types? ---")

interaction_results = []

for var in QUALITY_VARS:
    col_diff = f"{var}_Diff"

    mecanizado = df[df["Corte"] == "Mecanizado"][col_diff].dropna()
    manual     = df[df["Corte"] == "Manual"][col_diff].dropna()

    # Normality check on each group
    _, p_mec = stats.shapiro(mecanizado)
    _, p_man = stats.shapiro(manual)
    both_normal = (p_mec > 0.05) and (p_man > 0.05)

    if both_normal:
        # Welch two-sample t-test (does not assume equal variances)
        t_stat, p_val = stats.ttest_ind(mecanizado, manual, equal_var=False)
        test_used = "Welch t-test"
    else:
        # Mann-Whitney U — non-parametric alternative
        t_stat, p_val = stats.mannwhitneyu(mecanizado, manual, alternative="two-sided")
        test_used = "Mann-Whitney U"

    interaction_results.append({
        "Variable"          : var,
        "Mean diff Mec"     : round(mecanizado.mean(), 4),
        "Mean diff Manual"  : round(manual.mean(), 4),
        "Difference of bias": round(mecanizado.mean() - manual.mean(), 4),
        "Test used"         : test_used,
        "Statistic"         : round(t_stat, 4),
        "p-value"           : round(p_val, 4),
        "Bias differs between cut types": "Yes" if p_val < 0.05 else "No",
    })

interaction_df = pd.DataFrame(interaction_results)

print(f"\nDoes the bias differ between Mecanizado and Manual?")
print(interaction_df[["Variable", "Mean diff Mec", "Mean diff Manual",
                       "Difference of bias", "Test used", "p-value",
                       "Bias differs between cut types"]].to_string(index=False))

interaction_path = TABLES / "phase5_cut_interaction.csv"
interaction_df.to_csv(interaction_path, index=False, encoding="utf-8")
print(f"\nSaved: {interaction_path}")


# =============================================================================
# 4. VISUALIZATIONS
# =============================================================================

# --- 4a. Means comparison plot (primary variable) ---
# Shows the mean Pol%caña per mill with 95% CI error bars.
# Makes the mill differences visually clear.

print("\n--- Generating means comparison plot ---")

data_primary = long_df[long_df["Variable"] == PRIMARY_VAR]

fig, ax = plt.subplots(figsize=(7, 5))

means = data_primary.groupby("Mill")["Value"].mean()
sems  = data_primary.groupby("Mill")["Value"].sem()

for i, mill in enumerate(ALL_MILLS):
    ax.bar(
        i, means[mill],
        color=MILL_COLORS[mill], alpha=0.8, width=0.5,
        yerr=1.96 * sems[mill],
        capsize=5, error_kw=dict(linewidth=1.5)
    )
    ax.text(i, means[mill] + 1.96 * sems[mill] + 1,
            f"{means[mill]:.1f}", ha="center", fontsize=10, fontweight="bold")

ax.set_xticks(range(len(ALL_MILLS)))
ax.set_xticklabels(ALL_MILLS)
ax.set_ylabel(PRIMARY_VAR)
ax.set_title(f"Mean {PRIMARY_VAR} by Mill\n(error bars = 95% CI)", fontweight="bold")
ax.set_ylim(bottom=120)

plt.tight_layout()
fpath = FIG_DIR / "means_comparison.png"
plt.savefig(fpath, dpi=150)
plt.close()
print(f"  Saved: {fpath.name}")


# --- 4b. Cut type interaction plot ---
# Shows the mean bias (Santa Ana minus Reference) for Mecanizado vs Manual.
# If the bars are very different, the sampler behaves differently by cut type.

print("\n--- Generating cut type interaction plot ---")

primary_cut_plot = cut_df[cut_df["Variable"] == PRIMARY_VAR].copy()

fig, ax = plt.subplots(figsize=(6, 5))

colors = ["#e74c3c" if row["Significant"] == "Yes" else "#95a5a6"
          for _, row in primary_cut_plot.iterrows()]

bars = ax.bar(
    primary_cut_plot["Corte"],
    primary_cut_plot["Mean diff"],
    color=colors, alpha=0.85, width=0.4,
)

# Error bars using SD
ax.errorbar(
    primary_cut_plot["Corte"],
    primary_cut_plot["Mean diff"],
    yerr=primary_cut_plot["SD diff"],
    fmt="none", color="black", capsize=5, linewidth=1.5
)

ax.axhline(0, color="gray", linestyle="--", linewidth=1)
ax.set_ylabel(f"Mean diff — {PRIMARY_VAR} (Santa Ana − Reference)")
ax.set_title(f"Bias by Cut Type — {PRIMARY_VAR}", fontweight="bold")

for bar, (_, row) in zip(bars, primary_cut_plot.iterrows()):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.5,
        f"n={row['n']}\np={row['p-value']}",
        ha="center", fontsize=9
    )

plt.tight_layout()
fpath = FIG_DIR / "cut_type_interaction.png"
plt.savefig(fpath, dpi=150)
plt.close()
print(f"  Saved: {fpath.name}")


# --- 4c. Variance decomposition bar chart ---
# Shows what proportion of total Pol%caña variance is explained by
# mill, block, and residual — for each variable.

print("\n--- Generating variance decomposition chart ---")

var_decomp = []
for var in QUALITY_VARS:
    data = long_df[long_df["Variable"] == var].copy()
    model = ols("Value ~ C(Mill) + C(Block)", data=data).fit()
    table = sm.stats.anova_lm(model, typ=2)
    ss_total = table["sum_sq"].sum()
    var_decomp.append({
        "Variable": var,
        "Mill"    : table.loc["C(Mill)",  "sum_sq"] / ss_total * 100,
        "Block"   : table.loc["C(Block)", "sum_sq"] / ss_total * 100,
        "Residual": table.loc["Residual", "sum_sq"] / ss_total * 100,
    })

decomp_df = pd.DataFrame(var_decomp)

fig, ax = plt.subplots(figsize=(10, 5))

x = range(len(QUALITY_VARS))
ax.bar(x, decomp_df["Mill"],    label="Mill",     color="#e74c3c", alpha=0.85)
ax.bar(x, decomp_df["Block"],   label="Block",    color="#3498db", alpha=0.85, bottom=decomp_df["Mill"])
ax.bar(x, decomp_df["Residual"],label="Residual", color="#95a5a6", alpha=0.85, bottom=decomp_df["Mill"] + decomp_df["Block"])

ax.set_xticks(x)
ax.set_xticklabels(QUALITY_VARS, rotation=15, ha="right")
ax.set_ylabel("% of total variance")
ax.set_title("Variance Decomposition by Source", fontweight="bold")
ax.legend()

plt.tight_layout()
fpath = FIG_DIR / "variance_decomposition.png"
plt.savefig(fpath, dpi=150)
plt.close()
print(f"  Saved: {fpath.name}")

print("\nPhase 5 complete.")
