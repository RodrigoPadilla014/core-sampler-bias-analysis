# phase3_paired_tests.py
# Goal: formally test whether the differences found in Phase 2 are
# statistically significant, quantify the effect size, and report
# confidence intervals on the mean difference.

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns

from utils import (
    load_paired_data,
    MILL_SANTA_ANA, MILL_LA_UNION, MILL_PANTALEON,
    QUALITY_VARS, PRIMARY_VAR,
    MILL_COLORS, FIGURES, TABLES,
)

FIG_DIR = FIGURES / "phase3"
FIG_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", font_scale=1.1)


# =============================================================================
# LOAD
# =============================================================================

df = load_paired_data()

print("=" * 60)
print("PHASE 3 — Paired Statistical Tests")
print("=" * 60)
print(f"Paired triplets loaded: {len(df)}")


# =============================================================================
# 1. PAIRED T-TEST + EFFECT SIZE + CONFIDENCE INTERVALS
# =============================================================================
# For each quality variable we run three comparisons:
#   - Santa Ana vs La Union
#   - Santa Ana vs Pantaleon
#   - Santa Ana vs Reference average (mean of the two oblique mills)
#
# Cohen's d for paired data = mean(diff) / SD(diff)
# Interpretation: small = 0.2, medium = 0.5, large = 0.8
#
# The 95% CI on the mean difference is:
#   mean(diff) ± t_critical * SE
# where SE = SD(diff) / sqrt(n)

def paired_test(a, b, label_a, label_b, var):
    """
    Runs a paired t-test between series a and b.
    Returns a dict with all relevant statistics.
    """
    diff   = a - b
    n      = len(diff)
    mean_d = diff.mean()
    sd_d   = diff.std(ddof=1)
    se     = sd_d / np.sqrt(n)

    # Paired t-test
    t_stat, p_val = stats.ttest_rel(a, b)

    # Cohen's d for paired samples
    cohens_d = mean_d / sd_d

    # 95% confidence interval
    t_crit = stats.t.ppf(0.975, df=n - 1)
    ci_low = mean_d - t_crit * se
    ci_high = mean_d + t_crit * se

    return {
        "Variable"    : var,
        "Comparison"  : f"{label_a} vs {label_b}",
        "n"           : n,
        "Mean diff"   : round(mean_d, 4),
        "SD diff"     : round(sd_d, 4),
        "t statistic" : round(t_stat, 4),
        "p-value"     : round(p_val, 4),
        "Significant" : "Yes" if p_val < 0.05 else "No",
        "Cohen's d"   : round(cohens_d, 4),
        "CI low (95%)": round(ci_low, 4),
        "CI high (95%)": round(ci_high, 4),
    }


print("\n--- Running paired t-tests ---")

results = []
for var in QUALITY_VARS:
    sa  = df[f"{var}_{MILL_SANTA_ANA}"]
    lu  = df[f"{var}_{MILL_LA_UNION}"]
    pan = df[f"{var}_{MILL_PANTALEON}"]
    ref = df[f"{var}_Reference"]

    results.append(paired_test(sa, lu,  "Santa Ana", "La Union",   var))
    results.append(paired_test(sa, pan, "Santa Ana", "Pantaleon",  var))
    results.append(paired_test(sa, ref, "Santa Ana", "Reference",  var))

results_df = pd.DataFrame(results)

# Print a focused view on the primary variable first
primary = results_df[results_df["Variable"] == PRIMARY_VAR]
print(f"\nResults for primary variable: {PRIMARY_VAR}")
print(primary.to_string(index=False))

print(f"\nFull results (all variables):")
print(results_df.to_string(index=False))

results_path = TABLES / "phase3_paired_tests.csv"
results_df.to_csv(results_path, index=False, encoding="utf-8")
print(f"\nSaved: {results_path}")


# =============================================================================
# 2. FOREST PLOT
# =============================================================================
# A forest plot shows the mean difference and its 95% CI for each variable
# and comparison in a single, easy-to-read figure.
# The vertical line at zero represents "no bias" — if the CI crosses zero,
# the difference is not statistically significant.

print("\n--- Generating forest plot ---")

# We focus on the Santa Ana vs Reference comparison (the main finding)
forest_data = results_df[results_df["Comparison"] == "Santa Ana vs Reference"].copy()
forest_data = forest_data.sort_values("Mean diff", ascending=True).reset_index(drop=True)

fig, ax = plt.subplots(figsize=(9, 5))

colors = ["#e74c3c" if row["Significant"] == "Yes" else "#95a5a6"
          for _, row in forest_data.iterrows()]

for i, row in forest_data.iterrows():
    # Confidence interval line
    ax.plot(
        [row["CI low (95%)"], row["CI high (95%)"]],
        [i, i],
        color=colors[i], linewidth=2, solid_capstyle="round"
    )
    # Mean difference dot
    ax.scatter(row["Mean diff"], i, color=colors[i], s=80, zorder=5)

# Zero line — represents no bias
ax.axvline(0, color="black", linestyle="--", linewidth=1, label="No bias (diff = 0)")

# Labels
ax.set_yticks(range(len(forest_data)))
ax.set_yticklabels(forest_data["Variable"])
ax.set_xlabel("Mean difference (Santa Ana − Reference) with 95% CI")
ax.set_title("Forest Plot — Santa Ana vs Reference Mills", fontweight="bold")

# Legend
sig_patch   = mlines.Line2D([], [], color="#e74c3c", marker="o", markersize=7, label="Significant (p < 0.05)")
insig_patch = mlines.Line2D([], [], color="#95a5a6", marker="o", markersize=7, label="Not significant")
ax.legend(handles=[sig_patch, insig_patch, mlines.Line2D([], [], color="black", linestyle="--", label="No bias")], fontsize=9)

plt.tight_layout()
fpath = FIG_DIR / "forest_plot.png"
plt.savefig(fpath, dpi=150)
plt.close()
print(f"  Saved: {fpath.name}")


# =============================================================================
# 3. PAIRED DOT PLOT FOR PRIMARY VARIABLE
# =============================================================================
# Each dot represents one cane load. Lines connect the Santa Ana reading
# to the Reference reading for the same load.
# This makes the systematic upward bias of Santa Ana visually obvious.

print("\n--- Generating paired dot plot ---")

sa_vals  = df[f"{PRIMARY_VAR}_{MILL_SANTA_ANA}"]
ref_vals = df[f"{PRIMARY_VAR}_Reference"]

fig, ax = plt.subplots(figsize=(7, 6))

# Draw a line per triplet connecting Santa Ana to Reference
for sa, ref in zip(sa_vals, ref_vals):
    color = "#e74c3c" if sa > ref else "#3498db"
    ax.plot([0, 1], [sa, ref], color=color, alpha=0.3, linewidth=1)

# Plot the actual points
ax.scatter([0] * len(sa_vals),  sa_vals,  color="#e74c3c", s=40, zorder=5, label="Santa Ana (horizontal)")
ax.scatter([1] * len(ref_vals), ref_vals, color="#2ecc71", s=40, zorder=5, label="Reference (oblique)")

# Mark the means
ax.scatter([0], [sa_vals.mean()],  color="#c0392b", s=150, zorder=6, marker="D")
ax.scatter([1], [ref_vals.mean()], color="#27ae60", s=150, zorder=6, marker="D")

ax.set_xticks([0, 1])
ax.set_xticklabels(["Santa Ana\n(horizontal)", "Reference\n(oblique)"])
ax.set_ylabel(PRIMARY_VAR)
ax.set_title(f"Paired Dot Plot — {PRIMARY_VAR}\n(lines connect the same cane load)", fontweight="bold")
ax.legend(fontsize=9)

# Annotate mean difference
mean_diff = sa_vals.mean() - ref_vals.mean()
ax.annotate(
    f"Mean diff = +{mean_diff:.2f} kg/t",
    xy=(0.5, max(sa_vals.max(), ref_vals.max()) * 0.98),
    ha="center", fontsize=10, color="#e74c3c", fontweight="bold"
)

plt.tight_layout()
fpath = FIG_DIR / "paired_dot_plot.png"
plt.savefig(fpath, dpi=150)
plt.close()
print(f"  Saved: {fpath.name}")

print("\nPhase 3 complete.")
