# phase4_method_comparison.py
# Goal: characterize HOW the horizontal sampler bias behaves across the full
# range of cane quality using Bland-Altman plots, Passing-Bablok regression,
# and the Concordance Correlation Coefficient (CCC).

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from utils import (
    load_paired_data,
    MILL_SANTA_ANA, MILL_LA_UNION, MILL_PANTALEON,
    QUALITY_VARS, PRIMARY_VAR,
    MILL_COLORS, FIGURES, TABLES,
)

FIG_DIR = FIGURES / "phase4"
FIG_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", font_scale=1.1)


# =============================================================================
# LOAD
# =============================================================================

df = load_paired_data()

print("=" * 60)
print("PHASE 4 — Method Comparison")
print("=" * 60)
print(f"Paired triplets loaded: {len(df)}")


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def bland_altman_stats(a, b):
    """
    Computes Bland-Altman statistics between method a (Santa Ana)
    and method b (reference).
    Returns mean bias and limits of agreement.
    """
    diff  = a - b
    mean  = (a + b) / 2
    bias  = diff.mean()
    sd    = diff.std(ddof=1)
    loa_upper = bias + 1.96 * sd
    loa_lower = bias - 1.96 * sd
    return mean, diff, bias, sd, loa_upper, loa_lower


def passing_bablok(x, y):
    """
    Passing-Bablok regression — fits a line between two methods
    without assuming either is error-free.
    Returns slope and intercept with 95% confidence intervals.

    The method ranks all pairwise slopes (y_j - y_i) / (x_j - x_i)
    and takes the median as the slope estimate.
    """
    n = len(x)
    slopes = []

    for i in range(n):
        for j in range(i + 1, n):
            dx = x[j] - x[i]
            dy = y[j] - y[i]
            if dx != 0:
                slopes.append(dy / dx)

    slopes = np.array(sorted(slopes))

    # Remove slopes of exactly -1 (they indicate a specific degenerate case)
    slopes = slopes[slopes != -1]

    # Median slope
    slope = np.median(slopes)

    # Intercept
    intercept = np.median(y) - slope * np.median(x)

    # 95% CI using the ranked slopes
    k = len(slopes)
    z = 1.96
    ci_half = z * np.sqrt(n * (n - 1) / 2 * (2 * n + 5) / 18)
    m1 = int(np.floor((k - ci_half) / 2))
    m2 = int(np.ceil((k + ci_half) / 2)) + 1

    m1 = max(0, m1)
    m2 = min(k - 1, m2)

    slope_ci_low  = slopes[m1]
    slope_ci_high = slopes[m2]

    intercept_ci_low  = np.median(y) - slope_ci_high * np.median(x)
    intercept_ci_high = np.median(y) - slope_ci_low  * np.median(x)

    return {
        "slope"            : round(slope, 4),
        "slope_ci_low"     : round(slope_ci_low, 4),
        "slope_ci_high"    : round(slope_ci_high, 4),
        "intercept"        : round(intercept, 4),
        "intercept_ci_low" : round(intercept_ci_low, 4),
        "intercept_ci_high": round(intercept_ci_high, 4),
    }


def concordance_correlation(a, b):
    """
    Concordance Correlation Coefficient (CCC).
    Combines Pearson's r (precision) with how close the line is to
    the perfect 45-degree agreement line (accuracy).
    Range: -1 to 1. Values close to 1 = good agreement.
    """
    mean_a = a.mean()
    mean_b = b.mean()
    var_a  = a.var(ddof=1)
    var_b  = b.var(ddof=1)
    cov    = np.cov(a, b, ddof=1)[0, 1]

    ccc = (2 * cov) / (var_a + var_b + (mean_a - mean_b) ** 2)

    # Pearson r for comparison
    r, _ = stats.pearsonr(a, b)

    return round(ccc, 4), round(r, 4)


# =============================================================================
# 1. BLAND-ALTMAN PLOTS
# =============================================================================
# One plot per quality variable, comparing Santa Ana vs Reference average.
# We look for:
#   - Random scatter around the bias line → constant bias (good pattern)
#   - Upward or downward trend → proportional bias (problematic)

print("\n--- Generating Bland-Altman plots ---")

ba_results = []

for var in QUALITY_VARS:
    sa  = df[f"{var}_{MILL_SANTA_ANA}"]
    ref = df[f"{var}_Reference"]

    mean_val, diff, bias, sd, loa_upper, loa_lower = bland_altman_stats(sa, ref)

    # Test for proportional bias: correlate the difference with the mean
    r_prop, p_prop = stats.pearsonr(mean_val, diff)

    ba_results.append({
        "Variable"         : var,
        "Mean bias"        : round(bias, 4),
        "SD diff"          : round(sd, 4),
        "LoA upper (+1.96 SD)": round(loa_upper, 4),
        "LoA lower (-1.96 SD)": round(loa_lower, 4),
        "Proportional bias r" : round(r_prop, 4),
        "Proportional bias p" : round(p_prop, 4),
        "Proportional bias"   : "Yes" if p_prop < 0.05 else "No",
    })

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.scatter(mean_val, diff, color="#e74c3c", alpha=0.6, s=50, edgecolors="white", linewidths=0.5)

    # Bias and limits of agreement
    ax.axhline(bias,      color="#e74c3c", linestyle="-",  linewidth=1.8, label=f"Mean bias = {bias:.2f}")
    ax.axhline(loa_upper, color="#3498db", linestyle="--", linewidth=1.4, label=f"+1.96 SD = {loa_upper:.2f}")
    ax.axhline(loa_lower, color="#3498db", linestyle="--", linewidth=1.4, label=f"-1.96 SD = {loa_lower:.2f}")
    ax.axhline(0,         color="gray",    linestyle=":",  linewidth=1,   label="Zero (no bias)")

    # Shade the limits of agreement region
    ax.fill_between(
        [mean_val.min(), mean_val.max()],
        loa_lower, loa_upper,
        alpha=0.07, color="#3498db"
    )

    # Add proportional bias note
    prop_note = f"Proportional bias: r={r_prop:.2f}, p={p_prop:.3f}"
    ax.text(
        0.02, 0.97, prop_note,
        transform=ax.transAxes, fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7)
    )

    ax.set_xlabel(f"Mean of Santa Ana & Reference ({var})")
    ax.set_ylabel("Difference (Santa Ana − Reference)")
    ax.set_title(f"Bland-Altman Plot — {var}", fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")

    plt.tight_layout()
    fname = var.replace("%", "pct").replace("/", "_").replace(" ", "_").replace("(", "").replace(")", "")
    fpath = FIG_DIR / f"bland_altman_{fname}.png"
    plt.savefig(fpath, dpi=150)
    plt.close()
    print(f"  Saved: {fpath.name}")

ba_df = pd.DataFrame(ba_results)
print("\n--- Bland-Altman summary ---")
print(ba_df.to_string(index=False))

ba_path = TABLES / "phase4_bland_altman.csv"
ba_df.to_csv(ba_path, index=False, encoding="utf-8")
print(f"\nSaved: {ba_path}")


# =============================================================================
# 2. PASSING-BABLOK REGRESSION
# =============================================================================
# Plots Santa Ana vs Reference and fits the Passing-Bablok line.
# The perfect agreement line (slope=1, intercept=0) is shown for reference.
# If the confidence interval of the slope excludes 1 → proportional bias.
# If the confidence interval of the intercept excludes 0 → constant bias.

print("\n--- Running Passing-Bablok regression ---")

pb_results = []

for var in QUALITY_VARS:
    sa  = df[f"{var}_{MILL_SANTA_ANA}"].values
    ref = df[f"{var}_Reference"].values

    pb = passing_bablok(ref, sa)
    pb["Variable"] = var

    # Interpret results
    constant_bias     = not (pb["intercept_ci_low"] <= 0 <= pb["intercept_ci_high"])
    proportional_bias = not (pb["slope_ci_low"]      <= 1 <= pb["slope_ci_high"])
    pb["Constant bias"]     = "Yes" if constant_bias     else "No"
    pb["Proportional bias"] = "Yes" if proportional_bias else "No"

    pb_results.append(pb)

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(7, 6))

    ax.scatter(ref, sa, color="#e74c3c", alpha=0.6, s=50, edgecolors="white", linewidths=0.5)

    # Passing-Bablok line
    x_range = np.linspace(min(ref), max(ref), 200)
    ax.plot(
        x_range, pb["intercept"] + pb["slope"] * x_range,
        color="#e74c3c", linewidth=2,
        label=f"PB line: y = {pb['intercept']:.2f} + {pb['slope']:.2f}x"
    )

    # Perfect agreement line (slope=1, intercept=0)
    ax.plot(
        x_range, x_range,
        color="gray", linestyle="--", linewidth=1.5,
        label="Perfect agreement (y = x)"
    )

    ax.set_xlabel(f"Reference (oblique) — {var}")
    ax.set_ylabel(f"Santa Ana (horizontal) — {var}")
    ax.set_title(f"Passing-Bablok Regression — {var}", fontweight="bold")

    # Annotate bias diagnosis
    note = f"Constant bias: {pb['Constant bias']}  |  Proportional bias: {pb['Proportional bias']}"
    ax.text(
        0.02, 0.97, note,
        transform=ax.transAxes, fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7)
    )

    ax.legend(fontsize=9)
    plt.tight_layout()

    fname = var.replace("%", "pct").replace("/", "_").replace(" ", "_").replace("(", "").replace(")", "")
    fpath = FIG_DIR / f"passing_bablok_{fname}.png"
    plt.savefig(fpath, dpi=150)
    plt.close()
    print(f"  Saved: {fpath.name}")

pb_df = pd.DataFrame(pb_results)
print("\n--- Passing-Bablok summary ---")
print(pb_df[["Variable", "slope", "slope_ci_low", "slope_ci_high",
             "intercept", "intercept_ci_low", "intercept_ci_high",
             "Constant bias", "Proportional bias"]].to_string(index=False))

pb_path = TABLES / "phase4_passing_bablok.csv"
pb_df.to_csv(pb_path, index=False, encoding="utf-8")
print(f"\nSaved: {pb_path}")


# =============================================================================
# 3. CONCORDANCE CORRELATION COEFFICIENT
# =============================================================================

print("\n--- Computing Concordance Correlation Coefficients ---")

ccc_results = []
for var in QUALITY_VARS:
    sa  = df[f"{var}_{MILL_SANTA_ANA}"]
    ref = df[f"{var}_Reference"]
    ccc, r = concordance_correlation(sa, ref)
    ccc_results.append({
        "Variable"   : var,
        "Pearson r"  : r,
        "CCC"        : ccc,
        "Difference" : round(r - ccc, 4),  # gap reveals systematic bias
    })

ccc_df = pd.DataFrame(ccc_results)
print(ccc_df.to_string(index=False))

ccc_path = TABLES / "phase4_ccc.csv"
ccc_df.to_csv(ccc_path, index=False, encoding="utf-8")
print(f"\nSaved: {ccc_path}")

print("\nPhase 4 complete.")
