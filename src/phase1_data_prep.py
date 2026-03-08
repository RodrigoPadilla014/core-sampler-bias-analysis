# phase1_data_prep.py
# Goal: load the raw data, verify it is correctly paired, and produce a clean
# paired dataset that all subsequent phases will use.

import pandas as pd
import numpy as np
from utils import (
    load_raw_data,
    MILL_SANTA_ANA, MILL_LA_UNION, MILL_PANTALEON,
    ALL_MILLS, REFERENCE_MILLS,
    QUALITY_VARS, PRIMARY_VAR,
    DATA_PROCESSED, TABLES,
)


# =============================================================================
# 1. LOAD
# =============================================================================

print("=" * 60)
print("PHASE 1 — Data Preparation")
print("=" * 60)

df = load_raw_data()

print(f"\nRaw data loaded: {len(df)} rows, {len(df.columns)} columns")
print(f"Columns: {list(df.columns)}")


# =============================================================================
# 2. BASIC QUALITY CHECK
# =============================================================================

print("\n--- Missing values ---")
missing = df.isnull().sum()
missing = missing[missing > 0]
if missing.empty:
    print("No missing values found.")
else:
    print(missing)

print("\n--- Mills in the dataset ---")
print(df["Ingenio"].value_counts())

print("\n--- Weeks in the dataset ---")
print(f"Weeks: {sorted(df['Semana'].unique())}")

print("\n--- Cut types ---")
print(df["Corte"].value_counts())

print("\n--- Q or V (Quemazón / Verde) ---")
print(df["Q o V"].value_counts())


# =============================================================================
# 3. VERIFY PAIRING INTEGRITY
# =============================================================================
# The pairing key is: Semana + # Muestra
# For each (Semana, # Muestra) combination, we expect one row per mill.
# If a sample number is missing for any mill in a given week, it cannot be
# used as a true pair and should be flagged.

print("\n--- Checking pairing integrity ---")

# Count how many mills reported each (Semana, # Muestra) combination
pair_counts = (
    df.groupby(["Semana", "# Muestra"])["Ingenio"]
    .nunique()
    .reset_index(name="mills_reporting")
)

complete_pairs = pair_counts[pair_counts["mills_reporting"] == 3]
incomplete_pairs = pair_counts[pair_counts["mills_reporting"] < 3]

print(f"Complete triplets (all 3 mills): {len(complete_pairs)}")
print(f"Incomplete pairs (missing a mill): {len(incomplete_pairs)}")

if not incomplete_pairs.empty:
    print("\nIncomplete pairs:")
    print(incomplete_pairs)


# =============================================================================
# 4. BUILD THE PAIRED DATASET
# =============================================================================
# We pivot the data so that each row represents one (Semana, # Muestra) triplet,
# with separate columns for each mill's readings.
# This layout makes paired comparisons straightforward in all later phases.

print("\n--- Building paired dataset ---")

# Keep only complete triplets
valid_keys = complete_pairs[["Semana", "# Muestra"]]
df_valid = df.merge(valid_keys, on=["Semana", "# Muestra"])

# Pivot: one row per (Semana, # Muestra), columns for each mill x variable
pivot = df_valid.pivot_table(
    index=["Semana", "# Muestra", "Fecha", "Corte", "Q o V"],
    columns="Ingenio",
    values=QUALITY_VARS,
    aggfunc="first",  # each cell should already be unique
)

# Flatten the multi-level column names: "Pol%caña (kg/t)_Santa Ana" etc.
pivot.columns = [f"{var}_{mill}" for var, mill in pivot.columns]
pivot = pivot.reset_index()

print(f"Paired dataset shape: {pivot.shape}")
print(f"Number of complete triplets: {len(pivot)}")


# =============================================================================
# 5. ADD REFERENCE COLUMNS
# =============================================================================
# For each quality variable, compute the mean of the two reference mills
# (La Unión and Pantaleón — both use oblique core samplers).
# Then compute the difference: Santa Ana minus Reference.
# This difference isolates the horizontal sampler effect.

print("\n--- Adding reference and difference columns ---")

for var in QUALITY_VARS:
    col_sa  = f"{var}_{MILL_SANTA_ANA}"
    col_lu  = f"{var}_{MILL_LA_UNION}"
    col_pan = f"{var}_{MILL_PANTALEON}"
    col_ref = f"{var}_Reference"
    col_diff = f"{var}_Diff"    # positive = Santa Ana reads higher than reference

    pivot[col_ref]  = (pivot[col_lu] + pivot[col_pan]) / 2
    pivot[col_diff] = pivot[col_sa] - pivot[col_ref]


# =============================================================================
# 6. OUTLIER FLAGGING
# =============================================================================
# Flag rows where the difference in the primary variable (Pol%caña) is more
# than 3 standard deviations from the mean difference.
# We flag but do NOT remove — the decision is left to the analyst.

diff_col = f"{PRIMARY_VAR}_Diff"
mean_diff = pivot[diff_col].mean()
std_diff  = pivot[diff_col].std()

pivot["outlier_flag"] = (
    (pivot[diff_col] - mean_diff).abs() > 3 * std_diff
)

n_outliers = pivot["outlier_flag"].sum()
print(f"Outliers flagged (>3 SD in {PRIMARY_VAR} difference): {n_outliers}")

if n_outliers > 0:
    print(pivot[pivot["outlier_flag"]][["Semana", "# Muestra", diff_col]])


# =============================================================================
# 7. SAVE OUTPUTS
# =============================================================================

# Clean paired dataset — used by all other phases
output_path = DATA_PROCESSED / "paired_data.csv"
pivot.to_csv(output_path, index=False, encoding="utf-8")
print(f"\nPaired dataset saved to: {output_path}")

# Summary of differences — quick reference table
summary_rows = []
for var in QUALITY_VARS:
    diffs = pivot[f"{var}_Diff"]
    summary_rows.append({
        "Variable"  : var,
        "Mean diff" : round(diffs.mean(), 4),
        "SD diff"   : round(diffs.std(), 4),
        "Min diff"  : round(diffs.min(), 4),
        "Max diff"  : round(diffs.max(), 4),
        "n"         : len(diffs),
    })

summary_df = pd.DataFrame(summary_rows)
summary_path = TABLES / "phase1_difference_summary.csv"
summary_df.to_csv(summary_path, index=False, encoding="utf-8")

print(f"\n--- Difference summary (Santa Ana minus Reference) ---")
print(summary_df.to_string(index=False))
print(f"\nSummary table saved to: {summary_path}")

print("\nPhase 1 complete.")
