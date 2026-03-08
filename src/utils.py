# utils.py
# Shared constants and helper functions used across all phases.

import pandas as pd
from pathlib import Path

# --- Project paths ---
ROOT = Path(__file__).resolve().parent.parent
DATA_RAW        = ROOT / "data" / "raw"
DATA_PROCESSED  = ROOT / "data" / "processed"
FIGURES         = ROOT / "outputs" / "figures"
TABLES          = ROOT / "outputs" / "tables"

# --- Mills ---
MILL_SANTA_ANA  = "Santa Ana"
MILL_LA_UNION   = "La Unión"
MILL_PANTALEON  = "Pantaleón"
REFERENCE_MILLS = [MILL_LA_UNION, MILL_PANTALEON]   # oblique core samplers
ALL_MILLS       = [MILL_SANTA_ANA, MILL_LA_UNION, MILL_PANTALEON]

# Color assigned to each mill for consistent plots across all phases
MILL_COLORS = {
    MILL_SANTA_ANA : "#e74c3c",   # red   — the mill under study
    MILL_LA_UNION  : "#2ecc71",   # green
    MILL_PANTALEON : "#3498db",   # blue
}

# --- Variables of interest ---
# These are the cane quality measurements taken by the core sampler
QUALITY_VARS = [
    "%jugo",
    "Brix JE (%)",
    "Pol JE (%)",
    "Pza JE",
    "Fibra%Caña",
    "Pol%caña (kg/t)",
]

# Primary variable — sucrose per ton of cane, the most economically relevant
PRIMARY_VAR = "Pol%caña (kg/t)"


def load_raw_data() -> pd.DataFrame:
    """
    Reads the original CSV and returns a clean DataFrame.
    Handles the Windows-1252 encoding common in files exported from Excel in Spanish.
    """
    path = DATA_RAW / "Base de datos.csv"
    df = pd.read_csv(path, encoding="cp1252")

    # Convert date column to proper datetime format
    df["Fecha"] = pd.to_datetime(df["Fecha"], dayfirst=True)

    # Rename columns to remove encoding artifacts, just in case
    df.columns = df.columns.str.strip()

    return df


def load_paired_data() -> pd.DataFrame:
    """
    Reads the paired dataset produced by phase1_data_prep.py.
    This is what phases 2-7 should use.
    """
    path = DATA_PROCESSED / "paired_data.csv"
    df = pd.read_csv(path, encoding="utf-8")
    df["Fecha"] = pd.to_datetime(df["Fecha"])
    return df
