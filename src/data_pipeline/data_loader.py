"""
HydroCast — Data Loader
Loads, validates, and merges all datasets:
  - IDSP weekly district outbreak reports
  - IMD rainfall / climate data
  - NFHS-5 district-level WASH indicators
  - Maharashtra GeoJSON district boundaries

If real data files are not available, generate_synthetic_data()
produces realistic demo data so the pipeline runs end-to-end.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

# ── Internal imports
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import (
    IDSP_FILE, IMD_FILE, NFHS_FILE, GEOJSON_FILE,
    MAHARASHTRA_DISTRICTS, DISEASES, DATA_PROCESSED_DIR,
    CONFIG,
)

# ── Logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("hydrocast.data_loader")


# ══════════════════════════════════════════════════════════════════
# 1. IDSP LOADER
# ══════════════════════════════════════════════════════════════════

def load_idsp_data(filepath: Path = IDSP_FILE) -> pd.DataFrame:
    """
    Load IDSP weekly district-wise case counts from CSV.

    Expected CSV columns:
        district, week, year, cholera_cases, typhoid_cases,
        add_cases, hepatitis_cases

    Returns
    -------
    pd.DataFrame
        DatetimeIndex (Monday of each ISO week), columns:
        [district, cholera_cases, typhoid_cases, add_cases]
    """
    logger.info(f"Loading IDSP data from: {filepath}")

    df = pd.read_csv(filepath)

    # ── Standardise column names
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    required = {"district", "week_of_year", "year", "cholera_cases",
            "typhoid_cases", "add_cases"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"IDSP CSV missing columns: {missing}")

    # ── Convert week + year → Monday date
    df["date"] = pd.to_datetime(
        df["year"].astype(str) + "-W" + df["week_of_year"].astype(str).str.zfill(2) + "-1",
        format="%G-W%V-%u",
    )

    # ── Validate districts
    unknown = set(df["district"].unique()) - set(MAHARASHTRA_DISTRICTS)
    if unknown:
        logger.warning(f"Unknown districts in IDSP data (will keep): {unknown}")

    # ── Clean negatives, forward-fill missing weeks per district
    for col in ["cholera_cases", "typhoid_cases", "add_cases"]:
        df[col] = df[col].clip(lower=0)

    df = df.sort_values(["district", "date"])
    df[["cholera_cases", "typhoid_cases", "add_cases"]] = (
        df.groupby("district")[["cholera_cases", "typhoid_cases", "add_cases"]]
        .transform(lambda s: s.ffill())
    )

    df = df[["district", "date", "cholera_cases", "typhoid_cases", "add_cases"]]
    df = df.set_index("date")

    logger.info(f"IDSP loaded: {df.shape[0]} rows, "
                f"{df['district'].nunique()} districts")
    return df


# ══════════════════════════════════════════════════════════════════
# 2. IMD CLIMATE LOADER
# ══════════════════════════════════════════════════════════════════

def load_imd_data(filepath: Path = IMD_FILE) -> pd.DataFrame:
    """
    Load IMD weekly rainfall / climate data per district.

    Expected CSV columns:
        district, week, year, rainfall_mm, temperature_c, humidity_pct

    Adds:
        rainfall_anomaly_pct  — % deviation from district historical mean
        monsoon_flag          — True if month in [6, 7, 8, 9]

    Returns
    -------
    pd.DataFrame  indexed by date
    """
    logger.info(f"Loading IMD climate data from: {filepath}")

    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    required = {"district", "rainfall_mm", "temperature_c", "humidity_pct"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"IMD CSV missing columns: {missing}")

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=True)
    elif {"year", "week_of_year"}.issubset(df.columns):
        df["date"] = pd.to_datetime(
            df["year"].astype(str) + "-W" + df["week_of_year"].astype(str).str.zfill(2) + "-1",
            format="%G-W%V-%u",
        )
    elif {"year", "week"}.issubset(df.columns):
        df["date"] = pd.to_datetime(
            df["year"].astype(str) + "-W" + df["week"].astype(str).str.zfill(2) + "-1",
            format="%G-W%V-%u",
            errors="coerce",
        )
    else:
        raise ValueError(
            "IMD CSV must contain either 'date' or the pair "
            "('year', 'week_of_year') / ('year', 'week')."
        )

    df = df.dropna(subset=["date"]).copy()

    # ── Rainfall anomaly: (observed - district_mean) / district_mean * 100
    district_means = (
        df.groupby("district")["rainfall_mm"]
        .transform("mean")
        .replace(0, np.nan)        # avoid division by zero
    )
    df["rainfall_anomaly_pct"] = (
        (df["rainfall_mm"] - district_means) / district_means * 100
    ).fillna(0)

    # ── Monsoon flag (June–September = months 6–9)
    if "monsoon_flag" not in df.columns:
        df["monsoon_flag"] = df["date"].dt.month.isin([6, 7, 8, 9]).astype(int)

    df = df[["district", "date", "rainfall_mm", "rainfall_anomaly_pct",
             "temperature_c", "humidity_pct", "monsoon_flag"]]
    df = df.set_index("date")

    logger.info(f"IMD loaded: {df.shape[0]} rows")
    return df


# ══════════════════════════════════════════════════════════════════
# 3. NFHS-5 WASH LOADER
# ══════════════════════════════════════════════════════════════════

def load_nfhs5_data(filepath: Path = NFHS_FILE) -> pd.DataFrame:
    """
    Load NFHS-5 district-level static WASH indicators.

    Expected CSV columns:
        district, sanitation_coverage_pct, od_index,
        water_access_pct, population_density, urban_pct

    All numeric columns are normalized to [0, 1].

    Returns
    -------
    pd.DataFrame  indexed by district (one row per district)
    """
    logger.info(f"Loading NFHS-5 data from: {filepath}")

    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    required = {"district", "sanitation_coverage_pct", "od_index",
                "water_access_pct", "population_density", "urban_pct"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"NFHS-5 CSV missing columns: {missing}")

    numeric_cols = ["sanitation_coverage_pct", "od_index",
                    "water_access_pct", "population_density", "urban_pct"]

    # ── Min-max normalize to [0, 1]
    for col in numeric_cols:
        col_min = df[col].min()
        col_max = df[col].max()
        if col_max > col_min:
            df[col] = (df[col] - col_min) / (col_max - col_min)
        else:
            df[col] = 0.5

    df = df.set_index("district")
    logger.info(f"NFHS-5 loaded: {df.shape[0]} districts")
    return df


# ══════════════════════════════════════════════════════════════════
# 4. GEOJSON LOADER
# ══════════════════════════════════════════════════════════════════

def load_geojson(filepath: Path = GEOJSON_FILE) -> gpd.GeoDataFrame:
    """
    Load Maharashtra district boundary GeoJSON using GeoPandas.

    Returns
    -------
    gpd.GeoDataFrame  with a standardised 'district' column and geometry
    """
    logger.info(f"Loading GeoJSON from: {filepath}")

    gdf = gpd.read_file(filepath)

    # ── Standardise district name column
    for candidate in ["DISTRICT", "district", "NAME_2", "name", "dt_name"]:
        if candidate in gdf.columns:
            gdf = gdf.rename(columns={candidate: "district"})
            break

    if "district" not in gdf.columns:
        raise ValueError("GeoJSON has no recognisable district name column. "
                         "Add a column named 'district'.")

    logger.info(f"GeoJSON loaded: {len(gdf)} features")
    return gdf


# ══════════════════════════════════════════════════════════════════
# 5. MERGE ALL DATASETS
# ══════════════════════════════════════════════════════════════════

def merge_all_datasets(
    idsp_path:  Path = IDSP_FILE,
    imd_path:   Path = IMD_FILE,
    nfhs_path:  Path = NFHS_FILE,
) -> pd.DataFrame:
    """
    Load and merge IDSP + IMD + NFHS-5 into a single flat DataFrame.

    Join logic:
        IDSP ⟕ IMD   on (district, date)
        result ⟕ NFHS on district   (static, broadcast to all rows)

    Returns
    -------
    pd.DataFrame  with all features, indexed by (district, date)
    """
    logger.info("=== Starting full dataset merge ===")

    idsp  = load_idsp_data(idsp_path).reset_index()
    imd   = load_imd_data(imd_path).reset_index()
    nfhs  = load_nfhs5_data(nfhs_path)

    # ── IDSP + IMD on district + date
    merged = pd.merge(
        idsp, imd,
        on=["district", "date"],
        how="left",
        validate="many_to_one",
    )

    before = len(merged)
    logger.info(f"After IDSP+IMD merge: {before} rows")

    # ── + NFHS-5 (static) on district
    merged = merged.merge(
        nfhs.reset_index(),
        on="district",
        how="left",
    )

    after = len(merged)
    assert before == after, "Rows lost during NFHS-5 join — check district names!"

    # ── Validate no districts dropped
    final_districts = set(merged["district"].unique())
    expected        = set(MAHARASHTRA_DISTRICTS)
    dropped         = expected - final_districts
    if dropped:
        logger.warning(f"Districts missing from merged data: {dropped}")

    merged = merged.set_index(["district", "date"]).sort_index()
    logger.info(f"Final merged dataset: {merged.shape}")
    logger.info(f"Columns: {list(merged.columns)}")

    # ── Save processed
    out = DATA_PROCESSED_DIR / "merged_dataset.csv"
    merged.to_csv(out)
    logger.info(f"Saved merged dataset to: {out}")

    return merged


# ══════════════════════════════════════════════════════════════════
# 6. SYNTHETIC DATA GENERATOR
# ══════════════════════════════════════════════════════════════════

def generate_synthetic_data(n_weeks: int = 104) -> pd.DataFrame:
    """
    Generate realistic synthetic data for all 36 Maharashtra districts.

    Used when real IDSP/IMD/NFHS-5 files are not available.
    Simulates seasonal patterns, monsoon spikes, and WASH correlations.

    WARNING
    -------
    This is SYNTHETIC DATA. Replace with real IDSP files for
    actual research and publication.

    Parameters
    ----------
    n_weeks : int
        Number of weeks to simulate (default = 2 years = 104 weeks)

    Returns
    -------
    pd.DataFrame  same schema as merge_all_datasets()
    """
    warnings.warn(
        "\n" + "=" * 65 +
        "\nWARNING: Using SYNTHETIC DATA for demonstration.\n"
        "Replace data/raw/*.csv files with real IDSP/IMD/NFHS-5 data\n"
        "before running experiments for your research paper!\n" +
        "=" * 65,
        UserWarning, stacklevel=2
    )

    rng = np.random.default_rng(seed=CONFIG.training.seed)

    start_date = pd.Timestamp("2022-01-03")   # first Monday of 2022
    dates      = pd.date_range(start_date, periods=n_weeks, freq="W-MON")

    rows = []
    for district in MAHARASHTRA_DISTRICTS:
        # ── Static WASH profile (varies by district)
        dist_idx          = MAHARASHTRA_DISTRICTS.index(district)
        base_sanitation   = 0.3 + 0.5 * (dist_idx / len(MAHARASHTRA_DISTRICTS))
        od_index          = 1.0 - base_sanitation + rng.normal(0, 0.05)
        od_index          = float(np.clip(od_index, 0, 1))
        water_access      = float(np.clip(base_sanitation + 0.1 + rng.normal(0, 0.05), 0, 1))
        pop_density       = float(rng.uniform(100, 3000))
        urban_pct         = float(np.clip(0.2 + 0.6 * (dist_idx / len(MAHARASHTRA_DISTRICTS)), 0, 1))
        wash_index        = (base_sanitation * 0.4 + water_access * 0.4 + (1 - od_index) * 0.2)

        for i, date in enumerate(dates):
            month      = date.month
            week_of_yr = date.isocalendar().week
            monsoon    = 1 if month in [6, 7, 8, 9] else 0

            # ── Rainfall simulation (strong seasonal pattern)
            rainfall_base = 20 + 80 * monsoon + rng.normal(0, 10)
            rainfall_mm   = float(max(0, rainfall_base))
            rainfall_anom = float(rng.normal(0, 30) if monsoon else rng.normal(0, 10))

            temperature   = float(20 + 12 * np.sin(2 * np.pi * (month - 3) / 12) + rng.normal(0, 2))
            humidity      = float(40 + 40 * monsoon + rng.normal(0, 5))

            # ── Case counts (higher in monsoon, higher with poor sanitation)
            base_risk  = (1 - base_sanitation) * (1 + monsoon * 2.5)
            cholera    = int(max(0, rng.poisson(base_risk * 3 + rainfall_mm * 0.03)))
            typhoid    = int(max(0, rng.poisson(base_risk * 5 + temperature * 0.1)))
            add        = int(max(0, rng.poisson(base_risk * 15 + rainfall_mm * 0.1)))

            rows.append({
                "district":               district,
                "date":                   date,
                "cholera_cases":          cholera,
                "typhoid_cases":          typhoid,
                "add_cases":              add,
                "rainfall_mm":            rainfall_mm,
                "rainfall_anomaly_pct":   rainfall_anom,
                "temperature_c":          temperature,
                "humidity_pct":           humidity,
                "monsoon_flag":           monsoon,
                "sanitation_coverage_pct":base_sanitation,
                "od_index":               od_index,
                "water_access_pct":       water_access,
                "population_density":     pop_density,
                "urban_pct":              urban_pct,
                "wash_index":             wash_index,
                "week_of_year":           int(week_of_yr),
                "month":                  month,
            })

    df = pd.DataFrame(rows)
    df = df.set_index(["district", "date"]).sort_index()

    # ── Save synthetic dataset
    out = DATA_PROCESSED_DIR / "synthetic_dataset.csv"
    df.to_csv(out)
    logger.info(f"Synthetic dataset saved: {df.shape} → {out}")

    return df


# ══════════════════════════════════════════════════════════════════
# MAIN — Quick smoke test
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logger.info("Running data_loader smoke test with synthetic data...")

    df = generate_synthetic_data(n_weeks=52)

    print("\n── DataFrame head ──")
    print(df.head(10).to_string())
    print(f"\n── Shape: {df.shape}")
    print(f"── Columns: {list(df.columns)}")
    print(f"── Districts: {df.index.get_level_values('district').nunique()}")
    print(f"── Date range: {df.index.get_level_values('date').min()} "
          f"→ {df.index.get_level_values('date').max()}")
    print(f"── Null count:\n{df.isnull().sum()}")
    print("\n✅ data_loader.py smoke test passed.")
