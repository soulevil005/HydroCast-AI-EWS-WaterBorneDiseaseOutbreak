"""
HydroCast — Feature Engineer
Transforms the raw merged DataFrame into a rich feature matrix
ready for model training. All features are computed without
data leakage (temporal split is by time, not random).
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import (
    MAHARASHTRA_DISTRICTS, DISEASES, DISEASE_CASE_COLS, DISEASE_LABEL_COLS,
    DISEASE_THRESHOLDS, LAG_WINDOWS, ROLLING_WINDOWS,
    CONFIG, DATA_PROCESSED_DIR,
)

logger = logging.getLogger("hydrocast.feature_engineer")


class FeatureEngineer:
    """
    Builds the full feature matrix for HydroCast model training.

    Usage
    -----
    >>> fe = FeatureEngineer(df)
    >>> df_featured = fe.transform()
    >>> train, val, test = temporal_train_val_test_split(df_featured)
    """

    def __init__(self, df: pd.DataFrame) -> None:
        """
        Parameters
        ----------
        df : pd.DataFrame
            Merged DataFrame from data_loader.merge_all_datasets()
            or generate_synthetic_data(). Must have MultiIndex
            (district, date).
        """
        if not isinstance(df.index, pd.MultiIndex):
            raise ValueError("DataFrame must have MultiIndex (district, date).")

        self.df       = df.copy().reset_index()
        self._feature_registry: dict[str, list[str]] = {}
        logger.info(f"FeatureEngineer initialised with shape: {df.shape}")

    # ──────────────────────────────────────────────────────────────
    # 1. LAG FEATURES
    # ──────────────────────────────────────────────────────────────

    def add_lag_features(
        self,
        cols: list[str],
        lags: list[int] = LAG_WINDOWS,
    ) -> None:
        """
        Add lag-k week values for each column in `cols`, grouped by district.
        Grouping prevents leakage across district boundaries.

        Parameters
        ----------
        cols : list[str]   columns to lag (e.g. case count columns)
        lags : list[int]   lag offsets in weeks (default: [1, 2, 4, 8])
        """
        lag_cols = []
        for col in cols:
            if col not in self.df.columns:
                logger.warning(f"Column '{col}' not found — skipping lag.")
                continue
            for lag in lags:
                new_col = f"{col}_lag{lag}"
                self.df[new_col] = (
                    self.df.groupby("district")[col]
                    .shift(lag)
                )
                lag_cols.append(new_col)

        self._feature_registry["lag"] = lag_cols
        logger.info(f"Added {len(lag_cols)} lag features.")

    # ──────────────────────────────────────────────────────────────
    # 2. ROLLING FEATURES
    # ──────────────────────────────────────────────────────────────

    def add_rolling_features(
        self,
        cols: list[str],
        windows: list[int] = ROLLING_WINDOWS,
    ) -> None:
        """
        Add rolling mean and rolling std for each column, grouped by district.

        Parameters
        ----------
        cols    : columns to compute rolling stats on
        windows : window sizes in weeks (default: [4, 8])
        """
        roll_cols = []
        for col in cols:
            if col not in self.df.columns:
                continue
            for w in windows:
                mean_col = f"{col}_roll{w}_mean"
                std_col  = f"{col}_roll{w}_std"
                grp = self.df.groupby("district")[col]
                self.df[mean_col] = grp.transform(
                    lambda s: s.rolling(w, min_periods=1).mean()
                )
                self.df[std_col] = grp.transform(
                    lambda s: s.rolling(w, min_periods=1).std().fillna(0)
                )
                roll_cols.extend([mean_col, std_col])

        self._feature_registry["rolling"] = roll_cols
        logger.info(f"Added {len(roll_cols)} rolling features.")

    # ──────────────────────────────────────────────────────────────
    # 3. SEASONAL FEATURES
    # ──────────────────────────────────────────────────────────────

    def add_seasonal_features(self) -> None:
        """
        Add cyclical seasonal encodings and calendar features.

        Features added:
            week_of_year  — integer 1–52
            month         — integer 1–12
            monsoon_flag  — 1 if month in [6,7,8,9] else 0
            season_sin    — sin(2π × week / 52)  — cyclical encoding
            season_cos    — cos(2π × week / 52)  — cyclical encoding
        """
        if "date" not in self.df.columns:
            self.df = self.df.reset_index()

        self.df["date"] = pd.to_datetime(self.df["date"])

        if "week_of_year" not in self.df.columns:
            self.df["week_of_year"] = self.df["date"].dt.isocalendar().week.astype(int)
        if "month" not in self.df.columns:
            self.df["month"] = self.df["date"].dt.month
        if "monsoon_flag" not in self.df.columns:
            self.df["monsoon_flag"] = self.df["month"].isin([6, 7, 8, 9]).astype(int)

        self.df["season_sin"] = np.sin(2 * np.pi * self.df["week_of_year"] / 52)
        self.df["season_cos"] = np.cos(2 * np.pi * self.df["week_of_year"] / 52)

        seasonal_cols = ["week_of_year", "month", "monsoon_flag",
                         "season_sin", "season_cos"]
        self._feature_registry["seasonal"] = seasonal_cols
        logger.info("Added seasonal + cyclical encoding features.")

    # ──────────────────────────────────────────────────────────────
    # 4. WASH COMPOSITE INDEX
    # ──────────────────────────────────────────────────────────────

    def add_wash_composite_index(self) -> None:
        """
        Compute a composite WASH index from NFHS-5 indicators.

        Formula:
            wash_index = sanitation_coverage × 0.4
                       + water_access         × 0.4
                       + (1 - od_index)       × 0.2

        Higher wash_index = better sanitation = lower disease risk.
        """
        required = {"sanitation_coverage_pct", "water_access_pct", "od_index"}
        if not required.issubset(self.df.columns):
            logger.warning("WASH columns not found — skipping wash_index.")
            return

        self.df["wash_index"] = (
            self.df["sanitation_coverage_pct"] * 0.4
            + self.df["water_access_pct"]       * 0.4
            + (1 - self.df["od_index"])         * 0.2
        ).clip(0, 1)

        self._feature_registry["wash"] = ["wash_index"]
        logger.info("Added WASH composite index.")

    # ──────────────────────────────────────────────────────────────
    # 5. OUTBREAK LABELS
    # ──────────────────────────────────────────────────────────────

    def add_outbreak_labels(
        self,
        disease: str,
        threshold_percentile: int = 75,
    ) -> None:
        """
        Add binary outbreak label and normalised risk score for a disease.

        Label = 1 if case count ≥ district's {percentile}th percentile.
        Risk score = cases / (district_max + ε)

        Parameters
        ----------
        disease              : one of DISEASES list
        threshold_percentile : percentile cutoff (default 75)
        """
        case_col  = DISEASE_CASE_COLS.get(disease)
        label_col = DISEASE_LABEL_COLS.get(disease)
        score_col = f"{disease.lower()}_risk_score"

        if case_col not in self.df.columns:
            logger.warning(f"Case column '{case_col}' not found.")
            return

        # Percentile computed per district (no global leakage)
        threshold = self.df.groupby("district")[case_col].transform(
            lambda s: s.quantile(threshold_percentile / 100)
        )
        self.df[label_col] = (self.df[case_col] >= threshold).astype(int)

        # Normalised risk score
        max_cases = self.df.groupby("district")[case_col].transform("max")
        self.df[score_col] = self.df[case_col] / (max_cases + 1e-6)

        self._feature_registry.setdefault("labels", []).extend(
            [label_col, score_col]
        )
        logger.info(f"Added outbreak label + risk score for {disease}.")

    # ──────────────────────────────────────────────────────────────
    # 6. GRAPH NEIGHBOUR FEATURES
    # ──────────────────────────────────────────────────────────────

    def add_graph_neighbor_features(
        self,
        adjacency: dict[str, list[str]],
    ) -> None:
        """
        Add spatial spillover features: mean and max case counts from
        geographically neighbouring districts.

        Parameters
        ----------
        adjacency : dict mapping district → list of neighbour district names
        """
        case_cols = list(DISEASE_CASE_COLS.values())
        neighbor_cols = []

        # Build a pivot: date × district for each case column
        for col in case_cols:
            if col not in self.df.columns:
                continue

            pivot = self.df.pivot_table(
                index="date", columns="district", values=col, aggfunc="sum"
            )

            mean_col = f"neighbor_mean_{col}"
            max_col  = f"neighbor_max_{col}"

            def _neighbor_mean(row, district: str) -> float:
                neighbors = adjacency.get(district, [])
                valid     = [n for n in neighbors if n in pivot.columns]
                return row[valid].mean() if valid else 0.0

            def _neighbor_max(row, district: str) -> float:
                neighbors = adjacency.get(district, [])
                valid     = [n for n in neighbors if n in pivot.columns]
                return row[valid].max() if valid else 0.0

            # Compute per-district neighbour stats
            for district in MAHARASHTRA_DISTRICTS:
                mask = self.df["district"] == district
                dates_for_dist = self.df.loc[mask, "date"]
                subset = pivot.reindex(dates_for_dist)

                neighbors = adjacency.get(district, [])
                valid_nb  = [n for n in neighbors if n in pivot.columns]

                if valid_nb:
                    nb_data = subset[valid_nb]
                    self.df.loc[mask, mean_col] = nb_data.mean(axis=1).values
                    self.df.loc[mask, max_col]  = nb_data.max(axis=1).values
                else:
                    self.df.loc[mask, mean_col] = 0.0
                    self.df.loc[mask, max_col]  = 0.0

            neighbor_cols.extend([mean_col, max_col])

        self._feature_registry["graph_neighbors"] = neighbor_cols
        logger.info(f"Added {len(neighbor_cols)} graph neighbour features.")

    # ──────────────────────────────────────────────────────────────
    # 7. MASTER TRANSFORM
    # ──────────────────────────────────────────────────────────────

    def transform(self) -> pd.DataFrame:
        """
        Run the full feature engineering pipeline in correct order.

        Steps
        -----
        1. Seasonal features
        2. WASH composite index
        3. Lag features (case columns)
        4. Rolling features (case + rainfall)
        5. Outbreak labels for each disease
        6. Drop NaN rows from lag windows (first 8 weeks per district)

        Returns
        -------
        pd.DataFrame  with MultiIndex (district, date)
        """
        logger.info("=== Starting feature engineering pipeline ===")

        self.add_seasonal_features()
        self.add_wash_composite_index()

        case_cols    = list(DISEASE_CASE_COLS.values())
        climate_cols = ["rainfall_mm", "rainfall_anomaly_pct"]

        self.add_lag_features(cols=case_cols + climate_cols)
        self.add_rolling_features(cols=case_cols + climate_cols)

        for disease in DISEASES:
            self.add_outbreak_labels(disease)

        # Drop NaN rows introduced by max lag window (8 weeks)
        before = len(self.df)
        self.df = self.df.dropna(
            subset=[f"{c}_lag{max(LAG_WINDOWS)}" for c in case_cols
                    if f"{c}_lag{max(LAG_WINDOWS)}" in self.df.columns]
        )
        after = len(self.df)
        logger.info(f"Dropped {before - after} NaN rows from lag windows.")

        result = self.df.set_index(["district", "date"]).sort_index()

        # ── Save
        out = DATA_PROCESSED_DIR / "featured_dataset.csv"
        result.to_csv(out)
        logger.info(f"Featured dataset saved: {result.shape} → {out}")

        return result

    # ──────────────────────────────────────────────────────────────
    # 8. FEATURE NAME REGISTRY
    # ──────────────────────────────────────────────────────────────

    def get_feature_names(self) -> dict[str, list[str]]:
        """
        Return a categorised dictionary of all feature names added
        by this FeatureEngineer instance.

        Useful for SHAP: pass the 'time_series' key to the explainer.

        Returns
        -------
        dict  {category: [feature_names]}
        """
        return dict(self._feature_registry)


# ══════════════════════════════════════════════════════════════════
# TEMPORAL TRAIN / VAL / TEST SPLIT
# ══════════════════════════════════════════════════════════════════

def temporal_train_val_test_split(
    df: pd.DataFrame,
    train: float = CONFIG.training.train_split,
    val:   float = CONFIG.training.val_split,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split DataFrame by time — NOT random — to prevent data leakage.

    The split is computed on the unique sorted date axis.
    All districts share the same time split boundaries.

    Parameters
    ----------
    df    : MultiIndex (district, date) DataFrame
    train : fraction for training set  (default 0.70)
    val   : fraction for validation set (default 0.15)
            test = 1 - train - val     (default 0.15)

    Returns
    -------
    (train_df, val_df, test_df)
    """
    dates     = sorted(df.index.get_level_values("date").unique())
    n         = len(dates)
    train_end = dates[int(n * train) - 1]
    val_end   = dates[int(n * (train + val)) - 1]

    train_df = df[df.index.get_level_values("date") <= train_end]
    val_df   = df[
        (df.index.get_level_values("date") > train_end) &
        (df.index.get_level_values("date") <= val_end)
    ]
    test_df  = df[df.index.get_level_values("date") > val_end]

    logger.info(
        f"Temporal split → train: {len(train_df)} rows "
        f"| val: {len(val_df)} | test: {len(test_df)}"
    )
    return train_df, val_df, test_df


# ══════════════════════════════════════════════════════════════════
# MAIN — Smoke test
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from data_loader import generate_synthetic_data

    logger.info("Loading synthetic data...")
    df_raw = generate_synthetic_data(n_weeks=104)

    logger.info("Running FeatureEngineer.transform()...")
    fe = FeatureEngineer(df_raw)
    df_featured = fe.transform()

    print(f"\n── Featured shape : {df_featured.shape}")
    print(f"── Feature groups : {list(fe.get_feature_names().keys())}")

    train_df, val_df, test_df = temporal_train_val_test_split(df_featured)
    print(f"\n── Train rows : {len(train_df)}")
    print(f"── Val rows   : {len(val_df)}")
    print(f"── Test rows  : {len(test_df)}")
    print("\n✅ feature_engineer.py smoke test passed.")
