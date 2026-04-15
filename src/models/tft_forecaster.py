"""
HydroCast — Temporal Fusion Transformer Forecaster
Multi-horizon 2-week and 4-week outbreak probability forecasting.

Reference: Lim et al. 2020
"Temporal Fusion Transformers for Interpretable Multi-horizon
Time Series Forecasting"
https://arxiv.org/abs/1912.09363

Uses pytorch-forecasting library as backbone.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import CONFIG, DISEASES, MAHARASHTRA_DISTRICTS, RESULTS_DIR

logger = logging.getLogger("hydrocast.tft_forecaster")


# ══════════════════════════════════════════════════════════════════
# DATASET PREPARATION
# ══════════════════════════════════════════════════════════════════

def prepare_timeseries_dataset(
    df:   pd.DataFrame,
    mode: str = "train",
    max_encoder_length:    int = 52,
    max_prediction_length: int = 4,
):
    """
    Convert merged + featured DataFrame to pytorch-forecasting
    TimeSeriesDataSet.

    Parameters
    ----------
    df   : featured DataFrame from FeatureEngineer.transform()
           MultiIndex (district, date) or flat with those columns.
    mode : "train", "val", or "test"

    Returns
    -------
    pytorch_forecasting.TimeSeriesDataSet
    """
    try:
        from pytorch_forecasting import TimeSeriesDataSet
        from pytorch_forecasting.data import GroupNormalizer
    except ImportError:
        raise ImportError(
            "pytorch-forecasting not installed. "
            "Run: pip install pytorch-forecasting"
        )

    # ── Flatten MultiIndex if needed
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()
    else:
        df = df.copy()

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["district", "date"])

    # ── Create integer time index (weeks since start)
    min_date      = df["date"].min()
    df["time_idx"] = ((df["date"] - min_date).dt.days // 7).astype(int)

    # ── Normalise case columns to [0,1] per district
    for col in ["cholera_cases", "typhoid_cases", "add_cases"]:
        if col in df.columns:
            grp_max = df.groupby("district")[col].transform("max").replace(0, 1)
            df[f"{col}_norm"] = (df[col] / grp_max).fillna(0).clip(0, 1)

    # ── Define feature groups
    time_varying_known = [
        c for c in [
            "rainfall_mm", "rainfall_anomaly_pct", "temperature_c",
            "humidity_pct", "monsoon_flag", "season_sin", "season_cos",
            "week_of_year", "month",
        ] if c in df.columns
    ]

    time_varying_unknown = [
        c for c in df.columns
        if any(f"_lag" in c or f"_roll" in c for _ in [c])
        and c not in time_varying_known
    ] + [
        c for c in ["cholera_cases_norm", "typhoid_cases_norm", "add_cases_norm"]
        if c in df.columns
    ]

    static_reals = [
        c for c in [
            "sanitation_coverage_pct", "od_index", "water_access_pct",
            "population_density", "urban_pct", "wash_index",
        ] if c in df.columns
    ]

    # ── Target column: use ADD cases as primary (most common)
    target = "add_cases_norm" if "add_cases_norm" in df.columns else "add_cases"

    # ── Min encoder length
    min_encoder_length = max_encoder_length // 2

    dataset = TimeSeriesDataSet(
        data                       = df,
        time_idx                   = "time_idx",
        target                     = target,
        group_ids                  = ["district"],
        min_encoder_length         = min_encoder_length,
        max_encoder_length         = max_encoder_length,
        min_prediction_length      = 1,
        max_prediction_length      = max_prediction_length,
        static_categoricals        = ["district"],
        static_reals               = static_reals,
        time_varying_known_reals   = time_varying_known,
        time_varying_unknown_reals = time_varying_unknown,
        target_normalizer          = GroupNormalizer(
            groups=["district"], transformation="softplus"
        ),
        add_relative_time_idx      = True,
        add_target_scales          = True,
        add_encoder_length         = True,
        allow_missing_timesteps    = True,
    )

    logger.info(
        f"TimeSeriesDataSet [{mode}]: {len(dataset)} samples | "
        f"encoder={max_encoder_length}wk | prediction={max_prediction_length}wk"
    )
    return dataset


# ══════════════════════════════════════════════════════════════════
# MAHAWATCH FORECASTER WRAPPER
# ══════════════════════════════════════════════════════════════════

class HydroCastForecaster:
    """
    Wrapper around pytorch-forecasting's TemporalFusionTransformer
    for HydroCast multi-district outbreak probability forecasting.

    Usage
    -----
    >>> forecaster = HydroCastForecaster(config=CONFIG.tft)
    >>> train_ds = forecaster.prepare_timeseries_dataset(train_df, "train")
    >>> model    = forecaster.build_model(train_ds)
    >>> preds    = forecaster.predict(test_loader)
    """

    def __init__(self, config=None) -> None:
        self.config  = config or CONFIG.tft
        self.model   = None
        self.trainer = None

    def prepare_timeseries_dataset(
        self,
        df:   pd.DataFrame,
        mode: str = "train",
    ):
        """Delegate to module-level function."""
        return prepare_timeseries_dataset(
            df   = df,
            mode = mode,
            max_encoder_length    = self.config.max_encoder_length,
            max_prediction_length = self.config.max_prediction_length,
        )

    def build_model(self, train_dataset):
        """
        Build TemporalFusionTransformer from training dataset.

        Parameters
        ----------
        train_dataset : TimeSeriesDataSet

        Returns
        -------
        pytorch_forecasting.TemporalFusionTransformer
        """
        try:
            from pytorch_forecasting import TemporalFusionTransformer
            from pytorch_forecasting.metrics import QuantileLoss
        except ImportError:
            raise ImportError("Install pytorch-forecasting first.")

        self.model = TemporalFusionTransformer.from_dataset(
            train_dataset,
            learning_rate           = CONFIG.training.learning_rate,
            hidden_size             = self.config.hidden_size,
            attention_head_size     = self.config.attention_head_size,
            dropout                 = self.config.dropout,
            hidden_continuous_size  = self.config.hidden_continuous_size,
            loss                    = QuantileLoss(self.config.quantiles),
            log_interval            = 10,
            reduce_on_plateau_patience = 4,
        )

        n_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"TFT built: {n_params:,} parameters")
        return self.model

    def predict(
        self,
        dataloader,
        mode: str = "prediction",
    ) -> dict[str, dict[str, float]]:
        """
        Run inference and return per-district per-horizon probabilities.

        Returns
        -------
        dict  {district: {week+1: prob, week+2: prob, week+3: prob, week+4: prob}}
        """
        if self.model is None:
            raise RuntimeError("Call build_model() before predict().")

        self.model.eval()
        predictions = self.model.predict(dataloader, mode=mode)

        # predictions shape: (n_samples, max_prediction_length)
        result: dict[str, dict[str, float]] = {}
        pred_np = predictions.detach().cpu().numpy() if hasattr(predictions, "detach") \
                  else np.array(predictions)

        # Map back to districts (assumes same order as dataloader)
        for i, district in enumerate(MAHARASHTRA_DISTRICTS[:len(pred_np)]):
            row         = pred_np[i]
            horizon     = min(len(row), self.config.max_prediction_length)
            result[district] = {
                f"week+{h+1}": float(np.clip(row[h], 0, 1))
                for h in range(horizon)
            }

        return result

    def get_variable_importance(self) -> pd.DataFrame:
        """
        Extract TFT built-in variable importance scores.
        These are derived from the variable selection network attention.

        Returns
        -------
        pd.DataFrame  columns: [feature, importance]  sorted descending
        """
        if self.model is None:
            raise RuntimeError("Model not built yet.")

        try:
            interpretation = self.model.interpret_output(
                self.model.predict(return_x=True)[1]
            )
            encoder_imp = interpretation.get("encoder_variables", {})
            decoder_imp = interpretation.get("decoder_variables", {})
            static_imp  = interpretation.get("static_variables", {})

            rows = (
                [{"feature": k, "source": "encoder", "importance": float(v)}
                 for k, v in encoder_imp.items()] +
                [{"feature": k, "source": "decoder", "importance": float(v)}
                 for k, v in decoder_imp.items()] +
                [{"feature": k, "source": "static",  "importance": float(v)}
                 for k, v in static_imp.items()]
            )
            df = pd.DataFrame(rows).sort_values("importance", ascending=False)
            return df

        except Exception as e:
            logger.warning(f"Could not extract variable importance: {e}")
            return pd.DataFrame(columns=["feature", "source", "importance"])


# ══════════════════════════════════════════════════════════════════
# LIGHTWEIGHT LSTM FORECASTER (fallback if TFT not available)
# ══════════════════════════════════════════════════════════════════

class LSTMForecastHead(nn.Module):
    """
    Lightweight LSTM-based multi-horizon forecaster.
    Used as fallback when pytorch-forecasting is unavailable,
    or as the Bi-LSTM baseline in comparison experiments.

    Parameters
    ----------
    input_size  : number of features per timestep
    hidden_size : LSTM hidden dimension
    num_layers  : number of LSTM layers
    horizon     : number of weeks to forecast ahead
    n_diseases  : number of diseases to predict (3)
    """

    def __init__(
        self,
        input_size:  int   = 32,
        hidden_size: int   = 128,
        num_layers:  int   = 2,
        horizon:     int   = 4,
        n_diseases:  int   = 3,
        dropout:     float = 0.2,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        self.horizon    = horizon
        self.n_diseases = n_diseases
        dirs            = 2 if bidirectional else 1

        self.norm = nn.LayerNorm(input_size)
        self.lstm = nn.LSTM(
            input_size    = input_size,
            hidden_size   = hidden_size,
            num_layers    = num_layers,
            dropout       = dropout if num_layers > 1 else 0.0,
            bidirectional = bidirectional,
            batch_first   = True,
        )
        self.dropout = nn.Dropout(dropout)
        self.head    = nn.Sequential(
            nn.Linear(hidden_size * dirs, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, horizon * n_diseases),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, seq_len, input_size)

        Returns
        -------
        torch.Tensor  (batch, n_diseases, horizon)  sigmoid probabilities
        """
        x      = self.norm(x)
        out, _ = self.lstm(x)
        last   = self.dropout(out[:, -1, :])    # last timestep
        pred   = self.head(last)                 # (B, horizon * n_diseases)
        pred   = pred.view(-1, self.n_diseases, self.horizon)
        return torch.sigmoid(pred)


# ══════════════════════════════════════════════════════════════════
# MAIN — Smoke test (uses lightweight LSTM fallback)
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    batch    = 8
    seq_len  = 52
    in_size  = 20
    horizon  = 4

    x = torch.randn(batch, seq_len, in_size)

    # Test lightweight LSTM head
    lstm_model = LSTMForecastHead(
        input_size  = in_size,
        hidden_size = 128,
        num_layers  = 2,
        horizon     = horizon,
        n_diseases  = 3,
        dropout     = 0.2,
        bidirectional = False,
    )
    lstm_model.eval()
    with torch.no_grad():
        out = lstm_model(x)

    print(f"\n── LSTMForecastHead output shape : {out.shape}")
    print(f"   (batch={batch}, diseases=3, horizon={horizon})")
    print(f"── Num params : {sum(p.numel() for p in lstm_model.parameters()):,}")
    print(f"── Value range: [{out.min():.3f}, {out.max():.3f}] (sigmoid → [0,1])")

    # Test Bi-LSTM variant (baseline)
    bilstm = LSTMForecastHead(
        input_size=in_size, bidirectional=True
    )
    with torch.no_grad():
        out_bi = bilstm(x)
    print(f"── Bi-LSTM output shape : {out_bi.shape}")
    print("\n✅ tft_forecaster.py smoke test passed.")
