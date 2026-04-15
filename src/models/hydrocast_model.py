"""
HydroCast — Master Unified Model
Fuses GATv2 (spatial) + GRU (temporal) + TFT (forecast) + SEIR (physics)
into one end-to-end trainable architecture.

This is the novel contribution of the project:
the first multi-disease spatiotemporal AI-EWS for waterborne
disease outbreaks in Maharashtra with physics-informed constraints.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import CONFIG, DISEASES, MAHARASHTRA_DISTRICTS, DISTRICT_TO_IDX
from src.models.gatv2_encoder   import GATv2Encoder
from src.models.gru_temporal    import GRUTemporalEncoder
from src.models.tft_forecaster  import LSTMForecastHead
from src.models.seir_constraint import SEIRRegularizer

logger = logging.getLogger("hydrocast.model")


# ══════════════════════════════════════════════════════════════════
# DISEASE OUTPUT HEAD
# ══════════════════════════════════════════════════════════════════

class DiseaseHead(nn.Module):
    """
    Per-disease output head: predicts outbreak probabilities
    for a single disease over the forecast horizon.

    Architecture: Linear → ReLU → Dropout → Linear → Sigmoid

    Parameters
    ----------
    input_dim  : input embedding dimension (fusion_dim)
    hidden_dim : intermediate hidden dimension
    horizon    : number of weeks to forecast (4)
    dropout    : dropout probability
    """

    def __init__(
        self,
        input_dim:  int   = 128,
        hidden_dim: int   = 64,
        horizon:    int   = 4,
        dropout:    float = 0.2,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, input_dim)

        Returns
        -------
        torch.Tensor (batch, horizon) — sigmoid outbreak probabilities
        """
        return torch.sigmoid(self.net(x))


# ══════════════════════════════════════════════════════════════════
# CROSS-ATTENTION FUSION LAYER
# ══════════════════════════════════════════════════════════════════

class SpatioTemporalFusion(nn.Module):
    """
    Cross-attention fusion of spatial (GATv2) and temporal (GRU)
    embeddings. Lets each branch attend to the other.

    Architecture:
        concat(spatial, temporal) → CrossAttention → LayerNorm → residual

    Parameters
    ----------
    embed_dim  : dimension of each embedding (must match GATv2 out + GRU out)
    num_heads  : number of attention heads
    dropout    : dropout probability
    """

    def __init__(
        self,
        spatial_dim:  int   = 64,
        temporal_dim: int   = 64,
        num_heads:    int   = 4,
        dropout:      float = 0.2,
    ) -> None:
        super().__init__()

        self.fusion_dim = spatial_dim + temporal_dim

        # Project both to same dimension for cross-attention
        self.proj_spatial  = nn.Linear(spatial_dim, self.fusion_dim)
        self.proj_temporal = nn.Linear(temporal_dim, self.fusion_dim)

        # Multi-head cross-attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim   = self.fusion_dim,
            num_heads   = num_heads,
            dropout     = dropout,
            batch_first = True,
        )

        self.norm    = nn.LayerNorm(self.fusion_dim)
        self.dropout = nn.Dropout(dropout)

        # Feed-forward refinement
        self.ffn = nn.Sequential(
            nn.Linear(self.fusion_dim, self.fusion_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.fusion_dim * 2, self.fusion_dim),
        )
        self.norm2 = nn.LayerNorm(self.fusion_dim)

    def forward(
        self,
        spatial_emb:  torch.Tensor,
        temporal_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        spatial_emb  : (batch, spatial_dim)   from GATv2
        temporal_emb : (batch, temporal_dim)  from GRU

        Returns
        -------
        torch.Tensor (batch, fusion_dim)
        """
        # Project to fusion dim
        s = self.proj_spatial(spatial_emb).unsqueeze(1)   # (B, 1, F)
        t = self.proj_temporal(temporal_emb).unsqueeze(1) # (B, 1, F)

        # Concatenate as sequence: [spatial_token, temporal_token]
        seq = torch.cat([s, t], dim=1)  # (B, 2, F)

        # Self-attention over the 2-token sequence
        attended, _ = self.cross_attn(seq, seq, seq)
        attended     = self.norm(seq + self.dropout(attended))

        # Pool: mean of attended tokens
        pooled = attended.mean(dim=1)  # (B, F)

        # FFN + residual
        out = self.norm2(pooled + self.ffn(pooled))
        return out


# ══════════════════════════════════════════════════════════════════
# MAHAWATCH UNIFIED MODEL
# ══════════════════════════════════════════════════════════════════

class HydroCastModel(nn.Module):
    """
    Multi-disease spatiotemporal AI Early Warning System.

    Architecture
    ------------
    1. SPATIAL BRANCH
       GATv2Encoder(district_graph) → spatial_emb (36, 64)

    2. TEMPORAL BRANCH
       GRUTemporalEncoder(time_series) → temporal_emb (batch, 64)

    3. FUSION
       SpatioTemporalFusion(spatial + temporal) → fused_emb (batch, 128)

    4. OUTPUT HEADS (one per disease, independent)
       DiseaseHead × 3 → {Cholera, Typhoid, ADD}: (batch, 4) probs

    5. SEIR PHYSICS CONSTRAINT (applied in compute_loss())
       Not part of forward graph — regularises via loss term

    Parameters
    ----------
    node_feat_dim    : static node feature dimension (from graph)
    time_feat_dim    : time-varying feature dimension (from feature engineer)
    config           : ModelConfig from src.config
    """

    def __init__(
        self,
        node_feat_dim: int  = 6,
        time_feat_dim: int  = 32,
        config              = None,
    ) -> None:
        super().__init__()

        self.config    = config or CONFIG
        cfg_g  = self.config.gatv2
        cfg_r  = self.config.gru
        cfg_t  = self.config.training

        # ── 1. Spatial encoder (GATv2)
        self.spatial_encoder = GATv2Encoder(
            in_channels     = node_feat_dim,
            hidden_channels = cfg_g.hidden_channels,
            out_channels    = cfg_g.out_channels,
            num_heads       = cfg_g.num_heads,
            num_layers      = cfg_g.num_layers,
            dropout         = cfg_g.dropout,
            edge_dim        = cfg_g.edge_dim,
        )

        # ── 2. Temporal encoder (Bi-GRU)
        self.temporal_encoder = GRUTemporalEncoder(
            input_size    = time_feat_dim,
            hidden_size   = cfg_r.hidden_size,
            num_layers    = cfg_r.num_layers,
            dropout       = cfg_r.dropout,
            bidirectional = cfg_r.bidirectional,
            output_size   = cfg_r.output_size,
        )

        # ── 3. Fusion layer
        self.fusion = SpatioTemporalFusion(
            spatial_dim  = cfg_g.out_channels,
            temporal_dim = cfg_r.output_size,
            num_heads    = cfg_g.num_heads,
            dropout      = cfg_g.dropout,
        )
        fusion_dim = cfg_g.out_channels + cfg_r.output_size

        # ── 4. Disease-specific output heads
        self.disease_heads = nn.ModuleDict({
            disease: DiseaseHead(
                input_dim  = fusion_dim,
                hidden_dim = 64,
                horizon    = self.config.forecast_horizon,
                dropout    = cfg_g.dropout,
            )
            for disease in DISEASES
        })

        # ── Internal storage for SHAP
        self._last_spatial_emb:  Optional[torch.Tensor] = None
        self._last_temporal_emb: Optional[torch.Tensor] = None
        self._last_attn_weights: Optional[torch.Tensor] = None

        n_params = sum(p.numel() for p in self.parameters())
        logger.info(
            f"HydroCastModel initialised | "
            f"node_feat={node_feat_dim} | time_feat={time_feat_dim} | "
            f"params={n_params:,}"
        )

    def forward(
        self,
        graph_data,
        time_series:    torch.Tensor,
        district_indices: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Forward pass.

        Parameters
        ----------
        graph_data      : PyG Data object (x, edge_index, edge_attr)
        time_series     : (batch, seq_len, time_feat_dim)
        district_indices: (batch,) integer indices into graph nodes (optional)

        Returns
        -------
        dict with keys:
            'cholera'         : (batch, 4) outbreak probs
            'typhoid'         : (batch, 4) outbreak probs
            'add'             : (batch, 4) outbreak probs
            'spatial_emb'     : (batch, spatial_dim)
            'temporal_emb'    : (batch, temporal_dim)
            'fused_emb'       : (batch, fusion_dim)
            'attention_weights': (num_edges, num_heads) GATv2 attention
        """
        B = time_series.shape[0]

        # ── 1. Spatial branch
        spatial_all, attn_weights = self.spatial_encoder.get_attention_weights(
            x          = graph_data.x,
            edge_index = graph_data.edge_index,
            edge_attr  = graph_data.edge_attr,
        )
        # spatial_all: (num_nodes=36, spatial_dim)

        # Select the relevant node embeddings for each sample in the batch
        if district_indices is not None:
            spatial_emb = spatial_all[district_indices]  # (B, spatial_dim)
        else:
            # Default: use mean over all district embeddings
            spatial_emb = spatial_all.mean(dim=0, keepdim=True).expand(B, -1)

        # ── 2. Temporal branch
        _, temporal_emb = self.temporal_encoder(time_series)
        # temporal_emb: (B, temporal_dim)

        # ── 3. Fusion
        fused = self.fusion(spatial_emb, temporal_emb)  # (B, fusion_dim)

        # ── 4. Disease heads
        predictions = {
            disease.lower(): head(fused)
            for disease, head in self.disease_heads.items()
        }

        # ── Store for SHAP
        self._last_spatial_emb  = spatial_emb.detach()
        self._last_temporal_emb = temporal_emb.detach()
        self._last_attn_weights = attn_weights

        return {
            **predictions,
            "spatial_emb":      spatial_emb,
            "temporal_emb":     temporal_emb,
            "fused_emb":        fused,
            "attention_weights":attn_weights,
        }

    def compute_loss(
        self,
        predictions:      dict,
        targets:          dict,
        seir_regularizer: Optional[SEIRRegularizer] = None,
        district:         Optional[str] = None,
    ) -> dict:
        """
        Compute combined multi-task + SEIR physics loss.

        Loss = Σ_disease BCE(predictions[d], targets[d])
             + seir_weight × SEIR_loss

        Parameters
        ----------
        predictions      : output of forward() — {disease: (batch, 4) tensor}
        targets          : {disease: (batch, 4) binary labels}
        seir_regularizer : optional SEIR regulariser
        district         : district name for SEIR lookup

        Returns
        -------
        dict  {total_loss, task_loss, seir_loss}
        """
        sample_pred = next(
            predictions[key] for key in [d.lower() for d in DISEASES]
            if predictions.get(key) is not None
        )
        task_loss = torch.zeros((), device=sample_pred.device, dtype=sample_pred.dtype)
        for disease in DISEASES:
            key  = disease.lower()
            pred = predictions.get(key)
            tgt  = targets.get(key)
            if pred is not None and tgt is not None:
                loss = F.binary_cross_entropy(pred, tgt.to(device=pred.device, dtype=pred.dtype))
                task_loss = task_loss + loss

        # SEIR physics regularisation
        seir_loss = torch.zeros((), device=sample_pred.device, dtype=sample_pred.dtype)
        if seir_regularizer is not None and district is not None:
            for disease in DISEASES:
                key  = disease.lower()
                pred = predictions.get(key)
                if pred is not None:
                    seir_l = seir_regularizer.get_regularization_loss(
                        pred, district
                    )
                    seir_loss = seir_loss + seir_l

        total_loss = task_loss + seir_loss

        return {
            "total_loss": total_loss,
            "task_loss":  task_loss,
            "seir_loss":  seir_loss,
        }

    def predict_district(
        self,
        district_name: str,
        graph_data,
        time_series:   torch.Tensor,
    ) -> dict:
        """
        Convenience method: run inference for a single named district.

        Returns
        -------
        dict  {district, predictions, risk_levels, summary}
        """
        from src.config import RISK_THRESHOLDS, RISK_COLORS

        dist_idx = DISTRICT_TO_IDX.get(district_name)
        if dist_idx is None:
            raise ValueError(f"District '{district_name}' not in config.")

        idx_tensor = torch.tensor(
            [dist_idx],
            dtype=torch.long,
            device=time_series.device,
        )

        self.eval()
        with torch.no_grad():
            out = self.forward(graph_data, time_series, district_indices=idx_tensor)

        result: dict = {"district": district_name, "predictions": {}}

        for disease in DISEASES:
            key   = disease.lower()
            probs = out[key][0].cpu().numpy()   # (horizon,)

            # Take max probability across horizon as the alert score
            max_prob = float(probs.max())

            risk_level = "low"
            for level in ["critical", "high", "medium"]:
                if max_prob >= RISK_THRESHOLDS[level]:
                    risk_level = level
                    break

            result["predictions"][disease] = {
                "week_probs": probs.tolist(),
                "max_prob":   max_prob,
                "risk_level": risk_level,
                "color":      RISK_COLORS[risk_level],
            }

        # Summary text
        top_disease = max(
            result["predictions"],
            key=lambda d: result["predictions"][d]["max_prob"],
        )
        top_prob  = result["predictions"][top_disease]["max_prob"]
        top_level = result["predictions"][top_disease]["risk_level"]

        result["summary"] = (
            f"{district_name}: {top_level.upper()} risk — "
            f"{top_disease} outbreak probability {top_prob:.0%} "
            f"over 4-week horizon."
        )

        return result


# ══════════════════════════════════════════════════════════════════
# MODEL FACTORY
# ══════════════════════════════════════════════════════════════════

def build_hydrocast_model(
    node_feat_dim: int = 6,
    time_feat_dim: int = 32,
    config             = None,
) -> HydroCastModel:
    """
    Factory function: build and return a HydroCastModel instance.

    Parameters
    ----------
    node_feat_dim : static node feature dimension
    time_feat_dim : time-varying feature dimension per timestep
    config        : ModelConfig (uses CONFIG singleton if None)

    Returns
    -------
    HydroCastModel  (on appropriate device from config)
    """
    model  = HydroCastModel(
        node_feat_dim = node_feat_dim,
        time_feat_dim = time_feat_dim,
        config        = config or CONFIG,
    )
    device = (config or CONFIG).training.device
    model  = model.to(device)
    logger.info(f"Model moved to device: {device}")
    return model


# ══════════════════════════════════════════════════════════════════
# MAIN — Smoke test
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    from src.data_pipeline.data_loader  import generate_synthetic_data
    from src.data_pipeline.graph_builder import DistrictGraphBuilder

    # ── Synthetic data
    df      = generate_synthetic_data(n_weeks=20)
    builder = DistrictGraphBuilder(MAHARASHTRA_DISTRICTS)
    graph   = builder.build_pyg_data(static_df=df)

    node_feat_dim = graph.x.shape[1]   # 6 static features
    time_feat_dim = 20                  # synthetic time features

    # ── Build model
    model = build_hydrocast_model(
        node_feat_dim = node_feat_dim,
        time_feat_dim = time_feat_dim,
    )

    # ── Fake time-series batch
    batch_size = 8
    seq_len    = 52
    x_ts = torch.randn(batch_size, seq_len, time_feat_dim)

    # ── Forward pass
    out = model(graph, x_ts)

    print("\n── Forward pass outputs ──")
    for key, val in out.items():
        if isinstance(val, torch.Tensor):
            print(f"  {key:20s}: {tuple(val.shape)}")

    # ── Compute loss
    targets = {
        d.lower(): torch.randint(0, 2, (batch_size, 4)).float()
        for d in DISEASES
    }
    losses = model.compute_loss(out, targets)
    print(f"\n── Losses ──")
    for k, v in losses.items():
        print(f"  {k:15s}: {v.item():.4f}")

    # ── District prediction
    result = model.predict_district(
        district_name = "Raigad",
        graph_data    = graph,
        time_series   = x_ts[:1],
    )
    print(f"\n── District prediction ──")
    print(f"  {result['summary']}")
    for disease, preds in result["predictions"].items():
        print(f"  {disease}: {preds['risk_level'].upper()} "
              f"({preds['max_prob']:.2%}) | "
              f"weeks: {[f'{p:.2f}' for p in preds['week_probs']]}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n── Total parameters : {total_params:,}")
    print("\n✅ hydrocast_model.py smoke test passed.")
