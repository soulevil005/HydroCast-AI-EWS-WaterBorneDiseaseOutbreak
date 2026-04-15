"""
HydroCast — GATv2 Spatial Encoder
Graph Attention Network v2 for learning district-level
spatial disease propagation patterns.

Reference: Brody et al. 2022
"How Attentive are Graph Attention Networks?"
https://arxiv.org/abs/2105.14491
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, BatchNorm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import CONFIG, NUM_DISTRICTS, RESULTS_DIR

logger = logging.getLogger("hydrocast.gatv2_encoder")


class GATv2Encoder(nn.Module):
    """
    Multi-layer GATv2 spatial encoder.

    Takes static + dynamic node features on the Maharashtra district
    graph and outputs rich spatial embeddings for each district.

    Architecture (per layer):
        GATv2Conv → BatchNorm → ELU → Dropout
    With residual connections when dimensions match.

    Parameters
    ----------
    in_channels     : input node feature dimension
    hidden_channels : hidden layer dimension (from config: 64)
    out_channels    : output embedding dimension (from config: 64)
    num_heads       : number of attention heads (from config: 4)
    num_layers      : number of GATv2 layers (from config: 2)
    dropout         : dropout probability (from config: 0.2)
    edge_dim        : edge feature dimension (from config: 3)
    """

    def __init__(
        self,
        in_channels:     int   = 6,
        hidden_channels: int   = 64,
        out_channels:    int   = 64,
        num_heads:       int   = 4,
        num_layers:      int   = 2,
        dropout:         float = 0.2,
        edge_dim:        int   = 3,
    ) -> None:
        super().__init__()

        self.in_channels     = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels    = out_channels
        self.num_heads       = num_heads
        self.num_layers      = num_layers
        self.dropout         = dropout

        self.convs  = nn.ModuleList()
        self.bns    = nn.ModuleList()
        self.lins   = nn.ModuleList()   # residual projections

        # ── Build layers
        for i in range(num_layers):
            in_ch  = in_channels if i == 0 else hidden_channels
            out_ch = hidden_channels

            self.convs.append(
                GATv2Conv(
                    in_channels  = in_ch,
                    out_channels = out_ch // num_heads,
                    heads        = num_heads,
                    dropout      = dropout,
                    edge_dim     = edge_dim,
                    concat       = True,         # concat heads → out_ch
                    add_self_loops=True,
                )
            )
            self.bns.append(BatchNorm(out_ch))

            # Residual projection (only needed when dims differ)
            if in_ch != out_ch:
                self.lins.append(nn.Linear(in_ch, out_ch, bias=False))
            else:
                self.lins.append(nn.Identity())

        # ── Final projection to out_channels
        self.projection = nn.Linear(hidden_channels, out_channels)

        # ── Store last attention weights for explainability
        self._last_attention_weights: Optional[torch.Tensor] = None

        logger.info(
            f"GATv2Encoder: {num_layers} layers | "
            f"in={in_channels} → hidden={hidden_channels} → out={out_channels} | "
            f"heads={num_heads} | edge_dim={edge_dim}"
        )

    def forward(
        self,
        x:          torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr:  Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through GATv2 layers.

        Parameters
        ----------
        x          : node features  (num_nodes, in_channels)
        edge_index : COO edges      (2, num_edges)
        edge_attr  : edge features  (num_edges, edge_dim) or None

        Returns
        -------
        torch.Tensor  shape (num_nodes, out_channels)
            Spatial embeddings for each district node.
        """
        h = x
        for i, (conv, bn, lin) in enumerate(
            zip(self.convs, self.bns, self.lins)
        ):
            h_in = h
            h    = conv(h, edge_index, edge_attr=edge_attr)
            h    = bn(h)
            h    = F.elu(h)
            h    = F.dropout(h, p=self.dropout, training=self.training)

            # Residual connection
            residual = lin(h_in)
            if h.shape == residual.shape:
                h = h + residual

        # Final projection
        out = self.projection(h)
        return out

    def get_attention_weights(
        self,
        x:          torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr:  Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return spatial embeddings AND attention weights from the last layer.

        Attention weights indicate how much each district (node) attends
        to each of its neighbours — useful for SHAP spatial attribution.

        Returns
        -------
        (embeddings, attention_weights)
            embeddings       : (num_nodes, out_channels)
            attention_weights: (num_edges, num_heads) — last layer only
        """
        h = x
        attn_weights = None

        for i, (conv, bn, lin) in enumerate(
            zip(self.convs, self.bns, self.lins)
        ):
            h_in = h
            # GATv2Conv returns (out, (edge_index, alpha)) when return_attention_weights=True
            h, (_, alpha) = conv(
                h, edge_index,
                edge_attr=edge_attr,
                return_attention_weights=True,
            )
            h           = bn(h)
            h           = F.elu(h)
            h           = F.dropout(h, p=self.dropout, training=self.training)
            residual    = lin(h_in)
            if h.shape == residual.shape:
                h = h + residual
            attn_weights = alpha   # keep last layer's weights

        self._last_attention_weights = attn_weights
        return self.projection(h), attn_weights


class SpatialAttentionMap:
    """
    Stores and visualises GATv2 attention weights as a district heatmap.
    Shows which neighbouring districts influence each focal district most.
    """

    def __init__(self, districts: list[str], adjacency: dict) -> None:
        self.districts = districts
        self.adjacency = adjacency
        self.d2i       = {d: i for i, d in enumerate(districts)}

    def plot_attention(
        self,
        edge_index:      torch.Tensor,
        attn_weights:    torch.Tensor,
        focal_district:  str,
        save_path:       Optional[Path] = None,
    ) -> None:
        """
        Plot attention weights for a focal district as a bar chart.

        Parameters
        ----------
        edge_index     : (2, num_edges) COO edge tensor
        attn_weights   : (num_edges, num_heads) attention weights
        focal_district : district to show attention FROM
        save_path      : save PNG here (optional)
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not installed. Skipping attention plot.")
            return

        focal_idx = self.d2i.get(focal_district)
        if focal_idx is None:
            logger.error(f"District '{focal_district}' not found.")
            return

        src, dst = edge_index
        mask     = src == focal_idx
        nb_idxs  = dst[mask].cpu().numpy()
        weights  = attn_weights[mask].mean(dim=-1).detach().cpu().numpy()

        nb_names = [self.districts[i] for i in nb_idxs]

        # Sort by weight descending
        order    = np.argsort(weights)[::-1]
        nb_names = [nb_names[i] for i in order]
        weights  = weights[order]

        fig, ax = plt.subplots(figsize=(9, 5), facecolor="#070b14")
        ax.set_facecolor("#070b14")
        bars = ax.barh(nb_names, weights, color="#4d9fff", edgecolor="#2a5caa")
        ax.set_xlabel("Attention weight", color="white")
        ax.set_title(
            f"GATv2 Attention from '{focal_district}' to neighbours",
            color="white", fontsize=12,
        )
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight",
                        facecolor="#070b14")
            logger.info(f"Attention plot saved: {save_path}")
        else:
            plt.show()
        plt.close()


# ══════════════════════════════════════════════════════════════════
# MAIN — Smoke test
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    from src.data_pipeline.data_loader import generate_synthetic_data
    from src.data_pipeline.graph_builder import DistrictGraphBuilder, ADJACENCY_LIST
    from src.config import MAHARASHTRA_DISTRICTS

    df      = generate_synthetic_data(n_weeks=10)
    builder = DistrictGraphBuilder(MAHARASHTRA_DISTRICTS)
    pyg     = builder.build_pyg_data(static_df=df)

    model = GATv2Encoder(
        in_channels     = pyg.x.shape[1],
        hidden_channels = 64,
        out_channels    = 64,
        num_heads       = 4,
        num_layers      = 2,
        dropout         = 0.2,
        edge_dim        = 3,
    )
    model.eval()

    with torch.no_grad():
        emb = model(pyg.x, pyg.edge_index, pyg.edge_attr)
        emb_attn, attn = model.get_attention_weights(
            pyg.x, pyg.edge_index, pyg.edge_attr
        )

    print(f"\n── Embedding shape    : {emb.shape}")
    print(f"── Attn weight shape  : {attn.shape}")
    print(f"── Num params         : {sum(p.numel() for p in model.parameters()):,}")

    attn_map = SpatialAttentionMap(MAHARASHTRA_DISTRICTS, ADJACENCY_LIST)
    attn_map.plot_attention(
        edge_index=pyg.edge_index,
        attn_weights=attn,
        focal_district="Raigad",
        save_path=RESULTS_DIR / "gatv2_attention_raigad.png",
    )
    print("\n✅ gatv2_encoder.py smoke test passed.")
