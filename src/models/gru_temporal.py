"""
HydroCast — GRU Temporal Encoder
Bidirectional GRU for learning disease time-series patterns.
Also implements Bahdanau-style temporal attention — weights
reveal which weeks the model focuses on most (used in SHAP).
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
from src.config import CONFIG

logger = logging.getLogger("hydrocast.gru_temporal")


class TemporalAttention(nn.Module):
    """
    Bahdanau-style additive attention over GRU output sequence.

    Computes a context vector as a weighted sum of GRU hidden states.
    The attention weights show which weeks the model focuses on —
    monsoon weeks typically receive the highest weights for waterborne
    disease prediction.

    Parameters
    ----------
    hidden_size : dimension of each GRU output (bidirectional → ×2)
    """

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.W_query   = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_key     = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v         = nn.Linear(hidden_size, 1, bias=False)
        self.tanh      = nn.Tanh()

    def forward(
        self,
        query:  torch.Tensor,
        keys:   torch.Tensor,
        mask:   Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        query : (batch, hidden_size)   — final hidden state of GRU
        keys  : (batch, seq_len, hidden_size) — all GRU outputs
        mask  : (batch, seq_len) bool mask for padding (optional)

        Returns
        -------
        context       : (batch, hidden_size)   weighted context vector
        attention_weights : (batch, seq_len)   normalised attention scores
        """
        # query: (batch, hidden) → (batch, 1, hidden)
        q_expanded = self.W_query(query).unsqueeze(1)  # (B, 1, H)
        k_proj     = self.W_key(keys)                  # (B, T, H)

        energy = self.v(self.tanh(q_expanded + k_proj)).squeeze(-1)  # (B, T)

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-inf"))

        attn_weights = F.softmax(energy, dim=-1)         # (B, T)

        context = torch.bmm(
            attn_weights.unsqueeze(1), keys
        ).squeeze(1)                                      # (B, H)

        return context, attn_weights


class GRUTemporalEncoder(nn.Module):
    """
    Multi-layer Bidirectional GRU temporal encoder for disease
    time-series data.

    Architecture:
        LayerNorm → Bi-GRU (multi-layer) → TemporalAttention → Linear

    Parameters
    ----------
    input_size    : number of time-varying features per timestep
    hidden_size   : GRU hidden dimension (from config: 128)
    num_layers    : number of stacked GRU layers (from config: 2)
    dropout       : dropout between GRU layers (from config: 0.2)
    bidirectional : if True, use bidirectional GRU (recommended)
    output_size   : final embedding size (from config: 64)
    """

    def __init__(
        self,
        input_size:    int   = 32,
        hidden_size:   int   = 128,
        num_layers:    int   = 2,
        dropout:       float = 0.2,
        bidirectional: bool  = True,
        output_size:   int   = 64,
    ) -> None:
        super().__init__()

        self.input_size    = input_size
        self.hidden_size   = hidden_size
        self.num_layers    = num_layers
        self.bidirectional = bidirectional
        self.output_size   = output_size
        self.num_directions= 2 if bidirectional else 1
        self.gru_out_size  = hidden_size * self.num_directions

        # ── Input normalisation
        self.input_norm = nn.LayerNorm(input_size)

        # ── Bidirectional multi-layer GRU
        self.gru = nn.GRU(
            input_size   = input_size,
            hidden_size  = hidden_size,
            num_layers   = num_layers,
            dropout      = dropout if num_layers > 1 else 0.0,
            bidirectional= bidirectional,
            batch_first  = True,
        )

        # ── Temporal attention
        self.attention = TemporalAttention(hidden_size=self.gru_out_size)

        # ── Dropout between GRU and projection
        self.dropout = nn.Dropout(dropout)

        # ── Project to output size
        self.projection = nn.Sequential(
            nn.Linear(self.gru_out_size, output_size),
            nn.LayerNorm(output_size),
            nn.ELU(),
        )

        logger.info(
            f"GRUTemporalEncoder: input={input_size} | "
            f"hidden={hidden_size}×{self.num_directions}={self.gru_out_size} | "
            f"layers={num_layers} | out={output_size}"
        )

    def forward(
        self,
        x:       torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Parameters
        ----------
        x       : (batch, seq_len, input_size) time-series input
        lengths : (batch,) actual sequence lengths for packing (optional)

        Returns
        -------
        output  : (batch, seq_len, gru_out_size)  all timestep outputs
        hidden  : (batch, output_size)             final temporal embedding
        """
        x = self.input_norm(x)

        if lengths is not None:
            # Pack for variable-length sequences
            x_packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False,
            )
            gru_out_packed, h_n = self.gru(x_packed)
            gru_out, _ = nn.utils.rnn.pad_packed_sequence(
                gru_out_packed, batch_first=True,
            )
        else:
            gru_out, h_n = self.gru(x)   # gru_out: (B, T, gru_out_size)

        # Final hidden: concat last layer forward + backward
        if self.bidirectional:
            h_fwd = h_n[-2]   # last layer forward  (B, hidden)
            h_bwd = h_n[-1]   # last layer backward (B, hidden)
            h_cat = torch.cat([h_fwd, h_bwd], dim=-1)  # (B, gru_out_size)
        else:
            h_cat = h_n[-1]

        # Temporal attention over all timesteps
        context, _ = self.attention(query=h_cat, keys=gru_out)

        hidden = self.projection(self.dropout(context))
        return gru_out, hidden

    def encode_sequence(
        self,
        x:       torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Convenience method returning only the temporal embedding.

        Returns
        -------
        torch.Tensor  (batch, output_size)
        """
        _, hidden = self.forward(x, lengths)
        return hidden

    def get_temporal_attention(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return temporal embedding AND attention weights.

        Attention weights show which weeks drove the prediction —
        monsoon weeks (Jun–Sep) typically receive high weights.

        Returns
        -------
        hidden          : (batch, output_size)
        attention_weights : (batch, seq_len)
        """
        x = self.input_norm(x)
        gru_out, h_n = self.gru(x)

        if self.bidirectional:
            h_cat = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        else:
            h_cat = h_n[-1]

        context, attn_weights = self.attention(query=h_cat, keys=gru_out)
        hidden = self.projection(self.dropout(context))

        return hidden, attn_weights


# ══════════════════════════════════════════════════════════════════
# MAIN — Smoke test
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    batch_size = 8
    seq_len    = 52    # 1 year
    input_size = 20    # number of time-varying features

    x = torch.randn(batch_size, seq_len, input_size)

    model = GRUTemporalEncoder(
        input_size    = input_size,
        hidden_size   = 128,
        num_layers    = 2,
        dropout       = 0.2,
        bidirectional = True,
        output_size   = 64,
    )
    model.eval()

    with torch.no_grad():
        out, hidden = model(x)
        hidden2, attn = model.get_temporal_attention(x)

    print(f"\n── GRU output shape   : {out.shape}")
    print(f"── Hidden emb shape   : {hidden.shape}")
    print(f"── Attn weights shape : {attn.shape}")
    print(f"── Num params         : {sum(p.numel() for p in model.parameters()):,}")
    print(f"── Attn sum (should=1): {attn[0].sum().item():.4f}")
    print("\n✅ gru_temporal.py smoke test passed.")
