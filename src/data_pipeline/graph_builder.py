"""
HydroCast — District Graph Builder
Constructs a PyTorch Geometric graph of 36 Maharashtra districts.
Nodes = districts, Edges = geographic adjacency + river basin connectivity.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import (
    MAHARASHTRA_DISTRICTS, DISTRICT_CENTROIDS,
    DISTRICT_TO_IDX, NUM_DISTRICTS,
    RESULTS_DIR,
)

logger = logging.getLogger("hydrocast.graph_builder")


# ══════════════════════════════════════════════════════════════════
# GEOGRAPHIC ADJACENCY (real Maharashtra district neighbours)
# ══════════════════════════════════════════════════════════════════

ADJACENCY_LIST: dict[str, list[str]] = {
    "Mumbai City":      ["Mumbai Suburban", "Thane"],
    "Mumbai Suburban":  ["Mumbai City", "Thane", "Raigad", "Palghar"],
    "Thane":            ["Mumbai City", "Mumbai Suburban", "Raigad", "Palghar", "Nashik"],
    "Palghar":          ["Mumbai Suburban", "Thane", "Nashik", "Dhule"],
    "Raigad":           ["Mumbai Suburban", "Thane", "Pune", "Satara", "Ratnagiri"],
    "Ratnagiri":        ["Raigad", "Satara", "Kolhapur", "Sindhudurg"],
    "Sindhudurg":       ["Ratnagiri", "Kolhapur"],
    "Nashik":           ["Palghar", "Thane", "Ahmednagar", "Dhule", "Nandurbar", "Aurangabad"],
    "Dhule":            ["Palghar", "Nashik", "Nandurbar", "Jalgaon"],
    "Nandurbar":        ["Dhule", "Nashik", "Jalgaon"],
    "Jalgaon":          ["Dhule", "Nandurbar", "Nashik", "Ahmednagar", "Aurangabad", "Buldhana"],
    "Ahmednagar":       ["Nashik", "Jalgaon", "Pune", "Beed", "Osmanabad", "Aurangabad"],
    "Pune":             ["Raigad", "Ahmednagar", "Satara", "Solapur", "Beed"],
    "Satara":           ["Raigad", "Pune", "Solapur", "Sangli", "Kolhapur", "Ratnagiri"],
    "Sangli":           ["Satara", "Solapur", "Kolhapur"],
    "Solapur":          ["Pune", "Satara", "Sangli", "Osmanabad", "Latur"],
    "Kolhapur":         ["Satara", "Sangli", "Ratnagiri", "Sindhudurg"],
    "Aurangabad":       ["Nashik", "Jalgaon", "Ahmednagar", "Beed", "Jalna", "Buldhana"],
    "Jalna":            ["Aurangabad", "Beed", "Parbhani", "Buldhana"],
    "Beed":             ["Ahmednagar", "Pune", "Solapur", "Osmanabad", "Latur",
                         "Nanded", "Parbhani", "Jalna", "Aurangabad"],
    "Osmanabad":        ["Ahmednagar", "Solapur", "Latur", "Beed"],
    "Latur":            ["Solapur", "Osmanabad", "Beed", "Nanded"],
    "Nanded":           ["Beed", "Latur", "Parbhani", "Hingoli", "Yavatmal"],
    "Parbhani":         ["Jalna", "Beed", "Nanded", "Hingoli"],
    "Hingoli":          ["Buldhana", "Washim", "Nanded", "Parbhani"],
    "Buldhana":         ["Jalgaon", "Aurangabad", "Jalna", "Hingoli", "Washim", "Akola"],
    "Akola":            ["Buldhana", "Washim", "Amravati"],
    "Washim":           ["Buldhana", "Hingoli", "Akola", "Yavatmal"],
    "Amravati":         ["Akola", "Washim", "Yavatmal", "Wardha"],
    "Yavatmal":         ["Washim", "Nanded", "Amravati", "Wardha", "Chandrapur"],
    "Wardha":           ["Amravati", "Yavatmal", "Nagpur", "Chandrapur"],
    "Nagpur":           ["Wardha", "Bhandara", "Chandrapur"],
    "Bhandara":         ["Nagpur", "Gondia", "Chandrapur"],
    "Gondia":           ["Bhandara", "Chandrapur"],
    "Chandrapur":       ["Yavatmal", "Wardha", "Nagpur", "Bhandara", "Gondia", "Gadchiroli"],
    "Gadchiroli":       ["Chandrapur", "Gondia"],
}

# ── River basin connectivity (districts sharing the same river basin
#    get an extra edge with a higher weight)
RIVER_CONNECTIVITY: dict[str, list[str]] = {
    # Godavari basin
    "Nashik":      ["Aurangabad", "Jalna", "Parbhani", "Nanded"],
    "Aurangabad":  ["Nashik", "Jalna", "Parbhani"],
    "Jalna":       ["Nashik", "Aurangabad", "Parbhani", "Nanded"],
    "Parbhani":    ["Nashik", "Aurangabad", "Jalna", "Nanded"],
    "Nanded":      ["Nashik", "Jalna", "Parbhani"],
    # Krishna basin
    "Satara":      ["Sangli", "Solapur", "Kolhapur"],
    "Sangli":      ["Satara", "Solapur", "Kolhapur"],
    "Solapur":     ["Satara", "Sangli"],
    "Kolhapur":    ["Satara", "Sangli"],
    # Tapi basin
    "Dhule":       ["Jalgaon", "Nandurbar"],
    "Jalgaon":     ["Dhule", "Nandurbar"],
    "Nandurbar":   ["Dhule", "Jalgaon"],
}

# ── Maharashtra geographic regions (for graph visualisation colouring)
DISTRICT_REGIONS: dict[str, str] = {
    "Mumbai City": "Konkan",     "Mumbai Suburban": "Konkan",
    "Thane": "Konkan",           "Palghar": "Konkan",
    "Raigad": "Konkan",          "Ratnagiri": "Konkan",
    "Sindhudurg": "Konkan",
    "Nashik": "Western MH",      "Dhule": "Western MH",
    "Nandurbar": "Western MH",   "Jalgaon": "Western MH",
    "Ahmednagar": "Western MH",  "Pune": "Western MH",
    "Satara": "Western MH",      "Sangli": "Western MH",
    "Solapur": "Western MH",     "Kolhapur": "Western MH",
    "Aurangabad": "Marathwada",  "Jalna": "Marathwada",
    "Beed": "Marathwada",        "Osmanabad": "Marathwada",
    "Latur": "Marathwada",       "Nanded": "Marathwada",
    "Parbhani": "Marathwada",    "Hingoli": "Marathwada",
    "Buldhana": "Vidarbha",      "Akola": "Vidarbha",
    "Washim": "Vidarbha",        "Amravati": "Vidarbha",
    "Yavatmal": "Vidarbha",      "Wardha": "Vidarbha",
    "Nagpur": "Vidarbha",        "Bhandara": "Vidarbha",
    "Gondia": "Vidarbha",        "Chandrapur": "Vidarbha",
    "Gadchiroli": "Vidarbha",
}

REGION_COLORS: dict[str, str] = {
    "Konkan":     "#00e5a0",
    "Western MH": "#4d9fff",
    "Marathwada": "#ffb84d",
    "Vidarbha":   "#b47fff",
}


# ══════════════════════════════════════════════════════════════════
# HAVERSINE DISTANCE
# ══════════════════════════════════════════════════════════════════

def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute great-circle distance between two lat/lon points in km."""
    R = 6371.0
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = (math.sin(d_lat / 2) ** 2
         + math.cos(math.radians(lat1))
         * math.cos(math.radians(lat2))
         * math.sin(d_lon / 2) ** 2)
    return 2 * R * math.asin(math.sqrt(a))


# ══════════════════════════════════════════════════════════════════
# DISTRICT GRAPH BUILDER
# ══════════════════════════════════════════════════════════════════

class DistrictGraphBuilder:
    """
    Builds a PyTorch Geometric graph for 36 Maharashtra districts.

    Node features  : static WASH + demographic indicators
    Edge index     : geographic adjacency (undirected + self-loops)
    Edge attributes: [geographic_adj, river_basin, distance_normalised]
    """

    def __init__(self, districts: list[str] = MAHARASHTRA_DISTRICTS) -> None:
        self.districts     = districts
        self.n             = len(districts)
        self.d2i           = {d: i for i, d in enumerate(districts)}
        self._edge_pairs   : list[tuple[int, int]] = []
        self._edge_attrs   : list[list[float]] = []
        logger.info(f"DistrictGraphBuilder: {self.n} districts")

    # ──────────────────────────────────────────────────────────────
    # Build edge index (COO format)
    # ──────────────────────────────────────────────────────────────

    def build_edge_index(self) -> torch.Tensor:
        """
        Build COO-format edge_index tensor (2, num_edges).
        Includes self-loops and makes graph undirected.

        Returns
        -------
        torch.LongTensor  shape (2, E)
        """
        pairs: set[tuple[int, int]] = set()

        # ── Geographic adjacency edges
        for district, neighbours in ADJACENCY_LIST.items():
            if district not in self.d2i:
                continue
            src = self.d2i[district]
            for nb in neighbours:
                if nb not in self.d2i:
                    continue
                dst = self.d2i[nb]
                pairs.add((src, dst))
                pairs.add((dst, src))   # undirected

        # ── Self-loops
        for i in range(self.n):
            pairs.add((i, i))

        self._edge_pairs = sorted(pairs)
        edge_index = torch.tensor(self._edge_pairs, dtype=torch.long).t()
        logger.info(f"Edge index built: {edge_index.shape[1]} edges "
                    f"({len(pairs)} pairs incl. self-loops)")
        return edge_index

    # ──────────────────────────────────────────────────────────────
    # Build edge attributes
    # ──────────────────────────────────────────────────────────────

    def build_edge_attr(self) -> torch.Tensor:
        """
        Build edge feature matrix (num_edges, 3).

        Features per edge:
            [0] geographic_adjacent  (1.0 or 0.0)
            [1] shares_river_basin   (1.0 or 0.0)
            [2] haversine_distance_normalised  (0.0–1.0)

        Returns
        -------
        torch.FloatTensor  shape (E, 3)
        """
        if not self._edge_pairs:
            self.build_edge_index()

        # Precompute all pairwise distances for normalisation
        all_dists = {}
        for d1 in self.districts:
            lat1, lon1 = DISTRICT_CENTROIDS.get(d1, (19.5, 76.5))
            for d2 in self.districts:
                lat2, lon2 = DISTRICT_CENTROIDS.get(d2, (19.5, 76.5))
                all_dists[(d1, d2)] = _haversine_km(lat1, lon1, lat2, lon2)

        max_dist = max(all_dists.values()) if all_dists else 1.0

        attrs = []
        dist_list = {i: d for d, i in self.d2i.items()}

        for src_idx, dst_idx in self._edge_pairs:
            src_name = dist_list[src_idx]
            dst_name = dist_list[dst_idx]

            # Feature 0: geographic adjacency
            geo_adj = 1.0 if (dst_name in ADJACENCY_LIST.get(src_name, [])
                              or src_idx == dst_idx) else 0.0

            # Feature 1: river basin connectivity
            river = 1.0 if (dst_name in RIVER_CONNECTIVITY.get(src_name, [])
                             or src_name in RIVER_CONNECTIVITY.get(dst_name, [])
                             or src_idx == dst_idx) else 0.0

            # Feature 2: normalised distance (self-loop = 0)
            raw_dist = all_dists.get((src_name, dst_name), 0.0)
            dist_norm = 1.0 - (raw_dist / max_dist)   # closer = higher score

            attrs.append([geo_adj, river, dist_norm])

        edge_attr = torch.tensor(attrs, dtype=torch.float)
        logger.info(f"Edge attr built: {edge_attr.shape}")
        return edge_attr

    # ──────────────────────────────────────────────────────────────
    # Build node feature matrix
    # ──────────────────────────────────────────────────────────────

    def build_node_features(
        self,
        static_df: pd.DataFrame,
    ) -> torch.Tensor:
        """
        Build node feature matrix (num_nodes, num_features).

        Uses static WASH and demographic columns from NFHS-5 data.

        Parameters
        ----------
        static_df : DataFrame indexed by district or with a 'district' column

        Returns
        -------
        torch.FloatTensor  shape (36, num_static_features)
        """
        static_cols = [
            "sanitation_coverage_pct", "od_index", "water_access_pct",
            "population_density", "urban_pct", "wash_index",
        ]

        if isinstance(static_df.index, pd.MultiIndex):
            df_s = static_df.reset_index()
            df_s = df_s.groupby("district")[static_cols].mean()
        elif static_df.index.name == "district":
            df_s = static_df
        else:
            df_s = static_df.set_index("district")

        # Fill missing districts with median values
        node_feats = []
        for district in self.districts:
            if district in df_s.index:
                row = df_s.loc[district, static_cols].values.astype(np.float32)
            else:
                row = df_s[static_cols].median().values.astype(np.float32)
            node_feats.append(row)

        x = torch.tensor(np.array(node_feats), dtype=torch.float)
        logger.info(f"Node features built: {x.shape}")
        return x

    # ──────────────────────────────────────────────────────────────
    # Build full PyG Data object
    # ──────────────────────────────────────────────────────────────

    def build_pyg_data(
        self,
        static_df: pd.DataFrame,
        dynamic_df: Optional[pd.DataFrame] = None,
    ) -> Data:
        """
        Build a PyTorch Geometric Data object.

        Parameters
        ----------
        static_df  : DataFrame with static district features
        dynamic_df : (optional) DataFrame with time-varying features;
                     if provided, adds temporal node features

        Returns
        -------
        torch_geometric.data.Data
        """
        edge_index = self.build_edge_index()
        edge_attr  = self.build_edge_attr()
        x          = self.build_node_features(static_df)

        data = Data(
            x          = x,
            edge_index = edge_index,
            edge_attr  = edge_attr,
            num_nodes  = self.n,
        )

        data.district_names = self.districts

        logger.info(f"PyG Data object: {data}")
        return data

    # ──────────────────────────────────────────────────────────────
    # Visualise graph
    # ──────────────────────────────────────────────────────────────

    def visualize_graph(
        self,
        save_path: Optional[Path] = None,
    ) -> None:
        """
        Plot the Maharashtra district graph using networkx + matplotlib.
        Nodes are coloured by geographic region (Konkan, Western MH,
        Marathwada, Vidarbha). Edge thickness ∝ river basin connection.

        Parameters
        ----------
        save_path : Path to save PNG (optional)
        """
        try:
            import networkx as nx
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("networkx/matplotlib not installed. Skipping plot.")
            return

        G = nx.Graph()

        # Add nodes
        for district in self.districts:
            G.add_node(district, region=DISTRICT_REGIONS.get(district, "Unknown"))

        # Add edges from adjacency
        for district, neighbours in ADJACENCY_LIST.items():
            for nb in neighbours:
                if nb in self.d2i and not G.has_edge(district, nb):
                    river = nb in RIVER_CONNECTIVITY.get(district, [])
                    G.add_edge(district, nb, river=river)

        # Node positions = real lat/lon
        pos = {
            d: (DISTRICT_CENTROIDS[d][1], DISTRICT_CENTROIDS[d][0])
            for d in self.districts
            if d in DISTRICT_CENTROIDS
        }

        node_colors = [
            REGION_COLORS.get(DISTRICT_REGIONS.get(d, ""), "#888888")
            for d in G.nodes()
        ]
        edge_colors = [
            "#4d9fff" if G[u][v].get("river") else "#444444"
            for u, v in G.edges()
        ]
        edge_widths = [
            2.5 if G[u][v].get("river") else 0.8
            for u, v in G.edges()
        ]

        fig, ax = plt.subplots(figsize=(14, 10), facecolor="#070b14")
        ax.set_facecolor("#070b14")

        nx.draw_networkx_nodes(
            G, pos, node_color=node_colors,
            node_size=180, alpha=0.9, ax=ax,
        )
        nx.draw_networkx_edges(
            G, pos, edge_color=edge_colors,
            width=edge_widths, alpha=0.6, ax=ax,
        )
        nx.draw_networkx_labels(
            G, pos, font_size=5, font_color="white", ax=ax,
        )

        # Legend
        for region, color in REGION_COLORS.items():
            ax.scatter([], [], c=color, label=region, s=80)
        ax.scatter([], [], c="#4d9fff", label="River basin edge", marker="_", s=80)
        ax.legend(loc="lower left", fontsize=8,
                  facecolor="#0d1420", labelcolor="white",
                  edgecolor="#333")

        ax.set_title("HydroCast — Maharashtra District Graph",
                     color="white", fontsize=13, pad=12)
        ax.axis("off")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight",
                        facecolor="#070b14")
            logger.info(f"Graph saved to: {save_path}")
        else:
            plt.show()
        plt.close()


# ══════════════════════════════════════════════════════════════════
# MAIN — Smoke test
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from src.data_pipeline.data_loader import generate_synthetic_data

    df = generate_synthetic_data(n_weeks=52)

    builder = DistrictGraphBuilder(MAHARASHTRA_DISTRICTS)
    data    = builder.build_pyg_data(static_df=df)

    print("\n── PyG Data ──")
    print(data)
    print(f"\n── Num nodes     : {data.num_nodes}")
    print(f"── Num edges     : {data.num_edges}")
    print(f"── Node feat dim : {data.x.shape[1]}")
    print(f"── Edge feat dim : {data.edge_attr.shape[1]}")

    save = RESULTS_DIR / "district_graph.png"
    builder.visualize_graph(save_path=save)
    print(f"\n── Graph plot saved to: {save}")
    print("\n✅ graph_builder.py smoke test passed.")
