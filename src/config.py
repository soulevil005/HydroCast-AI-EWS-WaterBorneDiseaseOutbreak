"""
HydroCast — Central Configuration
All paths, hyperparameters, and constants for the entire project.
Never use magic numbers anywhere else — import from here.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import torch

# ══════════════════════════════════════════════
# ROOT PATHS
# ══════════════════════════════════════════════

ROOT_DIR          = Path(__file__).resolve().parent.parent
SRC_DIR           = ROOT_DIR / "src"
DATA_DIR          = SRC_DIR / "data" if (SRC_DIR / "data").exists() else ROOT_DIR / "data"
DATA_RAW_DIR      = DATA_DIR / "raw"
DATA_PROCESSED_DIR= DATA_DIR / "processed"
DATA_GEOJSON_DIR  = DATA_DIR / "geojson"
MODELS_DIR        = ROOT_DIR / "models"
RESULTS_DIR       = ROOT_DIR / "results"
LOGS_DIR          = ROOT_DIR / "logs"
SHAP_DIR          = RESULTS_DIR / "shap"
PLOTS_DIR         = RESULTS_DIR / "plots"

# Create directories if they don't exist
for _dir in [DATA_RAW_DIR, DATA_PROCESSED_DIR, DATA_GEOJSON_DIR,
             MODELS_DIR, RESULTS_DIR, LOGS_DIR, SHAP_DIR, PLOTS_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# ── Data file paths
IDSP_FILE    = DATA_RAW_DIR / "idsp_maharashtra_weekly.csv"
IMD_FILE     = DATA_RAW_DIR / "imd_rainfall_maharashtra.csv"
NFHS_FILE    = DATA_RAW_DIR / "nfhs5_maharashtra_district.csv"
GEOJSON_FILE = DATA_GEOJSON_DIR / "maharashtra_districts.geojson"
EPICLIM_FILE = DATA_PROCESSED_DIR / "epiclim_maharashtra_merged.csv"


# ══════════════════════════════════════════════
# DISTRICTS — All 36 Maharashtra Districts
# ══════════════════════════════════════════════

MAHARASHTRA_DISTRICTS: list[str] = [
    "Mumbai City", "Mumbai Suburban", "Thane", "Palghar", "Raigad",
    "Ratnagiri", "Sindhudurg", "Nashik", "Dhule", "Nandurbar",
    "Jalgaon", "Ahmednagar", "Pune", "Satara", "Sangli",
    "Solapur", "Kolhapur", "Aurangabad", "Jalna", "Beed",
    "Osmanabad", "Latur", "Nanded", "Parbhani", "Hingoli",
    "Buldhana", "Akola", "Washim", "Amravati", "Yavatmal",
    "Wardha", "Nagpur", "Bhandara", "Gondia", "Chandrapur",
    "Gadchiroli",
]

# Real approximate (lat, lon) centroids for all 36 districts
DISTRICT_CENTROIDS: dict[str, tuple[float, float]] = {
    "Mumbai City":      (18.940, 72.835),
    "Mumbai Suburban":  (19.120, 72.870),
    "Thane":            (19.218, 73.058),
    "Palghar":          (19.697, 72.765),
    "Raigad":           (18.519, 73.184),
    "Ratnagiri":        (17.000, 73.317),
    "Sindhudurg":       (16.350, 73.967),
    "Nashik":           (20.011, 73.790),
    "Dhule":            (20.899, 74.777),
    "Nandurbar":        (21.371, 74.248),
    "Jalgaon":          (21.004, 75.562),
    "Ahmednagar":       (19.095, 74.738),
    "Pune":             (18.520, 73.857),
    "Satara":           (17.687, 74.033),
    "Sangli":           (16.855, 74.567),
    "Solapur":          (17.687, 75.927),
    "Kolhapur":         (16.700, 74.233),
    "Aurangabad":       (19.877, 75.329),
    "Jalna":            (19.834, 75.883),
    "Beed":             (18.990, 75.756),
    "Osmanabad":        (18.178, 76.042),
    "Latur":            (18.407, 76.560),
    "Nanded":           (19.153, 77.308),
    "Parbhani":         (19.271, 76.774),
    "Hingoli":          (19.717, 77.149),
    "Buldhana":         (20.529, 76.184),
    "Akola":            (20.709, 77.007),
    "Washim":           (20.113, 77.133),
    "Amravati":         (20.932, 77.750),
    "Yavatmal":         (20.389, 78.133),
    "Wardha":           (20.745, 78.599),
    "Nagpur":           (21.149, 79.088),
    "Bhandara":         (21.167, 79.650),
    "Gondia":           (21.450, 80.200),
    "Chandrapur":       (19.952, 79.296),
    "Gadchiroli":       (19.750, 80.000),
}

# District to integer index mapping (for GNN node indexing)
DISTRICT_TO_IDX: dict[str, int] = {d: i for i, d in enumerate(MAHARASHTRA_DISTRICTS)}
IDX_TO_DISTRICT: dict[int, str] = {i: d for d, i in DISTRICT_TO_IDX.items()}
NUM_DISTRICTS: int = len(MAHARASHTRA_DISTRICTS)


# ══════════════════════════════════════════════
# DISEASES
# ══════════════════════════════════════════════

DISEASES: list[str] = ["Cholera", "Typhoid", "ADD"]

# Column names in the merged dataframe for each disease
DISEASE_CASE_COLS: dict[str, str] = {
    "Cholera": "cholera_cases",
    "Typhoid": "typhoid_cases",
    "ADD":     "add_cases",
}

DISEASE_LABEL_COLS: dict[str, str] = {
    "Cholera": "cholera_outbreak_label",
    "Typhoid": "typhoid_outbreak_label",
    "ADD":     "add_outbreak_label",
}

# Outbreak detection threshold (percentile of district case history)
DISEASE_THRESHOLDS: dict[str, int] = {
    "Cholera": 80,
    "Typhoid": 75,
    "ADD":     75,
}


# ══════════════════════════════════════════════
# ALERT RISK THRESHOLDS
# ══════════════════════════════════════════════

RISK_THRESHOLDS: dict[str, float] = {
    "critical": 0.80,
    "high":     0.60,
    "medium":   0.40,
    "low":      0.00,
}

RISK_COLORS: dict[str, str] = {
    "critical": "#ff3d5a",
    "high":     "#ffb84d",
    "medium":   "#4d9fff",
    "low":      "#00e5a0",
}


# ══════════════════════════════════════════════
# FEATURE DEFINITIONS
# ══════════════════════════════════════════════

TIME_FEATURES: list[str] = [
    "week_of_year", "month", "monsoon_flag",
    "season_sin", "season_cos",
]

CLIMATE_FEATURES: list[str] = [
    "rainfall_mm", "rainfall_anomaly_pct",
    "temperature_c", "humidity_pct",
]

STATIC_FEATURES: list[str] = [
    "sanitation_coverage_pct", "od_index",
    "water_access_pct", "population_density",
    "urban_pct", "wash_index", "flood_alert",
]

LAG_FEATURES_BASE: list[str] = ["cholera_cases", "typhoid_cases", "add_cases"]
LAG_WINDOWS: list[int] = [1, 2, 4, 8]
ROLLING_WINDOWS: list[int] = [4, 8]

GRAPH_FEATURES: list[str] = [
    "neighbor_mean_cholera", "neighbor_max_cholera",
    "neighbor_mean_typhoid", "neighbor_max_typhoid",
    "neighbor_mean_add",     "neighbor_max_add",
]


# ══════════════════════════════════════════════
# MODEL HYPERPARAMETERS
# ══════════════════════════════════════════════

@dataclass
class GATv2Config:
    """Graph Attention Network v2 spatial encoder config."""
    in_channels: int       = 64   # set dynamically from data
    hidden_channels: int   = 64
    out_channels: int      = 64
    num_heads: int         = 4
    num_layers: int        = 2
    dropout: float         = 0.2
    edge_dim: int          = 3    # [geo_adj, river_basin, distance]


@dataclass
class GRUConfig:
    """GRU temporal encoder config."""
    input_size: int        = 32   # set dynamically from data
    hidden_size: int       = 128
    num_layers: int        = 2
    dropout: float         = 0.2
    bidirectional: bool    = True
    output_size: int       = 64


@dataclass
class TFTConfig:
    """Temporal Fusion Transformer config."""
    hidden_size: int              = 64
    attention_head_size: int      = 4
    dropout: float                = 0.1
    hidden_continuous_size: int   = 32
    max_encoder_length: int       = 52   # 1 year of weekly data
    max_prediction_length: int    = 4    # predict 4 weeks ahead
    quantiles: list[float]        = field(
        default_factory=lambda: [0.05, 0.25, 0.5, 0.75, 0.95]
    )


@dataclass
class SEIRConfig:
    """SEIR physics-informed constraint config."""
    beta_init: float   = 0.3    # transmission rate
    gamma_init: float  = 0.1    # recovery rate
    sigma_init: float  = 0.2    # incubation rate (E→I)
    dt: float          = 1/7    # weekly timestep (fraction of year)
    seir_loss_weight: float = 0.1  # regularization weight


@dataclass
class TrainingConfig:
    """Training loop configuration."""
    seed: int              = 42
    batch_size: int        = 32
    learning_rate: float   = 1e-3
    weight_decay: float    = 1e-4
    epochs: int            = 25
    patience: int          = 15       # early stopping patience
    grad_clip: float       = 1.0      # max gradient norm
    train_split: float     = 0.70
    val_split: float       = 0.15
    test_split: float      = 0.15
    device: str            = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int       = 2
    pin_memory: bool       = True


@dataclass
class ModelConfig:
    """Master model configuration."""
    gatv2: GATv2Config     = field(default_factory=GATv2Config)
    gru: GRUConfig         = field(default_factory=GRUConfig)
    tft: TFTConfig         = field(default_factory=TFTConfig)
    seir: SEIRConfig       = field(default_factory=SEIRConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    fusion_dim: int        = 128   # GATv2_out + GRU_out
    num_diseases: int      = 3     # Cholera, Typhoid, ADD
    forecast_horizon: int  = 4     # weeks ahead


# ══════════════════════════════════════════════
# GRAPH CONFIG
# ══════════════════════════════════════════════

@dataclass
class GraphConfig:
    """District graph configuration."""
    num_nodes: int         = NUM_DISTRICTS     # 36
    edge_feature_dim: int  = 3                 # geo, river, distance
    add_self_loops: bool   = True
    undirected: bool       = True


# ══════════════════════════════════════════════
# MLFLOW & LOGGING
# ══════════════════════════════════════════════

MLFLOW_EXPERIMENT_NAME: str = "HydroCast_EWS"
MLFLOW_TRACKING_URI: str    = str(LOGS_DIR / "mlruns")

LOG_FORMAT: str = "%(asctime)s [%(levelname)s] %(name)s — %(message)s"
LOG_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"


# ══════════════════════════════════════════════
# EMERGENCY CONTACTS (Maharashtra)
# ══════════════════════════════════════════════

EMERGENCY_CONTACTS: dict[str, str] = {
    "Health Helpline":       "104",
    "Ambulance":             "108",
    "IDSP Control Room":     "1800-222-444",
    "Disaster Management":   "1078",
    "Water Supply BMC":      "022-24140000",
    "State Health Dept":     "022-22025050",
}


# ══════════════════════════════════════════════
# SINGLETON CONFIG OBJECT
# ══════════════════════════════════════════════

CONFIG = ModelConfig()


if __name__ == "__main__":
    import json
    print("=" * 60)
    print("HydroCast Configuration")
    print("=" * 60)
    print(f"Root dir     : {ROOT_DIR}")
    print(f"Districts    : {NUM_DISTRICTS}")
    print(f"Diseases     : {DISEASES}")
    print(f"Device       : {CONFIG.training.device}")
    print(f"Forecast     : {CONFIG.tft.max_prediction_length} weeks ahead")
    print(f"Encoder      : {CONFIG.tft.max_encoder_length} weeks history")
    print(f"SEIR weight  : {CONFIG.seir.seir_loss_weight}")
    print("=" * 60)
    print("All directories created successfully.")
