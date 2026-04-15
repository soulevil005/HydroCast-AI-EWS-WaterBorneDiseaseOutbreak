"""
HydroCast — Master Pipeline Orchestrator
One command to run the entire project end-to-end.

Usage
-----
# Full pipeline with synthetic data (no real IDSP files needed):
    python run_pipeline.py --mode all --synthetic

# Individual steps:
    python run_pipeline.py --mode data      --synthetic
    python run_pipeline.py --mode train     --synthetic
    python run_pipeline.py --mode baselines --synthetic
    python run_pipeline.py --mode eval
    python run_pipeline.py --mode explain
    python run_pipeline.py --mode predict
    python run_pipeline.py --mode dashboard
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path

# ── Make src importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt = "%H:%M:%S",
)
logger = logging.getLogger("hydrocast.pipeline")


# ══════════════════════════════════════════════════════════════════
# PIPELINE STEPS
# ══════════════════════════════════════════════════════════════════

def step_data(synthetic: bool = True) -> tuple:
    """Load, merge, and feature-engineer all datasets."""
    from src.config import MAHARASHTRA_DISTRICTS
    from src.data_pipeline.data_loader      import merge_all_datasets, generate_synthetic_data
    from src.data_pipeline.feature_engineer import FeatureEngineer, temporal_train_val_test_split
    from src.data_pipeline.graph_builder    import DistrictGraphBuilder

    logger.info("━" * 55)
    logger.info("STEP 1 — Data Pipeline")
    logger.info("━" * 55)

    if synthetic:
        df_raw = generate_synthetic_data(n_weeks=104)
    else:
        df_raw = merge_all_datasets()

    logger.info(f"Raw data shape: {df_raw.shape}")

    fe = FeatureEngineer(df_raw)
    df = fe.transform()
    logger.info(f"Featured data shape: {df.shape}")

    train_df, val_df, test_df = temporal_train_val_test_split(df)
    logger.info(f"Split — train:{len(train_df)} val:{len(val_df)} test:{len(test_df)}")

    builder = DistrictGraphBuilder(MAHARASHTRA_DISTRICTS)
    graph   = builder.build_pyg_data(static_df=df)
    logger.info(f"District graph: {graph.num_nodes} nodes, {graph.num_edges} edges")

    logger.info("✅ Data step complete.\n")
    return df, graph, fe


def step_train(df, graph, fe) -> tuple:
    """Train the HydroCast model."""
    from src.config import CONFIG
    from src.models.hydrocast_model import build_hydrocast_model
    from src.models.seir_constraint import SEIRRegularizer
    from src.training.train         import HydroCastTrainer, build_dataloaders

    logger.info("━" * 55)
    logger.info("STEP 2 — Model Training")
    logger.info("━" * 55)

    train_loader, val_loader, test_loader, time_feat_dim = build_dataloaders(
        df, graph, batch_size=CONFIG.training.batch_size,
    )

    model = build_hydrocast_model(
        node_feat_dim = graph.x.shape[1],
        time_feat_dim = time_feat_dim,
    )
    logger.info(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    trainer = HydroCastTrainer(model, graph, run_name="hydrocast_main")

    logger.info("Fitting SEIR models to districts...")
    trainer.seir_reg.fit_all_districts(df)

    results = trainer.fit(train_loader, val_loader)
    logger.info(f"Training complete — Best val F1: {results['best_val_f1']:.4f}")
    logger.info("✅ Train step complete.\n")

    return model, train_loader, val_loader, test_loader, time_feat_dim


def step_baselines(df, graph, time_feat_dim) -> None:
    """Train and evaluate all 6 baseline models."""
    from src.evaluation.baselines import BaselineEvaluator
    from src.training.train       import build_dataloaders
    from src.config               import CONFIG

    logger.info("━" * 55)
    logger.info("STEP 3 — Baseline Comparison")
    logger.info("━" * 55)

    train_loader, val_loader, test_loader, _ = build_dataloaders(
        df, graph, batch_size=CONFIG.training.batch_size,
    )

    evaluator = BaselineEvaluator(
        train_loader  = train_loader,
        val_loader    = val_loader,
        test_loader   = test_loader,
        time_feat_dim = time_feat_dim,
    )

    evaluator.train_sklearn_baselines()
    evaluator.train_deep_baselines()
    df_results = evaluator.evaluate_all()
    evaluator.plot_comparison(df_results)

    logger.info("✅ Baselines step complete.\n")


def step_evaluate(model, test_loader, graph) -> None:
    """Evaluate HydroCast on test set."""
    from src.evaluation.evaluate import ModelEvaluator
    from src.config              import CONFIG

    logger.info("━" * 55)
    logger.info("STEP 4 — Model Evaluation")
    logger.info("━" * 55)

    evaluator = ModelEvaluator(
        model         = model,
        test_loader   = test_loader,
        graph_data    = graph,
        device        = CONFIG.training.device,
    )
    evaluator.generate_report()
    logger.info("✅ Evaluation step complete.\n")


def step_explain(model, test_loader, graph, fe) -> None:
    """Compute SHAP explanations for all districts."""
    from src.explainability.shap_explainer import HydroCastExplainer
    from src.config                        import CONFIG
    import torch

    logger.info("━" * 55)
    logger.info("STEP 5 — SHAP Explainability")
    logger.info("━" * 55)

    # Use first batch as background
    x_bg, _, _ = next(iter(test_loader))
    explainer = HydroCastExplainer(
        model           = model,
        feature_names   = fe.get_feature_names(),
        background_data = x_bg,
        device          = CONFIG.training.device,
    )

    results = explainer.explain_all_districts(test_loader, graph)
    logger.info(f"SHAP complete for {len(results)} districts.")
    logger.info("✅ Explainability step complete.\n")


def step_predict(model, graph, df) -> dict:
    """Generate current-week predictions for all districts."""
    from src.config import MAHARASHTRA_DISTRICTS, RESULTS_DIR, CONFIG
    import pandas as pd
    import torch

    logger.info("━" * 55)
    logger.info("STEP 6 — Generate Predictions")
    logger.info("━" * 55)

    predictions = {}
    model.eval()
    seq_len = 52

    if isinstance(df.index, pd.MultiIndex):
        df_flat = df.reset_index()
    else:
        df_flat = df.copy()

    if "date" in df_flat.columns:
        df_flat["date"] = pd.to_datetime(df_flat["date"])

    exclude = {
        "district", "date", "cholera_cases", "typhoid_cases", "add_cases",
        "cholera_outbreak_label", "typhoid_outbreak_label", "add_outbreak_label",
    }
    feature_cols = [
        col for col in df_flat.columns
        if col not in exclude and pd.api.types.is_numeric_dtype(df_flat[col])
    ]
    time_feat_dim = getattr(getattr(model, "temporal_encoder", None), "input_size", len(feature_cols))

    with torch.no_grad():
        for district in MAHARASHTRA_DISTRICTS:
            district_df = df_flat[df_flat["district"] == district].sort_values("date")
            if len(district_df) < seq_len:
                continue

            ts_values = (
                district_df[feature_cols]
                .tail(seq_len)
                .fillna(0)
                .values.astype("float32")
            )
            if ts_values.shape[1] != time_feat_dim:
                raise ValueError(
                    f"Prediction feature width mismatch for {district}: "
                    f"expected {time_feat_dim}, got {ts_values.shape[1]}"
                )
            x_ts = torch.tensor(ts_values, dtype=torch.float32, device=CONFIG.training.device).unsqueeze(0)
            result = model.predict_district(
                district_name = district,
                graph_data    = graph.to(CONFIG.training.device),
                time_series   = x_ts,
            )
            latest = district_df.iloc[-1]
            top_disease = max(
                result["predictions"],
                key=lambda disease_name: result["predictions"][disease_name]["max_prob"],
            )
            result["context"] = {
                "rainfall_anomaly_pct": float(latest.get("rainfall_anomaly_pct", 0.0)),
                "sanitation_coverage_pct": float(latest.get("sanitation_coverage_pct", 0.0)),
                "case_count": int(latest.get(f"{top_disease.lower()}_cases", latest.get("add_cases", 0))),
                "week": str(latest.get("date", "")),
            }
            predictions[district] = result

    # Save predictions
    out = RESULTS_DIR / "predictions.json"
    with open(out, "w") as f:
        # Convert to JSON-serialisable format
        safe = {
            d: {
                "summary": r["summary"],
                "context": r.get("context", {}),
                "predictions": {
                    dis: {
                        "risk_level": info["risk_level"],
                        "max_prob":   round(info["max_prob"], 4),
                        "week_probs": [round(p, 4) for p in info["week_probs"]],
                    }
                    for dis, info in r["predictions"].items()
                }
            }
            for d, r in predictions.items()
        }
        json.dump(safe, f, indent=2)

    # Print risk table
    logger.info("\n── Current District Risk Table ──")
    for district, res in list(predictions.items())[:10]:
        preds = res["predictions"]
        top_d = max(preds, key=lambda d: preds[d]["max_prob"])
        prob  = preds[top_d]["max_prob"]
        level = preds[top_d]["risk_level"]
        logger.info(f"  {district:20s} {level.upper():10s} {top_d:8s} {prob:.2%}")

    logger.info(f"Predictions saved: {out}")
    logger.info("✅ Predict step complete.\n")
    return predictions


def step_dashboard() -> None:
    """Launch the Streamlit dashboard."""
    import subprocess

    logger.info("━" * 55)
    logger.info("STEP 7 — Launching Dashboard")
    logger.info("━" * 55)
    logger.info("Opening HydroCast dashboard at http://localhost:8501")
    logger.info("Press Ctrl+C to stop.\n")

    dashboard_path = Path(__file__).parent / "src" / "dashboard" / "app.py"

    if not dashboard_path.exists():
        logger.error(f"Dashboard file not found: {dashboard_path}")
        logger.info("Generate it using Prompt 16 from the coding guide.")
        return

    subprocess.run(
        ["streamlit", "run", str(dashboard_path),
         "--server.headless", "true",
         "--browser.gatherUsageStats", "false"],
        check=True,
    )


# ══════════════════════════════════════════════════════════════════
# STEP REGISTRY
# ══════════════════════════════════════════════════════════════════

STEP_MAP = {
    "data":      ["data"],
    "train":     ["data", "train"],
    "baselines": ["data", "baselines"],
    "eval":      ["data", "train", "eval"],
    "explain":   ["data", "train", "explain"],
    "predict":   ["data", "train", "predict"],
    "dashboard": ["data", "train", "predict", "dashboard"],
    "all":       ["data", "train", "baselines", "eval",
                  "explain", "predict", "dashboard"],
}


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="HydroCast — AI Early Warning System Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py --mode all --synthetic
  python run_pipeline.py --mode train --synthetic
  python run_pipeline.py --mode dashboard
  python run_pipeline.py --mode eval
        """,
    )
    parser.add_argument(
        "--mode",
        choices=list(STEP_MAP.keys()),
        default="all",
        help="Pipeline step(s) to run (default: all)",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic data — no real IDSP/IMD/NFHS-5 files needed",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override training epochs (default from config)",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Load existing checkpoint instead of training",
    )
    args = parser.parse_args()

    if args.epochs:
        from src.config import CONFIG
        CONFIG.training.epochs = args.epochs
        logger.info(f"Epochs overridden to: {args.epochs}")

    steps = STEP_MAP[args.mode]

    logger.info("=" * 55)
    logger.info("  HydroCast — AI Early Warning System")
    logger.info("  Waterborne Disease Outbreak Prediction")
    logger.info("  Maharashtra, India")
    logger.info("=" * 55)
    logger.info(f"Mode      : {args.mode}")
    logger.info(f"Steps     : {steps}")
    logger.info(f"Synthetic : {args.synthetic}")
    logger.info("=" * 55 + "\n")

    # ── State variables passed between steps
    df       = None
    graph    = None
    fe       = None
    model    = None
    train_loader = val_loader = test_loader = None
    time_feat_dim = 32

    # ── Execute steps in order
    if "data" in steps:
        df, graph, fe = step_data(synthetic=args.synthetic)

    if "train" in steps and not args.skip_train:
        model, train_loader, val_loader, test_loader, time_feat_dim = \
            step_train(df, graph, fe)

    if "baselines" in steps:
        step_baselines(df, graph, time_feat_dim)

    if "eval" in steps and model is not None:
        step_evaluate(model, test_loader, graph)

    if "explain" in steps and model is not None:
        step_explain(model, test_loader, graph, fe)

    if "predict" in steps and model is not None:
        step_predict(model, graph, df)

    if "dashboard" in steps:
        step_dashboard()

    logger.info("\n" + "=" * 55)
    logger.info("  ✅ HydroCast pipeline complete!")
    logger.info("=" * 55)


if __name__ == "__main__":
    main()
