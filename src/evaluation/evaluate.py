"""
HydroCast — Model Evaluator
Computes F1, precision, recall, AUC-ROC, RMSE per disease.
Also runs the ablation study (Table 3 in the paper) and
computes lead-time improvement (primary public health metric).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, mean_squared_error,
    mean_absolute_error,
)

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import DISEASES, RESULTS_DIR, PLOTS_DIR

logger = logging.getLogger("hydrocast.evaluator")


class ModelEvaluator:
    """
    Comprehensive evaluator for HydroCastModel.

    Produces all metrics needed for the research paper:
      - Table 2: baseline comparison (F1, precision, recall, AUC)
      - Table 3: ablation study
      - Table 4: forecasting metrics per horizon
      - Figure:  ROC curves, confusion matrices, lead-time plot

    Parameters
    ----------
    model       : trained HydroCastModel
    test_loader : DataLoader for test set
    graph_data  : static PyG Data object
    disease_names : list of disease names (default DISEASES)
    device      : torch device string
    """

    def __init__(
        self,
        model,
        test_loader,
        graph_data,
        disease_names: list[str] = None,
        device:        str       = "cpu",
    ) -> None:
        self.model         = model
        self.test_loader   = test_loader
        self.graph         = graph_data
        self.disease_names = disease_names or DISEASES
        self.device        = device

    # ──────────────────────────────────────────────────────────────
    # COLLECT PREDICTIONS
    # ──────────────────────────────────────────────────────────────

    def _collect_predictions(self) -> tuple[dict, dict]:
        """
        Run inference over the entire test set.

        Returns
        -------
        (all_preds, all_labels)
        Each is dict {disease_lower: np.array of shape (N,)}
        """
        self.model.eval()
        all_preds:  dict[str, list] = {d.lower(): [] for d in self.disease_names}
        all_labels: dict[str, list] = {d.lower(): [] for d in self.disease_names}
        all_probs:  dict[str, list] = {d.lower(): [] for d in self.disease_names}

        with torch.no_grad():
            for x_ts, y_labels, dist_idx in self.test_loader:
                x_ts     = x_ts.to(self.device)
                dist_idx = dist_idx.to(self.device)
                graph    = self.graph.to(self.device)

                out = self.model(graph, x_ts, district_indices=dist_idx)

                for i, disease in enumerate(self.disease_names):
                    key   = disease.lower()
                    probs = out[key].cpu().numpy()          # (B, horizon)
                    preds = (probs > 0.5).astype(int)
                    lbls  = y_labels[:, :, i].numpy()

                    all_probs[key].extend(probs.flatten().tolist())
                    all_preds[key].extend(preds.flatten().tolist())
                    all_labels[key].extend(lbls.flatten().tolist())

        return (
            {k: np.array(v) for k, v in all_preds.items()},
            {k: np.array(v) for k, v in all_labels.items()},
            {k: np.array(v) for k, v in all_probs.items()},
        )

    # ──────────────────────────────────────────────────────────────
    # CLASSIFICATION METRICS
    # ──────────────────────────────────────────────────────────────

    def compute_classification_metrics(self) -> pd.DataFrame:
        """
        Compute F1, Precision, Recall, AUC-ROC per disease
        and micro-averaged across all diseases.

        Returns
        -------
        pd.DataFrame  — formatted metrics table (Table 2 in paper)
        """
        preds, labels, probs = self._collect_predictions()

        rows = []
        all_preds_flat  = []
        all_labels_flat = []

        for disease in self.disease_names:
            key   = disease.lower()
            p, l, pr = preds[key], labels[key], probs[key]

            f1   = f1_score(l, p, average="binary", zero_division=0)
            prec = precision_score(l, p, average="binary", zero_division=0)
            rec  = recall_score(l, p, average="binary", zero_division=0)

            try:
                auc = roc_auc_score(l, pr)
            except ValueError:
                auc = float("nan")

            rows.append({
                "Disease":   disease,
                "F1":        round(f1, 4),
                "Precision": round(prec, 4),
                "Recall":    round(rec, 4),
                "AUC-ROC":   round(auc, 4),
                "Positives": int(l.sum()),
                "Total":     len(l),
            })

            all_preds_flat.extend(p.tolist())
            all_labels_flat.extend(l.tolist())

        # Micro-averaged row
        micro_f1 = f1_score(all_labels_flat, all_preds_flat,
                             average="micro", zero_division=0)
        rows.append({
            "Disease":   "MACRO AVG",
            "F1":        round(float(np.mean([r["F1"] for r in rows])), 4),
            "Precision": round(float(np.mean([r["Precision"] for r in rows])), 4),
            "Recall":    round(float(np.mean([r["Recall"] for r in rows])), 4),
            "AUC-ROC":   round(float(np.nanmean([r["AUC-ROC"] for r in rows])), 4),
            "Positives": "-",
            "Total":     len(all_labels_flat),
        })

        df = pd.DataFrame(rows)
        logger.info(f"\n{df.to_string(index=False)}")

        # Save
        out = RESULTS_DIR / "classification_metrics.csv"
        df.to_csv(out, index=False)
        logger.info(f"Metrics saved: {out}")
        return df

    # ──────────────────────────────────────────────────────────────
    # FORECASTING METRICS
    # ──────────────────────────────────────────────────────────────

    def compute_forecasting_metrics(self) -> pd.DataFrame:
        """
        Compute RMSE and MAE for continuous outbreak probability
        forecasts per horizon (week+1 to week+4).

        Returns
        -------
        pd.DataFrame  columns: [Disease, Horizon, RMSE, MAE]
        """
        self.model.eval()
        rows = []

        horizon_preds:  dict[str, list] = {d.lower(): [[] for _ in range(4)] for d in self.disease_names}
        horizon_labels: dict[str, list] = {d.lower(): [[] for _ in range(4)] for d in self.disease_names}

        with torch.no_grad():
            for x_ts, y_labels, dist_idx in self.test_loader:
                x_ts     = x_ts.to(self.device)
                dist_idx = dist_idx.to(self.device)
                out      = self.model(self.graph.to(self.device), x_ts,
                                      district_indices=dist_idx)

                for i, disease in enumerate(self.disease_names):
                    key   = disease.lower()
                    probs = out[key].cpu().numpy()    # (B, 4)
                    lbls  = y_labels[:, :, i].numpy() # (B, 4)

                    for h in range(min(4, probs.shape[1])):
                        horizon_preds[key][h].extend(probs[:, h].tolist())
                        horizon_labels[key][h].extend(lbls[:, h].tolist())

        for disease in self.disease_names:
            key = disease.lower()
            for h in range(4):
                p = np.array(horizon_preds[key][h])
                l = np.array(horizon_labels[key][h])
                if len(p) == 0:
                    continue
                rmse = float(np.sqrt(mean_squared_error(l, p)))
                mae  = float(mean_absolute_error(l, p))
                rows.append({
                    "Disease": disease,
                    "Horizon": f"Week+{h+1}",
                    "RMSE":    round(rmse, 4),
                    "MAE":     round(mae,  4),
                })

        df = pd.DataFrame(rows)
        out = RESULTS_DIR / "forecasting_metrics.csv"
        df.to_csv(out, index=False)
        logger.info(f"Forecasting metrics saved: {out}")
        return df

    # ──────────────────────────────────────────────────────────────
    # ABLATION STUDY
    # ──────────────────────────────────────────────────────────────

    def run_ablation_study(self, variants: dict) -> pd.DataFrame:
        """
        Evaluate each model variant and produce an ablation table.
        This becomes Table 3 in the research paper.

        Parameters
        ----------
        variants : dict {model_name: model_instance}
                   e.g. {"No GNN": ..., "No TFT": ..., "Full": model}

        Returns
        -------
        pd.DataFrame  columns: [Model, F1_Cholera, F1_Typhoid, F1_ADD, F1_Macro]
        """
        rows = []
        original_model = self.model

        for name, variant_model in variants.items():
            self.model = variant_model
            try:
                df_m = self.compute_classification_metrics()
                macro_f1 = df_m[df_m["Disease"] == "MACRO AVG"]["F1"].values[0]
                row = {"Model": name, "F1_Macro": float(macro_f1)}
                for disease in self.disease_names:
                    f1 = df_m[df_m["Disease"] == disease]["F1"].values
                    row[f"F1_{disease}"] = float(f1[0]) if len(f1) > 0 else 0.0
                rows.append(row)
            except Exception as e:
                logger.warning(f"Ablation variant '{name}' failed: {e}")
                rows.append({"Model": name, "F1_Macro": 0.0})

        self.model = original_model

        df = pd.DataFrame(rows).sort_values("F1_Macro", ascending=False)
        out = RESULTS_DIR / "ablation_study.csv"
        df.to_csv(out, index=False)
        logger.info(f"Ablation study saved:\n{df.to_string(index=False)}")
        return df

    # ──────────────────────────────────────────────────────────────
    # LEAD TIME
    # ──────────────────────────────────────────────────────────────

    def compute_lead_time(
        self,
        threshold: float = 0.80,
    ) -> dict:
        """
        For each confirmed outbreak in the test set, measure how many
        weeks in advance the model first exceeded the threshold.

        Parameters
        ----------
        threshold : probability threshold for alert (default 0.80)

        Returns
        -------
        dict  {mean_lead_weeks, median_lead_weeks, pct_caught_2wk_early}
        """
        self.model.eval()
        lead_times = []

        with torch.no_grad():
            for x_ts, y_labels, dist_idx in self.test_loader:
                x_ts  = x_ts.to(self.device)
                out   = self.model(self.graph.to(self.device), x_ts,
                                   district_indices=dist_idx.to(self.device))

                for i, disease in enumerate(self.disease_names):
                    key   = disease.lower()
                    probs = out[key].cpu().numpy()    # (B, 4)
                    lbls  = y_labels[:, :, i].numpy()

                    for b in range(probs.shape[0]):
                        if lbls[b].max() > 0:
                            # Find first week model exceeds threshold
                            alert_weeks = np.where(probs[b] >= threshold)[0]
                            if len(alert_weeks) > 0:
                                lead = int(alert_weeks[0]) + 1
                                lead_times.append(lead)

        if not lead_times:
            return {"mean_lead_weeks": 0, "median_lead_weeks": 0,
                    "pct_caught_2wk_early": 0.0}

        arr = np.array(lead_times)
        result = {
            "mean_lead_weeks":     float(arr.mean()),
            "median_lead_weeks":   float(np.median(arr)),
            "pct_caught_2wk_early":float((arr >= 2).mean() * 100),
            "n_outbreaks_detected":int(len(arr)),
        }
        logger.info(f"Lead time: {result}")
        return result

    # ──────────────────────────────────────────────────────────────
    # GENERATE FULL REPORT
    # ──────────────────────────────────────────────────────────────

    def generate_report(self, save_path: Optional[Path] = None) -> None:
        """
        Generate and save a complete evaluation report (CSV + plots).

        Parameters
        ----------
        save_path : directory to save all outputs (default RESULTS_DIR)
        """
        save_path = save_path or RESULTS_DIR
        save_path.mkdir(parents=True, exist_ok=True)

        logger.info("=== Generating Full Evaluation Report ===")

        # Classification metrics
        clf_df  = self.compute_classification_metrics()
        fore_df = self.compute_forecasting_metrics()
        lead    = self.compute_lead_time()

        # Save lead time
        with open(save_path / "lead_time.json", "w") as f:
            json.dump(lead, f, indent=2)

        # Save ROC curves
        self._save_roc_curves(save_path)

        # Save confusion matrices
        self._save_confusion_matrices(save_path)

        logger.info(f"\n{'='*50}")
        logger.info("EVALUATION SUMMARY")
        logger.info(f"{'='*50}")
        logger.info(clf_df[["Disease", "F1", "Precision", "Recall", "AUC-ROC"]].to_string(index=False))
        logger.info(f"\nLead time: {lead['mean_lead_weeks']:.1f} weeks avg | "
                    f"{lead['pct_caught_2wk_early']:.0f}% caught ≥2wk early")

    def _save_roc_curves(self, save_path: Path) -> None:
        try:
            import matplotlib.pyplot as plt
            from sklearn.metrics import roc_curve

            _, labels, probs = self._collect_predictions()
            fig, axes = plt.subplots(1, len(self.disease_names),
                                     figsize=(5 * len(self.disease_names), 5),
                                     facecolor="#070b14")
            colors = ["#ff3d5a", "#ffb84d", "#4d9fff"]

            for ax, disease, color in zip(axes, self.disease_names, colors):
                key = disease.lower()
                fpr, tpr, _ = roc_curve(labels[key], probs[key])
                auc = roc_auc_score(labels[key], probs[key])
                ax.set_facecolor("#0d1420")
                ax.plot(fpr, tpr, color=color, lw=2,
                        label=f"AUC = {auc:.3f}")
                ax.plot([0, 1], [0, 1], "--", color="#444")
                ax.set_title(disease, color="white")
                ax.set_xlabel("FPR", color="white")
                ax.set_ylabel("TPR", color="white")
                ax.tick_params(colors="white")
                ax.legend(loc="lower right", labelcolor="white",
                          facecolor="#0d1420")

            plt.suptitle("ROC Curves — HydroCast", color="white", y=1.02)
            plt.tight_layout()
            plt.savefig(save_path / "roc_curves.png", dpi=150,
                        bbox_inches="tight", facecolor="#070b14")
            plt.close()
            logger.info("ROC curves saved.")
        except Exception as e:
            logger.warning(f"Could not save ROC curves: {e}")

    def _save_confusion_matrices(self, save_path: Path) -> None:
        try:
            import matplotlib.pyplot as plt

            preds, labels, _ = self._collect_predictions()
            fig, axes = plt.subplots(1, len(self.disease_names),
                                     figsize=(5 * len(self.disease_names), 4),
                                     facecolor="#070b14")

            for ax, disease in zip(axes, self.disease_names):
                key = disease.lower()
                cm  = confusion_matrix(labels[key], preds[key])
                ax.set_facecolor("#0d1420")
                im = ax.imshow(cm, cmap="Blues")
                ax.set_title(disease, color="white")
                ax.set_xlabel("Predicted", color="white")
                ax.set_ylabel("True",      color="white")
                ax.tick_params(colors="white")
                for (r, c), val in np.ndenumerate(cm):
                    ax.text(c, r, str(val), ha="center", va="center",
                            color="white", fontsize=12)

            plt.suptitle("Confusion Matrices — HydroCast", color="white")
            plt.tight_layout()
            plt.savefig(save_path / "confusion_matrices.png", dpi=150,
                        bbox_inches="tight", facecolor="#070b14")
            plt.close()
            logger.info("Confusion matrices saved.")
        except Exception as e:
            logger.warning(f"Could not save confusion matrices: {e}")
