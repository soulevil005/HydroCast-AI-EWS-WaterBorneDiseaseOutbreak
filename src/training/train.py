"""
HydroCast — Training Loop
Full production-quality training with MLflow logging,
early stopping, gradient clipping, and checkpoint saving.
"""

from __future__ import annotations

import argparse
import logging
import random
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import (
    CONFIG, DISEASES, MAHARASHTRA_DISTRICTS,
    MODELS_DIR, LOGS_DIR, MLFLOW_EXPERIMENT_NAME, MLFLOW_TRACKING_URI,
)
from src.models.hydrocast_model  import build_hydrocast_model
from src.models.seir_constraint  import SEIRRegularizer

logger = logging.getLogger("hydrocast.trainer")


def set_global_seed(seed: int) -> None:
    """Keep direct training runs reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════
# SIMPLE DATASET WRAPPER
# ══════════════════════════════════════════════════════════════════

def build_dataloaders(
    df,
    graph_data,
    batch_size:  int = 32,
    seq_len:     int = 52,
    num_workers: int = 0,
):
    """
    Convert featured DataFrame to PyTorch DataLoaders.

    Returns
    -------
    (train_loader, val_loader, test_loader, time_feat_dim)
    """
    import pandas as pd
    from src.data_pipeline.feature_engineer import temporal_train_val_test_split

    if isinstance(df.index, pd.MultiIndex):
        df_flat = df.reset_index()
    else:
        df_flat = df.copy()

    df_flat["date"] = pd.to_datetime(df_flat["date"])
    df_flat = df_flat.sort_values(["district", "date"])

    # ── Feature columns (exclude labels and identifiers)
    exclude = {"district", "date", "cholera_cases", "typhoid_cases",
               "add_cases", "cholera_outbreak_label", "typhoid_outbreak_label",
               "add_outbreak_label"}
    feat_cols = [c for c in df_flat.columns if c not in exclude
                 and df_flat[c].dtype in [float, "float64", "float32",
                                           int, "int64", "int32"]]

    label_cols = ["cholera_outbreak_label", "typhoid_outbreak_label",
                  "add_outbreak_label"]
    label_cols = [c for c in label_cols if c in df_flat.columns]

    # ── Fill NaN
    df_flat[feat_cols]  = df_flat[feat_cols].fillna(0)
    df_flat[label_cols] = df_flat[label_cols].fillna(0) if label_cols else df_flat[[]]

    # ── Build sliding windows per district
    X_list, Y_list, D_list = [], [], []
    from src.config import DISTRICT_TO_IDX

    for district in MAHARASHTRA_DISTRICTS:
        sub = df_flat[df_flat["district"] == district].copy()
        if len(sub) < seq_len + 4:
            continue

        feats  = sub[feat_cols].values.astype(np.float32)
        labels = sub[label_cols].values.astype(np.float32) if label_cols else np.zeros((len(sub), 3), np.float32)
        dist_idx = DISTRICT_TO_IDX.get(district, 0)

        for i in range(len(sub) - seq_len - 4):
            X_list.append(feats[i : i + seq_len])
            Y_list.append(labels[i + seq_len : i + seq_len + 4])
            D_list.append(dist_idx)

    if not X_list:
        raise ValueError("No sliding windows built — check data length.")

    X = torch.tensor(np.array(X_list), dtype=torch.float32)
    Y = torch.tensor(np.array(Y_list), dtype=torch.float32)
    D = torch.tensor(D_list,           dtype=torch.long)

    # ── Temporal split by index (first 70% / next 15% / last 15%)
    n       = len(X)
    t_end   = int(n * CONFIG.training.train_split)
    v_end   = int(n * (CONFIG.training.train_split + CONFIG.training.val_split))

    def _seed_worker(worker_id: int) -> None:
        worker_seed = CONFIG.training.seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    def _make_loader(x, y, d, shuffle):
        ds = TensorDataset(x, y, d)
        generator = torch.Generator()
        generator.manual_seed(CONFIG.training.seed)
        return DataLoader(ds, batch_size=batch_size,
                          shuffle=shuffle, num_workers=num_workers,
                          worker_init_fn=_seed_worker,
                          generator=generator)

    train_loader = _make_loader(X[:t_end],    Y[:t_end],    D[:t_end],    True)
    val_loader   = _make_loader(X[t_end:v_end], Y[t_end:v_end], D[t_end:v_end], False)
    test_loader  = _make_loader(X[v_end:],    Y[v_end:],    D[v_end:],    False)

    time_feat_dim = X.shape[-1]
    logger.info(
        f"DataLoaders: train={len(train_loader.dataset)} | "
        f"val={len(val_loader.dataset)} | test={len(test_loader.dataset)} | "
        f"time_feat_dim={time_feat_dim}"
    )
    return train_loader, val_loader, test_loader, time_feat_dim


# ══════════════════════════════════════════════════════════════════
# TRAINER
# ══════════════════════════════════════════════════════════════════

class HydroCastTrainer:
    """
    Full training loop for HydroCastModel.

    Features
    --------
    - AdamW optimiser with weight decay
    - CosineAnnealingLR scheduler
    - Early stopping (patience = config.training.patience)
    - Gradient clipping (max_norm = config.training.grad_clip)
    - MLflow experiment tracking
    - Best checkpoint saving by validation F1
    - tqdm progress bars

    Parameters
    ----------
    model      : HydroCastModel instance
    graph_data : PyG Data object (static, shared across all batches)
    config     : ModelConfig (uses CONFIG singleton if None)
    run_name   : MLflow run name
    """

    def __init__(
        self,
        model:      nn.Module,
        graph_data,
        config      = None,
        run_name:   str = "hydrocast_run",
    ) -> None:
        self.model      = model
        self.graph      = graph_data
        self.config     = config or CONFIG
        self.run_name   = run_name
        self.device     = self.config.training.device

        # ── Optimiser & scheduler
        self.optimizer = AdamW(
            model.parameters(),
            lr           = self.config.training.learning_rate,
            weight_decay = self.config.training.weight_decay,
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max = self.config.training.epochs,
            eta_min = self.config.training.learning_rate * 0.01,
        )

        # ── Early stopping
        self.best_val_f1  = -np.inf
        self.patience_cnt = 0

        # ── SEIR regulariser
        self.seir_reg = SEIRRegularizer(
            seir_loss_weight = self.config.seir.seir_loss_weight,
        )

        # ── MLflow
        self._setup_mlflow()

        # ── Checkpoint dir
        MODELS_DIR.mkdir(parents=True, exist_ok=True)

    def _setup_mlflow(self) -> None:
        try:
            import mlflow
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
            self._mlflow = mlflow
            self._mlflow_active = True
        except ImportError:
            logger.warning("MLflow not installed — skipping experiment tracking.")
            self._mlflow_active = False

    def _log_metrics(self, metrics: dict, step: int) -> None:
        if self._mlflow_active:
            self._mlflow.log_metrics(metrics, step=step)

    # ──────────────────────────────────────────────────────────────
    # TRAIN ONE EPOCH
    # ──────────────────────────────────────────────────────────────

    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
    ) -> dict:
        """
        Train the model for one epoch.

        Returns
        -------
        dict  {total_loss, task_loss, seir_loss}
        """
        self.model.train()
        total_loss_sum = task_loss_sum = seir_loss_sum = 0.0
        n_batches = len(train_loader)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:03d} [TRAIN]",
                    leave=False, dynamic_ncols=True)

        for x_ts, y_labels, dist_idx in pbar:
            x_ts      = x_ts.to(self.device)
            y_labels  = y_labels.to(self.device)
            dist_idx  = dist_idx.to(self.device)

            # Build target dict
            targets = {
                "cholera": y_labels[:, :, 0],
                "typhoid": y_labels[:, :, 1],
                "add":     y_labels[:, :, 2],
            }

            # Move graph to device
            graph = self.graph.to(self.device)

            # Forward
            out = self.model(graph, x_ts, district_indices=dist_idx)

            # Loss
            losses = self.model.compute_loss(
                predictions      = out,
                targets          = targets,
                seir_regularizer = self.seir_reg,
                district         = MAHARASHTRA_DISTRICTS[dist_idx[0].item()],
            )

            # Backward
            self.optimizer.zero_grad()
            losses["total_loss"].backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.training.grad_clip,
            )
            self.optimizer.step()

            total_loss_sum += losses["total_loss"].item()
            task_loss_sum  += losses["task_loss"].item()
            seir_loss_sum  += losses["seir_loss"].item()

            pbar.set_postfix({
                "loss": f"{losses['total_loss'].item():.4f}",
                "seir": f"{losses['seir_loss'].item():.4f}",
            })

        return {
            "train/total_loss": total_loss_sum / n_batches,
            "train/task_loss":  task_loss_sum  / n_batches,
            "train/seir_loss":  seir_loss_sum  / n_batches,
        }

    # ──────────────────────────────────────────────────────────────
    # VALIDATE
    # ──────────────────────────────────────────────────────────────

    def validate(self, val_loader: DataLoader) -> dict:
        """
        Evaluate model on validation set.

        Returns
        -------
        dict  {val/total_loss, val/f1_cholera, val/f1_typhoid,
               val/f1_add, val/f1_macro}
        """
        self.model.eval()
        total_loss_sum = 0.0
        all_preds: dict[str, list] = {d.lower(): [] for d in DISEASES}
        all_labels: dict[str, list] = {d.lower(): [] for d in DISEASES}

        with torch.no_grad():
            for x_ts, y_labels, dist_idx in val_loader:
                x_ts     = x_ts.to(self.device)
                y_labels = y_labels.to(self.device)
                dist_idx = dist_idx.to(self.device)
                graph    = self.graph.to(self.device)

                out = self.model(graph, x_ts, district_indices=dist_idx)

                targets = {
                    "cholera": y_labels[:, :, 0],
                    "typhoid": y_labels[:, :, 1],
                    "add":     y_labels[:, :, 2],
                }
                losses = self.model.compute_loss(out, targets)
                total_loss_sum += losses["total_loss"].item()

                for i, disease in enumerate(DISEASES):
                    key  = disease.lower()
                    pred = (out[key] > 0.5).cpu().numpy().flatten()
                    lbl  = y_labels[:, :, i].cpu().numpy().flatten()
                    all_preds[key].extend(pred)
                    all_labels[key].extend(lbl)

        metrics = {"val/total_loss": total_loss_sum / len(val_loader)}
        f1s = []
        for disease in DISEASES:
            key = disease.lower()
            f1  = f1_score(all_labels[key], all_preds[key],
                           average="binary", zero_division=0)
            metrics[f"val/f1_{key}"] = f1
            f1s.append(f1)

        metrics["val/f1_macro"] = float(np.mean(f1s))
        return metrics

    # ──────────────────────────────────────────────────────────────
    # SAVE / LOAD CHECKPOINT
    # ──────────────────────────────────────────────────────────────

    def save_checkpoint(self, epoch: int, val_f1: float) -> Path:
        """Save model + optimizer state."""
        path = MODELS_DIR / f"checkpoint_epoch{epoch:03d}_f1{val_f1:.3f}.pt"
        torch.save({
            "epoch":      epoch,
            "model":      self.model.state_dict(),
            "optimizer":  self.optimizer.state_dict(),
            "scheduler":  self.scheduler.state_dict(),
            "val_f1":     val_f1,
            "config":     self.config,
        }, path)
        logger.info(f"Checkpoint saved: {path.name}")
        return path

    def load_checkpoint(self, path: Path) -> int:
        """Load checkpoint and restore model + optimizer state."""
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scheduler.load_state_dict(ckpt["scheduler"])
        logger.info(f"Checkpoint loaded: epoch={ckpt['epoch']} f1={ckpt['val_f1']:.3f}")
        return ckpt["epoch"]

    # ──────────────────────────────────────────────────────────────
    # MAIN FIT LOOP
    # ──────────────────────────────────────────────────────────────

    def fit(
        self,
        train_loader: DataLoader,
        val_loader:   DataLoader,
        resume_from:  Optional[Path] = None,
    ) -> dict:
        """
        Full training loop.

        Parameters
        ----------
        train_loader : training DataLoader
        val_loader   : validation DataLoader
        resume_from  : path to checkpoint to resume from (optional)

        Returns
        -------
        dict  {best_val_f1, best_epoch, history}
        """
        start_epoch = 0
        if resume_from and Path(resume_from).exists():
            start_epoch = self.load_checkpoint(Path(resume_from))

        history: list[dict] = []
        best_path: Optional[Path] = None

        run_ctx = (
            self._mlflow.start_run(run_name=self.run_name)
            if self._mlflow_active else _NullContext()
        )

        with run_ctx:
            if self._mlflow_active:
                self._mlflow.log_params({
                    "epochs":     self.config.training.epochs,
                    "batch_size": self.config.training.batch_size,
                    "lr":         self.config.training.learning_rate,
                    "patience":   self.config.training.patience,
                })

            for epoch in range(start_epoch + 1, self.config.training.epochs + 1):
                t0 = time.time()

                # Train
                train_metrics = self.train_epoch(train_loader, epoch)

                # Validate
                val_metrics = self.validate(val_loader)
                val_f1      = val_metrics["val/f1_macro"]

                # LR step
                self.scheduler.step()

                # Log
                all_metrics = {**train_metrics, **val_metrics,
                               "lr": self.scheduler.get_last_lr()[0]}
                self._log_metrics(all_metrics, step=epoch)
                history.append(all_metrics)

                elapsed = time.time() - t0
                logger.info(
                    f"Epoch {epoch:03d}/{self.config.training.epochs} | "
                    f"train_loss={train_metrics['train/total_loss']:.4f} | "
                    f"val_f1={val_f1:.4f} | "
                    f"lr={self.scheduler.get_last_lr()[0]:.2e} | "
                    f"time={elapsed:.1f}s"
                )

                # ── Early stopping + checkpoint
                if val_f1 > self.best_val_f1:
                    self.best_val_f1 = val_f1
                    self.patience_cnt = 0
                    best_path = self.save_checkpoint(epoch, val_f1)
                    if self._mlflow_active:
                        self._mlflow.log_artifact(str(best_path))
                else:
                    self.patience_cnt += 1
                    if self.patience_cnt >= self.config.training.patience:
                        logger.info(
                            f"Early stopping triggered at epoch {epoch}. "
                            f"Best val F1: {self.best_val_f1:.4f}"
                        )
                        break

        return {
            "best_val_f1": self.best_val_f1,
            "best_path":   best_path,
            "history":     history,
        }


class _NullContext:
    """No-op context manager when MLflow is unavailable."""
    def __enter__(self): return self
    def __exit__(self, *_): pass


# ══════════════════════════════════════════════════════════════════
# MAIN CLI
# ══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Train HydroCast EWS Model")
    parser.add_argument("--run-name",  default="hydrocast_run",
                        help="MLflow run name")
    parser.add_argument("--resume",    default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic data (no real files required)")
    parser.add_argument("--epochs",    type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    if args.epochs:
        CONFIG.training.epochs = args.epochs

    set_global_seed(CONFIG.training.seed)
    logger.info(f"Global seed fixed to: {CONFIG.training.seed}")

    # ── Load data
    from src.data_pipeline.data_loader      import merge_all_datasets, generate_synthetic_data
    from src.data_pipeline.feature_engineer import FeatureEngineer
    from src.data_pipeline.graph_builder    import DistrictGraphBuilder

    logger.info("Loading data...")
    if args.synthetic:
        df_raw = generate_synthetic_data(n_weeks=104)
    else:
        df_raw = merge_all_datasets()

    fe = FeatureEngineer(df_raw)
    df = fe.transform()

    builder = DistrictGraphBuilder(MAHARASHTRA_DISTRICTS)
    graph   = builder.build_pyg_data(static_df=df)

    # ── Build dataloaders
    train_loader, val_loader, test_loader, time_feat_dim = build_dataloaders(
        df, graph, batch_size=CONFIG.training.batch_size,
    )

    # ── Build model
    model = build_hydrocast_model(
        node_feat_dim = graph.x.shape[1],
        time_feat_dim = time_feat_dim,
    )

    # ── Fit SEIR
    trainer = HydroCastTrainer(model, graph, run_name=args.run_name)
    trainer.seir_reg.fit_all_districts(df)

    # ── Train
    results = trainer.fit(train_loader, val_loader, resume_from=args.resume)

    logger.info(f"\n{'='*50}")
    logger.info(f"Training complete!")
    logger.info(f"Best val F1 : {results['best_val_f1']:.4f}")
    logger.info(f"Checkpoint  : {results['best_path']}")


if __name__ == "__main__":
    main()
