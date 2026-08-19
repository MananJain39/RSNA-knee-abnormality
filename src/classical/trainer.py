"""Training orchestration for the global and cluster-specialist DINOv2 models.

The trainer accepts the dataset and collate protocols from the classical
package.  It deliberately keeps data construction outside this module so CLI
entry points can choose disk-backed or in-memory datasets without changing the
training loop.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from src.classical.cluster_specialist import filter_train_rows_for_cluster, make_specialist
from src.classical.collate import Batch, collate_batch
from src.classical.dataset import KneeMRIDatasetProtocol
from src.classical.dinov2_base import KneeModel
from src.config import (
    CLASSICAL_BATCH_SIZE,
    CLASSICAL_COSINE_T_MAX,
    CLASSICAL_EMA_DECAY_MAIN,
    CLASSICAL_EMA_DECAY_SPECIALIST,
    CLASSICAL_GRAD_ACCUM_STEPS,
    CLASSICAL_IMAGE_SIZE,
    CLASSICAL_LR,
    CLASSICAL_MAX_EPOCHS,
    CLASSICAL_USE_AMP,
    CLASSICAL_WARMUP_STEPS,
    CLASSICAL_WEIGHT_DECAY,
    LABEL_COLUMNS,
)
from src.experiments.tracker import ExperimentTracker
from src.imaging.slot_builder import SLOT_NAMES
from src.validation.metrics import macro_auc


class TrainerProtocol(Protocol):
    """Train the global branch or a single acquisition-cluster specialist."""

    def train_main(
        self,
        dataset: KneeMRIDatasetProtocol,
        val_dataset: KneeMRIDatasetProtocol,
        output_dir: Path,
    ) -> "TrainResult": ...

    def train_specialist(
        self,
        cluster_id: int,
        dataset: KneeMRIDatasetProtocol,
        val_dataset: KneeMRIDatasetProtocol,
        output_dir: Path,
    ) -> "TrainResult": ...


@dataclass(frozen=True)
class TrainResult:
    """Best checkpoint and metrics produced by one training run."""

    epoch: int
    best_val_macro_auc: float
    checkpoint_path: Path
    metrics_dict: dict[str, float]


class _EMA:
    """Exponential moving average of a model, retained for evaluation/saving."""

    def __init__(self, model: torch.nn.Module, decay: float):
        self.model = deepcopy(model).eval()
        self.decay = decay
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def update(self, model: torch.nn.Module) -> None:
        with torch.no_grad():
            for ema_value, value in zip(self.model.parameters(), model.parameters()):
                ema_value.mul_(self.decay).add_(value.detach(), alpha=1.0 - self.decay)
            for ema_buffer, buffer in zip(self.model.buffers(), model.buffers()):
                ema_buffer.copy_(buffer)

    def state_dict(self) -> dict:
        return self.model.state_dict()


class ClassicalTrainer:
    """AdamW + warmup/cosine + EMA training for hierarchical knee models."""

    def __init__(
        self,
        model_factory: Callable[[], KneeModel] = KneeModel,
        specialist_factory: Callable[[int, int], KneeModel] = make_specialist,
        batch_size: int = CLASSICAL_BATCH_SIZE,
        max_epochs: int = CLASSICAL_MAX_EPOCHS,
        device: str | torch.device | None = None,
        cluster_assignments: pd.DataFrame | None = None,
        resume_from: Path | None = None,
        tracker_factory: Callable[[str], ExperimentTracker] = ExperimentTracker,
        collate_fn: Callable[[list], Batch] = collate_batch,
    ) -> None:
        self.model_factory = model_factory
        self.specialist_factory = specialist_factory
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.cluster_assignments = cluster_assignments
        self.resume_from = Path(resume_from) if resume_from else None
        self.tracker_factory = tracker_factory
        self.collate_fn = collate_fn

    def train_main(
        self,
        dataset: KneeMRIDatasetProtocol,
        val_dataset: KneeMRIDatasetProtocol,
        output_dir: Path,
    ) -> TrainResult:
        """Train the global DINOv2 branch on every supplied training study."""
        return self._train(self.model_factory(), dataset, val_dataset, output_dir, "main")

    def train_specialist(
        self,
        cluster_id: int,
        dataset: KneeMRIDatasetProtocol,
        val_dataset: KneeMRIDatasetProtocol,
        output_dir: Path,
    ) -> TrainResult:
        """Train a small model using only studies routed to ``cluster_id``."""
        filtered_dataset = self._filter_dataset(dataset, cluster_id)
        return self._train(
            self.specialist_factory(cluster_id, CLASSICAL_IMAGE_SIZE),
            filtered_dataset,
            val_dataset,
            output_dir,
            f"specialist_cluster_{cluster_id}",
        )

    def _filter_dataset(self, dataset: KneeMRIDatasetProtocol, cluster_id: int):
        if not hasattr(dataset, "labels_df") or not hasattr(dataset, "study_uids"):
            raise TypeError(
                "Specialist training requires a KneeMRIDataset with labels_df and study_uids"
            )
        train_df = dataset.labels_df.reset_index()
        filtered = filter_train_rows_for_cluster(
            train_df, self.cluster_assignments, cluster_id
        )
        selected_uids = set(filtered["StudyInstanceUID"])
        indices = [i for i, uid in enumerate(dataset.study_uids) if uid in selected_uids]
        if not indices:
            raise ValueError(f"No training studies found for cluster {cluster_id}")
        return Subset(dataset, indices)

    def _train(
        self,
        model: KneeModel,
        dataset: KneeMRIDatasetProtocol,
        val_dataset: KneeMRIDatasetProtocol,
        output_dir: Path,
        run_name: str,
    ) -> TrainResult:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        model = model.to(self.device)
        optimizer = torch.optim.AdamW(
            model.parameter_groups(CLASSICAL_LR, CLASSICAL_WEIGHT_DECAY)
        )
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0 / max(CLASSICAL_WARMUP_STEPS, 1),
            total_iters=max(CLASSICAL_WARMUP_STEPS, 1),
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=CLASSICAL_COSINE_T_MAX
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[CLASSICAL_WARMUP_STEPS],
        )
        scaler = torch.cuda.amp.GradScaler(
            enabled=CLASSICAL_USE_AMP and self.device.type == "cuda"
        )
        ema = _EMA(model, self._ema_decay(run_name))
        start_epoch, best = self._resume(model, ema)
        tracker = self.tracker_factory(run_name)
        tracker.log_params({"batch_size": self.batch_size, "max_epochs": self.max_epochs})
        train_loader = self._loader(dataset, shuffle=True)
        val_loader = self._loader(val_dataset, shuffle=False)
        checkpoint_path = output_dir / "best.pt"
        final_metrics: dict[str, float] = {}

        for epoch in range(start_epoch, self.max_epochs):
            train_loss = self._train_epoch(
                model, train_loader, optimizer, scheduler, scaler, ema
            )
            val_auc = self._validate(ema.model, val_loader)
            final_metrics = {"train_loss": train_loss, "val_macro_auc": val_auc}
            tracker.log_step(epoch, final_metrics)
            if val_auc >= best:
                best = val_auc
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "ema_state_dict": ema.state_dict(),
                        "epoch": epoch,
                        "best_val_macro_auc": best,
                    },
                    checkpoint_path,
                )

        tracker.log_metrics({**final_metrics, "best_val_macro_auc": best})
        tracker.save()
        return TrainResult(self.max_epochs - 1, best, checkpoint_path, final_metrics)

    def _loader(self, dataset: KneeMRIDatasetProtocol, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            collate_fn=self.collate_fn,
        )

    def _resume(self, model: KneeModel, ema: _EMA) -> tuple[int, float]:
        if self.resume_from is None or not self.resume_from.exists():
            return 0, float("-inf")
        checkpoint = torch.load(self.resume_from, map_location=self.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        if checkpoint.get("ema_state_dict"):
            ema.model.load_state_dict(checkpoint["ema_state_dict"])
        return int(checkpoint["epoch"]) + 1, float(checkpoint["best_val_macro_auc"])

    def _ema_decay(self, run_name: str) -> float:
        if run_name.startswith("specialist"):
            return CLASSICAL_EMA_DECAY_SPECIALIST
        return CLASSICAL_EMA_DECAY_MAIN

    def _train_epoch(self, model, loader, optimizer, scheduler, scaler, ema) -> float:
        model.train()
        losses: list[float] = []
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(loader):
            slots, labels, weights, mask = self._to_tensors(batch)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                logits = self._forward_studies(model, slots, mask)
                loss = self._weighted_bce(logits, labels, weights) / CLASSICAL_GRAD_ACCUM_STEPS
            scaler.scale(loss).backward()
            if (step + 1) % CLASSICAL_GRAD_ACCUM_STEPS == 0 or step + 1 == len(loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                ema.update(model)
            losses.append(float(loss.detach().cpu()) * CLASSICAL_GRAD_ACCUM_STEPS)
        return float(np.mean(losses)) if losses else 0.0

    def _validate(self, model, loader: DataLoader) -> float:
        model.eval()
        labels, predictions = [], []
        with torch.no_grad():
            for batch in loader:
                slots, batch_labels, _, mask = self._to_tensors(batch)
                logits = self._forward_studies(model, slots, mask)
                labels.append(batch_labels.cpu().numpy())
                predictions.append(torch.sigmoid(logits).cpu().numpy())
        if not labels:
            return 0.0
        return macro_auc(
            pd.DataFrame(np.concatenate(labels), columns=LABEL_COLUMNS),
            pd.DataFrame(np.concatenate(predictions), columns=LABEL_COLUMNS),
        )

    def _to_tensors(self, batch: Batch):
        return tuple(torch.as_tensor(value, device=self.device) for value in batch)

    @staticmethod
    def _forward_studies(model, slots: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        per_slot = []
        for slot_index, _ in enumerate(SLOT_NAMES):
            batch_size, groups = slots.shape[:2][0], slots.shape[2]
            images = slots[:, slot_index].reshape(-1, *slots.shape[3:])
            logits = model.encode_slot(images).reshape(batch_size, groups, -1).mean(dim=1)
            per_slot.append(logits * mask[:, slot_index : slot_index + 1])
        summed = torch.stack(per_slot).sum(dim=0)
        return summed / mask.sum(dim=1, keepdim=True).clamp_min(1.0)

    @staticmethod
    def _weighted_bce(logits, labels, weights):
        safe_labels = torch.nan_to_num(labels, nan=0.5)
        safe_weights = torch.where(torch.isnan(labels), torch.zeros_like(weights), weights)
        bce = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, safe_labels, reduction="none"
        )
        return (bce * safe_weights).sum() / safe_weights.sum().clamp_min(1.0)


__all__ = ["TrainerProtocol", "TrainResult", "ClassicalTrainer"]
