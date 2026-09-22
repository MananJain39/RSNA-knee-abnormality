"""Study-level inference for the global DINOv2 model and routed specialists.

Model paths use ``{"main": path, "specialists": {cluster_id: path}}``.  A
flat ``{"specialist": path}`` mapping is also accepted when one shared
specialist checkpoint is intentionally used. Missing specialist routes fall
back cleanly to the main branch.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Protocol

import numpy as np
import pandas as pd
import torch

from src.classical.cluster_specialist import make_specialist, specialist_router
from src.classical.collate import Batch, collate_batch
from src.classical.dataset import KneeMRIDataset
from src.classical.dinov2_base import KneeModel
from src.classical.ensemble import rank_mean_logits, to_probabilities
from src.config import CLASSICAL_ENSEMBLE_WEIGHTS, CLASSICAL_IMAGE_SIZE, LABEL_COLUMNS
from src.imaging.slot_builder import SLOT_NAMES


class PredictorProtocol(Protocol):
    """Return one bounded score per requested study and diagnosis."""

    def predict_studies(
        self,
        model_paths: dict,
        study_uids: list[str],
        volume_index: dict,
        labels_df: pd.DataFrame,
        cluster_assignments: pd.DataFrame,
    ) -> pd.DataFrame: ...


class ClassicalPredictor:
    """Load/cache checkpoints and blend main plus cluster-routed predictions."""

    def __init__(
        self,
        device: str | torch.device | None = None,
        model_factory: Callable[[], KneeModel] = KneeModel,
        specialist_factory: Callable[[int, int], KneeModel] = make_specialist,
        dataset_factory: Callable[..., KneeMRIDataset] = KneeMRIDataset,
        collate_fn: Callable[[list], Batch] = collate_batch,
    ) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model_factory = model_factory
        self.specialist_factory = specialist_factory
        self.dataset_factory = dataset_factory
        self.collate_fn = collate_fn
        self._model_cache: dict[Path, torch.nn.Module] = {}

    def predict_studies(
        self,
        model_paths: dict,
        study_uids: list[str],
        volume_index: dict,
        labels_df: pd.DataFrame,
        cluster_assignments: pd.DataFrame,
    ) -> pd.DataFrame:
        """Predict requested studies, routing to a specialist when available."""
        if "main" not in model_paths:
            raise KeyError("model_paths must contain a 'main' checkpoint")
        main = self._load_checkpoint(Path(model_paths["main"]), self.model_factory)
        records = []
        for study_uid in study_uids:
            main_logits = self._predict_one(main, study_uid, volume_index, labels_df)
            specialist_path = self._specialist_path(model_paths, study_uid, cluster_assignments)
            if specialist_path is None:
                # ``rank_mean_logits`` ranks studies within each branch.  At
                # this point inference is one study at a time, so applying it
                # to a main-only ``(1, 12)`` array turns every rank into zero
                # and destroys the ordering needed by AUC.  One model needs
                # no rank blend; sigmoid preserves its logit ordering.
                scores = to_probabilities(main_logits)[0]
            else:
                branches = {"main": main_logits}
                cluster_id = specialist_router(study_uid, cluster_assignments)
                specialist = self._load_checkpoint(
                    specialist_path,
                    lambda: self.specialist_factory(cluster_id, CLASSICAL_IMAGE_SIZE),
                )
                branches["specialist"] = self._predict_one(
                    specialist, study_uid, volume_index, labels_df
                )
                scores = rank_mean_logits(branches, CLASSICAL_ENSEMBLE_WEIGHTS)[0]
            records.append({"StudyInstanceUID": study_uid, **dict(zip(LABEL_COLUMNS, scores))})
        return pd.DataFrame(records, columns=["StudyInstanceUID", *LABEL_COLUMNS])

    def _specialist_path(
        self,
        model_paths: dict,
        study_uid: str,
        cluster_assignments: pd.DataFrame,
    ) -> Path | None:
        try:
            cluster_id = specialist_router(study_uid, cluster_assignments)
        except KeyError:
            return None
        specialists = model_paths.get("specialists", {})
        path = specialists.get(cluster_id, model_paths.get("specialist"))
        return Path(path) if path else None

    def _load_checkpoint(
        self,
        path: Path,
        factory: Callable[[], KneeModel],
    ) -> torch.nn.Module:
        if path in self._model_cache:
            return self._model_cache[path]
        model = factory().to(self.device)
        checkpoint = torch.load(path, map_location=self.device)
        state = checkpoint.get("ema_state_dict") or checkpoint["model_state_dict"]
        model.load_state_dict(state)
        model.eval()
        self._model_cache[path] = model
        return model

    def _predict_one(
        self,
        model: torch.nn.Module,
        study_uid: str,
        volume_index: dict,
        labels_df: pd.DataFrame,
    ) -> np.ndarray:
        dataset = self.dataset_factory([study_uid], labels_df, volume_index)
        slots, _, _, mask = self.collate_fn([dataset[0]])
        slot_tensor = torch.as_tensor(slots, device=self.device)
        slot_mask = torch.as_tensor(mask, device=self.device)
        with torch.no_grad():
            per_slot = []
            for slot_index, _ in enumerate(SLOT_NAMES):
                images = slot_tensor[:, slot_index].reshape(-1, *slot_tensor.shape[3:])
                logits = model.encode_slot(images).reshape(1, slot_tensor.shape[2], -1).mean(dim=1)
                per_slot.append(logits * slot_mask[:, slot_index : slot_index + 1])
            summed = torch.stack(per_slot).sum(dim=0)
            logits = summed / slot_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        return logits.cpu().numpy()


__all__ = ["PredictorProtocol", "ClassicalPredictor"]
