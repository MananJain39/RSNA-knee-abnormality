"""DINOv2 backbone + per-diagnosis attention head for the RSNA knee model.

This is the main (not cluster-specialist) branch of the hierarchical pipeline.
Cluster specialists live in ``cluster_specialist.py`` and reuse the same
``PerDiagnosisHead`` so the rank-mean ensemble can blend the two output spaces
directly.

Inputs are 2.5D groups of 3 adjacent slices (see ``src/imaging/preprocessing``),
shape ``(B, 3, H, W)``.  Each slot is encoded by the same backbone and the
12-diagnosis attention head produces 12 logits per ``(B, group)``.  At
inference time the group logits are averaged (or rank-mean'd across slots) to
emit the final 12 probabilities per study.

The module is torch-optional: importing does not require torch, but
instantiating ``KneeModel`` does (used on the GPU side only).
"""

from __future__ import annotations

from src.config import (
    CLASSICAL_BATCH_SIZE,
    CLASSICAL_GRAD_ACCUM_STEPS,
    CLASSICAL_IMAGE_SIZE,
    CLASSICAL_LR,
    CLASSICAL_MAIN_BACKBONE,
    CLASSICAL_MAX_EPOCHS,
    CLASSICAL_N_GROUPS_PER_SLOT,
    CLASSICAL_SPECIALIST_BACKBONE,
    CLASSICAL_WARMUP_STEPS,
    CLASSICAL_WEIGHT_DECAY,
    LABEL_COLUMNS,
)
from src.imaging.slot_builder import SLOT_NAMES


def _import_torch():
    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:
        raise ImportError(
            "src.classical.dinov2_base requires torch and transformers. "
            "Install on GPU side: pip install torch transformers"
        ) from exc
    return torch, nn


def _import_transformers():
    try:
        from transformers import AutoModel
    except ImportError as exc:
        raise ImportError(
            "src.classical.dinov2_base requires transformers."
        ) from exc
    return AutoModel


class DINOv2Backbone:
    """Thin wrapper around the HuggingFace DINOv2 model.

    Forward returns the CLS token (``[B, D]``) and per-patch features
    (``[B, P, D]``) so the head can use either the global representation or
    attention over the patch tokens.
    """

    def __init__(
        self,
        model_name: str = CLASSICAL_MAIN_BACKBONE,
        freeze: bool = True,
        image_size: int = CLASSICAL_IMAGE_SIZE,
    ):
        torch, _ = _import_torch()
        AutoModel = _import_transformers()
        self.model_name = model_name
        self.image_size = image_size

        self.model = AutoModel.from_pretrained(model_name)

        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False

    @property
    def hidden_size(self) -> int:
        return int(self.model.config.hidden_size)

    def encode(self, pixel_values):
        """Run the backbone and return (cls, patch_tokens)."""
        torch, _ = _import_torch()
        outputs = self.model(pixel_values=pixel_values)
        last_hidden = outputs.last_hidden_state
        cls = last_hidden[:, 0]
        patches = last_hidden[:, 1:]
        return cls, patches


class PerDiagnosisHead:
    """Light per-diagnosis attention head.

    Each diagnosis gets its own attention query over the patch tokens,
    producing a single logit per diagnosis from the weighted patch sum.

    Output shape: ``[B, 12]`` (one logit per LABEL_COLUMNS).
    """

    def __init__(self, hidden_size: int, n_diagnoses: int = len(LABEL_COLUMNS)):
        torch, nn = _import_torch()
        self.n_diagnoses = n_diagnoses
        self.queries = nn.Parameter(torch.randn(n_diagnoses, hidden_size) * 0.02)
        self.scale = nn.Parameter(torch.tensor(float(hidden_size) ** -0.5))
        self.bias = nn.Parameter(torch.zeros(n_diagnoses))

    def __call__(self, patch_tokens):
        torch, _ = _import_torch()
        attn = torch.einsum("bpd,d->bdp", patch_tokens, self.queries) * self.scale
        weights = torch.softmax(attn, dim=-1)
        logits = torch.einsum("bdp,bpd->bd", weights, patch_tokens) + self.bias
        return logits


class KneeModel:
    """Composite model: DINOv2 backbone + per-diagnosis head.

    The model is shared across all 5 slots — only the image input changes
    per slot.  Outputs are accumulated by the trainer: ``logits_per_slot`` is
    shape ``(n_slots, n_groups, B, 12)`` and the inference pipeline averages
    the groups.
    """

    def __init__(
        self,
        backbone_name: str = CLASSICAL_MAIN_BACKBONE,
        image_size: int = CLASSICAL_IMAGE_SIZE,
        freeze_backbone: bool = True,
    ):
        self.backbone = DINOv2Backbone(
            model_name=backbone_name,
            freeze=freeze_backbone,
            image_size=image_size,
        )
        self.head = PerDiagnosisHead(self.backbone.hidden_size)
        self.image_size = image_size

    def encode_slot(self, pixel_values):
        """Encode a single ``(B, 3, H, W)`` batch (one slot's group batch)."""
        _, patches = self.backbone.encode(pixel_values)
        return self.head(patches)

    def parameter_groups(
        self,
        lr: float = CLASSICAL_LR,
        weight_decay: float = CLASSICAL_WEIGHT_DECAY,
    ):
        """Return parameter groups: backbone with one LR, head with another.

        Use with ``torch.optim.AdamW`` so the head learns faster than the
        (possibly frozen) backbone.
        """
        torch, _ = _import_torch()
        head_params = list(self.head.parameters()) if hasattr(self.head, "parameters") else []
        backbone_params = [
            p for p in self.backbone.model.parameters() if p.requires_grad
        ]

        groups = []
        if backbone_params:
            groups.append({
                "params": backbone_params,
                "lr": lr * 0.1,
                "weight_decay": weight_decay,
            })
        if head_params:
            groups.append({
                "params": head_params,
                "lr": lr,
                "weight_decay": weight_decay,
            })
        return groups


def expected_slot_layout(n_groups_per_slot: int = CLASSICAL_N_GROUPS_PER_SLOT) -> dict:
    """Return the expected per-slot shape grid for log/inspector code."""
    return {
        slot: n_groups_per_slot
        for slot in SLOT_NAMES
    }


def training_hyperparams() -> dict:
    """Snapshot the trainer-side hyperparameters in one place."""
    return {
        "main_backbone": CLASSICAL_MAIN_BACKBONE,
        "specialist_backbone": CLASSICAL_SPECIALIST_BACKBONE,
        "image_size": CLASSICAL_IMAGE_SIZE,
        "n_groups_per_slot": CLASSICAL_N_GROUPS_PER_SLOT,
        "batch_size": CLASSICAL_BATCH_SIZE,
        "grad_accum_steps": CLASSICAL_GRAD_ACCUM_STEPS,
        "lr": CLASSICAL_LR,
        "weight_decay": CLASSICAL_WEIGHT_DECAY,
        "warmup_steps": CLASSICAL_WARMUP_STEPS,
        "max_epochs": CLASSICAL_MAX_EPOCHS,
        "label_columns": list(LABEL_COLUMNS),
        "slots": list(SLOT_NAMES),
    }


__all__ = [
    "DINOv2Backbone",
    "PerDiagnosisHead",
    "KneeModel",
    "expected_slot_layout",
    "training_hyperparams",
]
