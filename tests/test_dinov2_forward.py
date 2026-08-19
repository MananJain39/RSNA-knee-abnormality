"""Tests for src.classical.dinov2_base — torch-gated end-to-end coverage.

These tests catch:
  - the ``nn.Module`` registration bug (parameters not seen by optimizers)
  - the per-diagnosis attention einsum collapse bug
  - state_dict round-trip correctness
  - parameter_groups() API correctness after the conversion

All tests are gated with ``pytest.importorskip("torch")`` so they degrade
gracefully when torch isn't installed. The HF DINOv2 backbone is mocked
with a small ``nn.Module`` stub so the tests don't depend on network access
to HuggingFace Hub.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


torch = pytest.importorskip("torch")
import torch.nn as nn

from src.classical.dinov2_base import (
    DINOv2Backbone,
    KneeModel,
    PerDiagnosisHead,
)


# === Mock helpers ===

class _StubBackbone(nn.Module):
    """Tiny stand-in for a HF DINOv2 model with the same output shape.

    Returns a 4D tensor with shape (B, 1 + P, D) where the first token is
    treated as CLS and the rest as patch tokens — matching the interface
    DINOv2Backbone.encode() expects.
    """

    def __init__(self, hidden_size: int = 16, n_patches: int = 5):
        super().__init__()
        self.config = type("Cfg", (), {"hidden_size": hidden_size})()
        self._proj = nn.Linear(3 * 16 * 16, hidden_size)
        self._n_patches = n_patches

    def forward(self, pixel_values):
        B = pixel_values.shape[0]
        flat = pixel_values.reshape(B, -1)
        tok = self._proj(flat).unsqueeze(1)
        patches = self._proj(flat).unsqueeze(1).repeat(1, self._n_patches, 1)
        out = type("O", (), {})()
        out.last_hidden_state = torch.cat([tok, patches], dim=1)
        return out


def _patched_backbone(hidden_size: int = 16, n_patches: int = 5):
    """Return a DINOv2Backbone whose inner HF model is a tiny stub."""
    def factory(model_name):
        return _StubBackbone(hidden_size=hidden_size, n_patches=n_patches)
    with patch("transformers.AutoModel.from_pretrained", side_effect=factory):
        return DINOv2Backbone(
            model_name="facebook/dinov2-small",
            freeze=True,
            image_size=16,
        )


# === PerDiagnosisHead (no backbone needed) ===

def test_per_diagnosis_head_is_nn_module():
    head = PerDiagnosisHead(hidden_size=16, n_diagnoses=12)
    assert isinstance(head, nn.Module)


def test_per_diagnosis_head_parameters_are_registered():
    head = PerDiagnosisHead(hidden_size=16, n_diagnoses=12)
    params = list(head.parameters())
    assert len(params) == 3
    assert sum(p.numel() for p in params) == 12 * 16 + 1 + 12


def test_per_diagnosis_head_forward_shape():
    head = PerDiagnosisHead(hidden_size=8, n_diagnoses=12)
    patch_tokens = torch.randn(2, 5, 8)
    logits = head(patch_tokens)
    assert logits.shape == (2, 12)
    assert not torch.isnan(logits).any()


def test_per_diagnosis_head_produces_distinct_logits():
    """Catches the einsum-collapse bug: every diagnosis must produce a
    distinct logit. With the bug, all 12 logits collapse to the same
    value plus per-diagnosis bias."""
    torch.manual_seed(0)
    head = PerDiagnosisHead(hidden_size=16, n_diagnoses=12)
    head.queries.data = torch.randn_like(head.queries.data)
    patch_tokens = torch.randn(4, 7, 16)
    logits = head(patch_tokens)
    std_per_row = logits.std(dim=1)
    assert (std_per_row > 1e-4).all(), (
        f"Per-diagnosis logits collapsed: std per row = {std_per_row.tolist()}"
    )


def test_per_diagnosis_head_state_dict_roundtrip():
    head = PerDiagnosisHead(hidden_size=8, n_diagnoses=12)
    patch_tokens = torch.randn(1, 4, 8)
    out_before = head(patch_tokens).clone()

    state = head.state_dict()
    head2 = PerDiagnosisHead(hidden_size=8, n_diagnoses=12)
    head2.load_state_dict(state)
    out_after = head2(patch_tokens)

    assert torch.allclose(out_before, out_after, atol=1e-6)


# === KneeModel + DINOv2Backbone (full pipeline, mocked HF) ===

def test_dinov2_backbone_is_nn_module():
    backbone = _patched_backbone()
    assert isinstance(backbone, nn.Module)


def test_knee_model_is_nn_module():
    with patch(
        "transformers.AutoModel.from_pretrained",
        side_effect=lambda name: _StubBackbone(),
    ):
        model = KneeModel(
            backbone_name="facebook/dinov2-small",
            image_size=16,
            freeze_backbone=True,
        )
    assert isinstance(model, nn.Module)


def test_knee_model_parameters_are_registered():
    """Catches the orphan-Parameter bug: optimizers see zero parameters
    when the head is not an nn.Module."""
    with patch(
        "transformers.AutoModel.from_pretrained",
        side_effect=lambda name: _StubBackbone(),
    ):
        model = KneeModel(
            backbone_name="facebook/dinov2-small",
            image_size=16,
            freeze_backbone=True,
        )

    head_params = list(model.head.parameters())
    backbone_params = [
        p for p in model.backbone.model.parameters() if p.requires_grad
    ]

    assert len(head_params) > 0, "Head parameters not registered"
    assert len(backbone_params) == 0, "Backbone should be frozen by default"


def test_knee_model_forward_shape():
    with patch(
        "transformers.AutoModel.from_pretrained",
        side_effect=lambda name: _StubBackbone(hidden_size=16, n_patches=5),
    ):
        model = KneeModel(
            backbone_name="facebook/dinov2-small",
            image_size=16,
            freeze_backbone=True,
        )
    model.eval()
    pixel_values = torch.randn(2, 3, 16, 16)
    with torch.no_grad():
        logits = model(pixel_values)
    assert logits.shape == (2, 12)


def test_knee_model_state_dict_roundtrip():
    with patch(
        "transformers.AutoModel.from_pretrained",
        side_effect=lambda name: _StubBackbone(hidden_size=16, n_patches=5),
    ):
        model = KneeModel(
            backbone_name="facebook/dinov2-small",
            image_size=16,
            freeze_backbone=True,
        )
    model.eval()
    pixel_values = torch.randn(1, 3, 16, 16)
    with torch.no_grad():
        out_before = model(pixel_values).clone()

    state = model.state_dict()
    with patch(
        "transformers.AutoModel.from_pretrained",
        side_effect=lambda name: _StubBackbone(hidden_size=16, n_patches=5),
    ):
        model2 = KneeModel(
            backbone_name="facebook/dinov2-small",
            image_size=16,
            freeze_backbone=True,
        )
    model2.load_state_dict(state)
    model2.eval()
    with torch.no_grad():
        out_after = model2(pixel_values)

    assert torch.allclose(out_before, out_after, atol=1e-5)


def test_knee_model_parameter_groups_returns_two_groups():
    with patch(
        "transformers.AutoModel.from_pretrained",
        side_effect=lambda name: _StubBackbone(),
    ):
        model = KneeModel(
            backbone_name="facebook/dinov2-small",
            image_size=16,
            freeze_backbone=False,
        )
    groups = model.parameter_groups()
    assert len(groups) == 2
    backbone_lrs = [g["lr"] for g in groups if g["lr"] < 1e-5]
    head_lrs = [g["lr"] for g in groups if g["lr"] >= 1e-5]
    assert len(backbone_lrs) == 1
    assert len(head_lrs) == 1
