"""Tests for predict_emission_views.py that need no GPU and no weights: the seed derivation equals the
parent repository's, the training config decides the reference image, and the panel split returns the
predicted views in camera order. Whole-model equality with the parent predictor is the gate's job
(scripts/internal/gate/gate.sh, GATE_ENTRY=fork).

    python -m pytest scripts/lightgen/test_predict_emission_views.py -q
"""

import os
import sys

import pytest
import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import predict_emission_views as pev  # noqa: E402


def test_sample_seed_matches_the_parent_seedutil():
    # Values from the parent's evaluation/newdata_eval/seedutil.py (parent commit 6ffabdc).
    assert pev.sample_seed("030adbdafd824eb7b6cc85b5a77f6f52", 0) == 15235939539786358340
    assert pev.sample_seed("030adbdafd824eb7b6cc85b5a77f6f52", 1) == 12401582440519789579
    assert pev.sample_seed("057ad09b6f014f18ae90aeaad48248dc", 0) == 15800993649544425029


def _cfg(ref_source=None):
    params = {} if ref_source is None else {"ref_source": ref_source}
    return OmegaConf.create({"data": {"params": {"train": [{"params": params}]}}})


def test_thumbnail_config_needs_the_thumbnail_dir(tmp_path):
    with pytest.raises(ValueError, match="pass --thumbnail_dir"):
        pev.resolve_ref_source(_cfg("thumbnail"), None)
    with pytest.raises(ValueError, match="not a directory"):
        pev.resolve_ref_source(_cfg("thumbnail"), str(tmp_path / "absent"))
    assert pev.resolve_ref_source(_cfg("thumbnail"), str(tmp_path)) == ("thumbnail", str(tmp_path))


def test_frontal_albedo_config_refuses_a_thumbnail_dir(tmp_path):
    assert pev.resolve_ref_source(_cfg(), None) == ("frontal_albedo", None)
    with pytest.raises(ValueError, match="never saw"):
        pev.resolve_ref_source(_cfg("frontal_albedo"), str(tmp_path))
    with pytest.raises(ValueError, match="not in"):
        pev.resolve_ref_source(_cfg("albedo"), None)


def test_split_panel_returns_the_predictions_in_camera_order():
    # validation_step's panel: rearrange(cat([gt, pred], dim=-2), "(b n) c h w -> b c h (n w)").
    b, n, c, h, w = 2, 6, 3, 4, 5
    gt, pred = torch.rand(b * n, c, h, w), torch.rand(b * n, c, h, w)
    both = torch.cat([gt, pred], dim=-2)
    panel = both.reshape(b, n, c, 2 * h, w).permute(0, 2, 3, 1, 4).reshape(b, c, 2 * h, n * w)
    assert torch.equal(pev._split_panel(panel, n), pred.reshape(b, n, c, h, w))
