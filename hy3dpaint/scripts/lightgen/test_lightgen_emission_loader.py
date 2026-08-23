"""pytest for LightgenEmissionDataset.

Can be run as:
  python test_lightgen_emission_loader.py  (standalone)
  pytest test_lightgen_emission_loader.py   (if pytest installed)
"""

import json
import os
import sys
import tempfile

import numpy as np
import torch
import pytest
from PIL import Image

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.data.dataloader import lightgen_emission_loader
from src.data.dataloader.lightgen_emission_loader import LightgenEmissionDataset


FIXTURE_ROOT = os.environ.get(
    "LIGHTGEN_FIXTURE_ROOT",
    "/cs/3dlg-jupiter-project/lightgen/uv_voxel_pipeline/out_multiview_pilot10"
)
# Support override for testing skip path via env var; default to local JSON
_DEFAULT_JSON = os.path.join(os.path.dirname(__file__), "pilot10_local.json")
EXAMPLES_JSON = os.environ.get("LIGHTGEN_EXAMPLES_JSON", _DEFAULT_JSON)


def test_lightgen_emission_dataset_shapes():
    """Test that LightgenEmissionDataset produces correct shapes and dtypes."""
    if not os.path.exists(EXAMPLES_JSON):
        pytest.skip(f"fixture root absent: {EXAMPLES_JSON}")

    d = LightgenEmissionDataset(EXAMPLES_JSON, num_view=6, image_size=512)

    # Check dataset size
    assert len(d) == 10, f"Expected 10 examples, got {len(d)}"

    # Test item 0
    b0 = d[0]
    assert b0["images_cond"].shape == (2, 3, 512, 512), f"images_cond shape: {b0['images_cond'].shape}"

    for key in ["images_albedo", "images_mr", "images_alpha", "images_normal", "images_position", "images_emission"]:
        assert b0[key].shape == (6, 3, 512, 512), f"{key} shape: {b0[key].shape}"

    # Test item 9
    b9 = d[9]
    assert b9["images_cond"].shape == (2, 3, 512, 512)
    for key in ["images_albedo", "images_mr", "images_alpha", "images_normal", "images_position", "images_emission"]:
        assert b9[key].shape == (6, 3, 512, 512)

    # Check dtypes
    assert b0["images_cond"].dtype == torch.float32
    assert b0["images_albedo"].dtype == torch.float32
    assert b0["images_mr"].dtype == torch.float32
    assert b0["images_alpha"].dtype == torch.float32
    assert b0["images_normal"].dtype == torch.float32
    assert b0["images_position"].dtype == torch.float32
    assert b0["images_emission"].dtype == torch.float32

    # Check value ranges
    for key in ["images_cond", "images_albedo", "images_mr", "images_alpha", "images_normal", "images_position", "images_emission"]:
        assert b0[key].min() >= 0.0, f"{key} min too low: {b0[key].min()}"
        assert b0[key].max() <= 1.0, f"{key} max too high: {b0[key].max()}"

    # Check that at least one shape has non-zero emission
    found_emission = False
    for i in range(len(d)):
        if d[i]["images_emission"].max() > 0.0:
            found_emission = True
            break
    assert found_emission, "No non-zero emission found in any example"

    print("✓ All assertions passed")


def _make_synthetic_fixture(root, n_shapes, size=32):
    """Write n_shapes tiny but structurally complete shape dirs; return their absolute paths."""
    dirs = []
    for k in range(n_shapes):
        d = os.path.join(root, f"shape{k:03d}")
        os.makedirs(os.path.join(d, "render_tex"), exist_ok=True)
        os.makedirs(os.path.join(d, "render_cond"), exist_ok=True)
        px = Image.fromarray(np.full((size, size, 3), k % 256, np.uint8))
        for v in range(6):
            for s in ("albedo", "emission", "mr", "alpha", "normal", "pos"):
                px.save(os.path.join(d, "render_tex", f"{v:03d}_{s}.png"))
        px.save(os.path.join(d, "render_cond", "000.png"))
        dirs.append(d)
    return dirs


def _write_json(path, dirs):
    with open(path, "w") as f:
        json.dump(dirs, f)
    return path


def test_corrupt_example_is_substituted():
    """One undecodable PNG must yield a valid substitute sample, not an exception.

    This is the guard that keeps a single bad file from killing a multi-day run.
    """
    with tempfile.TemporaryDirectory() as tmp:
        dirs = _make_synthetic_fixture(tmp, 8)
        # Truncate one file of shape 0 -- PIL opens the header fine and dies inside .load().
        victim = os.path.join(dirs[0], "render_tex", "003_emission.png")
        with open(victim, "r+b") as f:
            f.truncate(40)
        lightgen_emission_loader._REPORTED_BAD.clear()

        d = LightgenEmissionDataset(_write_json(os.path.join(tmp, "ex.json"), dirs),
                                    num_view=6, image_size=64)
        b = d[0]
        assert b["name"] != dirs[0], "expected a substitute example, got the corrupt one"
        assert b["images_emission"].shape == (6, 3, 64, 64)
        assert dirs[0] in lightgen_emission_loader._REPORTED_BAD, "the bad shape was not reported"
        # Reported once: a second visit must not re-add (set membership is the once-ness contract).
        n_reported = len(lightgen_emission_loader._REPORTED_BAD)
        d[0]
        assert len(lightgen_emission_loader._REPORTED_BAD) == n_reported


def test_systemic_corruption_raises():
    """If everything is unreadable the loader must fail loudly, not resample forever."""
    with tempfile.TemporaryDirectory() as tmp:
        dirs = _make_synthetic_fixture(tmp, 5)
        for d_ in dirs:
            with open(os.path.join(d_, "render_tex", "000_albedo.png"), "r+b") as f:
                f.truncate(40)
        lightgen_emission_loader._REPORTED_BAD.clear()
        ds = LightgenEmissionDataset(_write_json(os.path.join(tmp, "ex.json"), dirs),
                                     num_view=6, image_size=64)
        with pytest.raises(RuntimeError, match="distinct examples failed to load"):
            ds[0]


def _make_synthetic_thumbs(root, dirs, size=(48, 24), seed=0):
    """A non-square RGB thumbnail per shape dir, named <basename(dir)>.png under root.
    Non-square on purpose: upstream's load_image squashes to a square, and the test checks
    that squash, not a no-op."""
    os.makedirs(root, exist_ok=True)
    rng = np.random.default_rng(seed)
    for d in dirs:
        px = Image.fromarray(rng.integers(0, 256, (size[1], size[0], 3), dtype=np.uint8))
        px.save(os.path.join(root, os.path.basename(d) + ".png"))
    return root


def _upstream_squash(path, size):
    """Exactly what BaseDataset.load_image does to an RGB file: plain resize, /255, CHW."""
    im = Image.open(path).resize((size, size))
    return torch.from_numpy(np.asarray(im, np.float32) / 255.0).permute(2, 0, 1)


def test_thumbnail_ref_loads_thumbnail_not_render_cond():
    with tempfile.TemporaryDirectory() as tmp:
        dirs = _make_synthetic_fixture(tmp, 3)
        thumbs = _make_synthetic_thumbs(os.path.join(tmp, "thumbs"), dirs)
        lightgen_emission_loader._REPORTED_BAD.clear()
        ds = LightgenEmissionDataset(_write_json(os.path.join(tmp, "ex.json"), dirs), num_view=6,
                                     image_size=64, ref_source="thumbnail", thumbnail_dir=thumbs)
        b = ds[1]
        assert b["images_cond"].shape == (2, 3, 64, 64)
        expected = _upstream_squash(os.path.join(thumbs, "shape001.png"), 64)
        assert torch.equal(b["images_cond"][0], expected), "slot 0 is not the squashed thumbnail"
        assert torch.equal(b["images_cond"][0], b["images_cond"][1]), "slot 1 must equal slot 0"
        frontal = ds._img(os.path.join(dirs[1], "render_cond", "000.png"))
        assert not torch.equal(b["images_cond"][0], frontal), "thumbnail mode still used render_cond"
        assert b["images_albedo"].shape == (6, 3, 64, 64)  # the six views are untouched


def test_default_mode_is_frontal_albedo_unchanged():
    with tempfile.TemporaryDirectory() as tmp:
        dirs = _make_synthetic_fixture(tmp, 2)
        ds = LightgenEmissionDataset(_write_json(os.path.join(tmp, "ex.json"), dirs),
                                     num_view=6, image_size=64)
        assert ds.ref_source == "frontal_albedo" and ds.augment_ref is False
        b = ds[0]
        assert torch.equal(b["images_cond"][0],
                           ds._img(os.path.join(dirs[0], "render_cond", "000.png")))


def test_augment_ref_uses_upstream_augment_image(monkeypatch):
    import random as _random
    with tempfile.TemporaryDirectory() as tmp:
        dirs = _make_synthetic_fixture(tmp, 2)
        thumbs = _make_synthetic_thumbs(os.path.join(tmp, "thumbs"), dirs)
        js = _write_json(os.path.join(tmp, "ex.json"), dirs)
        plain = LightgenEmissionDataset(js, num_view=6, image_size=64, ref_source="thumbnail",
                                        thumbnail_dir=thumbs)
        aug = LightgenEmissionDataset(js, num_view=6, image_size=64, ref_source="thumbnail",
                                      thumbnail_dir=thumbs, augment_ref=True)
        assert aug.augment_ref is True
        # Coin flips, in call order: bg draw (<0.6 gray) | identity (<0.5 return as-is) |
        # rotate (<0.3) | scale (<0.5) | translate (<0.5) | perspective (<0.3).
        # 1) identity branch -> byte-identical to the unaugmented reference.
        monkeypatch.setattr(_random, "random", lambda: 0.0)
        assert torch.equal(aug[0]["images_cond"][0], plain[0]["images_cond"][0])
        # 2) non-identity, rotation only. The angle comes from random.uniform -- the module
        #    RNG, which the monkeypatch does not touch -- so seed it for determinism.
        seq = iter([0.0, 0.9, 0.1, 0.9, 0.9, 0.9] * 4)
        monkeypatch.setattr(_random, "random", lambda: next(seq))
        _random.seed(0)
        ref = aug[0]["images_cond"][0]
        base = plain[0]["images_cond"][0]
        assert ref.shape == (3, 64, 64) and float(ref.min()) >= 0.0 and float(ref.max()) <= 1.0
        assert float((ref - base).abs().max()) > 0.1, "augment_ref=True produced an unaugmented reference"


def test_missing_thumbnail_is_a_bad_example_not_a_fallback():
    with tempfile.TemporaryDirectory() as tmp:
        dirs = _make_synthetic_fixture(tmp, 6)
        thumbs = _make_synthetic_thumbs(os.path.join(tmp, "thumbs"), dirs[1:])  # shape000 has none
        lightgen_emission_loader._REPORTED_BAD.clear()
        ds = LightgenEmissionDataset(_write_json(os.path.join(tmp, "ex.json"), dirs), num_view=6,
                                     image_size=64, ref_source="thumbnail", thumbnail_dir=thumbs)
        b = ds[0]
        assert b["name"] != dirs[0], "expected a substitute example"
        assert dirs[0] in lightgen_emission_loader._REPORTED_BAD
        sub = _upstream_squash(os.path.join(thumbs, os.path.basename(b["name"]) + ".png"), 64)
        assert torch.equal(b["images_cond"][0], sub), "substitute must carry ITS thumbnail"


def test_empty_thumbnail_dir_raises_after_three():
    with tempfile.TemporaryDirectory() as tmp:
        dirs = _make_synthetic_fixture(tmp, 5)
        empty = os.path.join(tmp, "thumbs")
        os.makedirs(empty)
        lightgen_emission_loader._REPORTED_BAD.clear()
        ds = LightgenEmissionDataset(_write_json(os.path.join(tmp, "ex.json"), dirs), num_view=6,
                                     image_size=64, ref_source="thumbnail", thumbnail_dir=empty)
        with pytest.raises(RuntimeError, match="distinct examples failed to load"):
            ds[0]


def test_init_rejects_bad_reference_arguments():
    with tempfile.TemporaryDirectory() as tmp:
        dirs = _make_synthetic_fixture(tmp, 1)
        js = _write_json(os.path.join(tmp, "ex.json"), dirs)
        with pytest.raises(ValueError, match="ref_source"):
            LightgenEmissionDataset(js, ref_source="bogus")
        with pytest.raises(ValueError, match="thumbnail_dir"):
            LightgenEmissionDataset(js, ref_source="thumbnail")
        with pytest.raises(FileNotFoundError):
            LightgenEmissionDataset(js, ref_source="thumbnail", thumbnail_dir=os.path.join(tmp, "nope"))
        with pytest.raises(ValueError, match="augment_ref"):
            LightgenEmissionDataset(js, augment_ref=True)
        # thumbnail_dir without ref_source='thumbnail' must be rejected, never ignored:
        # ignoring it trains on render_cond/000.png while the caller believes otherwise.
        with pytest.raises(ValueError, match="thumbnail_dir is only used"):
            LightgenEmissionDataset(js, thumbnail_dir=os.path.dirname(js))



if __name__ == "__main__":
    test_lightgen_emission_dataset_shapes()
    test_corrupt_example_is_substituted()
    test_systemic_corruption_raises()
    print("✓ decode-guard tests passed")
    print("(thumbnail-reference tests need pytest: python -m pytest scripts/lightgen/test_lightgen_emission_loader.py)")
