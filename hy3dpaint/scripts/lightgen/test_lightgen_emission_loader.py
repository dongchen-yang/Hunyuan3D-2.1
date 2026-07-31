"""pytest for LightgenEmissionDataset.

Can be run as:
  python test_lightgen_emission_loader.py  (standalone)
  pytest test_lightgen_emission_loader.py   (if pytest installed)
"""

import os
import sys
import torch
import pytest

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

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

    for key in ["images_albedo", "images_mr", "images_normal", "images_position", "images_emission"]:
        assert b0[key].shape == (6, 3, 512, 512), f"{key} shape: {b0[key].shape}"

    # Test item 9
    b9 = d[9]
    assert b9["images_cond"].shape == (2, 3, 512, 512)
    for key in ["images_albedo", "images_mr", "images_normal", "images_position", "images_emission"]:
        assert b9[key].shape == (6, 3, 512, 512)

    # Check dtypes
    assert b0["images_cond"].dtype == torch.float32
    assert b0["images_albedo"].dtype == torch.float32
    assert b0["images_mr"].dtype == torch.float32
    assert b0["images_normal"].dtype == torch.float32
    assert b0["images_position"].dtype == torch.float32
    assert b0["images_emission"].dtype == torch.float32

    # Check value ranges
    for key in ["images_cond", "images_albedo", "images_mr", "images_normal", "images_position", "images_emission"]:
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


if __name__ == "__main__":
    test_lightgen_emission_dataset_shapes()
