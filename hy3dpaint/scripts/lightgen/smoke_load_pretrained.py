"""LightGen Task 2 smoke test: download the paint-pbr snapshot and strict-load it
through the custom `hunyuanpaintpbr` pipeline.

Answers, definitively:
  - does DiffusionPipeline.from_pretrained(<snapshot>, custom_pipeline=<hunyuanpaintpbr>)
    strict-load (no missing/unexpected keys)?
  - what wrapper class does pipe.unet come back as?
  - what is the inner UNet's conv_in shape (expect [320, 12, 3, 3])?
  - do learned_text_clip_albedo / learned_text_clip_mr exist?
  - is CUDA available, and does the pipeline move to it cleanly?

Prints `SNAPSHOT=<path>` on its own line — that path is `pretrained_snapshot_path`,
consumed by Task 7's config and Task 8's staging.

Run (from the Hunyuan3DPaint-emissive submodule root, hunyuanpaint conda env):
    conda run -n hunyuanpaint bash -c \
        'export PYTHONNOUSERSITE=1; python hy3dpaint/scripts/lightgen/smoke_load_pretrained.py'

PYTHONNOUSERSITE=1 is MANDATORY, not optional, for this specific workstation: it has a
stray ~/.local/lib/python3.10/site-packages (leftover from an old bare `pip install
--user`) that Python's site module puts AHEAD of the conda env's own site-packages on
sys.path for any python3.10 interpreter. Setting the env var from inside this script
would be too late (site processing already ran before `python` starts executing this
file) -- it must come from the launching shell, as in env_setup.sh and the command above.
"""
import os
import site
import sys

import torch
from huggingface_hub import snapshot_download

# hy3dpaint/ -- so `import hunyuanpaintpbr` would resolve if ever needed directly.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

if site.ENABLE_USER_SITE:
    print(
        "WARNING: user site-packages are enabled (site.ENABLE_USER_SITE=True). "
        "On this workstation that means ~/.local/lib/python3.10/site-packages can "
        "silently shadow this conda env's own packages. Launch with PYTHONNOUSERSITE=1 "
        "(see env_setup.sh) unless you've confirmed this machine doesn't have that dir.",
        file=sys.stderr,
    )

print(f"torch: {torch.__version__}  torch.version.cuda: {torch.version.cuda}  cuda_available: {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    print("BLOCKED: torch.cuda.is_available() is False -- cannot proceed with a GPU smoke test.")
    sys.exit(1)

# No explicit `revision=` pin: this follows the repo's current default branch, so a
# future upstream change to tencent/Hunyuan3D-2.1 could change what gets downloaded.
# The resolved commit is not lost, though -- huggingface_hub caches into
# .../snapshots/<commit_sha>/..., so the printed SNAPSHOT path below is itself the
# reproducibility record (observed 2026-07-30: commit 0b94677654c57bb9a6b6845cd7b704ccf551d327).
root = snapshot_download("tencent/Hunyuan3D-2.1", allow_patterns=["hunyuan3d-paintpbr-v2-1/*"])
snap = os.path.join(root, "hunyuan3d-paintpbr-v2-1")
print(f"SNAPSHOT={snap}")

from diffusers import DiffusionPipeline  # noqa: E402  (after sys.path insert)

custom_pipeline_dir = os.path.join(os.path.dirname(__file__), "..", "..", "hunyuanpaintpbr")
print(f"custom_pipeline dir: {os.path.abspath(custom_pipeline_dir)}")

pipe = DiffusionPipeline.from_pretrained(
    snap,
    custom_pipeline=custom_pipeline_dir,
    torch_dtype=torch.bfloat16,
)

print("pipeline class:", type(pipe).__name__, type(pipe).__module__)

unet = pipe.unet
inner = unet.unet if hasattr(unet, "unet") else unet

print("wrapper:", type(unet).__name__, "(module:", type(unet).__module__, ")")
print("inner unet:", type(inner).__name__, "(module:", type(inner).__module__, ")")
conv_in_shape = tuple(inner.conv_in.weight.shape)
print("conv_in:", conv_in_shape)
has_albedo_token = hasattr(inner, "learned_text_clip_albedo") or hasattr(unet, "learned_text_clip_albedo")
has_mr_token = hasattr(inner, "learned_text_clip_mr") or hasattr(unet, "learned_text_clip_mr")
print("has albedo token:", has_albedo_token)
print("has mr token:", has_mr_token)
print("use_ra:", getattr(unet, "use_ra", None))
print("use_dino:", getattr(unet, "use_dino", None))
print("use_learned_text_clip:", getattr(unet, "use_learned_text_clip", None))
print("pbr_setting:", getattr(unet, "pbr_setting", None))

# These are the four things this smoke test exists to answer definitively -- assert
# them (not just print) so a future regression (e.g. an upstream checkpoint change,
# or a diffusers upgrade that changes custom-component resolution) fails loudly with
# a non-zero exit code instead of silently printing a wrong value.
assert conv_in_shape == (320, 12, 3, 3), f"expected conv_in (320, 12, 3, 3), got {conv_in_shape}"
assert has_albedo_token, "learned_text_clip_albedo missing -- strict load should have caught this already"
assert has_mr_token, "learned_text_clip_mr missing -- strict load should have caught this already"

# Device/dtype sanity: move the whole pipeline to cuda and confirm the inner unet followed.
pipe = pipe.to("cuda")
conv_in_param = inner.conv_in.weight
print("post-.to(cuda) conv_in device/dtype:", conv_in_param.device, conv_in_param.dtype)
print("vae device/dtype:", next(pipe.vae.parameters()).device, next(pipe.vae.parameters()).dtype)
assert conv_in_param.device.type == "cuda", f"expected unet on cuda, got {conv_in_param.device}"
assert conv_in_param.dtype == torch.bfloat16, f"expected bfloat16, got {conv_in_param.dtype}"

print("SMOKE_TEST_OK")
