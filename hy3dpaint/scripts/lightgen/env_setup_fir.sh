#!/usr/bin/env bash
# LightGen: `hunyuanpaint` TRAINING-ONLY virtualenv on fir (Alliance Canada).
#
# Sibling of env_setup.sh, which is the conda record for the local workstation and venus.
# Like that file this is a REPLAYABLE RECORD, not a convenience wrapper: every deviation
# from requirements.txt is stated with its reason.
#
# Three things make this different from env_setup.sh:
#
#   1. NEVER conda on fir. Alliance supports virtualenv over its own wheelhouse and conda
#      performs worse there. `module load` + `virtualenv --no-download` + `pip --no-index`.
#
#   2. TRAINING-ONLY package set. requirements.txt has 43 entries; the multiview training
#      path imports 10 external packages. The rest exist for the gradio demo, mesh
#      post-processing and texture baking -- and they are exactly the ones least likely to
#      have an Alliance wheel (open3d, pymeshlab, cupy-cuda12x, rembg, realesrgan,
#      onnxruntime, pythreejs). Installing them would trade a 20-minute build for a day of
#      substitutions, for code the trainer never reaches.
#
#      The set was not guessed. It is the transitive import closure of train.py,
#      hunyuanpaintpbr/{model_emission,pipeline}.py, hunyuanpaintpbr/unet/{model,modules}.py,
#      src/data/objaverse_hunyuan.py and src/data/dataloader/lightgen_emission_loader.py,
#      following first-party imports and reporting external top-level names only. Adding
#      further entry points did not change the answer, which is the signal it is closed.
#
#      Two packages are in the list but NOT in that closure, and must not be dropped:
#        * wandb  -- pulled in at runtime by pytorch_lightning.loggers.WandbLogger, which
#                    train.py selects when LIGHTGEN_WANDB_NAME is set. A static scan cannot
#                    see it; the job dies at logger construction without it.
#        * pytest -- scripts/lightgen/test_lightgen_emission_loader.py.
#      accelerate / safetensors / huggingface_hub / tokenizers arrive transitively via
#      diffusers and transformers -- deliberately unpinned, let pip resolve them.
#
#   3. NO CUDA extensions are built, same as phase 1: no flash-attn, no custom_rasterizer,
#      no DifferentiableRenderer. H100 is sm_90 and stock torch wheels cover it, so unlike
#      venus (sm_120) there is no TORCH_CUDA_ARCH_LIST to set either.
#
# Usage (on a fir LOGIN node -- compute nodes have no outbound internet):
#   bash env_setup_fir.sh
#
set -euo pipefail

ROOT=/scratch/dya78/lightgen_mvpaint
mkdir -p "$ROOT"
cd "$ROOT"
mkdir -p log
TS="$(date +%Y%m%d-%H%M%S)"
LOG="log/${TS}-env-setup-fir.log"

# Alliance stack. cuda/12.6 rather than a newer one: torch 2.11.0's Alliance wheel is built
# against it, and loading a mismatched CUDA module is the classic way to get a torch that
# imports fine and then cannot see a GPU.
module load StdEnv/2023 python/3.12 cuda/12.6

# Same reasoning as env_setup.sh's Deviation 0: a stray ~/.local site-packages shadows the
# venv for any matching python version. Set it for install AND for every later invocation
# (the sbatch exports it too).
export PYTHONNOUSERSITE=1

{
    echo "=== env_setup_fir.sh $TS ==="
    python --version

    virtualenv --no-download env
    source env/bin/activate
    pip install --no-index --upgrade pip

    # torch 2.11.0 cp312 is in the Alliance wheelhouse (verified 2026-08-15 with
    # `avail_wheels torch --all-versions`); it is the same minor the venus conda env pins,
    # so the fork's code is running against a torch it has already been exercised on.
    pip install --no-index torch==2.11.0 torchvision

    # Versions pinned where requirements.txt pinned them. The unpinned ones float so the
    # wheelhouse can supply whatever build it has -- none of them is version-sensitive here.
    #
    # DEVIATION 1 (transformers 4.46.0 -> 4.46.2). The wheelhouse carries 4.46.2 and 4.46.3
    # but not 4.46.0. A patch bump inside the same minor; taken without ceremony.
    pip install --no-index \
        pytorch_lightning==1.9.5 \
        transformers==4.46.2 \
        numpy pillow einops omegaconf tqdm \
        wandb pytest

    # DEVIATION 2 (diffusers from PyPI, not the wheelhouse) -- the one real gap.
    # `avail_wheels diffusers --all-versions` offers 0.10.2 / 0.16.1 / 0.20.2 / 0.25.0 /
    # 0.32.2 / 0.37.x / 0.38.0. Our pin, 0.30.0, is in the hole between 0.25.2 and 0.32.2.
    #
    # Taking a neighbouring version is NOT safe here, which is why this is worth an internet
    # fetch. `hunyuanpaintpbr` does not merely call diffusers, it reaches inside it:
    # `UNet2p5DConditionModel` wraps `UNet2DConditionModel`, `attn_processor.py` replaces the
    # attention processors wholesale, and `modules.py` rebuilds `conv_in` in place. diffusers
    # refactored the attention-processor API and the UNet block plumbing more than once
    # between 0.30 and 0.32, so 0.32.2 is a version this fork has never been exercised
    # against -- and the failure mode is not a clean ImportError, it is subtly different
    # attention behaviour.
    #
    # Alliance's own guidance is --no-index first and PyPI when the wheelhouse lacks a
    # package, which is exactly this case. LOGIN NODES ONLY -- compute nodes have no
    # outbound internet, which is also why this must run at setup time and never from the
    # sbatch.
    pip install diffusers==0.30.0

    echo "=== versions ==="
    python - <<'PY'
import torch, torchvision, pytorch_lightning, diffusers, transformers
print("torch       ", torch.__version__, "cuda", torch.version.cuda)
print("torchvision ", torchvision.__version__)
print("lightning   ", pytorch_lightning.__version__)
print("diffusers   ", diffusers.__version__)
print("transformers", transformers.__version__)
PY
} 2>&1 | tee "$LOG"

echo
echo "wrote $ROOT/$LOG"
echo "NEXT: the import smoke test is the real gate --"
echo "  source $ROOT/env/bin/activate && cd $ROOT/Hunyuan3D-2.1-emissive/hy3dpaint &&"
echo "  python -c 'import hunyuanpaintpbr.model_emission;"
echo "             from src.data.dataloader.lightgen_emission_loader import LightgenEmissionDataset;"
echo "             print(\"imports OK\")'"
echo "An ImportError there names exactly the package this script missed. Add it HERE with a"
echo "comment saying why, do not pip-install it by hand and move on."
