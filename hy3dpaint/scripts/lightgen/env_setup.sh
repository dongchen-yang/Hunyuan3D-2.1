#!/usr/bin/env bash
# LightGen Task 2: `hunyuanpaint` conda env setup -- replayable install record.
#
# This is a RECORD of exactly what was installed and why, not just a convenience script.
# Every deviation from a naive `pip install -r requirements.txt` is called out below with
# the reason. Re-running this script from scratch should reproduce the same environment.
#
# Target: local RTX 4090 workstation now; this env is later replayed on venus (also
# Blackwell/Ada-class GPUs), hence the torch>=2.7 + cu128 pin (sm_120 Blackwell compat).
#
# Usage:
#   cd Hunyuan3DPaint-emissive
#   bash hy3dpaint/scripts/lightgen/env_setup.sh
#
set -euo pipefail

SUBMODULE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$SUBMODULE_ROOT"
mkdir -p log
TS="$(date +%Y%m%d-%H%M%S)"

# ---------------------------------------------------------------------------
# Deviation 0 (environment gotcha, not a package pin): this workstation's `/tmp`
# lives on the small root partition (99% full at time of writing, ~1.7G free) --
# `pip install torch` filled it and died with "No space left on device" mid-download.
# Redirect pip's TMPDIR to the scratch disk (hundreds of GB free) for every install
# in this script. Also export PYTHONNOUSERSITE=1: this machine has a stray
# ~/.local/lib/python3.10/site-packages (leftover from some earlier bare `pip install
# --user`) that Python's site module puts AHEAD of the conda env's own site-packages
# on sys.path for ANY python3.10 env -- without this, packages silently shadow the
# ones this script just installed. Both must be set for every `python`/`pip`
# invocation in this env, not just during install.
# Unconditional (not `${TMPDIR:-...}`) -- the whole point is this script must not
# depend on whatever TMPDIR the caller's shell happened to inherit from /tmp.
export TMPDIR=/local-scratch/localhome/dya78/tmp
mkdir -p "$TMPDIR"
export PYTHONNOUSERSITE=1

# ---------------------------------------------------------------------------
# TORCH_CUDA_ARCH_LIST: irrelevant today because nothing in this phase-1 setup
# compiles a CUDA extension (custom_rasterizer / DifferentiableRenderer are
# deliberately NOT built -- phase-1 training doesn't need them). Exported here
# anyway as the default for whoever later *does* need to compile: 8.9 covers
# this workstation's RTX 4090 (Ada), 12.0 covers venus's Blackwell target.
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9;12.0}"

# ---------------------------------------------------------------------------
# Step 1: presence-check, then create the env (global rule: check before install).
# NOTE: capture into a variable before grepping, not `conda env list | grep -q ...`
# directly -- under `pipefail`, `grep -q` exiting the instant it matches can send
# SIGPIPE to `conda env list` before it finishes writing, giving the pipeline a
# non-zero overall exit status (the SIGPIPE'd process, not the successful grep)
# even though the env *was* found -- which would silently flip this `if` to the
# wrong branch and then fail on `conda create` against an already-existing env.
existing_envs="$(conda env list)"
if grep -q '^hunyuanpaint ' <<<"$existing_envs"; then
    echo "conda env 'hunyuanpaint' already exists -- reusing (not validated: if it has"
    echo "the wrong python/torch version, delete it and rerun rather than debugging in place)."
else
    conda create -n hunyuanpaint python=3.10 -y
fi

# ---------------------------------------------------------------------------
# Step 2: torch + torchvision, cu128 wheels.
# NO flash-attn anywhere in this env (global constraint for this phase).
# Pinned to the exact versions resolved on 2026-07-30 (both newer than the
# >=2.7 floor the brief requires, which is fine per the brief -- "or newer" --
# but pinned exactly here since this file's job is to be *replayable*, not just
# "some torch>=2.7+cu128 pip happens to resolve today").
conda run -n hunyuanpaint bash -c \
    'pip install torch==2.11.0+cu128 torchvision==0.26.0+cu128 --index-url https://download.pytorch.org/whl/cu128' \
    2>&1 | tee "log/${TS}-hunyuanpaint-torch.log"

conda run -n hunyuanpaint bash -c \
    'python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"'

# ---------------------------------------------------------------------------
# Step 3: the rest of requirements.txt, minus torch/torchvision/torchaudio pins
# (those are handled above against the cu128 index instead of PyPI) and minus
# `bpy==4.0`.
#
# Deviation record (everything relaxed/dropped from requirements.txt, and why):
#   - `bpy==4.0` DROPPED, not relaxed. This version does not exist on PyPI --
#     the `bpy` project's first release is 4.2.0, and every `bpy` wheel ever
#     published (4.2.0 through 5.2.0, checked 2026-07-30) is cp311-only; there
#     has never been a cp310 wheel. Since Step 1 pins python=3.10 (per this
#     task's brief) `pip install bpy==4.0` cannot resolve under ANY torch
#     version -- this is a python-version conflict, not a torch conflict.
#     Confirmed unused by anything in the paint-pipeline load path: the only
#     importers in this submodule are `hy3dpaint/DifferentiableRenderer/mesh_utils.py`
#     (the DifferentiableRenderer subsystem this task's brief says NOT to build)
#     and `hy3dshape/tools/render/render.py` (the separate shape submodule, out
#     of scope for phase-1 paint/emission training). Safe to drop for this env.
#   - diffusers==0.30.0, pytorch-lightning==1.9.5, transformers==4.46.0,
#     accelerate==1.1.1, numpy==1.24.4 (downgraded from the 2.2.6 torch pulled
#     in) and every other exact pin in requirements.txt: installed AS PINNED,
#     no relaxation needed against torch 2.11.0+cu128. (This was the
#     anticipated risk per the task brief -- it did not materialize.)
#   - The two `--extra-index-url` mirrors at the top of requirements.txt
#     (Tencent/Aliyun) are intermittently flaky from this network -- pip logs
#     many "SSLError ... Retrying" warnings against them during Step 3. These
#     are non-fatal retries (pip falls through to PyPI / a working mirror);
#     the install still reports "Successfully installed ..." with every
#     package listed. Left the mirrors in place rather than stripping them --
#     harmless when reachable, and this is a replay record of what the
#     original repo ships, not a network config we own.
#   Full authoritative log: log/${TS}-hunyuanpaint-env.log
grep -viE '^(torch|torchvision|torchaudio)($|[=<>~])|^bpy==' requirements.txt > "$TMPDIR/req_notorch.txt"

conda run -n hunyuanpaint bash -c \
    "pip install -r '$TMPDIR/req_notorch.txt'" \
    2>&1 | tee "log/${TS}-hunyuanpaint-env.log"

# ---------------------------------------------------------------------------
# Step 4: smoke test -- strict-load the pretrained paint-pbr snapshot through the
# custom pipeline. Downloads a multi-GB HF snapshot on first run; let it finish.
conda run -n hunyuanpaint bash -c \
    'export PYTHONNOUSERSITE=1; python hy3dpaint/scripts/lightgen/smoke_load_pretrained.py' \
    2>&1 | tee "log/${TS}-hunyuanpaint-smoke.log"

echo "Done. See log/${TS}-hunyuanpaint-{torch,env,smoke}.log for the full record."
