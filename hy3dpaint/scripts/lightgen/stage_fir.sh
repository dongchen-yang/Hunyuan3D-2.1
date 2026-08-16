#!/usr/bin/env bash
# Stage the pretrained weights this fork needs into fir's HF_HOME, and prove they load
# OFFLINE -- which is the only condition that matters, because compute nodes have no
# outbound internet.
#
# RUN ON A LOGIN NODE. That is the whole point: it is the only place with internet, so
# every from_pretrained() the trainer will make has to be satisfiable from this cache.
#
# THERE ARE TWO REPOS, NOT ONE. This script exists because the first fir smoke
# (job 54933943) died after passing every data gate, loading five of six pipeline
# components, and reaching the model constructor:
#
#   OSError: We couldn't connect to 'https://huggingface.co' ... facebook/dinov2-giant is
#   not the path to a directory containing a file named preprocessor_config.json
#
# `tencent/Hunyuan3D-2.1` is the paint pipeline; `facebook/dinov2-giant` is the reference
# image encoder, pulled in separately by `Dino_v2(...)` at
# hunyuanpaintpbr/unet/model.py:119 whenever `unet.use_dino` is set. Staging only the
# Hunyuan snapshot looks complete right up until the constructor runs. Grep before you
# trust this list:
#   grep -rn 'from_pretrained(' hunyuanpaintpbr/ | grep -o '"[^"]*/[^"]*"' | sort -u
#
# The Hunyuan revision is PINNED so fir and venus run the same weights rather than
# "whatever main was on the day each was staged"; cfgs/lightgen-emission-74k-alpha-fir.yaml
# hard-codes the resulting path, and pinning is what makes that safe.
#
set -euo pipefail

ROOT=/scratch/dya78/lightgen_mvpaint
HUNYUAN_REV=0b94677654c57bb9a6b6845cd7b704ccf551d327

module load StdEnv/2023 python/3.12 cuda/12.6
source "$ROOT/env/bin/activate"
export PYTHONNOUSERSITE=1
export HF_HOME=$ROOT/hf_home

echo "=== fetching (online; login node) ==="
python - <<PY
from huggingface_hub import snapshot_download
h = snapshot_download("tencent/Hunyuan3D-2.1", revision="$HUNYUAN_REV",
                      allow_patterns=["hunyuan3d-paintpbr-v2-1/*"])
print("HUNYUAN", h)
d = snapshot_download("facebook/dinov2-giant")
print("DINOV2 ", d)
PY

echo
echo "=== verifying they load OFFLINE (the condition the compute node runs under) ==="
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python - <<PY
import warnings; warnings.filterwarnings("ignore")
from transformers import AutoImageProcessor, AutoModel
p = AutoImageProcessor.from_pretrained("facebook/dinov2-giant")
m = AutoModel.from_pretrained("facebook/dinov2-giant")
n = sum(x.numel() for x in m.parameters()) / 1e6
print(f"dinov2-giant OK offline: {type(p).__name__} / {type(m).__name__}, {n:.0f}M params")
PY

SNAP=$HF_HOME/hub/models--tencent--Hunyuan3D-2.1/snapshots/$HUNYUAN_REV/hunyuan3d-paintpbr-v2-1
[ -d "$SNAP" ] || { echo "ABORT: $SNAP missing -- the config hard-codes this path"; exit 1; }
N_W=$(find "$SNAP" \( -name '*.safetensors' -o -name '*.bin' \) | wc -l)
N_INC=$(find "$HF_HOME" -name '*.incomplete' | wc -l)
echo "hunyuan snapshot: $N_W weight files, $(du -sh "$HF_HOME/hub/models--tencent--Hunyuan3D-2.1/blobs" | cut -f1) of blobs, $N_INC incomplete"
[ "$N_W" -ge 4 ] || { echo "ABORT: expected >=4 weight files, found $N_W"; exit 1; }
[ "$N_INC" -eq 0 ] || { echo "ABORT: $N_INC incomplete downloads"; exit 1; }

# `du -sh` on a snapshot dir reports the SYMLINKS (48K) and tells you nothing -- always
# measure blobs, or du -L. That reads like a truncated download and is not one.
echo
echo "STAGE_FIR_OK"
