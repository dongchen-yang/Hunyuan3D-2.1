#!/usr/bin/env bash
# Bring one Alliance cluster up for the released-dataset training: venv, weights, data.
# Run on the cluster's LOGIN node (it has the network; most compute nodes do not), from the
# clone at <scratch>/lightgen_mvpaint/Hunyuan3D-2.1-emissive (branch lightgen):
#
#   bash hy3dpaint/scripts/lightgen/stage_alliance.sh [env|weights|data|check|all]    # default all
#
# Every step is idempotent: it checks first and skips what is already there. `all` ends with
# STAGE_ALLIANCE_OK. Long step: `data` (37 GB from Hugging Face); run the script under nohup
# with its output in <root>/log/ and poll that file.
#
# env and weights are scripts/lightgen/env_setup_fir.sh and stage_fir.sh with the root made a
# variable -- read those two for why each package and each pin is what it is (training-only
# package set, diffusers 0.30.0 from PyPI, the pinned Hunyuan3D-2.1 revision, dinov2-giant).
# The Alliance wheelhouse and module tree are the same CVMFS stack on every cluster.
#
# data: the four representations are separate tar shards on the Hub; training needs the
# multiview and thumbnail ones (train 37 + 37, val 1 + 1; test 1 + 1 for inference later),
# plus splits.json and checksums.sha256. ~37 GB, 82 files -- no inode cost worth the name.
set -euo pipefail

STEP=${1:-all}
ROOT=$(mkdir -p "${LIGHTGEN_MVPAINT_ROOT:-$HOME/scratch/lightgen_mvpaint}" && cd -P "${LIGHTGEN_MVPAINT_ROOT:-$HOME/scratch/lightgen_mvpaint}" && pwd -P)
REPO=$ROOT/Hunyuan3D-2.1-emissive
DATA=$ROOT/release_hf
HUNYUAN_REV=0b94677654c57bb9a6b6845cd7b704ccf551d327
HF_REPO=3dlg-hcvc/LightgenBench
CHECKSUMS_SHA256=16ad1851c5ad9080ea1bede027c57fba8809619e0c1a5e0a89b1cf0d89c5a3bc
mkdir -p "$ROOT/log"
cd "$ROOT"

module load StdEnv/2023 gcc python/3.12 cuda/12.6 opencv/4.12.0
export PYTHONNOUSERSITE=1

step_env() {
    if [ -x env/bin/python ] && env/bin/python -c "import torch, torchvision, pytorch_lightning, diffusers, transformers, omegaconf, einops, wandb" 2>/dev/null; then
        echo "[env] present: $(env/bin/python -c 'import torch, diffusers; print("torch", torch.__version__, "diffusers", diffusers.__version__)')"
        return 0
    fi
    echo "=== [env] building $ROOT/env ==="
    virtualenv --no-download env
    source env/bin/activate
    pip install --no-index --upgrade pip
    pip install --no-index torch==2.11.0 torchvision
    pip install --no-index pytorch_lightning==1.9.5 transformers==4.46.2 numpy pillow einops omegaconf tqdm wandb pytest
    pip install diffusers==0.30.0          # not in the wheelhouse; see env_setup_fir.sh DEVIATION 2
    python - <<'PY'
import torch, torchvision, pytorch_lightning, diffusers, transformers
print("torch       ", torch.__version__, "cuda", torch.version.cuda)
print("torchvision ", torchvision.__version__)
print("lightning   ", pytorch_lightning.__version__)
print("diffusers   ", diffusers.__version__)
print("transformers", transformers.__version__)
PY
    deactivate
}

step_weights() {
    source env/bin/activate
    export HF_HOME=$ROOT/hf_home
    SNAP=$HF_HOME/hub/models--tencent--Hunyuan3D-2.1/snapshots/$HUNYUAN_REV/hunyuan3d-paintpbr-v2-1
    if [ -d "$SNAP" ] && [ -d "$HF_HOME/hub/models--facebook--dinov2-giant" ] && [ "$(find "$HF_HOME" -name '*.incomplete' | wc -l)" -eq 0 ]; then
        echo "[weights] present: $SNAP + dinov2-giant"
    else
        echo "=== [weights] fetching ==="
        python - <<PY
from huggingface_hub import snapshot_download
print("HUNYUAN", snapshot_download("tencent/Hunyuan3D-2.1", revision="$HUNYUAN_REV", allow_patterns=["hunyuan3d-paintpbr-v2-1/*"]))
print("DINOV2 ", snapshot_download("facebook/dinov2-giant"))
PY
    fi
    [ -d "$SNAP" ] || { echo "ABORT: $SNAP missing after the fetch"; exit 1; }
    N_W=$(find "$SNAP" \( -name '*.safetensors' -o -name '*.bin' \) | wc -l)
    N_INC=$(find "$HF_HOME" -name '*.incomplete' | wc -l)
    echo "[weights] hunyuan snapshot: $N_W weight files, $N_INC incomplete"
    [ "$N_W" -ge 4 ] && [ "$N_INC" -eq 0 ] || { echo "ABORT: weights incomplete"; exit 1; }
    # the condition the compute node runs under
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -c "
import warnings; warnings.filterwarnings('ignore')
from transformers import AutoModel
m = AutoModel.from_pretrained('facebook/dinov2-giant')
print(f'[weights] dinov2-giant loads offline: {sum(x.numel() for x in m.parameters())/1e6:.0f}M params')"
    deactivate
}

step_data() {
    if [ -f "$ROOT/release_hf.verified" ]; then echo "[data] verified earlier: $(cat "$ROOT/release_hf.verified")"; return 0; fi
    source env/bin/activate
    mkdir -p "$DATA"
    echo "=== [data] downloading $HF_REPO -> $DATA ==="
    hf download "$HF_REPO" --repo-type dataset --local-dir "$DATA" --max-workers 8 \
        --include "data/train/multiview/*" "data/train/thumbnail/*" \
                  "data/val/multiview/*"   "data/val/thumbnail/*" \
                  "data/test/multiview/*"  "data/test/thumbnail/*" \
                  "splits.json" "checksums.sha256" > "$ROOT/log/hf-download.out" 2>&1 \
        || { tail -5 "$ROOT/log/hf-download.out"; echo "ABORT: hf download failed (full output: $ROOT/log/hf-download.out)"; exit 1; }
    GOT=$(sha256sum "$DATA/checksums.sha256" | cut -d' ' -f1)
    [ "$GOT" = "$CHECKSUMS_SHA256" ] || { echo "ABORT: checksums.sha256 is $GOT, expected $CHECKSUMS_SHA256"; exit 1; }
    grep -E '  (data/(train|val|test)/(multiview|thumbnail)/|splits\.json$)' "$DATA/checksums.sha256" > "$ROOT/log/release_need.sha256"
    N=$(wc -l < "$ROOT/log/release_need.sha256")
    [ "$N" -eq 79 ] || { echo "ABORT: expected 79 lines to verify (78 tars + splits.json), got $N"; exit 1; }
    echo "=== [data] sha256 of $N files ==="
    ( cd "$DATA" && xargs -P 8 -d '\n' -I{} sh -c 'printf "%s\n" "$1" | sha256sum -c --quiet -' _ {} < "$ROOT/log/release_need.sha256" ) \
        || { echo "ABORT: at least one file does not match checksums.sha256"; exit 1; }
    rm -rf "$DATA/.cache"              # the downloader's per-file bookkeeping; the tars are verified
    echo "$(date -Is) $N files sha256-verified against checksums.sha256 $CHECKSUMS_SHA256" > "$ROOT/release_hf.verified"
    echo "[data] $(cat "$ROOT/release_hf.verified"); $(du -shL "$DATA" | cut -f1)"
    deactivate
}

step_check() {
    source env/bin/activate
    ( cd "$REPO/hy3dpaint" && python - <<'PY'
import sys
sys.path.insert(0, ".")
import hunyuanpaintpbr.model_emission  # noqa: F401
from src.data.dataloader.lightgen_emission_loader import LightgenEmissionDataset as D
import cv2
assert D.REF_SOURCES == ("frontal_albedo", "thumbnail"), D.REF_SOURCES
print("[check] model + loader import OK; cv2", cv2.__version__)
PY
    )
    ( cd "$REPO/hy3dpaint" && python -m pytest -q scripts/lightgen/test_relayout_release.py scripts/lightgen/test_make_74k_jsons.py 2>&1 | tail -2 )
    deactivate
}

case "$STEP" in
    env) step_env ;;
    weights) step_weights ;;
    data) step_data ;;
    check) step_check ;;
    all) step_env; step_weights; step_data; step_check; echo "STAGE_ALLIANCE_OK $(hostname) $ROOT" ;;
    *) echo "usage: stage_alliance.sh [env|weights|data|check|all]"; exit 2 ;;
esac
