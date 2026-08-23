#!/bin/bash
# LightGen on cs-venus-05: pbr->emission multiview fine-tune WITH alpha, on the _agentic
# training split (36,255), REFERENCE IMAGE = the TexVerse thumbnail. A FRESH run.
# Config: cfgs/lightgen-emission-74k-alpha-thumb-agentic-venus05.yaml -- read its header.
# Spec:   lightgen docs/superpowers/specs/2026-08-23-hunyuan3d-paint-thumbnail-reference-design.md
# Plan:   lightgen docs/superpowers/plans/2026-08-23-hunyuan3d-paint-thumbnail-reference.md
#
# Built from train_74k_alpha_nonzero_nocopy_venus05.sbatch (the node, the gates, the GPU
# occupancy check) and train_74k_alpha_nonzero_nocopy_fir.sbatch (fresh start, chain of
# segments). Differences that matter:
#   * NO resume seed -- the first segment starts from the pretrained snapshot; any later
#     segment or requeue auto-resumes from <logdir>/checkpoints/last.ckpt because the logdir
#     (stamped with LIGHTGEN_RUN_TS) already exists. Same stamp on every segment or a segment
#     silently starts a fresh run at step 0 -- submit_chain_fir.sh computes it once.
#   * The train sha list and the thumbnails are fetched from the node's jupiter mount
#     (/3dlg-jupiter-project, writable, no /cs prefix on Solar nodes), so nothing has to be
#     staged from the workstation: agentic_train_shas.txt from trellis2_bw/lightgen_74k_newbake/
#     (md5-gated), thumbnails from dataset/thumbnails_emissive.tar (80,735 files, count-gated,
#     one byte anchor md5-gated).
#   * make_74k_jsons.py gets --thumbnail_dir, so A7 also proves every listed sha has a thumbnail.
#   * At the end (the segment that reaches max_steps, or any later no-op segment) the trough
#     checkpoints + config are rsynced to jupiter outputs/<run>/ckpts/ and sha256-verified there.
#
# PREREQUISITES already on the node from the _nonzero_nocopy run (gated below):
#   1. conda env `hunyuanpaint` at /localscratch/dya78/miniconda3.
#   2. HF snapshots in /localscratch/dya78/lightgen_mvpaint/hf_home (Hunyuan3D-2.1 pinned rev
#      0b94677654c57bb9a6b6845cd7b704ccf551d327 AND facebook/dinov2-giant).
#   3. Fixture /localscratch/dya78/lightgen_mvpaint/mv74k WITH the alpha maps (36 render_tex
#      PNGs per shape); make_74k_jsons's A7 gate is the proof, run fresh below.
#   4. v2_{val,test}_shas.txt in /localscratch/dya78/lightgen_mvpaint/v2_shas/.
#   5. The repo clone /localscratch/dya78/lightgen_mvpaint/Hunyuan3D-2.1-emissive on branch
#      lightgen (git pull --ff-only below brings the pushed loader/config/JSON-gate changes).
#
# Node pin cs-venus-05: 6x Pro6000 Max-Q; the co-tenant holds 2 cards, so this takes 4 cards /
# 48 CPUs / 320G (courtesy rule). 50,000 steps at the measured 5.8 s/step = ~80 h + 40
# validations; one 1-16:00 segment holds ~24k steps, so THREE segments (afterany chain) cover
# it with margin while staying out of the 3-day queue tier, as the previous venus-05 job did.
#SBATCH -w cs-venus-05
#SBATCH --gres=gpu:rtx_pro_6000_blackwell_max-q:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=320G
#SBATCH --time=1-16:00:00
#SBATCH --job-name=mvpaint_74k_alpha_thumb_agentic
#SBATCH --output=/localscratch/dya78/lightgen_mvpaint/%x-%j.out
set -euo pipefail

ROOT=/localscratch/dya78/lightgen_mvpaint
SHAS=$ROOT/v2_shas
J=/3dlg-jupiter-project/lightgen
CFG=cfgs/lightgen-emission-74k-alpha-thumb-agentic-venus05.yaml
NAME=74k_alpha_thumb_agentic_venus05
RUN='mvpaint_pbr->emission_74k_v2_alpha_thumb_agentic'
# md5 of agentic_train_shas.txt as built by the plan's Task 7 (36,255 lines). FILL IN before
# submitting; the gate below refuses to run while it is the marker.
AGENTIC_MD5=FILL_ME_FROM_TASK_7
EXPECT_TRAIN=36255
EXPECT_THUMBS=80735
ANCHOR=78e15dddf5314eebb1cf6473c894dda2.png
ANCHOR_MD5=2820b562140bbf73e9b031fd5aa53fa6

echo "=== $(date -Is) host=$(hostname) job=$SLURM_JOB_ID ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# GPU-OCCUPANCY GATE (ported unchanged from the _nonzero_nocopy venus-05 launcher): on a
# SHARED node the co-tenant's cards legitimately show usage, so check only OUR cards.
OURS=$(echo "${CUDA_VISIBLE_DEVICES:-}" | tr ',' ' ')
if [ -z "$OURS" ] && [ "$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)" -eq 4 ]; then
    OURS="0 1 2 3"
fi
if [ -n "$OURS" ]; then
    BUSY=""
    for g in $OURS; do
        USED=$(nvidia-smi -i "$g" --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
        [ "${USED:-0}" -gt 4096 ] && BUSY="$BUSY GPU$g=${USED}MiB"
    done
    if [ -n "$BUSY" ]; then
        echo "ABORT: allocated GPU(s) already occupied:$BUSY"; exit 4
    fi
    echo "[gate] allocated GPUs ($OURS) clean"
else
    echo "[gate] cannot identify our cards (no CUDA_VISIBLE_DEVICES, >4 visible) -- gate skipped"
fi

# --- gates --------------------------------------------------------------------------------
[ "$AGENTIC_MD5" != "FILL_ME_FROM_TASK_7" ] || { echo "ABORT: AGENTIC_MD5 not filled in (plan Task 7)"; exit 2; }
[ -d /localscratch/dya78/miniconda3/envs/hunyuanpaint ] || { echo "ABORT: no hunyuanpaint env"; exit 2; }
[ -d "$J" ] || { echo "ABORT: jupiter not mounted at $J"; exit 2; }
for f in v2_val_shas.txt v2_test_shas.txt; do
    [ -s "$SHAS/$f" ] || { echo "ABORT: missing $SHAS/$f (PREREQUISITES 4)"; exit 2; }
done
# The _agentic train list rides the jupiter mount; pin it by content, not by name.
mkdir -p "$SHAS"
[ -s "$SHAS/agentic_train_shas.txt" ] || cp "$J/trellis2_bw/lightgen_74k_newbake/agentic_train_shas.txt" "$SHAS/"
TRAIN_MD5=$(md5sum "$SHAS/agentic_train_shas.txt" | cut -d' ' -f1)
[ "$TRAIN_MD5" = "$AGENTIC_MD5" ] || { echo "ABORT: agentic_train_shas.txt md5 $TRAIN_MD5 != recorded $AGENTIC_MD5"; exit 2; }
[ "$(wc -l < "$SHAS/agentic_train_shas.txt")" -eq "$EXPECT_TRAIN" ] || { echo "ABORT: agentic_train_shas.txt is not $EXPECT_TRAIN lines"; exit 2; }
[ -d "$ROOT/hf_home/hub/models--tencent--Hunyuan3D-2.1" ] || { echo "ABORT: HF snapshot missing (PREREQUISITES 2)"; exit 2; }
[ -d "$ROOT/hf_home/hub/models--facebook--dinov2-giant" ] || { echo "ABORT: dinov2-giant snapshot missing (PREREQUISITES 2)"; exit 2; }

# --- thumbnails: the reference images, from jupiter's tar ---------------------------------
# Whole archive (12.6 GB, 2-3 min on the node's jupiter path), idempotent by count. Members
# are emissive_thumbnails/<sha>.png; the config's thumbnail_dir is the $ROOT/thumbnails link.
THUMBS=$ROOT/emissive_thumbnails
TAR=$J/dataset/thumbnails_emissive.tar
n_thumbs() { ls -U "$THUMBS" 2>/dev/null | wc -l; }
if [ "$(n_thumbs)" -ne "$EXPECT_THUMBS" ]; then
    [ -s "$TAR" ] || { echo "ABORT: $TAR missing"; exit 2; }
    echo "=== extracting thumbnails from $TAR (have $(n_thumbs), want $EXPECT_THUMBS) ==="
    t0=$SECONDS
    tar -xf "$TAR" -C "$ROOT"
    echo "extracted in $((SECONDS-t0))s"
fi
N_THUMB=$(n_thumbs)
[ "$N_THUMB" -eq "$EXPECT_THUMBS" ] || { echo "ABORT: $THUMBS holds $N_THUMB files, expected $EXPECT_THUMBS"; exit 3; }
A_MD5=$(md5sum "$THUMBS/$ANCHOR" | cut -d' ' -f1)
[ "$A_MD5" = "$ANCHOR_MD5" ] || { echo "ABORT: thumbnail anchor $ANCHOR md5 $A_MD5 != $ANCHOR_MD5 (not the TexVerse thumbnails the other baselines use)"; exit 3; }
ln -sfn emissive_thumbnails "$ROOT/thumbnails"
echo "[thumbs] $N_THUMB files at $THUMBS; anchor OK; $ROOT/thumbnails -> emissive_thumbnails"

source /localscratch/dya78/miniconda3/etc/profile.d/conda.sh
conda activate hunyuanpaint
cd $ROOT/Hunyuan3D-2.1-emissive
git pull --ff-only
cd hy3dpaint
export PYTHONNOUSERSITE=1
export TORCH_CUDA_ARCH_LIST="12.0"
export HF_HOME=$ROOT/hf_home
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# --- example JSONs ------------------------------------------------------------------------
# Rebuilt each job (idempotent). A7 proves the fixture holds the alpha maps AND, with
# --thumbnail_dir, that every listed sha has a thumbnail -- both before a batch is touched.
echo "=== building example JSONs ==="
python scripts/lightgen/make_74k_jsons.py \
    --train_shas  "$SHAS/agentic_train_shas.txt" \
    --val_shas    "$SHAS/v2_val_shas.txt" \
    --test_shas   "$SHAS/v2_test_shas.txt" \
    --val64_shas  scripts/lightgen/val64_v2_shas.txt \
    --fixture_root "$ROOT/mv74k" \
    --thumbnail_dir "$ROOT/thumbnails" \
    --out_train   "$ROOT/mv74k_train_agentic.json" \
    --out_val64   "$ROOT/mv74k_val64.json" \
    --expect_train "$EXPECT_TRAIN"

# --- train --------------------------------------------------------------------------------
export WANDB_MODE=offline
export WANDB_SERVICE_WAIT=300
export LIGHTGEN_WANDB_NAME="$RUN"
# Computed ONCE per campaign by submit_chain_fir.sh and passed via --export; the fallback
# only fires for a hand-submitted first segment. Every later segment MUST carry the same stamp.
export LIGHTGEN_RUN_TS=${LIGHTGEN_RUN_TS:-$(date +%Y%m%d-%H%M%S)}
echo "LIGHTGEN_RUN_TS=$LIGHTGEN_RUN_TS"
mkdir -p logs
LOGDIR="logs/lightgen-emission-74k-alpha-thumb-agentic-venus05-${NAME}-${LIGHTGEN_RUN_TS}"
if [ -f "$LOGDIR/checkpoints/last.ckpt" ]; then
    echo "[resume] $LOGDIR/checkpoints/last.ckpt exists -- train.py auto-resumes from it"
else
    echo "[fresh] no last.ckpt under $LOGDIR -- starting from the pretrained snapshot"
fi

nvidia-smi --query-gpu=index,memory.used --format=csv -l 120 > "$ROOT/vram-alpha-thumb-agentic-$SLURM_JOB_ID.log" &

set +e
python train.py \
    --base "$CFG" \
    --name "$NAME" \
    --logdir logs/ \
    --gpus 0,1,2,3
RC=$?
set -e
echo "=== $(date -Is) train.py exited rc=$RC ==="

# --- back the troughs up to jupiter (AGENTS.md: a run is not finished until its checkpoints
# are on jupiter and written down). Runs only when train.py RETURNS -- the segment that
# reaches max_steps, or a later no-op segment; a segment killed at its wall clock never gets
# here, which is fine: the next one does. Idempotent rsync; sha256 verified on the jupiter
# side (17.9 GB per trough, ~5 troughs -- budget 20-30 min).
if [ -d "$LOGDIR/checkpoints/troughs" ]; then
    JOUT="$J/outputs/$RUN"
    mkdir -p "$JOUT/ckpts"
    rsync -ah "$LOGDIR/checkpoints/troughs/" "$JOUT/ckpts/"
    cp -n "$CFG" "$JOUT/config.yaml" || true
    ( cd "$LOGDIR/checkpoints/troughs" && sha256sum trough-*.ckpt ) > "$JOUT/ckpts/SHA256SUMS.node"
    if ( cd "$JOUT/ckpts" && sha256sum -c SHA256SUMS.node ); then
        echo "[backup] troughs + config on jupiter at $JOUT, sha256 verified"
    else
        echo "[backup] WARNING: sha256 mismatch on jupiter -- re-run the rsync by hand"
    fi
fi
echo "=== $(date -Is) done, rc=$RC ==="
exit $RC
