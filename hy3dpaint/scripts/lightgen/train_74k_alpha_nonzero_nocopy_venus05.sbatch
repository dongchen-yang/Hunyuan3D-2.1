#!/bin/bash
# LightGen on cs-venus-05: CONTINUATION of the _nonzero_nocopy alpha run from the fir
# last.ckpt (~26,250 steps banked there), after fc10218 killed segments s2-s4 and the
# resubmitted chain sat >26 h in fir's whole-node H100 queue (2026-08-21, Dongchen's call).
#
# Recipe is IDENTICAL to the fir segments: bs4 x 4 GPUs x accumulate 1 = global batch 16,
# LR 7e-5, same 50k-step schedule; optimizer/scheduler state ride in the checkpoint, so
# the lineage continues -- only the hardware changes (H100-80G -> Pro6000 Blackwell 96G).
# Config: cfgs/lightgen-emission-74k-alpha-nonzero-nocopy-venus05.yaml (paths only vs fir).
#
# PREREQUISITES (gated below):
#   1. conda env `hunyuanpaint` at /localscratch/dya78/miniconda3 (the 20-channel 74k run's).
#   2. HF snapshots staged in /localscratch/dya78/lightgen_mvpaint/hf_home (same pinned rev).
#   3. Fixture /localscratch/dya78/lightgen_mvpaint/mv74k WITH the alpha maps -- 36
#      render_tex PNGs per shape. make_74k_jsons's A7 gate is the proof, run fresh below.
#   4. v2_shas/nonzero_nocopy_train_shas.txt + v2_{val,test}_shas.txt staged in
#      /localscratch/dya78/lightgen_mvpaint/v2_shas/ (md5-gated below).
#   5. The fir checkpoint staged at $STAGED_RESUME (sha256-verified against fir).
#
# Resume semantics: the FIRST run of this sbatch passes --resume $STAGED_RESUME (venus
# logdir does not exist yet). Any rerun finds <venus logdir>/checkpoints/last.ckpt and
# auto-resumes from it instead -- the staged fir ckpt is only the seed, so a requeue can
# never silently rewind the run to step 26,250.
#
# Node pin cs-venus-05: 6x Pro6000 Max-Q; yza440 holds 2 cards + 80 CPUs + 100G, so this
# takes 4 cards / 48 CPUs / 320G and leaves the co-tenant untouched (courtesy rule).
#SBATCH -w cs-venus-05
#SBATCH --gres=gpu:rtx_pro_6000_blackwell_max-q:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=320G
# ~23,750 remaining steps at ~4.1-4.5 s/step (the 20ch venus run measured 4.1 at bs2x2;
# same per-step sample count here) ≈ 28-30 h + 40 validations -> 40 h covers it without
# padding into the 3-day queue tier. last.ckpt every 1250 steps caps any timeout loss.
#SBATCH --time=1-16:00:00
#SBATCH --job-name=mvpaint_74k_alpha_nonzero_nocopy
#SBATCH --output=/localscratch/dya78/lightgen_mvpaint/%x-%j.out
set -euo pipefail

ROOT=/localscratch/dya78/lightgen_mvpaint
SHAS=$ROOT/v2_shas
STAGED_RESUME=$ROOT/fir_last_step26250.ckpt

echo "=== $(date -Is) host=$(hostname) job=$SLURM_JOB_ID ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# GPU-OCCUPANCY GATE (cf. fir fc10218 incident: a leaked process held 68.93 GiB and
# OOM'd two segments). On a SHARED node the co-tenant's cards legitimately show usage,
# so check only OUR cards: the ones in CUDA_VISIBLE_DEVICES when Slurm sets it, else --
# only if isolation shows exactly our 4 -- every visible card. (Codex review 2026-08-21:
# the earlier NGPU==4-only version silently skipped the gate in the 6-visible case.)
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
[ -d /localscratch/dya78/miniconda3/envs/hunyuanpaint ] || { echo "ABORT: no hunyuanpaint env"; exit 2; }
for f in nonzero_nocopy_train_shas.txt v2_val_shas.txt v2_test_shas.txt; do
    [ -s "$SHAS/$f" ] || { echo "ABORT: missing $SHAS/$f (PREREQUISITES 4)"; exit 2; }
done
TRAIN_MD5=$(md5sum "$SHAS/nonzero_nocopy_train_shas.txt" | cut -d' ' -f1)
[ "$TRAIN_MD5" = "b5a5d41699df7464f031b0fc5ba342cb" ] || {
    echo "ABORT: train sha list md5 $TRAIN_MD5 != recorded b5a5d41699df7464f031b0fc5ba342cb"; exit 2; }
[ -d "$ROOT/hf_home/hub/models--tencent--Hunyuan3D-2.1" ] || { echo "ABORT: HF snapshot missing (PREREQUISITES 2)"; exit 2; }
# dinov2 is loaded separately by the reference-image encoder (unet/model.py:119); staging
# only the Hunyuan repo looks complete until the constructor runs -- killed fir smoke 54933943.
[ -d "$ROOT/hf_home/hub/models--facebook--dinov2-giant" ] || { echo "ABORT: dinov2-giant snapshot missing (PREREQUISITES 2)"; exit 2; }

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
# Rebuilt each job (idempotent; fixture path is fixed on this node). The A7 gate (36
# render_tex PNGs per listed dir) is the proof the alpha maps are staged -- a 30-PNG
# pre-alpha fixture is rejected here rather than at the first training batch.
echo "=== building example JSONs ==="
python scripts/lightgen/make_74k_jsons.py \
    --train_shas  "$SHAS/nonzero_nocopy_train_shas.txt" \
    --val_shas    "$SHAS/v2_val_shas.txt" \
    --test_shas   "$SHAS/v2_test_shas.txt" \
    --val64_shas  scripts/lightgen/val64_v2_shas.txt \
    --fixture_root "$ROOT/mv74k" \
    --out_train   "$ROOT/mv74k_train_nonzero_nocopy.json" \
    --out_val64   "$ROOT/mv74k_val64.json" \
    --expect_train 63194

# --- train --------------------------------------------------------------------------------
export WANDB_MODE=offline
export WANDB_SERVICE_WAIT=300
export LIGHTGEN_WANDB_NAME='mvpaint_pbr->emission_74k_v2_alpha_nonzero_nocopy'
# SAME stamp as the fir segments -- this is a continuation, not a new run. The venus
# logdir name embeds the venus config/exp names, so it cannot collide with a fir dir.
export LIGHTGEN_RUN_TS=${LIGHTGEN_RUN_TS:-20260819-015735}
echo "LIGHTGEN_RUN_TS=$LIGHTGEN_RUN_TS"
mkdir -p logs

VENUS_LOGDIR="logs/lightgen-emission-74k-alpha-nonzero-nocopy-venus05-74k_alpha_nonzero_nocopy_venus05-${LIGHTGEN_RUN_TS}"
RESUME_ARGS=()
if [ -f "$VENUS_LOGDIR/checkpoints/last.ckpt" ]; then
    echo "[resume] venus last.ckpt exists -- auto-resume from $VENUS_LOGDIR"
else
    [ -f "$STAGED_RESUME" ] || { echo "ABORT: no venus last.ckpt AND no staged fir ckpt at $STAGED_RESUME"; exit 2; }
    # Verify the staged bytes ARE fir's last.ckpt (sha256 recorded at pull time,
    # 2026-08-21) -- a wrong-but-valid checkpoint would resume without any other error.
    STAGED_SHA=$(sha256sum "$STAGED_RESUME" | cut -d' ' -f1)
    [ "$STAGED_SHA" = "eb443d83dbe8acedde3ac4225ccc7cb4e35c2edc7ecbf54066fdc4c442a20961" ] || {
        echo "ABORT: staged fir ckpt sha256 $STAGED_SHA != recorded eb443d83…c442a20961"; exit 2; }
    echo "[resume] first venus segment -- seeding from staged fir ckpt $STAGED_RESUME (sha256 OK)"
    RESUME_ARGS=(--resume "$STAGED_RESUME")
fi

nvidia-smi --query-gpu=index,memory.used --format=csv -l 120 > "$ROOT/vram-alpha-nonzero-nocopy-$SLURM_JOB_ID.log" &

python train.py \
    --base cfgs/lightgen-emission-74k-alpha-nonzero-nocopy-venus05.yaml \
    --name 74k_alpha_nonzero_nocopy_venus05 \
    --logdir logs/ \
    --gpus 0,1,2,3 \
    "${RESUME_ARGS[@]}"

echo "=== $(date -Is) done, rc=$? ==="
