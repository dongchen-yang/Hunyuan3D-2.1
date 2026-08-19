#!/bin/bash
# LightGen on fir: pbr->emission multiview fine-tune WITH the alpha condition, full H100 node,
# on the _nonzero_nocopy training split (auto-filter v2: 71,646 -> 63,194).
#
# Plan: docs/superpowers/plans/2026-08-19-filtered-baselines-retrain.md (parent repo, Session C);
# baseline recipe: docs/superpowers/plans/2026-08-15-hunyuan3d-paint-alpha-on-fir.md.
# Config: cfgs/lightgen-emission-74k-alpha-nonzero-nocopy.yaml -- read its header for the
# sizing and the learning-rate reasoning; none of that is repeated here. This sbatch is
# train_74k_alpha_fir.sbatch with the train sha list swapped (63,194 lines, md5-gated
# below), --expect_train 63194, and NO --allow_missing: the one unrenderable sha
# (810a644a...) has crosstab new_max == 0, so the zero-emission predicate already removed
# it from the list and a zero-missing build is the expected, enforced outcome.
#
# PREREQUISITES (each one is gated below, so a missing piece fails in seconds, not at hour 3):
#   1. env      -- bash scripts/lightgen/env_setup_fir.sh   (login node; needs internet)
#   2. weights  -- bash scripts/lightgen/stage_fir.sh   (login node; needs internet)
#                  TWO repos, not one: tencent/Hunyuan3D-2.1 (pinned revision
#                  0b94677654c57bb9a6b6845cd7b704ccf551d327) AND facebook/dinov2-giant,
#                  which the reference-image encoder pulls in separately at
#                  hunyuanpaintpbr/unet/model.py:119. Staging only the first looks complete
#                  until the model constructor runs -- it is what killed smoke 54933943.
#   3. data     -- 12 tars in ~/projects/rrg-msavva/dya78/lightgen_mv74k_alpha/,
#                  247,435 entries each
#   4. splits   -- nonzero_nocopy_train_shas.txt (TRAIN, 63,194 lines, md5
#                  b5a5d41699df7464f031b0fc5ba342cb -- gated below) plus the unchanged
#                  v2_{val,test}_shas.txt, all in /scratch/dya78/lightgen_mvpaint/v2_shas/.
#                  These come from the PARENT repo (the train list from
#                  data_processing/uv_voxel_pipeline/pin_filtered_split.py via Task 3 of the
#                  2026-08-19 plan; val/test from data_processing/multiview_render_pipeline/)
#                  and are NOT in this submodule, so they must be scp'd separately. They are
#                  the only source of split membership: make_74k_jsons.py is sha-driven
#                  precisely so that a directory enumeration of the extracted fixture -- which
#                  holds train AND val AND test -- cannot silently train on the evaluation set.
#
# RESOURCES. 1 GPU : 12 CPU : 288 GB is the H100 RGU ratio; a full node is 4 GPUs : 48 CPUs : 1,152,000 MB
# and this asks for exactly that (see the --mem note below). Over-requesting multiplies the RGU charge, under-utilising (<=30%
# GPU) is penalised, so a full-node ask has to actually use all four cards -- it does.
#
# WALL TIME: 24 HOURS, and the campaign is a CHAIN of these, not one long job.
# Full-node H100 tiers are 3 h / 12 h / 1 d / 3 d / 7 d, backed by 100 / 100 / 90 / 60 / 30
# nodes -- so a 1-day ask sees 90 nodes against the 3-day tier's 60 and starts sooner, every
# segment. 50k steps will not fit in one 24 h segment (~17k steps at the extrapolated
# ~5 s/step, less validation and the extract), so expect roughly FOUR segments; submit them
# as a dependency chain with scripts/lightgen/submit_chain_fir.sh rather than by hand.
#
# The cost of short segments is what is lost at each kill: SLURM does not warn the trainer,
# so everything since the last `last.ckpt` write dies with it. That is why the config
# checkpoints every 1250 steps (~1.7 h, 7% of a segment) instead of the venus run's 2500.
#
#SBATCH --account=rrg-msavva_gpu
#SBATCH --job-name=mvpaint_74k_alpha_nonzero_nocopy
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
# --mem=0 means "all the memory on the node", which is what a whole-node job wants and is
# immune to RealMemory changing under us.
#
# Do NOT write --mem=1152G here. The RGU ratio (1 GPU : 12 CPU : 288 GB) is decimal GB, so
# 4 GPUs is 288*4 = 1152 GB = 1,152,000 MB -- exactly this node's RealMemory. But Slurm
# reads the G suffix as GiB, making 1152G = 1,179,648 MB, i.e. 27 GB MORE than the node
# has. The job is then simply unschedulable and sbatch says only
#   allocation failure: Requested node configuration is not available
# with no hint that memory is the culprit. Verified with --test-only 2026-08-16:
# 1152G fails, 1152000M / 0 / 1100G all schedule.
#SBATCH --mem=0
#SBATCH --gres=gpu:h100:4
#SBATCH --output=/scratch/dya78/lightgen_mvpaint/%x-%j.out
#
# RESUME after a requeue or at the end of a segment -- same stamp, same logdir, picks up
# last.ckpt automatically (train.py:190-194 auto-sets --resume from
# <logdir>/checkpoints/last.ckpt when the logdir already exists):
#   LIGHTGEN_RUN_TS=<the original stamp> sbatch scripts/lightgen/train_74k_alpha_nonzero_nocopy_fir.sbatch
# Without the stamp a resubmission starts a FRESH run at step 0 and silently discards the
# segment you were continuing. `max_steps` is global and lives in the checkpoint, so each
# segment continues toward 50,000 rather than restarting the count.

set -euo pipefail

ROOT=/scratch/dya78/lightgen_mvpaint
TARS=~/projects/rrg-msavva/dya78/lightgen_mv74k_alpha
SHAS=$ROOT/v2_shas
EXPECT_SHAPES=72420

echo "=== $(date -Is) host=$(hostname) job=$SLURM_JOB_ID ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# --- gates ------------------------------------------------------------------------------
[ -d "$ROOT/env" ] || { echo "ABORT: no venv at $ROOT/env -- run env_setup_fir.sh"; exit 2; }
for f in nonzero_nocopy_train_shas.txt v2_val_shas.txt v2_test_shas.txt; do
    [ -s "$SHAS/$f" ] || { echo "ABORT: missing $SHAS/$f (see PREREQUISITES 4)"; exit 2; }
done
# The filtered train list is the campaign's one variable -- pin it by content, not name.
# md5 recorded in the parent repo's docs/status/checkpoints.md (Task 3, 2026-08-19).
TRAIN_MD5=$(md5sum "$SHAS/nonzero_nocopy_train_shas.txt" | cut -d' ' -f1)
[ "$TRAIN_MD5" = "b5a5d41699df7464f031b0fc5ba342cb" ] || {
    echo "ABORT: nonzero_nocopy_train_shas.txt md5 $TRAIN_MD5 != recorded b5a5d41699df7464f031b0fc5ba342cb"; exit 2; }
N_TAR=$(ls "$TARS"/shard_*.tar 2>/dev/null | wc -l)
[ "$N_TAR" -eq 12 ] || { echo "ABORT: found $N_TAR tars in $TARS, expected 12"; exit 2; }

module load StdEnv/2023 python/3.12 cuda/12.6
source "$ROOT/env/bin/activate"
export PYTHONNOUSERSITE=1
export HF_HOME=$ROOT/hf_home
# Compute nodes have no outbound internet; without these, from_pretrained blocks on a
# metadata call and dies with a connection error instead of using the staged snapshot.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# --- stage the fixture to node-local NVMe -------------------------------------------------
# /project is Lustre and training I/O against it is forbidden; 2.75M small files over Lustre
# would also be far slower than paying this extraction once per job. The node reports
# TmpDisk=7200000 (7.2 TB) against ~137 GB extracted, so space is not a concern.
echo "=== extracting fixture -> \$SLURM_TMPDIR ==="
t0=$SECONDS
mkdir -p "$SLURM_TMPDIR/mv74k"
for f in "$TARS"/shard_*.tar; do
    tar -xf "$f" -C "$SLURM_TMPDIR/mv74k" &
done
wait
N_DIR=$(ls "$SLURM_TMPDIR/mv74k" | wc -l)
echo "extracted $N_DIR shape dirs in $((SECONDS-t0))s (expect $EXPECT_SHAPES)"
[ "$N_DIR" -eq "$EXPECT_SHAPES" ] || { echo "ABORT: extract is short"; exit 3; }

# --- code -------------------------------------------------------------------------------
cd "$ROOT/Hunyuan3D-2.1-emissive"
git pull --ff-only
cd hy3dpaint

# --- example JSONs ------------------------------------------------------------------------
# Rebuilt every job because $SLURM_TMPDIR paths change. Its A7 gate requires 36 render_tex
# PNGs per listed dir, so this doubles as the proof that the extract landed whole and carries
# alpha -- a pre-alpha 30-PNG fixture is rejected here rather than at the first training batch.
# expect_train 63194 = the full filtered list; NO --allow_missing, because the one shape
# that never rendered (810a644a..., crosstab new_max == 0) is already dropped by the
# zero-emission predicate -- Task 3 asserts it is absent from the list, so a single
# missing dir here is a real staging failure, not the known exception.
echo "=== building example JSONs ==="
python scripts/lightgen/make_74k_jsons.py \
    --train_shas  "$SHAS/nonzero_nocopy_train_shas.txt" \
    --val_shas    "$SHAS/v2_val_shas.txt" \
    --test_shas   "$SHAS/v2_test_shas.txt" \
    --val64_shas  scripts/lightgen/val64_v2_shas.txt \
    --fixture_root "$SLURM_TMPDIR/mv74k" \
    --out_train   "$ROOT/mv74k_train_nonzero_nocopy.json" \
    --out_val64   "$ROOT/mv74k_val64.json" \
    --expect_train 63194

# --- train --------------------------------------------------------------------------------
# WANDB offline: compute nodes have no internet, and we keep no credential on a shared
# cluster. Sync from the workstation afterwards:
#   wandb sync -p LightGen $ROOT/.../logs/lightgen-emission-74k-alpha-nonzero-nocopy-74k_alpha_nonzero_nocopy_fir-<TS>/wandb/offline-run-*
export WANDB_MODE=offline
export LIGHTGEN_WANDB_NAME='mvpaint_pbr->emission_74k_v2_alpha_nonzero_nocopy'
# Computed ONCE here, not in train.py: lightning's DDP launcher re-execs train.py per rank
# and each re-exec must inherit the same value, or the four ranks write four logdirs.
export LIGHTGEN_RUN_TS=${LIGHTGEN_RUN_TS:-$(date +%Y%m%d-%H%M%S)}
echo "LIGHTGEN_RUN_TS=$LIGHTGEN_RUN_TS"

mkdir -p logs
nvidia-smi --query-gpu=index,memory.used --format=csv -l 120 > "$ROOT/vram-alpha-nonzero-nocopy-$SLURM_JOB_ID.log" &

# HOST-memory sampler. Smoke 54934387 was OOM-killed at a 1125 GB cgroup limit while sacct
# reported MaxRSS of 21.5 GB -- because sacct SAMPLES (JobAcctGatherFrequency, ~30 s) and a
# fast allocation spike falls between samples. nvidia-smi above watches the wrong memory for
# that failure. This samples the cgroup's own counter every 2 s, which is the number the OOM
# killer actually acts on, so a repeat tells us WHEN and HOW MUCH rather than just "Killed".
# cgroup v2 first, v1 as fallback; `memory.peak` exists on newer kernels and is the cheapest
# high-water mark there is.
# The cgroup path must be DERIVED, not guessed. The first version of this sampler read
# /sys/fs/cgroup/memory.current directly and logged 0 for the whole run: on cgroup v2 that
# path is the ROOT cgroup, which a Slurm step cannot read. The step's own cgroup is the one
# named in /proc/self/cgroup (v2 format: "0::/<relative path>"), so resolve it at startup.
( CGROOT=/sys/fs/cgroup
  REL=$(awk -F: '$1=="0"{print $3}' /proc/self/cgroup 2>/dev/null)
  CG="$CGROOT$REL"
  [ -r "$CG/memory.current" ] || CG=$CGROOT   # v1 or unreadable: fall back, log will show 0
  echo "sampler: cgroup=$CG readable=$([ -r "$CG/memory.current" ] && echo yes || echo NO)"
  n=0
  while true; do
      cur=$(cat $CG/memory.current 2>/dev/null || cat $CGROOT/memory/memory.usage_in_bytes 2>/dev/null || echo 0)
      pk=$(cat $CG/memory.peak 2>/dev/null || echo 0)
      rss=$(awk '/^MemAvailable:/{print $2}' /proc/meminfo)
      echo "$(date +%H:%M:%S) cgroup_cur_GB=$((cur/1073741824)) cgroup_peak_GB=$((pk/1073741824)) host_avail_GB=$((rss/1048576))"
      # Per-process breakdown every 5th sample. The cgroup total says HOW MUCH; this says
      # WHO -- 4 big ranks and 40 small workers look identical in the total and need
      # completely different fixes (model/DDP vs num_workers).
      if [ $((n % 5)) -eq 0 ]; then
          ps -eo rss=,comm=,args= --sort=-rss 2>/dev/null | head -6 \
            | awk '{printf "    rss_GB=%.1f %s\n", $1/1048576, substr($0, index($0,$2), 60)}'
          echo "    n_python=$(pgrep -c -f 'python train.py' 2>/dev/null || echo 0)"
      fi
      n=$((n+1))
      sleep 2
  done ) > "$ROOT/hostmem-$SLURM_JOB_ID.log" 2>&1 &

python train.py \
    --base cfgs/lightgen-emission-74k-alpha-nonzero-nocopy.yaml \
    --name 74k_alpha_nonzero_nocopy_fir \
    --logdir logs/ \
    --gpus 0,1,2,3

echo "=== $(date -Is) done, rc=$? ==="
