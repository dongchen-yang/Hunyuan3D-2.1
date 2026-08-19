#!/usr/bin/env bash
# Queue the alpha training campaign as a chain of 24-hour segments on fir.
#
# WHY A CHAIN. Full-node H100 wall-time tiers are backed by 100 / 100 / 90 / 60 / 30 nodes
# for 3 h / 12 h / 1 d / 3 d / 7 d. A 1-day ask sees 90 nodes; a 3-day ask sees 60. Asking
# for 24 h every time starts sooner on every segment, and the whole 50k-step run does not
# fit in any single tier we would actually want to queue for anyway.
#
# WHY afterany AND NOT afterok. A segment that hits its 24 h wall clock is KILLED -- SLURM
# records TIMEOUT and a non-zero exit. That is the NORMAL end of a segment here, not a
# failure. `afterok` would refuse to start the successor in exactly the case the chain
# exists for. The cost of `afterany` is that a genuinely broken segment (bad config, OOM at
# step 0) also lets its successors run; they will fail the same way in minutes rather than
# burning an allocation, and the sbatch's own gates catch the common causes before the GPU
# is touched. Watch the first segment rather than trusting the chain blindly.
#
# WHY ONE SHARED LIGHTGEN_RUN_TS. train.py auto-resumes from
# <logdir>/checkpoints/last.ckpt when the logdir already exists (train.py:190-194), and the
# logdir is stamped with LIGHTGEN_RUN_TS. Every segment must therefore carry the SAME stamp
# or it starts a fresh run at step 0 in a new directory -- silently, with no error. That is
# the single most dangerous way to lose a campaign here, which is why the stamp is computed
# once, here, and passed explicitly to every segment via --export.
#
# `max_steps` is global and lives in the checkpoint, so the segments march toward 50,000
# between them rather than each doing 50,000. A segment that starts after the target is
# already reached exits within a few minutes without training -- harmless, but it is why
# you should not queue many more segments than you need.
#
# DO NOT CHANGE THE CONFIG OR MODEL CODE WHILE A CHAIN IS IN FLIGHT.
# Slurm spools the batch script at SUBMIT time, so editing the .sbatch cannot affect queued
# segments -- but every segment runs `git pull --ff-only` when it starts, so a commit to
# cfgs/ or hunyuanpaintpbr/ lands on segment N+1 and silently trains the rest of the campaign
# under different settings than segment 1. If a config change is genuinely needed, cancel the
# remaining segments, then start a NEW stamp rather than mixing two recipes into one
# checkpoint lineage.
#
# Usage:
#   bash scripts/lightgen/submit_chain_fir.sh [N_SEGMENTS]        # default 4
#   bash scripts/lightgen/submit_chain_fir.sh 2 20260815-235959   # extend an EXISTING run
#
# Which campaign it drives is an env override, defaulting to the original alpha run:
#   TRAIN_SBATCH=scripts/lightgen/train_74k_alpha_nonzero_nocopy_fir.sbatch \
#   JOB_PREFIX=mvpaint_alpha_nonzero_nocopy \
#     bash scripts/lightgen/submit_chain_fir.sh 4
# (JOB_PREFIX names the segments "<prefix>_sN"; %x in the sbatch's --output follows it.)
#
set -euo pipefail

N=${1:-4}
SBATCH_FILE="${TRAIN_SBATCH:-$(dirname "$0")/train_74k_alpha_fir.sbatch}"
JOB_PREFIX="${JOB_PREFIX:-mvpaint_alpha}"
[ -f "$SBATCH_FILE" ] || { echo "ABORT: $SBATCH_FILE not found"; exit 1; }

# Second argument continues an existing campaign; without it a new stamp starts a new run.
TS=${2:-$(date +%Y%m%d-%H%M%S)}
if [ -n "${2:-}" ]; then
    echo "[chain] EXTENDING existing run LIGHTGEN_RUN_TS=$TS"
else
    echo "[chain] NEW run LIGHTGEN_RUN_TS=$TS"
fi
echo "[chain] $N x 24h segments from $SBATCH_FILE"

dep=""
ids=()
for i in $(seq 1 "$N"); do
    # --export=ALL keeps the submitting environment (module paths etc.) and adds the stamp.
    out=$(sbatch --parsable $dep \
                 --export=ALL,LIGHTGEN_RUN_TS="$TS" \
                 --job-name="${JOB_PREFIX}_s$i" \
                 "$SBATCH_FILE")
    jid=${out%%;*}
    ids+=("$jid")
    echo "[chain] segment $i: job $jid${dep:+  (after ${ids[-2]})}"
    dep="--dependency=afterany:$jid"
done

echo
echo "[chain] LIGHTGEN_RUN_TS=$TS   jobs: ${ids[*]}"
echo "[chain] record that stamp -- extending or resuming this campaign later REQUIRES it."
echo "[chain] watch:   squeue -u \$USER -o '%.10i %.18j %.2t %.11M %.11L %R'"
echo "[chain] cancel:  scancel ${ids[*]}"
