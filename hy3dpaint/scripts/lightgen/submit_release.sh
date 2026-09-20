#!/usr/bin/env bash
# Submit the released-dataset training (4 x H100, one node) on ANY Alliance cluster.
# Run on the cluster's LOGIN node:
#
#   bash scripts/lightgen/submit_release.sh [--dry-run]
#
# It pulls the clone (login nodes have the network; most compute nodes do not), writes one job
# file per segment = a cluster-specific #SBATCH header + the run's constants + the shared body
# scripts/lightgen/train_release_body.sh, and submits them as an `afterany` chain. Each
# submission prints sbatch's "Submitted batch job N" line (compute-access hedge.sh reads it).
#
# Knobs (environment):
#   SEGMENTS=3           how many chained 24 h segments to queue (default 1)
#   STAMP=<ts>           the campaign's LIGHTGEN_RUN_TS. Default: a new one. REUSE it to extend or
#                        resume a campaign -- a new stamp starts a fresh run at step 0, silently.
#   DEPEND=<jobid>       the first segment waits for this job (afterany)
#   SEG_START=2          number the segments from here (names only)
#   SMOKE=1              bounded plumbing run: 60 steps, 1 h, its own stamp and name, checkpoints
#                        deleted at the end
#   WALLTIME=HH:MM:SS    default 24:00:00 (1 h for a smoke). 24 h is a queue decision: on fir it
#                        routes to the 90-node tier, 48/72 h to the 60-node one.
#   EXCLUDE=<node,...>   nodes to avoid
#   CLUSTER=<name>       override detection
#
# HEDGED RACE ACROSS CLUSTERS, then the chain on the winner:
#   every cluster:   STAMP=<ts> SEGMENTS=1 bash scripts/lightgen/submit_release.sh
#   the winner only: STAMP=<ts> SEGMENTS=2 SEG_START=2 DEPEND=<winner job> bash scripts/lightgen/submit_release.sh
# Only segment 1 is raced: under `afterany` a cancelled loser would release its successors.
#
# Per-cluster header facts: compute-access skill, alliance/overview.md (validated 2026-09-07 for
# the TRELLIS.2 launcher). trillium's sbatch is SciNet's wrapper: no --mem, no --gres,
# --gpus-per-node=N only, -t in minutes, a script file only.
set -euo pipefail

DRY=0; [ "${1:-}" = "--dry-run" ] && DRY=1
SMOKE=${SMOKE:-0}
SEGMENTS=${SEGMENTS:-1}
SEG_START=${SEG_START:-1}
DEPEND=${DEPEND:-}
EXCLUDE=${EXCLUDE:-}
STAMP=${STAMP:-$(date +%Y%m%d-%H%M%S)}
CLUSTER="${CLUSTER:-$(scontrol show config 2>/dev/null | awk -F'= ' '/^ClusterName/{print $2}')}"
# Physical paths: killarney's submit plugin rejects a cwd under /home, and ~/scratch is a
# symlink whose target differs per cluster (tamia: /scratch/d/dya78).
ROOT=$(cd -P "${LIGHTGEN_MVPAINT_ROOT:-$HOME/scratch/lightgen_mvpaint}" && pwd -P)
REPO=$ROOT/Hunyuan3D-2.1-emissive
DATA=$ROOT/release_hf
BODY=$REPO/hy3dpaint/scripts/lightgen/train_release_body.sh

if [ "$SMOKE" = 1 ]; then
    WALLTIME=${WALLTIME:-01:00:00}; NAME=release_alpha_thumb_smoke; JOB_PREFIX=mvpaint_relsmoke
    STAMP="smoke-$STAMP"; SEGMENTS=1
else
    WALLTIME=${WALLTIME:-24:00:00}; NAME=release_alpha_thumb; JOB_PREFIX=mvpaint_release
fi
RUN='mvpaint_pbr->emission_release_alpha_thumb'

# 4 GPUs of one node everywhere. CPUs and memory follow each cluster's per-GPU share.
GPU="#SBATCH --gres=gpu:h100:4"; CPUS=48; MEM="#SBATCH --mem=0"; TIME="#SBATCH --time=${WALLTIME}"
case "$CLUSTER" in
  fir)       ACCOUNT=rrg-msavva_gpu ;;                                  # whole node: 48 cpu, --mem=0 (the paper launcher's ask)
  nibi)      ACCOUNT=def-msavva_gpu; GPU="#SBATCH --gpus-per-node=h100:4"; MEM="#SBATCH --mem=960G" ;;      # half of 8 x H100, 112 cpu, 2000G
  rorqual)   ACCOUNT=def-msavva_gpu; GPU="#SBATCH --gpus-per-node=h100:4"; MEM="#SBATCH --mem=450G" ;;      # 4 x H100 nodes: 64 cpu, 510G
  killarney) ACCOUNT=aip-msavva;     GPU="#SBATCH --gpus-per-node=h100:4"; MEM="#SBATCH --mem=960G"; CPUS=24 ;;  # half of 8 x H100, 48 cpu, 2060G
  tamia)     ACCOUNT=aip-msavva;     MEM="#SBATCH --mem=450G" ;;        # whole 4 x H100 node: 48 cpu, 500000M
  grillium|trillium*) ACCOUNT=def-msavva; GPU="#SBATCH --gpus-per-node=4"; MEM=""
             TIME="#SBATCH -t $(( $(echo "$WALLTIME" | awk -F: '{print $1*60+$2}') ))" ;;   # minutes; whole node, memory fixed by the wrapper
  *) echo "unknown cluster '$CLUSTER' -- set CLUSTER=..."; exit 2 ;;
esac

# Login-node preflight: what the job would abort on, found before queueing.
fail() { echo "NOT READY on $CLUSTER: $1"; [ $DRY = 1 ] || exit 2; }
[ -x "$ROOT/env/bin/python" ] || fail "no venv at $ROOT/env (stage_alliance.sh env)"
[ -d "$ROOT/hf_home/hub/models--tencent--Hunyuan3D-2.1" ] && [ -d "$ROOT/hf_home/hub/models--facebook--dinov2-giant" ] \
    || fail "weight snapshots missing under $ROOT/hf_home (stage_alliance.sh weights)"
N_TAR=$( { ls "$DATA"/data/train/multiview/*.tar "$DATA"/data/train/thumbnail/*.tar "$DATA"/data/val/multiview/*.tar "$DATA"/data/val/thumbnail/*.tar 2>/dev/null || true; } | wc -l )
[ "$N_TAR" -eq 76 ] || fail "$N_TAR of 76 train/val multiview+thumbnail tars under $DATA (stage_alliance.sh data)"
[ -f "$ROOT/release_hf.verified" ] || fail "$ROOT/release_hf.verified missing: the tars were never sha256-checked (stage_alliance.sh data)"

# Update the code HERE (red line: never edit on the cluster, pull what was pushed), and pin the
# job to the commit it was submitted on.
cd -P "$REPO"
if [ $DRY = 0 ]; then
    git fetch -q origin lightgen && git checkout -q lightgen && git pull -q --ff-only origin lightgen \
        || { echo "ABORT: git pull --ff-only failed on $CLUSTER (local changes or a diverged branch) -- investigate, do not overwrite"; exit 2; }
fi
[ -z "$(git status --porcelain --untracked-files=no)" ] || { echo "ABORT: tracked files are modified in $REPO"; git status --short --untracked-files=no; exit 2; }
COMMIT=$(git rev-parse HEAD)
[ -f "$BODY" ] || { echo "ABORT: $BODY missing at $COMMIT"; exit 2; }

mkdir -p "$ROOT/log"
cd -P "$ROOT"            # killarney: submit from a physical /scratch cwd
dep=$DEPEND
for i in $(seq "$SEG_START" $(( SEG_START + SEGMENTS - 1 ))); do
    JOBNAME="${JOB_PREFIX}_s$i"; [ "$SMOKE" = 1 ] && JOBNAME=$JOB_PREFIX
    JOB="$ROOT/log/job_${NAME}_${STAMP}_s$i.sh"
    {
        echo "#!/bin/bash"
        echo "#SBATCH --account=$ACCOUNT"
        echo "#SBATCH --job-name=$JOBNAME"
        echo "$TIME"
        echo "#SBATCH --nodes=1"
        echo "#SBATCH --ntasks-per-node=1"
        echo "#SBATCH --cpus-per-task=$CPUS"
        [ -z "$MEM" ] || echo "$MEM"
        echo "$GPU"
        echo "#SBATCH --output=$ROOT/log/%x-%j.out"
        if [ "$SMOKE" != 1 ]; then echo "#SBATCH --mail-user=yangdongchen1@gmail.com"; echo "#SBATCH --mail-type=END"; fi
        [ -z "$dep" ] || echo "#SBATCH --dependency=afterany:$dep"
        [ -z "$EXCLUDE" ] || echo "#SBATCH --exclude=$EXCLUDE"
        echo "# ---- constants of this submission (written by submit_release.sh on $(date -Is)) ----"
        printf 'CLUSTER=%q\nROOT=%q\nREPO=%q\nDATA=%q\nSTAMP=%q\nCOMMIT=%q\nSMOKE=%q\nNAME=%q\nRUN=%q\nJOB_PREFIX=%q\n' \
            "$CLUSTER" "$ROOT" "$REPO" "$DATA" "$STAMP" "$COMMIT" "$SMOKE" "$NAME" "$RUN" "$JOB_PREFIX"
        echo "# ---- body: scripts/lightgen/train_release_body.sh at $COMMIT ----"
        cat "$BODY"
    } > "$JOB"
    if [ $DRY = 1 ]; then echo "[dry-run] wrote $JOB"; sed -n '1,/^# ---- body/p' "$JOB"; dep="<job of s$i>"; continue; fi
    out=$(sbatch "$JOB" 2>&1) || { echo "$out"; echo "ABORT: sbatch failed for segment $i on $CLUSTER"; exit 3; }
    echo "$out"
    jid=$(echo "$out" | grep -oE 'Submitted batch job [0-9]+' | grep -oE '[0-9]+' | tail -1)
    [ -n "$jid" ] || { echo "ABORT: no job id in sbatch's reply for segment $i"; exit 3; }
    echo "[submit] $CLUSTER segment $i: job $jid${dep:+ (after $dep)}  stamp=$STAMP commit=${COMMIT:0:7}"
    dep=$jid
done
[ $DRY = 1 ] || echo "[submit] LIGHTGEN_RUN_TS=$STAMP -- record it: extending or resuming this campaign REQUIRES it."
