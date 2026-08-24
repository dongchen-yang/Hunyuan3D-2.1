#!/bin/bash
# LightGen on FIR, CROSS-ACCOUNT: pbr->emission multiview fine-tune WITH alpha, on the
# _agentic training split (36,255), REFERENCE IMAGE = the TexVerse thumbnail. Submitted from
# Xingguang's account (rrg-angelx_gpu), not dya78's (rrg-msavva_gpu). A FRESH run.
# Config: cfgs/lightgen-emission-74k-alpha-thumb-agentic-fir-xg.yaml -- read its header first.
# Spec:   lightgen docs/superpowers/specs/2026-08-23-hunyuan3d-paint-thumbnail-reference-design.md
# Plan:   lightgen docs/superpowers/plans/2026-08-23-hunyuan3d-paint-thumbnail-reference.md
#
# WHY THIS SITE (Dongchen 2026-08-23/24, "不要管fir和solar了，我们就在fir-xg跑，记得用
# angel-rrg而不是msavva-rrg"): the dya78/rrg-msavva_gpu fir chain (56432310-12) sat PENDING
# (Priority) with no start in sight; the cs-venus-05 standby (248511-13) died SILENTLY within
# 1-2 seconds of each submission -- sacct shows CANCELLED by 0 on all three, no output log
# ever written. That is the known star2 bug in cluster-access skill
# references/star2/allocation.md ("Never pass sbatch --export=..."): submit_chain_fir.sh uses
# --export=ALL,LIGHTGEN_RUN_TS=..., which is fine on fir but fatally wrong on star2 -- so that
# standby was never a real fallback and nobody had noticed before 2026-08-24. Meanwhile
# rrg-angelx_gpu's fair-share priority measured ~2.6M vs rrg-msavva_gpu's ~1.9M for the
# identical TEXGen recipe (docs/status/checkpoints.md, "the fir cross-site run (collaborator
# account)") -- same discriminator applies here. This is an INDEPENDENT lineage from the
# dya78/rrg-msavva chain, not a resume of it; the dya78 chain is cancelled once this one is
# confirmed training, per the same precedent ("dya78's own fir copy was cancelled once the
# xya120 run was confirmed training").
#
# CROSS-ACCOUNT PLUMBING (2026-08-23, this session): everything the job only READS stays
# pointed at dya78's fir storage, now group-shared -- both users are in rrg-msavva:
#   - venv       /scratch/dya78/lightgen_mvpaint/env        chgrp rrg-msavva + chmod g+rX
#   - HF cache   /scratch/dya78/lightgen_mvpaint/hf_home    chgrp rrg-msavva + chmod g+rX
#   - sha lists  /scratch/dya78/lightgen_mvpaint/v2_shas     chgrp rrg-msavva + chmod g+rX
#   - fixture tars + thumbnail tar: already group rrg-msavva (staged straight to the shared
#     /project space by the original launcher's PREREQUISITES).
# Verified interactively before this launcher was written: xya120 can `ls`/`stat` into all of
# the above, and `source $VENV/bin/activate && python -c "import torch, pytorch_lightning,
# cv2"` printed 2.11.0 / 1.9.5 / 4.12.0 with no import error -- the shared venv's shared
# objects resolve fine cross-account (their RPATH does not depend on the caller's identity).
# Nothing dya78 needs write access to was touched; nothing xya120 needs write access to is
# shared -- $ROOT below is xya120's own /scratch, created and cloned into by hand before this
# script exists on fir (git clone -b lightgen, public GitHub URL, no deploy key needed).
#
# WHAT DIFFERS FROM THE dya78/fir TWIN (train_74k_alpha_thumb_agentic_fir.sbatch), and
# nothing else:
#   * --account=rrg-angelx_gpu, not rrg-msavva_gpu.
#   * ROOT is xya120's own /scratch/xya120/lightgen_mvpaint (code clone, logs, generated
#     JSONs, the thumbnails symlink) -- everything WRITABLE by the job.
#   * VENV and SHAS point at dya78's shared copies instead of living under $ROOT.
#   * TARS is the real /project/<gid> path, not `~/projects/rrg-msavva/...` and not dya78's
#     /home/dya78/projects/... symlink route either -- `~` would resolve to xya120's OWN home
#     (which holds nothing), and /home/dya78 itself is drwxr-x--- group `dya78` (dya78's own
#     private group), so xya120 cannot even traverse INTO the symlink that route depends on.
#     Found live: job 56538328's first real attempt aborted with "found 0 fixture tars" for
#     exactly this reason. /project/6049211 itself is drwxrws--- rrg-msavva -- go there directly.
#   * job-name carries an _xg suffix so squeue/sacct never conflate this with the dya78 chain.
#   * RUN keeps the SAME string ('mvpaint_pbr->emission_74k_v2_alpha_thumb_agentic') on
#     purpose -- once the dya78 chain is cancelled (per the precedent above) this becomes the
#     ONLY real lineage for that run name, and wandb/log messages should say so.
# GPU-occupancy gate, concurrency gate, cv2 preflight, fixture/thumbnail staging into
# $SLURM_TMPDIR, JSON building, training invocation, host-memory sampler, and the "AFTER THE
# RUN is manual" note are copied unchanged from the dya78/fir twin -- read that file's header
# for the reasoning behind each (wall-time tier choice, inode budget, the fc10218 leaked-GPU
# gate, wandb-core's 30 s port-file deadline, etc.).
#
# AFTER THE RUN -- THIS IS MANUAL, THE SCRIPT CANNOT DO IT. fir does not mount jupiter, so the
# troughs stay on fir /scratch. Pull <logdir>/checkpoints/troughs/ down to the workstation,
# rsync it up to jupiter outputs/<run>/ckpts/, sha256-verify BOTH ends, add the MANIFEST.md
# row and the docs/status/checkpoints.md entry. Because this runs under xya120's account, the
# workstation side needs SSH access to xya120@fir (or xya120 pushes to the tmux `fir-xg`
# session's reachable path) -- confirm the pull route before the run finishes, not after.
#SBATCH --account=rrg-angelx_gpu
#SBATCH --job-name=mvpaint_74k_alpha_thumb_agentic_xg
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --gres=gpu:h100:4
#SBATCH --output=/scratch/xya120/lightgen_mvpaint/%x-%j.out
#
# RESUME after a requeue or at a segment boundary -- SAME stamp, same logdir, picks up
# last.ckpt automatically. submit_chain_fir.sh computes the stamp once and exports it to
# every segment; a hand resubmission MUST pass it:
#   LIGHTGEN_RUN_TS=<the original stamp> sbatch scripts/lightgen/train_74k_alpha_thumb_agentic_fir_xg.sbatch
# Without it a resubmission starts a FRESH run at step 0 in a new directory, silently.
set -euo pipefail

ROOT=/scratch/xya120/lightgen_mvpaint
VENV=/scratch/dya78/lightgen_mvpaint/env
SHAS=/scratch/dya78/lightgen_mvpaint/v2_shas
# NOT /home/dya78/projects/rrg-msavva/... : that route requires traversing /home/dya78 itself,
# which is drwxr-x--- group `dya78` (dya78's own private group, not rrg-msavva) -- xya120 gets
# the "other" bits there, i.e. nothing, regardless of what the symlink inside points at.
# Confirmed 2026-08-24 (job 56538328's ABORT: 0 fixture tars found): the REAL absolute target
# of ~dya78/projects/rrg-msavva is /project/6049211 (drwxrws--- rrg-msavva) and
# /project/6049211/dya78 (drwxr-s--- rrg-msavva) -- both properly group-shared. Use that path
# directly so nothing depends on dya78's home directory.
TARS=/project/6049211/dya78/lightgen_mv74k_alpha
CFG=cfgs/lightgen-emission-74k-alpha-thumb-agentic-fir-xg.yaml
NAME=74k_alpha_thumb_agentic_fir_xg
RUN='mvpaint_pbr->emission_74k_v2_alpha_thumb_agentic'
AGENTIC_MD5=191a91bc295192517565eaac6089753b
EXPECT_TRAIN=36255
EXPECT_SHAPES=72420
EXPECT_THUMBS=80735
THUMB_TAR=$TARS/thumbnails_emissive.tar
THUMB_TAR_BYTES=12615577600
THUMB_TAR_MD5=04d4c4eacb0cd00701943fd8e5828a5a
ANCHOR=78e15dddf5314eebb1cf6473c894dda2.png
ANCHOR_MD5=2820b562140bbf73e9b031fd5aa53fa6
SNAP=/scratch/dya78/lightgen_mvpaint/hf_home/hub/models--tencent--Hunyuan3D-2.1/snapshots/0b94677654c57bb9a6b6845cd7b704ccf551d327/hunyuan3d-paintpbr-v2-1

echo "=== $(date -Is) host=$(hostname) job=$SLURM_JOB_ID user=$(whoami) account=rrg-angelx_gpu ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# GPU-OCCUPANCY GATE (unchanged from the dya78/fir twin).
BUSY=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
       | awk -F', ' '$2+0 > 4096 {printf "GPU%s=%sMiB ", $1, $2}')
if [ -n "$BUSY" ]; then
    echo "ABORT: leaked process(es) resident on this node before we start: $BUSY"
    echo "       Resubmit; consider sbatch --exclude=$(hostname -s)."
    exit 4
fi
echo "[gate] all 4 GPUs clean"

# --- gates ---------------------------------------------------------------------------------
[ -d "$VENV" ] || { echo "ABORT: no shared venv at $VENV -- has dya78 chgrp'd it to rrg-msavva?"; exit 2; }
[ -d "$ROOT/Hunyuan3D-2.1-emissive" ] || { echo "ABORT: no code clone at $ROOT/Hunyuan3D-2.1-emissive -- git clone -b lightgen it first"; exit 2; }
[ -d "$SNAP" ] || { echo "ABORT: pinned HF snapshot missing at $SNAP -- is dya78's hf_home chgrp'd to rrg-msavva?"; exit 2; }
[ -d "/scratch/dya78/lightgen_mvpaint/hf_home/hub/models--facebook--dinov2-giant" ] || { echo "ABORT: dinov2-giant snapshot missing"; exit 2; }
for f in agentic_train_shas.txt v2_val_shas.txt v2_test_shas.txt; do
    [ -s "$SHAS/$f" ] || { echo "ABORT: missing $SHAS/$f -- is dya78's v2_shas chgrp'd to rrg-msavva?"; exit 2; }
done
TRAIN_MD5=$(md5sum "$SHAS/agentic_train_shas.txt" | cut -d' ' -f1)
[ "$TRAIN_MD5" = "$AGENTIC_MD5" ] || { echo "ABORT: agentic_train_shas.txt md5 $TRAIN_MD5 != recorded $AGENTIC_MD5"; exit 2; }
[ "$(wc -l < "$SHAS/agentic_train_shas.txt")" -eq "$EXPECT_TRAIN" ] || { echo "ABORT: agentic_train_shas.txt is not $EXPECT_TRAIN lines"; exit 2; }
N_TAR=$( { ls "$TARS"/shard_*.tar 2>/dev/null || true; } | wc -l )
[ "$N_TAR" -eq 12 ] || { echo "ABORT: found $N_TAR fixture tars in $TARS, expected 12"; exit 2; }
[ -s "$THUMB_TAR" ] || { echo "ABORT: $THUMB_TAR missing"; exit 2; }
T_BYTES=$(stat -c %s "$THUMB_TAR" 2>/dev/null || echo 0)
[ "$T_BYTES" -eq "$THUMB_TAR_BYTES" ] || { echo "ABORT: $THUMB_TAR is $T_BYTES bytes, expected $THUMB_TAR_BYTES"; exit 2; }
echo "[gate] shared venv, both HF snapshots, 3 sha lists (train md5 OK), 12 fixture tars, thumbnail tar size"

# CONCURRENCY GATE (unchanged reasoning from the dya78/fir twin: $ROOT/thumbnails and the two
# example JSONs are shared per-root state a hand resubmission could clobber under a running
# segment).
OTHERS=$( { squeue -u "$USER" -h -n "$SLURM_JOB_NAME" -t R -o '%i' 2>/dev/null || true; } \
          | { grep -v "^${SLURM_JOB_ID}$" || true; } | tr '\n' ' ')
[ -z "$OTHERS" ] || { echo "ABORT: another segment of $SLURM_JOB_NAME is already RUNNING (job(s): $OTHERS)."; exit 5; }
echo "[gate] no other running segment of $SLURM_JOB_NAME"

# --- code ------------------------------------------------------------------------------------
cd "$ROOT/Hunyuan3D-2.1-emissive"
BRANCH=$(git rev-parse --abbrev-ref HEAD)
[ "$BRANCH" = "lightgen" ] || { echo "ABORT: xya120's clone is on branch '$BRANCH', expected lightgen"; exit 2; }
git pull --ff-only
cd hy3dpaint
[ -f "$CFG" ] || { echo "ABORT: $CFG not in the pulled clone"; exit 2; }
CFG_SNAP=$(awk -F': *' '/^[[:space:]]*pretrained_model_name_or_path:/ {print $2; exit}' "$CFG")
[ "$CFG_SNAP" = "$SNAP" ] || { echo "ABORT: $CFG names snapshot '$CFG_SNAP', but the gate checked '$SNAP'"; exit 2; }

module load StdEnv/2023 python/3.12 cuda/12.6 opencv/4.12.0
source "$VENV/bin/activate"
export PYTHONNOUSERSITE=1
export HF_HOME=/scratch/dya78/lightgen_mvpaint/hf_home
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# cv2 PREFLIGHT (unchanged from the dya78/fir twin -- test the code that will run, before
# either extract costs real time).
python - <<'PYCHK' || { echo "ABORT: the training env cannot import the loader (traceback above)."; exit 2; }
import sys
sys.path.insert(0, ".")
from src.data.dataloader.lightgen_emission_loader import LightgenEmissionDataset as D
assert D.REF_SOURCES == ("frontal_albedo", "thumbnail"), D.REF_SOURCES
import cv2, numpy as np
cv2.getPerspectiveTransform(np.float32([[0,0],[8,0],[8,8],[0,8]]),
                            np.float32([[1,1],[7,0],[8,7],[0,8]]))
print(f"[gate] loader imports OK; cv2 {cv2.__version__}, numpy {np.__version__}")
PYCHK

# --- stage the thumbnails to node-local NVMe, and repoint the config's static path ----------
echo "=== verifying thumbnail archive (md5 of 12.6 GB, ~2-3 min) ==="
t0=$SECONDS
T_MD5=$(md5sum "$THUMB_TAR" | cut -d' ' -f1)
[ "$T_MD5" = "$THUMB_TAR_MD5" ] || { echo "ABORT: $THUMB_TAR md5 $T_MD5 != recorded $THUMB_TAR_MD5"; exit 3; }
echo "md5 OK in $((SECONDS-t0))s"
echo "=== extracting thumbnails -> \$SLURM_TMPDIR ==="
t0=$SECONDS
tar -xf "$THUMB_TAR" -C "$SLURM_TMPDIR"
THUMBS=$SLURM_TMPDIR/emissive_thumbnails
N_THUMB=$( { ls -U "$THUMBS" 2>/dev/null || true; } | wc -l )
echo "extracted $N_THUMB thumbnails in $((SECONDS-t0))s (expect $EXPECT_THUMBS)"
[ "$N_THUMB" -eq "$EXPECT_THUMBS" ] || { echo "ABORT: $THUMBS holds $N_THUMB files, expected $EXPECT_THUMBS"; exit 3; }
A_MD5=$(md5sum "$THUMBS/$ANCHOR" 2>/dev/null | cut -d' ' -f1 || true)
[ "$A_MD5" = "$ANCHOR_MD5" ] || { echo "ABORT: thumbnail anchor $ANCHOR md5 '$A_MD5' != $ANCHOR_MD5"; exit 3; }
ln -sfn "$THUMBS" "$ROOT/thumbnails"
LINKED=$(readlink -f "$ROOT/thumbnails" || true)
[ "$LINKED" = "$(readlink -f "$THUMBS")" ] || { echo "ABORT: $ROOT/thumbnails resolves to '$LINKED', not $THUMBS"; exit 3; }
echo "[thumbs] $N_THUMB files, anchor OK; $ROOT/thumbnails -> $THUMBS"

# --- stage the fixture to node-local NVMe ---------------------------------------------------
echo "=== extracting fixture -> \$SLURM_TMPDIR ==="
t0=$SECONDS
mkdir -p "$SLURM_TMPDIR/mv74k"
pids=(); files=()
for f in "$TARS"/shard_*.tar; do
    tar -xf "$f" -C "$SLURM_TMPDIR/mv74k" &
    pids+=($!); files+=("$f")
done
fail=0
for i in "${!pids[@]}"; do
    if ! wait "${pids[$i]}"; then
        echo "ABORT-PENDING: tar failed for ${files[$i]}"
        fail=1
    fi
done
[ "$fail" -eq 0 ] || { echo "ABORT: at least one fixture shard failed to extract (see above)"; exit 3; }
N_DIR=$( { ls -U "$SLURM_TMPDIR/mv74k" 2>/dev/null || true; } | wc -l )
echo "extracted $N_DIR shape dirs in $((SECONDS-t0))s (expect $EXPECT_SHAPES)"
[ "$N_DIR" -eq "$EXPECT_SHAPES" ] || { echo "ABORT: fixture extract is short ($N_DIR of $EXPECT_SHAPES)"; exit 3; }

# --- example JSONs ----------------------------------------------------------------------------
echo "=== building example JSONs ==="
python scripts/lightgen/make_74k_jsons.py \
    --train_shas  "$SHAS/agentic_train_shas.txt" \
    --val_shas    "$SHAS/v2_val_shas.txt" \
    --test_shas   "$SHAS/v2_test_shas.txt" \
    --val64_shas  scripts/lightgen/val64_v2_shas.txt \
    --fixture_root "$SLURM_TMPDIR/mv74k" \
    --thumbnail_dir "$ROOT/thumbnails" \
    --out_train   "$ROOT/mv74k_train_agentic.json" \
    --out_val64   "$ROOT/mv74k_val64.json" \
    --expect_train "$EXPECT_TRAIN"

# --- train -------------------------------------------------------------------------------------
export WANDB_MODE=offline
export WANDB_SERVICE_WAIT=300
export LIGHTGEN_WANDB_NAME="$RUN"
export LIGHTGEN_RUN_TS=${LIGHTGEN_RUN_TS:-$(date +%Y%m%d-%H%M%S)}
echo "LIGHTGEN_RUN_TS=$LIGHTGEN_RUN_TS"

mkdir -p logs
LOGDIR="logs/lightgen-emission-74k-alpha-thumb-agentic-fir-xg-${NAME}-${LIGHTGEN_RUN_TS}"
if [ -f "$LOGDIR/checkpoints/last.ckpt" ]; then
    echo "[resume] $LOGDIR/checkpoints/last.ckpt exists -- train.py auto-resumes from it"
else
    echo "[fresh] no last.ckpt under $LOGDIR -- starting from the pretrained snapshot"
fi

nvidia-smi --query-gpu=index,memory.used --format=csv -l 120 > "$ROOT/vram-alpha-thumb-agentic-xg-$SLURM_JOB_ID.log" &

( set +e +o pipefail
  CGROOT=/sys/fs/cgroup
  REL=$(awk -F: '$1=="0"{print $3}' /proc/self/cgroup 2>/dev/null)
  CG="$CGROOT$REL"
  [ -r "$CG/memory.current" ] || CG=$CGROOT
  echo "sampler: cgroup=$CG readable=$([ -r "$CG/memory.current" ] && echo yes || echo NO)"
  n=0
  while true; do
      cur=$(cat $CG/memory.current 2>/dev/null || cat $CGROOT/memory/memory.usage_in_bytes 2>/dev/null || echo 0)
      pk=$(cat $CG/memory.peak 2>/dev/null || echo 0)
      rss=$(awk '/^MemAvailable:/{print $2}' /proc/meminfo)
      echo "$(date +%H:%M:%S) cgroup_cur_GB=$((cur/1073741824)) cgroup_peak_GB=$((pk/1073741824)) host_avail_GB=$((rss/1048576))"
      if [ $((n % 5)) -eq 0 ]; then
          ps -eo rss=,comm=,args= --sort=-rss 2>/dev/null | head -6 \
            | awk '{printf "    rss_GB=%.1f %s\n", $1/1048576, substr($0, index($0,$2), 60)}'
          echo "    n_python=$(pgrep -c -f 'python train.py' 2>/dev/null || echo 0)"
      fi
      n=$((n+1))
      sleep 2
  done ) > "$ROOT/hostmem-thumb-agentic-xg-$SLURM_JOB_ID.log" 2>&1 &

set +e
python train.py \
    --base "$CFG" \
    --name "$NAME" \
    --logdir logs/ \
    --gpus 0,1,2,3
RC=$?
set -e
echo "=== $(date -Is) train.py exited rc=$RC ==="
if [ -d "$LOGDIR/checkpoints/troughs" ]; then
    echo "[troughs] on FIR (xya120 account) at $PWD/$LOGDIR/checkpoints/troughs :"
    ls -la "$LOGDIR/checkpoints/troughs" || true
    echo "[troughs] fir cannot reach jupiter -- pull these to the workstation and forward them"
    echo "[troughs] to jupiter outputs/$RUN/ckpts/, sha256 BOTH ends, then add the MANIFEST row."
fi
echo "=== $(date -Is) done, rc=$RC ==="
exit $RC
