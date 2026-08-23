#!/bin/bash
# LightGen on FIR: pbr->emission multiview fine-tune WITH alpha, on the _agentic training
# split (36,255), REFERENCE IMAGE = the TexVerse thumbnail. A FRESH run (no resume seed).
# Config: cfgs/lightgen-emission-74k-alpha-thumb-agentic-fir.yaml -- read its header first;
#         it explains the thumbnail_dir symlink and why the segments are 48 h, not 24 h.
# Spec:   lightgen docs/superpowers/specs/2026-08-23-hunyuan3d-paint-thumbnail-reference-design.md
# Plan:   lightgen docs/superpowers/plans/2026-08-23-hunyuan3d-paint-thumbnail-reference.md
#
# SITE MOVE 2026-08-23 (Dongchen's call): written for cs-venus-05, moved here because another
# user's 24 h job took that node's four Max-Q cards minutes after the _nonzero_nocopy chain
# released them and holds them until 2026-08-24T12:09, while fir measured a start slot ~12 min
# out and runs this recipe at 3.80 s/step vs venus-05's 5.8.
#
# Built from train_74k_alpha_nonzero_nocopy_fir.sbatch (the fir node, the account, the
# $SLURM_TMPDIR fixture extract, the GPU gate, the cgroup memory sampler) with the
# thumbnail-reference staging from train_74k_alpha_thumb_agentic_venus05.sbatch. Differences
# from the venus twin that matter:
#   * fir does NOT mount jupiter. The venus twin self-serves BOTH the sha list and the
#     thumbnail tar from /3dlg-jupiter-project; here they must already be on fir, pushed from
#     the workstation, and this script only GATES them (md5 + count + byte anchor).
#   * The thumbnails are extracted to $SLURM_TMPDIR, not to a persistent directory: fir's
#     /scratch has ~99K free inodes (`lfs quota -u dya78 /scratch`; diskusage_report is STALE
#     and reads ~112K low), and 80,735 PNGs there would leave ~18K. $ROOT/thumbnails is then
#     repointed at it as an ABSOLUTE symlink so the config path can stay static. Copying the
#     venus twin's RELATIVE `ln -sfn emissive_thumbnails` here would leave a dangling link.
#   * NO jupiter backup block at the end -- fir cannot reach jupiter. See "AFTER THE RUN".
#
# PREREQUISITES (each gated below, so a missing piece fails in seconds, not at hour 3).
# All were verified present on fir 2026-08-23 unless marked NEW:
#   1. venv     -- /scratch/dya78/lightgen_mvpaint/env  (python 3.12.4, torch 2.11.0+computecanada)
#   2. weights  -- /scratch/dya78/lightgen_mvpaint/hf_home: tencent/Hunyuan3D-2.1 at the PINNED
#                  revision 0b94677654c57bb9a6b6845cd7b704ccf551d327 AND facebook/dinov2-giant
#                  (the reference-image encoder pulls the second one in separately at
#                  hunyuanpaintpbr/unet/model.py:119; staging only the first looks complete
#                  until the model constructor runs -- it is what killed smoke 54933943).
#   3. fixture  -- 12 shard_*.tar in ~/projects/rrg-msavva/dya78/lightgen_mv74k_alpha/,
#                  247,435 entries each, 72,420 shape dirs. The _agentic train set (36,255) is
#                  a strict SUBSET of _nonzero_nocopy (63,194), itself a subset of these
#                  72,420 -- verified 2026-08-23 -- so the tars did NOT need rebuilding.
#   4. splits   -- /scratch/dya78/lightgen_mvpaint/v2_shas/: v2_{val,test}_shas.txt (unchanged)
#                  and NEW agentic_train_shas.txt (36,255 lines, md5 gated below), pushed from
#                  the workstation because fir has no jupiter mount.
#   5. thumbs   -- NEW ~/projects/rrg-msavva/dya78/lightgen_mv74k_alpha/thumbnails_emissive.tar
#                  (12,615,577,600 B, 80,735 emissive_thumbnails/<sha>.png), pushed from the
#                  workstation. It lives on /project, not /scratch: /project has ~3.9M free
#                  inodes and is not purged, and a tar is one inode either way.
#   6. code     -- /scratch/dya78/lightgen_mvpaint/Hunyuan3D-2.1-emissive on branch lightgen.
#                  Slurm spools THIS script at submit time, so the in-job `git pull` below
#                  updates the config and the model code but never the .sbatch: pull on the
#                  login node before submitting.
#
# RESOURCES. 1 GPU : 12 CPU : 288 GB is the H100 RGU ratio; a full node is 4 GPUs : 48 CPUs :
# 1,152,000 MB and this asks for exactly that. Do NOT write --mem=1152G: Slurm reads the G
# suffix as GiB, making it 27 GB MORE than the node has, and the job becomes unschedulable
# with only "Requested node configuration is not available" as a hint. --mem=0 means "all the
# memory on the node" and is immune to RealMemory changing under us.
#
# WALL TIME 24 h, THREE segments -- REVISED 2026-08-23 16:20 after measurement, see below.
#
# The first submission asked 48 h and landed in gpubase_bynode_b4, where it sat PENDING with
# "Nodes required for job are DOWN, DRAINED or reserved for jobs in higher priority
# partitions" and an estimate that drifted 02:21 -> 03:00 -> 02:49 -> 03:30 without improving.
# Re-measured with sbatch --test-only at 16:19 that day:
#     --time=24:00:00 -> gpubase_bynode_b3, start 16:41 (22 minutes out)
#     --time=48:00:00 -> gpubase_bynode_b4, start 18:45
#     --time=72:00:00 -> gpubase_bynode_b4, start 18:45
# b3 is backed by 90 nodes against b4's 60, and the b4 reservation is structural -- waiting
# does not clear it. An earlier probe (a few hours before) had all three tiers returning the
# SAME slot, which is why the first submission took the longer tier; that stopped being true.
#
# ‼️ WHAT ACTUALLY HAPPENED, and the lesson worth more than the tier choice:
# the resubmitted b3 job landed at 2026-08-24T02:10, NOT the 16:41 that --test-only had just
# promised -- while a FRESH --test-only probe, run minutes later with identical resources,
# still said 16:41. Same scheduler, same ask, 9.5 h apart.
# **`sbatch --test-only` has no predictive power for where a job will actually sit in the
# queue.** Its will-run simulation assumes the job exists NOW and outranks everything pending;
# it does not model fairshare competition -- including our own concurrently RUNNING jobs on the
# same account, which are exactly what pushes the next one down. Use --test-only only to learn
# which PARTITION a --time value routes to; never quote its start time as an estimate, and
# never scancel a queued job on the strength of it.
# Net effect of this tier switch was ~1.3 h (b4 03:30 -> b3 02:10), not the ~11 h the probe
# implied. It was still worth taking, but it was not the trade it looked like.
#
# NOTE: `scontrol update TimeLimit=` does NOT re-route an already-queued job -- Slurm picks the
# partition at submit time -- so switching tiers requires scancel + resubmit. Verified.
#
# 50,000 steps at fir's measured effective 3.46 s/step (25,000 banked in one 24 h segment on
# fc10414) is ~48 h = TWO full 24 h segments; three are queued so the last one absorbs the
# per-segment staging (~10-20 min of extract) and any slow validation. That is two boundaries
# instead of the 48 h plan's one -- accepted for an ~11 h earlier start, because both known
# causes of this chain's historical boundary failures are now gated or fixed: a leaked process
# holding 68.93 GiB on fc10218's GPU 0 (the GPU gate below exits in seconds naming the card;
# resubmit with --exclude=fc10218) and wandb-core missing its 30 s port-file deadline
# (WANDB_SERVICE_WAIT below). The venus-05 chain of the previous campaign resumed correctly
# across its boundary (247612 TIMEOUT -> 248375 COMPLETED), so the resume path itself is sound.
#SBATCH --account=rrg-msavva_gpu
#SBATCH --job-name=mvpaint_74k_alpha_thumb_agentic
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --gres=gpu:h100:4
#SBATCH --output=/scratch/dya78/lightgen_mvpaint/%x-%j.out
#
# RESUME after a requeue or at a segment boundary -- SAME stamp, same logdir, picks up
# last.ckpt automatically (train.py auto-sets --resume from <logdir>/checkpoints/last.ckpt
# when the logdir already exists). submit_chain_fir.sh computes the stamp once and exports it
# to every segment; a hand resubmission MUST pass it:
#   LIGHTGEN_RUN_TS=<the original stamp> sbatch scripts/lightgen/train_74k_alpha_thumb_agentic_fir.sbatch
# Without it a resubmission starts a FRESH run at step 0 in a new directory, silently.
#
# AFTER THE RUN -- THIS IS MANUAL, THE SCRIPT CANNOT DO IT.
# fir does not mount jupiter, so the troughs stay on fir /scratch and AGENTS.md's rule ("a run
# is not finished until its checkpoints are on jupiter and written down") becomes a two-hop
# task for the workstation: pull <logdir>/checkpoints/troughs/ down, rsync it up to jupiter
# outputs/<run>/ckpts/, sha256-verify BOTH ends, add the MANIFEST.md row and the
# docs/status/checkpoints.md entry. That is exactly how the _nonzero_nocopy fir troughs
# (10k/20k) reached jupiter. ~90 GB for five troughs.
set -euo pipefail

ROOT=/scratch/dya78/lightgen_mvpaint
SHAS=$ROOT/v2_shas
TARS=~/projects/rrg-msavva/dya78/lightgen_mv74k_alpha
CFG=cfgs/lightgen-emission-74k-alpha-thumb-agentic-fir.yaml
NAME=74k_alpha_thumb_agentic_fir
RUN='mvpaint_pbr->emission_74k_v2_alpha_thumb_agentic'
AGENTIC_MD5=191a91bc295192517565eaac6089753b
EXPECT_TRAIN=36255
EXPECT_SHAPES=72420
EXPECT_THUMBS=80735
THUMB_TAR=$TARS/thumbnails_emissive.tar
THUMB_TAR_BYTES=12615577600
# Whole-archive md5, verified identical on jupiter and on the workstation copy that was pushed
# here 2026-08-23. Size + member count + one byte anchor do NOT establish archive identity: a
# same-size wrong archive passes all three, and tar headers checksum metadata, not member
# payloads. At ~2-3 min per segment against a 48 h segment this is cheap insurance on the
# reference images -- the one input whose corruption would train silently and wrongly.
THUMB_TAR_MD5=04d4c4eacb0cd00701943fd8e5828a5a
ANCHOR=78e15dddf5314eebb1cf6473c894dda2.png
ANCHOR_MD5=2820b562140bbf73e9b031fd5aa53fa6
SNAP=$ROOT/hf_home/hub/models--tencent--Hunyuan3D-2.1/snapshots/0b94677654c57bb9a6b6845cd7b704ccf551d327/hunyuan3d-paintpbr-v2-1

echo "=== $(date -Is) host=$(hostname) job=$SLURM_JOB_ID ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# GPU-OCCUPANCY GATE. A whole-node job is entitled to 4 clean GPUs, but segments s3/s4
# (55531930/31, 2026-08-20) OOM'd at model_to_device() on fc10218 because a process SLURM
# never cleaned up (PID 2342614) still held 68.93 GiB on GPU 0 -- each burned its extract
# before dying. Fail here in seconds, naming the card and the number. 4096 MiB of slack
# covers driver/ECC overhead without masking a real squatter.
BUSY=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
       | awk -F', ' '$2+0 > 4096 {printf "GPU%s=%sMiB ", $1, $2}')
if [ -n "$BUSY" ]; then
    echo "ABORT: leaked process(es) resident on this node before we start: $BUSY"
    echo "       Resubmit; consider sbatch --exclude=$(hostname -s)."
    exit 4
fi
echo "[gate] all 4 GPUs clean"

# --- gates ---------------------------------------------------------------------------------
[ -d "$ROOT/env" ] || { echo "ABORT: no venv at $ROOT/env -- run env_setup_fir.sh"; exit 2; }
[ -d "$SNAP" ] || { echo "ABORT: pinned HF snapshot missing at $SNAP (PREREQUISITES 2)"; exit 2; }
[ -d "$ROOT/hf_home/hub/models--facebook--dinov2-giant" ] || { echo "ABORT: dinov2-giant snapshot missing (PREREQUISITES 2)"; exit 2; }
for f in agentic_train_shas.txt v2_val_shas.txt v2_test_shas.txt; do
    [ -s "$SHAS/$f" ] || { echo "ABORT: missing $SHAS/$f -- push it from the workstation (fir has no jupiter mount)"; exit 2; }
done
# The campaign's one variable: pin it by content, not by name.
TRAIN_MD5=$(md5sum "$SHAS/agentic_train_shas.txt" | cut -d' ' -f1)
[ "$TRAIN_MD5" = "$AGENTIC_MD5" ] || { echo "ABORT: agentic_train_shas.txt md5 $TRAIN_MD5 != recorded $AGENTIC_MD5"; exit 2; }
[ "$(wc -l < "$SHAS/agentic_train_shas.txt")" -eq "$EXPECT_TRAIN" ] || { echo "ABORT: agentic_train_shas.txt is not $EXPECT_TRAIN lines"; exit 2; }
# `|| true` INSIDE the pipeline: with pipefail a no-match glob makes ls fail, and the
# assignment would then kill the job under errexit BEFORE the explicit ABORT below prints.
N_TAR=$( { ls "$TARS"/shard_*.tar 2>/dev/null || true; } | wc -l )
[ "$N_TAR" -eq 12 ] || { echo "ABORT: found $N_TAR fixture tars in $TARS, expected 12"; exit 2; }
[ -s "$THUMB_TAR" ] || { echo "ABORT: $THUMB_TAR missing -- push it from the workstation (fir has no jupiter mount)"; exit 2; }
T_BYTES=$(stat -c %s "$THUMB_TAR" 2>/dev/null || echo 0)
[ "$T_BYTES" -eq "$THUMB_TAR_BYTES" ] || { echo "ABORT: $THUMB_TAR is $T_BYTES bytes, expected $THUMB_TAR_BYTES"; exit 2; }
echo "[gate] venv, both HF snapshots, 3 sha lists (train md5 OK), 12 fixture tars, thumbnail tar size"

# CONCURRENCY GATE. $ROOT/thumbnails and the two example JSONs are SHARED, per-root state that
# every segment rewrites with its own node-local $SLURM_TMPDIR paths. The afterany chain never
# overlaps, so that is safe -- but a hand resubmission alongside a running segment would
# repoint the live run's symlink at another node's scratch and overwrite its JSONs, and the
# victim would fail at a batch, far from the cause. Refuse instead.
# Both `|| true`s are load-bearing under pipefail: squeue may fail, and `grep -v` returns 1
# when it filters out EVERY line -- which is the NORMAL case here, this job being the only
# match. Without the second one the gate would kill every healthy segment it was meant to protect.
OTHERS=$( { squeue -u "$USER" -h -n "$SLURM_JOB_NAME" -t R -o '%i' 2>/dev/null || true; } \
          | { grep -v "^${SLURM_JOB_ID}$" || true; } | tr '\n' ' ')
[ -z "$OTHERS" ] || { echo "ABORT: another segment of $SLURM_JOB_NAME is already RUNNING (job(s): $OTHERS). Two segments share \$ROOT/thumbnails and the example JSONs; run them in sequence (afterany), not side by side."; exit 5; }
echo "[gate] no other running segment of $SLURM_JOB_NAME"

# CODE FIRST, before either extract. A wrong branch, a failed pull or a missing config costs
# seconds here; behind the extracts it costs the ~2.97M-file fixture unpack plus 12.6 GB of
# thumbnails. Slurm spooled THIS .sbatch at submit time, so the pull below updates the config
# and the model code but never this file -- pull on the login node before submitting.
# --- code ------------------------------------------------------------------------------------
cd "$ROOT/Hunyuan3D-2.1-emissive"
BRANCH=$(git rev-parse --abbrev-ref HEAD)
[ "$BRANCH" = "lightgen" ] || { echo "ABORT: fir clone is on branch '$BRANCH', expected lightgen"; exit 2; }
git pull --ff-only
cd hy3dpaint
[ -f "$CFG" ] || { echo "ABORT: $CFG not in the pulled clone"; exit 2; }
CFG_SNAP=$(awk -F': *' '/^[[:space:]]*pretrained_model_name_or_path:/ {print $2; exit}' "$CFG")
[ "$CFG_SNAP" = "$SNAP" ] || { echo "ABORT: $CFG names snapshot '$CFG_SNAP', but the gate checked '$SNAP'"; exit 2; }

# opencv is REQUIRED as of the thumbnail-reference loader: it imports loader_util for
# BaseDataset.load_image/augment_image, and loader_util imports cv2 (augment_image's
# perspective branch calls cv2.getPerspectiveTransform/warpPerspective). fir's venv does NOT
# have cv2 -- env_setup_fir.sh never installed it, because no earlier fir run imported
# loader_util -- and Alliance deliberately ships an `opencv-noinstall` shim in the wheelhouse
# so `pip install opencv-python` FAILS by design. The module is the supported route.
# Verified 2026-08-23: cv2 4.12.0 against the venv's numpy 1.26.4, getPerspectiveTransform OK.
# (A `_alliance_opencv.pth` in the venv's site-packages points at the same directory, so the
# already-queued segments of this chain -- whose .sbatch was spooled before this line existed
# -- also find cv2. Both routes resolve to the same files; loading the module is harmless.)
module load StdEnv/2023 python/3.12 cuda/12.6 opencv/4.12.0
source "$ROOT/env/bin/activate"
export PYTHONNOUSERSITE=1
export HF_HOME=$ROOT/hf_home
# Compute nodes have no outbound internet; without these, from_pretrained blocks on a
# metadata call and dies with a connection error instead of using the staged snapshot.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# --- stage the thumbnails to node-local NVMe, and repoint the config's static path ----------
# FIRST, before the much larger fixture extract: these are the newly transferred, least-proven
# asset, so failing on them early is failing cheaply.
# NOT extracted to /scratch: ~99K free inodes there (`lfs quota -u dya78 /scratch`;
# diskusage_report reads ~112K low) and this is 80,735 files. Node-local costs no /scratch
# inode and is faster to read during training.
echo "=== verifying thumbnail archive (md5 of 12.6 GB, ~2-3 min) ==="
t0=$SECONDS
T_MD5=$(md5sum "$THUMB_TAR" | cut -d' ' -f1)
[ "$T_MD5" = "$THUMB_TAR_MD5" ] || { echo "ABORT: $THUMB_TAR md5 $T_MD5 != recorded $THUMB_TAR_MD5 -- this is NOT the archive that was staged; re-push it"; exit 3; }
echo "md5 OK in $((SECONDS-t0))s"
echo "=== extracting thumbnails -> \$SLURM_TMPDIR ==="
t0=$SECONDS
tar -xf "$THUMB_TAR" -C "$SLURM_TMPDIR"
THUMBS=$SLURM_TMPDIR/emissive_thumbnails
N_THUMB=$( { ls -U "$THUMBS" 2>/dev/null || true; } | wc -l )
echo "extracted $N_THUMB thumbnails in $((SECONDS-t0))s (expect $EXPECT_THUMBS)"
[ "$N_THUMB" -eq "$EXPECT_THUMBS" ] || { echo "ABORT: $THUMBS holds $N_THUMB files, expected $EXPECT_THUMBS"; exit 3; }
A_MD5=$(md5sum "$THUMBS/$ANCHOR" 2>/dev/null | cut -d' ' -f1 || true)
[ "$A_MD5" = "$ANCHOR_MD5" ] || { echo "ABORT: thumbnail anchor $ANCHOR md5 '$A_MD5' != $ANCHOR_MD5 (missing, or not the TexVerse thumbnails the other baselines use)"; exit 3; }
# ABSOLUTE target: $SLURM_TMPDIR is not a sibling of $ROOT, so the venus twin's relative form
# would dangle. -n so an existing symlink is REPLACED rather than followed into.
ln -sfn "$THUMBS" "$ROOT/thumbnails"
LINKED=$(readlink -f "$ROOT/thumbnails" || true)
[ "$LINKED" = "$(readlink -f "$THUMBS")" ] || { echo "ABORT: $ROOT/thumbnails resolves to '$LINKED', not $THUMBS (a real directory of that name is in the way -- remove it)"; exit 3; }
echo "[thumbs] $N_THUMB files, anchor OK; $ROOT/thumbnails -> $THUMBS"

# --- stage the fixture to node-local NVMe ---------------------------------------------------
# /project is Lustre and training I/O against it is forbidden; 2.97M small files over Lustre
# would also be far slower than paying this extraction once per segment.
# Each tar is waited on BY PID: a bare `wait` returns 0 regardless of what the children did,
# so a shard that died half-way would sail through to the count below -- and the count only
# sees top-level dirs, which a failing tar may well have finished creating. A7 would then
# check only the agentic train + val64 subset, and the loader's decode guard would quietly
# substitute whatever individual examples were truncated.
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
# Rebuilt every segment because $SLURM_TMPDIR paths change. A7 proves the fixture carries the
# alpha maps AND -- with --thumbnail_dir -- that every listed sha has a thumbnail, both before
# a batch is touched. --expect_train 36255, NO --allow_missing: the one unrenderable sha
# (810a644a...) is already absent from the agentic list, so zero missing is the enforced outcome.
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
# WANDB offline: compute nodes have no internet and we keep no credential on a shared cluster.
# Sync from the workstation afterwards:
#   wandb sync -p LightGen $ROOT/.../logs/lightgen-emission-74k-alpha-thumb-agentic-fir-$NAME-<TS>/wandb/offline-run-*
export WANDB_MODE=offline
# Segment s2 (55531929) died with wandb ServicePollForTokenError: wandb-core started but did
# not write its port file within the default 30 s. 300 s tolerates a slow service start.
export WANDB_SERVICE_WAIT=300
export LIGHTGEN_WANDB_NAME="$RUN"
# Computed ONCE by submit_chain_fir.sh and passed via --export; lightning's DDP launcher
# re-execs train.py per rank and each re-exec must inherit the same value, or the four ranks
# write four logdirs. The fallback only fires for a hand-submitted first segment.
export LIGHTGEN_RUN_TS=${LIGHTGEN_RUN_TS:-$(date +%Y%m%d-%H%M%S)}
echo "LIGHTGEN_RUN_TS=$LIGHTGEN_RUN_TS"

mkdir -p logs
LOGDIR="logs/lightgen-emission-74k-alpha-thumb-agentic-fir-${NAME}-${LIGHTGEN_RUN_TS}"
if [ -f "$LOGDIR/checkpoints/last.ckpt" ]; then
    echo "[resume] $LOGDIR/checkpoints/last.ckpt exists -- train.py auto-resumes from it"
else
    echo "[fresh] no last.ckpt under $LOGDIR -- starting from the pretrained snapshot"
fi

nvidia-smi --query-gpu=index,memory.used --format=csv -l 120 > "$ROOT/vram-alpha-thumb-agentic-$SLURM_JOB_ID.log" &

# HOST-memory sampler. Smoke 54934387 was OOM-killed at a 1125 GB cgroup limit while sacct
# reported MaxRSS of 21.5 GB -- sacct SAMPLES (~30 s) and a fast allocation spike falls between
# samples. This reads the cgroup's own counter every 2 s, which is what the OOM killer acts on.
# The cgroup path must be DERIVED, not guessed: /sys/fs/cgroup/memory.current is the ROOT
# cgroup on v2, which a Slurm step cannot read, so resolve the step's own from /proc/self/cgroup.
( set +e +o pipefail   # `ps | head -6` SIGPIPEs ps once head exits; under the inherited
                       # errexit/pipefail that would silently kill the very sampler we would
                       # need to explain a host OOM.
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
  done ) > "$ROOT/hostmem-thumb-agentic-$SLURM_JOB_ID.log" 2>&1 &

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
    echo "[troughs] on FIR at $PWD/$LOGDIR/checkpoints/troughs :"
    ls -la "$LOGDIR/checkpoints/troughs" || true
    echo "[troughs] fir cannot reach jupiter -- pull these to the workstation and forward them"
    echo "[troughs] to jupiter outputs/$RUN/ckpts/, sha256 BOTH ends, then add the MANIFEST row."
fi
echo "=== $(date -Is) done, rc=$RC ==="
exit $RC
