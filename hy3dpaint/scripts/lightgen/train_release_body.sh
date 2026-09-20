# LightGen: pbr->emission multiview fine-tune WITH alpha, thumbnail reference, on the RELEASED
# dataset (Hugging Face 3dlg-hcvc/LightgenBench) -- the JOB BODY, for any Alliance cluster.
#
# Not run by hand. scripts/lightgen/submit_release.sh writes a job file = a cluster-specific
# #SBATCH header + literal assignments of
#     CLUSTER ROOT REPO DATA STAMP COMMIT SMOKE NAME RUN JOB_PREFIX
# + this file, and submits that. Slurm spools the job file at submit time, so a queued segment
# is immune to later edits of this body; the config and the model code are pinned too, by the
# COMMIT gate below (there is no in-job `git pull`: nibi, rorqual and trillium compute nodes
# have no route to GitHub, and the pull happens on the login node at submit time instead).
#
# The recipe is the paper run's (scripts/lightgen/train_74k_alpha_thumb_agentic_fir.sbatch:
# same gates, same env, same train.py call, same 50,000 steps at global batch 16). What
# changed is the data staging: the release ships per-representation tar shards with the layout
#     <uuid>/multiview/*.png + transforms.json      <uuid>/thumbnail.png
# and relayout_release.py renames/links that into the fixture layout the loader reads. The
# views are the same renders (byte-identical to the old fixture where the bake is unchanged)
# and the thumbnails are the same TexVerse files.
#
# PREREQUISITES on the cluster, all under $ROOT (= <scratch>/lightgen_mvpaint), each gated:
#   env/           scripts/lightgen/stage_alliance.sh env      (python 3.12, torch 2.11.0)
#   hf_home/       scripts/lightgen/stage_alliance.sh weights  (Hunyuan3D-2.1 pinned + dinov2-giant)
#   release_hf/    scripts/lightgen/stage_alliance.sh data     (the HF tars, sha256-verified)
#   Hunyuan3D-2.1-emissive/   the clone, branch lightgen, at the commit the job was submitted on
#
# AFTER THE RUN: the troughs stay on the cluster's scratch. Pull <logdir>/checkpoints/troughs/
# to the workstation, forward to jupiter outputs/<run>/ckpts/, sha256 BOTH ends, MANIFEST row.
set -euo pipefail

# Hedged submission: a job cancelled in its first minute has cost queue position only.
sleep 60

CHECKSUMS_SHA256=16ad1851c5ad9080ea1bede027c57fba8809619e0c1a5e0a89b1cf0d89c5a3bc
SPLITS_SHA256=6ab3bae5453ba64a1c995a8c178c999cea6bb89f796918521752440b66dfe87e
EXPECT_TRAIN=36426
EXPECT_VAL=200
EXPECT_TEST=200
EXPECT_TARS=76          # train 37 + val 1, for each of multiview and thumbnail
CFG_TMPL=cfgs/lightgen-emission-release-alpha-thumb.yaml
SNAP=$ROOT/hf_home/hub/models--tencent--Hunyuan3D-2.1/snapshots/0b94677654c57bb9a6b6845cd7b704ccf551d327/hunyuan3d-paintpbr-v2-1
: "${SLURM_TMPDIR:?this body stages to node-local storage and needs SLURM_TMPDIR}"
STAGE=$SLURM_TMPDIR/lightgen_release

echo "=== $(date -Is) cluster=$CLUSTER host=$(hostname) job=$SLURM_JOB_ID stamp=$STAMP smoke=$SMOKE ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# GPU-OCCUPANCY GATE (from the paper launcher): a leaked process on an allocated card OOMs the
# run at model_to_device() after the whole staging. Fail in seconds, naming the card.
BUSY=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
       | awk -F', ' '$2+0 > 4096 {printf "GPU%s=%sMiB ", $1, $2}')
if [ -n "$BUSY" ]; then
    echo "ABORT: leaked process(es) resident on this node before we start: $BUSY"
    echo "       Resubmit; consider EXCLUDE=$(hostname -s)."
    exit 4
fi
echo "[gate] GPUs clean"

# --- gates ---------------------------------------------------------------------------------
[ -x "$ROOT/env/bin/python" ] || { echo "ABORT: no venv at $ROOT/env -- stage_alliance.sh env"; exit 2; }
[ -d "$SNAP" ] || { echo "ABORT: pinned HF snapshot missing at $SNAP -- stage_alliance.sh weights"; exit 2; }
[ -d "$ROOT/hf_home/hub/models--facebook--dinov2-giant" ] || { echo "ABORT: dinov2-giant snapshot missing -- stage_alliance.sh weights"; exit 2; }
for f in checksums.sha256 splits.json; do
    [ -s "$DATA/$f" ] || { echo "ABORT: missing $DATA/$f -- stage_alliance.sh data"; exit 2; }
done
# The campaign's data, pinned by content: the release's own checksum file and its split.
GOT=$(sha256sum "$DATA/checksums.sha256" | cut -d' ' -f1)
[ "$GOT" = "$CHECKSUMS_SHA256" ] || { echo "ABORT: checksums.sha256 is $GOT, expected $CHECKSUMS_SHA256 -- not the release this run was written for"; exit 2; }
GOT=$(sha256sum "$DATA/splits.json" | cut -d' ' -f1)
[ "$GOT" = "$SPLITS_SHA256" ] || { echo "ABORT: splits.json is $GOT, expected $SPLITS_SHA256"; exit 2; }
echo "[gate] venv, both HF snapshots, checksums.sha256 and splits.json pinned"

# CONCURRENCY GATE. Two segments of one campaign share a logdir (same STAMP) and would both
# write last.ckpt. The afterany chain never overlaps; a hand resubmission beside a running
# segment would. Both `|| true`s are load-bearing under pipefail (squeue may fail; grep -v
# returns 1 when it filters out every line, the NORMAL case). A smoke has its own stamp.
if [ "$SMOKE" != 1 ]; then
    OTHERS=$( { squeue -u "$USER" -h -t R -o '%i %j' 2>/dev/null || true; } \
              | { awk -v p="^${JOB_PREFIX}_s[0-9]+\$" -v me="$SLURM_JOB_ID" '$2 ~ p && $1 != me {print $1}' || true; } | tr '\n' ' ')
    [ -z "$OTHERS" ] || { echo "ABORT: another segment of $JOB_PREFIX is already RUNNING here (job(s): $OTHERS); run them in sequence (afterany)."; exit 5; }
    echo "[gate] no other running segment of $JOB_PREFIX"
fi

# --- code: the commit this job was submitted on, not whatever the clone holds now ------------
cd "$REPO"
BRANCH=$(git rev-parse --abbrev-ref HEAD)
[ "$BRANCH" = "lightgen" ] || { echo "ABORT: clone is on branch '$BRANCH', expected lightgen"; exit 2; }
HEAD_NOW=$(git rev-parse HEAD)
[ "$HEAD_NOW" = "$COMMIT" ] || { echo "ABORT: clone is at $HEAD_NOW but this job was submitted on $COMMIT -- the config or model code changed while the chain was queued. Cancel the chain and start a NEW stamp."; exit 2; }
cd hy3dpaint
[ -f "$CFG_TMPL" ] || { echo "ABORT: $CFG_TMPL not in the clone"; exit 2; }

# opencv comes from the module: the thumbnail loader imports loader_util, which imports cv2,
# and Alliance ships an opencv-noinstall shim so `pip install opencv-python` fails by design.
module load StdEnv/2023 gcc python/3.12 cuda/12.6 opencv/4.12.0
source "$ROOT/env/bin/activate"
export PYTHONNOUSERSITE=1
export HF_HOME=$ROOT/hf_home
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# trillium compute nodes mount $HOME read-only: per-user caches go to node-local storage there.
if ! ( : > "$HOME/.wtest.$$" ) 2>/dev/null; then
    export TRITON_CACHE_DIR=$SLURM_TMPDIR/.triton CUDA_CACHE_PATH=$SLURM_TMPDIR/.nv MPLCONFIGDIR=$SLURM_TMPDIR/.mpl \
           WANDB_CACHE_DIR=$SLURM_TMPDIR/.wandb WANDB_CONFIG_DIR=$SLURM_TMPDIR/.wandb_cfg XDG_CACHE_HOME=$SLURM_TMPDIR/.cache
    echo "HOME read-only: caches redirected to \$SLURM_TMPDIR"
else
    rm -f "$HOME/.wtest.$$"
fi

# LOADER PREFLIGHT, after the activation (so it tests the interpreter that trains) and before
# the staging (so a miss costs seconds).
python - <<'PYCHK' || { echo "ABORT: the training env cannot import the loader (traceback above)."; exit 2; }
import sys
sys.path.insert(0, ".")
from src.data.dataloader.lightgen_emission_loader import LightgenEmissionDataset as D
assert D.REF_SOURCES == ("frontal_albedo", "thumbnail"), D.REF_SOURCES
import cv2, numpy as np, torch
cv2.getPerspectiveTransform(np.float32([[0,0],[8,0],[8,8],[0,8]]),
                            np.float32([[1,1],[7,0],[8,7],[0,8]]))
print(f"[gate] loader imports OK; torch {torch.__version__}, cv2 {cv2.__version__}, numpy {np.__version__}, gpus {torch.cuda.device_count()}")
PYCHK

# --- verify, then stage the release to node-local storage -------------------------------------
mkdir -p "$STAGE/release" "$STAGE/thumbs" "$STAGE/shas" "$STAGE/cfg"
grep -E '  data/(train|val)/(multiview|thumbnail)/' "$DATA/checksums.sha256" > "$STAGE/need.sha256"
N_NEED=$(wc -l < "$STAGE/need.sha256")
[ "$N_NEED" -eq "$EXPECT_TARS" ] || { echo "ABORT: checksums.sha256 lists $N_NEED train/val multiview+thumbnail tars, expected $EXPECT_TARS"; exit 3; }
echo "=== sha256 of the $N_NEED tars ==="
t0=$SECONDS
( cd "$DATA" && xargs -P 16 -d '\n' -I{} sh -c 'printf "%s\n" "$1" | sha256sum -c --quiet -' _ {} < "$STAGE/need.sha256" ) \
    || { echo "ABORT: at least one tar is missing or does not match checksums.sha256 (see above) -- stage_alliance.sh data"; exit 3; }
echo "sha256 OK in $((SECONDS-t0))s"

echo "=== extracting -> $STAGE/release ==="
t0=$SECONDS
cut -d' ' -f3 "$STAGE/need.sha256" | ( cd "$DATA" && xargs -P 12 -d '\n' -I{} tar -xf {} -C "$STAGE/release" ) \
    || { echo "ABORT: at least one tar failed to extract (see above)"; exit 3; }
N_DIR=$( { ls -U "$STAGE/release" 2>/dev/null || true; } | wc -l )
echo "extracted $N_DIR shape dirs in $((SECONDS-t0))s (expect $((EXPECT_TRAIN+EXPECT_VAL)))"
[ "$N_DIR" -eq "$((EXPECT_TRAIN+EXPECT_VAL))" ] || { echo "ABORT: extract holds $N_DIR shape dirs, expected $((EXPECT_TRAIN+EXPECT_VAL))"; exit 3; }

# The split lists come out of the pinned splits.json; the sha-driven JSON builder below is what
# keeps a directory listing (train AND val are both unpacked) from deciding membership.
python - "$DATA/splits.json" "$STAGE/shas" <<'PYSPLIT'
import json, sys
sp = json.load(open(sys.argv[1]))
for k in ("train", "val", "test"):
    open(f"{sys.argv[2]}/{k}.txt", "w").write("\n".join(sp[k]) + "\n")
print("[splits]", {k: len(sp[k]) for k in ("train", "val", "test")})
PYSPLIT
for pair in "train:$EXPECT_TRAIN" "val:$EXPECT_VAL" "test:$EXPECT_TEST"; do
    [ "$(wc -l < "$STAGE/shas/${pair%%:*}.txt")" -eq "${pair##*:}" ] || { echo "ABORT: ${pair%%:*} list is not ${pair##*:} lines"; exit 3; }
done

echo "=== re-laying-out to the fixture layout ==="
python scripts/lightgen/relayout_release.py --root "$STAGE/release" --thumbs_out "$STAGE/thumbs" \
    --shas "$STAGE/shas/train.txt" "$STAGE/shas/val.txt"

# --- example JSONs: the paper's builder and its A0-A7 gates, unchanged -------------------------
echo "=== building example JSONs ==="
python scripts/lightgen/make_74k_jsons.py \
    --train_shas  "$STAGE/shas/train.txt" \
    --val_shas    "$STAGE/shas/val.txt" \
    --test_shas   "$STAGE/shas/test.txt" \
    --val64_shas  scripts/lightgen/val64_release_shas.txt \
    --fixture_root "$STAGE/release" \
    --thumbnail_dir "$STAGE/thumbs" \
    --out_train   "$STAGE/release_train.json" \
    --out_val64   "$STAGE/release_val64.json" \
    --expect_train "$EXPECT_TRAIN"

# --- the config this segment trains with --------------------------------------------------------
# Same basename as the template: train.py names the logdir after it.
CPUS=${SLURM_CPUS_PER_TASK:-48}
NUM_WORKERS=$(( CPUS / 4 - 2 )); [ "$NUM_WORKERS" -ge 2 ] || NUM_WORKERS=2
CFG=$STAGE/cfg/$(basename "$CFG_TMPL")
sed -e "s#__ROOT__#$ROOT#g" -e "s#__STAGE__#$STAGE#g" -e "s#__NUM_WORKERS__#$NUM_WORKERS#g" "$CFG_TMPL" > "$CFG"
if [ "$SMOKE" = 1 ]; then
    # Plumbing smoke: same model, batch, GPUs and data path; only the schedule is cut. Exact
    # whole-line matches on live lines, each required to hit once.
    python - "$CFG" <<'PYSMOKE'
import sys
p = sys.argv[1]
lines = open(p).read().split("\n")
edits = {"max_steps: 50000": "max_steps: 60", "every_n_train_steps: 1250": "every_n_train_steps: 40",
         "val_check_interval: 1250": "val_check_interval: 40", "save_top_k: 3": "save_top_k: 1"}
hits = dict.fromkeys(edits, 0)
for i, ln in enumerate(lines):
    if ln.strip() in edits:
        hits[ln.strip()] += 1
        lines[i] = ln.replace(ln.strip(), edits[ln.strip()])
assert all(v == 1 for v in hits.values()), hits
open(p, "w").write("\n".join(lines))
print("[smoke] schedule cut:", edits)
PYSMOKE
fi
if grep -n "__[A-Z_]*__" "$CFG"; then echo "ABORT: unfilled token in $CFG (line above)"; exit 2; fi
CFG_SNAP=$(awk -F': *' '/^[[:space:]]*pretrained_model_name_or_path:/ {print $2; exit}' "$CFG")
[ "$CFG_SNAP" = "$SNAP" ] || { echo "ABORT: $CFG names snapshot '$CFG_SNAP', but the gate checked '$SNAP'"; exit 2; }
echo "[cfg] $CFG  (num_workers $NUM_WORKERS from $CPUS CPUs)"

# --- train -------------------------------------------------------------------------------------
# WANDB offline everywhere: most compute nodes have no internet and no credential is kept on a
# shared cluster. Sync from the workstation afterwards: wandb sync -p LightGen <logdir>/wandb/offline-run-*
export WANDB_MODE=offline
export WANDB_SERVICE_WAIT=300
export LIGHTGEN_WANDB_NAME="$RUN"
# Baked into the job file by submit_release.sh, the same value for every segment of a chain
# and every rank of a segment: train.py resumes from the STAMP-named logdir's last.ckpt.
export LIGHTGEN_RUN_TS=$STAMP

mkdir -p logs "$ROOT/log"
LOGDIR="logs/$(basename "$CFG_TMPL" .yaml)-${NAME}-${STAMP}"
# PL's save_last versions around a last.ckpt it did not create (last-v1.ckpt, ...), and
# train.py resumes from `last.ckpt` BY NAME. Promote the newest before the resume check.
promote_last() {
    local d="$1" newest
    [ -d "$d" ] || return 0
    newest=$(ls -1t "$d"/last.ckpt "$d"/last-v*.ckpt 2>/dev/null | head -1)
    [ -n "$newest" ] || return 0
    if [ "$newest" != "$d/last.ckpt" ]; then
        echo "[promote] $(basename "$newest") is newer than last.ckpt -- promoting it"
        mv -f "$newest" "$d/last.ckpt" || { echo "ABORT: could not promote $newest"; exit 6; }
    fi
}
promote_last "$LOGDIR/checkpoints"
if [ -f "$LOGDIR/checkpoints/last.ckpt" ]; then
    echo "[resume] $LOGDIR/checkpoints/last.ckpt exists -- train.py auto-resumes from it"
else
    echo "[fresh] no last.ckpt under $LOGDIR -- starting from the pretrained snapshot"
fi

nvidia-smi --query-gpu=index,memory.used --format=csv -l 120 > "$ROOT/log/vram-$SLURM_JOB_ID.log" &

set +e
python train.py \
    --base "$CFG" \
    --name "$NAME" \
    --logdir logs/ \
    --gpus 0,1,2,3
RC=$?
set -e
echo "=== $(date -Is) train.py exited rc=$RC ==="
if [ "$SMOKE" = 1 ] && [ "$RC" -eq 0 ]; then
    echo "[smoke] checkpoints written:"; ls -la "$LOGDIR/checkpoints" || true
    rm -rf "$LOGDIR/checkpoints"   # ~18 GB each; a smoke's weights are not kept
    echo "[smoke] SMOKE_OK -- checkpoints removed, logs kept in $PWD/$LOGDIR"
fi
if [ -d "$LOGDIR/checkpoints/troughs" ]; then
    echo "[troughs] on $CLUSTER at $PWD/$LOGDIR/checkpoints/troughs :"
    ls -la "$LOGDIR/checkpoints/troughs" || true
    echo "[troughs] pull these to the workstation, forward to jupiter outputs/<run>/ckpts/, sha256 BOTH ends, MANIFEST row."
fi
echo "=== $(date -Is) done, rc=$RC ==="
exit $RC
