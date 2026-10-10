#!/bin/bash
# The training check (REFACTOR_TODO.md D5): the tree's real train.py on the check root
# (make_inputs.py --runs train), batch 1, deterministic mode with the math attention kernel,
# reseeded right before trainer.fit, every step recorded by traincheck/sitecustomize.py into
# <out_dir>/record.jsonl. The math kernel makes two runs byte-identical: with the fused kernels
# (flash, memory-efficient, cuDNN), which deterministic mode does not flag, the loss drifted by up to
# 3e-4 between runs on the RTX PRO 6000 Blackwell (measured 2026-10-10). It needs more GPU memory.
#
#   train-check.sh <tree> <out_dir> fresh|resume          (absolute paths; prints TRAINCHECK_DONE)
#
#   fresh   from the pretrained snapshot: 32 steps, validation at steps 16 and 32, one save per
#           checkpointer (main and trough) at step 32
#   resume  --resume from the release run's step-40,000 trough: 32 more steps, validation at 40,016
#           and 40,032, one save per checkpointer at 40,032; shows the code still loads a published
#           checkpoint's weights, optimizer, loops and callbacks. (Not 16 steps: the main checkpointer
#           ranks by val/emission_mse and fires before the validation of its own step, so a resumed
#           process saves only once a validation has run.)
# Needs a GPU with at least 48 GB (a 4090 cannot train this model). The saved checkpoints (about
# 18 GB each) are summarized into the record and deleted, also when the run aborts.
# Host overrides as in gate.sh: GATE_C, GATE_ACTIVATE, GATE_SNAPSHOT, GATE_LOCKS (default "none"
# here: nowhere with a lock file can train); TRAINCHECK_TMPBASE (a short dir with room for a temp file).
set -euo pipefail
case $# in 3) ;; *) echo "usage: train-check.sh <tree> <out_dir> fresh|resume"; exit 2 ;; esac
TREE=$1; OUT=$2; MODE=$3
HERE=$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")
C=${GATE_C:-/local-scratch/localhome/dya78/lightgen_eval_local/hunyuan_refactor_checks}
SNAP=${GATE_SNAPSHOT:-/localhome/dya78/.cache/huggingface/hub/models--tencent--Hunyuan3D-2.1/snapshots/0b94677654c57bb9a6b6845cd7b704ccf551d327/hunyuan3d-paintpbr-v2-1}
LOCKS=${GATE_LOCKS:-none}
ROOT=$C/inputs/train
RESUME_CKPT=$C/ckpts/mvpaint_alpha_thumb_lightgenbench_v1/ckpts/troughs/trough-step=00040000.ckpt
case $MODE in
  fresh)  SAVE=32; MAX=32;    RESUME=() ;;
  resume) SAVE=32; MAX=40032; RESUME=(--resume "$RESUME_CKPT") ;;
  *) echo "mode must be fresh or resume"; exit 2 ;;
esac
for f in "$TREE/hy3dpaint/train.py" "$ROOT/train.txt" "$ROOT/val.txt" "$SNAP/model_index.json"; do
  [ -e "$f" ] || { echo "ABORT: missing $f"; exit 2; }
done
[ "$MODE" = fresh ] || [ -e "$RESUME_CKPT" ] || { echo "ABORT: missing $RESUME_CKPT"; exit 2; }
[ -e "$OUT" ] && { echo "ABORT: $OUT exists"; exit 2; }
if [ "$LOCKS" != none ]; then
  fd=8
  for L in $LOCKS; do
    eval "exec $fd>\"$L\""
    flock -n $fd || { echo "ABORT: the GPU lock $L is held"; exit 7; }
    fd=$((fd + 1))
  done
fi
mkdir -p "$OUT"
set +u
if [ -n "${GATE_ACTIVATE:-}" ]; then source "$GATE_ACTIVATE"
else source /localhome/dya78/software/miniconda3/etc/profile.d/conda.sh; conda activate /localhome/dya78/software/miniconda3/envs/hunyuanpaint; fi
set -u
echo "host=$(hostname -s) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader -i "${CUDA_VISIBLE_DEVICES:-0}" | head -1)" | tee "$OUT/gpu.txt"

python - "$ROOT" "$OUT" <<'EOF'
import json, os, sys
root, out = sys.argv[1:]
for split in ("train", "val"):
    shas = [l.strip() for l in open(os.path.join(root, f"{split}.txt")) if l.strip()]
    json.dump([os.path.join(root, "shapes", s) for s in shas], open(os.path.join(out, f"{split}.json"), "w"))
EOF
sed -e "s|__SNAPSHOT__|$SNAP|" -e "s|__THUMBS__|$ROOT/thumbs|" -e "s|__TRAIN_JSON__|$OUT/train.json|" \
    -e "s|__VAL_JSON__|$OUT/val.json|" -e "s|__SAVE_EVERY__|$SAVE|" -e "s|__MAX_STEPS__|$MAX|" \
    "$HERE/traincheck/traincheck.yaml" > "$OUT/traincheck.yaml"
grep -qE '__(SNAPSHOT|THUMBS|TRAIN_JSON|VAL_JSON|SAVE_EVERY|MAX_STEPS)__' "$OUT/traincheck.yaml" && { echo "ABORT: an unfilled token in $OUT/traincheck.yaml"; exit 2; }

# Lightning writes a checkpoint through a temp file in $TMPDIR; the dataloader worker puts a socket
# there, so the path stays short.
mkdir -p "${TRAINCHECK_TMPBASE:=/local-scratch/localhome/dya78/tmp}"
TMPDIR=$(mktemp -d "$TRAINCHECK_TMPBASE/hc.XXXXXX"); export TMPDIR
trap 'rm -rf "$TMPDIR"; find "$OUT/logs" -name "*.ckpt" -delete 2>/dev/null || true' EXIT
# SLURM_JOB_NAME=bash: Lightning ignores the Slurm job and runs one process, as on a workstation.
unset LIGHTGEN_RUN_TS
export WANDB_MODE=disabled HF_MODULES_CACHE=$OUT/.hf_modules HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
       PYTHONNOUSERSITE=1 SLURM_JOB_NAME=bash HUNYUAN_GATE_DETERMINISTIC=1 HUNYUAN_GATE_MATH_SDP=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
       PYTHONPATH=$HERE/traincheck HUNYUAN_TRAINCHECK_OUT=$OUT/record.jsonl
if [ "${HUNYUAN_COVERAGE:-0}" = 1 ]; then   # step 7: line coverage of the tree under test, every process
  cat > "$OUT/.coveragerc" <<EOT
[run]
parallel = True
concurrency = multiprocessing,thread
source = $TREE/hy3dpaint
data_file = $OUT/.coverage
EOT
  export COVERAGE_PROCESS_START=$OUT/.coveragerc PYTHONPATH=$HERE/traincheck:${GATE_PYLIB:-$C/tooling/pylib}${PYTHONPATH:+:$PYTHONPATH}
fi
cd "$TREE/hy3dpaint"
python train.py --base "$OUT/traincheck.yaml" --gpus 0, --logdir "$OUT/logs" --name check "${RESUME[@]}"

REC=$OUT/record.jsonl
env -u HUNYUAN_TRAINCHECK_OUT python - "$REC" <<'EOF'
import json, math, sys
recs = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
train = [r for r in recs if r["kind"] == "train"]
if not train:
    sys.exit("ABORT: no training step in the record")
bad = [r["gs"] for r in train if not math.isfinite(r["loss"])]
if bad:
    sys.exit(f"ABORT: non-finite training loss at global steps {bad}")
EOF
SAVED=$(find "$OUT/logs" -name '*.ckpt' | sort)
[ "$(echo "$SAVED" | grep -c .)" = 2 ] || { echo "ABORT: expected two checkpoints (main and trough), got: $SAVED"; exit 3; }
for f in $SAVED; do
  env -u HUNYUAN_TRAINCHECK_OUT PYTHONPATH= python "$HERE/traincheck/summarize_ckpt.py" "$f" "$OUT/logs" >> "$REC"
done
if [ "${HUNYUAN_COVERAGE:-0}" = 1 ]; then
  ( unset COVERAGE_PROCESS_START; export PYTHONPATH=${GATE_PYLIB:-$C/tooling/pylib}; cd "$OUT" && python -m coverage combine --rcfile .coveragerc -q && python -m coverage json --rcfile .coveragerc -q -i -o coverage.json ) || { echo "ABORT: coverage combine failed"; exit 4; }
fi
echo TRAINCHECK_DONE
