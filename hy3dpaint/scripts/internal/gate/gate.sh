#!/bin/bash
# The inference gate (REFACTOR_TODO.md, Runs and the gate): the parent's predictor, frozen into a copy,
# run against a tree under test for one published run, seed 0, 30 sampling steps.
#
#   gate.sh <tree> <out_dir> paper|release [<shas_file>]          (absolute paths)
#
# <tree> holds the fork's hy3dpaint/ (a git archive of the commit under test). Without a shapes file
# every shape of the run's inputs runs (paper 204, release 200); `subset20` means the inputs'
# subset20.txt. Output: <out_dir>/<sha>/seed00/00{0..5}_emission.png, manifest.json, gpu.txt.
# Prints GATE_DONE when the predictor wrote every shape. Exit 7: a GPU lock is held (never waits).
#
# Defaults are the workstation's; another host overrides them:
#   GATE_C         the check root (inputs/, ckpts/, tooling/repo/), default
#                  /local-scratch/localhome/dya78/lightgen_eval_local/hunyuan_refactor_checks
#   GATE_ACTIVATE  a script to source that activates the hunyuanpaint env (default: the workstation's)
#   GATE_SNAPSHOT  the Hunyuan3D-2.1 paint snapshot dir (rev 0b94677…)
#   GATE_LOCKS     lock files to take, space separated, or "none" (inside a Slurm job)
#   GATE_DETMODE=1 PyTorch's deterministic mode through detmode/sitecustomize.py
#   GATE_ENTRY=fork  run the tree's own hy3dpaint/predict_emission_views.py (step 13) instead of the
#                  frozen parent predictor; the same inputs, flags and output layout
set -euo pipefail
case $# in 3|4) ;; *) echo "usage: gate.sh <tree> <out_dir> paper|release [<shas_file>|subset20]"; exit 2 ;; esac
TREE=$1; OUT=$2; RUN=$3; SHAS=${4:-}
HERE=$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")
C=${GATE_C:-/local-scratch/localhome/dya78/lightgen_eval_local/hunyuan_refactor_checks}
SNAP=${GATE_SNAPSHOT:-/localhome/dya78/.cache/huggingface/hub/models--tencent--Hunyuan3D-2.1/snapshots/0b94677654c57bb9a6b6845cd7b704ccf551d327/hunyuan3d-paintpbr-v2-1}
LOCKS=${GATE_LOCKS:-/local-scratch/localhome/dya78/lightgen_4090.lock /local-scratch/localhome/dya78/.locks/gpu0-rtx4090.lock}
case $RUN in
  paper)   CKRUN=mvpaint_74k_alpha_thumb_agentic;       CKFILE=trough-step=00050000.ckpt ;;
  release) CKRUN=mvpaint_alpha_thumb_lightgenbench_v1;  CKFILE=trough-step=00040000.ckpt ;;
  *) echo "run must be paper or release"; exit 2 ;;
esac
IN=$C/inputs/$RUN
CKPT=$C/ckpts/$CKRUN/ckpts/troughs/${CKFILE%.ckpt}.weights.ckpt   # slim_ckpt.py's weights-only copy
CFG=$C/ckpts/$CKRUN/config.yaml                                    # the run's own config (jupiter)
case ${GATE_ENTRY:-parent} in
  parent) PRED=$C/tooling/repo/evaluation/mvpaint_bridge/predict_emission_views.py; ROOTARG=(--hy3dpaint_root "$TREE/hy3dpaint") ;;
  fork)   PRED=$TREE/hy3dpaint/predict_emission_views.py; ROOTARG=() ;;
  *) echo "GATE_ENTRY must be parent or fork"; exit 2 ;;
esac
[ "$SHAS" = subset20 ] && SHAS=$IN/subset20.txt
SHAS=${SHAS:-$IN/shas.txt}
for f in "$TREE/hy3dpaint/train.py" "$CKPT" "$CFG" "$PRED" "$SHAS" "$SNAP/model_index.json"; do
  [ -e "$f" ] || { echo "ABORT: missing $f"; exit 2; }
done
[ -e "$OUT" ] && { echo "ABORT: $OUT exists"; exit 2; }

# One GPU job at a time on the workstation 4090, shared with other lanes; never wait for a lock.
if [ "$LOCKS" != none ]; then
  fd=8
  for L in $LOCKS; do
    eval "exec $fd>\"$L\""
    flock -n $fd || { echo "ABORT: the GPU lock $L is held"; exit 7; }
    fd=$((fd + 1))
  done
  echo "[gpu-lock] taken $(date +%H:%M:%S): $LOCKS"
fi

set +u
if [ -n "${GATE_ACTIVATE:-}" ]; then source "$GATE_ACTIVATE"
else source /localhome/dya78/software/miniconda3/etc/profile.d/conda.sh; conda activate /localhome/dya78/software/miniconda3/envs/hunyuanpaint; fi
set -u
mkdir -p "$OUT"
# diffusers copies ./hunyuanpaintpbr into the modules cache on every load: one cache per run, so two
# runs of different trees can never import each other's copy.
export HF_MODULES_CACHE=$OUT/.hf_modules HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONNOUSERSITE=1
if [ "${GATE_DETMODE:-0}" = 1 ]; then
  export HUNYUAN_GATE_DETERMINISTIC=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONPATH=$HERE/detmode${PYTHONPATH:+:$PYTHONPATH}
fi
if [ "${HUNYUAN_COVERAGE:-0}" = 1 ]; then   # step 7: line coverage of the tree under test, every process
  cat > "$OUT/.coveragerc" <<EOT
[run]
parallel = True
concurrency = multiprocessing,thread
source = $TREE/hy3dpaint
data_file = $OUT/.coverage
EOT
  export COVERAGE_PROCESS_START=$OUT/.coveragerc PYTHONPATH=$HERE/detmode:${GATE_PYLIB:-$C/tooling/pylib}${PYTHONPATH:+:$PYTHONPATH}
fi
echo "host=$(hostname -s) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader -i "${CUDA_VISIBLE_DEVICES:-0}" | head -1) detmode=${GATE_DETMODE:-0} entry=${GATE_ENTRY:-parent}" | tee "$OUT/gpu.txt"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader || true

# The examples JSON: this host's absolute paths of the listed shapes, in the shapes file's order.
JSON=$OUT/examples.json
python - "$IN" "$SHAS" "$JSON" <<'EOF'
import json, os, sys
root, shas_file, out = sys.argv[1:]
shas = [l.strip() for l in open(shas_file) if l.strip()]
if not shas:
    sys.exit("the shapes file lists no shape")
dirs = [os.path.join(root, "shapes", s) for s in shas]
missing = [d for d in dirs if not os.path.isdir(d)]
if missing:
    sys.exit(f"{len(missing)} shapes are not in {root}: {missing[:3]}")
json.dump(dirs, open(out, "w"))
print(f"[gate] {len(dirs)} shapes")
EOF
N=$(python -c "import json,sys; print(len(json.load(open(sys.argv[1]))))" "$JSON")
python "$PRED" --ckpt "$CKPT" --config "$CFG" --examples_json "$JSON" --thumbnail_dir "$IN/thumbs" \
  --snapshot "$SNAP" ${ROOTARG[@]+"${ROOTARG[@]}"} --seeds 1 --seed_base 0 --val_steps 30 --out "$OUT"
got=$(find "$OUT" -mindepth 2 -maxdepth 2 -type d -name seed00 | wc -l)
[ "$got" = "$N" ] || { echo "ABORT: $got of $N shapes written"; exit 1; }
rm -rf "$OUT/.hf_modules"
if [ "${HUNYUAN_COVERAGE:-0}" = 1 ]; then
  ( unset COVERAGE_PROCESS_START; export PYTHONPATH=${GATE_PYLIB:-$C/tooling/pylib}; cd "$OUT" && python -m coverage combine --rcfile .coveragerc -q && python -m coverage json --rcfile .coveragerc -q -i -o coverage.json ) || { echo "ABORT: coverage combine failed"; exit 4; }
fi
echo GATE_DONE
