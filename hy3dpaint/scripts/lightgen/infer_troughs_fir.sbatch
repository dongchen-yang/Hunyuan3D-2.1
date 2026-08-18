#!/bin/bash
# Stage-A scoring inference: run every LR-cycle-trough checkpoint over the 381-shape
# evaluation set, ON FIR, and emit only the predicted views.
#
# WHY ON FIR AND NOT THE WORKSTATION. The plan originally ran inference locally, which meant
# pulling each 17.9 GB checkpoint across the link -- 72 GB for four. The workstation has 101 GB
# free (94% used), so that does not fit alongside the outputs. But inference needs only the
# config, the checkpoint, the HF snapshot and the fixture, and ALL of those are already here;
# it is the SCORING that needs the TexVerse GLBs and df_metadata.parquet, which fir lacks.
# Splitting at that seam moves ~520 MB of predicted views instead of 72 GB of weights --
# measured 350 KB per shape per seed, so ~130 MB per checkpoint. 140x less data, and the
# disk problem disappears rather than being worked around.
#
# ONE SEED HERE, DELIBERATELY. This is Stage A: rank the troughs. Only the winner earns the
# 5-seed run that produces the reported number (see the plan's Task 7).
#
#SBATCH --account=def-msavva_gpu
#SBATCH --job-name=mvpaint_infer_troughs
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
# 288000M = the 1 GPU : 12 CPU : 288 GB ratio EXACTLY (the node is 1,152,000 MB for 4 GPUs),
# in decimal MB so the GiB-vs-GB trap cannot bite. Do NOT use --mem=0 here: that means "all
# the memory on the node", which is right for a whole-node job and is REJECTED for a
# single-GPU one -- `sbatch --test-only` returns only
#   allocation failure: Unspecified error
# naming neither memory nor the reason. Verified 2026-08-18: 0 fails, 288000M and 288G both
# schedule; 288000M is the one that matches the ratio rather than exceeding it by 7 GB.
#SBATCH --mem=288000M
#SBATCH --gres=gpu:h100:1
#SBATCH --output=/scratch/dya78/lightgen_mvpaint/%x-%j.out
#
# def-msavva_gpu, not rrg: this is evaluation, and the rrg allocation is for training.
# One GPU is enough -- the checkpoints are scored sequentially and each pass is ~80 min.
# PREREQUISITES staged from the workstation (neither lives in this submodule):
#   # predict_emission_views.py computes REPO_ROOT as parents[2] of its own path and imports
#   # evaluation.newdata_eval.seedutil, so it CANNOT be dropped in as a lone file -- it needs
#   # its position in a repo-shaped tree. Staging just the script gave
#   #   ModuleNotFoundError: No module named 'evaluation'
#   # ~1 minute into job 55307027, after the extract had already run.
#   mkdir -p $ROOT/repo/evaluation/{mvpaint_bridge,newdata_eval}
#   scp evaluation/mvpaint_bridge/predict_emission_views.py fir:$ROOT/repo/evaluation/mvpaint_bridge/
#   scp evaluation/newdata_eval/{seedutil.py,__init__.py}   fir:$ROOT/repo/evaluation/newdata_eval/
#   scp evaluation/newdata_eval/eval_sets/usable.txt         fir:/scratch/dya78/lightgen_mvpaint/v2_shas/usable381.txt
# Both are gated below, so a missing one fails in seconds rather than after the extract.
set -euo pipefail

ROOT=/scratch/dya78/lightgen_mvpaint
STAMP=20260816-013134
CK=$(ls -d $ROOT/Hunyuan3D-2.1-emissive/hy3dpaint/logs/*${STAMP}*/checkpoints 2>/dev/null | head -1)
OUT=$ROOT/trough_views_${STAMP}
SNAP=$ROOT/hf_home/hub/models--tencent--Hunyuan3D-2.1/snapshots/0b94677654c57bb9a6b6845cd7b704ccf551d327/hunyuan3d-paintpbr-v2-1
TARS=~/projects/rrg-msavva/dya78/lightgen_mv74k_alpha

[ -f "$ROOT/repo/evaluation/mvpaint_bridge/predict_emission_views.py" ] || { echo "ABORT: predict_emission_views.py not staged in the repo-shaped tree (see PREREQUISITES)"; exit 2; }
[ -f "$ROOT/repo/evaluation/newdata_eval/seedutil.py" ] || { echo "ABORT: seedutil.py not staged (see PREREQUISITES)"; exit 2; }
[ -s "$ROOT/v2_shas/usable381.txt" ] || { echo "ABORT: usable381.txt not staged (see PREREQUISITES)"; exit 2; }
[ -n "$CK" ] && [ -d "$CK/troughs" ] || { echo "ABORT: no troughs dir under $CK"; exit 2; }

echo "=== $(date -Is) host=$(hostname) job=$SLURM_JOB_ID ==="
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader
# GPU-OCCUPANCY GATE. Job 55318866 died with an opaque `CUDA error: out of memory` during
# model load on fc10512, while the identical job had just succeeded on fc10519 -- the
# allocations were byte-for-byte the same. fir's bygpu partitions hand out individual GPUs
# from shared nodes, so a card can arrive with another process's memory still resident.
# The old line printed memory.TOTAL, which is identical on a healthy and a busy card and
# therefore told us nothing. Fail here, naming the number, instead of 3 minutes later inside
# torch. 4096 MiB of slack covers driver/ECC overhead without masking a real squatter.
USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
if [ "${USED:-0}" -gt 4096 ]; then
    echo "ABORT: this GPU already has ${USED} MiB in use before we start -- another job is"
    echo "       resident on it. Resubmit; Slurm will usually place the retry elsewhere."
    exit 4
fi
echo "[gate] GPU free (${USED} MiB in use)"

module load StdEnv/2023 python/3.12 cuda/12.6
source "$ROOT/env/bin/activate"
export PYTHONNOUSERSITE=1 HF_HOME=$ROOT/hf_home HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

# Fixture -> node-local NVMe (same as training; ~55 s for all 12 shards).
mkdir -p "$SLURM_TMPDIR/mv74k"
for f in "$TARS"/shard_*.tar; do tar -xf "$f" -C "$SLURM_TMPDIR/mv74k" & done
wait
echo "extracted $(ls "$SLURM_TMPDIR/mv74k" | wc -l) shape dirs"

# The committed val381 list holds JUPITER paths, which do not exist here. Rewrite to this
# job's $SLURM_TMPDIR and assert every one resolves -- a silently short list would score a
# subset and land in the table as a real number.
python - "$ROOT/v2_shas/usable381.txt" "$SLURM_TMPDIR/mv74k" "$ROOT/val381_node.json" <<'PY'
import json, os, sys
lst, root, out = sys.argv[1], sys.argv[2], sys.argv[3]
shas = [l.strip() for l in open(lst) if l.strip()]
dirs = [os.path.join(root, s) for s in shas]
missing = [d for d in dirs if not os.path.isdir(os.path.join(d, "render_tex"))]
assert not missing, f"{len(missing)} eval dirs missing under {root}, e.g. {missing[:3]}"
noalpha = [d for d in dirs if not all(
    os.path.exists(os.path.join(d, "render_tex", f"{v:03d}_alpha.png")) for v in range(6))]
assert not noalpha, f"{len(noalpha)} eval dirs lack alpha maps, e.g. {noalpha[:3]}"
json.dump(dirs, open(out, "w"))
print(f"wrote {out}: {len(dirs)} dirs, all present, all alpha-complete")
PY

# Import gate. The previous failure surfaced only after the 55 s extract and the model load;
# checking the import first turns it into a 2-second abort.
python -c "
import sys; sys.path.insert(0, '$ROOT/repo')
from evaluation.newdata_eval.seedutil import sample_seed
print('[gate] seedutil imports OK')
" || { echo "ABORT: predict_emission_views.py's own imports do not resolve"; exit 2; }

cd "$ROOT/Hunyuan3D-2.1-emissive/hy3dpaint"
for CKPT in "$CK"/troughs/trough-step=*.ckpt; do
    TAG=$(basename "$CKPT" .ckpt | sed 's/trough-step=0*/t/')
    echo "=== $(date -Is) inference $TAG ($CKPT) ==="
    PYTHONPATH=$PWD python "$ROOT/repo/evaluation/mvpaint_bridge/predict_emission_views.py" \
        --config cfgs/lightgen-emission-74k-alpha-fir.yaml \
        --ckpt "$CKPT" \
        --snapshot "$SNAP" \
        --examples_json "$ROOT/val381_node.json" \
        --seeds 1 --seed_base 0 \
        --hy3dpaint_root "$ROOT/Hunyuan3D-2.1-emissive/hy3dpaint" \
        --out "$OUT/$TAG"
    echo "=== $TAG done: $(ls "$OUT/$TAG" | wc -l) shape dirs, $(du -sh "$OUT/$TAG" | cut -f1) ==="
done

echo "=== ALL TROUGHS DONE ==="
du -sh "$OUT"
