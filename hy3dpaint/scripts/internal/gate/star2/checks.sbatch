#!/bin/bash
# The refactor checks on one Solar node (after TEXGen-Emission's scripts/internal/gate/star2/checks.sbatch):
# inference gates and training checks for trees under test, one queue per GPU. GPU type = whatever the
# node has; compare a run only with runs on the same type (REFACTOR_TODO.md, How we work).
#
# Shared $HOME (H=~/hunyuan_checks), synced from the workstation before a submit:
#   rsync -a <tree>/ star2:hunyuan_checks/trees/<name>/                 (a git archive with hy3dpaint/)
#   rsync -a --delete hy3dpaint/scripts/internal/gate/ star2:hunyuan_checks/tooling/gate/
#   rsync -a $C/tooling/repo/ star2:hunyuan_checks/tooling/repo/        (the frozen predictor)
#   rsync -a $C/inputs/ star2:hunyuan_checks/inputs/                    (paper, release, train)
# Node staging (/localscratch/dya78/hunyuan_checks, once per node, flock'd): the two published troughs
# from jupiter (read-only), sha256-checked, and their weights-only copies made here with slim_ckpt.py,
# checked against the workstation's sha256; the inputs, checked against their MANIFEST.sha256.
#
# Submit: never `sbatch --export` on star2; bake the placeholders with sed and feed sbatch on stdin:
#   sed -e 's|__LABEL__|<label>|' -e 's|__TASKS__|tasks/<label>.txt|' -e 's|__NGPU__|3|' \
#       -e 's|__GRES__|gpu:3|' -e 's|__NCPU__|24|' -e 's|__MEM__|160G|' -e 's|__TIME__|03:00:00|' \
#       checks.sbatch | ssh star2 'sbatch --deadline=now+<time + 3 min>'
# No `-w` (a pinned job on our own nodes evicts another lab's running job); --deadline makes a job
# that cannot start at once go away instead of waiting.
# Task file (under $H), one task per line:  <gpu> <name> paper|release|fresh|resume[+cov][+fork] <tree> [<shas>]
#   <shas>: a file under $H, or "subset20" (the run's inputs subset20.txt); omitted: every shape.
# Output: $H/out/__LABEL__/<name>/ (the views, or record.jsonl), gpu.txt, logs/, status.txt
# (one line per task, then CHECKS_DONE or CHECKS_INCOMPLETE).
#SBATCH --job-name=hunyuan_checks___LABEL__
#SBATCH --gres=__GRES__
#SBATCH --cpus-per-task=__NCPU__
#SBATCH --mem=__MEM__
#SBATCH --time=__TIME__
#SBATCH --output=/home/dya78/hunyuan_checks/logs/checks-%j.out
set -uo pipefail
LABEL="__LABEL__"; NGPU=__NGPU__; TASKFILE="__TASKS__"
H=$HOME/hunyuan_checks
LS=/localscratch/dya78/hunyuan_checks; OUT=$LS/out/$LABEL; HOUT=$H/out/$LABEL
JUP=/3dlg-jupiter-project/lightgen; HFH=$JUP/trellis2_bw/eval_inputs/hf_home
SNAP=$HFH/hub/models--tencent--Hunyuan3D-2.1/snapshots/0b94677654c57bb9a6b6845cd7b704ccf551d327/hunyuan3d-paintpbr-v2-1
say() { echo "[checks $(date +%H:%M:%S)] $*"; }
SYNC="rsync -rlt"
die() { echo "CHECKS_FAIL label=$LABEL node=$(hostname -s) job=${SLURM_JOB_ID:-} reason=$*"
        mkdir -p "$HOUT" && echo "CHECKS_FAIL $*" > "$HOUT/status.txt"; exit 1; }
trap 'exit 143' TERM
say "node=$(hostname -s) job=${SLURM_JOB_ID:-} label=$LABEL gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | sort | uniq -c | xargs)"
[ -f "$H/tooling/gate/gate.sh" ] && [ -f "$H/tooling/repo/evaluation/mvpaint_bridge/predict_emission_views.py" ] || die "tooling missing under $H"
[ -f "$H/$TASKFILE" ] || die "no task file $H/$TASKFILE"
[ -e "$OUT" ] && { echo "CHECKS_FAIL $OUT exists; use a new label"; exit 1; }   # not die: it would overwrite that run's status
[ -d "$SNAP" ] || die "no snapshot at $SNAP"
[ -e "$HOUT" ] && { echo "CHECKS_FAIL $HOUT exists; use a new label"; exit 1; }
mkdir -p "$LS" "$OUT/logs" "$HOUT/logs" || die "mkdir"
# this job's own copy of the gate tooling: a later sync to $H cannot change scripts under a running job
GT=$OUT/gate; $SYNC "$H/tooling/gate/" "$GT/" || die "tooling copy"
# Slurm's device list for this job: queue g runs on its g-th entry
IFS=, read -ra DEVS <<< "${CUDA_VISIBLE_DEVICES:-$(seq -s, 0 $((NGPU-1)))}"
[ "${#DEVS[@]}" -ge "$NGPU" ] || die "the job has ${#DEVS[@]} GPUs, the tasks need $NGPU"
FREE_GB=$(df -BG --output=avail /localscratch | tail -1 | tr -dc 0-9)
say "/localscratch free: ${FREE_GB} GB"
# Staging takes ~55 GB once per node; a training check writes ~36 GB more.
NEED=80; grep -qE '^[0-9]+ [^ ]+ (fresh|resume)' "$H/$TASKFILE" && NEED=150
[ "$FREE_GB" -ge "$NEED" ] || [ -f "$LS/ckpts/mvpaint_alpha_thumb_lightgenbench_v1/ckpts/troughs/.ok_trough-step=00040000.weights.ckpt" -a "$FREE_GB" -ge 80 ] \
  || die "only ${FREE_GB} GB free on /localscratch, $NEED needed (staging ~55 GB, each training check writes ~36 GB)"

ACT=$OUT/activate.sh
cat > "$ACT" <<EOF
source $HOME/envs/hunyuanpaint/bin/activate
export HF_HOME=$HFH HF_HUB_CACHE=$HFH/hub TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
EOF

# ---- staging: checkpoints once per node, inputs and tooling every run ----
exec 8>"$LS/.stage.lock"; flock 8
stage_ckpt() {  # run file sha256 slim_sha256
  local d=$LS/ckpts/$1/ckpts/troughs f=$2
  if [ ! -f "$d/.ok_$f" ]; then
    mkdir -p "$d"
    $SYNC "$JUP/outputs/$1/config.yaml" "$LS/ckpts/$1/" || return 1
    say "stage: $1/$f"; $SYNC --inplace "$JUP/outputs/$1/ckpts/troughs/$f" "$d/$f.part" && mv "$d/$f.part" "$d/$f" || return 1
    [ "$(sha256sum "$d/$f" | cut -d' ' -f1)" = "$3" ] || { echo "sha256 mismatch $f"; return 1; }
    touch "$d/.ok_$f"
  fi
  local w=${f%.ckpt}.weights.ckpt
  if [ ! -f "$d/.ok_$w" ]; then
    ( set +u; source "$ACT"; set -u; python "$GT/slim_ckpt.py" "$d/$f" "$d/$w" ) || return 1
    [ "$(sha256sum "$d/$w" | cut -d' ' -f1)" = "$4" ] || { echo "sha256 mismatch $w (not the workstation's copy)"; return 1; }
    touch "$d/.ok_$w"
  fi
}
stage_ckpt mvpaint_74k_alpha_thumb_agentic 'trough-step=00050000.ckpt' \
  cbbc902b3ede9a87d8bf32b42f84db02406387a7f8ac4b559d31343d140f2d06 7ea7cdd756f35fb1599fc2bd188085b02bf6518e4cb0bbb6b25cf1a2b136d19d || die "paper ckpt"
stage_ckpt mvpaint_alpha_thumb_lightgenbench_v1 'trough-step=00040000.ckpt' \
  88260ff0a0e49ba049c904e791cdba0df3e629ffbd3cbf9da2699033b1fbe144 d21f3bc74ce61c3ff39a9bd8dfe5bc6c50158093df7d4eb8a2b8d8e2a7475c7e || die "release ckpt"
mkdir -p "$LS/inputs" || die "mkdir inputs"
for r in paper release train; do
  $SYNC --delete "$H/inputs/$r/" "$LS/inputs/$r/" && (cd "$LS/inputs/$r" && sha256sum -c --quiet MANIFEST.sha256) || die "inputs $r"
done
mkdir -p "$LS/tooling" && $SYNC --delete "$H/tooling/repo/" "$LS/tooling/repo/" || die "tooling"
[ ! -d "$H/tooling/pylib" ] || $SYNC --delete "$H/tooling/pylib/" "$LS/tooling/pylib/" || die "pylib"
flock -u 8
( set +u; source "$ACT"; set -u; python -c "import torch; print('[probe] torch', torch.__version__, torch.cuda.device_count(), 'GPUs')" ) || die "env probe"

export GATE_PYLIB=$LS/tooling/pylib GATE_C=$LS GATE_ACTIVATE=$ACT GATE_SNAPSHOT=$SNAP GATE_LOCKS=none TRAINCHECK_TMPBASE=/localscratch/dya78/tmp
TASKS=()
while read -r g name kind tree shas; do
  [ -n "${g:-}" ] || continue
  [ -d "$H/trees/$tree/hy3dpaint" ] || die "task $name: no tree $H/trees/$tree"
  case $shas in "") ;; subset20) ;; *) shas=$H/$shas ;; esac
  case ${kind%%+*} in paper|release|fresh|resume) ;; *) die "task file: unknown kind $kind" ;; esac
  for x in $(tr + ' ' <<< "${kind#"${kind%%+*}"}"); do
    case $x in cov|fork) ;; *) die "task file: unknown suffix +$x in $kind" ;; esac
  done
  TASKS+=("$g|$name|$kind|$H/trees/$tree|$shas")
done < "$H/$TASKFILE"
run_queue() {
  local gq=$1 t g name kind tree shas rc s cov entry
  for t in "${TASKS[@]}"; do
    IFS='|' read -r g name kind tree shas <<< "$t"
    [ "$g" = "$gq" ] || continue
    cov=0; case +$kind+ in *+cov+*) cov=1 ;; esac
    entry=auto; case +$kind+ in *+fork+*) entry=fork ;; esac
    kind=${kind%%+*}
    say "launch $name ($kind, $(basename "$tree"), coverage=$cov, entry=$entry) on gpu $g"; s=$(date +%s)
    case $kind in
      paper|release) GATE_ENTRY=$entry HUNYUAN_COVERAGE=$cov CUDA_VISIBLE_DEVICES=${DEVS[$g]} bash "$GT/gate.sh" "$tree" "$OUT/$name" "$kind" ${shas:+"$shas"} > "$OUT/logs/$name.log" 2>&1 ;;
      fresh|resume)  HUNYUAN_COVERAGE=$cov CUDA_VISIBLE_DEVICES=${DEVS[$g]} bash "$GT/train-check.sh" "$tree" "$OUT/$name" "$kind" > "$OUT/logs/$name.log" 2>&1 ;;
    esac
    rc=$?; echo "$name exit=$rc secs=$(( $(date +%s) - s )) $(tail -1 "$OUT/logs/$name.log")" | tee -a "$OUT/status.g$gq"
  done
}
PIDS=()
for g in $(seq 0 $((NGPU-1))); do run_queue "$g" & PIDS+=("$!"); done
for p in "${PIDS[@]}"; do wait "$p"; done
cat "$OUT"/status.g* > "$OUT/status.txt" 2>/dev/null
RC=0
[ "$(grep -c . "$OUT/status.txt")" = "${#TASKS[@]}" ] || RC=1
grep -v -q -E ' exit=0 secs=[0-9]+ (GATE_DONE|TRAINCHECK_DONE)$' "$OUT/status.txt" && RC=1

# ---- collect: the views, gpu.txt and the records go to $HOME ----
for d in "$OUT"/*/; do
  d=${d%/}; n=$(basename "$d")
  case $n in logs|gate) continue ;; esac
  mkdir -p "$HOUT/$n"
  $SYNC "$d"/gpu.txt "$HOUT/$n/" 2>/dev/null
  [ -f "$d/coverage.json" ] && { $SYNC "$d/coverage.json" "$HOUT/$n/" || RC=1; }
  if [ -f "$d/record.jsonl" ]; then $SYNC "$d/record.jsonl" "$d/traincheck.yaml" "$HOUT/$n/" || RC=1; continue; fi
  $SYNC --include='*/' --include='*.png' --include='manifest.json' --exclude='*' "$d/" "$HOUT/$n/" || RC=1
done
$SYNC "$OUT"/logs/*.log "$HOUT/logs/" || RC=1
[ $RC -eq 0 ] && echo "CHECKS_DONE label=$LABEL job=${SLURM_JOB_ID:-}" >> "$OUT/status.txt" || echo "CHECKS_INCOMPLETE label=$LABEL job=${SLURM_JOB_ID:-}" >> "$OUT/status.txt"
$SYNC "$OUT/status.txt" "$HOUT/status.txt.part" && mv "$HOUT/status.txt.part" "$HOUT/status.txt" || RC=1
# the node's copy goes only after everything reached $HOME
[ $RC -eq 0 ] && rm -rf "$OUT" || say "kept $OUT (collection incomplete)"
cat "$OUT/status.txt" 2>/dev/null || cat "$HOUT/status.txt"
[ $RC -eq 0 ] || exit 2
