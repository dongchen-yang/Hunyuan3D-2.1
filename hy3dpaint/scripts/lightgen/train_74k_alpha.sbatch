#!/bin/bash
# LightGen conditioning-parity launch script: pbr->emission multiview fine-tune WITH the alpha
# (glTF opacity) condition, on the v2-campaign 74k split. 50,000 steps at global batch 8, two
# GPUs on cs-venus-05 -- the same recipe as train_74k.sbatch, from the same pretrained init, so
# the two runs form an equal-compute A/B whose only difference is the extra condition.
#
# PREREQUISITES (all of these, or the run trains on the wrong thing / dies at hour 0):
#   1. The jupiter fixture must carry the alpha maps:
#        python data_processing/multiview_render_pipeline/append_alpha_views.py \
#          --shas <v2 train+val+test shas> --workers 5
#      36 render_tex PNGs per shape, not 30.
#   2. Those alpha maps must be staged onto the node beside the rest:
#        rsync the *_alpha.png files into /localscratch/dya78/lightgen_mvpaint/mv74k/<sha>/render_tex/
#      (an --include='*_alpha.png' pass over the existing tree; the other 30 are already there).
#   3. The example JSONs must be regenerated with the updated make_74k_jsons.py, whose A7 gate
#      now requires 36 PNGs -- that gate is what proves step 2 actually landed. Regenerating is
#      not optional even though the paths are unchanged: the old JSONs were certified against a
#      30-PNG payload and would happily list a shape whose alpha never arrived.
#
# Node pin: cs-venus-05 (RTX Pro 6000 Blackwell Max-Q x6, 96 GB each). Two of its GPUs, not more:
# the courtesy rule is to leave co-tenants room. Re-check availability before submitting with
#   scontrol show node cs-venus-05 | grep -E 'CfgTRES|AllocTRES'
#
# WANDB_MODE=offline: venus-05 has outbound HTTPS, but we deliberately keep no wandb credential
# on a shared cluster node. Sync afterwards from the workstation:
#   wandb sync -p LightGen /localscratch/.../logs/lightgen-emission-74k-alpha-74k_alpha-<TS>/wandb/offline-run-*
#
# Submit:  sbatch scripts/lightgen/train_74k_alpha.sbatch
# Resume a crashed run in place (same logdir, picks up last.ckpt automatically):
#   LIGHTGEN_RUN_TS=<the original stamp> sbatch scripts/lightgen/train_74k_alpha.sbatch
#SBATCH -w cs-venus-05
#SBATCH --gres=gpu:rtx_pro_6000_blackwell_max-q:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
# 3 days: the no-alpha run measured ~4.1 s/step -> ~57 h for 50,000 steps, and one extra VAE
# encode per sample does not move that materially. Same ~26% headroom, same friendlier queue
# tier than a 6-day ask. A resume starts from last.ckpt, so the wall clock only has to cover the
# remaining steps.
#SBATCH --time=3-00:00:00
#SBATCH --job-name=mvpaint_74k_alpha
#SBATCH --output=/localscratch/dya78/lightgen_mvpaint/%x-%j.out
set -euo pipefail
# NOTE: --output must not point into a not-yet-created subdir; Slurm opens the output file
# BEFORE this script can run mkdir, so the parent dir must always exist at submission time.
#
# One Slurm task, two GPUs: pytorch-lightning's DDP strategy spawns the second rank itself
# (--gpus 0,1 -> devices=2), and --cpus-per-task=16 therefore has to cover both ranks and all
# their dataloader workers. Do NOT set --ntasks-per-node=2 to "fix" DDP: that would start two
# independent trainers, each of which would then try to spawn its own second rank. train.py
# pins LightningEnvironment for exactly this launch shape.
source /localscratch/dya78/miniconda3/etc/profile.d/conda.sh
conda activate hunyuanpaint
cd /localscratch/dya78/lightgen_mvpaint/Hunyuan3D-2.1-emissive
git pull --ff-only
cd hy3dpaint
export PYTHONNOUSERSITE=1
export TORCH_CUDA_ARCH_LIST="12.0"
export HF_HOME=/localscratch/dya78/lightgen_mvpaint/hf_home
export WANDB_MODE=offline
export LIGHTGEN_WANDB_NAME='mvpaint_pbr->emission_74k_v2_alpha'
# Stamps the logdir so a resubmission cannot silently auto-resume an earlier run's last.ckpt.
# Exported BEFORE python and computed once here, not in train.py: lightning's DDP launcher
# re-execs train.py per rank, and each re-exec inherits this value rather than computing its own.
export LIGHTGEN_RUN_TS=${LIGHTGEN_RUN_TS:-$(date +%Y%m%d-%H%M%S)}
echo "LIGHTGEN_RUN_TS=$LIGHTGEN_RUN_TS"
mkdir -p logs
nvidia-smi --query-gpu=index,memory.used --format=csv -l 60 > /localscratch/dya78/lightgen_mvpaint/vram-alpha-$SLURM_JOB_ID.log &
python train.py --base cfgs/lightgen-emission-74k-alpha.yaml --name 74k_alpha --logdir logs/ --gpus 0,1
