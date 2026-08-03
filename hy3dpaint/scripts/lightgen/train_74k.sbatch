#!/bin/bash
# LightGen phase-2 launch script: pbr->emission multiview fine-tune on the v2-campaign 74k
# split, 50,000 steps at global batch 8, two GPUs on cs-venus-05.
#
# Node pin: cs-venus-05 (RTX Pro 6000 Blackwell Max-Q x6, 96 GB each) -- the node the pilot was
# verified on, with the conda env, the HF snapshot and the render fixtures already staged under
# /localscratch/dya78/lightgen_mvpaint/. Two of its GPUs, not more: at staging time 5 of 6 were
# free and the courtesy rule is to leave co-tenants room. Re-check with
# `scontrol show node cs-venus-05 | grep -E 'CfgTRES|AllocTRES'` before resubmitting.
#
# WANDB_MODE=offline: venus-05 does have outbound HTTPS to api.wandb.ai, but we deliberately
# keep no wandb credential on a shared cluster node. Sync afterwards from the workstation:
#   wandb sync -p LightGen /localscratch/.../logs/lightgen-emission-74k-74k-<TS>/wandb/offline-run-*
#
# Submit:  sbatch scripts/lightgen/train_74k.sbatch
# Resume a crashed run in place (same logdir, picks up last.ckpt automatically):
#   LIGHTGEN_RUN_TS=<the original stamp> sbatch scripts/lightgen/train_74k.sbatch
#SBATCH -w cs-venus-05
#SBATCH --gres=gpu:rtx_pro_6000_blackwell_max-q:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=6-12:00:00
#SBATCH --job-name=mvpaint_74k
#SBATCH --output=/localscratch/dya78/lightgen_mvpaint/%x-%j.out
set -euo pipefail
# NOTE: --output must not point into a not-yet-created subdir; Slurm opens the output file
# BEFORE this script can run mkdir, so the parent dir must always exist at submission time.
#
# One Slurm task, two GPUs: pytorch-lightning's DDP strategy spawns the second rank itself
# (--gpus 0,1 -> devices=2). Do NOT set --ntasks-per-node=2; that would start two independent
# trainers, each of which would then try to spawn its own second rank.
source /localscratch/dya78/miniconda3/etc/profile.d/conda.sh
conda activate hunyuanpaint
cd /localscratch/dya78/lightgen_mvpaint/Hunyuan3D-2.1-emissive
git pull --ff-only
cd hy3dpaint
export PYTHONNOUSERSITE=1
export TORCH_CUDA_ARCH_LIST="12.0"
export HF_HOME=/localscratch/dya78/lightgen_mvpaint/hf_home
export WANDB_MODE=offline
export LIGHTGEN_WANDB_NAME='mvpaint_pbr->emission_74k_v2'
# Stamps the logdir so a resubmission cannot silently auto-resume an earlier run's last.ckpt.
# Exported BEFORE python and computed once here, not in train.py: lightning's DDP launcher
# re-execs train.py per rank, and each re-exec inherits this value rather than computing its
# own. Override it on the command line to deliberately resume an existing run.
export LIGHTGEN_RUN_TS=${LIGHTGEN_RUN_TS:-$(date +%Y%m%d-%H%M%S)}
echo "LIGHTGEN_RUN_TS=$LIGHTGEN_RUN_TS"
mkdir -p logs
nvidia-smi --query-gpu=index,memory.used --format=csv -l 60 > /localscratch/dya78/lightgen_mvpaint/vram-$SLURM_JOB_ID.log &
python train.py --base cfgs/lightgen-emission-74k.yaml --name 74k --logdir logs/ --gpus 0,1
