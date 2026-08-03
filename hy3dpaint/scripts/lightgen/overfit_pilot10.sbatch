#!/bin/bash
# LightGen Task 9 launch script: pbr->emission overfit10 training on venus.
#
# Node pin: cs-venus-05 (RTX Pro 6000 Blackwell Max-Q x6, 96 GB). Chosen over
# cs-venus-19 (Blackwell SE x4) because at staging time (2026-07-30) venus-19 was
# fully ALLOCATED (0/4 GPUs free) while venus-05 had 1/6 free
# (`scontrol show node` CfgTRES gres/gpu=6 vs AllocTRES gres/gpu=5). Re-check
# availability with the node-cheatsheet recipe before resubmitting if venus-05 fills up
# -- swap to `-w cs-venus-19 --gres=gpu:rtx_pro_6000_blackwell_se:1` if so.
#
# WANDB_MODE=offline: cs-venus-05 DOES have outbound HTTPS to api.wandb.ai (confirmed
# 2026-07-30, see task-8-report.md) but this job deliberately does not log online anyway,
# to avoid putting a wandb API credential on a shared cluster node. Sync afterwards from
# the workstation: `wandb sync -p LightGen /localscratch/dya78/lightgen_mvpaint/.../logs/.../wandb/offline-run-*`.
#SBATCH -w cs-venus-05
#SBATCH --gres=gpu:rtx_pro_6000_blackwell_max-q:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --job-name=mvpaint_overfit10
#SBATCH --output=/localscratch/dya78/lightgen_mvpaint/%x-%j.out
set -euo pipefail
# NOTE: --output must not point into a not-yet-created subdir; Slurm opens the output file
# BEFORE this script can run mkdir, so the parent dir must always exist at submission time.
source /localscratch/dya78/miniconda3/etc/profile.d/conda.sh
conda activate hunyuanpaint
cd /localscratch/dya78/lightgen_mvpaint/Hunyuan3D-2.1-emissive
git pull --ff-only
cd hy3dpaint
export PYTHONNOUSERSITE=1
export TORCH_CUDA_ARCH_LIST="12.0"
export HF_HOME=/localscratch/dya78/lightgen_mvpaint/hf_home
export WANDB_MODE=offline
export LIGHTGEN_WANDB_NAME='mvpaint_pbr->emission_overfit10_v1'
mkdir -p logs
nvidia-smi --query-gpu=memory.used --format=csv -l 60 > /localscratch/dya78/lightgen_mvpaint/vram-$SLURM_JOB_ID.log &
python train.py --base cfgs/lightgen-emission-overfit10.yaml --name overfit10 --logdir logs/ --gpus 0,
