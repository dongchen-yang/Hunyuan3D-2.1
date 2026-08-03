#!/bin/bash
# LightGen phase-2 Task 3: 2-GPU DDP smoke on the 10-shape pilot fixture.
# Same node/GPU type/env as train_74k.sbatch, sized down to an hour. See
# cfgs/lightgen-emission-ddp-smoke.yaml for what it is proving.
#
# Submit twice with different LIGHTGEN_RUN_TS to also prove logdir isolation:
#   sbatch scripts/lightgen/ddp_smoke.sbatch
#SBATCH -w cs-venus-05
#SBATCH --gres=gpu:rtx_pro_6000_blackwell_max-q:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=1:00:00
#SBATCH --job-name=mvpaint_ddp_smoke
#SBATCH --output=/localscratch/dya78/lightgen_mvpaint/%x-%j.out
set -euo pipefail
source /localscratch/dya78/miniconda3/etc/profile.d/conda.sh
conda activate hunyuanpaint
cd /localscratch/dya78/lightgen_mvpaint/Hunyuan3D-2.1-emissive
git pull --ff-only
cd hy3dpaint
export PYTHONNOUSERSITE=1
export TORCH_CUDA_ARCH_LIST="12.0"
export HF_HOME=/localscratch/dya78/lightgen_mvpaint/hf_home
export WANDB_MODE=offline
export LIGHTGEN_WANDB_NAME='mvpaint_ddp_smoke'
export LIGHTGEN_RUN_TS=${LIGHTGEN_RUN_TS:-$(date +%Y%m%d-%H%M%S)}
echo "LIGHTGEN_RUN_TS=$LIGHTGEN_RUN_TS"
mkdir -p logs
nvidia-smi --query-gpu=index,memory.used --format=csv -l 20 > /localscratch/dya78/lightgen_mvpaint/vram-$SLURM_JOB_ID.log &
python train.py --base cfgs/lightgen-emission-ddp-smoke.yaml --name smoke --logdir logs/ --gpus 0,1
