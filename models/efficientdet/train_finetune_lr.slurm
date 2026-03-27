#!/bin/bash
#SBATCH --account=share-ie-idi
#SBATCH --job-name=effdet-ft-lr
#SBATCH --time=0-08:00:00
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --output=slurm-ft-lr-%j.out
#SBATCH --error=slurm-ft-lr-%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1

cd ${SLURM_SUBMIT_DIR}

module purge
module load Miniconda3/24.7.1-0

CONDA_BIN=/cluster/apps/eb/software/Miniconda3/24.7.1-0/bin/conda
ENV_PATH=/cluster/home/svanhigb/.conda/envs/efficientdet

srun ${CONDA_BIN} run -p ${ENV_PATH} --no-capture-output \
python models/efficientdet/train_efficientdet.py \
  --model efficientdet_d0 \
  --num-classes 1 \
  --img-size 512 \
  --epochs 50 \
  --batch-size 8 \
  --lr 5e-5 \
  --train-csv data_EfficientDet/output/train_efficientdet.csv \
  --val-csv data_EfficientDet/output/valid_efficientdet.csv \
  --output-dir models/efficientdet/weights_finetune_lr \
  --resume models/efficientdet/saved_models/efficientdet_d0_baseline_best.pth