#!/bin/bash
#SBATCH --account=share-ie-idi
#SBATCH --job-name=yolo-train-iphone
#SBATCH --time=0-24:00:00

#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8

#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

#SBATCH --mail-user=tryggvek@ntnu.no
#SBATCH --mail-type=ALL

WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}
set -euo pipefail
echo "Running from this directory: $SLURM_SUBMIT_DIR"
echo "Name of job: $SLURM_JOB_NAME"
echo "ID of job: $SLURM_JOB_ID"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"

module purge
module load Miniconda3/24.7.1-0

echo "Using conda at: $(command -v conda)"
conda env list | grep tdt4265-mini-project || true

# Quick runtime checks
conda run -n tdt4265-mini-project python -c "import torch; print('torch', torch.__version__); print('cuda_available', torch.cuda.is_available()); print('cuda_device_count', torch.cuda.device_count())"

# Run training
conda run -n tdt4265-mini-project python models/yolo/train_yolo.py --optuna --n-trials 30 --optuna-epochs 150 --continue-from-model "models/yolo/weights/v1/v2.pt"
