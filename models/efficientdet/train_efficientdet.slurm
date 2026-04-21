#!/bin/bash
#SBATCH --account=share-ie-idi
#SBATCH --job-name=effdet_d4_1024
#SBATCH --time=0-24:00:00
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --constraint=gpu32g|gpu40g|gpu80g
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1

WORKDIR="${SLURM_SUBMIT_DIR}"
cd "${WORKDIR}"

echo "Running from: ${SLURM_SUBMIT_DIR}"
echo "Job name: ${SLURM_JOB_NAME}"
echo "Job id: ${SLURM_JOB_ID}"
echo "Node list: ${SLURM_JOB_NODELIST}"

module purge
module load Miniconda3/24.7.1-0

CONDA_BIN=/cluster/apps/eb/software/Miniconda3/24.7.1-0/bin/conda
ENV_PATH=/cluster/home/svanhigb/.conda/envs/efficientdet

echo "Conda binary: ${CONDA_BIN}"
echo "Env path: ${ENV_PATH}"

${CONDA_BIN} run -p ${ENV_PATH} --no-capture-output python -c "
import sys, torch
print('python', sys.executable)
print('torch', torch.__version__)
print('cuda_available', torch.cuda.is_available())
print('device_count', torch.cuda.device_count())
print('device_name', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')
"

srun ${CONDA_BIN} run -p ${ENV_PATH} --no-capture-output \
python models/efficientdet/train_efficientdet.py \
  --model tf_efficientdet_d4 \
  --num-classes 1 \
  --img-size 1024 \
  --epochs 50 \
  --batch-size 8 \
  --lr 4e-5 \
  --num-workers ${SLURM_CPUS_PER_TASK} \
  --output-dir models/efficientdet/Roadpoles_v1/weights_img1024_BC8
  
