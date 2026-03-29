#!/bin/bash
#SBATCH --account=share-ie-idi
#SBATCH --job-name=effdet_d3_896
#SBATCH --time=0-00:30:00
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
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
python models/efficientdet/predict_submission.py \
  --model-path /cluster/home/svanhigb/TDT4265-datavision-mini-project/models/efficientdet/weights_finetune896_lr/best_model.pth \
  --model-name tf_efficientdet_d3 \
  --test-dir /cluster/projects/vc/courses/TDT17/ad/Poles2025/roadpoles_v1/test/images \
  --output-dir submissions_finetune896_lr \
  --img-size 896 \
  --conf-threshold 0.25 \
  --nms-threshold 0.4 \
  --create-zip