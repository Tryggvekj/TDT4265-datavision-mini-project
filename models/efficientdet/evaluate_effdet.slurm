#!/bin/bash
#SBATCH --account=share-ie-idi
#SBATCH --job-name=effdet_eval_896
#SBATCH --time=0-04:00:00
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:v100:1
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

module purge
module load Miniconda3/24.7.1-0

CONDA_BIN=/cluster/apps/eb/software/Miniconda3/24.7.1-0/bin/conda
ENV_PATH=/cluster/home/svanhigb/.conda/envs/efficientdet

srun ${CONDA_BIN} run -p ${ENV_PATH} --no-capture-output \
python models/efficientdet/evaluate_efficientdet.py \
  --model-path models/efficientdet/weights_finetune896_lr/best_model.pth
  --model tf_efficientdet_d3 \
  --num-classes 1 \
  --img-size 896 \
  --split valid \
  --batch-size 4 \
  --num-workers ${SLURM_CPUS_PER_TASK} \
  --output-dir models/efficientdet/saved_models/finetune896_lr_eval