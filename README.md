# TDT4265 Datavision Mini Project

Deep learning based pole detection project.

## 1. Connect to IDUN

To use VS Code Remote Explorer, add this to `~/.ssh/config`:

```sshconfig
Host login-node
	HostName idun-login1.hpc.ntnu.no
	User <your_username>
```

Then connect in VS Code to host `login-node`.

Clone or move into the project folder:

```bash
cd /cluster/home/<your_username>/TDT4265-datavision-mini-project
```

## 2. Activate Conda Environment

Load the module and activate the environment in each new shell:

```bash
module purge
module load Miniconda3/24.7.1-0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate tdt4265-mini-project
```

If the environment does not exist yet:

```bash
conda create -n tdt4265-mini-project python=3.10 -y
conda activate tdt4265-mini-project
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## 4. Train on IDUN (SLURM)

Submit training job:

```bash
sbatch models/yolo/train_yolo.slurm
```

For EfficientDet training, use:

```bash
sbatch data_EfficientDet/train_efficientdet.slurm
```

After running, you wil get a jobid which you can use to check output log and error log.

Monitor jobs:

```bash
squeue -u $USER
```

Check logs:

```bash
tail -f slurm-<JOBID>.out
tail -f slurm-<JOBID>.err
```