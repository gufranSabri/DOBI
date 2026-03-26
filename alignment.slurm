#!/bin/bash
#SBATCH --job-name=alignment
#SBATCH --output=slurm/%x.%j.out
#SBATCH --error=slurm/%x.%j.err
#SBATCH --open-mode=append
#SBATCH --account=aip-lsigal
#SBATCH --mail-user=g202302610@kfupm.edu.sa
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --gpus-per-node=l40s:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=60G
#SBATCH --time=12:00:00

module load StdEnv/2023 gcc/12.3 cuda/12.2 arrow/23.0.1 python/3.11.5
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
bash ./scripts/install.sh

timestamp=$(date +"%d_%H_%M_%S")
work_dir="./work_dir/aligned_50k__${timestamp}"

export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_DISABLE_XET=1
python alignment.py --work-dir "$work_dir" --slurm-mode