#!/bin/bash
#SBATCH --job-name=diffusion
#SBATCH --output=slurm/%x.%j.out
#SBATCH --error=slurm/%x.%j.err
#SBATCH --open-mode=append
#SBATCH --account=aip-lsigal
#SBATCH --mail-user=g202302610@kfupm.edu.sa
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --gpus-per-node=l40s:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=200G
#SBATCH --time=1-00:00:00

module load StdEnv/2023 gcc/12.3 cuda/13.2 arrow/23.0.1 python/3.11.5
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip

# SUBMIT THIS FROM THE REPO ROOT (i.e. `sbatch diffusion.slurm`). SBATCH --output and
# every path below are relative to that dir, where configs/, components/, scripts/,
# work_dir/, slurm/, and lm-evaluation-harness/ all live.
bash ./scripts/install.sh

# Pass an existing work_dir as $1 to resume a run that got cut off (e.g. hit the
# SLURM time limit) instead of starting a fresh one:
#   sbatch diffusion.slurm ./work_dir/Qwen1.5B-3B_diffusion_11_12_54_42
# main.py auto-detects the latest checkpoint inside it and resumes from there.
if [ -n "$1" ]; then
    work_dir="$1"
else
    timestamp=$(date +"%d_%H_%M_%S")
    work_dir="./work_dir/Qwen1.5B-3B_diffusion_${timestamp}"
fi

export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_DISABLE_XET=1

# Data-parallel (DDP) training: one process per GPU. HF Trainer reads LOCAL_RANK /
# WORLD_SIZE from torchrun and wraps the student in DistributedDataParallel. Each rank
# holds a full student+teacher copy on its own GPU; GRADIENT_ACCUMULATION_STEPS is
# divided by WORLD_SIZE in main.py so the effective batch size stays fixed and
# wall-clock drops ~linearly with GPU count.
NGPU=$(echo "$SLURM_GPUS_PER_NODE" | sed 's/.*://')
torchrun --standalone --nproc_per_node="$NGPU" main.py \
    --model-type diffusion --work-dir "$work_dir" --config configs/diffusion.yaml --slurm-mode
