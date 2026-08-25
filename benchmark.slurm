#!/bin/bash
#SBATCH --job-name=benchmark
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

module load StdEnv/2023 gcc/12.3 cuda/12.2 arrow/23.0.1 python/3.11.5
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip

# SUBMIT THIS FROM THE REPO ROOT.
# Usage:
#   sbatch benchmark.slurm hf   Qwen/Qwen2.5-1.5B-Instruct    [extra flags]
#   sbatch benchmark.slurm flow ./work_dir/<run>/flow_best    [extra flags]
bash ./scripts/install.sh

export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_DISABLE_XET=1

MODEL_TYPE="${1:?usage: sbatch benchmark.slurm <hf|flow> <model-or-checkpoint-path> [extra flags]}"
MODEL_PATH="${2:?usage: sbatch benchmark.slurm <hf|flow> <model-or-checkpoint-path> [extra flags]}"
shift 2

python benchmark.py \
    --model-type "$MODEL_TYPE" --model "$MODEL_PATH" \
    --work-dir "./work_dir/benchmark_${MODEL_TYPE}_$(date +%d_%H_%M_%S)" "$@"


# python benchmark.py --model-type flow --work-dir ~/scratch/DOBI_flow/flow_best