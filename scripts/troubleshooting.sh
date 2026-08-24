# salloc --gpus-per-node=l40s:1 --cpus-per-task=6 --mem=60G --time=12:00:00 --account=aip-lsigal
# salloc --gpus-per-node=l40s:1 --cpus-per-task=6 --mem=60G --time=3:00:00 --account=aip-lsigal
# salloc --gpus-per-node=l40s:1 --cpus-per-task=6 --mem=60G --time=3:00:00 --account=aip-lsigal

# salloc --gpus-per-node=h100:2 --cpus-per-task=6 --mem=60G --time=1:00:00 --account=aip-lsigal
# salloc --gpus-per-node=h100:1 --cpus-per-task=6 --mem=60G --time=3:00:00 --account=aip-lsigal
# salloc --gpus-per-node=h100:1 --cpus-per-task=6 --mem=60G --time=12:00:00 --account=aip-lsigal

module load StdEnv/2023 gcc/12.3 cuda/13.2 arrow/23.0.1 python/3.11.5
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip && bash ./scripts/install.sh

# ========================

# rm -r work_dir/test && python main.py

# ── Multi-GPU (DDP) ──────────────────────────────────────────────────────────
# One process per GPU, each holding a full teacher+student copy on its own
# cuda:{LOCAL_RANK}. GRADIENT_ACCUMULATION_STEPS is divided by the GPU count in
# flow.py's __main__, so the effective batch size stays fixed and wall-clock
# drops ~linearly. Allocate matching GPUs first (e.g. salloc --gpus-per-node=l40s:2 ...).

# Single-GPU (baseline / regression check):
# python diffusion.py

# Multi-GPU training (set --nproc_per_node to #GPUs allocated):
# torchrun --standalone --nproc_per_node=2 flow.py 

# Multi-GPU benchmark (set --num_processes to #GPUs; --limit for a quick subset):
# accelerate launch --multi_gpu --num_processes 2 benchmark.py --work-dir /home/ahmedubc/scratch/DOBI-ckp/Qwen1.5B-3B_diffusion_19_05_56_34 --model Qwen/Qwen2.5-1.5B-Instruct --num-steps 50