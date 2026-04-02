# salloc --gpus-per-node=l40s:1 --cpus-per-task=6 --mem=60G --time=12:00:00 --account=aip-lsigal
# salloc --gpus-per-node=l40s:1 --cpus-per-task=6 --mem=60G --time=3:00:00 --account=aip-lsigal

# salloc --gpus-per-node=h100:1 --cpus-per-task=6 --mem=60G --time=1:00:00 --account=aip-lsigal
# salloc --gpus-per-node=h100:1 --cpus-per-task=6 --mem=60G --time=3:00:00 --account=aip-lsigal
# salloc --gpus-per-node=h100:1 --cpus-per-task=6 --mem=60G --time=12:00:00 --account=aip-lsigal

module load StdEnv/2023 gcc/12.3 cuda/12.2 arrow/23.0.1 python/3.11.5
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip && bash ./scripts/install.sh

# ========================

rm -r work_dir/test && python main.py