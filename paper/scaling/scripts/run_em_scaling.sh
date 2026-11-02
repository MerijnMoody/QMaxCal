#!/bin/bash
#SBATCH --job-name=scaling_em
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=logs/em_%j.out
#SBATCH --error=logs/em_%j.out

# Wall-clock + memory comparison: EulerMaruyama vs ExponentialSplit
# at increasing N (4..14).  --mem=128G to clear the JIT-lowering host-RAM
# wall at N>=12.

PYTHON=~/qmaxcal_venv/bin/python
# Submit from the repository root: sbatch paper/scaling/scripts/run_em_scaling.sh
cd "${SLURM_SUBMIT_DIR:-$PWD}"
[ -f train_qmaxcal.py ] || { echo "Submit this script from the repository root." >&2; exit 1; }

$PYTHON -u paper/scaling/src/bench_em_scaling.py

echo "done $(date)"
