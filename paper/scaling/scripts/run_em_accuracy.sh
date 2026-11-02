#!/bin/bash
#SBATCH --job-name=scaling_emacc
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/emacc_%j.out
#SBATCH --error=logs/emacc_%j.out

# At fixed N=10, demonstrate that EM's accuracy depends on the random
# chain — some systems converge at small n_time_steps, others need
# 10-100x more.

PYTHON=~/qmaxcal_venv/bin/python
# Submit from the repository root: sbatch paper/scaling/scripts/run_em_accuracy.sh
cd "${SLURM_SUBMIT_DIR:-$PWD}"
[ -f train_qmaxcal.py ] || { echo "Submit this script from the repository root." >&2; exit 1; }

$PYTHON -u paper/scaling/src/bench_em_accuracy.py

echo "done $(date)"
