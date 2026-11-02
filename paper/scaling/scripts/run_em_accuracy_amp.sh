#!/bin/bash
#SBATCH --job-name=scaling_emamp
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/emamp_%j.out
#SBATCH --error=logs/emamp_%j.out

# Vary pulse amplitude at N=10 to expose where EM fails: stiffer
# Hamiltonians need finer dt for the same accuracy.

PYTHON=~/qmaxcal_venv/bin/python
# Submit from the repository root: sbatch paper/scaling/scripts/run_em_accuracy_amp.sh
cd "${SLURM_SUBMIT_DIR:-$PWD}"
[ -f train_qmaxcal.py ] || { echo "Submit this script from the repository root." >&2; exit 1; }

$PYTHON -u paper/scaling/src/bench_em_accuracy_amp.py

echo "done $(date)"
