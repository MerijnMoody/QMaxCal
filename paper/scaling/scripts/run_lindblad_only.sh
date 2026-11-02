#!/bin/bash
#SBATCH --job-name=scaling_lindblad_bigmem
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=256G
#SBATCH --time=02:00:00
#SBATCH --output=logs/lindblad_%j.out
#SBATCH --error=logs/lindblad_%j.out

# Push the Lindblad master-equation solver to high N until it OOMs.
# Reports wall-clock and peak GPU memory per N; stops at first failure.

PYTHON=~/qmaxcal_venv/bin/python
# Submit from the repository root: sbatch paper/scaling/scripts/run_lindblad_only.sh
cd "${SLURM_SUBMIT_DIR:-$PWD}"
[ -f train_qmaxcal.py ] || { echo "Submit this script from the repository root." >&2; exit 1; }

LINDBLAD_OUT=paper/scaling/results/lindblad_only_mem256.json $PYTHON -u paper/scaling/src/bench_lindblad_only.py

echo "done $(date)"
