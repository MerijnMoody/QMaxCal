#!/bin/bash
#SBATCH --job-name=scaling_mc
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/mc_%j.out
#SBATCH --error=logs/mc_%j.out

# x32 throughout.
#  Part 1: n_trajs sweep at N=10 with 10 reseeds  -> clean MC convergence
#  Part 2: SSE forward pass at N=12, 14, 16        -> high-N reach
PYTHON=~/qmaxcal_venv/bin/python
# Submit from the repository root: sbatch paper/scaling/scripts/run_mc_and_highN.sh
cd "${SLURM_SUBMIT_DIR:-$PWD}"
[ -f train_qmaxcal.py ] || { echo "Submit this script from the repository root." >&2; exit 1; }

$PYTHON -u paper/scaling/src/bench_mc_and_highN.py

echo "done $(date)"
