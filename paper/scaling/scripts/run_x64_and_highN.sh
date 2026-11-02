#!/bin/bash
#SBATCH --job-name=scaling_x64
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=01:30:00
#SBATCH --output=logs/x64_%j.out
#SBATCH --error=logs/x64_%j.out

# Two-part diagnostic in one job:
#  (1) precision check at N=10 with jax_enable_x64 — does the bias
#      collapse when we run SSE in complex128 like the Lindblad solver does?
#  (2) high-N SSE smoke at N=12, 14, 16 (x64) — does SSE work where
#      the Lindblad solver OOMs at N=12?
#
# 128 GB host RAM is enough for x64 Lindblad at N=10; we don't run
# Lindblad at N >= 12 (it OOMs in GPU memory regardless of host).

PYTHON=~/qmaxcal_venv/bin/python
# Submit from the repository root: sbatch paper/scaling/scripts/run_x64_and_highN.sh
cd "${SLURM_SUBMIT_DIR:-$PWD}"
[ -f train_qmaxcal.py ] || { echo "Submit this script from the repository root." >&2; exit 1; }

$PYTHON -u paper/scaling/src/bench_x64_and_highN.py

echo "done $(date)"
