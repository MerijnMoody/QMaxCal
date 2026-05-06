#!/bin/bash
#SBATCH --job-name=scaling_ernst_bigmem
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=256G
#SBATCH --time=02:00:00
#SBATCH --output=scaling/logs/ernst_%j.out
#SBATCH --error=scaling/logs/ernst_%j.out

# Push Ernst's Lindblad master-equation solver to high N until it OOMs.
# Reports wall-clock and peak GPU memory per N; stops at first failure.

PYTHON=~/qmaxcal_venv/bin/python
cd /home/mmoody/qmaxcal_paper

ERNST_OUT=scaling/results/ernst_only_mem256.json $PYTHON -u scaling/src/bench_ernst_only.py

echo "done $(date)"
