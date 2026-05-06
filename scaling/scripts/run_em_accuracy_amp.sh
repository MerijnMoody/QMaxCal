#!/bin/bash
#SBATCH --job-name=scaling_emamp
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=scaling/logs/emamp_%j.out
#SBATCH --error=scaling/logs/emamp_%j.out

# Vary pulse amplitude at N=10 to expose where EM fails: stiffer
# Hamiltonians need finer dt for the same accuracy.

PYTHON=~/qmaxcal_venv/bin/python
cd /home/mmoody/qmaxcal_paper

$PYTHON -u scaling/src/bench_em_accuracy_amp.py

echo "done $(date)"
