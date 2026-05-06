#!/bin/bash
#SBATCH --job-name=scaling_emacc
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=scaling/logs/emacc_%j.out
#SBATCH --error=scaling/logs/emacc_%j.out

# At fixed N=10, demonstrate that EM's accuracy depends on the random
# chain — some systems converge at small n_time_steps, others need
# 10-100x more.

PYTHON=~/qmaxcal_venv/bin/python
cd /home/mmoody/qmaxcal_paper

$PYTHON -u scaling/src/bench_em_accuracy.py

echo "done $(date)"
