#!/bin/bash
#SBATCH --job-name=scaling_x64
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=01:30:00
#SBATCH --output=scaling/logs/x64_%j.out
#SBATCH --error=scaling/logs/x64_%j.out

# Two-part diagnostic in one job:
#  (1) precision check at N=10 with jax_enable_x64 — does the bias
#      collapse when we run SSE in complex128 like Ernst does?
#  (2) high-N SSE smoke at N=12, 14, 16 (x64) — does SSE work where
#      Ernst OOMs at N=12?
#
# 128 GB host RAM is enough for x64 Lindblad at N=10; we don't run
# Lindblad at N >= 12 (it OOMs in GPU memory regardless of host).

PYTHON=~/qmaxcal_venv/bin/python
cd /home/mmoody/qmaxcal_paper

$PYTHON -u scaling/src/bench_x64_and_highN.py

echo "done $(date)"
