#!/bin/bash
#SBATCH --job-name=scaling_lindblad_bigmem
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=256G
#SBATCH --time=02:00:00
#SBATCH --output=scaling/logs/lindblad_%j.out
#SBATCH --error=scaling/logs/lindblad_%j.out

# Push the Lindblad master-equation solver to high N until it OOMs.
# Reports wall-clock and peak GPU memory per N; stops at first failure.

PYTHON=~/qmaxcal_venv/bin/python
ROOT="$(cd "$(dirname "$0")" && pwd)"
while [ ! -f "$ROOT/train_qmaxcal.py" ] && [ "$ROOT" != "/" ]; do ROOT="$(dirname "$ROOT")"; done
cd "$ROOT"

LINDBLAD_OUT=scaling/results/lindblad_only_mem256.json $PYTHON -u scaling/src/bench_lindblad_only.py

echo "done $(date)"
