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
ROOT="$(cd "$(dirname "$0")" && pwd)"
while [ ! -f "$ROOT/train_qmaxcal.py" ] && [ "$ROOT" != "/" ]; do ROOT="$(dirname "$ROOT")"; done
cd "$ROOT"

$PYTHON -u scaling/src/bench_em_accuracy_amp.py

echo "done $(date)"
