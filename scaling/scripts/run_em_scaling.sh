#!/bin/bash
#SBATCH --job-name=scaling_em
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=scaling/logs/em_%j.out
#SBATCH --error=scaling/logs/em_%j.out

# Wall-clock + memory comparison: EulerMaruyama vs ExponentialSplit
# at increasing N (4..14).  --mem=128G to clear the JIT-lowering host-RAM
# wall at N>=12.

PYTHON=~/qmaxcal_venv/bin/python
ROOT="$(cd "$(dirname "$0")" && pwd)"
while [ ! -f "$ROOT/train_qmaxcal.py" ] && [ "$ROOT" != "/" ]; do ROOT="$(dirname "$ROOT")"; done
cd "$ROOT"

$PYTHON -u scaling/src/bench_em_scaling.py

echo "done $(date)"
