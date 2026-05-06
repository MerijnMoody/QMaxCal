#!/bin/bash
#SBATCH --job-name=scaling_conv
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=scaling/logs/conv_%j.out
#SBATCH --error=scaling/logs/conv_%j.out

# Convergence sweep: at each N (where Lindblad is feasible), find how
# many time steps the diffusive SSE needs to match the master equation
# within target tolerance.

PYTHON=~/qmaxcal_venv/bin/python
ROOT="$(cd "$(dirname "$0")" && pwd)"
while [ ! -f "$ROOT/train_qmaxcal.py" ] && [ "$ROOT" != "/" ]; do ROOT="$(dirname "$ROOT")"; done
cd "$ROOT"

# Smoke test: quick verification on small N first
SMOKE=${SMOKE:-0}

if [ "$SMOKE" = "1" ]; then
    $PYTHON -u scaling/src/bench_convergence.py --smoke \
        --out scaling/results/convergence_smoke.json
else
    $PYTHON -u scaling/src/bench_convergence.py \
        --n-list 4 6 8 10 \
        --nts-list 1024 2048 4096 8192 \
        --n-trajs 512 \
        --n-reseeds 1 \
        --out scaling/results/convergence.json
fi

echo "done $(date)"
