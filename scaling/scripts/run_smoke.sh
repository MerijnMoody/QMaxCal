#!/bin/bash
#SBATCH --job-name=scaling_smoke
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=scaling/logs/smoke_%j.out
#SBATCH --error=scaling/logs/smoke_%j.out

# Smoke test for the SSE scaling experiment.  Restricts to small N
# (4, 6) and few n_trajs to verify the benchmark plumbing works
# end-to-end before launching the full sweep.

PYTHON=~/qmaxcal_venv/bin/python
cd /home/mmoody/qmaxcal_paper

echo "=== bench_wallclock --smoke ==="
$PYTHON -u scaling/src/bench_wallclock.py --smoke \
    --out scaling/results/wallclock_smoke.json

echo
echo "=== bench_variance --smoke ==="
$PYTHON -u scaling/src/bench_variance.py --smoke \
    --out scaling/results/variance_smoke.json

echo "done $(date)"
