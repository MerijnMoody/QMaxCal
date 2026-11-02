#!/bin/bash
#SBATCH --job-name=scaling_conv
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/conv_%j.out
#SBATCH --error=logs/conv_%j.out

# Convergence sweep: at each N (where Lindblad is feasible), find how
# many time steps the diffusive SSE needs to match the master equation
# within target tolerance.

PYTHON=~/qmaxcal_venv/bin/python
# Submit from the repository root: sbatch paper/scaling/scripts/run_convergence.sh
cd "${SLURM_SUBMIT_DIR:-$PWD}"
[ -f train_qmaxcal.py ] || { echo "Submit this script from the repository root." >&2; exit 1; }

# Smoke test: quick verification on small N first
SMOKE=${SMOKE:-0}

if [ "$SMOKE" = "1" ]; then
    $PYTHON -u paper/scaling/src/bench_convergence.py --smoke \
        --out paper/scaling/results/convergence_smoke.json
else
    $PYTHON -u paper/scaling/src/bench_convergence.py \
        --n-list 4 6 8 10 \
        --nts-list 1024 2048 4096 8192 \
        --n-trajs 512 \
        --n-reseeds 1 \
        --out paper/scaling/results/convergence.json
fi

echo "done $(date)"
