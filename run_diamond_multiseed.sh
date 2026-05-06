#!/bin/bash
#SBATCH --job-name=diamond_ms
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:30:00
#SBATCH --array=0-8
#SBATCH --output=logs/diamond_ms_%A_%a.out
#SBATCH --error=logs/diamond_ms_%A_%a.out

# Diamond / dark-route multi-seed sweep.
# {baseline, KL_W=0.5, KL_W=5} × 3 seeds = 9 tasks at γ=2, T=1.
#
# Reproduces `tab:dr3_multiseed` and the QMaxCal lines of
# `tab:dr3_robustness`.

PYTHON=~/qmaxcal_venv/bin/python
cd "$(dirname "$(realpath "$0")")"

KLS=(0.0 0.5 5.0)
i=$SLURM_ARRAY_TASK_ID
KL_IDX=$((i / 3))
SEED=$((i % 3))
KL=${KLS[$KL_IDX]}

echo "=== diamond ms  kl=$KL  seed=$SEED ==="

$PYTHON -u train_qmaxcal.py \
    --system systems.dark_route_v3 \
    --method euler \
    --gamma 2.0 --T_final 1.0 \
    --kl_weight $KL --drift_var_weight 0.0 \
    --fluence_weight 0.001 \
    --n_time_steps 256 --n_trajs 256 \
    --num_steps 5000 --lr 1e-3 \
    --n_modes 20 --init_scale 0.1 --seed $SEED \
    --snap_every 500 \
    --output_dir results/diamond/qmaxcal/multiseed

echo "done $(date)"
