#!/bin/bash
#SBATCH --job-name=stirap_ms
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --array=0-11
#SBATCH --output=logs/stirap_ms_%A_%a.out
#SBATCH --error=logs/stirap_ms_%A_%a.out

# STIRAP 3-level Λ system multi-seed sweep.
# 2 γ × {baseline, KL_W=1} × 3 seeds = 12 tasks.
#
# Reproduces `tab:stirap_multiseed` (γT ∈ {1, 10}, baseline & Wiener KL).

PYTHON=~/qmaxcal_venv/bin/python
# Submit from the repository root: sbatch paper/scripts/run_stirap_multiseed.sh
cd "${SLURM_SUBMIT_DIR:-$PWD}"
[ -f train_qmaxcal.py ] || { echo "Submit this script from the repository root." >&2; exit 1; }

GAMMAS=(0.5 5.0)
COMBO_KLS=(0.0 1.0 0.0 1.0 0.0 1.0)
COMBO_SEEDS=(0 0 1 1 2 2)

i=$SLURM_ARRAY_TASK_ID
G_IDX=$((i / 6))
COMBO_IDX=$((i % 6))
GAMMA=${GAMMAS[$G_IDX]}
KL=${COMBO_KLS[$COMBO_IDX]}
SEED=${COMBO_SEEDS[$COMBO_IDX]}

echo "=== stirap ms  γ=$GAMMA  kl=$KL  seed=$SEED ==="

$PYTHON -u train_qmaxcal.py \
    --system systems.stirap_3level \
    --method exp_split \
    --gamma $GAMMA --T_final 2.0 \
    --kl_weight $KL --drift_var_weight 0.0 \
    --fluence_weight 0.0 \
    --n_time_steps 256 --n_trajs 64 \
    --num_steps 10000 --lr 1e-3 \
    --n_modes 16 --init_scale 1.0 --seed $SEED \
    --snap_every 500 \
    --output_dir paper/results/stirap/qmaxcal/n64

echo "done $(date)"
