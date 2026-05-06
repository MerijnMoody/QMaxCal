#!/bin/bash
#SBATCH --job-name=ad_ms
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:30:00
#SBATCH --array=0-19
#SBATCH --output=logs/ad_ms_%A_%a.out
#SBATCH --error=logs/ad_ms_%A_%a.out

# Single-qubit amplitude damping multi-seed sweep.
# 5 γT × {baseline, KL_W=5} × 2 seeds (1,2) = 20 tasks; seed 0 is in
# `results/ampdamp/qmaxcal/multiseed_g{γ}/`-prefixed dirs already.
#
# Layout: gT_idx ∈ [0..4],  (KL,seed) combo_idx ∈ [0..3]
#   GAMMAS    = (0.1, 0.5, 1.0, 2.0, 5.0)
#   COMBO_KLS = (0.0, 0.0, 5.0, 5.0)
#   COMBO_SEEDS = (1, 2, 1, 2)
#
# Reproduces the cells of Tables `tab:ampdamp_fidelity_full` and
# `tab:ampdamp_variance_full`.

PYTHON=~/qmaxcal_venv/bin/python
cd "$(dirname "$(realpath "$0")")"

GAMMAS=(0.1 0.5 1.0 2.0 5.0)
COMBO_KLS=(0.0 0.0 5.0 5.0)
COMBO_SEEDS=(1 2 1 2)

i=$SLURM_ARRAY_TASK_ID
GT_IDX=$((i / 4))
COMBO_IDX=$((i % 4))
GAMMA=${GAMMAS[$GT_IDX]}
KL=${COMBO_KLS[$COMBO_IDX]}
SEED=${COMBO_SEEDS[$COMBO_IDX]}

echo "=== ampdamp ms  γ=$GAMMA  kl=$KL  seed=$SEED ==="

$PYTHON -u train_qmaxcal.py \
    --system systems.amplitude_damping \
    --method exp_split \
    --gamma $GAMMA --T_final 1.0 \
    --kl_weight $KL --drift_var_weight 0.0 \
    --fluence_weight 0.01 \
    --n_time_steps 256 --n_trajs 256 \
    --num_steps 5000 --lr 1e-3 \
    --n_modes 20 --init_scale 0.1 --seed $SEED \
    --snap_every 500 \
    --output_dir results/ampdamp/qmaxcal/multiseed_g${GAMMA}

echo "done $(date)"
