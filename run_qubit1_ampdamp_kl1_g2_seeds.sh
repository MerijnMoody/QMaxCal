#!/bin/bash
#SBATCH --job-name=q1_ad_kl1_sd
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --array=0-5
#SBATCH --output=logs/q1_ad_kl1_sd_%A_%a.out
#SBATCH --error=logs/q1_ad_kl1_sd_%A_%a.out

# Seeds 1,2 for the gamma*T=2, KL=1.0 phase row (seed 0 already done:
# 0.9785 / 0.9757 / 0.9775 at phi=0, pi/4, pi/2; baseline 0.9727).
# Tests the "moderate-lambda gain is phase-invariant and beats baseline"
# claim beyond single-seed noise.
#
#   task 0: phi=0     seed=1     task 3: phi=0     seed=2
#   task 1: phi=pi/4  seed=1     task 4: phi=pi/4  seed=2
#   task 2: phi=pi/2  seed=1     task 5: phi=pi/2  seed=2

PYTHON=~/qmaxcal_venv/bin/python
cd "$(dirname "$(realpath "$0")")"

PHIS=(0.0 0.78539816 1.57079633 0.0 0.78539816 1.57079633)
SEEDS=(1 1 1 2 2 2)
PHI=${PHIS[$SLURM_ARRAY_TASK_ID]}
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

echo "=== qubit1 amplitude damping  gamma=2.0  KL=1.0  phi=$PHI  seed=$SEED ==="

$PYTHON -u train_qmaxcal.py \
    --system systems.single_qubit_ampdamp --method exp_split \
    --gamma 2.0 --lo_phase $PHI --T_final 1.0 \
    --kl_weight 1.0 --drift_var_weight 0.0 \
    --fluence_weight 0.01 --fluence_warmup_steps 0 \
    --n_time_steps 256 --n_trajs 256 \
    --num_steps 5000 --lr 1e-3 \
    --n_modes 20 --init_scale 0.1 --seed $SEED \
    --snap_every 250 \
    --output_dir results/ampdamp/qmaxcal/lo_phase

echo "done $(date)"
