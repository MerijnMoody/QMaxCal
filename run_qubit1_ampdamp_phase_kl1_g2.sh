#!/bin/bash
#SBATCH --job-name=q1_ad_kl1_g2
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/q1_ad_kl1_g2_%A_%a.out
#SBATCH --error=logs/q1_ad_kl1_g2_%A_%a.out

# Dose-response for the phase-dependent KL_W effect at gamma*T=2:
# same as run_qubit1_ampdamp_phase_kl5_g2.sh but KL=1.0. At KL=5 the
# phases split +0.8pp / -1.7pp / -2.2pp vs baseline (F=0.9727); this run
# checks whether the gauge-toll distortion is already present at lambda=1.
#
#   task 0:  phi = 0
#   task 1:  phi = pi/4
#   task 2:  phi = pi/2

PYTHON=~/qmaxcal_venv/bin/python
cd "$(dirname "$(realpath "$0")")"

PHIS=(0.0 0.78539816 1.57079633)
PHI=${PHIS[$SLURM_ARRAY_TASK_ID]}

echo "=== qubit1 amplitude damping  gamma=2.0  KL=1.0  phi=$PHI ==="

$PYTHON -u train_qmaxcal.py \
    --system systems.single_qubit_ampdamp --method exp_split \
    --gamma 2.0 --lo_phase $PHI --T_final 1.0 \
    --kl_weight 1.0 --drift_var_weight 0.0 \
    --fluence_weight 0.01 --fluence_warmup_steps 0 \
    --n_time_steps 256 --n_trajs 256 \
    --num_steps 5000 --lr 1e-3 \
    --n_modes 20 --init_scale 0.1 --seed 0 \
    --snap_every 250 \
    --output_dir results/ampdamp/qmaxcal/lo_phase

echo "done $(date)"
