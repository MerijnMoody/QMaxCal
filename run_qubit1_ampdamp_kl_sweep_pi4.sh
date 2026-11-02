#!/bin/bash
#SBATCH --job-name=q1_ad_klsw
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --array=0-1
#SBATCH --output=logs/q1_ad_klsw_%A_%a.out
#SBATCH --error=logs/q1_ad_klsw_%A_%a.out

# KL_W dose-response at phi=pi/4 (the neutral quadrature), gamma*T=2:
# fills in KL in {0.5, 2.0} between the existing 1.0 (+0.3pp vs baseline)
# and 5.0 (-1.7pp) points, to hone in on where the gauge toll starts to
# outweigh the kernel-routing benefit.
#
#   task 0:  KL = 0.5
#   task 1:  KL = 2.0

PYTHON=~/qmaxcal_venv/bin/python
cd "$(dirname "$(realpath "$0")")"

KLS=(0.5 2.0)
KL=${KLS[$SLURM_ARRAY_TASK_ID]}

echo "=== qubit1 amplitude damping  gamma=2.0  KL=$KL  phi=pi/4 ==="

$PYTHON -u train_qmaxcal.py \
    --system systems.single_qubit_ampdamp --method exp_split \
    --gamma 2.0 --lo_phase 0.78539816 --T_final 1.0 \
    --kl_weight $KL --drift_var_weight 0.0 \
    --fluence_weight 0.01 --fluence_warmup_steps 0 \
    --n_time_steps 256 --n_trajs 256 \
    --num_steps 5000 --lr 1e-3 \
    --n_modes 20 --init_scale 0.1 --seed 0 \
    --snap_every 250 \
    --output_dir results/ampdamp/qmaxcal/lo_phase

echo "done $(date)"
