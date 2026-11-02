#!/bin/bash
#SBATCH --job-name=q1_ad_phase
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --array=0-1
#SBATCH --output=logs/q1_ad_phase_%A_%a.out
#SBATCH --error=logs/q1_ad_phase_%A_%a.out

# LO-phase dependence of KL_W on the
# single-qubit amplitude-damping task (|+> -> |Y>, gamma=0.1, T=1, u_x/u_y
# controls). L -> L e^{i phi} leaves the Lindblad dynamics invariant but
# rotates the monitored quadrature alpha_phi = 2 Re<e^{i phi} L>, so KL_W's
# zero set rotates with phi. Prediction: the discovered
# policy is unchanged, because the DFS mechanism (L|psi>=0) inside KL_W is
# phase-invariant.
#
# Hyperparameters match the original wiener_regularization amplitude_damping
# config (T=1, 256 time steps, 256 trajs, 20 modes, init 0.1, 5000 steps,
# lr 1e-3, fluence 0.01, seed 0) at the featured weight KL=1.0; phi=0 is the
# original run itself.
#
#   task 0:  phi = pi/4  (0.7854)
#   task 1:  phi = pi/2  (1.5708)

PYTHON=~/qmaxcal_venv/bin/python
cd "$(dirname "$(realpath "$0")")"

PHIS=(0.78539816 1.57079633)
PHI=${PHIS[$SLURM_ARRAY_TASK_ID]}

echo "=== qubit1 amplitude damping  gamma=0.1  KL=1.0  phi=$PHI ==="

$PYTHON -u train_qmaxcal.py \
    --system systems.single_qubit_ampdamp --method exp_split \
    --gamma 0.1 --lo_phase $PHI --T_final 1.0 \
    --kl_weight 1.0 --drift_var_weight 0.0 \
    --fluence_weight 0.01 --fluence_warmup_steps 0 \
    --n_time_steps 256 --n_trajs 256 \
    --num_steps 5000 --lr 1e-3 \
    --n_modes 20 --init_scale 0.1 --seed 0 \
    --snap_every 250 \
    --output_dir results/ampdamp/qmaxcal/lo_phase

echo "done $(date)"
