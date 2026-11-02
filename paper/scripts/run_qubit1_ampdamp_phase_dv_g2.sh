#!/bin/bash
#SBATCH --job-name=q1_ad_dv_g2
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/q1_ad_dv_g2_%A_%a.out
#SBATCH --error=logs/q1_ad_dv_g2_%A_%a.out

# Same as run_qubit1_ampdamp_phase_dv.sh but gamma=2.0 (very strong damping,
# gamma*T=2): R_DV=1.0, kl=0, phi in {0, pi/4, pi/2}.
#
#   task 0:  phi = 0
#   task 1:  phi = pi/4
#   task 2:  phi = pi/2

PYTHON=~/qmaxcal_venv/bin/python
# Submit from the repository root: sbatch paper/scripts/run_qubit1_ampdamp_phase_dv_g2.sh
cd "${SLURM_SUBMIT_DIR:-$PWD}"
[ -f train_qmaxcal.py ] || { echo "Submit this script from the repository root." >&2; exit 1; }

PHIS=(0.0 0.78539816 1.57079633)
PHI=${PHIS[$SLURM_ARRAY_TASK_ID]}

echo "=== qubit1 amplitude damping  gamma=2.0  DV=1.0  phi=$PHI ==="

$PYTHON -u train_qmaxcal.py \
    --system systems.single_qubit_ampdamp --method exp_split \
    --gamma 2.0 --lo_phase $PHI --T_final 1.0 \
    --kl_weight 0.0 --drift_var_weight 1.0 \
    --fluence_weight 0.01 --fluence_warmup_steps 0 \
    --n_time_steps 256 --n_trajs 256 \
    --num_steps 5000 --lr 1e-3 \
    --n_modes 20 --init_scale 0.1 --seed 0 \
    --snap_every 250 \
    --output_dir paper/results/ampdamp/qmaxcal/lo_phase

echo "done $(date)"
