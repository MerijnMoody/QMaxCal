#!/bin/bash
#SBATCH --job-name=q1_ad_kl_hi
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --array=0-5
#SBATCH --output=logs/q1_ad_kl_hi_%A_%a.out
#SBATCH --error=logs/q1_ad_kl_hi_%A_%a.out

# KL_W=1.0 at high noise, completing the noise x regularizer x phase grid:
# gamma in {0.5, 2.0} x phi in {0, pi/4, pi/2}, kl=1.0, dv=0. Counterparts
# to the R_DV=1.0 runs (run_qubit1_ampdamp_phase_dv.sh / _g2.sh) and the
# unregularized baselines (run_qubit1_ampdamp_noreg_hinoise.sh).
#
#   task 0:  gamma=0.5  phi=0
#   task 1:  gamma=0.5  phi=pi/4
#   task 2:  gamma=0.5  phi=pi/2
#   task 3:  gamma=2.0  phi=0
#   task 4:  gamma=2.0  phi=pi/4
#   task 5:  gamma=2.0  phi=pi/2

PYTHON=~/qmaxcal_venv/bin/python
# Submit from the repository root: sbatch paper/scripts/run_qubit1_ampdamp_phase_kl_hinoise.sh
cd "${SLURM_SUBMIT_DIR:-$PWD}"
[ -f train_qmaxcal.py ] || { echo "Submit this script from the repository root." >&2; exit 1; }

GAMMAS=(0.5 0.5 0.5 2.0 2.0 2.0)
PHIS=(0.0 0.78539816 1.57079633 0.0 0.78539816 1.57079633)
G=${GAMMAS[$SLURM_ARRAY_TASK_ID]}
PHI=${PHIS[$SLURM_ARRAY_TASK_ID]}

echo "=== qubit1 amplitude damping  gamma=$G  KL=1.0  phi=$PHI ==="

$PYTHON -u train_qmaxcal.py \
    --system systems.single_qubit_ampdamp --method exp_split \
    --gamma $G --lo_phase $PHI --T_final 1.0 \
    --kl_weight 1.0 --drift_var_weight 0.0 \
    --fluence_weight 0.01 --fluence_warmup_steps 0 \
    --n_time_steps 256 --n_trajs 256 \
    --num_steps 5000 --lr 1e-3 \
    --n_modes 20 --init_scale 0.1 --seed 0 \
    --snap_every 250 \
    --output_dir paper/results/ampdamp/qmaxcal/lo_phase

echo "done $(date)"
