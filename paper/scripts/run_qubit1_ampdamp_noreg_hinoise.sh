#!/bin/bash
#SBATCH --job-name=q1_ad_nr_hi
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --array=0-1
#SBATCH --output=logs/q1_ad_nr_hi_%A_%a.out
#SBATCH --error=logs/q1_ad_nr_hi_%A_%a.out

# Unregularized (kl=dv=0) baselines at phi=0 for the high-noise DV phase
# experiments (run_qubit1_ampdamp_phase_dv.sh gamma=0.5, _g2.sh gamma=2.0),
# so the DV=1.0 fidelities have a same-noise fidelity-only reference.
#
#   task 0:  gamma = 0.5
#   task 1:  gamma = 2.0

PYTHON=~/qmaxcal_venv/bin/python
# Submit from the repository root: sbatch paper/scripts/run_qubit1_ampdamp_noreg_hinoise.sh
cd "${SLURM_SUBMIT_DIR:-$PWD}"
[ -f train_qmaxcal.py ] || { echo "Submit this script from the repository root." >&2; exit 1; }

GAMMAS=(0.5 2.0)
G=${GAMMAS[$SLURM_ARRAY_TASK_ID]}

echo "=== qubit1 amplitude damping  gamma=$G  UNREGULARIZED  phi=0 ==="

$PYTHON -u train_qmaxcal.py \
    --system systems.single_qubit_ampdamp --method exp_split \
    --gamma $G --lo_phase 0.0 --T_final 1.0 \
    --kl_weight 0.0 --drift_var_weight 0.0 \
    --fluence_weight 0.01 --fluence_warmup_steps 0 \
    --n_time_steps 256 --n_trajs 256 \
    --num_steps 5000 --lr 1e-3 \
    --n_modes 20 --init_scale 0.1 --seed 0 \
    --snap_every 250 \
    --output_dir paper/results/ampdamp/qmaxcal/lo_phase

echo "done $(date)"
