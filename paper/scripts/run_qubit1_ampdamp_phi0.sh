#!/bin/bash
#SBATCH --job-name=q1_ad_phi0
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/q1_ad_phi0_%j.out
#SBATCH --error=logs/q1_ad_phi0_%j.out

# phi=0 baseline for the LO-phase experiment (run_qubit1_ampdamp_phase.sh),
# same trainer and settings as the phi=pi/4, pi/2 runs so the revision table
# is same-code across all three phases.

PYTHON=~/qmaxcal_venv/bin/python
# Submit from the repository root: sbatch paper/scripts/run_qubit1_ampdamp_phi0.sh
cd "${SLURM_SUBMIT_DIR:-$PWD}"
[ -f train_qmaxcal.py ] || { echo "Submit this script from the repository root." >&2; exit 1; }

echo "=== qubit1 amplitude damping  gamma=0.1  KL=1.0  phi=0 ==="

$PYTHON -u train_qmaxcal.py \
    --system systems.single_qubit_ampdamp --method exp_split \
    --gamma 0.1 --lo_phase 0.0 --T_final 1.0 \
    --kl_weight 1.0 --drift_var_weight 0.0 \
    --fluence_weight 0.01 --fluence_warmup_steps 0 \
    --n_time_steps 256 --n_trajs 256 \
    --num_steps 5000 --lr 1e-3 \
    --n_modes 20 --init_scale 0.1 --seed 0 \
    --snap_every 250 \
    --output_dir paper/results/ampdamp/qmaxcal/lo_phase

echo "done $(date)"
