#!/bin/bash
#SBATCH --job-name=ch6q149kl0p1s
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:30:00
#SBATCH --array=0-1
#SBATCH --output=logs/ch6q149kl0p1s_%A_%a.out
#SBATCH --error=logs/ch6q149kl0p1s_%A_%a.out

# Two extra KL=0.1 seeds (1, 2) to fill out the variance estimate.
# Seed 0 already exists from job 22439590.

PYTHON=~/qmaxcal_venv/bin/python
# Submit from the repository root: sbatch paper/results/chain6/qmaxcal/variance_q14_q9_flw0.02/run_kl0p1_seeds.sh
cd "${SLURM_SUBMIT_DIR:-$PWD}"
[ -f train_qmaxcal.py ] || { echo "Submit this script from the repository root." >&2; exit 1; }
VRDIR=paper/results/chain6/qmaxcal/variance_q14_q9_flw0.02

export PYTHONPATH=$PWD:$PYTHONPATH

SDS=(1 2)
SD=${SDS[$SLURM_ARRAY_TASK_ID]}

echo "=== chain6 Kingston q14-q9 variance  kl=0.1  dv=0  seed=$SD ==="

$PYTHON -u $VRDIR/train_qmaxcal.py \
    --system systems.spin_chain_6_kingston_q14_q9 --method exp_split \
    --T_final 30.0 --signal_scale 3.0 \
    --kl_weight 0.1 --drift_var_weight 0.0 \
    --fluence_weight 0.02 --fluence_max 0.0 --fluence_warmup_steps 500 \
    --n_time_steps 1024 --n_trajs 64 \
    --num_steps 3000 --lr 1e-3 \
    --n_modes 16 --init_scale 1.0 --seed $SD \
    --snap_every 250 \
    --output_dir $VRDIR

echo "done $(date)"
