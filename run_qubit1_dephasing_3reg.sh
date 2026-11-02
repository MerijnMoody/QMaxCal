#!/bin/bash
#SBATCH --job-name=q1_deph3
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/q1_deph3_%A_%a.out
#SBATCH --error=logs/q1_deph3_%A_%a.out

# Single qubit, pure dephasing (L = √κ σ_z), transfer |↑⟩ → |↓⟩ with a σ_x
# control. Minimal testbed where the Wiener-KL is provably ANTI-aligned with
# decoherence: KL_W = 2κT − 2×(dephasing-exposure integral), so minimizing it
# rewards equator residence — exactly where dephasing does its damage — while
# DV penalizes it. Prediction: KL_W < baseline ≤ DV in fidelity, with the
# KL_W run's own KL dropping below baseline's (right objective, wrong physics).
#
#   task 0:  baseline (no regularization)
#   task 1:  KL=0.02
#   task 2:  DV=0.02
#
# Trainer settings match run_chain4_dephasing_3reg.sh; fluence penalty kept
# so pulse speed has a real cost and the regularizers must take sides.

PYTHON=~/qmaxcal_venv/bin/python
cd "$(dirname "$(realpath "$0")")"

KLS=(0.0 0.02 0.0)
DVS=(0.0 0.0  0.02)

KL=${KLS[$SLURM_ARRAY_TASK_ID]}
DV=${DVS[$SLURM_ARRAY_TASK_ID]}

echo "=== qubit1 pure dephasing  kappa=0.3  kl=$KL  dv=$DV ==="

$PYTHON -u train_qmaxcal.py \
    --system systems.single_qubit_dephasing --method exp_split \
    --kappa 0.3 --T_final 3.0 \
    --kl_weight $KL --drift_var_weight $DV \
    --fluence_weight 0.005 --fluence_warmup_steps 2000 \
    --n_time_steps 256 --n_trajs 64 \
    --num_steps 10000 --lr 1e-3 \
    --n_modes 16 --init_scale 1.0 --seed 0 \
    --snap_every 500 \
    --output_dir results/dephasing/qmaxcal/pure_dephasing

echo "done $(date)"
