#!/bin/bash
#SBATCH --job-name=ch4_deph3
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/ch4_deph3_%A_%a.out
#SBATCH --error=logs/ch4_deph3_%A_%a.out

# chain4 endpoint-noise (dephasing-dominated) at ratio=4: three regularizers
# head-to-head to show the Wiener-KL is the wrong inductive bias here —
# KL's per-channel α² term is site-blind, so it should land ≈ baseline,
# while DV's site-aware bias exploits the low-noise interior.
#
#   task 0:  baseline (no regularization)
#   task 1:  KL=0.02
#   task 2:  DV=0.02
#
# All other settings copied from run_chain4_asymmetry.sh.

PYTHON=~/qmaxcal_venv/bin/python
cd "$(dirname "$(realpath "$0")")"

KLS=(0.0 0.02 0.0)
DVS=(0.0 0.0  0.02)

KL=${KLS[$SLURM_ARRAY_TASK_ID]}
DV=${DVS[$SLURM_ARRAY_TASK_ID]}

echo "=== chain4 dephasing ratio=4  kl=$KL  dv=$DV ==="

$PYTHON -u train_qmaxcal.py \
    --system systems.spin_chain_4 --method exp_split \
    --kappa_global 0.3 --gamma_global 0.05 \
    --kappa_edge_boost 4.0 --gamma_edge_boost 4.0 \
    --T_final 3.0 \
    --kl_weight $KL --drift_var_weight $DV \
    --fluence_weight 0.005 --fluence_warmup_steps 2000 \
    --n_time_steps 256 --n_trajs 64 \
    --num_steps 10000 --lr 1e-3 \
    --n_modes 16 --init_scale 1.0 --seed 0 \
    --snap_every 500 \
    --output_dir results/chain4/qmaxcal/endnoise_r4

echo "done $(date)"
