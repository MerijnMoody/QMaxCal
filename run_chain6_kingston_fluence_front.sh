#!/bin/bash
#SBATCH --job-name=ch6_kg_flfront
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-1
#SBATCH --output=logs/ch6_kg_flfront_%A_%a.out
#SBATCH --error=logs/ch6_kg_flfront_%A_%a.out

# Fluence-only Pareto front for the KL/DV-vs-fluence ablation on the
# Kingston q14-q9 chain. Extends the existing fluence-only axis
# (fl=0.0 from the flw0 run, fl=0.02 from variance_q14_q9_flw0.02) with
# two more LINEAR fluence weights, kl=dv=0. Together the 4 points trace
# what a fluence penalty alone can reach in the (fluence, fidelity) plane;
# the dv=16 runs at fl=0.02 are then compared against that front.
#
#   task 0:  fluence_weight=0.005  (gentle: extends the front's low end)
#   task 1:  fluence_weight=0.05   (aggressive: matched-fluence comparison)
#
# All other settings identical to run_chain6_kingston_q14_q9_linflu.sh /
# the variance-study runs (3000 steps, T=30, 1024 time steps, 64 trajs).

PYTHON=~/qmaxcal_venv/bin/python
cd "$(dirname "$(realpath "$0")")"

FLWS=(0.005 0.05)
FLW=${FLWS[$SLURM_ARRAY_TASK_ID]}

echo "=== chain6 Kingston q14-q9 fluence front  T=30us  flw=$FLW  kl=0  dv=0 ==="

$PYTHON -u train_qmaxcal.py \
    --system systems.spin_chain_6_kingston_q14_q9 --method exp_split \
    --T_final 30.0 --signal_scale 3.0 \
    --kl_weight 0.0 --drift_var_weight 0.0 \
    --fluence_weight $FLW --fluence_max 0.0 --fluence_warmup_steps 500 \
    --n_time_steps 1024 --n_trajs 64 \
    --num_steps 3000 --lr 1e-3 \
    --n_modes 16 --init_scale 1.0 --seed 0 \
    --snap_every 250 \
    --output_dir results/chain6/qmaxcal/kingston_q14_q9_linflu

echo "done $(date)"
