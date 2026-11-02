#!/bin/bash
#SBATCH --job-name=ch6_kg149L_flw0
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --array=0-1
#SBATCH --output=logs/ch6_kg149L_flw0_%A_%a.out
#SBATCH --error=logs/ch6_kg149L_flw0_%A_%a.out

# Same as run_chain6_kingston_q14_q9_linflu.sh but with fluence_weight=0.0
# (no fluence penalty at all), as the zero-penalty reference point.
#
# Tasks:
#   0:  fluence_weight=0.0,  dv = 0
#   1:  fluence_weight=0.0,  dv = 16

PYTHON=~/qmaxcal_venv/bin/python
cd "$(dirname "$(realpath "$0")")"

FLW=0.0
DVS=(0.0 16.0)
DV=${DVS[$SLURM_ARRAY_TASK_ID]}

echo "=== chain6 Kingston q14-q9 (old ctrl, LINEAR fluence)  T=30us  flw=$FLW  dv=$DV ==="

$PYTHON -u train_qmaxcal.py \
    --system systems.spin_chain_6_kingston_q14_q9 --method exp_split \
    --T_final 30.0 --signal_scale 3.0 \
    --kl_weight 0.0 --drift_var_weight $DV \
    --fluence_weight $FLW --fluence_max 0.0 --fluence_warmup_steps 500 \
    --n_time_steps 1024 --n_trajs 64 \
    --num_steps 3000 --lr 1e-3 \
    --n_modes 16 --init_scale 1.0 --seed 0 \
    --snap_every 250 \
    --output_dir results/chain6/qmaxcal/kingston_q14_q9_linflu

echo "done $(date)"
