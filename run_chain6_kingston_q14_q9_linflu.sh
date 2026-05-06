#!/bin/bash
#SBATCH --job-name=ch6_kg149L
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:30:00
#SBATCH --array=0-1
#SBATCH --output=logs/ch6_kg149L_%A_%a.out
#SBATCH --error=logs/ch6_kg149L_%A_%a.out

# Same q14-q9 chain as run_chain6_kingston_q14_q9.sh, but with a LINEAR
# fluence penalty (no hinge cap) at a single weight, intended to land
# fluence in the 10-20 range for comparison with the hinge-flmax=15 run.
#
# Tasks:
#   0:  fluence_weight=0.02,  dv = 0
#   1:  fluence_weight=0.02,  dv = 16

PYTHON=~/qmaxcal_venv/bin/python
cd /home/mmoody/qmaxcal_paper

FLW=0.01
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
