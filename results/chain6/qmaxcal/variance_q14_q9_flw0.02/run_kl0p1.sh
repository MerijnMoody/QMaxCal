#!/bin/bash
#SBATCH --job-name=ch6q149kl0p1
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:30:00
#SBATCH --output=logs/ch6q149kl0p1_%j.out
#SBATCH --error=logs/ch6q149kl0p1_%j.out

# KL=0.1 run (single task, seed=0).  Picked to match the loss
# contribution of dv=16 on this small-rate system:
# KL_cost ~ 1.1 vs DV_cost ~ 0.006, so kl_weight × 1.1 ≈ 16 × 0.006
# ⇒ kl_weight ≈ 0.1.  Earlier kl=1 and kl=16 attempts both stuck the
# optimizer in a high-fluence, low-fidelity basin (F < 0.5).

PYTHON=~/qmaxcal_venv/bin/python
TEMPLATE=/home/mmoody/qmaxcal_paper/experiments/template
VRDIR=$TEMPLATE/results/chain6/qmaxcal/variance_q14_q9_flw0.02

export PYTHONPATH=$TEMPLATE:$PYTHONPATH
cd $TEMPLATE

echo "=== chain6 Kingston q14-q9 variance  kl=0.1  dv=0  seed=0 ==="

$PYTHON -u $VRDIR/train_qmaxcal.py \
    --system systems.spin_chain_6_kingston_q14_q9 --method exp_split \
    --T_final 30.0 --signal_scale 3.0 \
    --kl_weight 0.1 --drift_var_weight 0.0 \
    --fluence_weight 0.02 --fluence_max 0.0 --fluence_warmup_steps 500 \
    --n_time_steps 1024 --n_trajs 64 \
    --num_steps 3000 --lr 1e-3 \
    --n_modes 16 --init_scale 1.0 --seed 0 \
    --snap_every 250 \
    --output_dir $VRDIR

echo "done $(date)"
