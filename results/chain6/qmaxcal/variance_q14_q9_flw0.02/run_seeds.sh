#!/bin/bash
#SBATCH --job-name=ch6q149var
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:30:00
#SBATCH --array=0-6
#SBATCH --output=logs/ch6q149var_%A_%a.out
#SBATCH --error=logs/ch6q149var_%A_%a.out

# Variance sweep for chain6 IBM Kingston q14->q9 with linear fluence flw=0.02.
# Seed=0 baseline & DV=16 already exist (symlinked from
# results/chain6/qmaxcal/kingston_q14_q9_linflu/); this job adds:
#   - 2 extra baseline seeds (1, 2)
#   - 2 extra DV=16 seeds   (1, 2)
#   - 3 KL=16 seeds          (0, 1, 2)
#
# Tasks (kl, dv, seed):
#   0: (0,  0,  1)   baseline seed 1
#   1: (0,  0,  2)   baseline seed 2
#   2: (0,  16, 1)   DV       seed 1
#   3: (0,  16, 2)   DV       seed 2
#   4: (16, 0,  0)   KL       seed 0
#   5: (16, 0,  1)   KL       seed 1
#   6: (16, 0,  2)   KL       seed 2

PYTHON=~/qmaxcal_venv/bin/python
TEMPLATE=/home/mmoody/qmaxcal_paper/experiments/template
VRDIR=$TEMPLATE/results/chain6/qmaxcal/variance_q14_q9_flw0.02

# Frozen train script lives in this directory; need template on PYTHONPATH
# so 'systems.spin_chain_6_kingston_q14_q9' resolves.
export PYTHONPATH=$TEMPLATE:$PYTHONPATH

cd $TEMPLATE

KLS=( 0.0   0.0   0.0   0.0   16.0  16.0  16.0)
DVS=( 0.0   0.0   16.0  16.0  0.0   0.0   0.0 )
SDS=( 1     2     1     2     0     1     2   )

KL=${KLS[$SLURM_ARRAY_TASK_ID]}
DV=${DVS[$SLURM_ARRAY_TASK_ID]}
SD=${SDS[$SLURM_ARRAY_TASK_ID]}

echo "=== chain6 Kingston q14-q9 variance  kl=$KL  dv=$DV  seed=$SD ==="

$PYTHON -u $VRDIR/train_qmaxcal.py \
    --system systems.spin_chain_6_kingston_q14_q9 --method exp_split \
    --T_final 30.0 --signal_scale 3.0 \
    --kl_weight $KL --drift_var_weight $DV \
    --fluence_weight 0.02 --fluence_max 0.0 --fluence_warmup_steps 500 \
    --n_time_steps 1024 --n_trajs 64 \
    --num_steps 3000 --lr 1e-3 \
    --n_modes 16 --init_scale 1.0 --seed $SD \
    --snap_every 250 \
    --output_dir $VRDIR

echo "done $(date)"
