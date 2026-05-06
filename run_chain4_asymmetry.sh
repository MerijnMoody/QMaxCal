#!/bin/bash
#SBATCH --job-name=ch4_asym
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --array=0-7
#SBATCH --output=logs/ch4_asym_%A_%a.out
#SBATCH --error=logs/ch4_asym_%A_%a.out

# chain4 endpoint-noise asymmetry sweep.
# Goal: see how the DV regulariser's "park on low-noise interior" strategy
# kicks in as the κ_edge / κ_int ratio grows from 1 (uniform) to 8 (extreme).
# ratio=4 is already on disk under results/chain4/qmaxcal/endnoise/.
#
# KL=0.02 is omitted from this sweep: Fig 3B shows KL ≈ baseline on chain4
# endnoise (KL's per-channel α² term is site-blind), so the asymmetry story
# is specifically about DV's site-aware inductive bias vs. the unregularised
# baseline.
#
# Layout (8 tasks): ratio ∈ {1.0, 1.5, 2.0, 8.0}  ×  policy ∈ {baseline, DV}
#
#   task 0:  ratio=1.0   baseline
#   task 1:  ratio=1.0   DV=0.02
#   task 2:  ratio=1.5   baseline
#   task 3:  ratio=1.5   DV=0.02
#   task 4:  ratio=2.0   baseline
#   task 5:  ratio=2.0   DV=0.02
#   task 6:  ratio=8.0   baseline
#   task 7:  ratio=8.0   DV=0.02

PYTHON=~/qmaxcal_venv/bin/python
cd "$(dirname "$(realpath "$0")")"

RATIOS=(1.0 1.0 1.5 1.5 2.0 2.0 8.0 8.0)
KLS=(   0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0)
DVS=(   0.0 0.02 0.0 0.02 0.0 0.02 0.0 0.02)

R=${RATIOS[$SLURM_ARRAY_TASK_ID]}
KL=${KLS[$SLURM_ARRAY_TASK_ID]}
DV=${DVS[$SLURM_ARRAY_TASK_ID]}

echo "=== chain4 asym  ratio=$R  kl=$KL  dv=$DV ==="

$PYTHON -u train_qmaxcal.py \
    --system systems.spin_chain_4 --method exp_split \
    --kappa_global 0.3 --gamma_global 0.05 \
    --kappa_edge_boost $R --gamma_edge_boost $R \
    --T_final 3.0 \
    --kl_weight $KL --drift_var_weight $DV \
    --fluence_weight 0.005 --fluence_warmup_steps 2000 \
    --n_time_steps 256 --n_trajs 64 \
    --num_steps 10000 --lr 1e-3 \
    --n_modes 16 --init_scale 1.0 --seed 0 \
    --snap_every 500 \
    --output_dir results/chain4/qmaxcal/asym_r${R}

echo "done $(date)"
