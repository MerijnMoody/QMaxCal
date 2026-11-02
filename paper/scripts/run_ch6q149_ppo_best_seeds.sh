#!/bin/bash
#SBATCH --job-name=ch6q149ppoS
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:30:00
#SBATCH --array=0-1
#SBATCH --output=logs/ch6q149ppoS_%A_%a.out
#SBATCH --error=logs/ch6q149ppoS_%A_%a.out

# Two extra PPO seeds at the best-cell config from the matched-fluence
# Phase-B sweep (job 22458197 task 3):
#   smoothness=0.001, kernel_std=4, fluence_penalty=0.02.
# Seed 0 already exists; this fills out seeds 1 and 2 for variance.

PYTHON=~/qmaxcal_venv/bin/python
# Submit from the repository root: sbatch paper/scripts/run_ch6q149_ppo_best_seeds.sh
cd "${SLURM_SUBMIT_DIR:-$PWD}"
[ -f train_qmaxcal.py ] || { echo "Submit this script from the repository root." >&2; exit 1; }

SDS=(1 2)
SD=${SDS[$SLURM_ARRAY_TASK_ID]}

echo "=== chain6 q14-q9 PPO best-cell  seed=$SD  smoothness=0.001  kernel_std=4  flu_pen=0.02 ==="

$PYTHON -u train_ppo.py \
    --system systems.spin_chain_6_kingston_q14_q9 \
    --T_final 30.0 --signal_scale 3.0 \
    --num_envs      16 \
    --num_updates   50000 \
    --num_minibatches 4 \
    --lr 5e-4 --ent_coef 0.0 --max_grad_norm 0.5 \
    --layer_size 256 --anneal_lr 0 \
    --n_action_steps 50 --log_fidelity 1 --fid_factor 1.0 \
    --smoothness_penalty_env 0.001 --pulse_area_penalty 0.0 \
    --fluence_penalty 0.02 \
    --mxstep 1000 --mx_step_penalty 25.0 --fixed_endpoints 1 \
    --kernel_window 25 --kernel_std 4.0 \
    --snap_every 5000 --max_wall_time_seconds 10800 \
    --seed $SD --output_dir paper/results/chain6/ppo/q14_q9_sm0.001_ks4.0

echo "done $(date)"
