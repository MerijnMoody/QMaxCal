#!/bin/bash
#SBATCH --job-name=ch6q149ppoB
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:30:00
#SBATCH --array=0-3
#SBATCH --output=logs/ch6q149ppoB_%A_%a.out
#SBATCH --error=logs/ch6q149ppoB_%A_%a.out

# 4-cell PPO Phase-B sweep on chain6 IBM Kingston q14->q9.
# smoothness × kernel_std, fid_weight fixed at 1.0,
# fluence_penalty=0.02 (matched to QMaxCal q14-q9).
# Chunked training with snapshots every 5000 updates and a 3 h
# wall-time cap; if PPO doesn't reach num_updates=50000 inside the
# cap, we still keep the latest snapshot.
#
# Tasks (smoothness, kernel_std):
#   0:  (0,     2)
#   1:  (0,     4)
#   2:  (0.001, 2)
#   3:  (0.001, 4)

PYTHON=~/qmaxcal_venv/bin/python
# Submit from the repository root: sbatch paper/scripts/run_ch6q149_ppo_sweep.sh
cd "${SLURM_SUBMIT_DIR:-$PWD}"
[ -f train_qmaxcal.py ] || { echo "Submit this script from the repository root." >&2; exit 1; }

SMS=(0.0   0.0   0.001 0.001)
KSS=(2.0   4.0   2.0   4.0  )
SM=${SMS[$SLURM_ARRAY_TASK_ID]}
KS=${KSS[$SLURM_ARRAY_TASK_ID]}

echo "=== chain6 q14-q9 PPO sweep  smoothness=$SM  kernel_std=$KS  flu_pen=0.02 ==="

$PYTHON -u train_ppo.py \
    --system systems.spin_chain_6_kingston_q14_q9 \
    --T_final 30.0 --signal_scale 3.0 \
    --num_envs      16 \
    --num_updates   50000 \
    --num_minibatches 4 \
    --lr 5e-4 --ent_coef 0.0 --max_grad_norm 0.5 \
    --layer_size 256 --anneal_lr 0 \
    --n_action_steps 50 --log_fidelity 1 --fid_factor 1.0 \
    --smoothness_penalty_env $SM --pulse_area_penalty 0.0 \
    --fluence_penalty 0.02 \
    --mxstep 1000 --mx_step_penalty 25.0 --fixed_endpoints 1 \
    --kernel_window 25 --kernel_std $KS \
    --snap_every 5000 --max_wall_time_seconds 10800 \
    --seed 0 --output_dir paper/results/chain6/ppo/q14_q9_sm${SM}_ks${KS}

echo "done $(date)"
