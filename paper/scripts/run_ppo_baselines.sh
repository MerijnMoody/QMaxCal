#!/bin/bash
#SBATCH --job-name=ppo_base
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/ppo_base_%A_%a.out
#SBATCH --error=logs/ppo_base_%A_%a.out

# PPO baselines for amp damping, STIRAP, and diamond.
# Best Phase-B configurations from Appendix `app:ppo` (per-system overrides).
# All add the squared-fluence penalty λ_flu = 0.001 to the PPO reward
# (the original linear pulse-area penalty w_A is set to 0).
#
#   task 0: AmpD γ=2          smoothness=0,     kernel_std=2, w_F=5
#   task 1: STIRAP γ=5, T=2   smoothness=0.001, kernel_std=2, w_F=5
#   task 2: Diamond γ=2       smoothness=0.001, kernel_std=2, w_F=1
#
# Single seed=0; multi-seed runs are launched separately for the headline
# tables (3 seeds for STIRAP/diamond, 4 seeds for AmpD).

PYTHON=~/qmaxcal_venv/bin/python
# Submit from the repository root: sbatch paper/scripts/run_ppo_baselines.sh
cd "${SLURM_SUBMIT_DIR:-$PWD}"
[ -f train_qmaxcal.py ] || { echo "Submit this script from the repository root." >&2; exit 1; }

case $SLURM_ARRAY_TASK_ID in
    0) SYS=systems.amplitude_damping; G=2.0; T=1.0
       SM=0.0;     KS=2; FF=5.0
       OUT=paper/results/ampdamp/ppo/g2.0_seed0 ;;
    1) SYS=systems.stirap_3level;     G=5.0; T=2.0
       SM=0.001;   KS=2; FF=5.0
       OUT=paper/results/stirap/ppo/g5.0_seed0 ;;
    2) SYS=systems.dark_route_v3;     G=2.0; T=1.0
       SM=0.001;   KS=2; FF=1.0
       OUT=paper/results/diamond/ppo/g2.0_seed0 ;;
esac

echo "=== ppo baseline  sys=$SYS  γ=$G  T=$T  sm=$SM  ks=$KS  ff=$FF ==="

$PYTHON -u train_ppo.py \
    --system $SYS \
    --gamma $G --T_final $T \
    --num_envs 16 --num_updates 50000 --num_minibatches 4 \
    --lr 5e-4 --ent_coef 0.0 --max_grad_norm 0.5 \
    --layer_size 256 --anneal_lr 1 \
    --n_action_steps 50 --log_fidelity 1 --fid_factor $FF \
    --smoothness_penalty_env $SM \
    --pulse_area_penalty 0.0 --fluence_penalty 0.001 \
    --mxstep 1000 --mx_step_penalty 25.0 --fixed_endpoints 1 \
    --kernel_window 25 --kernel_std $KS \
    --seed 0 \
    --output_dir $OUT

echo "done $(date)"
