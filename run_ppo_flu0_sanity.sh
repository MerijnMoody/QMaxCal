#!/bin/bash
#SBATCH --job-name=ppo_flu0
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/ppo_flu0_%A_%a.out
#SBATCH --error=logs/ppo_flu0_%A_%a.out

# Fluence-penalty sanity check: re-train the best Phase-B PPO configuration
# on each benchmark with λ_flu = 0 to confirm the added quadratic-fluence
# term is not handicapping PPO. Reproduces the `Fluence-penalty sanity
# check` paragraph in app:ppo.
#
# (For the IBM Kingston chain-6 the matching λ_flu=0 entry is in
#  `tab:ibm_chain_full`; that run uses run_ch6q149_ppo_*.sh.)

PYTHON=~/qmaxcal_venv/bin/python
cd "$(dirname "$(realpath "$0")")"

case $SLURM_ARRAY_TASK_ID in
    0) SYS=systems.amplitude_damping; G=2.0; T=1.0
       SM=0.0;     KS=2; FF=5.0
       OUT=results/ampdamp/ppo/g2.0_flu0_seed0 ;;
    1) SYS=systems.stirap_3level;     G=5.0; T=2.0
       SM=0.001;   KS=2; FF=5.0
       OUT=results/stirap/ppo/g5.0_flu0_seed0 ;;
    2) SYS=systems.dark_route_v3;     G=2.0; T=1.0
       SM=0.001;   KS=2; FF=1.0
       OUT=results/diamond/ppo/g2.0_flu0_seed0 ;;
esac

echo "=== ppo flu0 sanity  sys=$SYS  γ=$G  T=$T  sm=$SM ==="

$PYTHON -u train_ppo.py \
    --system $SYS \
    --gamma $G --T_final $T \
    --num_envs 16 --num_updates 50000 --num_minibatches 4 \
    --lr 5e-4 --ent_coef 0.0 --max_grad_norm 0.5 \
    --layer_size 256 --anneal_lr 1 \
    --n_action_steps 50 --log_fidelity 1 --fid_factor $FF \
    --smoothness_penalty_env $SM \
    --pulse_area_penalty 0.0 --fluence_penalty 0.0 \
    --mxstep 1000 --mx_step_penalty 25.0 --fixed_endpoints 1 \
    --kernel_window 25 --kernel_std $KS \
    --seed 0 \
    --output_dir $OUT

echo "done $(date)"
