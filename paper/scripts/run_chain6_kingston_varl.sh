#!/bin/bash
#SBATCH --job-name=ch6_kg_varl
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-1
#SBATCH --output=logs/ch6_kg_varl_%A_%a.out
#SBATCH --error=logs/ch6_kg_varl_%A_%a.out

# Direct decoherence penalty on the Kingston q14-q9 chain: regularize with
# the LO-phase-INVARIANT noise magnitude VarL = Sum_k E[int(<L†L>-|<L>|^2)dt]
# (the quantity in the LO-phase appendix; zero iff in the joint kernel),
# instead of the record-drift-based KL_W / implemented R_DV. kl=dv=0.
# Comparison set: variance_q14_q9_flw0.02 (baseline 0.836+-0.003,
# dv16 0.858+-0.005, 3 seeds each; same fl=0.02, 3000 steps, seed 0 here).
#
#   task 0:  var_l_weight = 1.0
#   task 1:  var_l_weight = 4.0

PYTHON=~/qmaxcal_venv/bin/python
# Submit from the repository root: sbatch paper/scripts/run_chain6_kingston_varl.sh
cd "${SLURM_SUBMIT_DIR:-$PWD}"
[ -f train_qmaxcal.py ] || { echo "Submit this script from the repository root." >&2; exit 1; }

VLS=(1.0 4.0)
VL=${VLS[$SLURM_ARRAY_TASK_ID]}

echo "=== chain6 Kingston q14-q9  T=30us  VarL penalty  vl=$VL  fl=0.02 ==="

$PYTHON -u train_qmaxcal.py \
    --system systems.spin_chain_6_kingston_q14_q9 --method exp_split \
    --T_final 30.0 --signal_scale 3.0 \
    --kl_weight 0.0 --drift_var_weight 0.0 \
    --var_l_weight $VL \
    --fluence_weight 0.02 --fluence_max 0.0 --fluence_warmup_steps 500 \
    --n_time_steps 1024 --n_trajs 64 \
    --num_steps 3000 --lr 1e-3 \
    --n_modes 16 --init_scale 1.0 --seed 0 \
    --snap_every 250 \
    --output_dir paper/results/chain6/qmaxcal/variance_q14_q9_flw0.02

echo "done $(date)"
