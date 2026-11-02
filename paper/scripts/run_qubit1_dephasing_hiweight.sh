#!/bin/bash
#SBATCH --job-name=q1_deph_hi
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --array=0-1
#SBATCH --output=logs/q1_deph_hi_%A_%a.out
#SBATCH --error=logs/q1_deph_hi_%A_%a.out

# Follow-up to run_qubit1_dephasing_3reg.sh: 10x regularizer weights, and
# num_steps cut to 1700 since the 0.02-weight runs converged by step ~1500.
# At weight 0.2 the KL_W anti-alignment (rewards equator residence) should
# visibly distort the pulse, while DV=0.2 should keep/sharpen the fast flip.
#
#   task 0:  KL=0.2
#   task 1:  DV=0.2

PYTHON=~/qmaxcal_venv/bin/python
# Submit from the repository root: sbatch paper/scripts/run_qubit1_dephasing_hiweight.sh
cd "${SLURM_SUBMIT_DIR:-$PWD}"
[ -f train_qmaxcal.py ] || { echo "Submit this script from the repository root." >&2; exit 1; }

KLS=(0.2 0.0)
DVS=(0.0 0.2)

KL=${KLS[$SLURM_ARRAY_TASK_ID]}
DV=${DVS[$SLURM_ARRAY_TASK_ID]}

echo "=== qubit1 pure dephasing  kappa=0.3  kl=$KL  dv=$DV  (1700 steps) ==="

$PYTHON -u train_qmaxcal.py \
    --system systems.single_qubit_dephasing --method exp_split \
    --kappa 0.3 --T_final 3.0 \
    --kl_weight $KL --drift_var_weight $DV \
    --fluence_weight 0.005 --fluence_warmup_steps 2000 \
    --n_time_steps 256 --n_trajs 64 \
    --num_steps 1700 --lr 1e-3 \
    --n_modes 16 --init_scale 1.0 --seed 0 \
    --snap_every 250 \
    --output_dir paper/results/dephasing/qmaxcal/pure_dephasing

echo "done $(date)"
