#!/bin/bash
#SBATCH --job-name=smooth
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=03:30:00
#SBATCH --array=0-3
#SBATCH --output=logs/smooth_%A_%a.out
#SBATCH --error=logs/smooth_%A_%a.out

# Abdelhafez 1st/2nd-derivative smoothness ablation on QMaxCal.
# 4 cells: amp damping × {kl=0, kl=5}  and  STIRAP × {kl=0, kl=1}.
# All use deriv1=1e-3, deriv2=1e-5, smooth_warmup=1500.
#
# Reproduces the headline cells of `tab:smooth_ablation` (Wiener KL is
# not interchangeable with C4/C5 derivative smoothness).

PYTHON=~/qmaxcal_venv/bin/python
# Submit from the repository root: sbatch paper/scripts/run_smoothness_ablation.sh
cd "${SLURM_SUBMIT_DIR:-$PWD}"
[ -f train_qmaxcal.py ] || { echo "Submit this script from the repository root." >&2; exit 1; }

D1=0.001
D2=0.00001
WARM=1500

case $SLURM_ARRAY_TASK_ID in
    0) SYS=systems.amplitude_damping; G=2.0; T=1.0; KL=0.0; FL=0.0
       NMODES=20; NTRAJ=256; NSTEPS=5000; INIT=0.1
       OUT=paper/results/ampdamp/qmaxcal/smooth ;;
    1) SYS=systems.amplitude_damping; G=2.0; T=1.0; KL=5.0; FL=0.0
       NMODES=20; NTRAJ=256; NSTEPS=5000; INIT=0.1
       OUT=paper/results/ampdamp/qmaxcal/smooth ;;
    2) SYS=systems.stirap_3level; G=5.0; T=2.0; KL=0.0; FL=0.0
       NMODES=16; NTRAJ=64;  NSTEPS=10000; INIT=1.0
       OUT=paper/results/stirap/qmaxcal/smooth ;;
    3) SYS=systems.stirap_3level; G=5.0; T=2.0; KL=1.0; FL=0.0
       NMODES=16; NTRAJ=64;  NSTEPS=10000; INIT=1.0
       OUT=paper/results/stirap/qmaxcal/smooth ;;
esac

echo "=== smooth  sys=$SYS  γ=$G  T=$T  kl=$KL  d1=$D1  d2=$D2  warm=$WARM ==="

$PYTHON -u train_qmaxcal.py \
    --system $SYS --method exp_split \
    --gamma $G --T_final $T \
    --kl_weight $KL --drift_var_weight 0.0 --fluence_weight $FL \
    --deriv1_weight $D1 --deriv2_weight $D2 --smooth_warmup_steps $WARM \
    --n_time_steps 256 --n_trajs $NTRAJ \
    --num_steps $NSTEPS --lr 1e-3 \
    --n_modes $NMODES --init_scale $INIT --seed 1 \
    --snap_every 500 \
    --output_dir $OUT

echo "done $(date)"
