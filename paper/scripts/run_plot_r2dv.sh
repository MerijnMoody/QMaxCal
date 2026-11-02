#!/bin/bash
#SBATCH --job-name=plot_r2dv
#SBATCH --partition=staging
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --output=logs/plot_r2dv_%j.out
# Submit from the repository root: sbatch paper/scripts/run_plot_r2dv.sh
cd "${SLURM_SUBMIT_DIR:-$PWD}"
[ -f train_qmaxcal.py ] || { echo "Submit this script from the repository root." >&2; exit 1; }
~/qmaxcal_venv/bin/python -u paper/figures/fig_2dv_3rho_seeds.py
echo done
