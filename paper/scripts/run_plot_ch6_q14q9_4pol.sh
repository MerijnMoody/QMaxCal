#!/bin/bash
#SBATCH --job-name=plot_4pol
#SBATCH --partition=staging
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --output=logs/plot_4pol_%j.out
# Submit from the repository root: sbatch paper/scripts/run_plot_ch6_q14q9_4pol.sh
cd "${SLURM_SUBMIT_DIR:-$PWD}"
[ -f train_qmaxcal.py ] || { echo "Submit this script from the repository root." >&2; exit 1; }
~/qmaxcal_venv/bin/python -u paper/figures/fig_chain6_q14_q9_4policies.py
echo done
