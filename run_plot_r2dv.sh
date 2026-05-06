#!/bin/bash
#SBATCH --job-name=plot_r2dv
#SBATCH --partition=staging
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --output=logs/plot_r2dv_%j.out
cd "$(dirname "$(realpath "$0")")"
~/qmaxcal_venv/bin/python -u figures/fig_2dv_3rho_seeds.py
echo done
