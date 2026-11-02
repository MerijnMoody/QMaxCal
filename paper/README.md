# Reproducing the paper

This folder holds the trained controllers, figure scripts and SLURM jobs behind
the figures and tables of the QMaxCal paper. The package itself (solver, systems,
trainers) lives in the repository root; see the [main README](../README.md).

**Run every command from the repository root**, not from this folder.

## Layout

```
paper/
├── figures/        One script per paper figure, plus the rendered .png/.pdf
├── results/        Saved controllers and snapshots behind every figure and table
│   ├── ampdamp/    Section 3: amplitude-damped qubit
│   │   ├── qmaxcal/multiseed_g{γ}/   multi-seed runs per γT
│   │   ├── qmaxcal/smooth/           smoothness ablation
│   │   ├── qmaxcal/lo_phase/         LO-phase appendix
│   │   └── ppo/                      PPO parameters and extracted pulses
│   ├── stirap/     Section 4: STIRAP (qmaxcal/{n64,smooth,smooth_low}, ppo/)
│   ├── diamond/    Section 4: diamond / dark-route system (qmaxcal/multiseed, ppo/, robustness/)
│   ├── chain4/     Section 4: 4-qubit chain (variance_asym_sweep_original/, endnoise_r4/)
│   ├── chain6/     Section 4: 6-qubit IBM Kingston chain (qmaxcal/, ppo/)
│   └── dephasing/  Appendix: single-qubit pure dephasing
├── scripts/        SLURM jobs that regenerate the data in results/
├── scaling/        Section 5 solver-scaling study (DESIGN.md, src/, scripts/, results/)
├── benchmarks/     SSE vs Lindblad solver micro-benchmark (used by scaling/)
└── eval/           Diamond robustness re-evaluation under perturbed γ
```

## Figures

Each figure regenerates from `results/` without retraining, in a few minutes or less:

```bash
# Section 3: amplitude damping
python paper/figures/fig_ampdamp_populations_4panel.py   # populations + variance, γT=2
python paper/figures/fig_ampdamp_alpha2_log.py           # Wiener-KL integrand on log axis
python paper/figures/fig_ampdamp_drift_4panel.py         # drift α(t) (appendix)

# Section 4: STIRAP and diamond
python paper/figures/fig_stirap_compare.py               # populations: baseline / KL / PPO
python paper/figures/fig_diamond_3panel_robust.py        # populations + γ-perturbation robustness

# Section 4: chain-4 and chain-6
python paper/figures/fig_2dv_3rho_seeds.py               # chain-4 ρ sweep
python paper/figures/fig_chain6_q14_q9_4policies.py      # IBM Kingston chain
```

The appendix summary plots (`qubit1_ampdamp_phase*.png`, `qubit1_dephasing_3reg.png`,
`chain4_dephasing_3reg.png`, `chain6_fluence_front_ablation.png`) are stored in
`figures/` as rendered images.

## Retraining

The jobs in `scripts/` are SLURM array jobs for one GPU each. Before the first
submission:

1. `mkdir -p logs` in the repository root (SLURM writes job output there and
   does not create the folder).
2. Edit the `PYTHON=` line in the scripts to point at your interpreter.

Then submit from the repository root, e.g. `sbatch paper/scripts/run_ampdamp_multiseed.sh`.
New runs are written next to the saved ones in `paper/results/`.

### QMaxCal

```bash
sbatch paper/scripts/run_ampdamp_multiseed.sh      # Sec. 3: 5 γT × 2 KL × 2 seeds
sbatch paper/scripts/run_stirap_multiseed.sh       # Sec. 4: 2 γ × 2 KL × 3 seeds
sbatch paper/scripts/run_diamond_multiseed.sh      # Sec. 4: {baseline, KL=0.5, KL=5} × 3 seeds
sbatch paper/scripts/run_smoothness_ablation.sh    # derivative penalties vs Wiener KL
sbatch paper/scripts/run_chain4_asymmetry.sh       # Sec. 4: chain-4 ρ sweep
sbatch paper/scripts/run_chain6_kingston_q14_q9_linflu.sh   # Sec. 4: chain-6 baseline + DV
sbatch paper/results/chain6/qmaxcal/variance_q14_q9_flw0.02/run_seeds.sh
sbatch paper/results/chain6/qmaxcal/variance_q14_q9_flw0.02/run_kl0p1.sh
sbatch paper/results/chain6/qmaxcal/variance_q14_q9_flw0.02/run_kl0p1_seeds.sh
```

The three chain-6 jobs in `results/chain6/.../variance_q14_q9_flw0.02/` use the
copy of `train_qmaxcal.py` stored in that folder, which is the version that
produced those runs.

### PPO baselines

```bash
sbatch paper/scripts/run_ppo_baselines.sh          # amp damping, STIRAP, diamond
sbatch paper/scripts/run_ppo_flu0_sanity.sh        # same with fluence penalty 0
sbatch paper/scripts/run_ch6q149_ppo_sweep.sh      # chain-6 hyperparameter sweep
sbatch paper/scripts/run_ch6q149_ppo_best_seeds.sh # chain-6 best configuration, more seeds
```

`paper/figures/extract_ppo_u_stack.py` turns saved PPO parameters into the
pulse arrays (`ppo_*_ustack.npz`) that the figure scripts read.

## Appendix experiments

### LO-phase dependence of the Wiener KL

The amplitude-damped qubit with jump operator L = √γ e^{iφ} σ₋, σ₋ = |0⟩⟨1|
(`systems/single_qubit_ampdamp.py`). The monitored quadrature is
e^{iφ}σ₋ + e^{−iφ}σ₊ = cos φ σx − sin φ σy. Results go to `results/ampdamp/qmaxcal/lo_phase/`.

```bash
sbatch paper/scripts/run_qubit1_ampdamp_phase_kl5_g2.sh     # γT=2, λ_KLW=5, φ ∈ {0, π/4, π/2}
sbatch paper/scripts/run_qubit1_ampdamp_phase_kl1_g2.sh     # γT=2, λ_KLW=1, seed 0
sbatch paper/scripts/run_qubit1_ampdamp_kl1_g2_seeds.sh     # γT=2, λ_KLW=1, seeds 1, 2
sbatch paper/scripts/run_qubit1_ampdamp_noreg_hinoise.sh    # unregularized, γT ∈ {0.5, 2}
sbatch paper/scripts/run_qubit1_ampdamp_kl_sweep_pi4.sh     # γT=2, φ=π/4, λ_KLW ∈ {0.5, 2}
sbatch paper/scripts/run_qubit1_ampdamp_phase.sh            # γT=0.1, λ_KLW=1, φ ∈ {π/4, π/2}
sbatch paper/scripts/run_qubit1_ampdamp_phi0.sh             # γT=0.1, λ_KLW=1, φ=0
sbatch paper/scripts/run_qubit1_ampdamp_phi0_noreg.sh       # γT=0.1, unregularized
sbatch paper/scripts/run_qubit1_ampdamp_phase_dv.sh         # γT=0.5, R_DV=1, three phases
sbatch paper/scripts/run_qubit1_ampdamp_phase_dv_g2.sh      # γT=2,   R_DV=1, three phases
sbatch paper/scripts/run_qubit1_ampdamp_phase_kl_hinoise.sh # γT ∈ {0.5, 2}, λ_KLW=1, three phases
```

The LO-phase table uses `f_exact_at_best_loss` from `controller_best_by_loss.pkl`.

### Regularizer comparisons

KL_W, R_DV and the LO-phase-invariant Var(L) penalty (`--var_l_weight`):

```bash
sbatch paper/scripts/run_qubit1_dephasing_3reg.sh           # single-qubit dephasing → results/dephasing/qmaxcal/pure_dephasing/
sbatch paper/scripts/run_qubit1_dephasing_hiweight.sh       # same, 10x weights
sbatch paper/scripts/run_chain4_dephasing_3reg.sh           # chain-4 endpoint noise → results/chain4/qmaxcal/endnoise_r4/
sbatch paper/scripts/run_chain6_kingston_varl.sh            # chain-6 Var(L) → results/chain6/qmaxcal/variance_q14_q9_flw0.02/
sbatch paper/scripts/run_chain6_kingston_fluence_front.sh   # chain-6 fluence-only front → results/chain6/qmaxcal/kingston_q14_q9_linflu/
sbatch paper/scripts/run_chain6_kingston_q14_q9_linflu_flw0.sh  # chain-6 linear fluence, weight 0
```

## Solver scaling (Section 5)

See `scaling/DESIGN.md`. The main benchmarks:

```bash
sbatch paper/scaling/scripts/run_em_scaling.sh      # Euler–Maruyama vs exponential splitting, wall-clock vs N
sbatch paper/scaling/scripts/run_lindblad_only.sh   # Lindblad solver out-of-memory scan
```

Results are written to `paper/scaling/results/*.json`. The Lindblad reference
solver needs `qiskit-dynamics` and `diffrax`.
