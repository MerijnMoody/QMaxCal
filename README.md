# QMaxCal — paper artifacts

Self-contained snapshot of the code and data needed to reproduce all
experiments and figures of the QMaxCal paper.

## Layout

```
qmaxcal_paper/
├── core/                              SSE / Lindblad solver
├── PPO/rl_working/                  PPO baseline + RL environment helpers
├── systems/                           Per-experiment Hamiltonians + jump ops
│   ├── amplitude_damping.py           Single-qubit T1 relaxation (Section 3)
│   ├── stirap_3level.py               STIRAP 3-level Λ system (Section 4)
│   ├── dark_route_v3.py               Diamond / dark-route 4-level system (Section 4)
│   ├── spin_chain_4.py                4-qubit chain with edge-noise asymmetry (Section 4)
│   ├── spin_chain_6_kingston_q14_q9.py 6-qubit IBM Kingston q14→q9 chain (Section 4)
│   ├── single_qubit_ampdamp.py        Amplitude damping with tunable LO phase φ (appendix)
│   └── single_qubit_dephasing.py      Single-qubit pure dephasing (appendix)
├── train_qmaxcal.py                   QMaxCal training (Fourier controller, SSE loss;
│                                      supports KL_W, drift-variance, fluence,
│                                      and Abdelhafez C4/C5 derivative penalties)
├── train_ppo.py                       PPO baseline (uses PPO/rl_working/)
├── figures/                           Paper figure scripts + .png outputs
│   ├── fig_ampdamp_populations_4panel.py  Section 3 amp-damping populations (4 panels)
│   ├── fig_ampdamp_alpha2_log.py          Wiener-KL integrand on log axis
│   ├── fig_ampdamp_drift_4panel.py        Drift α(t) per policy (appendix)
│   ├── fig_stirap_compare.py              STIRAP populations: baseline / KL / PPO
│   ├── fig_diamond_3panel_robust.py       Section 4 diamond + γ-robustness
│   ├── fig_2dv_3rho_seeds.py              Section 4 chain-4 ρ-sweep figure
│   ├── fig_chain6_q14_q9_4policies.py     Section 4 IBM Kingston chain
│   └── extract_ppo_u_stack.py           Helper: extract det. action sequence from PPO params
├── scaling/                           SSE scaling experiment (Section 5)
│   ├── DESIGN.md
│   ├── src/                           bench_*.py
│   ├── scripts/                       slurm submission scripts
│   └── results/                       JSON output (per cell)
├── results/                           Paper-figure data (saved snapshots)
│   ├── ampdamp/qmaxcal/multiseed_g{γ}/  multi-seed amp damping controllers
│   ├── ampdamp/qmaxcal/smooth/          smoothness-ablation controllers
│   ├── ampdamp/ppo/                     PPO params + extracted u_stack
│   ├── stirap/qmaxcal/{n64,smooth}/     STIRAP controllers (KL, smoothness)
│   ├── stirap/ppo/                      PPO params + extracted u_stack
│   ├── diamond/qmaxcal/multiseed/       diamond controllers (baseline + KL)
│   ├── diamond/ppo/                     PPO checkpoints
│   ├── diamond/robustness/              dr3_robustness_multiseed.npz (γ-perturbation eval)
│   ├── chain4/qmaxcal/variance_asym_sweep_original/
│   ├── chain6/qmaxcal/variance_q14_q9_flw0.02/
│   └── chain6/ppo/q14_q9_sm0.001_ks4.0/
├── benchmarks/                        Solver micro-benchmark (used by scaling/)
├── run_*.sh                           SLURM scripts to (re)produce data + figures
└── logs/                              SLURM stdout/stderr (gitignored)
```

## Reproducing the figures

Each figure regenerates from the saved snapshots in `results/`:

```bash
# Section 3 — amplitude damping
python figures/fig_ampdamp_populations_4panel.py   # populations + variance, γT=2
python figures/fig_ampdamp_alpha2_log.py           # Wiener-KL integrand on log axis
python figures/fig_ampdamp_drift_4panel.py         # drift α(t) (appendix)

# Section 4 — STIRAP and diamond
python figures/fig_stirap_compare.py               # populations: baseline/KL/PPO
python figures/fig_diamond_3panel_robust.py        # populations + γ-perturbation robustness

# Section 4 — chain-4 / chain-6
sbatch run_plot_r2dv.sh                            # → figures/fig_2dv_3rho_seeds.png
sbatch run_plot_ch6_q14q9_4pol.sh                  # → figures/fig_chain6_q14_q9_4policies.png
```

## Reproducing the data

QMaxCal training (no PPO dependency):
```bash
# Section 3 — amp damping multi-seed (5 γT × 2 KL × 2 seeds = 20 tasks)
sbatch run_ampdamp_multiseed.sh

# Section 4 — STIRAP multi-seed (2 γ × 2 KL × 3 seeds = 12 tasks)
sbatch run_stirap_multiseed.sh

# Section 4 — diamond multi-seed ({baseline, KL=0.5, KL=5} × 3 seeds = 9 tasks)
sbatch run_diamond_multiseed.sh

# Smoothness ablation (Abdelhafez C4/C5 vs Wiener-KL)
sbatch run_smoothness_ablation.sh

# Section 4 — chain-4 / chain-6
sbatch run_chain4_asymmetry.sh                   # chain-4 ρ-sweep
sbatch run_chain6_kingston_q14_q9_linflu.sh      # chain-6 baseline + DV
sbatch results/chain6/qmaxcal/variance_q14_q9_flw0.02/run_seeds.sh
sbatch results/chain6/qmaxcal/variance_q14_q9_flw0.02/run_kl0p1.sh
sbatch results/chain6/qmaxcal/variance_q14_q9_flw0.02/run_kl0p1_seeds.sh
```

PPO training (PPO/rl_working/ bundled in this repo):
```bash
# Phase-B winners for amp damping, STIRAP, and diamond
sbatch run_ppo_baselines.sh

# Fluence-penalty sanity check (λ_flu=0 across the same three systems)
sbatch run_ppo_flu0_sanity.sh

# Section 4 chain-6
sbatch run_ch6q149_ppo_sweep.sh
sbatch run_ch6q149_ppo_best_seeds.sh
```

## Additional experiments (appendix)

LO-phase dependence of the Wiener KL on the amplitude-damped qubit. The
jump operator is L = √γ e^{iφ} σ₋ with σ₋ = |0⟩⟨1|, so the monitored
quadrature is e^{iφ}σ₋ + e^{−iφ}σ₊ = cos φ σx − sin φ σy. Results go to
`results/ampdamp/qmaxcal/lo_phase/`.
```bash
sbatch run_qubit1_ampdamp_phase_kl5_g2.sh     # γT=2, λ_KLW=5, φ ∈ {0, π/4, π/2}
sbatch run_qubit1_ampdamp_phase_kl1_g2.sh     # γT=2, λ_KLW=1, seed 0
sbatch run_qubit1_ampdamp_kl1_g2_seeds.sh     # γT=2, λ_KLW=1, seeds 1, 2
sbatch run_qubit1_ampdamp_noreg_hinoise.sh    # unregularized, γT ∈ {0.5, 2}
sbatch run_qubit1_ampdamp_kl_sweep_pi4.sh     # γT=2, φ=π/4, λ_KLW ∈ {0.5, 2}
sbatch run_qubit1_ampdamp_phase.sh            # γT=0.1, λ_KLW=1, φ ∈ {π/4, π/2}
sbatch run_qubit1_ampdamp_phi0.sh             # γT=0.1, λ_KLW=1, φ=0
sbatch run_qubit1_ampdamp_phi0_noreg.sh       # γT=0.1, unregularized
sbatch run_qubit1_ampdamp_phase_dv.sh         # γT=0.5, R_DV=1, three phases
sbatch run_qubit1_ampdamp_phase_dv_g2.sh      # γT=2,   R_DV=1, three phases
sbatch run_qubit1_ampdamp_phase_kl_hinoise.sh # γT ∈ {0.5, 2}, λ_KLW=1, three phases
```
Each run directory holds `controller.pkl` (best exact-Lindblad snapshot) and
`controller_best_by_loss.pkl` (lowest training loss; used for the LO-phase table).

Regularizer comparisons (KL_W, R_DV and the LO-phase-invariant Var(L)
penalty, `--var_l_weight` in `train_qmaxcal.py`):
```bash
sbatch run_qubit1_dephasing_3reg.sh           # single-qubit dephasing → results/dephasing/qmaxcal/pure_dephasing/
sbatch run_qubit1_dephasing_hiweight.sh       # same, 10x weights
sbatch run_chain4_dephasing_3reg.sh           # chain-4 endpoint noise → results/chain4/qmaxcal/endnoise_r4/
sbatch run_chain6_kingston_varl.sh            # chain-6 Var(L) penalty → results/chain6/qmaxcal/variance_q14_q9_flw0.02/
sbatch run_chain6_kingston_fluence_front.sh   # chain-6 fluence-only front → results/chain6/qmaxcal/kingston_q14_q9_linflu/
sbatch run_chain6_kingston_q14_q9_linflu_flw0.sh  # chain-6 linear fluence, weight 0
```
Summary plots: `figures/qubit1_ampdamp_phase*.png`, `figures/qubit1_dephasing_3reg.png`,
`figures/chain4_dephasing_3reg.png`, `figures/chain6_fluence_front_ablation.png`.

## Scaling experiment

See `scaling/DESIGN.md`. Key benchmarks:

```bash
sbatch scaling/scripts/run_em_scaling.sh   # EM vs ExpSplit per-step wall-clock vs N
sbatch scaling/scripts/run_lindblad_only.sh  # Lindblad solver OOM scan
```

Results land in `scaling/results/*.json`; plots can be added on top.

## Dependencies

- `jax`, `flax`, `optax`, `dynamiqs`, `numpy`, `matplotlib`, `pickle` — for QMaxCal.
- `qiskit-dynamics`, `diffrax` — for the Lindblad reference solver in `scaling/`.
- For PPO retraining only: `gymnax`, `chex` (the PPO RL codebase
  is included under `PPO/rl_working/`).

The PPO baseline code in `PPO/rl_working/` is adapted from the RL codebase of
Ernst et al. (2025).

A python venv at `~/qmaxcal_venv/` is assumed by the SLURM scripts;
adapt the `PYTHON=` lines to your local interpreter.
