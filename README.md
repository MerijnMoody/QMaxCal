# QMaxCal — paper artifacts

Self-contained snapshot of the code and data needed to reproduce the
chain-experiment figures and the SSE scaling experiment from the QMaxCal
paper.

## Layout

```
qmaxcal_paper/
├── core/                              SSE / Lindblad solver
├── Ernst/rl_working/                  PPO + RL environment helpers (Ernst 2025 codebase)
├── systems/                           Per-experiment Hamiltonians + jump ops
│   ├── spin_chain_4.py                4-qubit chain with edge-noise asymmetry
│   └── spin_chain_6_kingston_q14_q9.py 6-qubit IBM Kingston q14→q9 chain
├── train_qmaxcal.py                   QMaxCal training (Fourier controller, SSE loss)
├── train_ppo.py                       PPO baseline (uses Ernst/rl_working/)
├── figures/                           Paper figure scripts + .png outputs
│   ├── fig_2dv_3rho_seeds.py          Section 4 chain-4 ρ-sweep figure
│   └── fig_chain6_q14_q9_4policies.py 4-policy comparison on the IBM Kingston chain
├── scaling/                           SSE scaling experiment (Section 5)
│   ├── DESIGN.md
│   ├── src/                           bench_*.py
│   ├── scripts/                       slurm submission scripts
│   └── results/                       JSON output (per cell)
├── results/                           Paper-figure data (saved snapshots)
│   ├── chain4/qmaxcal/variance_asym_sweep_original/
│   ├── chain6/qmaxcal/variance_q14_q9_flw0.02/
│   └── chain6/ppo/q14_q9_sm0.001_ks4.0/
├── benchmarks/                        Solver micro-benchmark (used by scaling/)
├── run_*.sh                           SLURM scripts to (re)produce data + figures
└── logs/                              SLURM stdout/stderr (gitignored)
```

## Reproducing the figures

The two paper figures regenerate from the saved snapshots in `results/`:

```bash
sbatch run_plot_r2dv.sh           # → figures/fig_2dv_3rho_seeds.png
sbatch run_plot_ch6_q14q9_4pol.sh # → figures/fig_chain6_q14_q9_4policies.png
```

## Reproducing the data

QMaxCal training (no Ernst dependency):
```bash
sbatch run_chain4_asymmetry.sh                   # chain-4 ρ-sweep
sbatch run_chain6_kingston_q14_q9_linflu.sh      # chain-6 baseline + DV
# Multi-seed variance scripts live inside the variance_* result dirs:
sbatch results/chain6/qmaxcal/variance_q14_q9_flw0.02/run_seeds.sh
sbatch results/chain6/qmaxcal/variance_q14_q9_flw0.02/run_kl0p1.sh
sbatch results/chain6/qmaxcal/variance_q14_q9_flw0.02/run_kl0p1_seeds.sh
```

PPO training (Ernst/rl_working/ bundled in this repo):
```bash
sbatch run_ch6q149_ppo_sweep.sh
sbatch run_ch6q149_ppo_best_seeds.sh
```

## Scaling experiment

See `scaling/DESIGN.md`. Key benchmarks:

```bash
sbatch scaling/scripts/run_em_scaling.sh   # EM vs ExpSplit per-step wall-clock vs N
sbatch scaling/scripts/run_ernst_only.sh   # Ernst Lindblad OOM scan
```

Results land in `scaling/results/*.json`; plots can be added on top.

## Dependencies

- `jax`, `flax`, `optax`, `dynamiqs`, `numpy`, `matplotlib`, `pickle` — for QMaxCal.
- `qiskit-dynamics`, `diffrax` — for the Lindblad reference solver in `scaling/`.
- For PPO retraining only: `gymnax`, `chex` (the Ernst RL codebase
  is included under `Ernst/rl_working/`).

A python venv at `~/qmaxcal_venv/` is assumed by the SLURM scripts;
adapt the `PYTHON=` lines to your local interpreter.
