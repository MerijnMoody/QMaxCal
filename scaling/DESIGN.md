# SSE sampling scaling experiment

## Question

> Does the trajectory approach scale better than master-equation simulation
> as system size grows?

## Three axes

| # | Axis | Measurement | What it shows |
|---|---|---|---|
| 1 | **Per-step wall-clock vs N** | Time for one forward simulation, for both solvers, at fixed n_trajs | Confirms theoretical $O(2^N)$ vs $O(4^N)$ |
| 2 | **Fidelity-variance vs n_trajs** | $\sigma(F)$ across reseeds, as a function of n_trajs at fixed N | How many trajectories you actually need |
| 3 | **Time-to-fixed-accuracy vs N** | Combine 1 and 2: $\text{wall-time}_\text{SSE}(N) = n_\text{trajs}^*(N, \sigma) \cdot t_\text{per-traj}(N)$ | Crossover where SSE overtakes Lindblad |

## Test system

Parameterised N-qubit XX+YY chain with random per-site couplings + per-site
$\sigma_-$/$\sigma_z$ Lindblad channels (re-using `solver_benchmark.py`'s
`random_chain` setup).

Closed-form Fourier control pulses (`n_modes=4`, random coefficients,
fixed amplitude). $T_\text{final}=1.0$ (dimensionless).

## Sweep ranges

- **N**: $\{4, 6, 8, 10, 12\}$ for both solvers; $\{14\}$ SSE-only
  (the Lindblad solver is expected to OOM around $N{\approx}13$ on a 40 GB A100).
- **n_trajs (axis 1, axis 2)**: $\{16, 32, 64, 128, 256, 512, 1024\}$.
- **Reseeds (axis 2)**: 50 RNG seeds per (N, n_trajs) cell.
- **Repeats for wall-clock (axis 1)**: 5 timed reps after 2 JIT warmups, report median.

## Metrics

- **Wall-clock** (`t_ms`): seconds per single forward pass, post-JIT.
- **Trace distance to Lindblad ground truth** (`trace_dist`): one number per (N, n_trajs).
- **Fidelity-estimate std** (`sigma_F`): std of $|\langle\psi_\text{tgt}|\psi_T\rangle|^2$ across reseeds.
- **OOM**: boolean for each (N, solver).

## Outputs

- `results/wallclock.json`: rows `(N, solver, n_trajs, t_ms, trace_dist, oom)`
- `results/variance.json`: rows `(N, n_trajs, sigma_F, n_reseeds)`
- `figures/scaling.png`: 1×3 panel (wall-clock vs N, $\sigma_F$ vs n_trajs, time-to-accuracy crossover)

## Risks (to address openly in the paper)

1. **n_trajs needed for stable gradients may be large**: if $\sigma_F$ at $n_\text{trajs}=64$ is large enough that we need $n_\text{trajs}=256$+, the SSE crossover moves to $N\geq 9$. Honest framing: *"the trajectory advantage materialises for $N \geq X$".*
2. **Lindblad memory ceiling**: the Lindblad solver stores $4^N$-element density matrix and several Dopri5 stage buffers. We expect OOM near $N{=}13$ on 40 GB A100, more aggressively if `qiskit-dynamics` doesn't free intermediates.
3. **Different time-discretisation**: SSE uses fixed $dt$, Lindblad uses adaptive Dopri5. Both are well-converged at the chosen tolerances, but a reviewer might ask. We control for this by checking `trace_dist` is at the per-trajectory sampling-noise floor across all $N$.

## Dev plan

1. Write `src/system.py` (parameterised chain — copy and lightly refactor `solver_benchmark.py`).
2. Write `src/bench_wallclock.py` (axis 1, also produces trace_dist).
3. Write `src/bench_variance.py` (axis 2, reseeds).
4. Smoke-test on $N\in\{4, 6\}$ (~5 min on a single A100).
5. Once smoke passes, scale to full $N$ range (one slurm job, ~3 h).
6. Plot.
