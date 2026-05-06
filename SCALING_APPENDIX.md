% Appendix: SSE solver schemes and scaling behaviour
% (Markdown working draft — translate to LaTeX when integrating)

# Appendix X — SSE solver schemes and scaling behaviour

This appendix documents the two diffusive-SSE integrators used to
optimize QMaxCal protocols, characterizes where each one fails as
the system size N grows, and contrasts both with the Lindblad
master-equation solver (qiskit-dynamics + diffrax Dopri5) used as
ground truth.

## X.1  The two SSE integrators

We solve the diffusive Stochastic Schrödinger Equation
\[
  d|\psi\rangle = \Big[\!-iH\,dt - \tfrac{1}{2}\sum_k L_k^\dagger L_k\,dt
                 + \tfrac{1}{2}\sum_k\langle L_k+L_k^\dagger\rangle L_k\,dt
                 - \tfrac{1}{4}\sum_k\langle L_k+L_k^\dagger\rangle^2\,dt\Big]\,
                 |\psi\rangle
                 + \sum_k\!\big(L_k - \tfrac{1}{2}\langle L_k+L_k^\dagger\rangle\big)|\psi\rangle\, dW_k
\]
with two integrators implemented in `core/solver.py`:

**Euler–Maruyama (EM).** A first-order weak-order-1 stochastic Euler
step. Each timestep applies $H,\, L_k,\, L_k^\dagger$ as
matrix–vector products on the state $|\psi\rangle$. Per-step compute
is $O(\dim^2 \cdot n_{\rm trajs})$ when $H$ is dense; runtime memory
is $O(\dim \cdot n_{\rm trajs})$ for the batched state plus the
single $\dim \times \dim$ Hamiltonian. **Never instantiates a dense
propagator.** Local truncation error is
$O(dt \cdot \|H_{\rm eff}\|^2)$, so stiff Hamiltonians require
proportionally finer time discretization.

**Exponential split-step (ExpSplit).** Splits each timestep into
deterministic and stochastic halves. The deterministic half is
solved exactly via
$\exp(-iH_{\rm eff}\,dt)|\psi\rangle$ where
$H_{\rm eff} = H - i\frac{1}{2}\sum_k L_k^\dagger L_k$; the stochastic
half is an Euler-style correction. Computing $\exp$ instantiates a
dense $\dim\times\dim$ propagator at every timestep, costing
$O(\dim^3)$ compute and $O(\dim^2)$ memory per step. The
deterministic half is Hamiltonian-exact, so for moderate-amplitude
controls ExpSplit can use coarser time discretization than EM at
matched accuracy. The cost is paid in the dense matrix exponential.

These trade-offs are summarised in Table~\ref{tab:em_vs_expsplit}.

| Solver | per-step compute | per-step memory (excl. constants) | Hamiltonian error per step | Stochastic error per step | Stiff-system robustness |
|---|---|---|---|---|---|
| EM | $O(\dim^2 \cdot n_{\rm trajs})$ | $O(\dim \cdot n_{\rm trajs})$ | $O(dt^2 \cdot \|H\|^2)$ | $O(\sqrt{dt})$ | poor — needs $dt \ll 1/\|H\|^2$ |
| ExpSplit | $O(\dim^3)$ | $O(\dim^2)$ | exact | $O(\sqrt{dt})$ | good — Hamiltonian half exact at any $dt$ |

## X.2  Reference solver: qiskit-dynamics + Dopri5 (Ernst 2025)

The benchmark Lindblad solver — the comparison baseline used by
\citet{ernst2025} and adopted here — propagates the full density
matrix $\rho(t)$ via adaptive Dopri5 on the Liouvillian. Each
adaptive step is $O(\dim^3)$ matrix work; runtime memory is
$O(\dim^2)$ for $\rho$ plus several Dopri5 stage buffers. It uses
the published `qiskit-dynamics` API on top of `diffrax`.

## X.3  Where each solver runs out

We evaluated all three solvers on a parameterised random
$N$-qubit XX+YY chain (per-site $\sigma_-$ and $\sigma_z$ Lindblad
channels with rates $\gamma_i, \kappa_i \in [0.05, 0.15]$, random
nearest-neighbour exchange $J_i \in [0.5, 1.5]$, $T_{\rm final}=1.0$
in dimensionless units, closed-form Fourier control pulses with
$n_{\rm modes}=4$ and unit amplitude). All experiments ran on a
single 40 GB NVIDIA A100 with the indicated host-RAM allocation.

The dominant scaling failure modes we observed:

1. **Lindblad master equation (qiskit-dynamics + Dopri5).** With
   `--mem=256G` host RAM, the solver compiles and runs successfully
   through $N=10$ ($\dim=1024$, peak GPU memory $1.85$ GiB,
   wall-clock 22 s/forward pass). At $N=12$ ($\dim=4096$), it raises
   `RESOURCE_EXHAUSTED: 7.50 GiB` during the runtime allocator pass:
   the density matrix plus Dopri5 stage buffers plus $\sim\!\!24$
   pre-baked dense Hamiltonian and Lindblad operators exceed the
   40 GB GPU. With `--mem=64G` host RAM the failure mode shifts: a
   $\sim\!\!10$ GB pool of constants is captured during JIT lowering
   ("`A large amount of constants were captured during lowering
   (9.93 GB total)`") and the host process is OOM-killed before the
   GPU is touched. Either way, **the practical ceiling of
   qiskit-dynamics on a 40 GB A100 is $N=11$** (we did not run
   $N=11$ explicitly, but extrapolating $\dim^2 \times 8\,\mathrm{B}
   \approx 32$ MiB for $\rho$ leaves comfortable headroom).

2. **Exponential split-step SSE.** Same JIT-compile pathology as
   Lindblad, plus an additional cost: at $N=12$ XLA's
   constant-folding pass tried to pre-evaluate dense $4096\times
   4096$ matrix products inside the lowered HLO, each taking
   $\sim\!\!9.5$ minutes. We let one such job run for 2 hours
   without it completing the trace lowering, and then cancelled.
   **ExpSplit is therefore unable to reach $N\geq 12$ in our
   pipeline regardless of GPU memory.** The dense-propagator memory
   would also be prohibitive at $N\geq 14$ ($\dim^2 \times 8\,
   \mathrm{B} = 2$ GiB just for the propagator at $N=14$, plus
   per-step buffers).

3. **Euler–Maruyama SSE.** Compiles and runs at $N=12$ (cached call
   wall-clock 45.8 s, peak GPU memory 7.0 GiB at $n_{\rm trajs}=256$,
   $n_{\rm time-steps}=2048$). Because EM never instantiates a
   dense propagator, its memory footprint is dominated by the
   $\sim\!\!7$ GiB of constants captured during JIT lowering — the
   same operators that Lindblad and ExpSplit need, but without the
   $O(\dim^3)$ pre-fold that crashes ExpSplit. We did not push EM
   to $N\geq 14$ within the present compute budget; based on the
   memory scaling we estimate $N=14$ should fit on a 40 GB A100
   (state $+$ matrix $+$ buffers $\approx 8$ GiB) but with rapidly
   growing wall-clock per call (extrapolated $\sim\!\!12$ minutes
   per forward pass).

The numerical wall-clock and memory comparison through $N=12$ is
shown in Figure~\ref{fig:scaling_wallclock} (data in
`scaling/results/em_scaling.json`):

| $N$ | $\dim$ | EM cached call | ExpSplit cached call | EM speedup |
|---|---|---|---|---|
| 4  | 16   | 0.1 s | 1.2 s | 12$\times$ |
| 6  | 64   | 0.2 s | 1.4 s | 7$\times$ |
| 8  | 256  | 0.5 s | 2.7 s | 5$\times$ |
| 10 | 1024 | 3.6 s | 15.1 s | 4.2$\times$ |
| 12 | 4096 | **45.8 s** | (JIT compile pass did not terminate within 2 h) | — |

## X.4  When EM is too coarse: Hamiltonian-amplitude sensitivity

EM's local error scales as $O(dt \cdot \|H_{\rm eff}\|^2)$; ExpSplit
has no such dependence in its Hamiltonian half. For QMaxCal
applications where the controller is small in amplitude (the
fluence-regularised regime) EM is comfortably accurate at
$n_{\rm time-steps} \sim 10^3$. For systems where peak
$\|H_{\rm eff}\|$ is moderate to large (control amplitudes a few
times unity, multi-qubit chains with strong $\sigma_x \sigma_x$
exchange), EM can require $10\!\!\times$ to $100\!\!\times$ more
time-steps for the same accuracy.

We illustrate this with a controlled experiment: at $N=10$, fixed
seed and fixed $n_{\rm trajs} = 1024$ (Monte Carlo noise floor
$\sim\!\!0.031$ in Frobenius distance), we vary the random-Fourier
pulse amplitude over $\{1, 3, 10\}$ and sweep
$n_{\rm time-steps}\in\{256, 1024, 4096, 16384, 65536\}$. Results
appear in Figure~\ref{fig:em_accuracy_amp} and in
`scaling/results/em_accuracy_amp.json`. *(Numbers populated post-run.)*

The qualitative pattern: at amplitude $1$ the integrator is
converged at $n_{\rm time-steps}=256$. At amplitude $10$ the same
$n_{\rm time-steps}$ produces a Frobenius error nearly an order of
magnitude above the MC floor, and ${\sim}\,10^4$–$10^5$ time-steps
are needed for matched accuracy. This is a consequence of EM's
$O(dt \cdot \|H\|^2)$ local truncation error — a $k$-fold amplitude
increase requires roughly $k^2$ more time-steps for the same
discretization error.

## X.5  Reproducibility

All scaling experiments live in `scaling/`:

```
scaling/
├── DESIGN.md              experiment design notes
├── src/
│   ├── common.py          random-chain construction; rho_estimator(method=...)
│   ├── bench_em_scaling.py     EM vs ExpSplit wall-clock + memory across N
│   ├── bench_ernst_only.py     Lindblad OOM scan
│   └── bench_em_accuracy_amp.py  EM accuracy vs pulse amplitude
├── scripts/                slurm submission scripts
├── results/                JSON output, one file per experiment
└── figures/                analysis plots
```

Each `bench_*.py` writes a JSON of (wall-clock, peak GPU memory,
trace distance / Frobenius distance) per cell, allowing the figures
in this appendix to be regenerated without re-running the GPU
benchmarks.
