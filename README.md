# QMaxCal

QMaxCal designs control pulses for open quantum systems by training directly on
simulated measurement trajectories. A pulse is a truncated Fourier series per
control channel. Training samples diffusive quantum trajectories (a stochastic
Schrödinger equation, SSE), differentiates through them with JAX, and maximises
the trajectory-averaged target fidelity plus optional regularizers, such as the
Wiener KL divergence between the measurement record and pure noise. Finished
pulses are scored with an exact Lindblad master-equation solver.

To reproduce the figures and tables of the paper, see [`paper/README.md`](paper/README.md).

## Installation

Tested with Python 3.13, JAX 0.9.2, flax 0.12.6 and optax 0.2.8. Install JAX for your hardware first
(see the [JAX install guide](https://docs.jax.dev/en/latest/installation.html)), then

```bash
pip install flax optax equinox jaxtyping numpy matplotlib
pip install jupyter                       # optional, for quickstart.ipynb
pip install gymnax chex qiskit-dynamics diffrax   # optional, PPO baseline only
```

`dynamiqs` is optional; `core/types.py` falls back to a built-in replacement when
it is not installed. All commands below are run from the repository root.

## Quickstart

Open [`quickstart.ipynb`](quickstart.ipynb). It builds a system, loads trained
controllers, evaluates them with the exact Lindblad solver and with SSE
trajectories, tabulates a folder of runs, and trains a small controller.

From the command line, a training run looks like this:

```bash
python train_qmaxcal.py --system systems.amplitude_damping --gamma 2.0 \
    --kl_weight 5.0 --fluence_weight 0.01 \
    --n_modes 20 --init_scale 0.1 --n_trajs 256 --num_steps 5000 \
    --output_dir runs/ampdamp
```

This takes about 30 minutes on one A100 GPU.
`python train_qmaxcal.py --system systems.<name> --help` lists every option,
including the system's own (e.g. `--gamma`, `--lo_phase`).

## Training options

The loss is

```
loss = (1 − F_SSE) + λ_KL·KL_W + λ_DV·R_DV + λ_flu·fluence + λ_VarL·Var(L) + smoothness terms
```

| Flag | Default | Meaning |
|---|---|---|
| `--kl_weight` | 0 | Wiener KL, ½ E[∫ Σ_k α_k² dt], with record drift α_k = 2 Re⟨L_k⟩ |
| `--drift_var_weight` | 0 | R_DV, the time variance of the record drift α |
| `--var_l_weight` | 0 | Var(L) = E[∫ (⟨L†L⟩ − \|⟨L⟩\|²) dt], invariant under the local-oscillator phase |
| `--fluence_weight` | 0 | Pulse energy ∫ \|u\|² dt (`--fluence_max` turns it into a hinge above a budget) |
| `--deriv1_weight`, `--deriv2_weight` | 0 | Penalties on the first and second time derivatives of the pulse |
| `--fluence_warmup_steps`, `--drift_var_warmup_steps`, `--smooth_warmup_steps` | 0 | Steps before the corresponding penalty switches on |
| `--n_modes` | 16 | Fourier modes per control channel |
| `--init_scale` | 1.0 | Scale of the random initial coefficients |
| `--n_trajs` | 128 | SSE trajectories per training step |
| `--n_time_steps` | 256 | SSE time steps over [0, T] |
| `--num_steps`, `--lr` | 10000, 1e-3 | Adam steps and learning rate |
| `--method` | `exp_split` | SSE integrator: `exp_split` (exponential splitting) or `euler` (Euler–Maruyama) |
| `--snap_every` | 500 | How often to evaluate with the exact Lindblad solver and save a snapshot |
| `--seed`, `--output_dir` | 0, `results` | Random seed and output folder |

## Outputs

Each run writes one folder named after the system and the loss weights, e.g.
`ampdamp_g2.0_T1.0_kl5.0_dv0.0_fl0.01_seed0/`, containing:

- `controller.pkl`: Fourier coefficients at the last training step (`best_dc`,
  `best_cos_coeffs`, `best_sin_coeffs`), plus `best_exact_fidelity` and
  `best_step` of the best exact-Lindblad snapshot.
- `controller_best_by_loss.pkl`: coefficients at the step with the lowest
  training loss, and their exact fidelity `f_exact_at_best_loss`.
- `snap_*.pkl`: periodic snapshots with populations, pulses and regularizer values.

To rebuild and evaluate a saved controller:

```python
import pickle, jax.numpy as jnp
from flax import nnx
import train_qmaxcal as tq
from systems import amplitude_damping

ck = pickle.load(open("runs/ampdamp/<run>/controller_best_by_loss.pkl", "rb"))
n_ctrl, n_modes = ck["best_cos_coeffs"].shape
ctrl = tq.FourierController(n_ctrl, n_modes, ck["T_final"], rngs=nnx.Rngs(0))
ctrl.dc, ctrl.cos_coeffs, ctrl.sin_coeffs = (nnx.Param(jnp.asarray(ck[k]))
    for k in ("best_dc", "best_cos_coeffs", "best_sin_coeffs"))

sysd = amplitude_damping.create_system(**ck["system_kwargs"])
times, pops, rho_T = tq.lindblad_trajectory(ctrl, sysd)   # exact Lindblad, RK4
```

`plot_snapshots.py` plots the latest snapshot of one or more run folders, and
`plot_with_variance.py` plots the mean ± std of SSE populations for saved controllers.

## Adding a system

A system is a module in `systems/` with two functions:

- `create_system(**kwargs)` returns a dict with
  - `H_sys` (drift Hamiltonian), `H_ops` (list of control operators), `jump_ops` (list of jump operators L_k), all as complex64 JAX arrays;
  - `psi0` and `target_state`, column vectors of shape (dim, 1);
  - `T_final`, `n_controls`, `control_labels`, `system_name` (used in the run folder name);
  - `populations`: dict of name → projector, used for plots and snapshots;
  - optionally `forbidden_states` (names in `populations` whose peak and integral are logged) and `control_transform` (`'abs'`, `'square'`, `'softplus'` to constrain the sign of the pulse).
- `add_system_args(parser)` adds the system's own command-line flags. Their
  values are passed to `create_system`.

`systems/amplitude_damping.py` is a short example. The PPO baseline also reads
`n_dim`, `signal_scale` and `action_range`.

## Repository layout

```
core/                 SSE solver (dssesolve), integrators, density-matrix utilities
systems/              Physical systems (one module each)
train_qmaxcal.py      QMaxCal trainer: Fourier controller, SSE loss, exact Lindblad evaluation
train_ppo.py          PPO baseline trainer (uses PPO/rl_working/)
PPO/rl_working/       PPO implementation and RL environment helpers
plot_snapshots.py     Plot the latest snapshot of run folders
plot_with_variance.py Plot SSE population mean ± std for saved controllers
quickstart.ipynb      Guided tour
paper/                Everything needed to reproduce the paper (see paper/README.md)
```

The PPO baseline code in `PPO/rl_working/` is adapted from the RL codebase of
Ernst et al. (2025).
