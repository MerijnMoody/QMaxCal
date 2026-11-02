"""Shared utilities for the SSE scaling benchmark.

Imports the existing solver_benchmark helpers, adds OOM-aware versions and
a population observable for the variance benchmark.
"""
import os, sys, time
_PAPER = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(_PAPER))   # repo root: core/, systems/
sys.path.insert(0, _PAPER)                    # paper/: benchmarks/

import numpy as np
import jax
import jax.numpy as jnp

# Reuse the chain construction + Fourier-pulse + Lindblad/SSE wrappers
# already validated in benchmarks/solver_benchmark.py.
from benchmarks.solver_benchmark import (   # noqa: E402
    random_chain,
    random_fourier_coeffs,
    make_H_con_fourier,
    lindblad_evolve,
    sse_evolve,
    trace_distance,
    time_after_warmup,
)
from core.types import ExponentialSplit, asqarray   # noqa: E402
from core.solver import dssesolve                   # noqa: E402


def rho_estimator(sys_dict, a, b, T, n_trajs, n_time_steps, method='exp_split'):
    """Build a JIT-able function that takes a PRNGKey and returns the
    SSE estimate of rho(T) = E_psi[|psi(T)><psi(T)|], averaged over
    n_trajs trajectories.

    method ∈ {'exp_split', 'euler_maruyama'} selects the integrator:
      - 'exp_split'      : ExponentialSplit (dense matrix-exp per step,
                            O(dim^3) per step, exact Hamiltonian part)
      - 'euler_maruyama' : Euler–Maruyama (matrix-vector only,
                            O(dim^2) per step with dense H, O(dim) sparse)
    """
    from core.types import EulerMaruyama  # local import: cheap
    H_ops_q = [asqarray(H) for H in sys_dict['H_ops']]
    H_con   = make_H_con_fourier(a, b, H_ops_q, T)
    psi0_q  = asqarray(sys_dict['psi0'])
    H_sys_q = asqarray(sys_dict['H_sys'])
    jump_q  = [asqarray(L) for L in sys_dict['jump_ops']]
    tsave   = jnp.array([0.0, T])
    dt      = T / n_time_steps
    integ   = EulerMaruyama(dt=dt) if method == 'euler_maruyama' else ExponentialSplit(dt=dt)

    def estimate(key):
        keys = jax.random.split(key, n_trajs)
        result = dssesolve(
            H_sys=H_sys_q, jump_ops=jump_q,
            psi0=psi0_q, tsave=tsave, keys=keys,
            H_con=H_con, method=integ,
            nsteps=n_time_steps, nsteps_per_save=n_time_steps,
        )
        # states.data shape: (n_trajs, n_save_times, dim, 1)
        psi_T = result.states.data[:, -1, :, 0]            # (n_trajs, dim)
        # rho_T = mean_traj [ |psi><psi| ] = (1/n_trajs) Σ ψψ†
        rho_T = jnp.einsum('nd,ne->de', psi_T, jnp.conj(psi_T)) / n_trajs
        return rho_T

    return estimate


def frobenius_distance(rho_a, rho_b):
    """||rho_a - rho_b||_F = sqrt( sum_ij |A_ij - B_ij|^2 ).
    Cheap (O(dim^2)) and a clean closeness metric; bounded by sqrt(2)
    for normalized density matrices.
    """
    diff = np.asarray(rho_a) - np.asarray(rho_b)
    return float(np.sqrt(np.sum(np.abs(diff) ** 2)))


def safe_eval(callable_, label):
    """Run callable, catch OOM/JAX errors, return (success, result, err_msg)."""
    try:
        out = callable_()
        if hasattr(out, 'block_until_ready'):
            out.block_until_ready()
        return True, out, None
    except Exception as exc:
        msg = f'{type(exc).__name__}: {exc}'
        print(f'  [{label}] FAILED: {msg[:200]}')
        return False, None, msg
