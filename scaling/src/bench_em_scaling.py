#!/usr/bin/env python3
"""Wall-clock + memory scaling of the SSE solver under
EulerMaruyama (no dense propagator) vs ExponentialSplit (dense expm).

For each N, run one forward pass at fixed n_trajs and n_time_steps,
record compile time, cached-call time, and peak GPU memory.

The expectation: EM should scale as O(dim^2) per step (matrix-vector
on dense H), ExpSplit as O(dim^3) per step (matrix exp).  At large N
the wall-clock gap should grow.
"""
import os, sys, json, time, gc
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import jax
jax.config.update('jax_enable_x64', False)

from common import random_chain, random_fourier_coeffs, rho_estimator

T_FINAL = 1.0
N_MODES = 4
PULSE_AMPLITUDE = 1.0
N_TRAJS = 256
N_TIME_STEPS = 2048

OUT = '/home/mmoody/qmaxcal_paper/scaling/results/em_scaling.json'


def gpu_mem_mib():
    try:
        return float(jax.devices()[0].memory_stats().get('peak_bytes_in_use', 0)) / (1024 ** 2)
    except Exception:
        return None


def time_method(sys_dict, a, b, method, label):
    """Compile + cached-time one forward pass."""
    est_fn  = rho_estimator(sys_dict, a, b, T_FINAL, N_TRAJS, N_TIME_STEPS,
                              method=method)
    est_jit = jax.jit(est_fn)
    t0 = time.perf_counter()
    rho = est_jit(jax.random.PRNGKey(0))
    rho.block_until_ready()
    t_first = time.perf_counter() - t0
    t0 = time.perf_counter()
    rho = est_jit(jax.random.PRNGKey(1))
    rho.block_until_ready()
    t_call = time.perf_counter() - t0
    mem = gpu_mem_mib()
    return t_first, t_call, mem


def main():
    print(f'EM vs ExpSplit scaling.  n_trajs={N_TRAJS}  n_time_steps={N_TIME_STEPS}')
    print(f'Device: {jax.devices()}')

    N_LIST = [4, 6, 8, 10, 12, 14]
    METHODS = ['euler_maruyama', 'exp_split']

    rows = []
    for N in N_LIST:
        dim = 2 ** N
        print(f'\n=== N = {N}  (dim = {dim}) ===')
        sys_dict = random_chain(N, seed=N * 17)
        a, b = random_fourier_coeffs(sys_dict['n_controls'], n_modes=N_MODES,
                                      amplitude=PULSE_AMPLITUDE, seed=N * 17)

        for method in METHODS:
            label = method.replace('_', ' ')
            try:
                t_first, t_call, mem = time_method(sys_dict, a, b, method, label)
                print(f'  {label:>16s}: first(compile+run)={t_first:7.1f}s  '
                      f'cached={t_call:6.1f}s  mem={mem:7.0f} MiB')
                rows.append(dict(N=N, dim=dim, method=method,
                                  n_trajs=N_TRAJS, n_time_steps=N_TIME_STEPS,
                                  t_first_s=float(t_first),
                                  t_cached_s=float(t_call),
                                  gpu_mem_mib=mem,
                                  oom=False, error=None))
                try:
                    jax.clear_caches()
                except Exception:
                    pass
            except Exception as exc:
                msg = f'{type(exc).__name__}: {str(exc)[:300]}'
                print(f'  {label:>16s}: FAILED  ({msg})')
                rows.append(dict(N=N, dim=dim, method=method,
                                  n_trajs=N_TRAJS, n_time_steps=N_TIME_STEPS,
                                  t_first_s=float('nan'),
                                  t_cached_s=float('nan'),
                                  gpu_mem_mib=gpu_mem_mib(),
                                  oom=True, error=msg))

            os.makedirs(os.path.dirname(OUT), exist_ok=True)
            with open(OUT, 'w') as f:
                json.dump(rows, f, indent=2)

        del sys_dict, a, b
        gc.collect()

    print(f'\nDone. {len(rows)} rows written to {OUT}.')


if __name__ == '__main__':
    main()
