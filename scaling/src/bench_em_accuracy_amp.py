#!/usr/bin/env python3
"""Show where EM fails: vary the pulse amplitude (effectively ||H_eff||)
at fixed N=10 and fixed n_trajs, sweep n_time_steps.

EM's local error scales as O(dt * ||H_eff||^2 * T), so a ~k-fold
amplitude increase requires k^2 times more time steps to hit the
same accuracy.  At amplitude=10 with too-coarse dt the EM solution
should diverge from the Lindblad master equation by O(0.1) or more
in Frobenius distance — well above the Monte Carlo noise floor.

Higher n_trajs (1024) chosen so the MC floor (~1/sqrt(1024) = 0.031)
sits below the bias contribution we want to expose.

Writes scaling/results/em_accuracy_amp.json.
"""
import os, sys, json, time, gc
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import jax
jax.config.update('jax_enable_x64', False)

from common import (random_chain, random_fourier_coeffs, ernst_evolve,
                     rho_estimator, frobenius_distance)

T_FINAL = 1.0
N_MODES = 4
N = 10
N_TRAJS = 1024
SEED = 17
NTS_LIST = [256, 1024, 4096, 16384, 65536]
AMPLITUDES = [1.0, 3.0, 10.0]

OUT = '<repo-root>/scaling/results/em_accuracy_amp.json'


def hamiltonian_norm(sys_dict, a, b, n_modes, T):
    """Worst-case operator norm of H_sys + sum_k g_k(t) H_op_k taken at
    t = 0 (a representative slice). Cheap proxy for "how stiff"."""
    H = np.asarray(sys_dict['H_sys']).astype(complex)
    # Evaluate H_con at t=0:  g_k(0) = sum_m a_{km} cos(0) + b_{km} sin(0) = sum_m a_{km}
    g0 = a.sum(axis=1)
    for k, gk in enumerate(g0):
        H = H + gk * np.asarray(sys_dict['H_ops'][k])
    eigs = np.linalg.eigvalsh((H + H.conj().T) * 0.5)
    return float(np.max(np.abs(eigs)))


def main():
    print(f'EM accuracy vs pulse amplitude. N={N}, seed={SEED}, '
          f'n_trajs={N_TRAJS}')
    print(f'amplitudes={AMPLITUDES}   nts={NTS_LIST}')

    sys_dict = random_chain(N, seed=SEED)

    rows = []
    for amp in AMPLITUDES:
        print(f'\n=== amplitude = {amp} ===')
        a, b = random_fourier_coeffs(sys_dict['n_controls'],
                                       n_modes=N_MODES,
                                       amplitude=amp, seed=SEED + 1)
        h_norm = hamiltonian_norm(sys_dict, a, b, N_MODES, T_FINAL)
        print(f'  ||H_sys + H_con(0)||_op ≈ {h_norm:.2f}')

        # Lindblad ground truth (one per amplitude)
        try:
            t0 = time.perf_counter()
            rho_truth = ernst_evolve(sys_dict, a, b, T_FINAL)
            t_truth = time.perf_counter() - t0
            print(f'  Lindblad ground truth: {t_truth:.1f}s')
        except Exception as exc:
            print(f'  Lindblad FAILED: {exc}')
            continue

        for nts in NTS_LIST:
            try:
                est_fn  = rho_estimator(sys_dict, a, b, T_FINAL,
                                          N_TRAJS, nts,
                                          method='euler_maruyama')
                est_jit = jax.jit(est_fn)
                t0 = time.perf_counter()
                _ = est_jit(jax.random.PRNGKey(0)).block_until_ready()
                t_compile = time.perf_counter() - t0
                t0 = time.perf_counter()
                rho = est_jit(jax.random.PRNGKey(1))
                rho.block_until_ready()
                t_call = time.perf_counter() - t0

                rho_np = np.asarray(rho)
                frob = frobenius_distance(rho_np, rho_truth)
                # NaN check: at very small dt and very large amp, EM can
                # produce numerically diverged states
                diverged = bool(np.isnan(rho_np).any() or
                                 np.isinf(rho_np).any())
                tag = ' DIVERGED' if diverged else ''
                print(f'  nts={nts:>6}  frob={frob:.4f}{tag}  '
                      f'compile={t_compile:5.1f}s  call={t_call:6.2f}s')
                rows.append(dict(amplitude=float(amp), seed=SEED, N=N,
                                  dim=sys_dict['dim'],
                                  H_norm=float(h_norm),
                                  n_time_steps=int(nts),
                                  n_trajs=N_TRAJS,
                                  frob=float(frob),
                                  diverged=diverged,
                                  t_compile_s=float(t_compile),
                                  t_call_s=float(t_call),
                                  oom=False, error=None))
            except Exception as exc:
                msg = f'{type(exc).__name__}: {str(exc)[:200]}'
                print(f'  nts={nts:>6}  FAILED: {msg}')
                rows.append(dict(amplitude=float(amp), seed=SEED, N=N,
                                  dim=sys_dict['dim'],
                                  H_norm=float(h_norm),
                                  n_time_steps=int(nts),
                                  n_trajs=N_TRAJS,
                                  frob=float('nan'),
                                  diverged=False,
                                  t_compile_s=float('nan'),
                                  t_call_s=float('nan'),
                                  oom=True, error=msg))

            os.makedirs(os.path.dirname(OUT), exist_ok=True)
            with open(OUT, 'w') as f:
                json.dump(rows, f, indent=2)

        del rho_truth
        gc.collect()
        try:
            jax.clear_caches()
        except Exception:
            pass

    print(f'\nDone. {len(rows)} rows written to {OUT}.')


if __name__ == '__main__':
    main()
