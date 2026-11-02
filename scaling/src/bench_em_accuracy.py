#!/usr/bin/env python3
"""EM accuracy is system-dependent: at fixed N, the n_time_steps you
need depends on the spectral content of the random Hamiltonian.

Diagnostic: at N=10, draw several random chains (different RNG seeds,
hence different H_sys / H_ops / jump_ops scales).  For each, sweep
n_time_steps and measure ||rho_EM - rho_Lindblad||_F.

Expected: some seeds converge cleanly at modest n_time_steps; others
need 10-100x more steps to hit the same accuracy floor.

Writes scaling/results/em_accuracy.json.
"""
import os, sys, json, time, gc
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import jax
jax.config.update('jax_enable_x64', False)

from common import (random_chain, random_fourier_coeffs, lindblad_evolve,
                     rho_estimator, frobenius_distance)

T_FINAL = 1.0
N_MODES = 4
PULSE_AMPLITUDE = 1.0
N = 10
N_TRAJS = 256
NTS_LIST = [256, 1024, 4096, 16384, 65536]
SEEDS = [17, 42, 101, 314, 1729]   # five different "systems"

OUT = '<repo-root>/scaling/results/em_accuracy.json'


def hamiltonian_norms(sys_dict):
    """Return spectral-radius-style scalars for H_sys and per-site Lindblad
    rates — useful as a "this system is hard" indicator."""
    H = np.asarray(sys_dict['H_sys'])
    eigs = np.linalg.eigvalsh((H + H.conj().T) * 0.5)
    H_norm = float(np.max(np.abs(eigs)))
    L_total = float(sum(np.linalg.norm(np.asarray(L), 'fro')
                          for L in sys_dict['jump_ops']))
    return H_norm, L_total


def main():
    print(f'EM accuracy diagnostic at N={N}, n_trajs={N_TRAJS}, n_modes={N_MODES}')
    print(f'Seeds: {SEEDS}   nts grid: {NTS_LIST}')

    rows = []
    for seed in SEEDS:
        print(f'\n=== chain seed = {seed} ===')
        sys_dict = random_chain(N, seed=seed)
        a, b = random_fourier_coeffs(sys_dict['n_controls'], n_modes=N_MODES,
                                      amplitude=PULSE_AMPLITUDE, seed=seed + 1)

        H_norm, L_total = hamiltonian_norms(sys_dict)
        print(f'  ||H_sys||_op = {H_norm:.3f}   sum ||L_k||_F = {L_total:.3f}')

        # --- Lindblad ground truth (one per seed) ---
        try:
            t0 = time.perf_counter()
            rho_truth = lindblad_evolve(sys_dict, a, b, T_FINAL)
            t_truth = time.perf_counter() - t0
            print(f'  Lindblad ground truth: {t_truth:.1f}s')
        except Exception as exc:
            print(f'  Lindblad FAILED: {exc}')
            continue

        # --- EM at increasing n_time_steps ---
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
                print(f'  nts={nts:>6}  frob={frob:.5f}   '
                      f'compile={t_compile:5.1f}s  call={t_call:6.2f}s')
                rows.append(dict(seed=seed, N=N, dim=sys_dict['dim'],
                                  n_time_steps=int(nts), n_trajs=N_TRAJS,
                                  H_norm=float(H_norm),
                                  L_total=float(L_total),
                                  frob=float(frob),
                                  t_compile_s=float(t_compile),
                                  t_call_s=float(t_call),
                                  oom=False, error=None))
            except Exception as exc:
                msg = f'{type(exc).__name__}: {str(exc)[:200]}'
                print(f'  nts={nts:>6}  FAILED: {msg}')
                rows.append(dict(seed=seed, N=N, dim=sys_dict['dim'],
                                  n_time_steps=int(nts), n_trajs=N_TRAJS,
                                  H_norm=float(H_norm),
                                  L_total=float(L_total),
                                  frob=float('nan'),
                                  t_compile_s=float('nan'),
                                  t_call_s=float('nan'),
                                  oom=True, error=msg))

            os.makedirs(os.path.dirname(OUT), exist_ok=True)
            with open(OUT, 'w') as f:
                json.dump(rows, f, indent=2)

        del sys_dict, a, b, rho_truth
        gc.collect()
        try:
            jax.clear_caches()
        except Exception:
            pass

    print(f'\nDone. {len(rows)} rows written to {OUT}.')


if __name__ == '__main__':
    main()
