#!/usr/bin/env python3
"""How fine does the SSE time discretization need to be to match Lindblad
ground truth at each N?

For each N (where Lindblad is feasible, i.e. N <= 10):
  1. Compute rho_lindblad (ground truth, deterministic)
  2. For each n_time_steps in a grid:
       - Run SSE with N_RESEEDS independent keys at fixed n_trajs
       - Compute rho_sse_avg = mean across reseeds
                            (= n_trajs * N_RESEEDS effective samples)
       - bias       = ||rho_sse_avg - rho_lindblad||_F
       - mc_noise   = std across reseeds of ||rho_sse_seed - rho_sse_avg||_F
       - Wall-clock per call

The 'bias' isolates the integrator discretization error: it stays
non-zero even as n_trajs * N_RESEEDS -> infinity, but should fall
roughly as 1/n_time_steps^p (p=2 for ExponentialSplit Hamiltonian
part, p=1 for the stochastic Euler step) as n_time_steps -> infinity.

Output: paper/scaling/results/convergence.json — rows
  (N, n_time_steps, n_trajs, n_reseeds, bias, mc_noise, t_call_ms)

Run: python -u bench_convergence.py [--smoke]
"""
import os, sys, json, argparse, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import jax
jax.config.update('jax_enable_x64', False)

from common import (random_chain, random_fourier_coeffs, lindblad_evolve,
                     rho_estimator, frobenius_distance)

T_FINAL = 1.0
N_MODES = 4
PULSE_AMPLITUDE = 1.0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--smoke', action='store_true',
                   help='Smoke: N in {4,6}, n_time_steps in {1024,2048,4096}, 5 reseeds')
    p.add_argument('--n-list', type=int, nargs='+', default=None)
    p.add_argument('--nts-list', type=int, nargs='+', default=None,
                   help='Time-step counts to sweep')
    p.add_argument('--n-trajs', type=int, default=None)
    p.add_argument('--n-reseeds', type=int, default=None)
    p.add_argument('--out', type=str,
                   default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'convergence.json'))
    return p.parse_args()


def main():
    args = parse_args()
    if args.smoke:
        N_LIST     = [4, 6]
        NTS_LIST   = [1024, 2048, 4096]
        N_TRAJS    = 256
        N_RESEEDS  = 5
    else:
        N_LIST     = args.n_list    or [4, 6, 8, 10]
        NTS_LIST   = args.nts_list  or [1024, 2048, 4096, 8192, 16384]
        N_TRAJS    = args.n_trajs   or 256
        N_RESEEDS  = args.n_reseeds or 5

    print(f'Convergence benchmark: N={N_LIST}  '
          f'n_time_steps={NTS_LIST}  n_trajs={N_TRAJS}  reseeds={N_RESEEDS}')

    rows = []
    for N in N_LIST:
        print(f'\n=== N = {N}  (dim = {2**N}) ===')
        sys_dict = random_chain(N, seed=N * 17)
        a, b = random_fourier_coeffs(sys_dict['n_controls'], n_modes=N_MODES,
                                      amplitude=PULSE_AMPLITUDE, seed=N * 17)

        # --- Lindblad ground truth ---
        try:
            t0 = time.perf_counter()
            rho_truth = lindblad_evolve(sys_dict, a, b, T_FINAL)
            t_truth = time.perf_counter() - t0
            print(f'  Lindblad ground truth: t={t_truth:.1f}s')
        except Exception as exc:
            msg = f'{type(exc).__name__}: {str(exc)[:200]}'
            print(f'  Lindblad ground truth FAILED ({msg}) — skipping this N')
            rows.append(dict(N=N, dim=sys_dict['dim'], n_time_steps=None,
                              n_trajs=N_TRAJS, n_reseeds=N_RESEEDS,
                              bias=float('nan'), mc_noise=float('nan'),
                              t_call_ms=float('nan'),
                              oom=True, error=msg))
            continue

        # --- Convergence sweep over n_time_steps ---
        for nts in NTS_LIST:
            try:
                est_fn  = rho_estimator(sys_dict, a, b, T_FINAL, N_TRAJS, nts)
                est_jit = jax.jit(est_fn)

                t0 = time.perf_counter()
                _ = est_jit(jax.random.PRNGKey(0)).block_until_ready()
                t_compile = time.perf_counter() - t0

                rhos = []
                t0 = time.perf_counter()
                for r in range(N_RESEEDS):
                    rho_sse = np.asarray(est_jit(jax.random.PRNGKey(r + 1)))
                    rhos.append(rho_sse)
                t_total = time.perf_counter() - t0

                rho_avg  = np.mean(rhos, axis=0)
                bias     = frobenius_distance(rho_avg, rho_truth)
                # MC noise: per-reseed deviation from the across-reseed mean
                per_seed_dists = [frobenius_distance(rho, rho_avg) for rho in rhos]
                mc_noise = float(np.std(per_seed_dists))

                t_call_ms = float(t_total / N_RESEEDS * 1e3)
                print(f'  n_time_steps={nts:>5}  bias={bias:.4f}  '
                      f'mc_noise={mc_noise:.4f}  '
                      f'compile={t_compile:5.1f}s  per_call={t_call_ms:6.0f} ms')
                rows.append(dict(N=N, dim=sys_dict['dim'], n_time_steps=int(nts),
                                  n_trajs=N_TRAJS, n_reseeds=N_RESEEDS,
                                  bias=float(bias),
                                  mc_noise=float(mc_noise),
                                  per_seed_dists=[float(d) for d in per_seed_dists],
                                  t_compile_s=float(t_compile),
                                  t_call_ms=t_call_ms,
                                  oom=False, error=None))
            except Exception as exc:
                msg = f'{type(exc).__name__}: {str(exc)[:200]}'
                print(f'  n_time_steps={nts:>5}  FAILED  ({msg})')
                rows.append(dict(N=N, dim=sys_dict['dim'], n_time_steps=int(nts),
                                  n_trajs=N_TRAJS, n_reseeds=N_RESEEDS,
                                  bias=float('nan'), mc_noise=float('nan'),
                                  per_seed_dists=[],
                                  t_compile_s=float('nan'),
                                  t_call_ms=float('nan'),
                                  oom=True, error=msg))

            os.makedirs(os.path.dirname(args.out), exist_ok=True)
            with open(args.out, 'w') as f:
                json.dump(rows, f, indent=2)

    print(f'\nDone. {len(rows)} rows written to {args.out}.')


if __name__ == '__main__':
    main()
