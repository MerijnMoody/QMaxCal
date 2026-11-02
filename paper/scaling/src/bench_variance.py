#!/usr/bin/env python3
"""Axis-2: how reliably does the SSE estimate of rho(T) converge with n_trajs?

For each (N, n_trajs):
  - Build the random chain
  - Compute rho_lindblad = ground-truth Lindblad master-equation rho(T)
  - JIT-compile the SSE rho(T) estimator at this n_trajs
  - For each of N_RESEEDS independent PRNGKeys, run SSE once, get rho_sse
  - Record Frobenius distance ||rho_sse - rho_lindblad||_F per reseed
  - Report mean / std across reseeds

Frobenius distance is O(dim^2), much cheaper than trace distance's
O(dim^3) eigvalsh, so this scales to N=14.

For N where Lindblad OOMs we fall back to a self-consistency check:
take the mean across reseeds as a proxy ground truth and measure each
reseed's distance to that.

Writes results to paper/scaling/results/variance.json.
"""
import os, sys, json, argparse, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import jax
jax.config.update('jax_enable_x64', False)

from common import (random_chain, random_fourier_coeffs, lindblad_evolve,
                     rho_estimator, frobenius_distance, trace_distance)

T_FINAL = 1.0
N_MODES = 4
PULSE_AMPLITUDE = 1.0
N_TIME_STEPS_SSE = 2048


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--smoke', action='store_true',
                   help='Smoke test: N in {4,8}, n_trajs in {64,256}, 10 reseeds')
    p.add_argument('--n-list', type=int, nargs='+', default=None)
    p.add_argument('--n-trajs', type=int, nargs='+', default=None)
    p.add_argument('--n-reseeds', type=int, default=None)
    p.add_argument('--out', type=str,
                   default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'variance.json'))
    return p.parse_args()


def main():
    args = parse_args()
    if args.smoke:
        N_LIST       = [4, 8]
        N_TRAJS_LIST = [64, 256]
        N_RESEEDS    = 10
    else:
        N_LIST       = args.n_list    or [4, 8, 10, 12, 14]
        N_TRAJS_LIST = args.n_trajs   or [16, 64, 256, 1024]
        N_RESEEDS    = args.n_reseeds or 30

    print(f'Variance benchmark: N={N_LIST}  n_trajs={N_TRAJS_LIST}  '
          f'reseeds={N_RESEEDS}')

    rows = []
    for N in N_LIST:
        print(f'\n=== N = {N}  (dim = {2**N}) ===')
        sys_dict = random_chain(N, seed=N * 17)
        a, b = random_fourier_coeffs(sys_dict['n_controls'], n_modes=N_MODES,
                                      amplitude=PULSE_AMPLITUDE, seed=N * 17)

        # ---- Ground truth (Lindblad), if feasible ----
        rho_truth = None
        try:
            t0 = time.perf_counter()
            rho_truth = lindblad_evolve(sys_dict, a, b, T_FINAL)
            t_truth = time.perf_counter() - t0
            print(f'  Lindblad ground truth: t={t_truth:.1f}s  '
                  f'shape={rho_truth.shape}')
        except Exception as exc:
            msg = f'{type(exc).__name__}: {str(exc)[:200]}'
            print(f'  Lindblad ground truth FAILED ({msg}) — using SSE-mean as proxy')

        for n_trajs in N_TRAJS_LIST:
            try:
                est_fn  = rho_estimator(sys_dict, a, b, T_FINAL,
                                         n_trajs, N_TIME_STEPS_SSE)
                est_jit = jax.jit(est_fn)

                # Warmup (JIT compile)
                t0 = time.perf_counter()
                _ = est_jit(jax.random.PRNGKey(0)).block_until_ready()
                t_compile = time.perf_counter() - t0

                # Reseeds
                rhos = []
                t0 = time.perf_counter()
                for r in range(N_RESEEDS):
                    rho_sse = np.asarray(est_jit(jax.random.PRNGKey(r + 1)))
                    rhos.append(rho_sse)
                t_total = time.perf_counter() - t0

                # Distance to ground truth (or to mean if no ground truth)
                if rho_truth is not None:
                    dists = [frobenius_distance(rho, rho_truth) for rho in rhos]
                    metric = 'frob_to_lindblad'
                else:
                    rho_mean = np.mean(rhos, axis=0)
                    dists = [frobenius_distance(rho, rho_mean) for rho in rhos]
                    metric = 'frob_to_sse_mean'

                dists_arr = np.asarray(dists)
                mean_d = float(dists_arr.mean())
                std_d  = float(dists_arr.std())
                print(f'  n_trajs={n_trajs:>4}  mean_d={mean_d:.4f}  '
                      f'std_d={std_d:.4f}  ({metric})  '
                      f'compile={t_compile:.1f}s  reseeds={t_total:.1f}s '
                      f'({t_total/N_RESEEDS*1e3:.0f} ms each)')
                rows.append(dict(N=N, dim=sys_dict['dim'], n_trajs=int(n_trajs),
                                  n_reseeds=N_RESEEDS,
                                  metric=metric,
                                  mean_dist=mean_d, std_dist=std_d,
                                  raw_dists=[float(d) for d in dists],
                                  t_compile_s=float(t_compile),
                                  t_per_call_ms=float(t_total/N_RESEEDS*1e3),
                                  oom=False, error=None))
            except Exception as exc:
                msg = f'{type(exc).__name__}: {str(exc)[:200]}'
                print(f'  n_trajs={n_trajs:>4}  FAILED  ({msg})')
                rows.append(dict(N=N, dim=sys_dict['dim'], n_trajs=int(n_trajs),
                                  n_reseeds=N_RESEEDS,
                                  metric='failed',
                                  mean_dist=float('nan'), std_dist=float('nan'),
                                  raw_dists=[], t_compile_s=float('nan'),
                                  t_per_call_ms=float('nan'),
                                  oom=True, error=msg))

            # Write incrementally
            os.makedirs(os.path.dirname(args.out), exist_ok=True)
            with open(args.out, 'w') as f:
                json.dump(rows, f, indent=2)

    print(f'\nDone. {len(rows)} rows total.')


if __name__ == '__main__':
    main()
