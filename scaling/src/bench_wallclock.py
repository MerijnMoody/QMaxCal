#!/usr/bin/env python3
"""Axis-1: wall-clock vs N for Lindblad (qiskit-dynamics) and SSE.

For each N in the sweep:
  - Build the random chain
  - Time one forward pass of the Lindblad master-equation solver
  - Time one forward pass of the SSE solver at multiple n_trajs values
  - Record trace distance to ground-truth Lindblad

Catches OOM/JAX errors per cell and continues.
Writes results to scaling/results/wallclock.json.

Run with: python -u bench_wallclock.py [--smoke]
  --smoke restricts to N in {4,6} for fast verification (~5 min).
"""
import os, sys, json, argparse, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import jax
jax.config.update('jax_enable_x64', False)

from common import (random_chain, random_fourier_coeffs, lindblad_evolve,
                     sse_evolve, trace_distance, time_after_warmup,
                     safe_eval)

T_FINAL = 1.0
N_MODES = 4
PULSE_AMPLITUDE = 1.0
N_TIME_STEPS_SSE = 2048


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--smoke', action='store_true',
                   help='Smoke test: N in {4,6}, n_trajs in {64,256}')
    p.add_argument('--n-list', type=int, nargs='+', default=None)
    p.add_argument('--n-trajs', type=int, nargs='+', default=None)
    p.add_argument('--out', type=str,
                   default='<repo-root>/scaling/results/wallclock.json')
    return p.parse_args()


def main():
    args = parse_args()
    if args.smoke:
        N_LIST       = [4, 6]
        N_TRAJS_LIST = [64, 256]
    else:
        N_LIST       = args.n_list   or [4, 6, 8, 10, 12, 14]
        N_TRAJS_LIST = args.n_trajs  or [64, 256, 1024]

    print(f'Wall-clock benchmark: N={N_LIST}  n_trajs={N_TRAJS_LIST}')
    print(f'  T={T_FINAL}  n_modes={N_MODES}  n_time_steps_sse={N_TIME_STEPS_SSE}')

    rows = []
    for N in N_LIST:
        print(f'\n=== N = {N}  (dim = {2**N}) ===')
        sys_dict = random_chain(N, seed=N * 17)
        a, b = random_fourier_coeffs(sys_dict['n_controls'], n_modes=N_MODES,
                                      amplitude=PULSE_AMPLITUDE, seed=N * 17)

        # ---- Lindblad master equation ----
        rho_lindblad = None
        try:
            def lindblad_call():
                return lindblad_evolve(sys_dict, a, b, T_FINAL)
            t_lindblad, rho_lindblad = time_after_warmup(lindblad_call, n_warmup=2, n_runs=5)
            print(f'  Lindblad Dopri5 DM:             t = {t_lindblad*1e3:9.1f} ms')
            rows.append(dict(N=N, dim=sys_dict['dim'], method='lindblad',
                              n_trajs=None, t_ms=t_lindblad * 1e3,
                              trace_dist=0.0, oom=False, error=None))
        except Exception as exc:
            msg = f'{type(exc).__name__}: {str(exc)[:200]}'
            print(f'  Lindblad Dopri5 DM:             FAILED  ({msg})')
            rows.append(dict(N=N, dim=sys_dict['dim'], method='lindblad',
                              n_trajs=None, t_ms=float('nan'),
                              trace_dist=float('nan'), oom=True, error=msg))

        # ---- SSE at multiple n_trajs ----
        for n_trajs in N_TRAJS_LIST:
            try:
                def sse_call(_n=n_trajs):
                    return sse_evolve(sys_dict, a, b, T_FINAL,
                                       n_trajs=_n, n_time_steps=N_TIME_STEPS_SSE)
                t_sse, rho_sse = time_after_warmup(sse_call, n_warmup=2, n_runs=5)
                err = (trace_distance(rho_lindblad, rho_sse)
                        if rho_lindblad is not None else float('nan'))
                err_str = f'{err:.4f}' if rho_lindblad is not None else '   --   '
                print(f'  SSE n_trajs={n_trajs:>4}:               '
                      f't = {t_sse*1e3:9.1f} ms   trace_dist = {err_str}')
                rows.append(dict(N=N, dim=sys_dict['dim'], method='sse',
                                  n_trajs=int(n_trajs), t_ms=t_sse * 1e3,
                                  trace_dist=err, oom=False, error=None))
            except Exception as exc:
                msg = f'{type(exc).__name__}: {str(exc)[:200]}'
                print(f'  SSE n_trajs={n_trajs:>4}:               FAILED  ({msg})')
                rows.append(dict(N=N, dim=sys_dict['dim'], method='sse',
                                  n_trajs=int(n_trajs), t_ms=float('nan'),
                                  trace_dist=float('nan'), oom=True, error=msg))

        # Write incrementally so partial progress is preserved on crash
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, 'w') as f:
            json.dump(rows, f, indent=2)
        print(f'  → wrote {len(rows)} rows to {args.out}')

    print(f'\nDone. {len(rows)} rows total.')


if __name__ == '__main__':
    main()
