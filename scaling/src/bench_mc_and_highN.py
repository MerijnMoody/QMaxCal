#!/usr/bin/env python3
"""Two clean scaling tests in one job (x32 throughout):

1. MC CONVERGENCE at N=10. Fix nts=2048, sweep n_trajs in
   {64, 256, 1024, 4096}, run N_RESEEDS independent estimates per
   cell, report mean + std of ||rho_sse - rho_lindblad||_F.
   Expected behavior: mean/std fall as 1/sqrt(n_trajs) — log-log slope -0.5.

2. HIGH-N SSE SMOKE at N=12, 14, 16. One SSE forward pass each at
   nts=2048, n_trajs=512.  Records compile time, call time, peak GPU
   memory.  No Lindblad comparison (the Lindblad solver OOMs at N>=12 on a 40 GB A100).

Writes scaling/results/mc_and_highN.json.
"""
import os, sys, json, time, gc
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import jax
jax.config.update('jax_enable_x64', False)  # x32 throughout

from common import (random_chain, random_fourier_coeffs, lindblad_evolve,
                     rho_estimator, frobenius_distance)

T_FINAL = 1.0
N_MODES = 4
PULSE_AMPLITUDE = 1.0

OUT = '<repo-root>/scaling/results/mc_and_highN.json'


def gpu_mem_mib():
    try:
        return float(jax.devices()[0].memory_stats().get('peak_bytes_in_use', 0)) / (1024 ** 2)
    except Exception:
        return None


def main():
    print(f'Device: {jax.devices()}')
    rows = []

    # =====================================================================
    # PART 1: MC convergence sweep at N=10
    # =====================================================================
    N = 10
    NTS = 2048
    N_TRAJS_LIST = [64, 256, 1024, 4096]
    N_RESEEDS = 10

    print(f'\n=== MC CONVERGENCE at N={N}  nts={NTS}  reseeds={N_RESEEDS} ===')
    sys_dict = random_chain(N, seed=N * 17)
    a, b = random_fourier_coeffs(sys_dict['n_controls'], n_modes=N_MODES,
                                  amplitude=PULSE_AMPLITUDE, seed=N * 17)

    t0 = time.perf_counter()
    rho_truth = lindblad_evolve(sys_dict, a, b, T_FINAL)
    t_truth = time.perf_counter() - t0
    print(f'  Lindblad ground truth: t={t_truth:.1f}s')

    for n_trajs in N_TRAJS_LIST:
        try:
            est_fn  = rho_estimator(sys_dict, a, b, T_FINAL, n_trajs, NTS)
            est_jit = jax.jit(est_fn)

            t0 = time.perf_counter()
            _ = est_jit(jax.random.PRNGKey(0)).block_until_ready()
            t_compile = time.perf_counter() - t0

            dists = []
            t0 = time.perf_counter()
            for r in range(N_RESEEDS):
                rho_sse = np.asarray(est_jit(jax.random.PRNGKey(r + 1)))
                dists.append(frobenius_distance(rho_sse, rho_truth))
            t_total = time.perf_counter() - t0

            d = np.asarray(dists)
            mean_d = float(d.mean())
            std_d  = float(d.std())
            print(f'  n_trajs={n_trajs:>5}  mean_d={mean_d:.4f}  std_d={std_d:.4f}  '
                  f'compile={t_compile:.1f}s  reseeds={t_total:.1f}s '
                  f'({t_total/N_RESEEDS*1e3:.0f} ms each)')
            rows.append(dict(part='mc', N=N, dim=sys_dict['dim'],
                              n_time_steps=NTS, n_trajs=int(n_trajs),
                              n_reseeds=N_RESEEDS,
                              mean_dist=mean_d, std_dist=std_d,
                              raw_dists=[float(x) for x in dists],
                              t_compile_s=float(t_compile),
                              t_per_call_ms=float(t_total/N_RESEEDS*1e3),
                              oom=False, error=None))
        except Exception as exc:
            msg = f'{type(exc).__name__}: {str(exc)[:200]}'
            print(f'  n_trajs={n_trajs:>5}  FAILED: {msg}')
            rows.append(dict(part='mc', N=N, dim=sys_dict['dim'],
                              n_time_steps=NTS, n_trajs=int(n_trajs),
                              n_reseeds=N_RESEEDS,
                              mean_dist=float('nan'), std_dist=float('nan'),
                              raw_dists=[],
                              t_compile_s=float('nan'),
                              t_per_call_ms=float('nan'),
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

    # =====================================================================
    # PART 2: high-N SSE smoke
    # =====================================================================
    HIGH_N_LIST = [12, 14, 16]
    HIGH_NTS    = 2048
    HIGH_NTRAJS = 512

    print(f'\n=== HIGH-N SSE SMOKE  (x32, nts={HIGH_NTS}, n_trajs={HIGH_NTRAJS}) ===')
    for N in HIGH_N_LIST:
        dim = 2 ** N
        dm_mib = (dim ** 2) * 8 / (1024 ** 2)   # complex64 propagator size
        print(f'\n  N={N}  dim={dim}  '
              f'(propagator at x32 ≈ {dm_mib:.0f} MiB)')

        try:
            sys_dict = random_chain(N, seed=N * 17)
            a, b = random_fourier_coeffs(sys_dict['n_controls'],
                                           n_modes=N_MODES,
                                           amplitude=PULSE_AMPLITUDE,
                                           seed=N * 17)

            est_fn  = rho_estimator(sys_dict, a, b, T_FINAL,
                                     HIGH_NTRAJS, HIGH_NTS)
            est_jit = jax.jit(est_fn)

            t0 = time.perf_counter()
            rho_sse = est_jit(jax.random.PRNGKey(0))
            rho_sse.block_until_ready()
            t_compile_plus_call = time.perf_counter() - t0

            t0 = time.perf_counter()
            rho_sse = est_jit(jax.random.PRNGKey(1))
            rho_sse.block_until_ready()
            t_call = time.perf_counter() - t0
            mem = gpu_mem_mib()

            print(f'    first call (compile+run)={t_compile_plus_call:.1f}s  '
                  f'cached call={t_call:.1f}s  mem={mem:.0f} MiB  ✓')
            rows.append(dict(part='high_N', N=N, dim=dim,
                              n_time_steps=HIGH_NTS, n_trajs=HIGH_NTRAJS,
                              t_first_call_s=float(t_compile_plus_call),
                              t_cached_call_s=float(t_call),
                              gpu_mem_mib=mem,
                              oom=False, error=None))

            del sys_dict, a, b, est_fn, est_jit, rho_sse
            gc.collect()
            try:
                jax.clear_caches()
            except Exception:
                pass

        except Exception as exc:
            msg = f'{type(exc).__name__}: {str(exc)[:300]}'
            print(f'    FAILED: {msg}')
            rows.append(dict(part='high_N', N=N, dim=dim,
                              n_time_steps=HIGH_NTS, n_trajs=HIGH_NTRAJS,
                              t_first_call_s=float('nan'),
                              t_cached_call_s=float('nan'),
                              gpu_mem_mib=gpu_mem_mib(),
                              oom=True, error=msg))
            print('    Stopping high-N scan after first failure.')
            with open(OUT, 'w') as f:
                json.dump(rows, f, indent=2)
            break

        with open(OUT, 'w') as f:
            json.dump(rows, f, indent=2)

    with open(OUT, 'w') as f:
        json.dump(rows, f, indent=2)
    print(f'\nDone. {len(rows)} rows written to {OUT}.')


if __name__ == '__main__':
    main()
