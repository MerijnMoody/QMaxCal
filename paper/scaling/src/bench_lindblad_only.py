#!/usr/bin/env python3
"""Push the Lindblad master-equation solver to large N until it OOMs.

For each N in {6, 8, 10, 12, 13, 14, 15}:
  - Build the random chain
  - Run lindblad_evolve once (no warmup-and-median; we only want to know
    if it succeeds and a rough wall-clock at this N)
  - Record peak GPU memory if available
  - Catch OOM/JAX errors and continue

This finds the N at which the density-matrix (Lindblad) solver runs out of
GPU memory (or where the wall-clock becomes prohibitive).  The
companion claim for the paper is "SSE reaches N=X+1 where the Lindblad solver OOMs."

Writes results to paper/scaling/results/lindblad_only.json.
"""
import os, sys, json, time, gc
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import jax
jax.config.update('jax_enable_x64', False)

from common import random_chain, random_fourier_coeffs, lindblad_evolve

T_FINAL = 1.0
N_MODES = 4
PULSE_AMPLITUDE = 1.0
N_LIST = [6, 8, 10, 12, 13, 14, 15]
OUT = os.environ.get(
    'LINDBLAD_OUT',
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'lindblad_only.json'),
)


def gpu_mem_mib():
    """Return peak GPU memory used so far in MiB, or None if unavailable."""
    try:
        # JAX exposes per-device memory stats on cuda
        dev = jax.devices()[0]
        stats = dev.memory_stats() or {}
        return float(stats.get('peak_bytes_in_use', 0)) / (1024 ** 2)
    except Exception:
        return None


def main():
    print(f'Lindblad OOM scan: N={N_LIST}  (T={T_FINAL}, n_modes={N_MODES})')
    print(f'Device: {jax.devices()}')

    rows = []
    for N in N_LIST:
        dim = 2 ** N
        # Density matrix size in MiB (complex64 = 8 bytes per entry)
        dm_mib = (dim ** 2) * 8 / (1024 ** 2)
        print(f'\n=== N = {N}  (dim = {dim}, ρ matrix ≈ {dm_mib:.1f} MiB) ===')

        sys_dict = random_chain(N, seed=N * 17)
        a, b = random_fourier_coeffs(sys_dict['n_controls'], n_modes=N_MODES,
                                      amplitude=PULSE_AMPLITUDE, seed=N * 17)

        try:
            # Warmup (JIT compile + first run)
            t0 = time.perf_counter()
            rho1 = lindblad_evolve(sys_dict, a, b, T_FINAL)
            t_warmup = time.perf_counter() - t0
            mem_after_warmup = gpu_mem_mib()
            print(f'  warmup: t={t_warmup:.1f}s   '
                  f'peak GPU mem={mem_after_warmup:.0f} MiB' if mem_after_warmup else
                  f'  warmup: t={t_warmup:.1f}s')

            # One timed run (post-compile)
            t0 = time.perf_counter()
            rho2 = lindblad_evolve(sys_dict, a, b, T_FINAL)
            t_run = time.perf_counter() - t0
            mem_after_run = gpu_mem_mib()
            print(f'  run:    t={t_run:.1f}s    '
                  f'peak GPU mem={mem_after_run:.0f} MiB' if mem_after_run else
                  f'  run:    t={t_run:.1f}s')

            rows.append(dict(N=N, dim=dim, dm_mib_theoretical=dm_mib,
                              t_warmup_s=float(t_warmup),
                              t_run_s=float(t_run),
                              gpu_mem_mib=mem_after_run,
                              oom=False, error=None))
        except Exception as exc:
            msg = f'{type(exc).__name__}: {str(exc)[:300]}'
            print(f'  FAILED: {msg}')
            rows.append(dict(N=N, dim=dim, dm_mib_theoretical=dm_mib,
                              t_warmup_s=float('nan'),
                              t_run_s=float('nan'),
                              gpu_mem_mib=gpu_mem_mib(),
                              oom=True, error=msg))
            # Don't try larger N if we OOM; assume it'll only get worse
            print('  Stopping scan after first failure.')
            break

        # Save incrementally + clean up between N values
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        with open(OUT, 'w') as f:
            json.dump(rows, f, indent=2)
        del rho1, rho2, sys_dict, a, b
        gc.collect()
        try:
            jax.clear_caches()
        except Exception:
            pass

    # Final write
    with open(OUT, 'w') as f:
        json.dump(rows, f, indent=2)
    print(f'\nDone. {len(rows)} rows written to {OUT}.')


if __name__ == '__main__':
    main()
