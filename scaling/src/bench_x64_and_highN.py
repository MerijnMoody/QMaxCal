#!/usr/bin/env python3
"""Two diagnostics in one job:

1. PRECISION CHECK at N=10. Re-run the convergence sweep with
   jax_enable_x64=True (i.e. complex128 throughout) so SSE matches
   the Lindblad solver's internal precision. If bias collapses to ~1e-3 or below, the
   floor we saw in the float32 sweep was complex64 round-off; if it
   stays at ~0.04, there's a structural mismatch.

2. HIGH-N SSE SMOKE at N=12, 14 (and 16 if memory permits). One SSE
   forward pass each, no Lindblad comparison (the Lindblad solver OOMs above N=11
   on a 40 GB A100). Records wall-clock and peak GPU memory to confirm
   SSE actually runs where Lindblad doesn't.

Writes scaling/results/x64_and_highN.json.
"""
import os, sys, json, time, gc
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# x64 must be set BEFORE importing jax-using modules
import jax
jax.config.update('jax_enable_x64', True)

import numpy as np
import jax.numpy as jnp

from common import (random_chain, random_fourier_coeffs, lindblad_evolve,
                     rho_estimator, frobenius_distance)

T_FINAL = 1.0
N_MODES = 4
PULSE_AMPLITUDE = 1.0

OUT = '<repo-root>/scaling/results/x64_and_highN.json'


def gpu_mem_mib():
    try:
        return float(jax.devices()[0].memory_stats().get('peak_bytes_in_use', 0)) / (1024 ** 2)
    except Exception:
        return None


def upcast(sys_dict):
    """Upcast a chain dict's arrays from complex64 to complex128 for x64 runs."""
    out = dict(sys_dict)
    out['H_sys']    = np.asarray(sys_dict['H_sys'],    dtype=np.complex128)
    out['H_ops']    = [np.asarray(H, dtype=np.complex128) for H in sys_dict['H_ops']]
    out['jump_ops'] = [np.asarray(L, dtype=np.complex128) for L in sys_dict['jump_ops']]
    out['psi0']     = np.asarray(sys_dict['psi0'],     dtype=np.complex128)
    return out


def main():
    print(f'jax x64 enabled: {jax.config.read("jax_enable_x64")}')
    print(f'Device: {jax.devices()}')

    rows = []

    # =====================================================================
    # PART 1: precision check at N=10
    # =====================================================================
    N = 10
    NTS_LIST = [1024, 2048, 4096, 8192]
    N_TRAJS = 512

    print(f'\n=== PRECISION CHECK at N={N} (x64 enabled) ===')
    sys_dict = upcast(random_chain(N, seed=N * 17))
    a, b = random_fourier_coeffs(sys_dict['n_controls'], n_modes=N_MODES,
                                  amplitude=PULSE_AMPLITUDE, seed=N * 17)
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    try:
        t0 = time.perf_counter()
        rho_truth = lindblad_evolve(sys_dict, a, b, T_FINAL)
        t_truth = time.perf_counter() - t0
        print(f'  Lindblad ground truth: t={t_truth:.1f}s  shape={rho_truth.shape}  '
              f'dtype={rho_truth.dtype}')
    except Exception as exc:
        msg = f'{type(exc).__name__}: {str(exc)[:200]}'
        print(f'  Lindblad ground truth FAILED: {msg}')
        rho_truth = None

    for nts in NTS_LIST:
        try:
            est_fn  = rho_estimator(sys_dict, a, b, T_FINAL, N_TRAJS, nts)
            est_jit = jax.jit(est_fn)

            t0 = time.perf_counter()
            _ = est_jit(jax.random.PRNGKey(0)).block_until_ready()
            t_compile = time.perf_counter() - t0

            t0 = time.perf_counter()
            rho_sse = np.asarray(est_jit(jax.random.PRNGKey(1)))
            t_call = time.perf_counter() - t0

            bias = (frobenius_distance(rho_sse, rho_truth)
                     if rho_truth is not None else float('nan'))
            mem = gpu_mem_mib()
            print(f'  N={N}  nts={nts:>5}  bias={bias:.4f}  '
                  f'compile={t_compile:.1f}s  call={t_call:.1f}s  '
                  f'mem={mem:.0f} MiB')
            rows.append(dict(part='precision', N=N, dim=sys_dict['dim'],
                              n_time_steps=int(nts), n_trajs=N_TRAJS,
                              x64=True, dtype='complex128',
                              bias=float(bias),
                              t_compile_s=float(t_compile),
                              t_call_s=float(t_call),
                              gpu_mem_mib=mem,
                              oom=False, error=None))
        except Exception as exc:
            msg = f'{type(exc).__name__}: {str(exc)[:200]}'
            print(f'  N={N}  nts={nts:>5}  FAILED: {msg}')
            rows.append(dict(part='precision', N=N, dim=sys_dict['dim'],
                              n_time_steps=int(nts), n_trajs=N_TRAJS,
                              x64=True, dtype='complex128',
                              bias=float('nan'),
                              t_compile_s=float('nan'),
                              t_call_s=float('nan'),
                              gpu_mem_mib=gpu_mem_mib(),
                              oom=True, error=msg))

        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        with open(OUT, 'w') as f:
            json.dump(rows, f, indent=2)

    # Free up before the high-N runs
    del sys_dict, a, b, rho_truth
    gc.collect()
    try:
        jax.clear_caches()
    except Exception:
        pass

    # =====================================================================
    # PART 2: high-N SSE smoke (no Lindblad comparison)
    # =====================================================================
    HIGH_N_LIST = [12, 14, 16]
    HIGH_NTS    = 2048
    HIGH_NTRAJS = 512

    print(f'\n=== HIGH-N SSE SMOKE  (x64, nts={HIGH_NTS}, n_trajs={HIGH_NTRAJS}) ===')
    for N in HIGH_N_LIST:
        dim = 2 ** N
        dm_mib_x64 = (dim ** 2) * 16 / (1024 ** 2)
        print(f'\n  N={N}  dim={dim}  '
              f'(propagator at x64 ≈ {dm_mib_x64:.0f} MiB)')

        try:
            sys_dict_h = upcast(random_chain(N, seed=N * 17))
            a_h, b_h = random_fourier_coeffs(sys_dict_h['n_controls'],
                                               n_modes=N_MODES,
                                               amplitude=PULSE_AMPLITUDE,
                                               seed=N * 17)
            a_h = np.asarray(a_h, dtype=np.float64)
            b_h = np.asarray(b_h, dtype=np.float64)

            est_fn  = rho_estimator(sys_dict_h, a_h, b_h, T_FINAL,
                                     HIGH_NTRAJS, HIGH_NTS)
            est_jit = jax.jit(est_fn)

            t0 = time.perf_counter()
            _ = est_jit(jax.random.PRNGKey(0)).block_until_ready()
            t_compile = time.perf_counter() - t0

            t0 = time.perf_counter()
            rho_sse = est_jit(jax.random.PRNGKey(1))
            rho_sse.block_until_ready()
            t_call = time.perf_counter() - t0
            mem = gpu_mem_mib()

            print(f'    compile={t_compile:.1f}s  call={t_call:.1f}s  '
                  f'mem={mem:.0f} MiB  ✓')
            rows.append(dict(part='high_N', N=N, dim=dim,
                              n_time_steps=HIGH_NTS, n_trajs=HIGH_NTRAJS,
                              x64=True, dtype='complex128',
                              bias=None,
                              t_compile_s=float(t_compile),
                              t_call_s=float(t_call),
                              gpu_mem_mib=mem,
                              oom=False, error=None))

            del sys_dict_h, a_h, b_h, est_fn, est_jit, rho_sse
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
                              x64=True, dtype='complex128',
                              bias=None,
                              t_compile_s=float('nan'),
                              t_call_s=float('nan'),
                              gpu_mem_mib=gpu_mem_mib(),
                              oom=True, error=msg))
            print('    Stopping high-N scan after first failure.')
            break

        with open(OUT, 'w') as f:
            json.dump(rows, f, indent=2)

    with open(OUT, 'w') as f:
        json.dump(rows, f, indent=2)
    print(f'\nDone. {len(rows)} rows written to {OUT}.')


if __name__ == '__main__':
    main()
