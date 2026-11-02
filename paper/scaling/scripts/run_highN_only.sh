#!/bin/bash
#SBATCH --job-name=scaling_highN
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=01:30:00
#SBATCH --output=logs/highN_%j.out
#SBATCH --error=logs/highN_%j.out

# High-N SSE smoke only.  Skip the MC-convergence part (already done in
# 22508137 Part 1).  --mem=128G is needed because JIT lowering captures
# ~7 GB of dense Hamiltonian/Lindblad constants at N=12.

PYTHON=~/qmaxcal_venv/bin/python
# Submit from the repository root: sbatch paper/scaling/scripts/run_highN_only.sh
cd "${SLURM_SUBMIT_DIR:-$PWD}"
[ -f train_qmaxcal.py ] || { echo "Submit this script from the repository root." >&2; exit 1; }

LINDBLAD_OUT=paper/scaling/results/highN_only.json $PYTHON -u - <<'EOF'
import os, sys, json, time, gc
sys.path.insert(0, 'paper/scaling/src')

import numpy as np
import jax
jax.config.update('jax_enable_x64', False)

from common import random_chain, random_fourier_coeffs, rho_estimator

T_FINAL = 1.0
N_MODES = 4
PULSE_AMPLITUDE = 1.0

HIGH_N_LIST = [12, 14, 16]
HIGH_NTS    = 2048
HIGH_NTRAJS = 512

OUT = os.environ.get('LINDBLAD_OUT', 'paper/scaling/results/highN_only.json')


def gpu_mem_mib():
    try:
        return float(jax.devices()[0].memory_stats().get('peak_bytes_in_use', 0)) / (1024 ** 2)
    except Exception:
        return None


print(f'Device: {jax.devices()}')
rows = []
for N in HIGH_N_LIST:
    dim = 2 ** N
    dm_mib = (dim ** 2) * 8 / (1024 ** 2)
    print(f'\n=== N={N}  dim={dim}  (propagator x32 ≈ {dm_mib:.0f} MiB) ===')

    try:
        sys_dict = random_chain(N, seed=N * 17)
        a, b = random_fourier_coeffs(sys_dict['n_controls'], n_modes=N_MODES,
                                      amplitude=PULSE_AMPLITUDE, seed=N * 17)

        est_fn  = rho_estimator(sys_dict, a, b, T_FINAL, HIGH_NTRAJS, HIGH_NTS)
        est_jit = jax.jit(est_fn)

        t0 = time.perf_counter()
        rho_sse = est_jit(jax.random.PRNGKey(0))
        rho_sse.block_until_ready()
        t_first = time.perf_counter() - t0

        t0 = time.perf_counter()
        rho_sse = est_jit(jax.random.PRNGKey(1))
        rho_sse.block_until_ready()
        t_call = time.perf_counter() - t0
        mem = gpu_mem_mib()

        print(f'  first call (compile+run)={t_first:.1f}s  '
              f'cached call={t_call:.1f}s  mem={mem:.0f} MiB  ✓')
        rows.append(dict(N=N, dim=dim, n_time_steps=HIGH_NTS,
                          n_trajs=HIGH_NTRAJS,
                          t_first_call_s=float(t_first),
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
        print(f'  FAILED: {msg}')
        rows.append(dict(N=N, dim=dim, n_time_steps=HIGH_NTS,
                          n_trajs=HIGH_NTRAJS,
                          t_first_call_s=float('nan'),
                          t_cached_call_s=float('nan'),
                          gpu_mem_mib=gpu_mem_mib(),
                          oom=True, error=msg))
        with open(OUT, 'w') as f:
            json.dump(rows, f, indent=2)
        print('  Stopping after first failure.')
        break

    with open(OUT, 'w') as f:
        json.dump(rows, f, indent=2)

with open(OUT, 'w') as f:
    json.dump(rows, f, indent=2)
print(f'\nDone. {len(rows)} rows written to {OUT}.')
EOF

echo "done $(date)"
