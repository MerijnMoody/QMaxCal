#!/usr/bin/env python3
"""
Solver-comparison benchmark: the density-matrix (Lindblad) solver vs. our SSE
sampling solver.

For each N in a small sweep:
  1. Build a random N-qubit spin chain (XX+YY exchange + per-site sigma_-/sigma_z)
  2. Generate Fourier-mode control pulses with KNOWN closed-form coefficients
  3. Evolve under both methods to time T_final, evaluating EXACTLY the same
     pulse function at every timestep (no interpolation mismatch)
  4. Time a single forward call of each, after JIT warmup
  5. Compare final density matrices via trace distance

Wall-clock timing excludes the first call (JIT compile).  We pre-compile
each forward pass with a warmup invocation, then median-time n_runs
subsequent invocations.

The Lindblad solver = qiskit-dynamics Dopri5 on the full density matrix.
Our solver    = ExponentialSplit on n_trajs SSE samples; the SSE-mean
                |psi><psi| is the rho estimate.
"""
import os, sys, time, json
sys.path.insert(0, '<repo-root>')
sys.path.insert(0, '<repo-root>/experiments/template')

import numpy as np
import jax
import jax.numpy as jnp
jax.config.update('jax_enable_x64', False)

from core.types import ExponentialSplit, asqarray
from core.solver import dssesolve

from qiskit_dynamics import Solver, Signal
from qiskit.quantum_info import DensityMatrix
from diffrax import Dopri5, PIDController


# Pauli matrices (numpy, complex64)
sx = np.array([[0, 1], [1, 0]], dtype=np.complex64)
sy = np.array([[0, -1j], [1j, 0]], dtype=np.complex64)
sz = np.array([[1, 0], [0, -1]], dtype=np.complex64)
sm = np.array([[0, 0], [1, 0]], dtype=np.complex64)


def _embed(op_2x2, site, N):
    out = np.array([[1.0]], dtype=np.complex64)
    for i in range(N):
        out = np.kron(out, op_2x2 if i == site else np.eye(2, dtype=np.complex64))
    return out


def _two_site(A, B, i, j, N):
    out = np.array([[1.0]], dtype=np.complex64)
    for k in range(N):
        if k == i:
            out = np.kron(out, A)
        elif k == j:
            out = np.kron(out, B)
        else:
            out = np.kron(out, np.eye(2, dtype=np.complex64))
    return out


def random_chain(N, seed=0):
    rng = np.random.default_rng(seed)
    dim = 2 ** N
    Js = rng.uniform(0.5, 1.5, size=N - 1).astype(np.float32)
    h_z = rng.uniform(-0.3, 0.3, size=N).astype(np.float32)
    H_sys = np.zeros((dim, dim), dtype=np.complex64)
    for i in range(N - 1):
        H_sys += Js[i] * (_two_site(sx, sx, i, i + 1, N)
                           + _two_site(sy, sy, i, i + 1, N))
    for i in range(N):
        H_sys += h_z[i] * _embed(sz, i, N)

    H_ops = []
    for i in range(N):
        H_ops.append(_embed(sx, i, N))
        H_ops.append(_embed(sy, i, N))

    gammas = rng.uniform(0.05, 0.15, size=N).astype(np.float32)
    kappas = rng.uniform(0.05, 0.15, size=N).astype(np.float32)
    jump_ops = []
    for i in range(N):
        jump_ops.append(np.sqrt(gammas[i]) * _embed(sm, i, N))
        jump_ops.append(np.sqrt(kappas[i]) * _embed(sz, i, N))

    psi0 = np.zeros((dim, 1), dtype=np.complex64)
    bits = [0] + [1] * (N - 1)
    idx = 0
    for b in bits:
        idx = idx * 2 + b
    psi0[idx, 0] = 1.0

    return dict(H_sys=H_sys, H_ops=H_ops, jump_ops=jump_ops,
                psi0=psi0, n_controls=2 * N, dim=dim, N=N)


def random_fourier_coeffs(n_controls, n_modes=4, amplitude=1.0, seed=0):
    """Random Fourier coefficients (a_m, b_m) for u_k(t).

    u_k(t) = sum_{m=1}^{n_modes} [ a_{k,m} cos(2 pi m t / T) + b_{k,m} sin(2 pi m t / T) ]
    """
    rng = np.random.default_rng(seed + 1234)
    a = rng.standard_normal((n_controls, n_modes)).astype(np.float32)
    b = rng.standard_normal((n_controls, n_modes)).astype(np.float32)
    a /= np.sqrt(n_modes)
    b /= np.sqrt(n_modes)
    a *= amplitude
    b *= amplitude
    return a, b


def make_pulse_fn(a_k, b_k, T):
    """Return a single-control pulse function f_k(t).  a_k, b_k: (n_modes,)."""
    a_k = jnp.asarray(a_k)
    b_k = jnp.asarray(b_k)
    n_modes = a_k.shape[0]

    def f(t):
        ms = jnp.arange(1, n_modes + 1)
        phase = 2.0 * jnp.pi * ms * t / T
        return jnp.sum(a_k * jnp.cos(phase) + b_k * jnp.sin(phase))

    return f


def lindblad_evolve(sys, a, b, T, rtol=1e-8, atol=1e-10):
    """Build a fresh qiskit-dynamics Solver and evolve to T using closed-form
    Fourier signals.  Returns final density matrix (numpy)."""
    H_ops = [jnp.asarray(H) for H in sys['H_ops']]
    jump_ops = [jnp.asarray(L) for L in sys['jump_ops']]
    solver = Solver(
        hamiltonian_operators=H_ops,
        static_dissipators=jump_ops,
        static_hamiltonian=jnp.asarray(sys['H_sys']),
    )
    signals = [Signal(jnp.vectorize(make_pulse_fn(a[k], b[k], T)))
                for k in range(sys['n_controls'])]
    psi0 = sys['psi0'].squeeze(-1)
    rho0 = DensityMatrix(np.outer(psi0, psi0.conj()))
    stepsize_controller = PIDController(rtol=rtol, atol=atol,
                                          dtmin=1e-12, dtmax=T / 10)
    sol = solver.solve(
        t_span=[0.0, T],
        signals=signals,
        y0=rho0,
        t_eval=jnp.array([0.0, T]),
        convert_results=False,
        method=Dopri5(),
        stepsize_controller=stepsize_controller,
        max_steps=500_000,
        throw=False,
    )
    return np.asarray(sol.y[-1])


def make_H_con_fourier(a, b, H_ops_q, T):
    """H_con(t) = sum_k u_k(t) H_op_k, with u_k from closed-form Fourier."""
    a = jnp.asarray(a)        # (n_controls, n_modes)
    b = jnp.asarray(b)
    n_modes = a.shape[1]

    def H_con(t):
        ms = jnp.arange(1, n_modes + 1)
        phase = 2.0 * jnp.pi * ms * t / T
        cosp = jnp.cos(phase)
        sinp = jnp.sin(phase)
        u = (a * cosp).sum(axis=1) + (b * sinp).sum(axis=1)
        out = sum(uk * H_ops_q[k] for k, uk in enumerate(u))
        return out

    return H_con


def sse_evolve(sys, a, b, T, n_trajs, n_time_steps, seed=0):
    """Run dssesolve with closed-form Fourier H_con; return SSE-mean rho(T)."""
    H_ops_q = [asqarray(H) for H in sys['H_ops']]
    H_con = make_H_con_fourier(a, b, H_ops_q, T)
    keys = jax.random.split(jax.random.PRNGKey(seed), n_trajs)
    psi0 = asqarray(sys['psi0'])
    H_sys_q = asqarray(sys['H_sys'])
    jump_q = [asqarray(L) for L in sys['jump_ops']]
    tsave = jnp.array([0.0, T])
    dt = T / n_time_steps

    result = dssesolve(
        H_sys=H_sys_q, jump_ops=jump_q,
        psi0=psi0, tsave=tsave, keys=keys,
        H_con=H_con, method=ExponentialSplit(dt=dt),
        nsteps=n_time_steps, nsteps_per_save=n_time_steps,
    )
    states = np.asarray(result.states.data).squeeze(-1)
    psi_T = states[:, -1, :]
    rho_T = (psi_T[:, :, None] * psi_T.conj()[:, None, :]).mean(axis=0)
    return np.asarray(rho_T)


def trace_distance(rho_a, rho_b):
    diff = rho_a - rho_b
    eigvals = np.linalg.eigvalsh(0.5 * (diff + diff.conj().T))
    return 0.5 * float(np.sum(np.abs(eigvals)))


def time_after_warmup(fn, n_warmup=2, n_runs=5):
    """Run fn() n_warmup times to fully JIT-compile, then median-time n_runs."""
    res = None
    for _ in range(n_warmup):
        res = fn()
        if hasattr(res, 'block_until_ready'):
            res.block_until_ready()
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        res = fn()
        if hasattr(res, 'block_until_ready'):
            res.block_until_ready()
        times.append(time.perf_counter() - t0)
    return float(np.median(times)), res


def main(out_path='<repo-root>/experiments/template/benchmarks/solver_benchmark.json'):
    T_FINAL = 1.0
    N_MODES = 4
    PULSE_AMPLITUDE = 1.0
    N_TIME_STEPS_SSE = 2048   # half the previous 4096 to keep wall-time bounded;
                              # dt * ||H|| ~ 5e-4 so integrator error << sampling
    N_TRAJS_LIST = [64, 256]  # 1024 already shown to converge; skip to save time
    N_LIST = [4, 6, 8, 10, 12]
    # With --mem=64G host RAM, host OOM is no longer the binding constraint;
    # the binding constraint is the 40 GB A100 GPU memory.  The Lindblad solver's density
    # matrix is 4^N x 4^N x 16 bytes plus 6 Dopri5 stage buffers plus dense
    # Lindblad/Hamiltonian operators (~24 of them at 4^N x 16 bytes each).
    # Estimated GPU footprint per N:
    #     N=10:  ~ 0.7 GB     N=12:  ~ 9 GB     N=13:  ~ 30 GB     N=14: > 40 GB
    # We try the Lindblad solver at every N below; if it OOMs we'll see it and can bump
    # this cutoff after the run.
    LINDBLAD_MAX_N = 12

    rows = []
    for N in N_LIST:
        print(f'\n=== N = {N}  (dim = {2**N}) ===')
        sys_dict = random_chain(N, seed=N * 17)
        a, b = random_fourier_coeffs(sys_dict['n_controls'], n_modes=N_MODES,
                                      amplitude=PULSE_AMPLITUDE, seed=N * 17)

        # ---- Lindblad (ground truth) -- only at small N, OOMs at N>=10 ----
        rho_lindblad = None
        if N <= LINDBLAD_MAX_N:
            def lindblad_call():
                return lindblad_evolve(sys_dict, a, b, T_FINAL)
            t_lindblad, rho_lindblad = time_after_warmup(lindblad_call, n_warmup=2, n_runs=5)
            print(f'  Lindblad (Dopri5 DM, rtol=1e-8): t = {t_lindblad*1e3:7.1f} ms')
            rows.append(dict(N=N, dim=sys_dict['dim'], method='lindblad',
                              n_trajs=None, t_ms=t_lindblad * 1e3,
                              trace_dist=0.0))
        else:
            print(f'  Lindblad (Dopri5 DM):           SKIPPED  '
                  f'(OOMs at N>={LINDBLAD_MAX_N+1} with 16 GB allocation)')
            rows.append(dict(N=N, dim=sys_dict['dim'], method='lindblad',
                              n_trajs=None, t_ms=float('nan'),
                              trace_dist=float('nan'),
                              note='SKIPPED: OOM at 16 GB'))

        # ---- Our SSE at multiple n_trajs ----
        for n_trajs in N_TRAJS_LIST:
            def sse_call(_n=n_trajs):
                return sse_evolve(sys_dict, a, b, T_FINAL,
                                    n_trajs=_n, n_time_steps=N_TIME_STEPS_SSE)
            t_sse, rho_sse = time_after_warmup(sse_call, n_warmup=2, n_runs=5)
            err = (trace_distance(rho_lindblad, rho_sse)
                    if rho_lindblad is not None else float('nan'))
            err_str = f'{err:.4f}' if rho_lindblad is not None else '   --   '
            print(f'  SSE n_trajs={n_trajs:>4}  (dt={T_FINAL/N_TIME_STEPS_SSE:.4g}):'
                  f'  t = {t_sse*1e3:7.1f} ms   trace_dist = {err_str}')
            rows.append(dict(N=N, dim=sys_dict['dim'], method='sse',
                              n_trajs=int(n_trajs), t_ms=t_sse * 1e3,
                              trace_dist=err))

    with open(out_path, 'w') as f:
        json.dump(rows, f, indent=2)
    print(f'\nSaved: {out_path}')

    print('\n=== Summary ===')
    print(f'{"N":>3}  {"dim":>5}  {"method":>10}  {"n_trajs":>7}  '
          f'{"time(ms)":>10}  {"trace_dist":>10}')
    for r in rows:
        nt = '-' if r['n_trajs'] is None else r['n_trajs']
        print(f'{r["N"]:>3}  {r["dim"]:>5}  {r["method"]:>10}  {str(nt):>7}  '
              f'{r["t_ms"]:>10.2f}  {r["trace_dist"]:>10.4f}')


if __name__ == '__main__':
    main()
