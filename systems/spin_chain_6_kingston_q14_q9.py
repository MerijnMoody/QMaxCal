"""
6-qubit chain on IBM Kingston, qubits 14-13-12-11-10-9.
Goal: transfer from q14 (site 0) to q9 (site 5).

Calibration snapshot 2026-05-03: noisy bottleneck at q10 (site 4),
T2~17us, located ONE HOP BEFORE the target.  This forces the optimizer
to pass excitation through a strongly-decohering site to reach q9, so
DV's "park on quiet, dash through noisy" strategy applies on the
last-mile leg rather than the first-leg evacuation seen in q10-q15.

Per-site rates (us^-1) from raw T1, T2:

    site i :  qubit  T1 (us)  T2 (us)   gamma     kappa     gamma+kappa
       0   :    14    147.5   191.2    0.00678   0.00092    0.00770
       1   :    13    402.9   352.0    0.00248   0.00080    0.00328   <- quietest
       2   :    12    355.1   272.8    0.00282   0.00113    0.00394
       3   :    11    402.7    93.3    0.00248   0.00474    0.00722
       4   :    10    346.5    17.0    0.00289   0.02873    0.03162   <- noisy bottleneck
       5   :     9    137.8    61.8    0.00725   0.00628    0.01353

Asymmetry: (gamma+kappa)_max / (gamma+kappa)_min = 0.0316 / 0.0033 ~ 9.6x.
"""
import numpy as np
import jax.numpy as jnp


N = 6

# IBM Kingston, qubits 14-13-12-11-10-9 (snapshot 2026-05-03)
KING_T1_US = (147.497, 402.850, 355.054, 402.659, 346.472, 137.849)
KING_T2_US = (191.161, 351.986, 272.768,  93.323,  16.982,  61.777)


def _kron_list(ops):
    out = ops[0]
    for op in ops[1:]:
        out = jnp.kron(out, op)
    return out


def _single_site(op_2x2, site, n_sites=N):
    I2 = jnp.eye(2, dtype=jnp.complex64)
    ops = [I2] * n_sites
    ops[site] = jnp.asarray(op_2x2, dtype=jnp.complex64)
    return _kron_list(ops)


def _two_site(A, B, i, j, n_sites=N):
    I2 = jnp.eye(2, dtype=jnp.complex64)
    ops = [I2] * n_sites
    ops[i] = jnp.asarray(A, dtype=jnp.complex64)
    ops[j] = jnp.asarray(B, dtype=jnp.complex64)
    return _kron_list(ops)


sx = jnp.array([[0, 1], [1, 0]], dtype=jnp.complex64)
sy = jnp.array([[0, -1j], [1j, 0]], dtype=jnp.complex64)
sz = jnp.array([[1, 0], [0, -1]], dtype=jnp.complex64)
sm = jnp.array([[0, 0], [1, 0]], dtype=jnp.complex64)


def _rates(T1_us, T2_us):
    T1 = np.asarray(T1_us, dtype=np.float64)
    T2 = np.asarray(T2_us, dtype=np.float64)
    inv_Tphi = np.clip(1.0 / T2 - 1.0 / (2.0 * T1), 0.0, None)
    return 1.0 / T1, 0.5 * inv_Tphi


def create_system(T_final: float = 30.0,
                  T1_us=KING_T1_US, T2_us=KING_T2_US,
                  signal_scale: float = 3.0,
                  **_):
    """Old-style: 5 tunable XX+YY bond couplings as the only controls."""
    dim = 2 ** N
    gamma, kappa = _rates(T1_us, T2_us)

    H_sys = jnp.zeros((dim, dim), dtype=jnp.complex64)

    H_ops = []
    control_labels = []
    for i in range(N - 1):
        H_xx = _two_site(sx, sx, i, i + 1)
        H_yy = _two_site(sy, sy, i, i + 1)
        H_ops.append(0.5 * (H_xx + H_yy))
        control_labels.append(f'g_{i}{i+1}')

    jump_ops = []
    for i in range(N):
        jump_ops.append(jnp.sqrt(jnp.array(float(gamma[i]), dtype=jnp.float32))
                         * _single_site(sm, i))
        jump_ops.append(jnp.sqrt(jnp.array(float(kappa[i]), dtype=jnp.float32))
                         * _single_site(sz, i))

    def basis(bits):
        idx = 0
        for b in bits:
            idx = idx * 2 + b
        v = jnp.zeros((dim, 1), dtype=jnp.complex64)
        return v.at[idx, 0].set(1.0)

    psi0   = basis([0, 1, 1, 1, 1, 1])      # |100000> = q14 excited
    target = basis([1, 1, 1, 1, 1, 0])      # |000001> = q9  excited

    populations = {}
    for i in range(N):
        proj = 0.5 * (jnp.eye(dim, dtype=jnp.complex64) + _single_site(sz, i))
        populations[f'P(site {i} ↑)'] = proj
    populations['⟨n_tot⟩'] = sum(
        0.5 * (jnp.eye(dim, dtype=jnp.complex64) + _single_site(sz, i))
        for i in range(N)
    )

    return dict(
        H_sys=H_sys, H_ops=H_ops, jump_ops=jump_ops,
        psi0=psi0, target_state=target,
        T_final=float(T_final), n_dim=dim, n_controls=N - 1,
        populations=populations,
        forbidden_states=[f'P(site {i} ↑)' for i in range(1, N - 1)],
        control_labels=control_labels,
        signal_scale=float(signal_scale),
        action_range='signed',
        system_name=f'chain6kingq14q9_T{T_final:g}',
    )


def add_system_args(parser):
    parser.add_argument('--T_final', type=float, default=30.0,
                        help='Protocol duration (us)')
    parser.add_argument('--signal_scale', type=float, default=3.0,
                        help='Soft amplitude bound on each XX+YY coupling')
