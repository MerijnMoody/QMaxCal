"""
4-level dark route with amplitude damping.

Levels: |b⟩=0, |t⟩=1, |d⟩=2, |dump⟩=3.
|b⟩ and |t⟩ decay to |dump⟩ at rate γ; |d⟩ is stable.
Goal: transfer |b⟩ → |t⟩ via the stable dark state |d⟩.

Controls:
  Ω_bt(t) · (|b⟩⟨t| + h.c.)/2
  Ω_bd(t) · (|b⟩⟨d| + h.c.)/2
  Ω_dt(t) · (|d⟩⟨t| + h.c.)/2
"""
import jax.numpy as jnp


def _basis(n, dim):
    v = jnp.zeros((dim, 1), dtype=jnp.complex64)
    return v.at[n, 0].set(1.0)


def _hop(n, m, dim):
    op = jnp.zeros((dim, dim), dtype=jnp.complex64)
    return op.at[n, m].set(1.0)


def create_system(gamma: float = 2.0, T_final: float = 1.0, **_):
    dim = 4
    sg  = jnp.sqrt(jnp.array(gamma, dtype=jnp.float32))

    H_sys = jnp.zeros((dim, dim), dtype=jnp.complex64)

    H_ops = [
        0.5 * (_hop(0, 1, dim) + _hop(1, 0, dim)),   # b <-> t
        0.5 * (_hop(0, 2, dim) + _hop(2, 0, dim)),   # b <-> d
        0.5 * (_hop(2, 1, dim) + _hop(1, 2, dim)),   # d <-> t
    ]
    jump_ops = [
        sg * _hop(3, 0, dim),   # |dump><b|
        sg * _hop(3, 1, dim),   # |dump><t|
    ]
    psi0   = _basis(0, dim)     # |b>
    target = _basis(1, dim)     # |t>

    # Diagonal projectors for populations
    def proj(i):
        return jnp.zeros((dim, dim), dtype=jnp.complex64).at[i, i].set(1.0)

    populations = {
        '|b⟩':    proj(0),
        '|t⟩':    proj(1),
        '|d⟩':    proj(2),
        '|dump⟩': proj(3),
    }

    return dict(
        H_sys=H_sys, H_ops=H_ops, jump_ops=jump_ops,
        psi0=psi0, target_state=target,
        T_final=T_final, n_dim=dim, n_controls=3,
        populations=populations,
        control_labels=['Ω_bt', 'Ω_bd', 'Ω_dt'],
        signal_scale=50.0,
        action_range='signed',
        system_name=f'darkroute_g{gamma}_T{T_final}',
    )


def add_system_args(parser):
    parser.add_argument('--gamma',   type=float, default=2.0, help='Decay rate to dump')
    parser.add_argument('--T_final', type=float, default=1.0)
