"""
Three-level Λ-system STIRAP.

Levels: |g⟩ = 0 (initial), |e⟩ = 1 (excited, decays), |f⟩ = 2 (target).
Transition dipoles: g↔e (pump, Ω_p), e↔f (Stokes, Ω_s).
Decay: L = √γ |g⟩⟨e|  (spontaneous emission e→g).

Dark state:
    |D⟩ = (Ω_s|g⟩ − Ω_p|f⟩) / √(Ω_p² + Ω_s²)
satisfies both H|D⟩ = 0 and L|D⟩ = 0 (since L annihilates both |g⟩ and |f⟩).

Canonical STIRAP protocol:
    apply Stokes *first* with Ω_s ≫ Ω_p so |D⟩ ≈ |g⟩,
    then ramp down Stokes and ramp up pump so mixing angle θ = arctan(Ω_p/Ω_s)
    rotates adiabatically 0 → π/2, carrying the state from |g⟩ to −|f⟩
    without ever populating the lossy |e⟩.

Goal: |g⟩ → |f⟩.
"""
import jax.numpy as jnp


def _basis(n, dim):
    v = jnp.zeros((dim, 1), dtype=jnp.complex64)
    return v.at[n, 0].set(1.0)


def _hop(n, m, dim):
    op = jnp.zeros((dim, dim), dtype=jnp.complex64)
    return op.at[n, m].set(1.0)


def create_system(gamma: float = 1.0, T_final: float = 2.0, **_):
    dim = 3
    G, E, F = 0, 1, 2

    H_sys = jnp.zeros((dim, dim), dtype=jnp.complex64)

    H_ops = [
        0.5 * (_hop(G, E, dim) + _hop(E, G, dim)),   # Ω_p pump  (g ↔ e)
        0.5 * (_hop(E, F, dim) + _hop(F, E, dim)),   # Ω_s Stokes (e ↔ f)
    ]

    sg = jnp.sqrt(jnp.array(gamma, dtype=jnp.float32))
    L  = sg * _hop(G, E, dim)                         # |g⟩⟨e|: spontaneous emission
    jump_ops = [L]

    psi0   = _basis(G, dim)   # |g⟩
    target = _basis(F, dim)   # |f⟩

    def proj_diag(i):
        return jnp.zeros((dim, dim), dtype=jnp.complex64).at[i, i].set(1.0)

    populations = {
        '|g⟩': proj_diag(G),
        '|e⟩ (bright)': proj_diag(E),
        '|f⟩ (target)': proj_diag(F),
    }

    return dict(
        H_sys=H_sys, H_ops=H_ops, jump_ops=jump_ops,
        psi0=psi0, target_state=target,
        T_final=T_final, n_dim=dim, n_controls=2,
        populations=populations,
        forbidden_states=['|e⟩ (bright)'],
        control_labels=['Ω_p', 'Ω_s'],
        signal_scale=5.0,
        action_range='unsigned',
        system_name=f'stirap3_g{gamma}_T{T_final}',
    )


def add_system_args(parser):
    parser.add_argument('--gamma',   type=float, default=1.0,
                        help='Spontaneous emission rate e→g')
    parser.add_argument('--T_final', type=float, default=2.0)
