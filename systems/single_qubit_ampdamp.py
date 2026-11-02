"""
Single qubit, amplitude damping with tunable local-oscillator phase.

Same amplitude-damping task as systems/amplitude_damping.py, with an LO phase:

    H(t) = u_x(t) σ_x + u_y(t) σ_y        (2 controls)
    L    = √γ e^{iφ} σ₋                    (amplitude damping, γ = 0.1)
    ψ₀   = |+⟩ = (|0⟩ + |1⟩)/√2
    ψ_t  = |Y⟩ = (|0⟩ + i|1⟩)/√2
    T    = 1.0

The LO phase φ multiplies the jump operator: the Lindblad master equation for
ρ is exactly invariant under L → L e^{iφ}, but the diffusive unraveling is
not — the solver's record drift is α_φ = 2 Re⟨e^{iφ} L⟩ (solver.py exp_p),
i.e. the monitored quadrature rotates with φ. Hence KL_W = ½E[∫ Σ α_φ² dt]
depends on φ (its zero set rotates), while R_DV = Var(L) is φ-invariant.
φ = 0 reproduces the original experiment exactly.
"""
import jax.numpy as jnp


sx = jnp.array([[0, 1], [1, 0]], dtype=jnp.complex64)
sy = jnp.array([[0, -1j], [1j, 0]], dtype=jnp.complex64)
sm = jnp.array([[0, 1], [0, 0]], dtype=jnp.complex64)   # σ₋ = |0⟩⟨1|
I2 = jnp.eye(2, dtype=jnp.complex64)


def create_system(gamma: float = 0.1, lo_phase: float = 0.0,
                  T_final: float = 1.0, **_):
    """Build the amplitude-damping qubit.

    gamma:    decay rate.
    lo_phase: local-oscillator phase φ; L = √γ e^{iφ} σ₋.
    """
    H_sys = jnp.zeros((2, 2), dtype=jnp.complex64)

    H_ops = [sx, sy]

    phase = jnp.exp(1j * jnp.array(lo_phase, dtype=jnp.float32))
    jump_ops = [jnp.sqrt(jnp.array(gamma, dtype=jnp.float32)) * phase * sm]

    psi0   = jnp.array([[1.0], [1.0]],  dtype=jnp.complex64) / jnp.sqrt(2.0)  # |+⟩
    target = jnp.array([[1.0], [1.0j]], dtype=jnp.complex64) / jnp.sqrt(2.0)  # |Y⟩

    populations = {
        'P(1)':  jnp.array([[0, 0], [0, 1]], dtype=jnp.complex64),  # excited pop
        'P(+y)': 0.5 * (I2 + sy),                                   # target witness
    }

    return dict(
        H_sys=H_sys, H_ops=H_ops, jump_ops=jump_ops,
        psi0=psi0, target_state=target,
        T_final=T_final, n_dim=2, n_controls=2,
        populations=populations,
        forbidden_states=[],
        control_labels=['u_x', 'u_y'],
        signal_scale=3.0,
        action_range='signed',
        system_name=f'ampdamp_g{gamma}_T{T_final}_phi{lo_phase:.4f}',
    )


def add_system_args(parser):
    parser.add_argument('--gamma',    type=float, default=0.1,
                        help='Amplitude damping rate γ')
    parser.add_argument('--lo_phase', type=float, default=0.0,
                        help='LO phase φ (radians); L = √γ e^{iφ} σ₋')
    parser.add_argument('--T_final',  type=float, default=1.0)
