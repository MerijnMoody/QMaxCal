"""
Single-qubit amplitude damping (T₁ relaxation).

Hilbert space: 2 levels {|0⟩, |1⟩}.
Hamiltonian: H(t) = u_x(t)·σ_x + u_y(t)·σ_y    (no σ_z control, no static term)
Jump op:    L = √γ · σ_-   with σ_- = |0⟩⟨1|.

Task: |+⟩ → |Y⟩
    |+⟩ = (|0⟩ + |1⟩)/√2
    |Y⟩ = (|0⟩ + i|1⟩)/√2

The drift α = 2 Re⟨L⟩ = 2√γ · Re⟨σ_-⟩ = 2√γ · Re(ψ_0* ψ_1).
For |+⟩  : α = 2√γ · ½  = √γ        (nonzero — it pays KL!)
For |Y⟩  : α = 2√γ · 0  = 0          (target is in ker drift)
For |1⟩  : α = 0                      (no |0⟩ amplitude)
For |0⟩  : α = 0                      (no |1⟩ amplitude)

So the *only* way to keep α small along the protocol is to avoid coherent
superpositions of |0⟩ and |1⟩ with non-orthogonal phases — but the task
requires going from |+⟩ to |Y⟩, both of which ARE such superpositions.
The optimal Wiener-regularized strategy is to rotate the relative phase of
the superposition (|+⟩ → |Y⟩ via σ_z-like pulses) while passing through
|i⟩-type states that have purely imaginary ⟨σ_-⟩ (where Re part vanishes).

This is a small but informative system — it tests the regularizer on the
simplest possible "non-trivial Wiener KL" case. Used in the original
wiener_regularization paper-style sweep over λ ∈ {0, 0.01, ..., 5.0}.
"""
import jax.numpy as jnp


SIGMA_X = jnp.array([[0, 1], [1, 0]],   dtype=jnp.complex64)
SIGMA_Y = jnp.array([[0, -1j], [1j, 0]], dtype=jnp.complex64)
SIGMA_M = jnp.array([[0, 1], [0, 0]],   dtype=jnp.complex64)


def create_system(gamma: float = 0.1, T_final: float = 1.0, **_):
    dim = 2
    sg  = jnp.sqrt(jnp.array(gamma, dtype=jnp.float32))

    H_sys = jnp.zeros((dim, dim), dtype=jnp.complex64)
    H_ops = [SIGMA_X, SIGMA_Y]
    jump_ops = [sg * SIGMA_M]

    sqrt2 = jnp.sqrt(jnp.array(2.0, dtype=jnp.float32))
    psi0   = jnp.array([1.0, 1.0  ], dtype=jnp.complex64).reshape(2, 1) / sqrt2   # |+⟩
    target = jnp.array([1.0, 1.0j ], dtype=jnp.complex64).reshape(2, 1) / sqrt2   # |Y⟩

    # Populations and useful projectors
    p0 = jnp.array([[1.0, 0.0], [0.0, 0.0]], dtype=jnp.complex64)   # |0⟩⟨0|
    p1 = jnp.array([[0.0, 0.0], [0.0, 1.0]], dtype=jnp.complex64)   # |1⟩⟨1|
    p_plus  = (psi0   @ jnp.conj(psi0.T))
    p_Y     = (target @ jnp.conj(target.T))

    populations = {
        '|0⟩':       p0,
        '|1⟩ (decays)': p1,
        '|+⟩':       p_plus,
        '|Y⟩':       p_Y,
    }

    return dict(
        H_sys=H_sys, H_ops=H_ops, jump_ops=jump_ops,
        psi0=psi0, target_state=target,
        T_final=T_final, n_dim=dim, n_controls=2,
        populations=populations,
        forbidden_states=['|1⟩ (decays)'],
        control_labels=['u_x', 'u_y'],
        signal_scale=5.0,
        action_range='signed',
        system_name=f'ampdamp_g{gamma}_T{T_final}',
    )


def add_system_args(parser):
    parser.add_argument('--gamma',   type=float, default=0.1,
                        help='Amplitude damping rate (T₁⁻¹)')
    parser.add_argument('--T_final', type=float, default=1.0)
