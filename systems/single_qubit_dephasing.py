"""
Single qubit, pure dephasing — the minimal transfer experiment.

Hilbert space: 1 qubit, dim = 2.

1 control:
    H_c = u(t) · σ_x

1 Lindblad operator:
    L = √κ · σ_z        (pure dephasing; no T₁ decay)

Task: flip the spin,  ψ₀ = |↑⟩  →  ψ_t = |↓⟩.

The poles of the Bloch sphere are σ_z eigenstates and hence dephasing-immune;
damage accrues only while the state is transverse, at rate ∝ κ·(1 − ⟨σ_z⟩²).
The physically optimal open-loop policy is: sit on a pole, cross the equator
as fast as the fluence budget allows, sit on the target pole.

This makes the qubit the sharpest KL-vs-DV testbed: with α = 2√κ⟨σ_z⟩ the
Wiener-KL is  ½E[∫α²dt] = 2κT − 2κ∫(1−⟨σ_z⟩²)dt  — up to a constant, the
NEGATIVE of the dephasing-exposure integral. Minimizing KL_W maximizes time
on the equator, i.e. it rewards exactly the decoherence-maximizing states,
while DV penalizes equator residence (trajectories there diffuse in ⟨σ_z⟩).
"""
import jax.numpy as jnp


sx = jnp.array([[0, 1], [1, 0]], dtype=jnp.complex64)
sz = jnp.array([[1, 0], [0, -1]], dtype=jnp.complex64)
I2 = jnp.eye(2, dtype=jnp.complex64)


def create_system(kappa: float = 0.3, T_final: float = 3.0, **_):
    """Build the single dephasing qubit.

    kappa: dephasing rate (the only noise channel).
    """
    H_sys = jnp.zeros((2, 2), dtype=jnp.complex64)

    H_ops = [sx]

    jump_ops = [jnp.sqrt(jnp.array(kappa, dtype=jnp.float32)) * sz]

    psi0   = jnp.array([[1.0], [0.0]], dtype=jnp.complex64)   # |↑⟩
    target = jnp.array([[0.0], [1.0]], dtype=jnp.complex64)   # |↓⟩

    populations = {
        'P(↓)':  0.5 * (I2 - sz),   # target population
        'P(+x)': 0.5 * (I2 + sx),   # transverse marker (0.5 at both poles)
    }

    return dict(
        H_sys=H_sys, H_ops=H_ops, jump_ops=jump_ops,
        psi0=psi0, target_state=target,
        T_final=T_final, n_dim=2, n_controls=1,
        populations=populations,
        forbidden_states=[],
        control_labels=['u_x'],
        signal_scale=3.0,
        action_range='signed',
        system_name=f'qubit1_deph_k{kappa}_T{T_final}',
    )


def add_system_args(parser):
    parser.add_argument('--kappa',   type=float, default=0.3,
                        help='Dephasing rate κ')
    parser.add_argument('--T_final', type=float, default=3.0)
