"""
system.py - Quantum system definitions and controllers.

Defines the quantum system structure, control parameterizations,
and Hamiltonian construction utilities.
"""
from __future__ import annotations

from typing import NamedTuple, Callable
import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Scalar

from .types import QArray, asqarray, dq


# =============================================================================
# System Definition
# =============================================================================

class SystemDefinition(NamedTuple):
    """
    Complete specification of a quantum control problem.

    Attributes:
        n_dim: Hilbert space dimension
        psi0: Initial quantum state |ψ₀⟩
        target_state: Target quantum state |ψ_target⟩
        jump_ops: List of Lindblad jump operators [L₁, L₂, ...]
        H_sys: System (drift) Hamiltonian
        H_con_func: Function that takes a controller and returns H_control(t)
        n_controls: Number of control parameters
    """
    n_dim: int
    psi0: QArray
    target_state: QArray
    jump_ops: list[QArray]
    H_sys: QArray
    H_con_func: Callable[[nnx.Module], Callable[[Scalar], QArray]]
    n_controls: int


# =============================================================================
# Controller Parameterizations
# =============================================================================

class FourierController(nnx.Module):
    """
    Parameterizes control pulses as Fourier series.

    u_i(t) = a₀ᵢ + Σₖ [aₖᵢ cos(k ω₀ t) + bₖᵢ sin(k ω₀ t)]

    where ω₀ = 2π/T_final ensures periodicity over the control interval.

    Attributes:
        n_controls: Number of control channels
        n_fourier_basis: Number of Fourier modes per channel
        T_final: Total protocol duration
        coeffs: Trainable Fourier coefficients, shape (n_controls, 1 + 2*n_fourier_basis)
    """

    def __init__(
        self,
        n_controls: int,
        n_fourier_basis: int,
        T_final: float,
        *,
        rngs: nnx.Rngs,
        init_scale: float = 0.01
    ):
        self.n_controls = n_controls
        self.n_fourier_basis = n_fourier_basis
        self.T_final = T_final
        self.omega0 = 2 * jnp.pi / T_final

        # Coefficients: [a₀, a₁, ..., aₙ, b₁, ..., bₙ] for each control
        key = rngs.params()
        self.coeffs = nnx.Param(
            jax.random.normal(key, (n_controls, 1 + 2 * n_fourier_basis)) * init_scale
        )

    def __call__(self, t: Scalar) -> jax.Array:
        """
        Evaluate all control pulses at time t.

        Args:
            t: Time point

        Returns:
            Array of shape (n_controls,) with control values
        """
        k = jnp.arange(1, self.n_fourier_basis + 1)
        cos_terms = jnp.cos(k * self.omega0 * t)
        sin_terms = jnp.sin(k * self.omega0 * t)

        # Basis vector: [1, cos(ωt), cos(2ωt), ..., sin(ωt), sin(2ωt), ...]
        basis_vector = jnp.concatenate([jnp.array([1.0]), cos_terms, sin_terms])
        return self.coeffs.value @ basis_vector

    def compute_fluence(self) -> Scalar:
        """
        Compute total control fluence: U = ∫(u₁² + u₂² + ...)dt

        Using Parseval's theorem for Fourier series:
        ∫|u(t)|² dt = T × [|a₀|² + ½ Σₖ (|aₖ|² + |bₖ|²)]

        Returns:
            Total fluence (scalar)
        """
        # coeffs shape: (n_controls, 1 + 2*n_fourier_basis)
        # Layout: [a₀, a₁, ..., aₙ, b₁, ..., bₙ]
        a0 = self.coeffs.value[:, 0]  # DC components
        ak = self.coeffs.value[:, 1:self.n_fourier_basis+1]  # Cosine coefficients
        bk = self.coeffs.value[:, self.n_fourier_basis+1:]  # Sine coefficients

        dc_contribution = jnp.sum(a0**2) * self.T_final
        ac_contribution = 0.5 * jnp.sum(ak**2 + bk**2) * self.T_final
        return dc_contribution + ac_contribution

    def get_pulse_sequence(self) -> tuple[jax.Array, jax.Array]:
        """Get pulse sampled for plotting (same as get_smooth_pulse)."""
        return self.get_smooth_pulse()

    def get_smooth_pulse(self, n_points: int = 1000) -> tuple[jax.Array, jax.Array]:
        """
        Get pulse sampled at many points for smooth plotting.

        Returns:
            (times, controls): times shape (n_points,), controls shape (n_controls, n_points)
        """
        times = jnp.linspace(0, self.T_final, n_points)
        controls = jax.vmap(self)(times).T
        return times, controls


class ConstantController(nnx.Module):
    """Simple constant control for testing."""

    def __init__(self, n_controls: int, values: jax.Array | None = None):
        self.n_controls = n_controls
        if values is None:
            values = jnp.zeros(n_controls)
        self.values = nnx.Param(jnp.asarray(values))

    def __call__(self, t: Scalar) -> jax.Array:
        return self.values.value


# =============================================================================
# Hamiltonian Construction
# =============================================================================

def build_H_con_modulated(
    model: nnx.Module,
    H_ops_q: list[QArray]
) -> Callable[[Scalar], QArray]:
    """
    Construct time-dependent control Hamiltonian from a controller.

    H_con(t) = Σᵢ uᵢ(t) Hᵢ

    Args:
        model: Controller module with __call__(t) -> (n_controls,)
        H_ops_q: List of control Hamiltonian basis operators

    Returns:
        Callable that evaluates H_con at any time t
    """
    # Create time functions for each control channel
    time_funcs = [lambda t, i=i: model(t)[i] for i in range(len(H_ops_q))]

    # Build modulated Hamiltonian
    H_total = dq.modulated(time_funcs[0], H_ops_q[0])
    for i in range(1, len(H_ops_q)):
        H_total = H_total + dq.modulated(time_funcs[i], H_ops_q[i])

    return H_total


# =============================================================================
# Standard Quantum Systems
# =============================================================================

def create_qubit_system(
    gamma: float = 1.0,
    psi0: jax.Array | None = None,
    target: jax.Array | None = None,
    dtype=jnp.complex64
) -> tuple[SystemDefinition, list[QArray]]:
    """
    Create a standard qubit (2-level) system.

    Args:
        gamma: Decay rate
        psi0: Initial state (default: |0⟩)
        target: Target state (default: |1⟩)
        dtype: Data type

    Returns:
        (SystemDefinition, list of control Hamiltonian operators)
    """
    # Pauli matrices
    sigma_x = jnp.array([[0, 1], [1, 0]], dtype=dtype)
    sigma_y = jnp.array([[0, -1j], [1j, 0]], dtype=dtype)
    sigma_z = jnp.array([[1, 0], [0, -1]], dtype=dtype)
    sigma_m = jnp.array([[0, 0], [1, 0]], dtype=dtype)  # |0⟩⟨1|

    # Jump operator (decay)
    L = jnp.sqrt(gamma) * sigma_m

    # Control operators
    H_ops = [asqarray(sigma_x)]

    # Default states
    if psi0 is None:
        psi0 = jnp.array([1, 0], dtype=dtype).reshape(2, 1)
    if target is None:
        target = jnp.array([0, 1], dtype=dtype).reshape(2, 1)

    system = SystemDefinition(
        n_dim=2,
        psi0=asqarray(psi0),
        target_state=asqarray(target),
        jump_ops=[asqarray(L)],
        H_sys=asqarray(jnp.zeros((2, 2), dtype=dtype)),
        H_con_func=lambda m: build_H_con_modulated(m, H_ops),
        n_controls=1
    )

    return system, H_ops


def create_qubit_dephasing_system(
    gamma: float = 1.0,
    psi0: jax.Array | None = None,
    target: jax.Array | None = None,
    dtype=jnp.complex64
) -> tuple[SystemDefinition, list[QArray]]:
    """
    Create a qubit with dephasing noise (σ_z measurement).

    This is the "hard mode" system where the target is a superposition
    state that is sensitive to dephasing.

    Args:
        gamma: Dephasing rate
        psi0: Initial state (default: |0⟩)
        target: Target state (default: |+⟩)
        dtype: Data type

    Returns:
        (SystemDefinition, list of control Hamiltonian operators)
    """
    sigma_x = jnp.array([[0, 1], [1, 0]], dtype=dtype)
    sigma_z = jnp.array([[1, 0], [0, -1]], dtype=dtype)

    # Dephasing jump operator
    L = jnp.sqrt(gamma) * sigma_z

    H_ops = [asqarray(sigma_x)]

    # Default: |0⟩ → |+⟩
    if psi0 is None:
        psi0 = jnp.array([1, 0], dtype=dtype).reshape(2, 1)
    if target is None:
        target = jnp.array([1, 1], dtype=dtype).reshape(2, 1) / jnp.sqrt(2)

    system = SystemDefinition(
        n_dim=2,
        psi0=asqarray(psi0),
        target_state=asqarray(target),
        jump_ops=[asqarray(L)],
        H_sys=asqarray(jnp.zeros((2, 2), dtype=dtype)),
        H_con_func=lambda m: build_H_con_modulated(m, H_ops),
        n_controls=1
    )

    return system, H_ops


def create_three_level_lambda_system(
    gamma: float = 0.5,
    dtype=jnp.complex64
) -> tuple[SystemDefinition, list[QArray]]:
    """
    Create a three-level Lambda system for STIRAP-like control.

    States: |0⟩ (ground), |1⟩ (excited/forbidden), |2⟩ (target)
    Goal: Transfer |0⟩ → |2⟩ while minimizing time in |1⟩

    Args:
        gamma: Decay rate from excited state
        dtype: Data type

    Returns:
        (SystemDefinition, list of control Hamiltonian operators)
    """
    def projector(i, j):
        P = jnp.zeros((3, 3), dtype=dtype)
        return P.at[i, j].set(1)

    # Energy levels
    omega0, omega1, omega2 = 0.0, 5.0, 1.8

    # System Hamiltonian
    H_sys = (omega0 * projector(0, 0) +
             omega1 * projector(1, 1) +
             omega2 * projector(2, 2))

    # Control Hamiltonians: coupling |0⟩-|1⟩ and |1⟩-|2⟩
    H_01 = projector(0, 1) + projector(1, 0)
    H_12 = projector(1, 2) + projector(2, 1)
    H_control = H_01 + H_12

    # Decay from excited state
    decay_strength = jnp.sqrt(gamma * 0.1)
    L_10 = decay_strength * projector(0, 1)  # |1⟩ → |0⟩
    L_12 = decay_strength * projector(2, 1)  # |1⟩ → |2⟩

    H_ops = [asqarray(H_control)]

    # States
    psi0 = jnp.array([1, 0, 0], dtype=dtype).reshape(3, 1)
    target = jnp.array([0, 0, 1], dtype=dtype).reshape(3, 1)

    system = SystemDefinition(
        n_dim=3,
        psi0=asqarray(psi0),
        target_state=asqarray(target),
        jump_ops=[asqarray(L_10), asqarray(L_12)],
        H_sys=asqarray(H_sys),
        H_con_func=lambda m: build_H_con_modulated(m, H_ops),
        n_controls=1
    )

    return system, H_ops
