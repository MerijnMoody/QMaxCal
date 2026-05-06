"""
thermodynamics.py - Forward-backward protocols and entropy production.

Implements the Manzano & Zambrini (2022) framework for computing
entropy production in driven quantum systems via time-reversal.

Key concepts:
- Forward process: System driven by control u(t), generates measurement record I(t)
- Backward process: Time-reversed evolution using I(T-t) and reversed operators
- Entropy production: Distinguishability between forward and backward processes

The quantum Crooks fluctuation theorem states:
    ⟨e^{-S_tot}⟩ = 1

where S_tot is the total entropy production along a trajectory.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Scalar
from jax.scipy.integrate import trapezoid

from .types import QArray, asqarray


# =============================================================================
# Time-Reversal Operator Θ
# =============================================================================

def get_time_reversal_unitary(dim: int) -> Array:
    """
    Get the unitary part of the time-reversal operator Θ = U_Θ K.

    For spin-1/2 (dim=2): U_Θ = i σ_y = [[0, 1], [-1, 0]]

    The full time-reversal operator is anti-unitary: Θ = U_Θ K
    where K is complex conjugation.

    Properties:
    - Θ |↑⟩ = |↓⟩ (up to phase)
    - Θ |↓⟩ = -|↑⟩
    - Θ² = -1 for spin-1/2

    For an operator A: Θ A Θ† = U_Θ A* U_Θ†

    Args:
        dim: Hilbert space dimension (must be 2 for now)

    Returns:
        U_Θ: Unitary part of time-reversal operator
    """
    if dim != 2:
        raise NotImplementedError(
            f"Time-reversal only implemented for spin-1/2 (dim=2), got dim={dim}"
        )

    # U_Θ = i σ_y for spin-1/2
    # σ_y = [[0, -i], [i, 0]]
    # i σ_y = [[0, 1], [-1, 0]]
    U_theta = jnp.array([[0., 1.], [-1., 0.]], dtype=jnp.complex128)

    return U_theta


def time_reverse_operator(A: Array, U_theta: Array) -> Array:
    """
    Apply time-reversal to an operator: Θ A Θ† = U_Θ A* U_Θ†

    Args:
        A: Operator to time-reverse
        U_theta: Unitary part of Θ

    Returns:
        Time-reversed operator
    """
    A_conj = A.conj()  # Complex conjugate (not adjoint!)
    return U_theta @ A_conj @ U_theta.conj().T


def time_reverse_state(psi: Array, U_theta: Array) -> Array:
    """
    Apply time-reversal to a state: Θ|ψ⟩ = U_Θ |ψ*⟩

    Args:
        psi: State vector to time-reverse
        U_theta: Unitary part of Θ

    Returns:
        Time-reversed state
    """
    return U_theta @ psi.conj()


# =============================================================================
# Projective Measurements
# =============================================================================

def sample_projective_measurement(
    psi: Array,
    key: jax.random.PRNGKey,
    basis: str = 'computational'
) -> tuple[int, Scalar, Array]:
    """
    Perform projective measurement on a pure state.

    Args:
        psi: State vector, shape (dim,) or (dim, 1)
        key: PRNG key for sampling
        basis: 'computational' for σ_z eigenbasis {|0⟩, |1⟩}

    Returns:
        (outcome, log_prob, post_state):
        - outcome: Measurement result (0 or 1)
        - log_prob: log |⟨n|ψ⟩|²
        - post_state: Post-measurement eigenstate |n⟩
    """
    psi = psi.squeeze()
    dim = psi.shape[0]

    # Probabilities in computational basis
    probs = jnp.abs(psi)**2

    # Sample outcome
    outcome = jax.random.choice(key, dim, p=probs)

    # Log probability of the outcome
    log_prob = jnp.log(jnp.clip(probs[outcome], 1e-15, 1.0))

    # Post-measurement state (eigenstate)
    post_state = jnp.zeros(dim, dtype=psi.dtype)
    post_state = post_state.at[outcome].set(1.0)

    return outcome, log_prob, post_state.reshape(-1, 1)


def compute_backward_projection_log_prob(
    psi_bwd_final: Array,
    psi_0: Array,
    U_theta: Array = None
) -> Scalar:
    """
    Compute log |⟨Θψ_0|ψ_bwd⟩|² for backward projection.

    This computes the probability of the backward trajectory ending at
    the time-reversed initial state.

    Args:
        psi_bwd_final: Final backward state, shape (dim,) or (dim, 1)
        psi_0: Initial state (forward process), shape (dim,) or (dim, 1)
        U_theta: Time-reversal unitary (computed if None)

    Returns:
        log_prob: log |⟨Θψ_0|ψ_bwd⟩|²
    """
    psi_bwd = psi_bwd_final.squeeze()
    psi_0 = psi_0.squeeze()
    dim = psi_0.shape[0]

    if U_theta is None:
        U_theta = get_time_reversal_unitary(dim)

    # Time-reverse the initial state: Θ|ψ_0⟩ = U_Θ |ψ_0*⟩
    theta_psi_0 = U_theta @ psi_0.conj()

    # Overlap |⟨Θψ_0|ψ_bwd⟩|²
    overlap = jnp.abs(jnp.vdot(theta_psi_0, psi_bwd))**2

    return jnp.log(jnp.clip(overlap, 1e-15, 1.0))


# =============================================================================
# Reverse Operators (Multiple conventions)
# =============================================================================

def compute_reverse_operators(
    H_sys: QArray,
    jump_ops: list[QArray],
    method: str = 'theta'
) -> tuple[QArray, list[QArray]]:
    """
    Compute time-reversed operators for the backward process.

    Three methods available:

    1. 'theta' (default): Full quantum time-reversal using Θ = i σ_y K
       - H_rev = Θ H Θ† = U_Θ H* U_Θ†
       - L_rev = Θ L† Θ† = U_Θ (L†)* U_Θ† = U_Θ L^T U_Θ†

       For spin-1/2:
       - Θ σ_x Θ† = -σ_x
       - Θ σ_y Θ† = -σ_y
       - Θ σ_z Θ† = -σ_z
       - Θ σ_+ Θ† = -σ_-
       - Θ σ_- Θ† = -σ_+

    2. 'adjoint': Simple adjoint reversal (Manzano "naive" protocol)
       - H_rev = H* (complex conjugate)
       - L_rev = L† (adjoint)

    3. 'theta_L': Θ reversal for H, but L† for jump operators
       - H_rev = Θ H Θ†
       - L_rev = L†

    Args:
        H_sys: System Hamiltonian
        jump_ops: List of jump operators
        method: 'theta', 'adjoint', or 'theta_L'

    Returns:
        (H_rev, L_rev_list): Reversed Hamiltonian and jump operators
    """
    H_data = H_sys.data if hasattr(H_sys, 'data') else H_sys
    dim = H_data.shape[0]

    if method == 'adjoint':
        # Original "naive" reversal: H* and L†
        H_rev = asqarray(H_data.conj())
        L_rev_list = []
        for L in jump_ops:
            L_data = L.data if hasattr(L, 'data') else L
            L_tilde = L_data.conj().T  # L†
            L_rev_list.append(asqarray(L_tilde))
        return H_rev, L_rev_list

    elif method in ['theta', 'theta_L']:
        # Full quantum time-reversal using Θ
        U_theta = get_time_reversal_unitary(dim)

        # H_rev = Θ H Θ† = U_Θ H* U_Θ†
        H_rev_data = time_reverse_operator(H_data, U_theta)
        H_rev = asqarray(H_rev_data)

        L_rev_list = []
        for L in jump_ops:
            L_data = L.data if hasattr(L, 'data') else L

            if method == 'theta':
                # L_rev = Θ L† Θ† = U_Θ (L†)* U_Θ† = U_Θ L^T U_Θ†
                # Note: (L†)* = (L*)^T = L^T (transpose, not adjoint)
                L_dag = L_data.conj().T  # L†
                L_tilde = time_reverse_operator(L_dag, U_theta)
            else:  # theta_L
                # Just use L† without Θ transformation
                L_tilde = L_data.conj().T

            L_rev_list.append(asqarray(L_tilde))

        return H_rev, L_rev_list

    else:
        raise ValueError(f"Unknown method: {method}. Use 'theta', 'adjoint', or 'theta_L'")


# =============================================================================
# Backward Evolution
# =============================================================================

def evolve_backward(
    H_rev: Array,
    L_rev_list: list[Array],
    start_state: Array,
    signal_reversed: Array,
    dt: float,
    H_con_bwd_func: callable = None,
    t_axis_bwd: Array = None,
    time_reverse_start: bool = False
) -> tuple[Array, Array]:
    """
    Evolve the system backwards in time using time-reversed measurement record.

    This implements the backward stochastic master equation:

        dψ_B = [-iH_rev - ½L†L + ½⟨L†L⟩]ψ_B dt + (L - ⟨L⟩)ψ_B dW_B

    where dW_B is derived from the reversed measurement signal.

    Args:
        H_rev: Reversed system Hamiltonian (raw array)
        L_rev_list: List of reversed jump operators (raw arrays)
        start_state: Starting state for backward evolution (typically target or final state)
        signal_reversed: Time-reversed measurement record I(T-t), shape (n_steps,)
        dt: Time step
        H_con_bwd_func: Optional function t -> H_con(T-t) for active backward driving
        t_axis_bwd: Time points for backward evolution
        time_reverse_start: If True, apply Θ to start_state before evolution

    Returns:
        (drifts, states): Backward drifts α_B(t) and state trajectory
    """
    # Optionally time-reverse the starting state
    if time_reverse_start:
        dim = start_state.shape[0]
        U_theta = get_time_reversal_unitary(dim)
        start_state = time_reverse_state(start_state.squeeze(), U_theta).reshape(-1, 1)
    # Extract raw arrays from QArrays to avoid tracing issues inside jax.lax.scan
    L_raw = L_rev_list[0].data if hasattr(L_rev_list[0], 'data') else L_rev_list[0]
    H_rev_raw = H_rev.data if hasattr(H_rev, 'data') else H_rev

    # Compute observable: L + L†
    L_dag = L_raw.conj().T
    obs = L_raw + L_dag

    n_steps = signal_reversed.shape[0]
    if t_axis_bwd is None:
        t_axis_bwd = jnp.arange(n_steps) * dt

    def step_fn(psi, inputs):
        I_t, t_curr = inputs

        # Total Hamiltonian (passive + optional active control)
        H_total = H_rev_raw
        if H_con_bwd_func is not None:
            H_con_val = H_con_bwd_func(t_curr)
            # Handle both QArray and raw array returns
            if hasattr(H_con_val, 'data'):
                H_total = H_total + H_con_val.data
            else:
                H_total = H_total + H_con_val

        # Compute drift (measurement expectation)
        psi_dag = psi.conj().T
        exp_L = (psi_dag @ obs @ psi).squeeze().real

        # Innovation from measurement record
        dW = (I_t - exp_L) * dt

        # SME evolution terms
        term1 = -1j * H_total @ psi * dt

        L_psi = L_raw @ psi
        LdagL_psi = L_dag @ L_psi

        # Diffusive SSE terms
        term2 = -0.5 * (LdagL_psi - exp_L * L_psi + 0.25 * (exp_L**2) * psi) * dt
        term3 = (L_psi - 0.5 * exp_L * psi) * dW

        psi_next = psi + term1 + term2 + term3
        psi_next = psi_next / jnp.linalg.norm(psi_next)

        return psi_next, (exp_L, psi)

    scan_inputs = (signal_reversed, t_axis_bwd)
    _, (drifts_bwd, states_bwd) = jax.lax.scan(step_fn, start_state, scan_inputs)

    return drifts_bwd, states_bwd


def evolve_backward_with_innovations(
    H_rev: Array,
    L_rev_list: list[Array],
    start_state: Array,
    innovations_reversed: Array,
    dt: float,
    H_con_bwd_func: callable = None,
    t_axis_bwd: Array = None,
) -> tuple[Array, Array]:
    """
    Evolve backward using pre-computed innovations from forward process.

    This fixes the eigenstate-sticking issue by using the forward innovations
    dW_F(τ-t) directly, rather than computing innovations from the backward state.

    The key insight is that for microscopic reversibility, the backward process
    should be driven by the SAME noise realization as the forward process,
    just time-reversed.

    Args:
        H_rev: Reversed system Hamiltonian
        L_rev_list: List of reversed jump operators
        start_state: Starting state (typically Θ|n_τ⟩ for TPM scheme)
        innovations_reversed: Time-reversed forward innovations dW_F(τ-t)
        dt: Time step
        H_con_bwd_func: Optional backward control Hamiltonian
        t_axis_bwd: Time axis for backward evolution

    Returns:
        (drifts, states): Backward drifts and state trajectory
    """
    L_raw = L_rev_list[0].data if hasattr(L_rev_list[0], 'data') else L_rev_list[0]
    H_rev_raw = H_rev.data if hasattr(H_rev, 'data') else H_rev

    L_dag = L_raw.conj().T
    obs = L_raw + L_dag

    n_steps = innovations_reversed.shape[0]
    if t_axis_bwd is None:
        t_axis_bwd = jnp.arange(n_steps) * dt

    def step_fn(psi, inputs):
        dW, t_curr = inputs

        # Total Hamiltonian
        H_total = H_rev_raw
        if H_con_bwd_func is not None:
            H_con_val = H_con_bwd_func(t_curr)
            if hasattr(H_con_val, 'data'):
                H_total = H_total + H_con_val.data
            else:
                H_total = H_total + H_con_val

        # Compute drift for output (but NOT for innovation)
        psi_dag = psi.conj().T
        exp_L = (psi_dag @ obs @ psi).squeeze().real

        # SME evolution using PRE-COMPUTED innovations
        term1 = -1j * H_total @ psi * dt

        L_psi = L_raw @ psi
        LdagL_psi = L_dag @ L_psi

        # Diffusive terms - using forward innovations dW directly
        term2 = -0.5 * (LdagL_psi - exp_L * L_psi + 0.25 * (exp_L**2) * psi) * dt
        term3 = (L_psi - 0.5 * exp_L * psi) * dW

        psi_next = psi + term1 + term2 + term3
        psi_next = psi_next / jnp.linalg.norm(psi_next)

        return psi_next, (exp_L, psi)

    scan_inputs = (innovations_reversed, t_axis_bwd)
    _, (drifts_bwd, states_bwd) = jax.lax.scan(step_fn, start_state, scan_inputs)

    return drifts_bwd, states_bwd


# =============================================================================
# Entropy Production
# =============================================================================

def compute_entropy_production(
    alpha_F: Array,
    alpha_B: Array,
    measurements: Array,
    dt: float
) -> Array:
    """
    Compute total entropy production for each trajectory.

    The entropy production is given by:

        S_tot = ∫(α_F - α_B)·I(t) dt - ½∫(α_F² - α_B²) dt

    where:
    - α_F = ⟨L + L†⟩_F is the forward drift
    - α_B = ⟨L + L†⟩_B is the backward drift
    - I(t) is the measurement record

    This satisfies the integral fluctuation theorem: ⟨e^{-S_tot}⟩ = 1

    Args:
        alpha_F: Forward drifts, shape (n_trajs, n_times)
        alpha_B: Backward drifts, shape (n_trajs, n_times)
        measurements: Measurement records I(t), shape (n_trajs, n_times)
        dt: Time step

    Returns:
        S_tot: Entropy production per trajectory, shape (n_trajs,)
    """
    # Term 1: ∫(α_F - α_B)·I dt (stochastic work)
    term_1 = jnp.sum((alpha_F - alpha_B) * measurements * dt, axis=1)

    # Term 2: ½∫(α_F² - α_B²) dt (drift correction)
    term_2 = 0.5 * jnp.sum((alpha_F**2 - alpha_B**2) * dt, axis=1)

    S_tot = term_1 - term_2
    return S_tot


def compute_entropy_production_proxy(
    alpha_F: Array,
    alpha_B: Array,
    dt: float
) -> Scalar:
    """
    Compute entropy production proxy (Girsanov-like cost).

    This is a simplified entropy cost that can be used during training:

        J = ½ E[∫(α_F - α_B)² dt]

    This measures the distinguishability between forward and backward
    processes and is minimized when the processes are thermodynamically
    reversible.

    Args:
        alpha_F: Forward drifts, shape (n_trajs, n_times)
        alpha_B: Backward drifts, shape (n_trajs, n_times)
        dt: Time step

    Returns:
        Mean entropy proxy across trajectories
    """
    diff_sq = (alpha_F - alpha_B)**2
    integral_per_traj = jnp.sum(diff_sq * dt, axis=1)
    return 0.5 * jnp.mean(integral_per_traj)


# =============================================================================
# Girsanov KL Divergence
# =============================================================================

def compute_girsanov_kl(
    log_p_p: Array,
    log_p_q: Array
) -> Scalar:
    """
    Compute KL divergence via likelihood ratios (Girsanov theorem).

    The Girsanov theorem states that for diffusion processes:

        KL(P || Q) = E_P[log(dP/dQ)]

    where the likelihood ratio is accumulated during integration.

    This is mathematically equivalent to:

        KL(P || Q) = ½ E_P[∫(α_P - α_Q)² dt]

    where α_P, α_Q are the drifts under processes P and Q.

    Args:
        log_p_p: Log-likelihood under controlled process, shape (n_trajs,)
        log_p_q: Log-likelihood under uncontrolled process, shape (n_trajs,)

    Returns:
        KL divergence estimate
    """
    ll_ratio = log_p_p - log_p_q
    return jnp.mean(ll_ratio)


def compute_girsanov_integral(
    exp_p: Array,
    exp_q: Array,
    tsave: Array
) -> Scalar:
    """
    Compute KL divergence via drift integral (Girsanov theorem).

    KL(P || Q) = ½ E_P[∫(α_P - α_Q)² dt]

    This should match compute_girsanov_kl when computed correctly.

    Args:
        exp_p: Drifts under controlled process, shape (n_trajs, n_times, n_channels)
        exp_q: Drifts under uncontrolled process, shape (n_trajs, n_times, n_channels)
        tsave: Time array

    Returns:
        KL divergence estimate via integral
    """
    # Sum over measurement channels
    drift_diff_sq = jnp.sum((exp_p - exp_q)**2, axis=-1)

    # Integrate over time
    integral_per_traj = trapezoid(drift_diff_sq, tsave, axis=1)

    return 0.5 * jnp.mean(integral_per_traj)


# =============================================================================
# Fluctuation Theorem Verification
# =============================================================================

def verify_crooks_fluctuation_theorem(S_tot: Array) -> dict:
    """
    Verify the quantum Crooks fluctuation theorem.

    The integral fluctuation theorem states:
        ⟨e^{-S_tot}⟩ = 1

    This is a fundamental check that the forward-backward protocol
    is correctly implemented.

    Args:
        S_tot: Entropy production per trajectory, shape (n_trajs,)

    Returns:
        Dictionary with:
        - 'mean_S': Mean entropy production
        - 'jarzynski': ⟨e^{-S_tot}⟩ (should be ~1)
        - 'std_S': Standard deviation of entropy production
        - 'passed': Whether |jarzynski - 1| < tolerance
    """
    mean_S = float(jnp.mean(S_tot))
    std_S = float(jnp.std(S_tot))
    jarzynski = float(jnp.mean(jnp.exp(-S_tot)))

    # Tolerance depends on number of samples
    n_samples = len(S_tot)
    tolerance = 3.0 / jnp.sqrt(n_samples)  # Rough 3-sigma bound

    return {
        'mean_S': mean_S,
        'jarzynski': jarzynski,
        'std_S': std_S,
        'passed': abs(jarzynski - 1.0) < max(0.1, tolerance),
        'tolerance': float(tolerance)
    }
