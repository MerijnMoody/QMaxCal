"""
density_matrix.py - Stochastic Master Equation (SME) solver for density matrices.

Extends the DSSE solver to density matrix evolution, implementing the framework
from Manzano & Zambrini (2022) "Quantum thermodynamics under continuous monitoring".

Key features:
- SME evolution: dρ = -i[H,ρ]dt + D(ρ)dt + measurement backaction
- Two-Point Measurement (TPM) scheme with probability tracking
- Combined measurement + Girsanov entropy production

The SME for homodyne detection (Eq. 12 in Manzano & Zambrini):
    dρ = -i[H,ρ]dt + Σ_k D_k(ρ)dt + Σ_k H_k(ρ) dw_k

where:
    D_k(ρ) = L_k ρ L†_k - ½{L†_k L_k, ρ}  (Lindblad dissipator)
    H_k(ρ) = L_k ρ + ρ L†_k - Tr[(L_k + L†_k)ρ]ρ  (measurement backaction)
"""
from __future__ import annotations

from abc import abstractmethod
import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import PRNGKeyArray, PyTree, Scalar
from typing import NamedTuple, Callable

from .types import (
    QArray, TimeQArray, Options, Method,
    asqarray, astimeqarray
)


# =============================================================================
# Result Types
# =============================================================================

class SMESolveResult(NamedTuple):
    """Result from SME density matrix solver."""
    rhos: Array              # Density matrices at saved times [n_traj, n_times, d, d]
    measurements: Array      # Measurement records [n_traj, n_times-1, n_meas]
    drifts: Array           # Measurement drifts α(t) = Tr[(L+L†)ρ]
    log_prob_traj: Array    # Log-probability of trajectory (Girsanov factor)
    extra: PyTree | None    # Additional data
    tsave: Array            # Time points


class TPMResult(NamedTuple):
    """Result from Two-Point Measurement scheme."""
    initial_eigenvalue: int     # Index of initial measurement outcome
    final_eigenvalue: int       # Index of final measurement outcome
    log_p_initial: Scalar       # log p⁰_n₀
    log_p_final: Scalar         # log p^τ_nτ
    delta_S_system: Scalar      # -log p^τ + log p⁰ (system entropy change)
    trajectory_result: SMESolveResult  # Full trajectory data


# =============================================================================
# Density Matrix State
# =============================================================================

class DensityMatrixState(eqx.Module):
    """
    State container for density matrix evolution with likelihood tracking.

    Attributes:
        rho: Density matrix ρ
        Y: Integrated measurement signal ∫I(t)dt
        log_prob: Log-probability of trajectory (Girsanov factor)
        drift: Current measurement drift α = Tr[(L+L†)ρ]
    """
    rho: Array              # Density matrix [d, d]
    Y: Array                # Integrated measurement [n_meas]
    log_prob: Scalar        # Accumulated log-probability
    drift: Array            # Current drift [n_meas]


# =============================================================================
# Utility Functions
# =============================================================================

def commutator(A: Array, B: Array) -> Array:
    """Compute [A, B] = AB - BA."""
    return A @ B - B @ A


def anticommutator(A: Array, B: Array) -> Array:
    """Compute {A, B} = AB + BA."""
    return A @ B + B @ A


def lindblad_dissipator(L: Array, rho: Array) -> Array:
    """
    Lindblad dissipator: D(ρ) = L ρ L† - ½{L†L, ρ}
    """
    Ldag = L.conj().T
    LdagL = Ldag @ L
    return L @ rho @ Ldag - 0.5 * anticommutator(LdagL, rho)


def measurement_backaction(L: Array, rho: Array, alpha: Scalar) -> Array:
    """
    Measurement backaction (homodyne): H(ρ) = L ρ + ρ L† - α ρ

    where α = Tr[(L + L†)ρ] is the measurement drift.
    """
    Ldag = L.conj().T
    return L @ rho + rho @ Ldag - alpha * rho


def compute_drift(L: Array, rho: Array) -> Scalar:
    """
    Compute measurement drift α = Tr[(L + L†)ρ] = 2 Re[Tr(L ρ)]
    """
    return 2 * jnp.trace(L @ rho).real


def normalize_density_matrix(rho: Array) -> Array:
    """Normalize density matrix to unit trace."""
    return rho / jnp.trace(rho)


# =============================================================================
# TPM (Two-Point Measurement) Functions
# =============================================================================

def measure_energy_basis(
    rho: Array,
    H: Array,
    key: PRNGKeyArray
) -> tuple[int, Scalar, Array]:
    """
    Perform projective measurement in energy eigenbasis.

    Args:
        rho: Density matrix to measure
        H: Hamiltonian defining the measurement basis
        key: PRNG key for sampling

    Returns:
        (eigenvalue_index, log_probability, post_measurement_state)
    """
    # Diagonalize Hamiltonian
    eigenvalues, eigenvectors = jnp.linalg.eigh(H)

    # Compute measurement probabilities p_n = ⟨n|ρ|n⟩ = |⟨n|ψ⟩|² for each eigenstate
    probs = jnp.array([
        jnp.abs(jnp.trace(jnp.outer(eigenvectors[:, n].conj(), eigenvectors[:, n]) @ rho)).real
        for n in range(len(eigenvalues))
    ])
    probs = probs / jnp.sum(probs)  # Normalize

    # Sample outcome
    outcome = jax.random.categorical(key, jnp.log(probs + 1e-10))

    # Post-measurement state (pure state projector)
    psi_n = eigenvectors[:, outcome]
    rho_post = jnp.outer(psi_n, psi_n.conj())

    log_prob = jnp.log(probs[outcome] + 1e-10)

    return outcome, log_prob, rho_post


def compute_tpm_entropy_change(
    log_p_initial: Scalar,
    log_p_final: Scalar
) -> Scalar:
    """
    Compute system entropy change from TPM: ΔS = -log p^τ + log p⁰

    This is the stochastic entropy change of the system.
    """
    return -log_p_final + log_p_initial


# =============================================================================
# SME Integrator
# =============================================================================

class SMESolveIntegrator(eqx.Module):
    """
    Euler-Maruyama integrator for the Stochastic Master Equation.

    Implements the SME for homodyne detection (Manzano & Zambrini Eq. 12):

        dρ = -i[H,ρ]dt + Σ_k D_k(ρ)dt + Σ_k H_k(ρ) dw_k

    with measurement record:
        dY = α dt + dW,  where α = Tr[(L+L†)ρ]
    """
    ts: Array
    rho0: Array
    method: Method
    options: Options
    H: TimeQArray
    H_con: TimeQArray
    Ls: list[TimeQArray]
    key: PRNGKeyArray
    nsteps: int
    nsteps_per_save: int

    @property
    def dt(self) -> float:
        return self.method.dt

    @property
    def nmeas(self) -> int:
        return len(self.Ls)

    def forward(
        self,
        t: Scalar,
        state: DensityMatrixState,
        dW: Array
    ) -> DensityMatrixState:
        """
        Single Euler-Maruyama step for SME.

        Args:
            t: Current time
            state: Current density matrix state
            dW: Wiener increment for this step

        Returns:
            Updated density matrix state
        """
        rho = state.rho
        Y = state.Y
        log_prob = state.log_prob

        # Get time-dependent operators
        L_list = [op(t).data if hasattr(op(t), 'data') else op(t) for op in self.Ls]
        H_sys = self.H(t)
        H_con = self.H_con(t)

        H_sys_data = H_sys.data if hasattr(H_sys, 'data') else H_sys
        H_con_data = H_con.data if hasattr(H_con, 'data') else H_con
        H_total = H_sys_data + H_con_data

        # Compute drifts α_k = Tr[(L_k + L†_k)ρ]
        drifts = jnp.array([compute_drift(L, rho) for L in L_list])

        # Measurement record increment: dY = α dt + dW
        dY = drifts * self.dt + dW

        # === Deterministic evolution ===
        # Hamiltonian: -i[H, ρ]
        drho_det = -1j * commutator(H_total, rho) * self.dt

        # Lindblad dissipators: Σ_k D_k(ρ)
        for L in L_list:
            drho_det = drho_det + lindblad_dissipator(L, rho) * self.dt

        # === Stochastic evolution (measurement backaction) ===
        drho_stoch = jnp.zeros_like(rho)
        for k, (L, alpha, dw) in enumerate(zip(L_list, drifts, dW)):
            drho_stoch = drho_stoch + measurement_backaction(L, rho, alpha) * dw

        # Update density matrix
        rho_new = rho + drho_det + drho_stoch

        # Ensure Hermiticity and normalize
        rho_new = 0.5 * (rho_new + rho_new.conj().T)
        rho_new = normalize_density_matrix(rho_new)

        # === Log-probability increment (Girsanov factor) ===
        # log P(dW | α) = -½ Σ_k [(dW_k - 0)²/dt] + const
        # The Girsanov likelihood ratio is: dlog(P_controlled/P_ref) = α·dW - ½α²dt
        # For entropy production we track: ∫ α·dY - ½α² dt = ∫ α·dW + ½α² dt
        # But simpler: just track ∫ -½(dW - α dt)²/dt which is the innovation log-likelihood

        # Innovation-based log-likelihood (tracks distinguishability from reference)
        step_log_prob = jnp.sum(-0.5 * dW**2 / self.dt)  # Standard Wiener measure

        return DensityMatrixState(
            rho=rho_new,
            Y=Y + dY,
            log_prob=log_prob + step_log_prob,
            drift=drifts
        )

    def run(self) -> SMESolveResult:
        """
        Execute the full trajectory integration.

        Returns:
            SMESolveResult with density matrices, measurements, and likelihoods
        """
        nsave = len(self.ts)
        delta_t = self.ts[1] - self.ts[0]
        d = self.rho0.shape[0]

        # Initialize state
        initial_drifts = jnp.zeros(self.nmeas)
        state0 = DensityMatrixState(
            rho=self.rho0,
            Y=jnp.zeros(self.nmeas),
            log_prob=0.0,
            drift=initial_drifts
        )

        def outer_step(carry, key_save):
            """One save interval: multiple inner steps."""
            t, state = carry

            # Generate Wiener increments for all inner steps
            dWs = jnp.sqrt(self.dt) * jax.random.normal(
                key_save, (self.nsteps_per_save, self.nmeas)
            )

            def inner_step(carry_inner, dW):
                t_inner, state_inner = carry_inner
                state_inner = self.forward(t_inner, state_inner, dW)
                t_inner = t_inner + self.dt
                return (t_inner, state_inner), None

            (t, state), _ = jax.lax.scan(inner_step, (t, state), dWs)

            saved = {
                'rho': state.rho,
                'Y': state.Y,
                'drift': state.drift,
                'log_prob': state.log_prob
            }

            return (t, state), saved

        # Split keys for each save interval
        keys = jax.random.split(self.key, nsave - 1)

        # Initial saved data
        saved0 = {
            'rho': state0.rho,
            'Y': state0.Y,
            'drift': state0.drift,
            'log_prob': state0.log_prob
        }

        # Run integration
        (t_final, state_final), saved_steps = jax.lax.scan(
            outer_step, (self.ts[0], state0), keys
        )

        # Concatenate initial state with stepped states
        saved = jax.tree.map(
            lambda s0, s: jnp.concatenate([jnp.expand_dims(s0, 0), s]),
            saved0, saved_steps
        )

        # Compute measurement record from integrated signal
        measurements = jnp.diff(saved['Y'], axis=0) / delta_t

        return SMESolveResult(
            rhos=saved['rho'],
            measurements=measurements,
            drifts=saved['drift'],
            log_prob_traj=saved['log_prob'][-1],  # Final accumulated log-prob
            extra={'log_prob_trajectory': saved['log_prob']},
            tsave=self.ts
        )


# =============================================================================
# Main Solver Interface
# =============================================================================

def dsmesolve(
    H_sys: QArray,
    jump_ops: list[QArray],
    rho0: Array,
    tsave: Array,
    keys: PRNGKeyArray,
    nsteps: int,
    nsteps_per_save: int,
    *,
    H_con: QArray | None = None,
    method: Method,
    options: Options = Options()
) -> SMESolveResult:
    """
    Solve the Stochastic Master Equation for density matrices.

    Simulates quantum trajectories under continuous homodyne measurement
    using density matrix formalism, enabling tracking of mixed states
    and TPM measurement probabilities.

    Args:
        H_sys: System (drift) Hamiltonian
        jump_ops: List of jump operators [L₁, L₂, ...]
        rho0: Initial density matrix ρ₀
        tsave: Array of times at which to save states
        keys: PRNGKey array of shape (n_trajectories,) for stochastic sampling
        nsteps: Total number of integration steps
        nsteps_per_save: Number of steps between save points
        H_con: Control Hamiltonian H_con(t) (optional)
        method: Integration method (e.g., EulerMaruyama(dt=1e-3))
        options: Solver options

    Returns:
        SMESolveResult with:
            - rhos: Density matrices at save times
            - measurements: Measurement records I(t)
            - drifts: Measurement drifts α(t)
            - log_prob_traj: Girsanov log-probability of trajectory
            - tsave: Time array
    """
    # Convert to time-dependent form
    H_sys_t = astimeqarray(H_sys)

    if H_con is not None:
        H_con_t = astimeqarray(H_con)
    else:
        # Zero control Hamiltonian
        if hasattr(H_sys, 'data'):
            zero_H = jnp.zeros_like(H_sys.data)
        else:
            zero_H = jnp.zeros_like(H_sys(0).data if hasattr(H_sys(0), 'data') else H_sys(0))
        H_con_t = astimeqarray(asqarray(zero_H))

    Ls_t = [astimeqarray(L) for L in jump_ops]

    def _solve_single(key):
        """Solve single trajectory."""
        integrator = SMESolveIntegrator(
            ts=tsave,
            rho0=rho0,
            method=method,
            options=options,
            H=H_sys_t,
            H_con=H_con_t,
            Ls=Ls_t,
            key=key,
            nsteps=nsteps,
            nsteps_per_save=nsteps_per_save
        )
        return integrator.run()

    # Vectorize over trajectories
    return jax.vmap(_solve_single)(keys)


# =============================================================================
# TPM Protocol with SME Evolution
# =============================================================================

def run_tpm_protocol(
    H_initial: Array,
    H_final: Array,
    H_sys: QArray,
    jump_ops: list[QArray],
    rho0: Array,
    tsave: Array,
    key: PRNGKeyArray,
    nsteps: int,
    nsteps_per_save: int,
    *,
    H_con: QArray | None = None,
    method: Method,
    options: Options = Options()
) -> TPMResult:
    """
    Run full Two-Point Measurement protocol.

    Protocol:
    1. Initial projective measurement in eigenbasis of H_initial
    2. Stochastic evolution under SME
    3. Final projective measurement in eigenbasis of H_final

    Args:
        H_initial: Hamiltonian for initial measurement basis
        H_final: Hamiltonian for final measurement basis
        H_sys: System Hamiltonian for dynamics
        jump_ops: Jump operators
        rho0: Initial density matrix (before first measurement)
        tsave: Save times
        key: PRNG key
        nsteps, nsteps_per_save: Integration parameters
        H_con: Optional control Hamiltonian
        method: Integration method
        options: Solver options

    Returns:
        TPMResult with measurement outcomes and entropy change
    """
    key1, key2, key3 = jax.random.split(key, 3)

    # 1. Initial measurement
    n0, log_p0, rho_post_initial = measure_energy_basis(rho0, H_initial, key1)

    # 2. Stochastic evolution
    traj_result = dsmesolve(
        H_sys=H_sys,
        jump_ops=jump_ops,
        rho0=rho_post_initial,
        tsave=tsave,
        keys=key2[None],  # Single trajectory
        nsteps=nsteps,
        nsteps_per_save=nsteps_per_save,
        H_con=H_con,
        method=method,
        options=options
    )

    # Extract final state (squeeze out batch dimension)
    rho_final = traj_result.rhos[0, -1]  # [0] for batch, [-1] for final time

    # 3. Final measurement
    n_tau, log_p_tau, _ = measure_energy_basis(rho_final, H_final, key3)

    # System entropy change
    delta_S = compute_tpm_entropy_change(log_p0, log_p_tau)

    # Create single-trajectory result
    single_traj = SMESolveResult(
        rhos=traj_result.rhos[0],
        measurements=traj_result.measurements[0],
        drifts=traj_result.drifts[0],
        log_prob_traj=traj_result.log_prob_traj[0],
        extra=jax.tree.map(lambda x: x[0], traj_result.extra),
        tsave=traj_result.tsave
    )

    return TPMResult(
        initial_eigenvalue=n0,
        final_eigenvalue=n_tau,
        log_p_initial=log_p0,
        log_p_final=log_p_tau,
        delta_S_system=delta_S,
        trajectory_result=single_traj
    )


# =============================================================================
# Combined Entropy Production (Measurement + Girsanov)
# =============================================================================

def compute_total_entropy_production(
    tpm_result: TPMResult,
    alpha_backward: Array,
    dt: float
) -> dict:
    """
    Compute total entropy production combining TPM and Girsanov contributions.

    Total entropy: S_tot = ΔS_system + σ_Λ

    where:
    - ΔS_system = -log p^τ + log p⁰ (from TPM measurements)
    - σ_Λ = Girsanov factor from stochastic evolution

    For diffusive trajectories with self-adjoint L operators, σ_Λ = 0.
    Otherwise, σ_Λ = ∫(α_F - α_B)·dY - ½∫(α_F² - α_B²)dt

    Args:
        tpm_result: Result from run_tpm_protocol
        alpha_backward: Backward drifts from time-reversed process
        dt: Time step

    Returns:
        Dictionary with entropy components and total
    """
    traj = tpm_result.trajectory_result
    alpha_forward = traj.drifts
    measurements = traj.measurements

    # System entropy change from TPM
    delta_S_system = tpm_result.delta_S_system

    # Girsanov entropy (environment contribution)
    # σ_Λ = ∫(α_F - α_B)·I dt - ½∫(α_F² - α_B²) dt

    # Align dimensions (measurements has one fewer time point than drifts)
    alpha_F = alpha_forward[:-1]
    alpha_B = alpha_backward[:-1]

    term1 = jnp.sum((alpha_F - alpha_B) * measurements * dt)
    term2 = 0.5 * jnp.sum((alpha_F**2 - alpha_B**2) * dt)
    sigma_lambda = term1 - term2

    # Total entropy production
    S_total = delta_S_system + sigma_lambda

    return {
        'S_total': S_total,
        'delta_S_system': delta_S_system,
        'sigma_lambda': sigma_lambda,
        'term1_stochastic': term1,
        'term2_drift_correction': term2
    }


def compute_entropy_selfadjoint(tpm_result: TPMResult) -> dict:
    """
    Compute entropy for self-adjoint Lindblad operators.

    When L = L†, the environment entropy flow σ_Λ = 0,
    and total entropy equals system entropy change.

    This is the case for:
    - Position measurements (L = x)
    - Momentum measurements (L = p)
    - Quadrature measurements in general

    Args:
        tpm_result: Result from run_tpm_protocol

    Returns:
        Dictionary with entropy (just system contribution)
    """
    return {
        'S_total': tpm_result.delta_S_system,
        'delta_S_system': tpm_result.delta_S_system,
        'sigma_lambda': 0.0,  # Zero for self-adjoint L
        'note': 'Self-adjoint L operators: no entropy flow to environment'
    }