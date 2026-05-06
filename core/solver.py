"""
solver.py - Diffusive Stochastic Schrödinger Equation (DSSE) solver.

Implements the Euler-Maruyama integrator for quantum trajectories under
continuous homodyne measurement with optional control Hamiltonians.

The solver tracks two parallel evolutions:
- psi_p: State under controlled Hamiltonian H_sys + H_con (physical)
- psi_q: State under uncontrolled Hamiltonian H_sys (guidance/reference)

This dual tracking enables computation of likelihood ratios and
Girsanov-based information costs.
"""
from __future__ import annotations

from abc import abstractmethod
import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import PRNGKeyArray, PyTree, Scalar

from .types import (
    QArray, TimeQArray, DSSESolveResult, Options, Method,
    asqarray, astimeqarray
)


# =============================================================================
# Likelihood State
# =============================================================================

class LikelihoodState(eqx.Module):
    """
    State container for dual-process likelihood tracking.

    Attributes:
        psi_p: Quantum state under controlled (physical) dynamics
        psi_q: Quantum state under uncontrolled (reference) dynamics
        Y: Integrated measurement signal ∫I(t)dt
        log_p_p: Log-likelihood of trajectory under controlled process
        log_p_q: Log-likelihood of trajectory under uncontrolled process
    """
    psi_p: QArray      # Controlled state
    psi_q: QArray      # Reference state
    Y: Array           # Integrated measurement
    log_p_p: Scalar    # Log-likelihood (controlled)
    log_p_q: Scalar    # Log-likelihood (uncontrolled)


# =============================================================================
# Base Integrator Classes
# =============================================================================

class DiffusiveSolveSaveMixin(eqx.Module):
    """Mixin for saving trajectory data."""
    y0: QArray
    Es: list[QArray] | None
    options: Options

    def postprocess_saved(self, saved: PyTree, ylast: LikelihoodState) -> PyTree:
        return saved


class DiffusiveSolveIntegrator(DiffusiveSolveSaveMixin):
    """
    Base class for diffusive SDE integrators.

    Implements the outer loop structure for trajectory integration
    with save points and batched inner steps.
    """
    ts: Array
    y0: QArray
    method: Method
    options: Options
    H: TimeQArray
    H_con: TimeQArray
    Ls: list[TimeQArray]
    Es: list[QArray] | None
    key: PRNGKeyArray
    nsteps: int
    nsteps_per_save: int

    @property
    def dt(self) -> float:
        return self.method.dt

    @property
    @abstractmethod
    def nmeas(self) -> int:
        """Number of measurement channels."""
        pass

    @abstractmethod
    def forward(self, t: Scalar, y: LikelihoodState, dW: Array) -> LikelihoodState:
        """Single integration step."""
        pass

    def run(self) -> DSSESolveResult:
        """
        Execute the full trajectory integration.

        Returns:
            DSSESolveResult with states, measurements, and extra data
        """
        nsave = len(self.ts)
        delta_t = self.ts[1] - self.ts[0]

        # Initialize dual states
        y0 = LikelihoodState(
            psi_p=self.y0,
            psi_q=self.y0,
            Y=jnp.zeros(self.nmeas),
            log_p_p=0.0,
            log_p_q=0.0
        )

        def outer_step(carry, key_save):
            """One save interval: multiple inner steps."""
            t, y = carry

            # Generate Wiener increments for all inner steps
            dWs = jnp.sqrt(self.dt) * jax.random.normal(
                key_save, (self.nsteps_per_save, self.nmeas)
            )

            def inner_step(carry_inner, dW):
                """Single Euler-Maruyama step."""
                t_inner, y_inner = carry_inner
                y_inner = self.forward(t_inner, y_inner, dW)
                t_inner = t_inner + self.dt
                return (t_inner, y_inner), None

            (t, y), _ = jax.lax.scan(inner_step, (t, y), dWs)

            # Save state data
            saved = {
                'states': y.psi_p.data,
                'Isave': y.Y
            }
            if self.options.save_extra:
                saved['extra'] = self.options.save_extra(t, y)

            return (t, y), saved

        # Split keys for each save interval
        keys = jax.random.split(self.key, nsave - 1)

        # Initial saved data
        saved0 = {
            'states': y0.psi_p.data,
            'Isave': y0.Y,
            'extra': self.options.save_extra(self.ts[0], y0) if self.options.save_extra else None
        }

        # Run integration
        (t_final, y_final), saved_steps = jax.lax.scan(
            outer_step, (self.ts[0], y0), keys
        )

        # Concatenate initial state with stepped states
        saved = jax.tree.map(
            lambda s0, s: jnp.concatenate([jnp.expand_dims(s0, 0), s]),
            saved0, saved_steps
        )

        # Compute measurement record from integrated signal
        measurements = jnp.diff(saved['Isave'], axis=0) / delta_t

        return DSSESolveResult(
            states=asqarray(saved['states']),
            expects=None,
            measurements=measurements,
            extra=saved['extra'],
            tsave=self.ts
        )


# =============================================================================
# Euler-Maruyama Integrator
# =============================================================================

class DSSESolveEulerMayuramaIntegrator(DiffusiveSolveIntegrator):
    """
    Euler-Maruyama integrator for the Diffusive Stochastic Schrödinger Equation.

    Implements the stochastic master equation for homodyne detection:

        dψ = [-iH - ½L†L + ½⟨L†L⟩]ψ dt + (L - ⟨L⟩)ψ dW

    with measurement record:

        dY = ⟨L + L†⟩ dt + dW

    The integrator tracks both controlled (H_sys + H_con) and uncontrolled
    (H_sys only) evolutions for computing likelihood ratios.
    """

    @property
    def nmeas(self) -> int:
        return len(self.Ls)

    def forward(self, t: Scalar, y: LikelihoodState, dW: Array) -> LikelihoodState:
        """
        Single Euler-Maruyama step for both controlled and reference processes.

        Args:
            t: Current time
            y: Current likelihood state
            dW: Wiener increment for this step

        Returns:
            Updated likelihood state
        """
        psi_p, psi_q, Y = y.psi_p, y.psi_q, y.Y
        log_p_p, log_p_q = y.log_p_p, y.log_p_q

        # Get time-dependent operators
        L = [op(t) for op in self.Ls]
        H_sys = self.H(t)
        H_con = self.H_con(t)
        H_p = H_sys + H_con  # Controlled Hamiltonian

        # =====================================================================
        # Physical (controlled) evolution
        # =====================================================================
        Lpsi_p = [_L @ psi_p for _L in L]

        # Drift: α_p = ⟨L + L†⟩ = 2 Re⟨ψ|L|ψ⟩
        exp_p = jnp.stack([
            2 * (psi_p.dag() @ _Lpsi_p).data.squeeze().real
            for _Lpsi_p in Lpsi_p
        ])

        # Measurement increment
        dY = exp_p * self.dt + dW

        # State evolution (Ito form)
        # dψ = -iH ψ dt - ½(L†L - ⟨L†L⟩)ψ dt + (L - ⟨L⟩)ψ dW
        dpsi_p = (
            -1j * self.dt * H_p @ psi_p
            - 0.5 * self.dt * sum([
                _L.dag() @ _Lpsi_p - _exp * _Lpsi_p + 0.25 * _exp**2 * psi_p
                for _L, _Lpsi_p, _exp in zip(L, Lpsi_p, exp_p)
            ])
            + sum([
                (_Lpsi_p - 0.5 * _exp * psi_p) * _dw
                for _Lpsi_p, _exp, _dw in zip(Lpsi_p, exp_p, dW)
            ])
        )

        # Normalize
        next_psi_p_data = psi_p.data + dpsi_p.data
        next_psi_p = asqarray(next_psi_p_data / jnp.linalg.norm(next_psi_p_data))

        # Log-likelihood increment: log p(dW | drift=0)
        step_log_p_p = jnp.sum(
            jax.scipy.stats.norm.logpdf(dW, loc=0.0, scale=jnp.sqrt(self.dt))
        )

        # =====================================================================
        # Reference (uncontrolled) evolution
        # =====================================================================
        H_q = H_sys  # No control

        Lpsi_q = [_L @ psi_q for _L in L]
        exp_q = jnp.stack([
            2 * (psi_q.dag() @ _Lpsi_q).data.squeeze().real
            for _Lpsi_q in Lpsi_q
        ])

        # Innovation: what the reference process "sees" given the actual measurement
        dW_q = dY - exp_q * self.dt

        dpsi_q = (
            -1j * self.dt * H_q @ psi_q
            - 0.5 * self.dt * sum([
                _L.dag() @ _Lpsi_q - _exp * _Lpsi_q + 0.25 * _exp**2 * psi_q
                for _L, _Lpsi_q, _exp in zip(L, Lpsi_q, exp_q)
            ])
            + sum([
                (_Lpsi_q - 0.5 * _exp * psi_q) * _dw
                for _Lpsi_q, _exp, _dw in zip(Lpsi_q, exp_q, dW_q)
            ])
        )

        next_psi_q_data = psi_q.data + dpsi_q.data
        next_psi_q = asqarray(next_psi_q_data / jnp.linalg.norm(next_psi_q_data))

        # Log-likelihood for reference process
        step_log_p_q = jnp.sum(
            jax.scipy.stats.norm.logpdf(dW_q, loc=0.0, scale=jnp.sqrt(self.dt))
        )

        return LikelihoodState(
            psi_p=next_psi_p,
            psi_q=next_psi_q,
            Y=Y + dY,
            log_p_p=log_p_p + step_log_p_p,
            log_p_q=log_p_q + step_log_p_q
        )


# =============================================================================
# Heun (Stochastic RK2) Integrator
# =============================================================================

class DSSESolveHeunIntegrator(DiffusiveSolveIntegrator):
    """
    Heun (stochastic RK2) integrator for the Diffusive Stochastic Schrödinger Equation.

    Uses predictor-corrector scheme for O(dt) strong convergence instead of O(√dt).

    The Heun method:
    1. Predict: ψ̃ = ψ + f(ψ)dt + g(ψ)dW  (Euler step)
    2. Correct: ψ' = ψ + ½[f(ψ) + f(ψ̃)]dt + ½[g(ψ) + g(ψ̃)]dW
    """

    @property
    def nmeas(self) -> int:
        return len(self.Ls)

    def _compute_drift_diffusion(self, t: Scalar, psi: QArray, H: QArray):
        """
        Compute drift and diffusion terms for SSE.

        Returns:
            drift: -iH ψ - ½Σ(L†L - ⟨L+L†⟩L + ¼⟨L+L†⟩²)ψ
            diffusion: List of (L - ½⟨L+L†⟩)ψ for each jump op
            exp_vals: ⟨L + L†⟩ for each jump op
        """
        L = [op(t) for op in self.Ls]
        Lpsi = [_L @ psi for _L in L]

        # Drift coefficient: α = ⟨L + L†⟩ = 2 Re⟨ψ|L|ψ⟩
        exp_vals = jnp.stack([
            2 * (psi.dag() @ _Lpsi).data.squeeze().real
            for _Lpsi in Lpsi
        ])

        # Deterministic drift: -iH ψ - ½Σ(L†L ψ - α L ψ + ¼α² ψ)
        drift = (
            -1j * H @ psi
            - 0.5 * sum([
                _L.dag() @ _Lpsi - _exp * _Lpsi + 0.25 * _exp**2 * psi
                for _L, _Lpsi, _exp in zip(L, Lpsi, exp_vals)
            ])
        )

        # Stochastic diffusion: (L - ½α)ψ for each jump op
        diffusion = [
            _Lpsi - 0.5 * _exp * psi
            for _Lpsi, _exp in zip(Lpsi, exp_vals)
        ]

        return drift, diffusion, exp_vals

    def forward(self, t: Scalar, y: LikelihoodState, dW: Array) -> LikelihoodState:
        """
        Single Heun (predictor-corrector) step for both controlled and reference processes.
        """
        psi_p, psi_q, Y = y.psi_p, y.psi_q, y.Y
        log_p_p, log_p_q = y.log_p_p, y.log_p_q

        # Get time-dependent Hamiltonians
        H_sys = self.H(t)
        H_con = self.H_con(t)
        H_p = H_sys + H_con  # Controlled
        H_q = H_sys           # Reference (uncontrolled)

        # =====================================================================
        # Physical (controlled) evolution - Heun method
        # =====================================================================

        # Step 1: Compute drift and diffusion at current state
        drift_p, diff_p, exp_p = self._compute_drift_diffusion(t, psi_p, H_p)

        # Step 2: Euler predictor
        dpsi_p_euler = drift_p.data * self.dt + sum([
            d.data * dw for d, dw in zip(diff_p, dW)
        ])
        psi_p_pred_data = psi_p.data + dpsi_p_euler
        psi_p_pred = asqarray(psi_p_pred_data / jnp.linalg.norm(psi_p_pred_data))

        # Step 3: Compute drift and diffusion at predicted state
        drift_p_pred, diff_p_pred, _ = self._compute_drift_diffusion(t + self.dt, psi_p_pred, H_p)

        # Step 4: Heun corrector (average of drifts and diffusions)
        dpsi_p_heun = (
            0.5 * (drift_p.data + drift_p_pred.data) * self.dt
            + 0.5 * sum([
                (d.data + dp.data) * dw
                for d, dp, dw in zip(diff_p, diff_p_pred, dW)
            ])
        )

        next_psi_p_data = psi_p.data + dpsi_p_heun
        next_psi_p = asqarray(next_psi_p_data / jnp.linalg.norm(next_psi_p_data))

        # Measurement increment
        dY = exp_p * self.dt + dW

        # Log-likelihood
        step_log_p_p = jnp.sum(
            jax.scipy.stats.norm.logpdf(dW, loc=0.0, scale=jnp.sqrt(self.dt))
        )

        # =====================================================================
        # Reference (uncontrolled) evolution - Heun method
        # =====================================================================

        # Step 1: Compute drift and diffusion at current state
        drift_q, diff_q, exp_q = self._compute_drift_diffusion(t, psi_q, H_q)

        # Innovation for reference process
        dW_q = dY - exp_q * self.dt

        # Step 2: Euler predictor
        dpsi_q_euler = drift_q.data * self.dt + sum([
            d.data * dw for d, dw in zip(diff_q, dW_q)
        ])
        psi_q_pred_data = psi_q.data + dpsi_q_euler
        psi_q_pred = asqarray(psi_q_pred_data / jnp.linalg.norm(psi_q_pred_data))

        # Step 3: Compute drift and diffusion at predicted state
        drift_q_pred, diff_q_pred, _ = self._compute_drift_diffusion(t + self.dt, psi_q_pred, H_q)

        # Step 4: Heun corrector
        dpsi_q_heun = (
            0.5 * (drift_q.data + drift_q_pred.data) * self.dt
            + 0.5 * sum([
                (d.data + dp.data) * dw
                for d, dp, dw in zip(diff_q, diff_q_pred, dW_q)
            ])
        )

        next_psi_q_data = psi_q.data + dpsi_q_heun
        next_psi_q = asqarray(next_psi_q_data / jnp.linalg.norm(next_psi_q_data))

        # Log-likelihood for reference
        step_log_p_q = jnp.sum(
            jax.scipy.stats.norm.logpdf(dW_q, loc=0.0, scale=jnp.sqrt(self.dt))
        )

        return LikelihoodState(
            psi_p=next_psi_p,
            psi_q=next_psi_q,
            Y=Y + dY,
            log_p_p=log_p_p + step_log_p_p,
            log_p_q=log_p_q + step_log_p_q
        )


# =============================================================================
# Exponential Split-Step Integrator
# =============================================================================

class DSSESolveExponentialSplitIntegrator(DiffusiveSolveIntegrator):
    """
    Exponential split-step integrator for the Diffusive SSE.

    Splits the evolution into:
    1. Deterministic: exp(-i H_eff dt) where H_eff = H - i½L†L (exact)
    2. Stochastic: Euler step for measurement backaction

    This eliminates discretization error in the Hamiltonian dynamics,
    leaving only O(√dt) error from the stochastic part.
    """

    @property
    def nmeas(self) -> int:
        return len(self.Ls)

    def _build_effective_hamiltonian(self, t: Scalar, H: QArray) -> Array:
        """Build H_eff = H - i½ΣL†L (non-Hermitian effective Hamiltonian)."""
        L = [op(t) for op in self.Ls]
        H_eff = H.data.astype(jnp.complex128)
        for _L in L:
            H_eff = H_eff - 0.5j * (_L.dag() @ _L).data
        return H_eff

    def _apply_exponential(self, H_eff: Array, psi: QArray) -> QArray:
        """Apply exp(-i H_eff dt) to state using matrix exponential."""
        propagator = jax.scipy.linalg.expm(-1j * H_eff * self.dt)
        new_data = propagator @ psi.data
        return asqarray(new_data)

    def _apply_stochastic_step(self, t: Scalar, psi: QArray, dW: Array):
        """
        Apply remaining SSE terms not in the exponential.

        The exponential handles: -iH - ½L†L
        This step handles: +½⟨L+L†⟩Lψ dt - ¼⟨L+L†⟩²ψ dt + (L - ½⟨L+L†⟩)ψ dW

        Full SSE drift: -iH - ½L†L + ½⟨L+L†⟩L - ¼⟨L+L†⟩²
        Exponential has: -iH - ½L†L
        Remaining drift: +½⟨L+L†⟩L - ¼⟨L+L†⟩²
        """
        L = [op(t) for op in self.Ls]
        Lpsi = [_L @ psi for _L in L]

        # Expectation values: α = ⟨L + L†⟩ = 2 Re⟨ψ|L|ψ⟩
        exp_vals = jnp.stack([
            2 * (psi.dag() @ _Lpsi).data.squeeze().real
            for _Lpsi in Lpsi
        ])

        # Deterministic correction (terms not in exponential):
        # +½⟨L+L†⟩Lψ - ¼⟨L+L†⟩²ψ
        det_correction = sum([
            (0.5 * _exp * _Lpsi - 0.25 * _exp**2 * psi) * self.dt
            for _Lpsi, _exp in zip(Lpsi, exp_vals)
        ])

        # Stochastic increment: Σ(L - ½⟨L+L†⟩)ψ dW
        dpsi_stoch = sum([
            (_Lpsi - 0.5 * _exp * psi) * _dw
            for _Lpsi, _exp, _dw in zip(Lpsi, exp_vals, dW)
        ])

        new_data = psi.data + det_correction.data + dpsi_stoch.data
        new_psi = asqarray(new_data / jnp.linalg.norm(new_data))

        return new_psi, exp_vals

    def forward(self, t: Scalar, y: LikelihoodState, dW: Array) -> LikelihoodState:
        """
        Single exponential split-step for both controlled and reference processes.

        Order: Deterministic (exact) -> Stochastic (Euler)
        """
        psi_p, psi_q, Y = y.psi_p, y.psi_q, y.Y
        log_p_p, log_p_q = y.log_p_p, y.log_p_q

        # Get Hamiltonians
        H_sys = self.H(t)
        H_con = self.H_con(t)
        H_p = asqarray(H_sys.data + H_con.data)  # Controlled
        H_q = H_sys  # Reference

        # =====================================================================
        # Physical (controlled) evolution
        # =====================================================================

        # Step 1: Exact deterministic evolution
        H_eff_p = self._build_effective_hamiltonian(t, H_p)
        psi_p_det = self._apply_exponential(H_eff_p, psi_p)

        # Step 2: Stochastic step
        next_psi_p, exp_p = self._apply_stochastic_step(t, psi_p_det, dW)

        # Measurement increment
        dY = exp_p * self.dt + dW

        # Log-likelihood
        step_log_p_p = jnp.sum(
            jax.scipy.stats.norm.logpdf(dW, loc=0.0, scale=jnp.sqrt(self.dt))
        )

        # =====================================================================
        # Reference (uncontrolled) evolution
        # =====================================================================

        # Step 1: Exact deterministic evolution
        H_eff_q = self._build_effective_hamiltonian(t, H_q)
        psi_q_det = self._apply_exponential(H_eff_q, psi_q)

        # Get exp_q before stochastic step for innovation
        L = [op(t) for op in self.Ls]
        Lpsi_q = [_L @ psi_q_det for _L in L]
        exp_q = jnp.stack([
            2 * (psi_q_det.dag() @ _Lpsi).data.squeeze().real
            for _Lpsi in Lpsi_q
        ])

        # Innovation for reference process
        dW_q = dY - exp_q * self.dt

        # Step 2: Stochastic step with innovation
        next_psi_q, _ = self._apply_stochastic_step(t, psi_q_det, dW_q)

        # Log-likelihood for reference
        step_log_p_q = jnp.sum(
            jax.scipy.stats.norm.logpdf(dW_q, loc=0.0, scale=jnp.sqrt(self.dt))
        )

        return LikelihoodState(
            psi_p=next_psi_p,
            psi_q=next_psi_q,
            Y=Y + dY,
            log_p_p=log_p_p + step_log_p_p,
            log_p_q=log_p_q + step_log_p_q
        )


# =============================================================================
# Strang Split-Step Integrator
# =============================================================================

class DSSESolveStrangSplitIntegrator(DiffusiveSolveIntegrator):
    """
    Strang (symmetric) split-step integrator for the Diffusive SSE.

    Uses symmetric splitting for O(dt²) accuracy:
    1. Half-step deterministic: exp(-i H_eff dt/2)
    2. Full stochastic step: Euler for measurement backaction
    3. Half-step deterministic: exp(-i H_eff dt/2)

    This gives better accuracy than simple splitting while maintaining
    proper handling of the stochastic terms.
    """

    @property
    def nmeas(self) -> int:
        return len(self.Ls)

    def _build_effective_hamiltonian(self, t: Scalar, H: QArray) -> Array:
        """Build H_eff = H - i½ΣL†L (non-Hermitian effective Hamiltonian)."""
        L = [op(t) for op in self.Ls]
        H_eff = H.data.astype(jnp.complex128)
        for _L in L:
            H_eff = H_eff - 0.5j * (_L.dag() @ _L).data
        return H_eff

    def _apply_half_exponential(self, H_eff: Array, psi: QArray) -> QArray:
        """Apply exp(-i H_eff dt/2) to state."""
        propagator = jax.scipy.linalg.expm(-1j * H_eff * self.dt * 0.5)
        new_data = propagator @ psi.data
        # Don't normalize here - we'll normalize at the end
        return asqarray(new_data)

    def _apply_stochastic_step(self, t: Scalar, psi: QArray, dW: Array):
        """
        Apply remaining SSE terms not in the exponential.

        The exponential handles: -iH - ½L†L
        This step handles: +½⟨L+L†⟩Lψ dt - ¼⟨L+L†⟩²ψ dt + (L - ½⟨L+L†⟩)ψ dW
        """
        L = [op(t) for op in self.Ls]
        Lpsi = [_L @ psi for _L in L]

        # Expectation values: α = ⟨L + L†⟩ = 2 Re⟨ψ|L|ψ⟩
        exp_vals = jnp.stack([
            2 * (psi.dag() @ _Lpsi).data.squeeze().real
            for _Lpsi in Lpsi
        ])

        # Deterministic correction (terms not in exponential):
        # +½⟨L+L†⟩Lψ - ¼⟨L+L†⟩²ψ
        det_correction = sum([
            (0.5 * _exp * _Lpsi - 0.25 * _exp**2 * psi) * self.dt
            for _Lpsi, _exp in zip(Lpsi, exp_vals)
        ])

        # Stochastic increment: Σ(L - ½⟨L+L†⟩)ψ dW
        dpsi_stoch = sum([
            (_Lpsi - 0.5 * _exp * psi) * _dw
            for _Lpsi, _exp, _dw in zip(Lpsi, exp_vals, dW)
        ])

        new_data = psi.data + det_correction.data + dpsi_stoch.data
        new_psi = asqarray(new_data)  # Don't normalize yet

        return new_psi, exp_vals

    def forward(self, t: Scalar, y: LikelihoodState, dW: Array) -> LikelihoodState:
        """
        Single Strang split-step for both controlled and reference processes.

        Order: Half-det -> Full-stoch -> Half-det
        """
        psi_p, psi_q, Y = y.psi_p, y.psi_q, y.Y
        log_p_p, log_p_q = y.log_p_p, y.log_p_q

        # Get Hamiltonians
        H_sys = self.H(t)
        H_con = self.H_con(t)
        H_p = asqarray(H_sys.data + H_con.data)  # Controlled
        H_q = H_sys  # Reference

        # =====================================================================
        # Physical (controlled) evolution - Strang splitting
        # =====================================================================

        # Build effective Hamiltonian
        H_eff_p = self._build_effective_hamiltonian(t, H_p)

        # Step 1: Half-step deterministic
        psi_p_half1 = self._apply_half_exponential(H_eff_p, psi_p)

        # Step 2: Full stochastic step (at midpoint)
        psi_p_stoch, exp_p = self._apply_stochastic_step(t + 0.5*self.dt, psi_p_half1, dW)

        # Step 3: Half-step deterministic
        H_eff_p_end = self._build_effective_hamiltonian(t + self.dt, H_p)
        psi_p_half2 = self._apply_half_exponential(H_eff_p_end, psi_p_stoch)

        # Normalize final state
        next_psi_p = asqarray(psi_p_half2.data / jnp.linalg.norm(psi_p_half2.data))

        # Measurement increment (use expectation at start for consistency)
        dY = exp_p * self.dt + dW

        # Log-likelihood
        step_log_p_p = jnp.sum(
            jax.scipy.stats.norm.logpdf(dW, loc=0.0, scale=jnp.sqrt(self.dt))
        )

        # =====================================================================
        # Reference (uncontrolled) evolution - Strang splitting
        # =====================================================================

        # Build effective Hamiltonian
        H_eff_q = self._build_effective_hamiltonian(t, H_q)

        # Step 1: Half-step deterministic
        psi_q_half1 = self._apply_half_exponential(H_eff_q, psi_q)

        # Get exp_q for innovation calculation
        L = [op(t + 0.5*self.dt) for op in self.Ls]
        Lpsi_q = [_L @ psi_q_half1 for _L in L]
        exp_q = jnp.stack([
            2 * (psi_q_half1.dag() @ _Lpsi).data.squeeze().real
            for _Lpsi in Lpsi_q
        ])

        # Innovation for reference process
        dW_q = dY - exp_q * self.dt

        # Step 2: Full stochastic step with innovation
        psi_q_stoch, _ = self._apply_stochastic_step(t + 0.5*self.dt, psi_q_half1, dW_q)

        # Step 3: Half-step deterministic
        H_eff_q_end = self._build_effective_hamiltonian(t + self.dt, H_q)
        psi_q_half2 = self._apply_half_exponential(H_eff_q_end, psi_q_stoch)

        # Normalize final state
        next_psi_q = asqarray(psi_q_half2.data / jnp.linalg.norm(psi_q_half2.data))

        # Log-likelihood for reference
        step_log_p_q = jnp.sum(
            jax.scipy.stats.norm.logpdf(dW_q, loc=0.0, scale=jnp.sqrt(self.dt))
        )

        return LikelihoodState(
            psi_p=next_psi_p,
            psi_q=next_psi_q,
            Y=Y + dY,
            log_p_p=log_p_p + step_log_p_p,
            log_p_q=log_p_q + step_log_p_q
        )


# =============================================================================
# Main Solver Interface
# =============================================================================

def dssesolve(
    H_sys: QArray,
    jump_ops: list[QArray],
    psi0: QArray,
    tsave: Array,
    keys: PRNGKeyArray,
    nsteps: int,
    nsteps_per_save: int,
    *,
    H_con: QArray | None = None,
    method: Method,
    options: Options = Options()
) -> DSSESolveResult:
    """
    Solve the Diffusive Stochastic Schrödinger Equation.

    Simulates quantum trajectories under continuous homodyne measurement,
    tracking both controlled and uncontrolled dynamics for likelihood
    ratio computation.

    Args:
        H_sys: System (drift) Hamiltonian
        jump_ops: List of jump operators [L₁, L₂, ...]
        psi0: Initial state |ψ₀⟩
        tsave: Array of times at which to save states
        keys: PRNGKey array of shape (n_trajectories,) for stochastic sampling
        nsteps: Total number of integration steps
        nsteps_per_save: Number of steps between save points
        H_con: Control Hamiltonian H_con(t) (optional)
        method: Integration method (e.g., EulerMaruyama(dt=1e-3))
        options: Solver options

    Returns:
        DSSESolveResult with:
            - states: Quantum states at save times
            - measurements: Measurement records I(t)
            - extra: Dictionary with 'log_p_p', 'log_p_q', 'exp_p', 'exp_q'
            - tsave: Time array
    """
    # Convert to time-dependent form
    H_sys_t = astimeqarray(H_sys)

    if H_con is not None:
        H_con_t = astimeqarray(H_con)
    else:
        # Zero control Hamiltonian
        zero_H = jnp.zeros_like(
            H_sys.data if isinstance(H_sys, QArray) else H_sys(0).data
        )
        H_con_t = astimeqarray(asqarray(zero_H))

    Ls_t = [astimeqarray(L) for L in jump_ops]

    # Define extra data to save
    def save_extra(t, y: LikelihoodState):
        L_of_t = [L(t) for L in Ls_t]
        exp_p = jnp.stack([
            2 * (y.psi_p.dag() @ (L @ y.psi_p)).data.squeeze().real
            for L in L_of_t
        ])
        exp_q = jnp.stack([
            2 * (y.psi_q.dag() @ (L @ y.psi_q)).data.squeeze().real
            for L in L_of_t
        ])
        return {
            'log_p_p': y.log_p_p,
            'log_p_q': y.log_p_q,
            'exp_p': exp_p,
            'exp_q': exp_q
        }

    options = options.replace(save_extra=save_extra)

    def _solve_single(key):
        """Solve single trajectory."""
        # Select integrator based on method
        method_name = getattr(method, 'name', 'euler_maruyama')
        if method_name == "heun":
            IntegratorClass = DSSESolveHeunIntegrator
        elif method_name == "exponential_split":
            IntegratorClass = DSSESolveExponentialSplitIntegrator
        elif method_name == "strang_split":
            IntegratorClass = DSSESolveStrangSplitIntegrator
        else:
            IntegratorClass = DSSESolveEulerMayuramaIntegrator

        integrator = IntegratorClass(
            ts=tsave,
            y0=psi0,
            method=method,
            options=options,
            H=H_sys_t,
            H_con=H_con_t,
            Ls=Ls_t,
            Es=None,
            key=key,
            nsteps=nsteps,
            nsteps_per_save=nsteps_per_save
        )
        return integrator.run()

    # Vectorize over trajectories
    return jax.vmap(_solve_single)(keys)
