"""3-level Λ STIRAP env with drift variance penalty.

Computes the temporal variance of the mean drift:
    ᾱ(t) = Tr[(L + L†)ρ(t)]
    DriftVar = (1/τ)∫ᾱ(t)² dt - ((1/τ)∫ᾱ(t) dt)²

This is exactly computable from the Lindblad equation (no SSE needed).
"""
from __future__ import annotations

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import jax.numpy as jnp
from jax import lax
from jax.scipy.integrate import trapezoid
from scipy.signal.windows import blackman
import chex

from single_stirap_3level_env import SimpleStirap3Level
from utils.signal_functions import gen_gauss_kernel, SignalToolClass
from diffrax import (backward_hermite_coefficients, CubicInterpolation,
                     Dopri5, PIDController, RESULTS)
from qiskit_dynamics import Signal


class SimpleStirap3LevelDriftVar(SimpleStirap3Level):
    """3-level Λ STIRAP with drift variance penalty."""

    def __init__(
        self,
        log_fidelity: int = 1,
        fid_factor: float = 1.0,
        x_penalty: float = 0.1,
        pulse_area_penalty: float = 0.0,
        smoothness_penalty_env: float = 0.1,
        drift_var_penalty: float = 0.0,  # New: drift variance weight
        kernel_window: int = 25,
        kernel_std: float = 4.0,
        **kwargs,
    ):
        # Pop conflicting kwargs before passing to parent
        kwargs.pop("log_fidelity", None)
        kwargs.pop("fid_factor", None)
        kwargs.pop("x_penalty", None)
        kwargs.pop("pulse_area_penalty", None)
        super().__init__(
            fid_factor=fid_factor,
            x_penalty=x_penalty,
            pulse_area_penalty=pulse_area_penalty,
            log_fidelity=log_fidelity,
            **kwargs,
        )
        self.smoothness_penalty_env = float(smoothness_penalty_env)
        self.drift_var_penalty = float(drift_var_penalty)

        # Blackman baseline
        bm = jnp.asarray(jnp.abs(blackman(self._n_steps)),
                         dtype=self.float_dtype)
        self.bm_waveform = self._omega_0 * bm
        self.baseline_area = trapezoid(self.bm_waveform, self.ts_action)

        # Gaussian smoothing kernel
        gkernel = gen_gauss_kernel(kernel_window, gauss_std=kernel_std)
        self.signal_tools = SignalToolClass(
            gkernel, gkernel, self._t_total, self.bm_waveform
        )
        self.baseline_smoothness = self.signal_tools.get_baseline_smoothness(
            self.bm_waveform, method="second_derivative",
        )

        # Precompute L + L† for drift computation
        # L = √γ |g⟩⟨e| = √γ |0⟩⟨1|
        # L + L† = √γ (|0⟩⟨1| + |1⟩⟨0|)
        L_plus_Ldag = jnp.zeros((3, 3), dtype=self.complex_dtype)
        L_plus_Ldag = L_plus_Ldag.at[0, 1].set(jnp.sqrt(self._gamma))
        L_plus_Ldag = L_plus_Ldag.at[1, 0].set(jnp.sqrt(self._gamma))
        self.L_plus_Ldag = L_plus_Ldag

        # Override log_vals to include drift_var
        self.log_vals = {
            "fid": 0,
            "x-state-avg": 0,
            "area-s": 0,
            "area-p": 0,
            "env-s": jnp.zeros(self._n_steps),
            "env-p": jnp.zeros(self._n_steps),
            "max-steps-reached": 0,
            "num-steps": 0,
            "drift-var": 0,  # New: drift variance logging
        }

    def calc_step(self, key: chex.PRNGKey, action: chex.Array):
        """Run simulation and return full trajectory for drift variance."""
        omega_p = action[0:self._n_steps]
        omega_s = action[self._n_steps:2 * self._n_steps]

        # Cubic interpolation
        env_pump_coeff = backward_hermite_coefficients(self.ts_action, omega_p)
        env_stokes_coeff = backward_hermite_coefficients(self.ts_action, omega_s)
        env_pump = CubicInterpolation(ts=self.ts_action, coeffs=env_pump_coeff)
        env_stokes = CubicInterpolation(ts=self.ts_action, coeffs=env_stokes_coeff)

        def f_pump(t):
            return env_pump.evaluate(t)

        def f_stokes(t):
            return env_stokes.evaluate(t)

        signal_list = [Signal(jnp.vectorize(f_pump)), Signal(jnp.vectorize(f_stokes))]

        stepsize_controller = PIDController(rtol=1e-8, atol=1e-6)
        sol = self.solver.solve(
            t_span=[self.ts_sim[0], self.ts_sim[-1]],
            signals=signal_list,
            y0=self.yi,
            t_eval=self.ts_action,
            convert_results=False,
            method=Dopri5(),
            max_steps=self.mxstep,
            throw=False,
            stepsize_controller=stepsize_controller,
        )

        max_steps_reached = (sol.result == RESULTS.max_steps_reached)
        num_steps = sol.stats["num_steps"]
        rho_traj = sol.y  # Full trajectory: (n_steps, 3, 3)
        rho_f = rho_traj[-1]

        # Excited state population average
        x_state_avg = jnp.mean(jnp.abs(rho_traj[:, 1, 1]))

        # Compute temporal drift variance
        # ᾱ(t) = Tr[(L + L†)ρ(t)] = √γ (ρ₀₁ + ρ₁₀) = 2√γ Re[ρ₀₁]
        # Vectorized: Tr[A @ B] = sum_ij A_ij * B_ji
        # einsum 'ij,tji->t' computes trace for all timesteps at once
        alpha_t = jnp.einsum('ij,tji->t', self.L_plus_Ldag, rho_traj).real

        # Temporal variance: Var_t[ᾱ] = E[ᾱ²] - E[ᾱ]²
        alpha_mean = trapezoid(alpha_t, self.ts_action) / self._t_total
        alpha_sq_mean = trapezoid(alpha_t**2, self.ts_action) / self._t_total
        drift_var = alpha_sq_mean - alpha_mean**2

        return rho_f, x_state_avg, max_steps_reached, num_steps, drift_var

    def _calc_reward(self, action, sim_result):
        rho_f, x_state_avg, mx_reached, num_steps, drift_var = sim_result

        # Fidelity
        from utils.shared_quantum_functions import mixed_pure_state_fidelity
        max_fid = mixed_pure_state_fidelity(self.psi_des, rho_f)
        if self.log_fidelity:
            fid_reward = jnp.log(1.0 / (1.0 - max_fid + 1e-12))
        else:
            fid_reward = max_fid

        omega_ge = action[0:self._n_steps]
        omega_ef = action[self._n_steps:2 * self._n_steps]

        # Pulse area (Blackman-normalized)
        area_ge = trapezoid(jnp.abs(omega_ge) * self._omega_0, self.ts_action)
        area_ef = trapezoid(jnp.abs(omega_ef) * self._omega_0, self.ts_action)
        area_norm = 0.5 * (area_ge / self.baseline_area
                           + area_ef / self.baseline_area)

        # Fluence (for logging)
        fluence = trapezoid(omega_ge ** 2 + omega_ef ** 2, self.ts_action)

        # Smoothness penalty
        s_ge = self.signal_tools.calculate_smoothness(
            omega_ge, method="second_derivative"
        )
        s_ef = self.signal_tools.calculate_smoothness(
            omega_ef, method="second_derivative"
        )
        relu = lambda x: jnp.maximum(0.0, x)
        smooth_pen = relu(
            (s_ge + s_ef) / (2.0 * self.baseline_smoothness) - 1.0
        )

        # Solver overflow handling
        def true_fn(_):
            return 1, 0.0, 0.0, 0.0, 0.0

        def false_fn(_):
            return 0, fid_reward, max_fid, x_state_avg, drift_var

        mx_pen, fid_reward, max_fid, x_state_avg, drift_var = lax.cond(
            mx_reached, true_fn, false_fn, operand=None
        )

        reward = (
            self.fid_factor * fid_reward
            - self.x_penalty * x_state_avg
            - self.pulse_area_penalty * area_norm
            - self.smoothness_penalty_env * smooth_pen
            - self.drift_var_penalty * drift_var
            - self.mx_step_penalty * mx_pen
        )
        return reward, max_fid, x_state_avg, fluence, mx_reached, num_steps, drift_var

    def step_env(self, key, state, action, params):
        """Perform single episode with drift variance tracking."""
        new_timestep = state.timestep + 1

        # Process actions
        omega_p = action[0:self._n_steps]
        omega_s = action[self._n_steps:2 * self._n_steps]

        # Scale from [-1,1] to [0,1] and clip
        omega_p = jnp.clip(omega_p * 0.5 + 0.5, 0.0, 1.0)
        omega_s = jnp.clip(omega_s * 0.5 + 0.5, 0.0, 1.0)

        # Smooth
        omega_p = self.signal_tools.drive_smoother(omega_p.astype(self.float_dtype))
        omega_s = self.signal_tools.drive_smoother(omega_s.astype(self.float_dtype))

        # Scale by omega_0
        omega_p = omega_p * self._omega_0
        omega_s = omega_s * self._omega_0

        action_scaled = jnp.concatenate([omega_p, omega_s])

        # Run simulation
        sim_result = self.calc_step(key, action_scaled)
        reward, max_fid, x_state_avg, fluence, mx_reached, num_steps, drift_var = \
            self._calc_reward(action_scaled, sim_result)

        # Update state
        area_p = trapezoid(jnp.abs(omega_p), self.ts_action)
        area_s = trapezoid(jnp.abs(omega_s), self.ts_action)

        from single_stirap_3level_env import EnvState3Level
        updated_state = EnvState3Level(
            reward=reward,
            fid=max_fid,
            x_state_pop_avg=x_state_avg,
            area_s=area_s,
            area_p=area_p,
            env_s=omega_s.astype(self.float_dtype),
            env_p=omega_p.astype(self.float_dtype),
            max_steps_reached=mx_reached,
            num_steps=num_steps,
            timestep=new_timestep,
        )

        # Add drift_var to info dict (use dash to match log_vals)
        info = self.get_info(updated_state)
        info["drift-var"] = drift_var

        return (
            lax.stop_gradient(self.get_obs()),
            lax.stop_gradient(updated_state),
            reward,
            True,
            lax.stop_gradient(info),
        )
