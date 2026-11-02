"""3-level Λ STIRAP env with reward MATCHED to the original 4/5-level PPO setup.

Identical physics to SimpleStirap3Level, but the reward shape is the one
the original PPO paper actually uses (SimpleStirap):

    R = c_F · log(1/(1−F))                                # diverges at F→1
        − c_x · ⟨P_|e⟩⟩_avg
        − α · ½(A_s/A_0 + A_p/A_0)                         # Blackman-normalized
        − c_s · ReLU((S_s + S_p)/(2·S_0) − 1)              # envelope smoothness
        − c_mx · 1[solver overflow]

This removes the unfair handicap we gave the PPO baseline by using the 3-level env's
default linear reward (log_fidelity=0, no smoothness, quadratic fluence).

Defaults mirror SimpleStirap in single_stirap_env.py.
"""
from __future__ import annotations

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import jax.numpy as jnp
from jax import lax
from jax.scipy.integrate import trapezoid
from scipy.signal.windows import blackman

from single_stirap_3level_env import SimpleStirap3Level
from utils.signal_functions import gen_gauss_kernel, SignalToolClass


class SimpleStirap3LevelMatched(SimpleStirap3Level):
    """3-level Λ STIRAP with the original 4/5-level reward shape."""

    def __init__(
        self,
        # inherited / new reward weights — defaults from SimpleStirap
        log_fidelity: int = 1,
        fid_factor: float = 1.0,
        x_penalty: float = 0.1,
        pulse_area_penalty: float = 0.1,
        smoothness_penalty_env: float = 0.1,
        kernel_window: int = 25,
        kernel_std: float = 4.0,
        **kwargs,
    ):
        # Force the base class to use linear fid (we'll override with log here)
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

        # Blackman baseline (same as SimpleStirap)
        bm = jnp.asarray(jnp.abs(blackman(self._n_steps)),
                         dtype=self.float_dtype)
        self.bm_waveform = self._omega_0 * bm
        self.baseline_area = trapezoid(self.bm_waveform, self.ts_action)

        # Gaussian-smoothed Blackman → baseline smoothness
        gkernel = gen_gauss_kernel(kernel_window, gauss_std=kernel_std)
        self.signal_tools = SignalToolClass(
            gkernel, gkernel, self._t_total, self.bm_waveform
        )
        self.baseline_smoothness = self.signal_tools.get_baseline_smoothness(
            self.bm_waveform, method="second_derivative",
        )

    def _calc_reward(self, action, sim_result):
        rho_f, x_state_avg, mx_reached, num_steps = sim_result

        # fidelity
        from utils.shared_quantum_functions import mixed_pure_state_fidelity
        max_fid = mixed_pure_state_fidelity(self.psi_des, rho_f)
        if self.log_fidelity:
            fid_reward = jnp.log(1.0 / (1.0 - max_fid + 1e-12))
        else:
            fid_reward = max_fid

        omega_ge = action[0:self._n_steps]
        omega_ef = action[self._n_steps:2 * self._n_steps]

        # Original-paper style: LINEAR area normalized by Blackman baseline.
        # Note: |omega_*| because Blackman is positive but raw actions can swing.
        area_ge = trapezoid(jnp.abs(omega_ge) * self._omega_0, self.ts_action)
        area_ef = trapezoid(jnp.abs(omega_ef) * self._omega_0, self.ts_action)
        area_norm = 0.5 * (area_ge / self.baseline_area
                           + area_ef / self.baseline_area)

        # Quadratic fluence (kept for logging consistency with previous runs)
        fluence = trapezoid(omega_ge ** 2 + omega_ef ** 2, self.ts_action)

        # Smoothness (second-derivative) normalized to Blackman baseline
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

        # Solver-overflow zeroing (mirrors SimpleStirap3Level)
        def true_fn(_):
            return 1, 0.0, 0.0, 0.0

        def false_fn(_):
            return 0, fid_reward, max_fid, x_state_avg

        mx_pen, fid_reward, max_fid, x_state_avg = lax.cond(
            mx_reached, true_fn, false_fn, operand=None
        )

        reward = (
            self.fid_factor * fid_reward
            - self.x_penalty * x_state_avg
            - self.pulse_area_penalty * area_norm
            - self.smoothness_penalty_env * smooth_pen
            - self.mx_step_penalty * mx_pen
        )
        return reward, max_fid, x_state_avg, fluence, mx_reached, num_steps
