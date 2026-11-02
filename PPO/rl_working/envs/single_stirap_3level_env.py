"""3-level Λ STIRAP environment.

Simplified version of SimpleStirap for 3-level system:
- States: |g⟩ (ground), |e⟩ (excited), |f⟩ (target)
- Controls: pump envelope (g↔e) and Stokes envelope (e↔f)
- No detuning controls (resonant operation)
"""
from __future__ import annotations

from typing import Tuple, Optional, List
from scipy.signal.windows import blackman
import jax
import jax.numpy as jnp
from jax import lax
from jax.scipy.integrate import trapezoid
from gymnax.environments import spaces
from flax import struct
import chex
from diffrax import (backward_hermite_coefficients, CubicInterpolation,
                     Dopri5, PIDController, RESULTS)

from environment_template import SingleStepEnvironment
from utils.signal_functions import gen_gauss_kernel, SignalToolClass
from utils.shared_quantum_functions import (
    statevector_to_rho,
    pure_state_fidelity,
    mixed_pure_state_fidelity,
)
from hamiltonian.stirap_hamiltonian import get_hamiltonian_operators
from qiskit_dynamics import Solver, Signal

# Check if GPU is available
if "cuda" in str(jax.devices()):
    processor_array_type = "jax"
    processor = "gpu"
else:
    jax.config.update("jax_platform_name", "cpu")
    processor_array_type = "jax_sparse"
    jax.config.update("jax_enable_x64", True)
    processor = "cpu"


@struct.dataclass
class EnvState3Level:
    """Environment state for 3-level STIRAP."""
    reward: float
    fid: float
    x_state_pop_avg: float
    area_s: float
    area_p: float
    env_s: jnp.ndarray
    env_p: jnp.ndarray
    max_steps_reached: bool
    num_steps: int
    timestep: int


@struct.dataclass
class EnvParams3Level:
    """Environment parameters for 3-level STIRAP."""
    args: List
    tf: float = 1.0
    min_action: Optional[float] = -1.0
    max_action: Optional[float] = 1.0


class SimpleStirap3Level(SingleStepEnvironment):
    """3-level Λ STIRAP environment with pump and Stokes control."""

    def __init__(
            self,
            gamma: float,
            omega_0: float,
            mxstep: int,
            t_total: float = 1.0,
            x_penalty: float = 0.1,
            fid_factor: float = 1.0,
            log_fidelity: int = 0,
            pulse_area_penalty: float = 0.1,
            mx_step_penalty: float = 25,
            n_action_steps: int = 50,
            dissipator: bool = True,
            gaussian_kernel_window_length: int = 25,
            convolution_std: float = 1.0,
            **kwargs,  # Accept and ignore extra args from matched env
    ):
        """
        Args:
            gamma: Decay rate of excited state (MHz)
            omega_0: Maximum Rabi frequency (MHz)
            mxstep: Maximum ODE solver steps
            t_total: Total evolution time (μs)
            x_penalty: Penalty for excited state population
            fid_factor: Reward factor for fidelity
            log_fidelity: Use log(1/(1-F)) reward if 1, else linear F
            pulse_area_penalty: Penalty for pulse area
            mx_step_penalty: Penalty for solver overflow
            n_action_steps: Number of control timesteps
            dissipator: Include decay (Lindblad) if True
            gaussian_kernel_window_length: Smoothing kernel size
            convolution_std: Smoothing kernel std
        """
        # Convert to angular frequency
        self._gamma = gamma * 2 * jnp.pi
        self._omega_0 = omega_0 * 2 * jnp.pi
        self._n_steps = n_action_steps
        self._t_total = t_total
        self._resolution = 10
        self.mxstep = mxstep
        self.log_fidelity = log_fidelity
        self.mx_step_penalty = mx_step_penalty

        # dtype
        if processor == "gpu":
            self.float_dtype = jnp.float32
            self.complex_dtype = jnp.complex64
        else:
            self.float_dtype = jnp.float64
            self.complex_dtype = jnp.complex128

        # Time arrays
        self.sim_steps = self._n_steps * self._resolution
        self.ts_sim = jnp.linspace(0.0, self._t_total, self.sim_steps)
        self.ts_action = jnp.linspace(0.0, self._t_total, self._n_steps)

        # 3-level Hamiltonian
        self.ham_ops, self.c_ops = get_hamiltonian_operators(
            3, self._gamma, _dtype=self.complex_dtype)
        self.H_p, self.H_s = self.ham_ops[:]

        # Initial state: |g⟩
        self.psi0 = jnp.array([1, 0, 0], dtype=self.complex_dtype)
        self.rho0 = statevector_to_rho(self.psi0)

        # Target state: |f⟩
        self.psi_des = jnp.array([0, 0, 1], dtype=self.complex_dtype)
        self.rho_des = statevector_to_rho(self.psi_des)

        # Solver
        self.dissipator = dissipator
        if self.dissipator:
            self.solver = Solver(
                hamiltonian_operators=[self.H_p, self.H_s],
                static_dissipators=self.c_ops,
                validate=False,
                array_library=processor_array_type,
            )
            self.yi = self.rho0
        else:
            self.solver = Solver(
                hamiltonian_operators=[self.H_p, self.H_s],
                validate=False,
                array_library=processor_array_type,
            )
            self.yi = self.psi0

        # Reward parameters
        self.fid_factor = fid_factor
        self.x_penalty = x_penalty
        self.pulse_area_penalty = pulse_area_penalty

        # Smoothing
        self.gauss_kernel = gen_gauss_kernel(
            gaussian_kernel_window_length, gauss_std=convolution_std)
        self.bm_waveform = self._omega_0 * jnp.array(
            jnp.abs(blackman(self._n_steps)), dtype=self.float_dtype)
        self.signal_tools = SignalToolClass(
            self.gauss_kernel, self.gauss_kernel, self._t_total, self.bm_waveform)
        self.baseline_area = trapezoid(self.bm_waveform, self.ts_action)

        # Logging
        self.log_vals = {
            "fid": 0,
            "x-state-avg": 0,
            "area-s": 0,
            "area-p": 0,
            "env-s": jnp.zeros(self._n_steps),
            "env-p": jnp.zeros(self._n_steps),
            "max-steps-reached": 0,
            "num-steps": 0,
        }

    @property
    def default_params(self) -> EnvParams3Level:
        return EnvParams3Level(
            args=jnp.array([self._gamma, self._omega_0], dtype=self.float_dtype),
            tf=self._t_total,
        )

    def calc_step(self, key: chex.PRNGKey, action: chex.Array):
        """Run simulation with given control pulses."""
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
        rho_f = sol.y[-1]

        # Excited state population (state index 1)
        if self.dissipator:
            x_state_avg = jnp.mean(jnp.abs(sol.y[:, 1, 1]))
        else:
            x_state_avg = jnp.mean(jnp.abs(sol.y[:, 1])**2)

        return rho_f, x_state_avg, max_steps_reached, num_steps

    def _calc_reward(self, action, sim_result):
        """Calculate reward from simulation result."""
        rho_f, x_state_avg, mx_reached, num_steps = sim_result

        # Fidelity
        if self.dissipator:
            max_fid = mixed_pure_state_fidelity(self.psi_des, rho_f)
        else:
            max_fid = pure_state_fidelity(self.psi_des, rho_f)

        if self.log_fidelity:
            fid_reward = jnp.log(1.0 / (1.0 - max_fid + 1e-12))
        else:
            fid_reward = max_fid

        omega_p = action[0:self._n_steps]
        omega_s = action[self._n_steps:2 * self._n_steps]

        # Pulse areas
        area_p = trapezoid(jnp.abs(omega_p), self.ts_action)
        area_s = trapezoid(jnp.abs(omega_s), self.ts_action)
        fluence = trapezoid(omega_p**2 + omega_s**2, self.ts_action)

        # Handle solver overflow
        def true_fn(_):
            return 1, 0.0, 0.0, 0.0

        def false_fn(_):
            return 0, fid_reward, max_fid, x_state_avg

        mx_pen, fid_reward, max_fid, x_state_avg = lax.cond(
            mx_reached, true_fn, false_fn, operand=None)

        reward = (
            self.fid_factor * fid_reward
            - self.x_penalty * x_state_avg
            - self.pulse_area_penalty * (area_p + area_s) / (2 * self.baseline_area)
            - self.mx_step_penalty * mx_pen
        )

        return reward, max_fid, x_state_avg, fluence, mx_reached, num_steps

    def step_env(
        self, key: chex.PRNGKey, state: EnvState3Level, action: chex.Array,
        params: EnvParams3Level
    ) -> Tuple[chex.Array, EnvState3Level, chex.Array, bool, dict]:
        """Perform single episode."""
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
        reward, max_fid, x_state_avg, fluence, mx_reached, num_steps = self._calc_reward(
            action_scaled, sim_result)

        # Update state
        area_p = trapezoid(jnp.abs(omega_p), self.ts_action)
        area_s = trapezoid(jnp.abs(omega_s), self.ts_action)

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

        return (
            lax.stop_gradient(self.get_obs()),
            lax.stop_gradient(updated_state),
            reward,
            True,  # Single-step env, always done
            lax.stop_gradient(self.get_info(updated_state)),
        )

    def reset(self, key: chex.PRNGKey,
              params: EnvParams3Level) -> Tuple[chex.Array, EnvState3Level]:
        state = EnvState3Level(
            reward=0.0,
            fid=0.0,
            x_state_pop_avg=0.0,
            area_s=0.0,
            area_p=0.0,
            env_s=jnp.zeros(self._n_steps, dtype=self.float_dtype),
            env_p=jnp.zeros(self._n_steps, dtype=self.float_dtype),
            max_steps_reached=False,
            num_steps=0,
            timestep=0,
        )
        return self.get_obs(params), state

    def get_obs(self, params: Optional[EnvParams3Level] = None) -> chex.Array:
        return jnp.zeros((1,), dtype=self.float_dtype)

    def get_info(self, env_state: EnvState3Level) -> dict:
        return {
            "reward": env_state.reward,
            "fid": env_state.fid,
            "x-state-avg": env_state.x_state_pop_avg,
            "area-s": env_state.area_s,
            "area-p": env_state.area_p,
            "env-s": env_state.env_s,
            "env-p": env_state.env_p,
            "max-steps-reached": env_state.max_steps_reached,
            "num-steps": env_state.num_steps,
            "timestep": env_state.timestep,
        }

    @property
    def name(self) -> str:
        return "3-level Lambda STIRAP"

    @property
    def num_actions(self, params: Optional[EnvParams3Level] = None) -> int:
        return int(2 * self._n_steps)

    def action_space(self, params: Optional[EnvParams3Level] = None) -> spaces.Box:
        if params is None:
            params = self.default_params
        return spaces.Box(
            low=params.min_action,
            high=params.max_action,
            shape=(int(2 * self._n_steps),),
            dtype=self.float_dtype,
        )

    def observation_space(self, params: Optional[EnvParams3Level] = None) -> spaces.Box:
        return spaces.Box(0, 1.0, shape=(1,), dtype=self.float_dtype)
