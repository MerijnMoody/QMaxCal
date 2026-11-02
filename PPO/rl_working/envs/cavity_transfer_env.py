"""
Cavity-assisted state transfer environment for PPO.

Two qubits coupled through a lossy cavity mode:
  H = g1(t)(σ+^(1) a + σ-^(1) a†) + g2(t)(σ+^(2) a + σ-^(2) a†)
  L = √κ a

Hilbert space: |qubit1, cavity, qubit2⟩ with dim = 2 × n_photon × 2

Task: Transfer excitation from qubit 1 to qubit 2: |1,0,0⟩ → |0,0,1⟩
The dark mode |D⟩ ∝ g2|1,0,0⟩ - g1|0,0,1⟩ has zero cavity photon.
"""
from __future__ import annotations

from functools import partial
from typing import Any, Dict, Optional, Tuple

import chex
import jax
import jax.numpy as jnp
from diffrax import Dopri5, PIDController, RESULTS
from flax import struct
from gymnax.environments import environment, spaces
from jax import lax
from qiskit_dynamics import Solver, Signal

from rl_working.envs.utils.signal_functions import gen_gauss_kernel
from rl_working.envs.utils.shared_quantum_functions import mixed_pure_state_fidelity


@struct.dataclass
class EnvParams:
    """Environment parameters for cavity transfer."""
    kappa: float = 5.0       # Cavity decay rate
    g0: float = 10.0         # Coupling strength
    n_photon: int = 3        # Photon number truncation
    t_total: float = 1.0
    min_action: float = -1.0
    max_action: float = 1.0


@struct.dataclass
class EnvState:
    """Environment state."""
    timestep: int
    reward: float
    fid: float
    cavity_pop_avg: float  # Average cavity photon number
    area_g1: float
    area_g2: float
    env_g1: jnp.ndarray
    env_g2: jnp.ndarray
    max_steps_reached: bool
    num_steps: int


class CavityTransferEnv(environment.Environment):
    """Cavity-assisted state transfer environment."""

    def __init__(
        self,
        kappa: float = 5.0,
        g0: float = 10.0,
        n_photon: int = 3,
        t_total: float = 1.0,
        n_action_steps: int = 50,
        log_fidelity: int = 1,
        fid_factor: float = 1.0,
        cavity_penalty: float = 0.1,  # Penalize cavity population
        pulse_area_penalty: float = 0.0,
        smoothness_penalty_env: float = 0.1,
        kernel_window: int = 25,
        kernel_std: float = 4.0,
        mxstep: int = 4000,
        mx_step_penalty: float = 25.0,
        fixed_endpoints: int = 1,
        dissipator: bool = True,
        **kwargs,
    ):
        super().__init__()

        self._kappa = kappa * 2 * jnp.pi
        self._g0 = g0 * 2 * jnp.pi
        self._n_photon = n_photon
        self._t_total = t_total
        self._n_steps = n_action_steps

        # Dimensions
        self.dim_q = 2
        self.dim_c = n_photon
        self.dim_total = self.dim_q * self.dim_c * self.dim_q

        # Data types
        self.float_dtype = jnp.float32
        self.complex_dtype = jnp.complex64

        # Build operators
        self._build_operators()

        # Time grid
        self.ts_action = jnp.linspace(0, t_total, n_action_steps, dtype=self.float_dtype)

        # Solver setup
        if dissipator:
            self.solver = Solver(
                hamiltonian_operators=[self.H_g1, self.H_g2],
                static_dissipators=self.c_ops,
                validate=False,
                array_library="jax",
            )
            self.yi = self.rho0
        else:
            self.solver = Solver(
                hamiltonian_operators=[self.H_g1, self.H_g2],
                validate=False,
                array_library="jax",
            )
            self.yi = self.psi0

        self.dissipator = dissipator
        self.mxstep = mxstep

        # Reward parameters
        self.log_fidelity = log_fidelity
        self.fid_factor = fid_factor
        self.cavity_penalty = cavity_penalty
        self.pulse_area_penalty = pulse_area_penalty
        self.smoothness_penalty_env = smoothness_penalty_env
        self.mx_step_penalty = mx_step_penalty
        self.fixed_endpoints = fixed_endpoints

        # Smoothing kernel
        self.gauss_kernel = gen_gauss_kernel(kernel_window, gauss_std=kernel_std)

        # Logging
        self.log_vals = {
            "fid": 0,
            "cavity-pop-avg": 0,
            "area-g1": 0,
            "area-g2": 0,
            "env-g1": jnp.zeros(self._n_steps),
            "env-g2": jnp.zeros(self._n_steps),
            "max-steps-reached": 0,
            "num-steps": 0,
        }

        self.default_params = EnvParams(
            kappa=kappa,
            g0=g0,
            n_photon=n_photon,
            t_total=t_total,
        )

    def _build_operators(self):
        """Build Hamiltonian and jump operators."""
        dim = self.dim_total
        dtype = self.complex_dtype

        # Identity operators
        I_q = jnp.eye(self.dim_q, dtype=dtype)
        I_c = jnp.eye(self.dim_c, dtype=dtype)

        # Qubit operators
        sigma_plus = jnp.array([[0, 1], [0, 0]], dtype=dtype)
        sigma_minus = jnp.array([[0, 0], [1, 0]], dtype=dtype)

        # Cavity annihilation
        a = jnp.diag(jnp.sqrt(jnp.arange(1, self.dim_c, dtype=jnp.float32)), k=1).astype(dtype)
        a_dag = a.conj().T

        def kron3(A, B, C):
            return jnp.kron(jnp.kron(A, B), C)

        # H_g1 = g0 * (σ+^(1) a + σ-^(1) a†) - couples qubit 1 to cavity
        self.H_g1 = self._g0 * (kron3(sigma_plus, a, I_q) + kron3(sigma_minus, a_dag, I_q))

        # H_g2 = g0 * (σ+^(2) a + σ-^(2) a†) - couples qubit 2 to cavity
        self.H_g2 = self._g0 * (kron3(I_q, a, sigma_plus) + kron3(I_q, a_dag, sigma_minus))

        # Cavity annihilation for full space
        self.a_full = kron3(I_q, a, I_q)

        # Jump operator: L = √κ a
        L = jnp.sqrt(self._kappa) * self.a_full
        self.c_ops = [L]

        # Initial state: |1,0,0⟩ (qubit 1 excited, cavity vacuum, qubit 2 ground)
        self.psi0 = jnp.zeros((dim, 1), dtype=dtype)
        self.psi0 = self.psi0.at[1, 0].set(1.0)
        self.rho0 = self.psi0 @ jnp.conj(self.psi0.T)

        # Target state: |0,0,1⟩
        self.psi_des = jnp.zeros((dim, 1), dtype=dtype)
        self.psi_des = self.psi_des.at[2 * self.dim_c, 0].set(1.0)

        # Cavity number operator: a†a
        self.a_dag_a = self.a_full.conj().T @ self.a_full

    @property
    def default_params(self) -> EnvParams:
        return self._default_params

    @default_params.setter
    def default_params(self, params: EnvParams):
        self._default_params = params

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: EnvState,
        action: jnp.ndarray,
        params: Optional[EnvParams] = None,
    ) -> Tuple[chex.Array, EnvState, float, bool, Dict[str, Any]]:
        if params is None:
            params = self.default_params

        sim_result = self.calc_step(action)
        reward = self._calc_reward(action, sim_result)
        rho_f, cavity_pop_avg, mx_reached, num_steps = sim_result

        # Process action for logging
        g1_raw = action[: self._n_steps]
        g2_raw = action[self._n_steps:]

        g1 = jnp.convolve(g1_raw, self.gauss_kernel, mode="same")
        g2 = jnp.convolve(g2_raw, self.gauss_kernel, mode="same")

        if self.fixed_endpoints:
            g1 = g1 * jnp.sin(jnp.pi * jnp.arange(self._n_steps) / (self._n_steps - 1)) ** 2
            g2 = g2 * jnp.sin(jnp.pi * jnp.arange(self._n_steps) / (self._n_steps - 1)) ** 2

        g1 = jnp.clip(g1 * 0.5 + 0.5, 0.0, 1.0) * self._g0
        g2 = jnp.clip(g2 * 0.5 + 0.5, 0.0, 1.0) * self._g0

        from jax.scipy.integrate import trapezoid
        area_g1 = trapezoid(g1, self.ts_action)
        area_g2 = trapezoid(g2, self.ts_action)

        # Fidelity
        if self.dissipator:
            max_fid = mixed_pure_state_fidelity(self.psi_des, rho_f)
        else:
            max_fid = jnp.abs(jnp.vdot(self.psi_des, rho_f)) ** 2

        new_state = EnvState(
            timestep=state.timestep + 1,
            reward=reward,
            fid=max_fid,
            cavity_pop_avg=cavity_pop_avg,
            area_g1=area_g1,
            area_g2=area_g2,
            env_g1=g1,
            env_g2=g2,
            max_steps_reached=mx_reached,
            num_steps=num_steps,
        )

        done = True
        info = self.get_info(new_state)

        return self.get_obs(params), new_state, reward, done, info

    def calc_step(self, action):
        """Run simulation with given action."""
        g1_raw = action[: self._n_steps]
        g2_raw = action[self._n_steps:]

        # Smooth
        g1 = jnp.convolve(g1_raw, self.gauss_kernel, mode="same")
        g2 = jnp.convolve(g2_raw, self.gauss_kernel, mode="same")

        # Fixed endpoints
        if self.fixed_endpoints:
            window = jnp.sin(jnp.pi * jnp.arange(self._n_steps) / (self._n_steps - 1)) ** 2
            g1 = g1 * window
            g2 = g2 * window

        # Scale to [0, 1]
        g1 = jnp.clip(g1 * 0.5 + 0.5, 0.0, 1.0)
        g2 = jnp.clip(g2 * 0.5 + 0.5, 0.0, 1.0)

        # Create signals using cubic interpolation (same as STIRAP env)
        from diffrax import CubicInterpolation, backward_hermite_coefficients
        g1_coeff = backward_hermite_coefficients(self.ts_action, g1)
        g2_coeff = backward_hermite_coefficients(self.ts_action, g2)
        g1_interp = CubicInterpolation(ts=self.ts_action, coeffs=g1_coeff)
        g2_interp = CubicInterpolation(ts=self.ts_action, coeffs=g2_coeff)

        def f_g1(t):
            return g1_interp.evaluate(t)

        def f_g2(t):
            return g2_interp.evaluate(t)

        sig_g1 = Signal(jnp.vectorize(f_g1))
        sig_g2 = Signal(jnp.vectorize(f_g2))

        stepsize_controller = PIDController(
            rtol=1e-5, atol=1e-7, dtmin=1e-10, dtmax=self._t_total / 10
        )

        sol = self.solver.solve(
            t_span=[self.ts_action[0], self.ts_action[-1]],
            signals=[sig_g1, sig_g2],
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
        rho_traj = sol.y
        rho_f = rho_traj[-1]

        # Average cavity photon number
        if self.dissipator:
            cavity_pops = jnp.array([jnp.real(jnp.trace(self.a_dag_a @ rho_traj[i]))
                                     for i in range(rho_traj.shape[0])])
        else:
            cavity_pops = jnp.array([jnp.real(jnp.vdot(rho_traj[i], self.a_dag_a @ rho_traj[i]))
                                     for i in range(rho_traj.shape[0])])
        cavity_pop_avg = jnp.mean(cavity_pops)

        return rho_f, cavity_pop_avg, max_steps_reached, num_steps

    def _calc_reward(self, action, sim_result):
        """Calculate reward."""
        rho_f, cavity_pop_avg, mx_reached, num_steps = sim_result

        # Fidelity
        if self.dissipator:
            max_fid = mixed_pure_state_fidelity(self.psi_des, rho_f)
        else:
            max_fid = jnp.abs(jnp.vdot(self.psi_des, rho_f)) ** 2

        if self.log_fidelity:
            fid_reward = jnp.log(1.0 / (1.0 - max_fid + 1e-12))
        else:
            fid_reward = max_fid

        # Pulse areas
        g1_raw = action[: self._n_steps]
        g2_raw = action[self._n_steps:]
        g1 = jnp.convolve(g1_raw, self.gauss_kernel, mode="same")
        g2 = jnp.convolve(g2_raw, self.gauss_kernel, mode="same")

        if self.fixed_endpoints:
            window = jnp.sin(jnp.pi * jnp.arange(self._n_steps) / (self._n_steps - 1)) ** 2
            g1 = g1 * window
            g2 = g2 * window

        g1 = jnp.clip(g1 * 0.5 + 0.5, 0.0, 1.0) * self._g0
        g2 = jnp.clip(g2 * 0.5 + 0.5, 0.0, 1.0) * self._g0

        from jax.scipy.integrate import trapezoid
        area_g1 = trapezoid(g1, self.ts_action)
        area_g2 = trapezoid(g2, self.ts_action)
        area_norm = (area_g1 + area_g2) / (2 * self._g0 * self._t_total)

        # Smoothness penalty
        def second_derivative_smoothness(signal, ts):
            d2 = jnp.diff(signal, n=2)
            dt = ts[1] - ts[0]
            return jnp.sum(d2 ** 2) / (dt ** 2)

        s_g1 = second_derivative_smoothness(g1, self.ts_action)
        s_g2 = second_derivative_smoothness(g2, self.ts_action)
        smooth_pen = (s_g1 + s_g2) / 1e6  # Normalize

        # Overflow handling
        def true_fn(_):
            return 1, 0.0, 0.0, 0.0

        def false_fn(_):
            return 0, fid_reward, max_fid, cavity_pop_avg

        mx_pen, fid_reward, max_fid, cavity_pop_avg = lax.cond(
            mx_reached, true_fn, false_fn, operand=None
        )

        reward = (
            self.fid_factor * fid_reward
            - self.cavity_penalty * cavity_pop_avg
            - self.pulse_area_penalty * area_norm
            - self.smoothness_penalty_env * smooth_pen
            - self.mx_step_penalty * mx_pen
        )

        return reward

    @partial(jax.jit, static_argnums=(0,))
    def reset(
        self, key: chex.PRNGKey, params: Optional[EnvParams] = None
    ) -> Tuple[chex.Array, EnvState]:
        if params is None:
            params = self.default_params

        state = EnvState(
            timestep=0,
            reward=0.0,
            fid=0.0,
            cavity_pop_avg=0.0,
            area_g1=0.0,
            area_g2=0.0,
            env_g1=jnp.zeros(self._n_steps),
            env_g2=jnp.zeros(self._n_steps),
            max_steps_reached=False,
            num_steps=0,
        )
        return self.get_obs(params), state

    def get_obs(self, params: Optional[EnvParams] = None) -> chex.Array:
        return jnp.zeros((1,), dtype=self.float_dtype)

    def get_info(self, env_state: EnvState) -> dict:
        return {
            "reward": env_state.reward,
            "fid": env_state.fid,
            "cavity-pop-avg": env_state.cavity_pop_avg,
            "area-g1": env_state.area_g1,
            "area-g2": env_state.area_g2,
            "env-g1": env_state.env_g1,
            "env-g2": env_state.env_g2,
            "max-steps-reached": env_state.max_steps_reached,
            "num-steps": env_state.num_steps,
            "timestep": env_state.timestep,
        }

    @property
    def name(self) -> str:
        return "Cavity-Assisted State Transfer"

    @property
    def num_actions(self) -> int:
        return int(2 * self._n_steps)

    def action_space(self, params: Optional[EnvParams] = None) -> spaces.Box:
        if params is None:
            params = self.default_params
        return spaces.Box(
            low=params.min_action,
            high=params.max_action,
            shape=(int(2 * self._n_steps),),
            dtype=self.float_dtype,
        )

    def observation_space(self, params: Optional[EnvParams] = None) -> spaces.Box:
        return spaces.Box(0, 1.0, shape=(1,), dtype=self.float_dtype)
