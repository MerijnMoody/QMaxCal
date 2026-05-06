#!/usr/bin/env python3
"""
Generic Ernst-style PPO trainer. Takes a system module path and trains a
Gaussian policy that outputs piecewise-constant controls (smoothed + scaled)
feeding a qiskit_dynamics Lindblad ODE solver.

After training, runs one deterministic rollout with the learned policy and
saves a snap_*.pkl with the same schema as train_qmaxcal.py so plot_snapshots.py
can visualise both.

Example:
    python train_ppo.py --system systems.cavity_transfer --num_updates 500 \
        --output_dir results/ppo_test
"""
from __future__ import annotations

import os, sys, pickle, argparse, importlib, time
from functools import partial
from typing import Any, Dict, Optional, Tuple

# Both sys.paths required: Ernst/ for ppo + envs, template/ for systems/
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)
sys.path.insert(0, '/home/mmoody/qmaxcal_paper/Ernst')

import chex
import jax
import jax.numpy as jnp
import numpy as np
from diffrax import Dopri5, PIDController, RESULTS, CubicInterpolation, backward_hermite_coefficients
from flax import struct
from gymnax.environments import environment, spaces
from qiskit_dynamics import Solver, Signal

import rl_working.ppo as ppo
from rl_working.envs.utils.signal_functions import gen_gauss_kernel
from rl_working.envs.utils.shared_quantum_functions import mixed_pure_state_fidelity


# ============================================================
# Generic system env
# ============================================================

@struct.dataclass
class EnvParams:
    T_final: float = 1.0


@struct.dataclass
class EnvState:
    timestep:  int
    reward:    float
    fid:       float
    pop_avg:   jnp.ndarray   # (n_pops,)  average of each population over episode
    pulse_rms: jnp.ndarray   # (n_controls,)
    env_u:     jnp.ndarray   # (n_controls, n_action_steps) — post-processed signals
    max_steps_reached: bool
    num_steps: int


class SystemEnv(environment.Environment):
    """Generic Ernst-style env that reads a system dict."""

    def __init__(self, system_mod_path: str, system_kwargs: dict,
                 n_action_steps: int = 50,
                 log_fidelity: int = 1,
                 fid_factor: float = 1.0,
                 pulse_area_penalty: float = 0.0,
                 fluence_penalty: float = 0.0,
                 smoothness_penalty_env: float = 0.1,
                 kernel_window: int = 25,
                 kernel_std: float = 4.0,
                 mxstep: int = 4000,
                 mx_step_penalty: float = 25.0,
                 fixed_endpoints: int = 1,
                 **_kwargs):
        super().__init__()

        sys_mod = importlib.import_module(system_mod_path)
        sysdict = sys_mod.create_system(**system_kwargs)

        # Quantum operators (plain jax arrays, complex64 for qiskit_dynamics)
        self.H_ops   = [jnp.asarray(H, dtype=jnp.complex64) for H in sysdict['H_ops']]
        self.c_ops   = [jnp.asarray(L, dtype=jnp.complex64) for L in sysdict['jump_ops']]
        psi0         = jnp.asarray(sysdict['psi0'],         dtype=jnp.complex64)
        self.psi0    = psi0
        self.rho0    = psi0 @ jnp.conj(psi0.T)
        self.target  = jnp.asarray(sysdict['target_state'], dtype=jnp.complex64)

        self.T_final       = float(sysdict['T_final'])
        self.n_controls    = int(sysdict['n_controls'])
        self.n_dim         = int(sysdict['n_dim'])
        self.signal_scale  = float(sysdict.get('signal_scale', 1.0))
        self.control_labels = list(sysdict.get('control_labels',
                                                [f'u_{i}' for i in range(self.n_controls)]))

        # action_range: 'signed' | 'unsigned' | list[str]
        ar = sysdict.get('action_range', 'signed')
        if isinstance(ar, str):
            ar = [ar] * self.n_controls
        self.action_range_per_ch = ar  # list of 'signed'/'unsigned'

        # Populations (complex64 projectors)
        pop_items = list(sysdict['populations'].items())
        self.pop_names = [n for n, _ in pop_items]
        self.pop_ops   = jnp.stack([jnp.asarray(P, dtype=jnp.complex64)
                                     for _, P in pop_items])   # (n_pops, d, d)

        self._n_steps = n_action_steps
        self.ts_action = jnp.linspace(0, self.T_final, n_action_steps, dtype=jnp.float32)

        # Reward / shaping params
        self.log_fidelity            = log_fidelity
        self.fid_factor              = fid_factor
        self.pulse_area_penalty      = pulse_area_penalty
        self.fluence_penalty         = fluence_penalty
        self.smoothness_penalty_env  = smoothness_penalty_env
        self.mxstep                  = mxstep
        self.mx_step_penalty         = mx_step_penalty
        self.fixed_endpoints         = fixed_endpoints
        self.gauss_kernel            = gen_gauss_kernel(kernel_window, gauss_std=kernel_std)

        # log_vals declares which scalar/array fields LogWrapper accumulates per episode.
        # Keys MUST match entries in the info dict returned by step().
        self.log_vals = {
            'fid': 0.0,
            'pop_avg':  jnp.zeros(len(self.pop_names)),
            'pulse_rms': jnp.zeros(self.n_controls),
            'env_u':    jnp.zeros((self.n_controls, self._n_steps)),
            'pulse_area': 0.0,
            'fluence_sq': 0.0,
            'max_steps_reached': 0.0,
            'num_steps': 0,
        }

        # Lindblad solver
        self.solver = Solver(
            hamiltonian_operators=self.H_ops,
            static_dissipators=self.c_ops,
            validate=False,
            array_library='jax',
        )

        self._default_params = EnvParams(T_final=self.T_final)

    # -----------------------------------------
    @property
    def default_params(self) -> EnvParams:
        return self._default_params

    @property
    def name(self) -> str:
        return 'system_env'

    def action_space(self, params):
        return spaces.Box(low=-1.0, high=1.0,
                          shape=(self.n_controls * self._n_steps,),
                          dtype=jnp.float32)

    def observation_space(self, params):
        return spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=jnp.float32)

    def state_space(self, params):
        return self.observation_space(params)

    def get_obs(self, params):
        return jnp.zeros((1,), dtype=jnp.float32)

    def is_terminal(self, state, params):
        return True

    # -----------------------------------------
    def _postprocess_channel(self, raw, ch_idx):
        """Smooth → optional fixed-endpoint window → scale per action_range."""
        u = jnp.convolve(raw, self.gauss_kernel, mode='same')
        if self.fixed_endpoints:
            window = jnp.sin(jnp.pi * jnp.arange(self._n_steps) / (self._n_steps - 1)) ** 2
            u = u * window
        # u is in (approximately) [-1, 1]
        if self.action_range_per_ch[ch_idx] == 'unsigned':
            u = jnp.clip(0.5 * u + 0.5, 0.0, 1.0)
        else:  # signed
            u = jnp.clip(u, -1.0, 1.0)
        return u * self.signal_scale

    def _split_action(self, action):
        return action.reshape(self.n_controls, self._n_steps)

    def _postprocess(self, action):
        raw = self._split_action(action)
        return jnp.stack([self._postprocess_channel(raw[i], i)
                          for i in range(self.n_controls)])   # (n_controls, n_steps)

    # -----------------------------------------
    def _solve(self, u_stack):
        """Run Lindblad ODE with stacked post-processed control signals.

        Returns (rho_traj (T, d, d), max_steps_reached, num_steps).
        """
        signals = []
        for i in range(self.n_controls):
            ui = u_stack[i]
            coeff  = backward_hermite_coefficients(self.ts_action, ui)
            interp = CubicInterpolation(ts=self.ts_action, coeffs=coeff)
            def f(t, _interp=interp):
                return _interp.evaluate(t)
            signals.append(Signal(jnp.vectorize(f)))

        stepsize_controller = PIDController(
            rtol=1e-5, atol=1e-7, dtmin=1e-10, dtmax=self.T_final / 10
        )
        sol = self.solver.solve(
            t_span=[self.ts_action[0], self.ts_action[-1]],
            signals=signals,
            y0=self.rho0,
            t_eval=self.ts_action,
            convert_results=False,
            method=Dopri5(),
            max_steps=self.mxstep,
            throw=False,
            stepsize_controller=stepsize_controller,
        )
        return sol.y, sol.result == RESULTS.max_steps_reached, sol.stats['num_steps']

    def _populations_of(self, rho_traj):
        """Return (n_pops, T) population trajectory from rho_traj (T, d, d)."""
        # trace(P @ rho)  for each t, each P
        return jnp.real(jnp.einsum('pij,tji->pt', self.pop_ops, rho_traj))

    # -----------------------------------------
    @partial(jax.jit, static_argnums=(0,))
    def step(self, key: chex.PRNGKey, state: EnvState, action: jnp.ndarray,
             params: Optional[EnvParams] = None):
        if params is None:
            params = self.default_params

        u_stack = self._postprocess(action)                  # (n_ctrls, T)
        rho_traj, max_reached, num_steps = self._solve(u_stack)
        rho_f = rho_traj[-1]

        fid = mixed_pure_state_fidelity(self.target, rho_f)

        # Reward
        r = self.fid_factor * (jnp.log(fid + 1e-8) if self.log_fidelity else fid)

        # Pulse-area penalty (linear:  Σ_i ∫|u_i|dt)
        from jax.scipy.integrate import trapezoid
        pulse_area = jnp.sum(jnp.stack([trapezoid(jnp.abs(u_stack[i]), self.ts_action)
                                         for i in range(self.n_controls)]))
        r = r - self.pulse_area_penalty * pulse_area

        # Squared fluence penalty (matches QMaxCal "fluence" metric:  Σ_i ∫u_i²dt)
        fluence_sq = jnp.sum(jnp.stack([trapezoid(u_stack[i] ** 2, self.ts_action)
                                          for i in range(self.n_controls)]))
        r = r - self.fluence_penalty * fluence_sq

        # Smoothness penalty (2nd derivative magnitude)
        sm = jnp.sum(jnp.abs(jnp.diff(u_stack, n=2, axis=-1)))
        r = r - self.smoothness_penalty_env * sm

        # Max-steps penalty
        r = r - self.mx_step_penalty * jnp.where(max_reached, 1.0, 0.0)

        # Average population over time for logging
        pop_traj = self._populations_of(rho_traj)            # (n_pops, T)
        pop_avg  = jnp.mean(pop_traj, axis=-1)
        pulse_rms = jnp.sqrt(jnp.mean(u_stack ** 2, axis=-1))

        new_state = EnvState(
            timestep=state.timestep + 1, reward=r, fid=fid,
            pop_avg=pop_avg, pulse_rms=pulse_rms, env_u=u_stack,
            max_steps_reached=max_reached, num_steps=num_steps,
        )
        info = {
            'fid': fid, 'reward': r,
            'pop_avg': pop_avg, 'pulse_rms': pulse_rms,
            'env_u': u_stack,
            'pulse_area': pulse_area, 'fluence_sq': fluence_sq,
            'max_steps_reached': jnp.where(max_reached, 1.0, 0.0),
            'num_steps': num_steps,
            'returned_episode_returns': r, 'returned_episode_lengths': 1,
            'returned_episode': True,
        }
        return self.get_obs(params), new_state, r, True, info

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key, params=None):
        if params is None:
            params = self.default_params
        state = EnvState(
            timestep=0, reward=0.0, fid=0.0,
            pop_avg=jnp.zeros(len(self.pop_names)),
            pulse_rms=jnp.zeros(self.n_controls),
            env_u=jnp.zeros((self.n_controls, self._n_steps)),
            max_steps_reached=False, num_steps=0,
        )
        return self.get_obs(params), state


# ============================================================
# Snapshot: run deterministic policy rollout + save QMaxCal-schema snap
# ============================================================

def rollout_and_save_snap(env: SystemEnv, params, config, output_path: str,
                           step_label: int, system_name: str):
    """Evaluate policy deterministically (action = π mean), compute populations
    over time via Lindblad, save snap with QMaxCal schema."""
    # Rebuild network to get deterministic action
    import flax.linen as nn
    network = ppo.CombinedActorCritic(
        env.action_space(env.default_params).shape[0],
        activation=config['ACTIVATION'],
        layer_size=config['LAYER_SIZE'],
    )
    obs = jnp.zeros(env.observation_space(env.default_params).shape)
    pi, _ = network.apply(params, obs)
    action = pi.mode()   # deterministic

    u_stack = env._postprocess(action)
    rho_traj, max_reached, _num_steps = env._solve(u_stack)

    # populations (n_pops, T)
    pop_traj = env._populations_of(rho_traj)
    pop_names = env.pop_names

    # Fidelity
    fid = float(mixed_pure_state_fidelity(env.target, rho_traj[-1]))

    # Pulse-area and squared fluence (deterministic, on the eval rollout)
    from jax.scipy.integrate import trapezoid
    pulse_area = float(jnp.sum(jnp.stack([
        trapezoid(jnp.abs(u_stack[i]), env.ts_action)
        for i in range(env.n_controls)])))
    fluence_sq = float(jnp.sum(jnp.stack([
        trapezoid(u_stack[i] ** 2, env.ts_action)
        for i in range(env.n_controls)])))

    # Build snap
    snap = dict(
        step=step_label, f_exact=fid,
        kl_cost=0.0, drift_var_cost=0.0, fluence=fluence_sq,
        pulse_area=pulse_area,
        t_traj=np.asarray(env.ts_action),
        populations={n: np.asarray(pop_traj[i]) for i, n in enumerate(pop_names)},
        t_ctrl=np.asarray(env.ts_action),
        control_signals=np.asarray(u_stack.T),              # (T, n_controls)
        control_labels=env.control_labels,
        system_name=system_name,
        method='ppo',
    )
    with open(output_path, 'wb') as f:
        pickle.dump(snap, f)
    print(f"  [snap] saved {output_path}  F={fid:.4f}  "
          f"pulse_area={pulse_area:.3f}  fluence²={fluence_sq:.3f}  "
          f"max_steps_reached={bool(max_reached)}")


# ============================================================
# Driver
# ============================================================

def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--system', type=str, required=True)
    pre_args, _ = pre.parse_known_args()
    sys_mod = importlib.import_module(pre_args.system)

    p = argparse.ArgumentParser(parents=[pre])
    p.add_argument('--output_dir',      type=str,   default='results_ppo')
    p.add_argument('--seed',            type=int,   default=0)
    # PPO hyperparams
    p.add_argument('--lr',              type=float, default=5e-4)
    p.add_argument('--num_envs',        type=int,   default=16)
    p.add_argument('--num_updates',     type=int,   default=1000)
    p.add_argument('--num_minibatches', type=int,   default=4)
    p.add_argument('--clip_eps',        type=float, default=0.2)
    p.add_argument('--ent_coef',        type=float, default=0.0)
    p.add_argument('--max_grad_norm',   type=float, default=0.5)
    p.add_argument('--layer_size',      type=int,   default=256)
    p.add_argument('--anneal_lr',       type=int,   default=0)
    p.add_argument('--log_freq',        type=int,   default=25)
    p.add_argument('--activation',      type=str,   default='relu6')
    # Env shaping
    p.add_argument('--n_action_steps',  type=int,   default=50)
    p.add_argument('--log_fidelity',    type=int,   default=1)
    p.add_argument('--fid_factor',      type=float, default=1.0)
    p.add_argument('--pulse_area_penalty',     type=float, default=0.0,
                   help='Linear penalty on Σ_i ∫|u_i|dt')
    p.add_argument('--fluence_penalty',        type=float, default=0.0,
                   help='Squared-fluence penalty Σ_i ∫u_i²dt (matches QMaxCal "fluence")')
    p.add_argument('--smoothness_penalty_env', type=float, default=0.1)
    p.add_argument('--kernel_window',   type=int,   default=25)
    p.add_argument('--kernel_std',      type=float, default=4.0)
    p.add_argument('--mxstep',          type=int,   default=4000)
    p.add_argument('--mx_step_penalty', type=float, default=25.0)
    p.add_argument('--fixed_endpoints', type=int,   default=1)
    # Chunked training + intermediate snapshots
    p.add_argument('--snap_every',      type=int,   default=0,
                   help='Save a snapshot every N PPO updates (0 = save only at end). '
                        'When >0, training is broken into chunks of this size and '
                        'a deterministic-rollout snap is saved after each chunk.')
    p.add_argument('--max_wall_time_seconds', type=float, default=0.0,
                   help='Stop training early once the elapsed wall time exceeds this '
                        'threshold (still saves a final snapshot). 0 disables.')

    sys_mod.add_system_args(p)
    args = p.parse_args()
    assert args.num_envs % args.num_minibatches == 0

    generic = {'system', 'output_dir', 'seed', 'lr', 'num_envs', 'num_updates',
               'num_minibatches', 'clip_eps', 'ent_coef', 'max_grad_norm',
               'layer_size', 'anneal_lr', 'log_freq', 'activation',
               'n_action_steps', 'log_fidelity', 'fid_factor', 'pulse_area_penalty', 'fluence_penalty',
               'smoothness_penalty_env', 'kernel_window', 'kernel_std', 'mxstep',
               'mx_step_penalty', 'fixed_endpoints',
               'snap_every', 'max_wall_time_seconds'}
    system_kwargs = {k: v for k, v in vars(args).items() if k not in generic}

    env_params = {
        'system_mod_path':   pre_args.system,
        'system_kwargs':     system_kwargs,
        'n_action_steps':    args.n_action_steps,
        'log_fidelity':      args.log_fidelity,
        'fid_factor':        args.fid_factor,
        'pulse_area_penalty':      args.pulse_area_penalty,
        'fluence_penalty':         args.fluence_penalty,
        'smoothness_penalty_env':  args.smoothness_penalty_env,
        'kernel_window':     args.kernel_window,
        'kernel_std':        args.kernel_std,
        'mxstep':            args.mxstep,
        'mx_step_penalty':   args.mx_step_penalty,
        'fixed_endpoints':   args.fixed_endpoints,
    }

    # Register + wire plot element names
    ppo.envs_class_dict['system_env'] = SystemEnv
    _orig = ppo.get_plot_elem_names
    def _plot_names(env_name):
        if env_name == 'system_env':
            return [f'env-{lbl}' for lbl in SystemEnv(**env_params).control_labels]
        return _orig(env_name)
    ppo.get_plot_elem_names = _plot_names

    # Build sysdict_raw once to name the output dir
    sysdict_raw = sys_mod.create_system(**system_kwargs)
    run_name = f"{sysdict_raw['system_name']}_ppo_seed{args.seed}"
    run_dir  = os.path.join(args.output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    print(f"Run: {run_name}")

    config = {
        'SEED': args.seed, 'LR': args.lr,
        'NUM_ENVS': args.num_envs, 'NUM_STEPS': 1,
        'NUM_UPDATES': args.num_updates, 'UPDATE_EPOCHS': 4,
        'NUM_MINIBATCHES': args.num_minibatches,
        'MINIBATCH_SIZE': args.num_envs // args.num_minibatches,
        'GAMMA': 0.99, 'GAE_LAMBDA': 0.95,
        'CLIP_EPS': args.clip_eps, 'ENT_COEF': args.ent_coef,
        'VF_COEF': 0.5, 'MAX_GRAD_NORM': args.max_grad_norm,
        'ACTIVATION': args.activation, 'LAYER_SIZE': args.layer_size,
        'ENV_NAME': 'system_env', 'ENV_PARAMS': env_params,
        'ANNEAL_LR': args.anneal_lr, 'DEBUG': False, 'DEBUG_NOJIT': False,
        'LOGGING': True, 'LOG_FREQ': max(1, args.log_freq),
        'LOG_WAND': False, 'LOCAL_LOGGING': True,
        'LOCAL_SAVE_NAME': run_name,
    }

    # ----------------------------------------------------------
    # Training: optionally chunked with periodic snapshots and a
    # wall-time cap.  When --snap_every is 0 we run the whole
    # num_updates in a single jit call (original behaviour).
    # ----------------------------------------------------------
    rng = jax.random.PRNGKey(args.seed)
    snap_env = SystemEnv(**env_params)  # for deterministic eval / snapshots

    def _save_snap_at(step_label, params):
        snap_path = os.path.join(run_dir, f'snap_{step_label:06d}.pkl')
        rollout_and_save_snap(snap_env, params, config, snap_path,
                               step_label=step_label,
                               system_name=sysdict_raw['system_name'])

    def _save_ckpt(params):
        with open(os.path.join(run_dir, 'ppo_params.pkl'), 'wb') as f:
            pickle.dump({'params': params,
                         'config': {k: v for k, v in config.items()
                                       if k != 'ENV_PARAMS'},
                         'env_params': env_params}, f)

    chunk = args.snap_every if args.snap_every > 0 else args.num_updates
    chunk = min(chunk, args.num_updates)
    chunk_config = dict(config); chunk_config['NUM_UPDATES'] = chunk
    train = jax.jit(ppo.PPO_make_train(chunk_config), static_argnames=())

    t0 = time.time()
    runner_state = None
    updates_done = 0

    while updates_done < args.num_updates:
        rng, sub_rng = jax.random.split(rng)
        outs = jax.block_until_ready(
            train(sub_rng) if runner_state is None
            else train(sub_rng, runner_state)
        )
        runner_state = outs['runner_state']
        updates_done += chunk

        # Pull params, save intermediate snapshot
        params = jax.tree_util.tree_map(jnp.asarray, runner_state[0].params)
        if args.snap_every > 0:
            _save_snap_at(updates_done, params)
        elapsed = time.time() - t0
        print(f"  chunk done.  updates={updates_done}/{args.num_updates}  "
              f"elapsed={elapsed:.1f}s")

        # Wall-time check
        if args.max_wall_time_seconds > 0 and elapsed >= args.max_wall_time_seconds:
            print(f"  [stop] wall-time cap {args.max_wall_time_seconds:.0f}s "
                  f"reached after {updates_done} updates")
            break

    # Save final checkpoint + snapshot (always, even if a snap_every snap
    # already happened at this exact step — overwrites with same content).
    _save_ckpt(params)
    if args.snap_every == 0 or updates_done % args.snap_every != 0:
        _save_snap_at(updates_done, params)
    print(f"Done.  total updates={updates_done}  total wall time={time.time() - t0:.1f}s")


if __name__ == '__main__':
    main()
