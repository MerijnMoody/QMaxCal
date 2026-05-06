"""Driver for Ernst's 5-level Λ STIRAP (dim=5, hyperfine parasitic branch) under PPO.

Wraps the existing `SimpleStirap` class with dim=5 so that PPO's env table
resolves the right class. Exposes the reward hyperparameters (pulse_area_penalty,
smoothness_penalty_env/det, log_fidelity, x_penalty, fid_factor) as CLI so a
pap sweep matches the 3-level drivers.

Run:
    cd /home/zmensch/MaxCal-VQE/reference/RL4qcWpc
    python -m rl_working.train_stirap5 --pulse_area_penalty 0.1 --seed 0
"""
from __future__ import annotations

import argparse
import os
import pickle
import time

import jax
import jax.numpy as jnp

import rl_working.ppo as ppo
from rl_working.envs.single_stirap_env import SimpleStirap

ppo.envs_class_dict["simple_stirap5"] = SimpleStirap
_orig_get_plot_elem_names = ppo.get_plot_elem_names


def _patched_plot_elem_names(env_name):
    if env_name == "simple_stirap5":
        return ["det-p", "det-s", "env-p", "env-s", "noise-amp", "noise-freq"]
    return _orig_get_plot_elem_names(env_name)


ppo.get_plot_elem_names = _patched_plot_elem_names


def build_config(args):
    env_params = {
        "gamma": args.gamma,
        "omega_0": args.omega_0,
        "delta_s": args.delta_s,
        "delta_p": args.delta_p,
        "mxstep": args.mxstep,
        "t_total": args.t_total,
        "x_penalty": args.x_penalty,
        "fid_factor": args.fid_factor,
        "log_fidelity": args.log_fidelity,
        "pulse_area_penalty": args.pulse_area_penalty,
        "smoothness_calc_amp": "second_derivative",
        "smoothness_calc_det": "second_derivative",
        "convolution_std_amp": args.kernel_std,
        "convolution_std_freq": args.kernel_std,
        "smoothness_cutoff_freq": args.smoothness_cutoff,
        "smoothness_penalty_env": args.smoothness_penalty_env,
        "smoothness_penalty_det": args.smoothness_penalty_det,
        "mx_step_penalty": args.mx_step_penalty,
        "fixed_endpoints": args.fixed_endpoints,
        "gaussian_kernel_window_length": [args.kernel_window_amp,
                                          args.kernel_window_freq],
        "initial_state": [1, 0],
        "final_state": [0, 1],
        "n_action_steps": args.n_action_steps,
        "dissipator": True,
        "dim": 5,
        "x_detuning": args.x_detuning,
        "noise": "None",
        "gaussian_noise_scaling": [0.0, 0.0],
    }
    return {
        "SEED": args.seed,
        "LR": args.lr,
        "NUM_ENVS": args.num_envs,
        "NUM_STEPS": 1,
        "NUM_UPDATES": args.num_updates,
        "UPDATE_EPOCHS": 4,
        "NUM_MINIBATCHES": args.num_minibatches,
        "MINIBATCH_SIZE": args.num_envs // args.num_minibatches,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": args.clip_eps,
        "ENT_COEF": args.ent_coef,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": args.max_grad_norm,
        "ACTIVATION": "relu6",
        "LAYER_SIZE": args.layer_size,
        "ENV_NAME": "simple_stirap5",
        "ENV_PARAMS": env_params,
        "ANNEAL_LR": args.anneal_lr,
        "DEBUG": False,
        "DEBUG_NOJIT": False,
        "LOGGING": True,
        "LOG_FREQ": max(1, args.log_freq),
        "LOG_WAND": False,
        "LOCAL_LOGGING": True,
        "LOCAL_SAVE_NAME": args.save_name,
    }


def main():
    p = argparse.ArgumentParser()
    # RL hyperparams
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--num_envs", type=int, default=16)
    p.add_argument("--num_updates", type=int, default=5000)
    p.add_argument("--num_minibatches", type=int, default=4)
    p.add_argument("--clip_eps", type=float, default=0.2)
    p.add_argument("--ent_coef", type=float, default=0.0)
    p.add_argument("--max_grad_norm", type=float, default=0.5)
    p.add_argument("--layer_size", type=int, default=256)
    p.add_argument("--anneal_lr", type=int, default=0)
    p.add_argument("--log_freq", type=int, default=25)
    p.add_argument("--save_name", type=str, default="run")

    # Env / physics — defaults mirror Ernst 5-level paper setup
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--omega_0", type=float, default=30.0)
    p.add_argument("--delta_s", type=float, default=30.0)
    p.add_argument("--delta_p", type=float, default=30.0)
    p.add_argument("--t_total", type=float, default=1.0)
    p.add_argument("--n_action_steps", type=int, default=50)
    p.add_argument("--x_detuning", type=float, default=100.0)
    p.add_argument("--mxstep", type=int, default=4000)

    # Reward shape — Ernst paper defaults for 5-level
    p.add_argument("--x_penalty", type=float, default=0.5)
    p.add_argument("--fid_factor", type=float, default=3.0)
    p.add_argument("--log_fidelity", type=int, default=1)
    p.add_argument("--pulse_area_penalty", type=float, default=0.1)
    p.add_argument("--smoothness_penalty_env", type=float, default=0.1)
    p.add_argument("--smoothness_penalty_det", type=float, default=0.1)
    p.add_argument("--smoothness_cutoff", type=float, default=5.0)
    p.add_argument("--kernel_std", type=float, default=1.0)
    p.add_argument("--kernel_window_amp", type=int, default=50)
    p.add_argument("--kernel_window_freq", type=int, default=25)
    p.add_argument("--mx_step_penalty", type=float, default=25.0)
    p.add_argument("--fixed_endpoints", type=int, default=1)

    args = p.parse_args()
    assert args.num_envs % args.num_minibatches == 0
    config = build_config(args)

    print(f"env: {config['ENV_NAME']}  (5-level Λ, dim=5, Ernst paper)")
    print(f"pulse_area_penalty={args.pulse_area_penalty}  "
          f"smoothness_env={args.smoothness_penalty_env}  "
          f"smoothness_det={args.smoothness_penalty_det}  "
          f"x_penalty={args.x_penalty}  fid_factor={args.fid_factor}  "
          f"log_fidelity={args.log_fidelity}")
    print(f"seed={args.seed}  num_updates={args.num_updates}")

    rng = jax.random.PRNGKey(args.seed)
    train = jax.jit(ppo.PPO_make_train(config))
    t0 = time.time()
    outs = jax.block_until_ready(train(rng))
    print(f"training done in {time.time() - t0:.1f} s")

    out_dir = f"saved_runs/{config['ENV_NAME']}"
    os.makedirs(out_dir, exist_ok=True)
    tag = (f"pap{args.pulse_area_penalty}_sm{args.smoothness_penalty_env}_"
           f"seed{args.seed}_upd{args.num_updates}")
    path = os.path.join(out_dir, f"{tag}.pkl")
    runner_state = outs["runner_state"]
    train_state = runner_state[0]
    payload = {
        "config": {k: v for k, v in config.items() if k != "ENV_PARAMS"},
        "env_params": config["ENV_PARAMS"],
        "params": jax.tree_util.tree_map(jnp.asarray, train_state.params),
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f)
    print(f"saved: {path}")


if __name__ == "__main__":
    main()
