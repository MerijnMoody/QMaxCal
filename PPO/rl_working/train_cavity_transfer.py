"""Driver for cavity-assisted state transfer with PPO.

Two qubits coupled through a lossy cavity mode:
  H = g1(t)(σ+^(1) a + σ-^(1) a†) + g2(t)(σ+^(2) a + σ-^(2) a†)
  L = √κ a

Task: Transfer |1,0,0⟩ → |0,0,1⟩ using the dark mode strategy.

Run:
    python -m rl_working.train_cavity_transfer --kappa 5.0 --g0 10.0
"""
from __future__ import annotations

import argparse
import os
import pickle
import time

import jax
import jax.numpy as jnp

import rl_working.ppo as ppo
from rl_working.envs.cavity_transfer_env import CavityTransferEnv

ppo.envs_class_dict["cavity_transfer"] = CavityTransferEnv
_orig_get_plot_elem_names = ppo.get_plot_elem_names


def _patched_plot_elem_names(env_name):
    if env_name == "cavity_transfer":
        return ["env-g1", "env-g2"]
    return _orig_get_plot_elem_names(env_name)


ppo.get_plot_elem_names = _patched_plot_elem_names


def build_config(args):
    env_params = {
        "kappa": args.kappa,
        "g0": args.g0,
        "n_photon": args.n_photon,
        "t_total": args.t_total,
        "n_action_steps": args.n_action_steps,
        "log_fidelity": args.log_fidelity,
        "fid_factor": args.fid_factor,
        "cavity_penalty": args.cavity_penalty,
        "pulse_area_penalty": args.pulse_area_penalty,
        "smoothness_penalty_env": args.smoothness_penalty_env,
        "mxstep": args.mxstep,
        "mx_step_penalty": args.mx_step_penalty,
        "fixed_endpoints": args.fixed_endpoints,
        "kernel_window": args.kernel_window,
        "kernel_std": args.kernel_std,
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
        "ENV_NAME": "cavity_transfer",
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

    # env hyperparams
    p.add_argument("--kappa", type=float, default=5.0, help="Cavity decay rate")
    p.add_argument("--g0", type=float, default=10.0, help="Coupling strength")
    p.add_argument("--n_photon", type=int, default=3, help="Photon number truncation")
    p.add_argument("--t_total", type=float, default=1.0)
    p.add_argument("--n_action_steps", type=int, default=50)
    p.add_argument("--log_fidelity", type=int, default=1)
    p.add_argument("--fid_factor", type=float, default=1.0)
    p.add_argument("--cavity_penalty", type=float, default=0.0, help="Cavity pop penalty")
    p.add_argument("--pulse_area_penalty", type=float, default=0.0)
    p.add_argument("--smoothness_penalty_env", type=float, default=0.1)
    p.add_argument("--mxstep", type=int, default=4000)
    p.add_argument("--mx_step_penalty", type=float, default=25.0)
    p.add_argument("--fixed_endpoints", type=int, default=1)
    p.add_argument("--kernel_window", type=int, default=25)
    p.add_argument("--kernel_std", type=float, default=4.0)

    args = p.parse_args()
    assert args.num_envs % args.num_minibatches == 0
    config = build_config(args)

    print(f"env: {config['ENV_NAME']}")
    print(f"kappa={args.kappa}  g0={args.g0}  n_photon={args.n_photon}  t_total={args.t_total}")
    print(f"log_fidelity={args.log_fidelity}  cavity_penalty={args.cavity_penalty}  "
          f"pulse_area_penalty={args.pulse_area_penalty}  smoothness_penalty_env={args.smoothness_penalty_env}")
    print(f"seed={args.seed}  num_updates={args.num_updates}")

    rng = jax.random.PRNGKey(args.seed)
    train = jax.jit(ppo.PPO_make_train(config))
    t0 = time.time()
    outs = jax.block_until_ready(train(rng))
    print(f"training done in {time.time() - t0:.1f} s")

    out_dir = f"saved_runs/{config['ENV_NAME']}"
    os.makedirs(out_dir, exist_ok=True)
    tag = (f"kappa{args.kappa}_g0{args.g0}_np{args.n_photon}_"
           f"cp{args.cavity_penalty}_logF{args.log_fidelity}_"
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
