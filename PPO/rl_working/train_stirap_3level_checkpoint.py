"""Driver for 3-level Λ STIRAP with checkpoint saving.

Saves checkpoints whenever a new best mean fidelity is found.
"""
from __future__ import annotations

import argparse
import os
import pickle
import time

import jax
import jax.numpy as jnp

# Global state for checkpoint saving
_checkpoint_state = {
    'best_fid': 0.0,
    'save_dir': None,
    'config': None,
}


def _save_checkpoint_callback(train_state_params, step, mean_fid, config):
    """Save checkpoint if this is a new best fidelity."""
    global _checkpoint_state

    if mean_fid > _checkpoint_state['best_fid']:
        _checkpoint_state['best_fid'] = mean_fid

        save_dir = _checkpoint_state['save_dir']
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            ckpt_path = os.path.join(save_dir, 'best_checkpoint.pkl')
            payload = {
                'config': _checkpoint_state['config'],
                'params': jax.tree_util.tree_map(jnp.asarray, train_state_params),
                'step': int(step),
                'best_fid': float(mean_fid),
            }
            with open(ckpt_path, 'wb') as f:
                pickle.dump(payload, f)
            print(f"  >> Saved checkpoint at step {step} (fid={mean_fid:.4f})")


# Import and patch ppo
import rl_working.ppo as ppo
from rl_working.envs.single_stirap_3level_driftvar_env import SimpleStirap3LevelDriftVar

ppo.envs_class_dict["simple_stirap_3level_driftvar"] = SimpleStirap3LevelDriftVar


def build_config(args):
    env_params = {
        "gamma": args.gamma,
        "omega_0": args.omega_0,
        "t_total": args.t_total,
        "n_action_steps": args.n_action_steps,
        "x_penalty": args.x_penalty,
        "fid_factor": args.fid_factor,
        "pulse_area_penalty": args.pulse_area_penalty,
        "mxstep": args.mxstep,
        "mx_step_penalty": args.mx_step_penalty,
        "fixed_endpoints": args.fixed_endpoints,
        "log_fidelity": args.log_fidelity,
        "smoothness_penalty_env": args.smoothness_penalty_env,
        "drift_var_penalty": args.drift_var_penalty,
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
        "ENV_NAME": "simple_stirap_3level_driftvar",
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
    global _checkpoint_state

    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--num_envs", type=int, default=16)
    p.add_argument("--num_updates", type=int, default=50000)  # 5x more
    p.add_argument("--num_minibatches", type=int, default=4)
    p.add_argument("--clip_eps", type=float, default=0.2)
    p.add_argument("--ent_coef", type=float, default=0.0)
    p.add_argument("--max_grad_norm", type=float, default=0.5)
    p.add_argument("--layer_size", type=int, default=256)
    p.add_argument("--anneal_lr", type=int, default=0)
    p.add_argument("--log_freq", type=int, default=25)
    p.add_argument("--save_name", type=str, default="run")
    p.add_argument("--save_dir", type=str, default="saved_runs/stirap_hardest")

    # env hyperparams
    p.add_argument("--gamma", type=float, default=100.0)
    p.add_argument("--omega_0", type=float, default=30.0)
    p.add_argument("--t_total", type=float, default=0.1)
    p.add_argument("--n_action_steps", type=int, default=50)
    p.add_argument("--x_penalty", type=float, default=0.0)
    p.add_argument("--fid_factor", type=float, default=1.0)
    p.add_argument("--pulse_area_penalty", type=float, default=0.0)
    p.add_argument("--mxstep", type=int, default=4000)
    p.add_argument("--mx_step_penalty", type=float, default=25.0)
    p.add_argument("--fixed_endpoints", type=int, default=1)
    p.add_argument("--log_fidelity", type=int, default=1)
    p.add_argument("--smoothness_penalty_env", type=float, default=0.0)
    p.add_argument("--drift_var_penalty", type=float, default=0.0)
    p.add_argument("--kernel_window", type=int, default=25)
    p.add_argument("--kernel_std", type=float, default=4.0)

    args = p.parse_args()
    assert args.num_envs % args.num_minibatches == 0
    config = build_config(args)

    # Setup checkpoint state
    _checkpoint_state['save_dir'] = args.save_dir
    _checkpoint_state['config'] = {
        'env_params': config["ENV_PARAMS"],
        'config': {k: v for k, v in config.items() if k != "ENV_PARAMS"},
    }
    _checkpoint_state['best_fid'] = 0.0

    print(f"env: {config['ENV_NAME']}")
    print(f"gamma={args.gamma}  t_total={args.t_total}")
    print(f"log_fidelity={args.log_fidelity}  "
          f"pulse_area_penalty={args.pulse_area_penalty}  "
          f"smoothness_penalty_env={args.smoothness_penalty_env}  "
          f"drift_var_penalty={args.drift_var_penalty}  "
          f"x_penalty={args.x_penalty}  fid_factor={args.fid_factor}  "
          f"seed={args.seed}  num_updates={args.num_updates}")
    print(f"Checkpoints will be saved to: {args.save_dir}")

    # Run training in chunks to allow checkpointing
    chunk_size = 5000
    num_chunks = args.num_updates // chunk_size

    rng = jax.random.PRNGKey(args.seed)

    # Initialize
    from rl_working.envs.single_stirap_3level_driftvar_env import SimpleStirap3LevelDriftVar
    from rl_working.ppo import CombinedActorCritic, LogWrapper, VecEnv
    import optax
    from flax.training.train_state import TrainState

    env = SimpleStirap3LevelDriftVar(**config["ENV_PARAMS"])
    env = LogWrapper(env)
    env = VecEnv(env)
    env_params = env.default_params

    network = CombinedActorCritic(
        env.action_space(env_params).shape[0],
        activation=config["ACTIVATION"],
        layer_size=config["LAYER_SIZE"],
    )
    rng, _rng = jax.random.split(rng)
    init_x = jnp.zeros(env.observation_space(env_params).shape)
    network_params = network.init(_rng, init_x)

    tx = optax.chain(
        optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
        optax.adam(config["LR"], eps=1e-5),
    )
    train_state = TrainState.create(
        apply_fn=network.apply,
        params=network_params,
        tx=tx,
    )

    # Run training in chunks
    total_steps = 0
    best_fid = 0.0

    for chunk_idx in range(num_chunks):
        print(f"\n=== Chunk {chunk_idx + 1}/{num_chunks} (steps {total_steps} - {total_steps + chunk_size}) ===")

        # Create config for this chunk
        chunk_config = config.copy()
        chunk_config["NUM_UPDATES"] = chunk_size

        train_fn = jax.jit(ppo.PPO_make_train(chunk_config))

        t0 = time.time()
        outs = jax.block_until_ready(train_fn(rng))
        print(f"Chunk done in {time.time() - t0:.1f} s")

        runner_state = outs["runner_state"]
        train_state = runner_state[0]

        # Check if we need to save checkpoint by loading episodic data
        data_fpath = f"episodic_data/{config['ENV_NAME']}/{config['ENV_NAME']}_{args.save_name}.pkl"
        if os.path.exists(data_fpath):
            with open(data_fpath, 'rb') as f:
                data = pickle.load(f)
            if data:
                latest = data[-1]
                current_fid = float(latest.get('mean_fidelity', 0))
                if current_fid > best_fid:
                    best_fid = current_fid
                    # Save checkpoint
                    os.makedirs(args.save_dir, exist_ok=True)
                    ckpt_path = os.path.join(args.save_dir, 'best_checkpoint.pkl')
                    payload = {
                        'config': {k: v for k, v in config.items() if k != "ENV_PARAMS"},
                        'env_params': config["ENV_PARAMS"],
                        'params': jax.tree_util.tree_map(jnp.asarray, train_state.params),
                        'step': total_steps + chunk_size,
                        'best_fid': best_fid,
                    }
                    with open(ckpt_path, 'wb') as f:
                        pickle.dump(payload, f)
                    print(f"  >> Saved checkpoint (fid={best_fid:.4f})")

        total_steps += chunk_size
        rng, _ = jax.random.split(rng)

    # Final save
    out_dir = args.save_dir
    os.makedirs(out_dir, exist_ok=True)
    tag = (f"g{args.gamma}_t{args.t_total}_pap{args.pulse_area_penalty}_"
           f"sm{args.smoothness_penalty_env}_xp{args.x_penalty}_"
           f"logF{args.log_fidelity}_seed{args.seed}_upd{args.num_updates}")
    path = os.path.join(out_dir, f"{tag}.pkl")
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
