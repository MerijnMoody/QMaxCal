#!/usr/bin/env python3
"""Extract the PPO deterministic u_stack from saved PPO params.
Save as .npz so the plotting script can load without importing ppo (which
forces x64 on CPU and breaks dynamiqs's complex64 path)."""
import os, sys, pickle
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "PPO"))
sys.path.insert(0, os.path.join(ROOT, "PPO", "rl_working"))

import numpy as np
import jax.numpy as jnp
import train_ppo as tp


def main():
    params_path = sys.argv[1]
    out_path = sys.argv[2]
    gamma = float(sys.argv[3]) if len(sys.argv) > 3 else 2.0
    T_final = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0

    with open(params_path, "rb") as f:
        ck = pickle.load(f)
    params, config, env_params = ck["params"], ck["config"], ck["env_params"]
    sys_kwargs = dict(env_params.get("system_kwargs", {}))
    sys_kwargs["gamma"] = gamma
    sys_kwargs["T_final"] = T_final
    env_params["system_kwargs"] = sys_kwargs
    env = tp.SystemEnv(**env_params)

    network = tp.ppo.CombinedActorCritic(
        env.action_space(env.default_params).shape[0],
        activation=config["ACTIVATION"],
        layer_size=config["LAYER_SIZE"],
    )
    obs = jnp.zeros(env.observation_space(env.default_params).shape)
    pi, _ = network.apply(params, obs)
    action = pi.mode()
    u_stack = env._postprocess(action)        # (n_controls, n_action_steps)
    ts_action = np.asarray(env.ts_action, dtype=np.float32)
    u_arr = np.asarray(u_stack, dtype=np.float32).T  # (T, n_controls)

    np.savez(out_path, ts=ts_action, u=u_arr)
    print(f"saved {out_path}: ts shape={ts_action.shape}, u shape={u_arr.shape}")


if __name__ == "__main__":
    main()
