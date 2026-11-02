#!/usr/bin/env python3
"""Fig 1C — SSE-trajectory variance at γT=2 with 4 panels:
baseline / Wiener-KL / DV / PPO.

For QMaxCal (baseline, KL, DV=1): load FourierController, run SSE.
For PPO: load PPO params, compute deterministic u_stack via policy,
wrap as a tabulated controller, then run SSE on the same Lindblad SDE.
"""
import os, sys, pickle
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core.types import ExponentialSplit, asqarray
from core.solver import dssesolve
from train_qmaxcal import FourierController, build_H_con
import importlib
sys_mod = importlib.import_module("systems.amplitude_damping")


N_TRAJS = 128
N_TIME_STEPS = 256

POP_COLOR = {"|0⟩": "steelblue", "|1⟩ (decays)": "crimson",
             "|+⟩": "darkorange", "|Y⟩": "seagreen"}
POP_LABEL = {"|0⟩": r"$|0\rangle$", "|1⟩ (decays)": r"$|1\rangle$",
             "|+⟩": r"$|+\rangle$", "|Y⟩": r"$|Y\rangle$"}
POP_ORDER = ["|0⟩", "|1⟩ (decays)", "|+⟩", "|Y⟩"]


def load_qmaxcal_controller(path):
    with open(path, "rb") as f:
        ck = pickle.load(f)
    n_modes = int(ck["n_modes"])
    T_final = float(ck["T_final"])
    n_controls = int(ck["best_cos_coeffs"].shape[0])
    rngs = nnx.Rngs(0)
    ctrl = FourierController(n_controls=n_controls, n_modes=n_modes,
                             T_final=T_final, rngs=rngs)
    ctrl.dc = nnx.Param(ck["best_dc"])
    ctrl.cos_coeffs = nnx.Param(ck["best_cos_coeffs"])
    ctrl.sin_coeffs = nnx.Param(ck["best_sin_coeffs"])
    return ctrl


def load_ppo_u_stack(npz_path):
    """Load pre-extracted PPO u_stack from npz (avoids importing train_ppo
    which would force x64 on CPU and break dssesolve's complex64 path)."""
    d = np.load(npz_path)
    return d["ts"], d["u"]


class TabulatedController:
    """Linear-interpolation 'controller' from a precomputed u(t) table."""
    def __init__(self, ts, u_table):
        # ts: (T,), u_table: (T, n_controls)
        self.ts = jnp.asarray(ts, dtype=jnp.float32)
        self.u_table = jnp.asarray(u_table, dtype=jnp.float32)
        self.n_controls = u_table.shape[1]

    def __call__(self, t):
        # Interpolate per-control
        return jnp.stack([jnp.interp(t, self.ts, self.u_table[:, i])
                          for i in range(self.n_controls)])


def sample_populations(ctrl, sysdict, n_trajs=N_TRAJS, n_time_steps=N_TIME_STEPS, seed=12345):
    T_final = sysdict["T_final"]
    dt = T_final / n_time_steps
    tsave = jnp.linspace(0, T_final, n_time_steps + 1)
    H_ops_q = [asqarray(H) for H in sysdict["H_ops"]]
    sys_q = dict(
        H_sys=asqarray(sysdict["H_sys"]),
        H_ops=H_ops_q,
        jump_ops=[asqarray(L) for L in sysdict["jump_ops"]],
        psi0=asqarray(sysdict["psi0"]),
        T_final=T_final,
    )
    H_con = build_H_con(ctrl, sys_q["H_ops"])
    keys = jax.random.split(jax.random.PRNGKey(seed), n_trajs)
    result = dssesolve(
        H_sys=sys_q["H_sys"], jump_ops=sys_q["jump_ops"],
        psi0=sys_q["psi0"], tsave=tsave, keys=keys,
        H_con=H_con, method=ExponentialSplit(dt=dt),
        nsteps=n_time_steps, nsteps_per_save=1,
    )
    states = np.asarray(result.states.data).squeeze(-1)

    pop_traj = {}
    for name, P in sysdict["populations"].items():
        P_np = np.asarray(P)
        Pψ = np.einsum("ij,ntj->nti", P_np, states)
        expect = np.einsum("nti,nti->nt", np.conj(states), Pψ).real
        pop_traj[name] = expect
    return np.asarray(tsave), pop_traj


def main():
    sysdict = sys_mod.create_system(gamma=2.0, T_final=1.0)

    BASE_QMC   = os.path.join(ROOT, "results", "ampdamp", "qmaxcal", "multiseed_g2.0")
    BASE_PPO = os.path.join(ROOT, "results", "ampdamp", "ppo")

    PANELS = [
        ("Baseline ($\\lambda_{R_{\\mathrm{DV}}}{=}\\lambda_{\\mathrm{KL}_\\mathrm{W}}{=}0$)", "qmc",
         f"{BASE_QMC}/ampdamp_g2.0_T1.0_kl0.0_dv0.0_fl0.01_seed1/controller.pkl"),
        ("Wiener KL ($\\lambda_{\\mathrm{KL}_\\mathrm{W}}{=}5$)",     "qmc",
         f"{BASE_QMC}/ampdamp_g2.0_T1.0_kl5.0_dv0.0_fl0.01_seed1/controller.pkl"),
        ("Drift variance ($\\lambda_{R_{\\mathrm{DV}}}{=}5$)", "qmc",
         f"{BASE_QMC}/ampdamp_g2.0_T1.0_kl0.0_dv5.0_fl0.01_seed1/controller.pkl"),
        ("PPO",                    "ppo",
         os.path.join(BASE_PPO, "ppo_ampdamp_g2_seed1_ustack.npz")),
    ]

    plt.rcParams.update({
        "font.size": 13, "axes.titlesize": 14,
        "axes.labelsize": 13, "xtick.labelsize": 11,
        "ytick.labelsize": 11, "legend.fontsize": 11,
    })

    fig, axes = plt.subplots(1, 4, figsize=(18.0, 3.6), sharey=True)
    mean_vars = []

    for col, (title, kind, path) in enumerate(PANELS):
        if kind == "qmc":
            ctrl = load_qmaxcal_controller(path)
        else:
            ts_act, u_arr = load_ppo_u_stack(path)
            ctrl = TabulatedController(ts_act, u_arr)
        print(f"[{title}] sampling {N_TRAJS} SSE trajectories...")
        t, pop_traj = sample_populations(ctrl, sysdict)

        ax = axes[col]
        total_var = 0.0
        for name in POP_ORDER:
            arr = pop_traj[name]
            mu = arr.mean(axis=0)
            sd = arr.std(axis=0)
            c = POP_COLOR[name]
            ax.plot(t, mu, color=c, lw=2.0, label=POP_LABEL[name])
            ax.fill_between(t, mu - sd, mu + sd, color=c, alpha=0.18, linewidth=0)
            total_var += float(np.trapezoid(sd ** 2, t))
        mean_vars.append(total_var)

        ax.set_title(title)
        ax.set_xlabel("t")
        ax.set_xlim(0, t[-1])
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        if col == 3:
            ax.legend(loc="upper right", framealpha=0.5)
    axes[0].set_ylabel("Population")

    plt.tight_layout(pad=0.5)
    out = os.path.join(HERE, "fig_ampdamp_populations_4panel.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.savefig(out.replace(".png", ".pdf"), bbox_inches="tight")
    print(f"\nsaved {out}")

    print("\nSummary (sum_k ∫ Var[P_k] dt):")
    for (title, *_), v in zip(PANELS, mean_vars):
        print(f"  {title:>30}  total_var={v:.5f}")


if __name__ == "__main__":
    main()
