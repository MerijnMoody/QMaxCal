#!/usr/bin/env python3
"""STIRAP populations side-by-side, same layout as fig1C amp damping.
Columns: Baseline (kl=0) / Wiener-KL (kl=1) / PPO. γ=5, T=2.
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
sys_mod = importlib.import_module("systems.stirap_3level")


N_TRAJS = 128
N_TIME_STEPS = 256

POP_COLOR = {"|g⟩": "steelblue", "|e⟩ (bright)": "crimson",
             "|f⟩ (target)": "seagreen"}
POP_LABEL = {"|g⟩": r"$|g\rangle$", "|e⟩ (bright)": r"$|e\rangle$",
             "|f⟩ (target)": r"$|f\rangle$"}
POP_ORDER = ["|g⟩", "|e⟩ (bright)", "|f⟩ (target)"]


def load_qmc(path):
    with open(path, "rb") as f:
        ck = pickle.load(f)
    n_modes = int(ck["n_modes"]); T_final = float(ck["T_final"])
    n_controls = int(ck["best_cos_coeffs"].shape[0])
    rngs = nnx.Rngs(0)
    ctrl = FourierController(n_controls=n_controls, n_modes=n_modes,
                             T_final=T_final, rngs=rngs)
    ctrl.dc = nnx.Param(ck["best_dc"])
    ctrl.cos_coeffs = nnx.Param(ck["best_cos_coeffs"])
    ctrl.sin_coeffs = nnx.Param(ck["best_sin_coeffs"])
    return ctrl


def load_ppo_npz(npz_path):
    d = np.load(npz_path)
    return d["ts"], d["u"]


class TabulatedController:
    def __init__(self, ts, u_table):
        self.ts = jnp.asarray(ts, dtype=jnp.float32)
        self.u_table = jnp.asarray(u_table, dtype=jnp.float32)
        self.n_controls = u_table.shape[1]

    def __call__(self, t):
        return jnp.stack([jnp.interp(t, self.ts, self.u_table[:, i])
                          for i in range(self.n_controls)])


def sample_populations(ctrl, sysdict, n_trajs=N_TRAJS, n_time_steps=N_TIME_STEPS, seed=12345):
    T_final = sysdict["T_final"]
    dt = T_final / n_time_steps
    tsave = jnp.linspace(0, T_final, n_time_steps + 1)
    sys_q = dict(
        H_sys=asqarray(sysdict["H_sys"]),
        H_ops=[asqarray(H) for H in sysdict["H_ops"]],
        jump_ops=[asqarray(L) for L in sysdict["jump_ops"]],
        psi0=asqarray(sysdict["psi0"]),
        T_final=T_final,
    )
    H_con = build_H_con(ctrl, sys_q["H_ops"])
    keys = jax.random.split(jax.random.PRNGKey(seed), n_trajs)
    res = dssesolve(H_sys=sys_q["H_sys"], jump_ops=sys_q["jump_ops"],
                    psi0=sys_q["psi0"], tsave=tsave, keys=keys, H_con=H_con,
                    method=ExponentialSplit(dt=dt),
                    nsteps=n_time_steps, nsteps_per_save=1)
    states = np.asarray(res.states.data).squeeze(-1)

    pop_traj = {}
    for name, P in sysdict["populations"].items():
        P_np = np.asarray(P)
        Pψ = np.einsum("ij,ntj->nti", P_np, states)
        expect = np.einsum("nti,nti->nt", np.conj(states), Pψ).real
        pop_traj[name] = expect
    return np.asarray(tsave), pop_traj


def main():
    sysdict = sys_mod.create_system(gamma=5.0, T_final=2.0)

    BASE_QMC   = os.path.join(ROOT, "results", "stirap", "qmaxcal", "n64")
    BASE_PPO = os.path.join(ROOT, "results", "stirap", "ppo")

    PANELS = [
        ("Baseline ($\\lambda_{R_{\\mathrm{DV}}}{=}\\lambda_{\\mathrm{KL}_\\mathrm{W}}{=}0$)",   "qmc",
         f"{BASE_QMC}/stirap3_g5.0_T2.0_kl0.0_dv0.0_fl0.0_seed1/controller.pkl"),
        ("Wiener KL ($\\lambda_{\\mathrm{KL}_\\mathrm{W}}{=}1$)",  "qmc",
         f"{BASE_QMC}/stirap3_g5.0_T2.0_kl1.0_dv0.0_fl0.0_seed1/controller.pkl"),
        ("PPO",                 "ppo",
         os.path.join(BASE_PPO, "ppo_stirap_flu0_g5_seed0_ustack.npz")),
    ]

    plt.rcParams.update({
        "font.size": 13, "axes.titlesize": 14,
        "axes.labelsize": 13, "xtick.labelsize": 11,
        "ytick.labelsize": 11, "legend.fontsize": 11,
    })

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.6), sharey=True)
    mean_vars = []

    for col, (title, kind, path) in enumerate(PANELS):
        if kind == "qmc":
            ctrl = load_qmc(path)
        else:
            ts_act, u_arr = load_ppo_npz(path)
            ctrl = TabulatedController(ts_act, u_arr)
        print(f"[{title}] sampling {N_TRAJS} SSE trajectories...")
        t, pop_traj = sample_populations(ctrl, sysdict)

        ax = axes[col]
        total_var = 0.0
        for name in POP_ORDER:
            if name not in pop_traj:
                continue
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
        if col == 2:
            ax.legend(loc="upper right", framealpha=0.5)
    axes[0].set_ylabel("Population")

    plt.tight_layout(pad=0.5)
    out = os.path.join(HERE, "fig_stirap_compare.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.savefig(out.replace(".png", ".pdf"), bbox_inches="tight")
    print(f"\nsaved {out}")

    print("\nSummary (sum_k ∫ Var[P_k] dt):")
    for (title, *_), v in zip(PANELS, mean_vars):
        print(f"  {title:>30}  total_var={v:.5f}")


if __name__ == "__main__":
    main()
