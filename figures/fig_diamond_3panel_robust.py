#!/usr/bin/env python3
"""Diamond / dark-route 3-panel figure for the body:
panel 1: baseline populations; panel 2: Wiener-KL populations;
panel 3: γ-perturbation robustness curve (replaces PPO populations
panel; PPO populations move to the appendix).
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
sys_mod = importlib.import_module("systems.dark_route_v3")


N_TRAJS = 128
N_TIME_STEPS = 256

POP_COLOR = {"|b⟩": "steelblue", "|t⟩": "crimson",
             "|d⟩": "seagreen",  "|dump⟩": "gray"}
POP_LABEL = {"|b⟩": r"$|b\rangle$", "|t⟩": r"$|t\rangle$",
             "|d⟩": r"$|d\rangle$", "|dump⟩": r"$|\mathrm{dump}\rangle$"}
POP_ORDER = ["|b⟩", "|t⟩", "|d⟩", "|dump⟩"]


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
    sysdict = sys_mod.create_system(gamma=2.0, T_final=1.0)
    BASE_QMC = os.path.join(ROOT, "results", "diamond", "qmaxcal", "multiseed")

    POP_PANELS = [
        ("Baseline ($\\lambda_{R_{\\mathrm{DV}}}{=}\\lambda_{\\mathrm{KL}_\\mathrm{W}}{=}0$)",
         f"{BASE_QMC}/darkroute_g2.0_T1.0_kl0.0_dv0.0_fl0.001_seed1/controller.pkl"),
        ("Wiener KL ($\\lambda_{\\mathrm{KL}_\\mathrm{W}}{=}5$)",
         f"{BASE_QMC}/darkroute_g2.0_T1.0_kl5.0_dv0.0_fl0.001_seed1/controller.pkl"),
    ]

    plt.rcParams.update({
        "font.size": 13, "axes.titlesize": 14,
        "axes.labelsize": 13, "xtick.labelsize": 11,
        "ytick.labelsize": 11, "legend.fontsize": 10,
    })

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.6))

    # --- population panels (no shared y; we set ylim manually) ---
    for col, (title, path) in enumerate(POP_PANELS):
        ctrl = load_qmc(path)
        print(f"[{title}] sampling {N_TRAJS} SSE trajectories...")
        t, pop_traj = sample_populations(ctrl, sysdict)
        ax = axes[col]
        for name in POP_ORDER:
            arr = pop_traj[name]
            mu = arr.mean(axis=0); sd = arr.std(axis=0)
            c = POP_COLOR[name]
            ax.plot(t, mu, color=c, lw=2.0, label=POP_LABEL[name])
            ax.fill_between(t, mu - sd, mu + sd, color=c, alpha=0.18, linewidth=0)
        ax.set_title(title)
        ax.set_xlabel(r"$t$")
        ax.set_xlim(0, t[-1])
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        if col == 1:
            ax.legend(loc="upper right", framealpha=0.5, ncol=2)
    axes[0].set_ylabel("Population")

    # --- robustness panel ---
    ax = axes[2]
    npz = np.load(os.path.join(ROOT, "results", "diamond", "robustness", "dr3_robustness_multiseed.npz"))
    gammas = npz["gammas"]

    # PPO numbers from app:diamond `tab:dr3_robustness` (3-seed mean ± std)
    ppo_g    = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 5.0])
    ppo_mean = np.array([0.772, 0.681, 0.601, 0.531, 0.470, 0.288])
    ppo_std  = np.array([0.001, 0.002, 0.002, 0.002, 0.002, 0.001])

    series = [
        ("Baseline",                               npz["F_means_lam0.0"], npz["F_stds_lam0.0"], "gray", "o", "-"),
        (r"Wiener KL ($\lambda_{\mathrm{KL}_\mathrm{W}}{=}5$)",   npz["F_means_lam5.0"], npz["F_stds_lam5.0"], "navy", "D", "-"),
    ]
    for label, mu, sd, color, marker, ls in series:
        ax.plot(gammas, mu, color=color, lw=1.6, marker=marker, ms=5, ls=ls, label=label)
        ax.fill_between(gammas, mu - sd, mu + sd, color=color, alpha=0.18, linewidth=0)
    ax.plot(ppo_g, ppo_mean, color="crimson", lw=1.6, marker="^", ms=5, ls="--", label="PPO")
    ax.fill_between(ppo_g, ppo_mean - ppo_std, ppo_mean + ppo_std, color="crimson", alpha=0.18, linewidth=0)

    # mark the training γ
    ax.axvline(2.0, color="k", lw=0.8, alpha=0.4, ls=":")
    ax.text(2.05, 0.28, r"$\gamma_{\mathrm{train}}$", color="k", alpha=0.6, fontsize=10)

    ax.set_title(r"$\gamma$-perturbation robustness")
    ax.set_xlabel(r"$\gamma_{\mathrm{test}}$")
    ax.set_ylabel(r"$F_{\mathrm{exact}}$")
    ax.set_xlim(0.95, 5.05)
    ax.set_ylim(0.25, 0.95)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", framealpha=0.5, fontsize=9)

    plt.tight_layout(pad=0.6)
    out = os.path.join(HERE, "fig_diamond_3panel_robust.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.savefig(out.replace(".png", ".pdf"), bbox_inches="tight")
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
