#!/usr/bin/env python3
"""Amp damping drift α_k(t) = 2·Re⟨ψ|L_k|ψ⟩ over trajectories.
Same 4-panel layout as fig1C_variance_4panel: Baseline / Wiener KL / Drift-variance / PPO.
Shows that Wiener KL drives α→0 (kernel of L), drift-variance stabilizes α to a
nonzero constant (low variance, not kernel), and baseline/PPO leave α large.
"""
import os, sys, pickle
HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(HERE)
ROOT = os.path.dirname(PAPER)   # repo root
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
    d = np.load(npz_path); return d["ts"], d["u"]


class TabulatedController:
    def __init__(self, ts, u_table):
        self.ts = jnp.asarray(ts, dtype=jnp.float32)
        self.u_table = jnp.asarray(u_table, dtype=jnp.float32)
        self.n_controls = u_table.shape[1]

    def __call__(self, t):
        return jnp.stack([jnp.interp(t, self.ts, self.u_table[:, i])
                          for i in range(self.n_controls)])


def sample_drift(ctrl, sysdict, n_trajs=N_TRAJS, n_time_steps=N_TIME_STEPS, seed=12345):
    """Returns t, alpha_kt of shape (n_jumps, n_trajs, n_times)."""
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
    # exp_p shape: (n_trajs, n_times, n_jumps) — α_k = 2·Re⟨L_k+L_k†⟩
    alpha = np.asarray(res.extra["exp_p"])     # (n_trajs, n_times, n_jumps)
    alpha = np.transpose(alpha, (2, 0, 1))     # (n_jumps, n_trajs, n_times)
    return np.asarray(tsave), alpha


def main():
    sysdict = sys_mod.create_system(gamma=2.0, T_final=1.0)
    BASE_QMC   = os.path.join(PAPER, "results", "ampdamp", "qmaxcal", "multiseed_g2.0")
    BASE_PPO = os.path.join(PAPER, "results", "ampdamp", "ppo")

    PANELS = [
        ("Baseline ($\\lambda_{\\mathrm{KL}_\\mathrm{W}}{=}\\lambda_{R_{\\mathrm{DV}}}{=}0$)", "qmc",
         f"{BASE_QMC}/ampdamp_g2.0_T1.0_kl0.0_dv0.0_fl0.01_seed1/controller.pkl"),
        ("Wiener KL ($\\lambda_{\\mathrm{KL}_\\mathrm{W}}{=}5$)", "qmc",
         f"{BASE_QMC}/ampdamp_g2.0_T1.0_kl5.0_dv0.0_fl0.01_seed1/controller.pkl"),
        ("Drift variance ($\\lambda_{R_{\\mathrm{DV}}}{=}5$)", "qmc",
         f"{BASE_QMC}/ampdamp_g2.0_T1.0_kl0.0_dv5.0_fl0.01_seed1/controller.pkl"),
        ("PPO", "ppo",
         os.path.join(BASE_PPO, "ppo_ampdamp_g2_seed1_ustack.npz")),
    ]

    plt.rcParams.update({
        "font.size": 13, "axes.titlesize": 14,
        "axes.labelsize": 13, "xtick.labelsize": 11,
        "ytick.labelsize": 11, "legend.fontsize": 11,
    })

    fig, axes = plt.subplots(1, 4, figsize=(18.0, 3.6), sharey=True)
    summaries = []

    for col, (title, kind, path) in enumerate(PANELS):
        if kind == "qmc":
            ctrl = load_qmc(path)
        else:
            ts_act, u_arr = load_ppo_npz(path)
            ctrl = TabulatedController(ts_act, u_arr)
        print(f"[{title}] sampling drift over {N_TRAJS} SSE trajectories...")
        t, alpha = sample_drift(ctrl, sysdict)   # (n_jumps, n_trajs, n_times)

        ax = axes[col]
        # amp damping has 1 jump op (L=√γ σ-). Plot α(t) per traj as faint lines + mean.
        a0 = alpha[0]                     # (n_trajs, n_times)
        mu = a0.mean(axis=0)
        sd = a0.std(axis=0)
        # faint trajectory bundle
        for k in range(min(32, a0.shape[0])):
            ax.plot(t, a0[k], color="steelblue", lw=0.5, alpha=0.15)
        ax.plot(t, mu, color="black", lw=2.2, label=r"$\mathbb{E}[\alpha(t)]$")
        ax.fill_between(t, mu - sd, mu + sd, color="steelblue", alpha=0.30,
                        linewidth=0, label=r"$\pm\,1\sigma$")
        ax.axhline(0, color="crimson", lw=1.0, ls="--", alpha=0.7,
                   label=r"$\alpha=0$ ($\ker L$)")

        var_int = float(np.trapezoid(sd ** 2, t))
        sq_int  = float(np.trapezoid(np.mean(a0 ** 2, axis=0), t))   # ½∫α² dt is KL_W
        summaries.append((title, var_int, sq_int))
        print(f"  Var(α) integral = {var_int:.4f}   ⟨α²⟩ integral = {sq_int:.4f}")

        ax.set_title(title)
        ax.set_xlabel("t")
        ax.set_xlim(0, t[-1])
        ax.grid(True, alpha=0.3)
        if col == 3:
            ax.legend(loc="upper right", framealpha=0.5)
    axes[0].set_ylabel(r"$\alpha(t) = 2\,\mathrm{Re}\langle\psi|L|\psi\rangle$")

    plt.tight_layout(pad=0.5)
    out = os.path.join(HERE, "fig_ampdamp_drift_4panel.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.savefig(out.replace(".png", ".pdf"), bbox_inches="tight")
    print(f"\nsaved {out}")

    print("\nDrift summary (∫Var[α]dt and ∫⟨α²⟩dt):")
    for title, v, sq in summaries:
        print(f"  {title:>50}  ∫Var[α]={v:.4f}  ∫⟨α²⟩={sq:.4f}")


if __name__ == "__main__":
    main()
