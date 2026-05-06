#!/usr/bin/env python3
"""Single-panel plot of ⟨α²⟩(t) on log y-axis for amp damping at γT=2.
The KL_W integrand: shows KL_W on the floor (kernel reached) vs DV plateauing
at baseline level (drift constant but nonzero, kernel not reached).
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


def load_ernst_npz(npz_path):
    d = np.load(npz_path); return d["ts"], d["u"]


class TabulatedController:
    def __init__(self, ts, u_table):
        self.ts = jnp.asarray(ts, dtype=jnp.float32)
        self.u_table = jnp.asarray(u_table, dtype=jnp.float32)
        self.n_controls = u_table.shape[1]

    def __call__(self, t):
        return jnp.stack([jnp.interp(t, self.ts, self.u_table[:, i])
                          for i in range(self.n_controls)])


def sample_alpha2(ctrl, sysdict, n_trajs=N_TRAJS, n_time_steps=N_TIME_STEPS, seed=12345):
    """Returns t, ⟨α²⟩(t) averaged over trajectories — shape (n_times,)."""
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
    alpha = np.asarray(res.extra["exp_p"])           # (n_trajs, n_times, n_jumps)
    alpha2 = (alpha ** 2).sum(axis=-1)                # (n_trajs, n_times) — sum over jumps
    return np.asarray(tsave), alpha2.mean(axis=0)     # E_traj[α²](t)


def main():
    sysdict = sys_mod.create_system(gamma=2.0, T_final=1.0)
    BASE_QMC   = os.path.join(ROOT, "results", "ampdamp", "qmaxcal", "multiseed_g2.0")
    BASE_ERNST = os.path.join(ROOT, "results", "ampdamp", "ppo")

    METHODS = [
        ("Baseline ($\\lambda_{\\mathrm{KL}_\\mathrm{W}}{=}\\lambda_{R_{\\mathrm{DV}}}{=}0$)",
         "qmc", f"{BASE_QMC}/ampdamp_g2.0_T1.0_kl0.0_dv0.0_fl0.01_seed1/controller.pkl",
         "darkgray", "-"),
        ("Wiener KL ($\\lambda_{\\mathrm{KL}_\\mathrm{W}}{=}5$)",
         "qmc", f"{BASE_QMC}/ampdamp_g2.0_T1.0_kl5.0_dv0.0_fl0.01_seed1/controller.pkl",
         "steelblue", "-"),
        ("Drift variance ($\\lambda_{R_{\\mathrm{DV}}}{=}5$)",
         "qmc", f"{BASE_QMC}/ampdamp_g2.0_T1.0_kl0.0_dv5.0_fl0.01_seed1/controller.pkl",
         "darkorange", "-"),
        ("PPO", "ernst",
         os.path.join(BASE_ERNST, "ernst_ampdamp_g2_seed1_ustack.npz"),
         "crimson", "--"),
    ]

    plt.rcParams.update({
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}",
        "font.family": "serif",
        "font.size": 13, "axes.titlesize": 14,
        "axes.labelsize": 13, "xtick.labelsize": 11,
        "ytick.labelsize": 11, "legend.fontsize": 11,
    })

    fig, ax = plt.subplots(1, 1, figsize=(10.0, 4.2))

    summaries = []
    for label, kind, path, color, ls in METHODS:
        ctrl = load_qmc(path) if kind == "qmc" else (
            (lambda ts, u: TabulatedController(ts, u))(*load_ernst_npz(path)))
        print(f"[{label}] sampling...")
        t, a2 = sample_alpha2(ctrl, sysdict)
        ax.plot(t, a2, color=color, lw=2.0, ls=ls, label=label)
        kl_int = 0.5 * float(np.trapezoid(a2, t))
        summaries.append((label, kl_int))
        print(f"  ½∫⟨α²⟩dt = {kl_int:.4f}  (this IS the KL_W training objective)")

    ax.set_yscale("log")
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$\mathbb{E}_{P_\theta}[\alpha(t)^2]$")
    ax.set_xlim(0, t[-1])
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower left", framealpha=0.5)

    plt.tight_layout(pad=0.5)
    out = os.path.join(HERE, "fig_ampdamp_alpha2_log.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.savefig(out.replace(".png", ".pdf"), bbox_inches="tight")
    print(f"\nsaved {out}")

    print("\n½∫⟨α²⟩dt summary (= KL_W cost at evaluation):")
    for label, v in summaries:
        print(f"  {label:>50}  {v:.4f}")


if __name__ == "__main__":
    main()
