#!/usr/bin/env python3

import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
"""dark_route_v3 multi-seed robustness sweep.

Loads all (λ, seed) controllers from the multi-seed runs, re-evaluates each at
perturbed γ values, plots F(γ) per λ with error bars (mean ± std across seeds).

Inputs:
  results/dark_route_v3_repro/     (single-seed reproduction, seed=0 only)
  results/dark_route_v3_multiseed/ (additional seeds 0,1,2 at λ ∈ {0, 0.5, 5})
"""
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from systems.dark_route_v3 import create_system as dr3_create_system
from train_qmaxcal import FourierController, lindblad_trajectory

MULTISEED = Path(ROOT) / "results" / "diamond" / "qmaxcal" / "multiseed"

LAMBDAS = [0.0, 0.5, 5.0]
SEEDS   = [0, 1, 2]
GAMMAS_TEST = [1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 5.0]
GAMMA_TRAIN = 2.0

OUT = Path(ROOT) / "results" / "diamond" / "robustness"
OUT.mkdir(exist_ok=True, parents=True)


def find_controller(lam: float, seed: int):
    """Look for controller_best_by_loss.pkl (preferred) or controller.pkl."""
    name = f"darkroute_g{GAMMA_TRAIN}_T1.0_kl{lam}_dv0.0_fl0.001_seed{seed}"
    for fname in ("controller_best_by_loss.pkl", "controller.pkl"):
        p = MULTISEED / name / fname
        if p.exists():
            return p
    return None


def load_controller(lam: float, seed: int):
    p = find_controller(lam, seed)
    if p is None:
        return None
    with open(p, "rb") as f:
        d = pickle.load(f)
    n_modes = d["n_modes"]; T = d["T_final"]
    rngs = nnx.Rngs(seed)
    ctrl = FourierController(n_controls=3, n_modes=n_modes, T_final=T, rngs=rngs, init_scale=0.1)
    ctrl.dc.value         = jnp.asarray(d["best_dc"])
    ctrl.cos_coeffs.value = jnp.asarray(d["best_cos_coeffs"])
    ctrl.sin_coeffs.value = jnp.asarray(d["best_sin_coeffs"])
    return ctrl


def eval_F(ctrl, gamma_test):
    sys_g = dr3_create_system(gamma=gamma_test, T_final=1.0)
    _, _, rho_f = lindblad_trajectory(ctrl, sys_g, n_steps=400)
    tgt = sys_g["target_state"]
    tgt_proj = (tgt @ jnp.conj(tgt.T)).astype(jnp.complex64)
    return float(jnp.real(jnp.trace(tgt_proj @ rho_f)))


def main():
    # results[lam][seed] = array of F values across GAMMAS_TEST
    results = {lam: {} for lam in LAMBDAS}
    for lam in LAMBDAS:
        for seed in SEEDS:
            ctrl = load_controller(lam, seed)
            if ctrl is None:
                print(f"  MISSING: λ={lam}, seed={seed}")
                continue
            Fs = np.array([eval_F(ctrl, g) for g in GAMMAS_TEST])
            results[lam][seed] = Fs
            print(f"λ={lam} seed={seed}: " + ", ".join(f"γ={g}:F={F:.3f}" for g, F in zip(GAMMAS_TEST, Fs)))

    # Build mean ± std arrays
    F_means = {lam: None for lam in LAMBDAS}
    F_stds  = {lam: None for lam in LAMBDAS}
    for lam in LAMBDAS:
        if not results[lam]:
            continue
        stack = np.stack([v for v in results[lam].values()])
        F_means[lam] = stack.mean(axis=0)
        F_stds[lam]  = stack.std(axis=0, ddof=1) if stack.shape[0] > 1 else np.zeros(stack.shape[1])

    # ----- plot -----
    fig, ax = plt.subplots(1, 1, figsize=(8, 5.5))
    colors = {0.0: "#1f77b4", 0.5: "#ff7f0e", 5.0: "#d62728"}
    for lam in LAMBDAS:
        if F_means[lam] is None:
            continue
        n_seeds = len(results[lam])
        ax.errorbar(GAMMAS_TEST, F_means[lam], yerr=F_stds[lam],
                    fmt="o-", capsize=4, lw=2, ms=8,
                    label=f"λ={lam}  (n={n_seeds} seeds)",
                    color=colors[lam])
    ax.axvline(GAMMA_TRAIN, color="grey", ls="--", lw=1, label="γ_train=2.0")
    ax.set_xlabel("γ (test) — protocols trained at γ=2")
    ax.set_ylabel("F_exact (Lindblad, mean ± std over seeds)")
    ax.set_title("dark_route_v3 robustness: F vs γ, multi-seed")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    out = OUT / "dr3_robustness_multiseed.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.savefig(OUT / "dr3_robustness_multiseed.pdf")
    print(f"\nsaved {out}")

    # Print summary
    print("\n=== Summary table (mean F ± std) ===")
    header = "γ   | " + "  | ".join(f"λ={lam}" for lam in LAMBDAS)
    print(header)
    print("-" * len(header))
    for i, g in enumerate(GAMMAS_TEST):
        row = f"{g:>4.2f} | " + "  | ".join(
            f"{F_means[lam][i]:.3f}±{F_stds[lam][i]:.3f}" if F_means[lam] is not None else "  N/A   "
            for lam in LAMBDAS)
        print(row)

    # Save raw data
    np.savez(OUT / "dr3_robustness_multiseed.npz",
             gammas=np.array(GAMMAS_TEST),
             **{f"F_means_lam{lam}": F_means[lam] for lam in LAMBDAS if F_means[lam] is not None},
             **{f"F_stds_lam{lam}": F_stds[lam] for lam in LAMBDAS if F_stds[lam] is not None})


if __name__ == "__main__":
    main()
