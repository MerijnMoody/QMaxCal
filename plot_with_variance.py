#!/usr/bin/env python3
"""
Plot side-by-side population trajectories with SSE-trajectory variance shading.

For each run dir:
    1. Load controller.pkl
    2. Reconstruct the system and the FourierController
    3. Sample N SSE trajectories with the trained policy
    4. Compute populations per trajectory
    5. Plot mean ± std as line + shaded band

Usage:
    python plot_with_variance.py \\
        --system systems.spin_chain_4 \\
        --runs path/to/baseline,path/to/kl,path/to/dv \\
        --titles "Baseline,KL,DV" \\
        --n-trajs 64 \\
        --out chain4_variance.png
"""
import os, sys, pickle, argparse, importlib
sys.path.insert(0, '/home/mmoody/qmaxcal_paper')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from core.types import ExponentialSplit, asqarray
from core.solver import dssesolve
from train_qmaxcal import FourierController, build_H_con, _get_transform


POP_COLORS = ['steelblue', 'crimson', 'darkorange', 'gray',
              'mediumpurple', 'seagreen', 'saddlebrown', 'deeppink']
CTRL_COLORS = ['steelblue', 'crimson', 'darkorange', 'seagreen', 'mediumpurple']


def load_controller(path):
    with open(path, 'rb') as f:
        ckpt = pickle.load(f)
    n_modes    = int(ckpt['n_modes'])
    T_final    = float(ckpt['T_final'])
    n_controls = int(ckpt['best_cos_coeffs'].shape[0])
    rngs = nnx.Rngs(0)
    ctrl = FourierController(n_controls=n_controls, n_modes=n_modes,
                              T_final=T_final, rngs=rngs)
    ctrl.dc         = nnx.Param(ckpt['best_dc'])
    ctrl.cos_coeffs = nnx.Param(ckpt['best_cos_coeffs'])
    ctrl.sin_coeffs = nnx.Param(ckpt['best_sin_coeffs'])
    return ctrl, ckpt


def sample_populations(ctrl, sysdict, n_trajs=64, n_time_steps=256, seed=12345):
    """Sample N SSE trajectories, return per-traj population time series."""
    T_final = sysdict['T_final']
    dt      = T_final / n_time_steps
    tsave   = jnp.linspace(0, T_final, n_time_steps + 1)
    H_ops_q = [asqarray(H) for H in sysdict['H_ops']]
    sys_q = dict(
        H_sys=asqarray(sysdict['H_sys']),
        H_ops=H_ops_q,
        jump_ops=[asqarray(L) for L in sysdict['jump_ops']],
        psi0=asqarray(sysdict['psi0']),
        target_state=asqarray(sysdict['target_state']),
        T_final=T_final,
        control_transform=sysdict.get('control_transform'),
    )
    H_con = build_H_con(ctrl, sys_q['H_ops'], transform=sys_q['control_transform'])

    keys = jax.random.split(jax.random.PRNGKey(seed), n_trajs)
    result = dssesolve(
        H_sys=sys_q['H_sys'], jump_ops=sys_q['jump_ops'],
        psi0=sys_q['psi0'], tsave=tsave, keys=keys,
        H_con=H_con, method=ExponentialSplit(dt=dt),
        nsteps=n_time_steps, nsteps_per_save=1,
    )
    states = np.asarray(result.states.data).squeeze(-1)   # (N, T+1, dim) complex

    # populations per traj: |⟨pop|ψ⟩|² for projector pop
    # pop matrices in sysdict['populations'] are projectors on full Hilbert space
    pop_traj = {}
    for name, P in sysdict['populations'].items():
        P_np = np.asarray(P)
        # ⟨ψ|P|ψ⟩ for each (traj, t)
        # = sum_ij conj(ψ_i) P_ij ψ_j
        Pψ      = np.einsum('ij,ntj->nti', P_np, states)
        expect  = np.einsum('nti,nti->nt', np.conj(states), Pψ).real
        pop_traj[name] = expect
    return np.asarray(tsave), pop_traj


def render(runs, titles, system_mod, system_kwargs, n_trajs, out_path, suptitle=None):
    n = len(runs)
    sysdict = system_mod.create_system(**system_kwargs)
    pop_names = list(sysdict['populations'].keys())

    fig, axes = plt.subplots(2, n, figsize=(5.5 * n, 9),
                              gridspec_kw={'hspace': 0.45, 'wspace': 0.3},
                              squeeze=False)

    for col, (run_dir, title) in enumerate(zip(runs, titles)):
        # Load controller for this run
        ctrl_path = os.path.join(run_dir, 'controller.pkl')
        ctrl, ckpt = load_controller(ctrl_path)
        print(f"[{title}] sampling {n_trajs} trajectories from {run_dir}...")

        t, pop_traj = sample_populations(ctrl, sysdict, n_trajs=n_trajs)

        # Top: populations with mean ± std
        ax_pop = axes[0, col]
        for i, name in enumerate(pop_names):
            arr = pop_traj[name]
            mu  = arr.mean(axis=0)
            sd  = arr.std(axis=0)
            ls  = '-' if i < 4 else ('--' if i < 7 else ':')
            ax_pop.plot(t, mu, color=POP_COLORS[i % len(POP_COLORS)],
                        lw=2, ls=ls, label=name)
            ax_pop.fill_between(t, mu - sd, mu + sd,
                                 color=POP_COLORS[i % len(POP_COLORS)], alpha=0.15)
        ax_pop.set_title(f"{title}   (F={ckpt.get('best_exact_fidelity', float('nan')):.3f})",
                         fontsize=11)
        ax_pop.set_xlabel('t'); ax_pop.set_ylabel('Population (mean ± σ)')
        ax_pop.set_xlim(0, t[-1]); ax_pop.set_ylim(-0.05, 1.05)
        ax_pop.grid(True, alpha=0.3)
        ax_pop.legend(fontsize=8, loc='upper right')

        # Bottom: controls (deterministic, transform-applied)
        tfm   = _get_transform(sysdict.get('control_transform'))
        t_ctrl = np.linspace(0, sysdict['T_final'], 200)
        sigs = np.asarray(tfm(jax.vmap(ctrl)(jnp.array(t_ctrl))))
        labels = sysdict.get('control_labels', [f'u_{i}' for i in range(sigs.shape[1])])

        ax_ctrl = axes[1, col]
        for i in range(sigs.shape[1]):
            ax_ctrl.plot(t_ctrl, sigs[:, i],
                          color=CTRL_COLORS[i % len(CTRL_COLORS)], lw=2,
                          label=labels[i])
        rms = [float(np.sqrt(np.mean(sigs[:, i] ** 2))) for i in range(sigs.shape[1])]
        ax_ctrl.set_title(' '.join(f'{labels[i]}:RMS={rms[i]:.2f}' for i in range(sigs.shape[1])),
                          fontsize=9)
        ax_ctrl.set_xlabel('t'); ax_ctrl.set_ylabel('amplitude')
        ax_ctrl.grid(True, alpha=0.3)
        ax_ctrl.legend(fontsize=8)

    if suptitle:
        plt.suptitle(suptitle, fontsize=13, y=1.01)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--system', type=str, required=True,
                   help='Dotted system module path')
    p.add_argument('--runs',   type=str, required=True,
                   help='Comma-separated run dirs')
    p.add_argument('--titles', type=str, default=None)
    p.add_argument('--system-kwargs', type=str, default='',
                   help='key=value,key=value extra system args')
    p.add_argument('--n-trajs', type=int, default=64)
    p.add_argument('--out',    type=str, default='populations_variance.png')
    p.add_argument('--suptitle', type=str, default=None)
    args = p.parse_args()

    sys_mod = importlib.import_module(args.system)
    sys_kw = {}
    if args.system_kwargs:
        for kv in args.system_kwargs.split(','):
            k, v = kv.split('=')
            try:   sys_kw[k] = float(v)
            except ValueError: sys_kw[k] = v

    runs = [r.strip() for r in args.runs.split(',')]
    titles = ([t.strip() for t in args.titles.split(',')] if args.titles
              else [os.path.basename(r) for r in runs])

    render(runs, titles, sys_mod, sys_kw, args.n_trajs, args.out, suptitle=args.suptitle)


if __name__ == '__main__':
    main()
