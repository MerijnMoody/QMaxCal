#!/usr/bin/env python3
"""
2x3 grid: top row dv=0.02, bottom row baseline (dv=0), columns rho=2,4,8.
Each panel shows population mean +/- std across 3 seeds.
No suptitle.  Above each panel only the (rho, dv) labels.
"""
import os, glob, pickle, re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = '/home/mmoody/qmaxcal_paper'
SITE_COLOR = ['steelblue', 'crimson', 'darkorange', 'seagreen']


def final_snap(d):
    """Return (F, step, snap) for the snapshot at the highest step number."""
    real = os.path.realpath(d)
    cands = []
    for p in glob.glob(f'{real}/snap_*.pkl'):
        m = re.match(r'snap_(\d+)\.pkl$', os.path.basename(p))
        if not m:
            continue
        cands.append((int(m.group(1)), p))
    if not cands:
        return None
    cands.sort()
    step, path = cands[-1]
    try:
        s = pickle.load(open(path, 'rb'))
    except Exception:
        return None
    F = float(s.get('f_exact', float('nan')))
    return (F, step, s)


def load_best_seed(rho, dv):
    """Pick the (final-snapshot dict of the) seed with the highest final F across {0,1,2}."""
    base = f'{ROOT}/results/chain4/qmaxcal/variance_asym_sweep_original/r{rho:.1f}'
    dv_str = f'{dv}'
    candidates = []
    for seed in (0, 1, 2):
        d = (f'{base}/chain4_k0.3_g0.05_boost{rho:.1f}_T3.0_'
             f'kl0.0_dv{dv_str}_fl0.005_seed{seed}')
        pick = final_snap(d)
        if pick is not None and np.isfinite(pick[0]):
            F, step, snap = pick
            candidates.append((F, seed, snap))
    if not candidates:
        return None
    candidates.sort()
    return candidates[-1][2]   # snap of seed with highest final F


def plot_single(ax, snap):
    if snap is None:
        ax.text(0.5, 0.5, '(no data)', ha='center', va='center',
                transform=ax.transAxes, color='gray')
        return
    t = np.asarray(snap['t_traj'])
    for i, color in enumerate(SITE_COLOR):
        P_i = np.asarray(snap['populations'][f'P(site {i} ↑)'])
        ax.plot(t, P_i, color=color, lw=2.0, label=f'site {i}')
    ax.set_xlim(0, t[-1])
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)


fig, axes = plt.subplots(2, 3, figsize=(13.5, 6.4),
                          sharey=True, sharex=False,
                          gridspec_kw={'wspace': 0.10, 'hspace': 0.30})

# Top row: dv=0.02 (best seed); bottom row: baseline (best seed)
for col, rho in enumerate([2.0, 4.0, 8.0]):
    snap_dv = load_best_seed(rho, 0.02)
    snap_bl = load_best_seed(rho, 0.0)

    ax_top = axes[0, col]
    plot_single(ax_top, snap_dv)
    ax_top.set_title(rf'$\rho={rho:g},\ \lambda_{{R_{{\mathrm{{DV}}}}}}=0.02$',
                     fontsize=11)

    ax_bot = axes[1, col]
    plot_single(ax_bot, snap_bl)
    ax_bot.set_title(rf'$\rho={rho:g},\ \lambda_{{R_{{\mathrm{{DV}}}}}}'
                     rf'=\lambda_{{\mathrm{{KL}}_{{\mathrm{{W}}}}}}=0$',
                     fontsize=11)

    if col == 0:
        ax_top.set_ylabel('P(site $i$ excited)', fontsize=11)
        ax_bot.set_ylabel('P(site $i$ excited)', fontsize=11)
    ax_bot.set_xlabel('t', fontsize=11)

axes[0, -1].legend(fontsize=9, loc='upper right', framealpha=0.5)

out = f'{ROOT}/figures/fig_2dv_3rho_seeds.png'
plt.savefig(out, dpi=160, bbox_inches='tight')
print(f'Saved: {out}')
