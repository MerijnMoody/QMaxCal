#!/usr/bin/env python3
"""Two-row comparison of policies on chain-6 IBM Kingston q14->q9
(T_final=30 us, matched lambda_flu=0.02 unless otherwise noted):

  Top row    : Baseline | Wiener KL | Drift-variance | PPO (lambda_flu=0.02)
  Bottom row : (empty) | (empty) | PPO (lambda_flu=0.02) | PPO (lambda_flu=0)

For each policy we pick the seed with the highest exact-Lindblad
fidelity at its best snapshot, then take *that* snapshot.  The
bottom row repeats the matched-fluence PPO panel alongside an
unconstrained (lambda_flu=0) PPO run so the fluence cost of removing
the regulariser is visible at a glance.
"""
import os, glob, pickle, re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PAPER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

QMAX_BASE       = f'{PAPER}/results/chain6/qmaxcal/variance_q14_q9_flw0.02'
PPO_BASE        = f'{PAPER}/results/chain6/ppo/q14_q9_sm0.001_ks4.0'
PPO_NOFLU_BASE  = f'{PAPER}/results/chain6/ppo/q14_q9_sm0.001_ks4.0_nofluence'

# Top row: 4 main policies
TOP_POLICIES = [
    ('Baseline ($\\lambda_{KL}=\\lambda_{DV}=0$)',
     [f'{QMAX_BASE}/chain6kingq14q9_T30_kl0.0_dv0.0_fl0.02_seed{s}' for s in range(3)]),
    ('Wiener KL ($\\lambda_{KL}=0.1$)',
     [f'{QMAX_BASE}/chain6kingq14q9_T30_kl0.1_dv0.0_fl0.02_seed{s}' for s in range(3)]),
    ('Drift-variance ($\\lambda_{DV}=16$)',
     [f'{QMAX_BASE}/chain6kingq14q9_T30_kl0.0_dv16.0_fl0.02_seed{s}' for s in range(3)]),
    ('PPO ($\\lambda_{flu}=0.02$)',
     [f'{PPO_BASE}/chain6kingq14q9_T30_ppo_seed{s}' for s in range(3)]),
]

# Bottom row: the two PPO panels are centered in cols 1 and 2; cols 0 and 3 are blank
BOTTOM_POLICIES = [
    None,
    ('PPO ($\\lambda_{flu}=0.02$)',
     [f'{PPO_BASE}/chain6kingq14q9_T30_ppo_seed{s}' for s in range(3)]),
    ('PPO ($\\lambda_{flu}=0$)',
     [f'{PPO_NOFLU_BASE}/chain6kingq14q9_T30_ppo_seed{s}' for s in range(3)]),
    None,
]

SITE_COLOR = ['steelblue', 'crimson', 'darkorange',
              'seagreen', 'mediumpurple', 'saddlebrown']


def best_F_snap(d):
    cands = []
    for p in glob.glob(f'{d}/snap_*.pkl'):
        m = re.match(r'snap_(\d+)\.pkl$', os.path.basename(p))
        if not m:
            continue
        try:
            s = pickle.load(open(p, 'rb'))
        except Exception:
            continue
        F = float(s.get('f_exact', float('nan')))
        if np.isfinite(F):
            cands.append((F, int(m.group(1)), s))
    if not cands:
        return None
    cands.sort()
    return cands[-1]


def best_seed_snap(seed_dirs):
    best = None
    for d in seed_dirs:
        if not os.path.isdir(d):
            continue
        pick = best_F_snap(d)
        if pick is None:
            continue
        F, step, snap = pick
        if best is None or F > best[0]:
            best = (F, step, d, snap)
    return best


def render_panel(ax, label, seed_dirs):
    pick = best_seed_snap(seed_dirs)
    if pick is None:
        ax.text(0.5, 0.5, '(no data)', ha='center', va='center',
                transform=ax.transAxes, color='gray')
        ax.set_title(label, fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
        return None
    F, step, run_dir, snap = pick
    seed_match = re.search(r'_seed(\d+)$', run_dir)
    seed = int(seed_match.group(1)) if seed_match else -1
    flu  = float(snap.get('fluence', np.nan))
    t    = np.asarray(snap['t_traj'])
    for i in range(6):
        P_i = np.asarray(snap['populations'].get(f'P(site {i} ↑)', [np.nan]))
        ax.plot(t, P_i, color=SITE_COLOR[i], lw=1.7, label=f'site {i}')
    ax.set_title(label, fontsize=11)
    ax.set_xlim(0, t[-1])
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    return dict(F=F, step=step, seed=seed, flu=flu, label=label)


# ------- layout: 2 rows × 4 columns -------
fig, axes = plt.subplots(2, len(TOP_POLICIES),
                          figsize=(4.6 * len(TOP_POLICIES), 8.8),
                          gridspec_kw={'wspace': 0.22, 'hspace': 0.40})

print(f'{"row":>4s}  {"policy":>32s}  {"seed":>4s}  {"step":>5s}  '
      f'{"F":>7s}  {"flu²":>8s}')

# Top row
for col, entry in enumerate(TOP_POLICIES):
    label, seed_dirs = entry
    info = render_panel(axes[0, col], label, seed_dirs)
    axes[0, col].set_xlabel('t  (μs)', fontsize=11)
    if col == 0:
        axes[0, col].set_ylabel('P(site $i$ excited)', fontsize=11)
    if col == len(TOP_POLICIES) - 1:
        axes[0, col].legend(fontsize=8, loc='upper right',
                             framealpha=0.5, ncol=2)
    if info:
        print(f'{"top":>4s}  {info["label"]:>32s}  {info["seed"]:>4d}  '
              f'{info["step"]:>5d}  {info["F"]:.4f}  {info["flu"]:8.2f}')

# Bottom row: blank cols 0, 1, populated cols 2, 3
for col, entry in enumerate(BOTTOM_POLICIES):
    if entry is None:
        axes[1, col].set_visible(False)
        continue
    label, seed_dirs = entry
    info = render_panel(axes[1, col], label, seed_dirs)
    axes[1, col].set_xlabel('t  (μs)', fontsize=11)
    # ylabel on the left-most visible bottom panel
    if col == 0 or not axes[1, col - 1].get_visible():
        axes[1, col].set_ylabel('P(site $i$ excited)', fontsize=11)
    if info:
        print(f'{"bot":>4s}  {info["label"]:>32s}  {info["seed"]:>4d}  '
              f'{info["step"]:>5d}  {info["F"]:.4f}  {info["flu"]:8.2f}')

out = f'{PAPER}/figures/fig_chain6_q14_q9_4policies.png'
plt.savefig(out, dpi=160, bbox_inches='tight')
print(f'\nSaved: {out}')
