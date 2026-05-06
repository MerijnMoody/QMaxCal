#!/usr/bin/env python3
"""
Shared plotter for QMaxCal + PPO snapshots.

Reads the latest snap_*.pkl in each run dir and plots a 2×N grid:
  top row: populations (one line per entry in snap['populations'])
  bottom row: controls (one line per channel)

Usage:
    python plot_snapshots.py --runs results/run_a,results/run_b --out fig.png
"""
import pickle, glob, os, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


POP_COLORS = ['steelblue', 'crimson', 'darkorange', 'gray',
              'mediumpurple', 'seagreen', 'saddlebrown', 'deeppink']
CTRL_COLORS = ['steelblue', 'crimson', 'darkorange', 'seagreen']


def load_latest(run_dir):
    snaps = sorted(glob.glob(os.path.join(run_dir, 'snap_*.pkl')))
    if not snaps:
        return None
    with open(snaps[-1], 'rb') as f:
        return pickle.load(f)


def render(runs, titles, out_path, suptitle=None):
    n = len(runs)
    fig, axes = plt.subplots(2, n, figsize=(4.5 * n, 8),
                              gridspec_kw={'hspace': 0.45, 'wspace': 0.3},
                              squeeze=False)

    for col, (run_dir, title) in enumerate(zip(runs, titles)):
        snap = load_latest(run_dir)
        if snap is None:
            print(f"[warn] no snapshots in {run_dir}")
            continue

        ax_pop  = axes[0, col]
        ax_ctrl = axes[1, col]

        t    = np.array(snap['t_traj'])
        pops = snap['populations']
        for i, (name, series) in enumerate(pops.items()):
            ls = '-' if i < 2 else ('--' if i == 2 else ':')
            ax_pop.plot(t, np.array(series), color=POP_COLORS[i % len(POP_COLORS)],
                        lw=2, ls=ls, label=name)

        f_exact  = snap.get('f_exact', None)
        kl_cost  = snap.get('kl_cost', None)
        dv_cost  = snap.get('drift_var_cost', None)
        step     = snap.get('step', None)
        title_bits = [title]
        if step is not None:
            title_bits.append(f'step={step}')
        if f_exact is not None:
            title_bits.append(f'F={f_exact:.3f}')
        if kl_cost is not None:
            title_bits.append(f'KL={kl_cost:.3f}')
        if dv_cost is not None:
            title_bits.append(f'DV={dv_cost:.3f}')
        ax_pop.set_title('  '.join(title_bits), fontsize=10)
        ax_pop.set_xlabel('Time')
        ax_pop.set_ylabel('Population')
        ax_pop.set_xlim(0, t[-1])
        ax_pop.set_ylim(-0.05, 1.05)
        ax_pop.legend(fontsize=8, loc='upper right')
        ax_pop.grid(True, alpha=0.3)

        tc   = np.array(snap['t_ctrl'])
        sigs = np.array(snap['control_signals'])   # (T, n_controls)
        labels = snap.get('control_labels', [f'u_{i}' for i in range(sigs.shape[1])])
        for i in range(sigs.shape[1]):
            ax_ctrl.plot(tc, sigs[:, i], color=CTRL_COLORS[i % len(CTRL_COLORS)],
                         lw=2, label=labels[i])

        # RMS annotations
        rms_bits = [f'{labels[i]}:{np.sqrt(np.mean(sigs[:, i]**2)):.2f}'
                    for i in range(sigs.shape[1])]
        ctrl_title = ' '.join(rms_bits)
        if sigs.shape[1] == 2:
            asym = np.sqrt(np.mean((sigs[:, 0] - sigs[:, 1])**2))
            sym  = np.sqrt(np.mean((sigs[:, 0] + sigs[:, 1])**2))
            ctrl_title = f'asym={asym:.2f}  sym={sym:.2f}  |  ' + ctrl_title
        ax_ctrl.set_title(ctrl_title, fontsize=9)
        ax_ctrl.set_xlabel('Time')
        ax_ctrl.set_ylabel('Amplitude')
        ax_ctrl.set_xlim(0, tc[-1])
        ax_ctrl.legend(fontsize=8)
        ax_ctrl.grid(True, alpha=0.3)

    if suptitle:
        plt.suptitle(suptitle, fontsize=13, y=1.01)

    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--runs',   type=str, required=True,
                   help='Comma-separated list of run directories')
    p.add_argument('--titles', type=str, default=None,
                   help='Comma-separated titles (default: basename of each run)')
    p.add_argument('--out',    type=str, default='snapshots.png')
    p.add_argument('--suptitle', type=str, default=None)
    args = p.parse_args()

    runs = [r.strip() for r in args.runs.split(',')]
    if args.titles:
        titles = [t.strip() for t in args.titles.split(',')]
    else:
        titles = [os.path.basename(os.path.normpath(r)) for r in runs]

    assert len(titles) == len(runs), "titles and runs must have same length"
    render(runs, titles, args.out, suptitle=args.suptitle)


if __name__ == '__main__':
    main()
