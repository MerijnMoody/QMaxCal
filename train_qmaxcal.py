#!/usr/bin/env python3
"""
Generic QMaxCal trainer. Takes a system module path (e.g. `systems.dicke_singlet`)
and trains a FourierController with SSE-based loss + optional regularizers.

Example:
    python train_qmaxcal.py --system systems.dark_route_v3 --kl_weight 1.0 \
        --method exp_split --output_dir results/test
"""
import os, sys, pickle, argparse, importlib
sys.path.insert(0, '/home/mmoody/qmaxcal_paper')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax
import jax.numpy as jnp
from flax import nnx
import optax

from core.types import ExponentialSplit, EulerMaruyama, asqarray, dq
from core.solver import dssesolve


# ============================================================
# Controller
# ============================================================

class FourierController(nnx.Module):
    def __init__(self, n_controls, n_modes, T_final, rngs, init_scale=1.0):
        self.n_controls = n_controls
        self.n_modes    = n_modes
        self.T_final    = T_final
        scale = init_scale / jnp.sqrt(n_modes)
        self.dc         = nnx.Param(jax.random.normal(rngs.params(), (n_controls,)) * init_scale * 0.1)
        self.cos_coeffs = nnx.Param(jax.random.normal(rngs.params(), (n_controls, n_modes)) * scale)
        self.sin_coeffs = nnx.Param(jax.random.normal(rngs.params(), (n_controls, n_modes)) * scale)

    def __call__(self, t):
        phase     = 2 * jnp.pi * t / self.T_final
        modes     = jnp.arange(1, self.n_modes + 1)
        cos_terms = jnp.cos(jnp.outer(modes, phase).T)
        sin_terms = jnp.sin(jnp.outer(modes, phase).T)
        return self.dc + jnp.sum(self.cos_coeffs * cos_terms, axis=-1) \
                       + jnp.sum(self.sin_coeffs * sin_terms, axis=-1)


_TRANSFORMS = {
    None:        lambda u: u,
    'identity':  lambda u: u,
    'abs':       lambda u: jnp.abs(u),
    'square':    lambda u: u ** 2,
    'softplus':  lambda u: jax.nn.softplus(u),
}


def _get_transform(name_or_callable):
    if callable(name_or_callable):
        return name_or_callable
    if name_or_callable in _TRANSFORMS:
        return _TRANSFORMS[name_or_callable]
    raise ValueError(f"unknown control_transform: {name_or_callable!r}")


def build_H_con(ctrl, H_ops_q, transform=None):
    tfm = _get_transform(transform)
    fns = [lambda t, i=i: tfm(ctrl(t))[i] for i in range(len(H_ops_q))]
    H = dq.modulated(fns[0], H_ops_q[0])
    for i in range(1, len(H_ops_q)):
        H = H + dq.modulated(fns[i], H_ops_q[i])
    return H


# ============================================================
# Exact Lindblad RK4 trajectory (for snapshots + plotting)
# ============================================================

def lindblad_trajectory(ctrl, sysdict, n_steps=None):
    """Run exact Lindblad RK4 with trained controller.

    Returns (times, {name: pop_series}) for each entry in sysdict['populations'].

    If ``n_steps`` is None, picks an adaptive value so that the RK4 step
    size satisfies dt <= 0.01 (whatever units ``T_final`` is in).  This
    keeps the integrator stable for systems with strong drive Hamiltonians,
    e.g. local microwave drives at ~200 rad/us on transmons.
    """
    psi0 = sysdict['psi0']
    rho  = (psi0 @ jnp.conj(psi0.T)).astype(jnp.complex64)
    Ls   = sysdict['jump_ops']
    T    = sysdict['T_final']
    if n_steps is None:
        n_steps = max(300, int(T / 0.01))
    dt   = T / n_steps
    H_ops = sysdict['H_ops']
    tfm  = _get_transform(sysdict.get('control_transform', None))

    def lrhs(r, H):
        d = -1j * (H @ r - r @ H)
        for L in Ls:
            Ld = jnp.conj(L.T)
            d = d + L @ r @ Ld - 0.5 * (Ld @ L @ r + r @ Ld @ L)
        return d

    def H_at(t):
        cvec = tfm(ctrl(jnp.array(t)))
        H = sysdict['H_sys'].astype(jnp.complex64)
        for c, H_op in zip(cvec, H_ops):
            H = H + c * H_op
        return H.astype(jnp.complex64)

    names = list(sysdict['populations'].keys())
    projs = [sysdict['populations'][n].astype(jnp.complex64) for n in names]

    times = [0.0]
    pops  = {n: [float(jnp.real(jnp.trace(P @ rho)))] for n, P in zip(names, projs)}

    for step in range(n_steps):
        t  = step * dt
        H1 = H_at(t)
        k1 = lrhs(rho, H1)
        H2 = H_at(t + 0.5 * dt)
        k2 = lrhs(rho + 0.5 * dt * k1, H2)
        k3 = lrhs(rho + 0.5 * dt * k2, H2)
        H3 = H_at(t + dt)
        k4 = lrhs(rho + dt * k3, H3)
        rho = rho + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)

        t_next = (step + 1) * dt
        times.append(float(t_next))
        for n, P in zip(names, projs):
            pops[n].append(float(jnp.real(jnp.trace(P @ rho))))

    return jnp.array(times), {n: jnp.array(v) for n, v in pops.items()}, rho


# ============================================================
# Loss
# ============================================================

def compute_loss(ctrl, sys_q, keys, tsave, n_time_steps,
                 kl_weight=0.0, drift_var_weight=0.0, fluence_weight=0.0,
                 fluence_max=0.0, method='exp_split'):
    dt = sys_q['T_final'] / n_time_steps
    H_con = build_H_con(ctrl, sys_q['H_ops'], transform=sys_q.get('control_transform'))
    integ = ExponentialSplit(dt=dt) if method == 'exp_split' else EulerMaruyama(dt=dt)

    result = dssesolve(
        H_sys=sys_q['H_sys'], jump_ops=sys_q['jump_ops'],
        psi0=sys_q['psi0'], tsave=tsave, keys=keys,
        H_con=H_con, method=integ,
        nsteps=n_time_steps, nsteps_per_save=1,
    )

    # exp_p: (n_trajs, n_times, n_jumps) → (n_trajs, n_jumps, n_times)
    exp_p = jnp.transpose(result.extra['exp_p'], (0, 2, 1))

    # Fidelity
    final_psi = result.states.data[:, -1, :, :]
    target    = sys_q['target_state'].data
    overlaps  = jnp.einsum('ij,nij->n', jnp.conj(target), final_psi)
    mean_fidelity = jnp.mean(jnp.abs(overlaps) ** 2)

    # KL = ½ E[∫ α² dt]
    alpha    = exp_p
    alpha_sq = jnp.sum(alpha ** 2, axis=1)
    kl_cost  = 0.5 * jnp.mean(jnp.trapezoid(alpha_sq, tsave, axis=-1))

    # Sum of per-channel temporal variances:
    #   DV = Σ_k E_ψ[ Var_t[α_k] ]  =  Σ_k E_ψ[ ⟨α_k²⟩_t − ⟨α_k⟩_t² ]
    T_final         = tsave[-1]
    alpha_mean_k    = jnp.trapezoid(alpha,      tsave, axis=-1) / T_final   # (n_trajs, n_jumps)
    alpha_sq_mean_k = jnp.trapezoid(alpha ** 2, tsave, axis=-1) / T_final   # (n_trajs, n_jumps)
    var_per_channel = alpha_sq_mean_k - alpha_mean_k ** 2                    # (n_trajs, n_jumps)
    drift_var_cost  = jnp.mean(jnp.sum(var_per_channel, axis=-1))

    # Fluence ∫ |g|² dt — use physical (post-transform) controls
    tfm      = _get_transform(sys_q.get('control_transform'))
    t_sample = jnp.linspace(0, ctrl.T_final, 100)
    controls = jax.vmap(ctrl)(t_sample)
    phys_g   = tfm(controls)
    fluence  = jnp.trapezoid(jnp.sum(phys_g ** 2, axis=-1), t_sample)

    # Hinge-constraint form when fluence_max > 0; plain linear penalty otherwise.
    fluence_excess = jnp.where(fluence_max > 0,
                                jnp.maximum(0.0, fluence - fluence_max),
                                fluence)

    loss = (1.0 - mean_fidelity) \
         + kl_weight * kl_cost \
         + drift_var_weight * drift_var_cost \
         + fluence_weight * fluence_excess
    return loss, dict(fidelity=mean_fidelity, kl_cost=kl_cost,
                      drift_var_cost=drift_var_cost,
                      fluence=fluence, fluence_excess=fluence_excess, loss=loss)


# ============================================================
# Wrap raw arrays into QArrays for dssesolve
# ============================================================

def _wrap_system(sysdict):
    """Return a copy of sysdict with H_sys/H_ops/jump_ops/psi0/target wrapped in asqarray."""
    return {
        **sysdict,
        'H_sys':        asqarray(sysdict['H_sys']),
        'H_ops':        [asqarray(H) for H in sysdict['H_ops']],
        'jump_ops':     [asqarray(L) for L in sysdict['jump_ops']],
        'psi0':         asqarray(sysdict['psi0']),
        'target_state': asqarray(sysdict['target_state']),
    }


# ============================================================
# Training
# ============================================================

def train(system_mod, system_kwargs, output_dir,
          kl_weight=0.0, drift_var_weight=0.0, fluence_weight=0.0,
          fluence_warmup_steps=0, fluence_max=0.0,
          drift_var_warmup_steps=0,
          seed=0, n_time_steps=256, n_trajs=128, num_steps=10000,
          lr=1e-3, n_modes=16, init_scale=1.0, snap_every=500,
          method='exp_split'):

    sysdict_raw = system_mod.create_system(**system_kwargs)
    sys_q       = _wrap_system(sysdict_raw)

    run_name = f"{sysdict_raw['system_name']}_kl{kl_weight}_dv{drift_var_weight}_fl{fluence_weight}_seed{seed}"
    run_dir  = os.path.join(output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    print(f"Run: {run_name}")
    print(f"  method={method}  n_time_steps={n_time_steps}  n_trajs={n_trajs}  "
          f"num_steps={num_steps}")

    T_final = sysdict_raw['T_final']
    n_ctrls = sysdict_raw['n_controls']

    rngs = nnx.Rngs(seed)
    ctrl = FourierController(n_controls=n_ctrls, n_modes=n_modes,
                             T_final=T_final, rngs=rngs, init_scale=init_scale)

    static, params = nnx.split(ctrl)
    opt       = optax.adam(lr)
    opt_state = opt.init(params)
    tsave     = jnp.linspace(0, T_final, n_time_steps + 1)

    def loss_fn(p, keys, fluence_w, dv_w):
        m = nnx.merge(static, p)
        return compute_loss(m, sys_q, keys, tsave, n_time_steps,
                            kl_weight=kl_weight, drift_var_weight=dv_w,
                            fluence_weight=fluence_w, fluence_max=fluence_max,
                            method=method)

    @jax.jit
    def train_step(p, s, keys, fluence_w, dv_w):
        (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(p, keys, fluence_w, dv_w)
        updates, ns = opt.update(grads, s, p)
        return optax.apply_updates(p, updates), ns, metrics

    key = jax.random.PRNGKey(seed)
    best_fidelity = -1.0
    best_step     = 0
    best_snap     = None
    # Track best-by-training-loss every step (not just at snapshot cadence).
    # Diamond-style runs can fluctuate heavily, so the snapshot-only "best"
    # may miss the best optimizer pass.
    best_train_loss   = float('inf')
    best_train_step   = 0
    best_train_params = params
    best_train_metrics = None
    t_ctrl_plot   = jnp.linspace(0, T_final, 200)

    for step in range(num_steps):
        key, sub = jax.random.split(key)
        keys = jax.random.split(sub, n_trajs)
        fw = jnp.array(0.0 if step < fluence_warmup_steps else fluence_weight)
        dvw = jnp.array(0.0 if step < drift_var_warmup_steps else drift_var_weight)
        params, opt_state, metrics = train_step(params, opt_state, keys, fw, dvw)

        loss_val = float(metrics['loss'])
        if loss_val < best_train_loss:
            best_train_loss    = loss_val
            best_train_step    = step
            best_train_params  = params
            best_train_metrics = metrics

        if step % 100 == 0:
            print(f"  step {step:5d}: F={float(metrics['fidelity']):.4f}  "
                  f"loss={loss_val:.4f}  (best_loss={best_train_loss:.4f}@{best_train_step})  "
                  f"KL={float(metrics['kl_cost']):.4f}  "
                  f"DV={float(metrics['drift_var_cost']):.4f}  "
                  f"Flu={float(metrics['fluence']):.4f}")

        if step % snap_every == 0 or step == num_steps - 1:
            ctrl_eval = nnx.merge(static, params)
            t_traj, pops, rho_f = lindblad_trajectory(ctrl_eval, sysdict_raw)

            tgt      = sysdict_raw['target_state']
            tgt_proj = (tgt @ jnp.conj(tgt.T)).astype(jnp.complex64)
            f_exact  = float(jnp.real(jnp.trace(tgt_proj @ rho_f)))

            tfm_plot = _get_transform(sysdict_raw.get('control_transform'))
            sigs = tfm_plot(jax.vmap(ctrl_eval)(t_ctrl_plot))

            snap = dict(
                step=step, f_exact=f_exact,
                kl_cost=float(metrics['kl_cost']),
                drift_var_cost=float(metrics['drift_var_cost']),
                fluence=float(metrics['fluence']),
                t_traj=jnp.asarray(t_traj),
                populations={k: jnp.asarray(v) for k, v in pops.items()},
                t_ctrl=jnp.asarray(t_ctrl_plot),
                control_signals=jnp.asarray(sigs),
                control_labels=sysdict_raw['control_labels'],
                system_name=sysdict_raw['system_name'],
                method=method,
            )
            # Forbidden-state stats (if the system declared them)
            forbidden = sysdict_raw.get('forbidden_states', [])
            forbidden_stats = {}
            for name in forbidden:
                if name in pops:
                    p = jnp.asarray(pops[name])
                    forbidden_stats[name] = {
                        'max': float(jnp.max(p)),
                        'integral': float(jnp.trapezoid(p, t_traj)),
                        'final': float(p[-1]),
                    }
            snap['forbidden_stats'] = forbidden_stats

            with open(os.path.join(run_dir, f'snap_{step:06d}.pkl'), 'wb') as f:
                pickle.dump(snap, f)
            fb_msg = "  ".join(f"{n}:max={s['max']:.3f},∫={s['integral']:.3f}"
                                for n, s in forbidden_stats.items())
            print(f"    [snap] step={step}  F_exact={f_exact:.4f}"
                  + (f"  [forbidden: {fb_msg}]" if fb_msg else ""))

            if f_exact > best_fidelity:
                best_fidelity = f_exact
                best_step     = step
                best_snap     = snap

    # Final checkpoint (best by snapshot f_exact)
    ctrl_final = nnx.merge(static, params)
    ckpt = dict(
        system_mod=system_mod.__name__, system_kwargs=system_kwargs,
        kl_weight=kl_weight, drift_var_weight=drift_var_weight,
        fluence_weight=fluence_weight, seed=seed,
        n_modes=n_modes, T_final=T_final, method=method,
        best_exact_fidelity=best_fidelity, best_step=best_step,
        best_dc=ctrl_final.dc.value,
        best_cos_coeffs=ctrl_final.cos_coeffs.value,
        best_sin_coeffs=ctrl_final.sin_coeffs.value,
        t_ctrl=jnp.asarray(t_ctrl_plot),
        best_control_signals=best_snap['control_signals'],
    )
    with open(os.path.join(run_dir, 'controller.pkl'), 'wb') as f:
        pickle.dump(ckpt, f)
    print(f"Done. Best F (by snapshot f_exact) = {best_fidelity:.4f} at step {best_step}")

    # --- Best-by-training-loss: evaluate the step-level best params with exact Lindblad ---
    ctrl_bt  = nnx.merge(static, best_train_params)
    t_bt, pops_bt, rho_bt = lindblad_trajectory(ctrl_bt, sysdict_raw)
    tgt      = sysdict_raw['target_state']
    tgt_proj = (tgt @ jnp.conj(tgt.T)).astype(jnp.complex64)
    f_bt     = float(jnp.real(jnp.trace(tgt_proj @ rho_bt)))
    tfm_bt   = _get_transform(sysdict_raw.get('control_transform'))
    sigs_bt  = tfm_bt(jax.vmap(ctrl_bt)(t_ctrl_plot))
    forbidden_stats_bt = {}
    for name in sysdict_raw.get('forbidden_states', []):
        if name in pops_bt:
            p = jnp.asarray(pops_bt[name])
            forbidden_stats_bt[name] = {
                'max':      float(jnp.max(p)),
                'integral': float(jnp.trapezoid(p, t_bt)),
                'final':    float(p[-1]),
            }
    snap_bt  = dict(
        step=best_train_step, f_exact=f_bt,
        kl_cost=float(best_train_metrics['kl_cost']),
        drift_var_cost=float(best_train_metrics['drift_var_cost']),
        fluence=float(best_train_metrics['fluence']),
        train_loss=best_train_loss,
        t_traj=jnp.asarray(t_bt),
        populations={k: jnp.asarray(v) for k, v in pops_bt.items()},
        t_ctrl=jnp.asarray(t_ctrl_plot),
        control_signals=jnp.asarray(sigs_bt),
        control_labels=sysdict_raw['control_labels'],
        system_name=sysdict_raw['system_name'],
        method=method,
        forbidden_stats=forbidden_stats_bt,
    )
    with open(os.path.join(run_dir, 'snap_best_by_loss.pkl'), 'wb') as f:
        pickle.dump(snap_bt, f)
    ckpt_bt = dict(
        system_mod=system_mod.__name__, system_kwargs=system_kwargs,
        kl_weight=kl_weight, drift_var_weight=drift_var_weight,
        fluence_weight=fluence_weight, seed=seed,
        n_modes=n_modes, T_final=T_final, method=method,
        best_train_loss=best_train_loss, best_train_step=best_train_step,
        f_exact_at_best_loss=f_bt,
        best_dc=ctrl_bt.dc.value,
        best_cos_coeffs=ctrl_bt.cos_coeffs.value,
        best_sin_coeffs=ctrl_bt.sin_coeffs.value,
        t_ctrl=jnp.asarray(t_ctrl_plot),
        best_control_signals=jnp.asarray(sigs_bt),
    )
    with open(os.path.join(run_dir, 'controller_best_by_loss.pkl'), 'wb') as f:
        pickle.dump(ckpt_bt, f)
    print(f"Done. Best by train loss = {best_train_loss:.4f} at step {best_train_step}  "
          f"-> F_exact = {f_bt:.4f}")


# ============================================================
# CLI
# ============================================================

def main():
    # Two-pass parsing: system first, so the system can add its own args.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--system', type=str, required=True,
                     help='Dotted path, e.g. systems.dark_route_v3')
    pre_args, _ = pre.parse_known_args()

    system_mod = importlib.import_module(pre_args.system)

    p = argparse.ArgumentParser(parents=[pre])
    p.add_argument('--output_dir',       type=str,   default='results')
    p.add_argument('--seed',             type=int,   default=0)
    p.add_argument('--method',           type=str,   default='exp_split',
                   choices=['exp_split', 'euler'])
    # regularizers
    p.add_argument('--kl_weight',        type=float, default=0.0)
    p.add_argument('--drift_var_weight', type=float, default=0.0)
    p.add_argument('--fluence_weight',   type=float, default=0.0)
    p.add_argument('--fluence_warmup_steps', type=int, default=0,
                   help='Apply fluence only after this many steps (ramp-on)')
    p.add_argument('--drift_var_warmup_steps', type=int, default=0,
                   help='Apply DV only after this many steps (lets fidelity term find a basin first)')
    p.add_argument('--fluence_max',      type=float, default=0.0,
                   help='If >0, use hinge constraint: penalty = fluence_weight*max(0, fluence - fluence_max). '
                        'If 0, uses plain linear penalty (default).')
    # training hyperparams
    p.add_argument('--n_time_steps',     type=int,   default=256)
    p.add_argument('--n_trajs',          type=int,   default=128)
    p.add_argument('--num_steps',        type=int,   default=10000)
    p.add_argument('--lr',               type=float, default=1e-3)
    p.add_argument('--n_modes',          type=int,   default=16)
    p.add_argument('--init_scale',       type=float, default=1.0)
    p.add_argument('--snap_every',       type=int,   default=500)

    system_mod.add_system_args(p)
    args = p.parse_args()

    # Extract system kwargs: anything the system added goes into system_kwargs.
    # Easiest way: gather whatever attributes are on `args` but not on the generic list.
    generic = {'system', 'output_dir', 'seed', 'method', 'kl_weight', 'drift_var_weight',
               'fluence_weight', 'fluence_warmup_steps', 'fluence_max',
               'drift_var_warmup_steps',
               'n_time_steps', 'n_trajs', 'num_steps', 'lr',
               'n_modes', 'init_scale', 'snap_every'}
    system_kwargs = {k: v for k, v in vars(args).items() if k not in generic}

    train(
        system_mod=system_mod, system_kwargs=system_kwargs, output_dir=args.output_dir,
        kl_weight=args.kl_weight, drift_var_weight=args.drift_var_weight,
        fluence_weight=args.fluence_weight,
        fluence_warmup_steps=args.fluence_warmup_steps,
        fluence_max=args.fluence_max,
        drift_var_warmup_steps=args.drift_var_warmup_steps,
        seed=args.seed,
        n_time_steps=args.n_time_steps, n_trajs=args.n_trajs,
        num_steps=args.num_steps, lr=args.lr, n_modes=args.n_modes,
        init_scale=args.init_scale, snap_every=args.snap_every,
        method=args.method,
    )


if __name__ == '__main__':
    main()
