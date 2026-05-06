"""
training.py - Training loop and loss functions for quantum control.

Provides various loss functions for optimizing quantum controllers:
1. Fidelity-only: Pure state transfer optimization
2. Fidelity + KL: Thermodynamically-informed control
3. Fidelity + Entropy: Forward-backward entropy minimization
"""
from __future__ import annotations

from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
import optax
from flax import nnx
from jax.scipy.integrate import trapezoid

from .types import QArray, DSSESolveResult, EulerMaruyama, asqarray
from .solver import dssesolve
from .system import SystemDefinition, FourierController, build_H_con_modulated
from .thermodynamics import (
    compute_reverse_operators,
    evolve_backward,
    compute_entropy_production_proxy,
)


# =============================================================================
# Loss Functions
# =============================================================================

def compute_fidelity(
    final_states: jax.Array,
    target_state: QArray
) -> jax.Array:
    """
    Compute average fidelity from trajectory ensemble.

    Args:
        final_states: Final states from trajectories, shape (n_trajs, dim, 1)
        target_state: Target state |ψ_target⟩

    Returns:
        Fidelity ⟨target|ρ|target⟩
    """
    # Average density matrix
    rho_final = jnp.mean(
        final_states @ jnp.swapaxes(final_states, -1, -2).conj(),
        axis=0
    )
    target_vec = target_state.data.squeeze()
    fidelity = jnp.real(jnp.conj(target_vec) @ rho_final @ target_vec)
    return fidelity


def fidelity_loss_fn(
    c_params,
    static,
    key: jax.Array,
    system: SystemDefinition,
    H_ops_q: list[QArray],
    tsave: jax.Array,
    n_trajs: int,
    nsteps_total: int,
    nsteps_per_save: int,
    dt: float
) -> tuple[jax.Array, tuple]:
    """
    Simple fidelity-based loss function.

    Loss = 1 - Fidelity

    Args:
        c_params: Controller parameters (from nnx.split)
        static: Static controller definition
        key: PRNG key
        system: System definition
        H_ops_q: Control Hamiltonian operators
        tsave: Time save points
        n_trajs: Number of trajectories
        nsteps_total: Total integration steps
        nsteps_per_save: Steps between saves
        dt: Integration timestep

    Returns:
        (loss, (infidelity,))
    """
    model = nnx.merge(static, c_params)
    H_con = build_H_con_modulated(model, H_ops_q)
    keys = jax.random.split(key, n_trajs)

    result = dssesolve(
        H_sys=system.H_sys,
        jump_ops=system.jump_ops,
        psi0=system.psi0,
        tsave=tsave,
        keys=keys,
        H_con=H_con,
        method=EulerMaruyama(dt=dt),
        nsteps=nsteps_total,
        nsteps_per_save=nsteps_per_save
    )

    final_states = result.states.data[:, -1, :, :]
    fidelity = compute_fidelity(final_states, system.target_state)
    infidelity = 1.0 - fidelity

    return infidelity, (infidelity,)


def fidelity_kl_loss_fn(
    c_params,
    static,
    key: jax.Array,
    system: SystemDefinition,
    H_ops_q: list[QArray],
    tsave: jax.Array,
    n_trajs: int,
    nsteps_total: int,
    nsteps_per_save: int,
    dt: float,
    kl_weight: float = 1.0
) -> tuple[jax.Array, tuple]:
    """
    Fidelity + KL divergence loss (Girsanov cost).

    Loss = (1 - Fidelity) + kl_weight * KL(P || Q)

    where KL is computed via likelihood ratios between controlled
    and uncontrolled processes.

    Args:
        kl_weight: Weight on KL divergence term

    Returns:
        (loss, (infidelity, kl_cost))
    """
    model = nnx.merge(static, c_params)
    H_con = build_H_con_modulated(model, H_ops_q)
    keys = jax.random.split(key, n_trajs)

    result = dssesolve(
        H_sys=system.H_sys,
        jump_ops=system.jump_ops,
        psi0=system.psi0,
        tsave=tsave,
        keys=keys,
        H_con=H_con,
        method=EulerMaruyama(dt=dt),
        nsteps=nsteps_total,
        nsteps_per_save=nsteps_per_save
    )

    # Fidelity
    final_states = result.states.data[:, -1, :, :]
    fidelity = compute_fidelity(final_states, system.target_state)
    infidelity = 1.0 - fidelity

    # KL divergence via likelihoods
    kl_cost = jnp.mean(
        result.extra['log_p_p'][:, -1] - result.extra['log_p_q'][:, -1]
    )

    loss = infidelity + kl_weight * kl_cost
    return loss, (infidelity, kl_cost)


def fidelity_entropy_loss_fn(
    c_params,
    static,
    key: jax.Array,
    system: SystemDefinition,
    H_ops_q: list[QArray],
    H_rev: QArray,
    L_rev_list: list[QArray],
    tsave: jax.Array,
    n_trajs: int,
    nsteps_total: int,
    nsteps_per_save: int,
    dt: float,
    T_final: float,
    entropy_weight: float = 1.0
) -> tuple[jax.Array, tuple]:
    """
    Fidelity + Forward-backward entropy loss (Manzano protocol).

    Loss = (1 - Fidelity) + entropy_weight * S_proxy

    where S_proxy = ½∫(α_F - α_B)² dt is the entropy production proxy.

    Args:
        H_rev: Reversed system Hamiltonian
        L_rev_list: Reversed jump operators
        T_final: Total evolution time
        entropy_weight: Weight on entropy term

    Returns:
        (loss, (infidelity, entropy_cost))
    """
    model = nnx.merge(static, c_params)
    H_con = build_H_con_modulated(model, H_ops_q)
    keys = jax.random.split(key, n_trajs)
    dt_save = tsave[1] - tsave[0]

    # Forward evolution
    result = dssesolve(
        H_sys=system.H_sys,
        jump_ops=system.jump_ops,
        psi0=system.psi0,
        tsave=tsave,
        keys=keys,
        H_con=H_con,
        method=EulerMaruyama(dt=dt),
        nsteps=nsteps_total,
        nsteps_per_save=nsteps_per_save
    )

    states = result.states.data
    measurements = result.measurements[..., 0]

    # Forward drift α_F = ⟨L + L†⟩
    L_op = system.jump_ops[0]
    obs = L_op + L_op.dag()

    def get_drift(s):
        return (asqarray(s).dag() @ obs @ asqarray(s)).squeeze().real

    alpha_F = jax.vmap(jax.vmap(get_drift))(states)[:, :-1]

    # Backward evolution
    I_bwd = measurements[:, ::-1]
    t_axis_bwd = tsave[:-1]

    def H_con_bwd_func(t):
        """Active backward driving: u(T-t)"""
        return H_con(T_final - t)

    def run_backward(I_single):
        drifts, _ = evolve_backward(
            H_rev.data,
            [L.data for L in L_rev_list],
            system.target_state.data,
            I_single,
            dt_save,
            H_con_bwd_func=H_con_bwd_func,
            t_axis_bwd=t_axis_bwd
        )
        return drifts

    alpha_bwd_raw = jax.vmap(run_backward)(I_bwd)
    alpha_B = alpha_bwd_raw[:, ::-1]  # Map back to forward time

    # Entropy cost
    entropy_cost = compute_entropy_production_proxy(alpha_F, alpha_B, dt_save)

    # Fidelity
    final_states = states[:, -1, :, :]
    fidelity = compute_fidelity(final_states, system.target_state)
    infidelity = 1.0 - fidelity

    loss = infidelity + entropy_weight * entropy_cost
    return loss, (infidelity, entropy_cost)


# =============================================================================
# Training Loop
# =============================================================================

def train_controller(
    system: SystemDefinition,
    H_ops_q: list[QArray],
    n_fourier_basis: int = 12,
    num_steps: int = 200,
    n_trajs: int = 100,
    dt: float = 1e-3,
    T_final: float = 2.0,
    lr: float = 1e-2,
    loss_type: str = 'fidelity',
    kl_weight: float = 1.0,
    entropy_weight: float = 1.0,
    seed: int = 0,
    verbose: bool = True
) -> tuple[nnx.Module, list[dict]]:
    """
    Train a Fourier controller for quantum state transfer.

    Args:
        system: System definition
        H_ops_q: Control Hamiltonian operators
        n_fourier_basis: Number of Fourier modes
        num_steps: Training iterations
        n_trajs: Trajectories per gradient estimate
        dt: Integration timestep
        T_final: Protocol duration
        lr: Learning rate
        loss_type: One of 'fidelity', 'fidelity_kl', 'fidelity_entropy'
        kl_weight: Weight for KL term (if loss_type='fidelity_kl')
        entropy_weight: Weight for entropy term (if loss_type='fidelity_entropy')
        seed: Random seed
        verbose: Print progress

    Returns:
        (trained_controller, history)
    """
    # Setup time grid
    tsave = jnp.linspace(0, T_final, 101)
    nsteps_total = int(round(T_final / dt))
    nsteps_per_save = nsteps_total // (len(tsave) - 1)

    # Initialize controller
    controller = FourierController(
        n_controls=system.n_controls,
        n_fourier_basis=n_fourier_basis,
        T_final=T_final,
        rngs=nnx.Rngs(seed)
    )

    static, params = nnx.split(controller)
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)

    # Setup reverse operators if needed
    H_rev, L_rev_list = None, None
    if loss_type == 'fidelity_entropy':
        H_rev, L_rev_list = compute_reverse_operators(system.H_sys, system.jump_ops)

    # Select loss function
    if loss_type == 'fidelity':
        @jax.jit
        def loss_fn(p, k):
            return fidelity_loss_fn(
                p, static, k, system, H_ops_q,
                tsave, n_trajs, nsteps_total, nsteps_per_save, dt
            )
    elif loss_type == 'fidelity_kl':
        @jax.jit
        def loss_fn(p, k):
            return fidelity_kl_loss_fn(
                p, static, k, system, H_ops_q,
                tsave, n_trajs, nsteps_total, nsteps_per_save, dt,
                kl_weight=kl_weight
            )
    elif loss_type == 'fidelity_entropy':
        @partial(jax.jit, static_argnames=('nsteps_total', 'nsteps_per_save'))
        def loss_fn(p, k):
            return fidelity_entropy_loss_fn(
                p, static, k, system, H_ops_q,
                H_rev, L_rev_list,
                tsave, n_trajs, nsteps_total, nsteps_per_save, dt,
                T_final, entropy_weight=entropy_weight
            )
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    # Training loop
    @jax.jit
    def update(params, opt_state, key):
        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, key)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss, aux

    key = jax.random.PRNGKey(seed)
    history = []

    for step in range(num_steps):
        key, subkey = jax.random.split(key)
        params, opt_state, loss, aux = update(params, opt_state, subkey)

        record = {'step': step, 'loss': float(loss)}
        if loss_type == 'fidelity':
            record['infidelity'] = float(aux[0])
        elif loss_type == 'fidelity_kl':
            record['infidelity'] = float(aux[0])
            record['kl_cost'] = float(aux[1])
        elif loss_type == 'fidelity_entropy':
            record['infidelity'] = float(aux[0])
            record['entropy_cost'] = float(aux[1])

        history.append(record)

        if verbose and step % 20 == 0:
            if loss_type == 'fidelity':
                print(f"Step {step:04d} | Loss: {loss:.4f} | Infid: {aux[0]:.4f}")
            elif loss_type == 'fidelity_kl':
                print(f"Step {step:04d} | Loss: {loss:.4f} | Infid: {aux[0]:.4f} | KL: {aux[1]:.4f}")
            elif loss_type == 'fidelity_entropy':
                print(f"Step {step:04d} | Loss: {loss:.4f} | Infid: {aux[0]:.4f} | S: {aux[1]:.4f}")

    return nnx.merge(static, params), history
