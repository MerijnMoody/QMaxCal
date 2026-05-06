"""
4-qubit XX spin chain — excitation transfer with site-resolved decoherence.

Hilbert space: 4 qubits, dim = 16. Basis |q₀ q₁ q₂ q₃⟩ in standard kron order
(q₀ slowest → q₃ fastest index).

3 tunable XX nearest-neighbor couplings (the controls):
    H_{i,i+1} = ½ (σ_x^i σ_x^{i+1} + σ_y^i σ_y^{i+1})
              = σ_+^i σ_-^{i+1} + σ_-^i σ_+^{i+1}
These are excitation-preserving (commute with total Sz), so single-excitation
dynamics stay in a 4-dim subspace + vacuum, but the optimizer can still
temporarily leave it by symmetry-breaking phases picked up through the noise.

8 Lindblad operators — 2 per site:
    L_{κ,i} = √κ_i · σ_z^i      (local dephasing on site i)         i = 0..3
    L_{γ,i} = √γ_i · σ_-^i      (local T₁ decay on site i)         i = 0..3

Default rates are inhomogeneous (site-dependent) so the chain has "fast"
and "slow" sites and dark-path selection actually matters.

Task: transfer excitation from site 0 to site 3:
    ψ₀ = |1000⟩   →   ψ_t = |0001⟩

Pathways:
  (A) Bucket brigade: strong pulses g₀₁, g₁₂, g₂₃ in sequence. Populates each
      middle site fully — pays dephasing at sites 1 and 2.
  (B) Perfect State Transfer (PST, Christandl et al. 2004): couplings set as
      g_{i,i+1} = J·√(i+1)(N−i−1)/N executes a σ_x^(⊗N) mirror isometry in
      time π/(2J) with SMALL peak occupation at middle sites.
  (C) Dark-mode / eigenmode transfer: uniform couplings give standing-wave
      eigenmodes. Parametric ramp drives the boundary-localized eigenmode.

(B) and (C) differ in how much time they spend on each intermediate site —
which the Wiener-KL directly penalises via the per-site dephasing channels.
"""
import jax.numpy as jnp


N = 4   # 4 qubits


def _kron_list(ops):
    """Kronecker product of a list of 2x2 matrices."""
    out = ops[0]
    for op in ops[1:]:
        out = jnp.kron(out, op)
    return out


def _single_site(op_2x2, site, n_sites=N):
    """Embed a 2x2 operator on `site` into the full n_sites Hilbert space."""
    I2 = jnp.eye(2, dtype=jnp.complex64)
    ops = [I2] * n_sites
    ops[site] = jnp.asarray(op_2x2, dtype=jnp.complex64)
    return _kron_list(ops)


def _two_site(A_2x2, B_2x2, i, j, n_sites=N):
    """Embed A⊗B on sites (i, j) into full Hilbert space."""
    I2 = jnp.eye(2, dtype=jnp.complex64)
    ops = [I2] * n_sites
    ops[i] = jnp.asarray(A_2x2, dtype=jnp.complex64)
    ops[j] = jnp.asarray(B_2x2, dtype=jnp.complex64)
    return _kron_list(ops)


# 2x2 Paulis
sx  = jnp.array([[0, 1], [1, 0]],  dtype=jnp.complex64)
sy  = jnp.array([[0, -1j], [1j, 0]], dtype=jnp.complex64)
sz  = jnp.array([[1, 0], [0, -1]], dtype=jnp.complex64)
sp  = jnp.array([[0, 1], [0, 0]], dtype=jnp.complex64)   # σ₊ = |↑⟩⟨↓|
sm  = jnp.array([[0, 0], [1, 0]], dtype=jnp.complex64)   # σ₋ = |↓⟩⟨↑|


def create_system(kappa_global: float = 0.3, gamma_global: float = 0.05,
                  kappa_edge_boost: float = 2.0,
                  gamma_edge_boost: float = 1.0,
                  T_final: float = 3.0, **_):
    """Build the 4-site spin chain.

    kappa_global:  baseline dephasing rate on each site.
    gamma_global:  baseline T₁ decay rate.
    kappa_edge_boost: multiplier on the endpoint dephasings (sites 0, 3) — the
        "bath contact" is stronger at the ends, a physically common pattern.
    """
    dim = 2 ** N

    H_sys = jnp.zeros((dim, dim), dtype=jnp.complex64)

    # Control Hamiltonians: 3 nearest-neighbor XX couplings
    H_ops = []
    for i in range(N - 1):
        H_xx = _two_site(sx, sx, i, i + 1)
        H_yy = _two_site(sy, sy, i, i + 1)
        H_ops.append(0.5 * (H_xx + H_yy))    # = σ₊ᵢ σ₋_{i+1} + σ₋ᵢ σ₊_{i+1}

    # Site-resolved decoherence rates (inhomogeneous)
    kappas = [kappa_edge_boost * kappa_global,   # site 0 (strong)
              kappa_global,                      # site 1
              kappa_global,                      # site 2
              kappa_edge_boost * kappa_global]   # site 3 (strong)
    gammas = [gamma_edge_boost * gamma_global,   # site 0
              gamma_global,                      # site 1
              gamma_global,                      # site 2
              gamma_edge_boost * gamma_global]   # site 3

    jump_ops = []
    for i in range(N):
        jump_ops.append(jnp.sqrt(jnp.array(kappas[i], dtype=jnp.float32))
                         * _single_site(sz, i))
        jump_ops.append(jnp.sqrt(jnp.array(gammas[i], dtype=jnp.float32))
                         * _single_site(sm, i))

    # Initial and target: excitation localized at site 0 / site 3
    # Spin convention: σ_z|↑⟩ = +|↑⟩, σ_z|↓⟩ = −|↓⟩. Computational basis index:
    # |q₀q₁q₂q₃⟩ with qᵢ = 0 for |↑⟩, 1 for |↓⟩ (standard jnp.kron order).
    def basis(bits):
        idx = 0
        for b in bits:
            idx = idx * 2 + b
        v = jnp.zeros((dim, 1), dtype=jnp.complex64)
        return v.at[idx, 0].set(1.0)

    # "Excitation" = |↑⟩ at a site, |↓⟩ elsewhere (we call |↑⟩ the excited).
    # Put excitation at site 0: bits = (0, 1, 1, 1)
    psi0   = basis([0, 1, 1, 1])   # |↑↓↓↓⟩
    target = basis([1, 1, 1, 0])   # |↓↓↓↑⟩   (excitation at site 3)

    # Populations: per-site excitation P(q_i = ↑)
    populations = {}
    for i in range(N):
        # P(site i excited) = projector onto  σ_z_i-eigenvalue +1 subspace
        # = (I + σ_z_i)/2
        proj = 0.5 * (jnp.eye(dim, dtype=jnp.complex64)
                       + _single_site(sz, i))
        populations[f'P(site {i} ↑)'] = proj

    # Also track total excitation count
    populations['⟨n_tot⟩'] = sum(
        0.5 * (jnp.eye(dim, dtype=jnp.complex64) + _single_site(sz, i))
        for i in range(N)
    )

    return dict(
        H_sys=H_sys, H_ops=H_ops, jump_ops=jump_ops,
        psi0=psi0, target_state=target,
        T_final=T_final, n_dim=dim, n_controls=N - 1,
        populations=populations,
        forbidden_states=[f'P(site {i} ↑)' for i in [1, 2]],  # middle sites
        control_labels=[f'g_{i}{i+1}' for i in range(N - 1)],
        signal_scale=3.0,
        action_range='signed',
        system_name=(f'chain4_k{kappa_global}_g{gamma_global}'
                      f'_boost{kappa_edge_boost}_T{T_final}'),
    )


def add_system_args(parser):
    parser.add_argument('--kappa_global', type=float, default=0.3,
                        help='Baseline dephasing rate κ per site')
    parser.add_argument('--gamma_global', type=float, default=0.05,
                        help='Baseline T₁ decay rate γ per site')
    parser.add_argument('--kappa_edge_boost', type=float, default=2.0,
                        help='Endpoint dephasing multiplier (stronger bath contact)')
    parser.add_argument('--gamma_edge_boost', type=float, default=1.0,
                        help='Endpoint T₁ decay multiplier (extra noise on initial/final sites)')
    parser.add_argument('--T_final',      type=float, default=3.0)
