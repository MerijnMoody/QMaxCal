"""
types.py - Core type definitions and dynamiqs compatibility layer.

Provides either the real `dynamiqs` library or a minimal mock version
if the library is not installed, ensuring a consistent API.
"""
from __future__ import annotations

from typing import NamedTuple, Callable
import equinox as eqx
import jax.numpy as jnp
from jax import Array
from jaxtyping import PyTree


# =============================================================================
# Try to import dynamiqs, fall back to mock implementation
# =============================================================================

_USING_REAL_DYNAMIQS = False

try:
    import dynamiqs as _dq
    from dynamiqs import *
    from dynamiqs.integrators._utils import astimeqarray, asqarray
    _USING_REAL_DYNAMIQS = True
    dq = _dq

except ImportError:
    # ==========================================================================
    # Mock Implementation of dynamiqs types
    # ==========================================================================

    class QArray:
        """Mock quantum array type compatible with dynamiqs API."""

        def __init__(self, data):
            self.data = jnp.asarray(data)

        @property
        def shape(self):
            return self.data.shape

        def dag(self):
            """Hermitian conjugate (dagger)."""
            return QArray(self.data.T.conj())

        def __matmul__(self, other):
            if isinstance(other, QArray):
                return QArray(self.data @ other.data)
            return QArray(self.data @ other)

        def __add__(self, other):
            if isinstance(other, (TimeQArray, Sum)):
                return Sum([self, other])
            if isinstance(other, QArray):
                return QArray(self.data + other.data)
            return QArray(self.data + other)

        def __sub__(self, other):
            if isinstance(other, QArray):
                return QArray(self.data - other.data)
            return QArray(self.data - other)

        def __mul__(self, scalar):
            return QArray(self.data * scalar)

        def __rmul__(self, scalar):
            return QArray(scalar * self.data)

        def __radd__(self, other):
            # Handle sum() which starts with 0
            if other == 0:
                return self
            return QArray(other + self.data)

        def __repr__(self):
            return f"QArray(shape={self.shape})"

    # Register QArray as a JAX PyTree
    import jax
    jax.tree_util.register_pytree_node(
        QArray,
        lambda q: ((q.data,), None),  # flatten: returns (children, aux_data)
        lambda _, children: QArray(children[0])  # unflatten
    )

    def asqarray(x):
        """Convert to QArray if not already."""
        if isinstance(x, QArray):
            return x
        return QArray(x)

    class TimeQArray:
        """Time-dependent quantum array: f(t) * Q."""

        def __init__(self, f: Callable, q_obj: QArray):
            self.f = f
            self.q_obj = q_obj

        def __call__(self, t):
            return asqarray(self.f(t) * self.q_obj.data)

        def __add__(self, other):
            return Sum([self, other])

    class Sum:
        """Sum of time-dependent quantum arrays."""

        def __init__(self, terms: list):
            self.terms = list(terms)

        def __call__(self, t):
            total = self.terms[0](t).data if hasattr(self.terms[0](t), 'data') else self.terms[0](t)
            for term in self.terms[1:]:
                val = term(t)
                total = total + (val.data if hasattr(val, 'data') else val)
            return asqarray(total)

        def __add__(self, other):
            self.terms.append(other)
            return self

    def astimeqarray(x):
        """Convert to time-dependent QArray."""
        if callable(x):
            return x
        if isinstance(x, (TimeQArray, Sum)):
            return x
        # Return a constant function
        return lambda t: asqarray(x)

    # Mock dq module
    class _DQModule:
        @staticmethod
        def modulated(f, q):
            return TimeQArray(f, q)

    dq = _DQModule()


# =============================================================================
# Solver Options and Method Types
# =============================================================================

class Options(eqx.Module):
    """Solver options."""
    save_states: bool = True
    save_extra: Callable | None = None

    def replace(self, **kwargs):
        return eqx.tree_at(
            lambda tree: [getattr(tree, k) for k in kwargs],
            self,
            list(kwargs.values())
        )


class Method(eqx.Module):
    """Integration method specification."""
    dt: float
    name: str = "euler_maruyama"


def EulerMaruyama(dt: float) -> Method:
    """Create Euler-Maruyama integration method. O(√dt) weak convergence."""
    return Method(dt=dt, name="euler_maruyama")


def Heun(dt: float) -> Method:
    """Create Heun (stochastic RK2) integration method. O(dt) strong convergence."""
    return Method(dt=dt, name="heun")


def ExponentialSplit(dt: float) -> Method:
    """
    Create exponential split-step integration method.

    Uses matrix exponential for deterministic part (exact for linear dynamics)
    and Euler for stochastic part. Eliminates discretization error in Hamiltonian.
    """
    return Method(dt=dt, name="exponential_split")


def StrangSplit(dt: float) -> Method:
    """
    Create Strang (symmetric) split-step integration method.

    Half-step deterministic -> Full stochastic -> Half-step deterministic.
    Gives O(dt²) splitting error while handling stochastic part properly.
    """
    return Method(dt=dt, name="strang_split")


# =============================================================================
# Result Types
# =============================================================================

class DSSESolveResult(NamedTuple):
    """Result from DSSE solver."""
    states: QArray          # Quantum states at saved times
    expects: Array | None   # Expectation values (optional)
    measurements: Array     # Measurement records
    extra: PyTree | None    # Additional data (likelihoods, drifts, etc.)
    tsave: Array            # Time points


def is_using_real_dynamiqs() -> bool:
    """Check if using real dynamiqs library."""
    return _USING_REAL_DYNAMIQS
