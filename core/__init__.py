"""
QMaxCal Clean - Quantum Maximum Caliber Framework

A framework for thermodynamically-informed quantum control optimization
using the Maximum Caliber formalism and stochastic Schrödinger equations.
"""

from .types import QArray, TimeQArray, DSSESolveResult, Options, Method, EulerMaruyama
from .types import asqarray, astimeqarray, dq
from .system import SystemDefinition, FourierController, build_H_con_modulated
from .solver import dssesolve, LikelihoodState
from .thermodynamics import (
    compute_reverse_operators,
    evolve_backward,
    compute_entropy_production,
    compute_girsanov_kl,
)
from .density_matrix import (
    SMESolveResult,
    TPMResult,
    DensityMatrixState,
    dsmesolve,
    run_tpm_protocol,
    compute_total_entropy_production,
    compute_entropy_selfadjoint,
    measure_energy_basis,
    lindblad_dissipator,
    measurement_backaction,
)

__all__ = [
    # Types
    "QArray", "TimeQArray", "DSSESolveResult", "Options", "Method", "EulerMaruyama",
    "asqarray", "astimeqarray", "dq",
    # System
    "SystemDefinition", "FourierController", "build_H_con_modulated",
    # Solver (pure state)
    "dssesolve", "LikelihoodState",
    # Solver (density matrix)
    "SMESolveResult", "TPMResult", "DensityMatrixState",
    "dsmesolve", "run_tpm_protocol",
    # Thermodynamics
    "compute_reverse_operators", "evolve_backward",
    "compute_entropy_production", "compute_girsanov_kl",
    "compute_total_entropy_production", "compute_entropy_selfadjoint",
    "measure_energy_basis", "lindblad_dissipator", "measurement_backaction",
]
