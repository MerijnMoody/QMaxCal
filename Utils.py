
from dataclasses import dataclass
from typing import Optional, List
import torch
import numpy as np
from scipy.linalg import logm, sqrtm

@dataclass
class BasisConfig:
    """Configuration for basis functions"""
    type: str  # 'fourier' or 'bspline'
    degree: int
    n_params: int
    period: Optional[float] = None  # for Fourier basis
    knots: Optional[torch.Tensor] = None  # for B-splines

def matrix_logarithm(A: torch.Tensor) -> torch.Tensor:
    """Compute matrix logarithm using scipy's logm"""
    A_np = A.detach().cpu().numpy()
    log_A_np = logm(A_np)
    return torch.tensor(log_A_np, dtype=A.dtype, device=A.device)

def matrix_sqrtm(A: torch.Tensor) -> torch.Tensor:
    """Compute matrix square root using scipy's sqrtm"""
    A_np = A.detach().cpu().numpy()
    sqrt_A_np = sqrtm(A_np)
    return torch.tensor(sqrt_A_np, dtype=A.dtype, device=A.device)

def fidelity(rho: torch.Tensor, sigma: torch.Tensor) -> float:
    """Compute quantum fidelity between two density matrices"""
    sqrt_rho = matrix_sqrtm(rho)
    interm_matrix = sqrt_rho @ sigma @ sqrt_rho
    sqrt_interm_matrix = matrix_sqrtm(interm_matrix)
    return torch.norm(torch.trace(sqrt_interm_matrix)).real

def fourier_basis(t: torch.Tensor, i: int, degree: int, period: float = 1.0) -> torch.Tensor:
    """Evaluate Fourier basis function"""
    if i == 0:
        return torch.ones_like(t)
    harmonic = (i + 1) // 2
    if i % 2 == 1:
        return torch.sin(2 * torch.pi * harmonic * t / period)
    return torch.cos(2 * torch.pi * harmonic * t / period)

def bspline_basis(t: torch.Tensor, i: int, degree: int, knots: torch.Tensor) -> torch.Tensor:
    """Evaluate B-spline basis function"""
    n_knots = len(knots)
    if i < 0 or i >= n_knots - degree - 1:
        raise ValueError(f"Invalid basis function index: {i}")
    
    basis = torch.zeros(len(t), n_knots - 1, dtype=t.dtype, device=t.device)
    basis[:, i] = 1.0
    
    for d in range(1, degree + 1):
        for j in range(n_knots - d - 1):
            left = (t - knots[j]) / (knots[j + d] - knots[j]) if knots[j + d] != knots[j] else 0.0
            right = (knots[j + d + 1] - t) / (knots[j + d + 1] - knots[j + 1]) if knots[j + d + 1] != knots[j + 1] else 0.0
            basis[:, j] = left * basis[:, j] + right * basis[:, j + 1]
    
    return basis[i]

def evaluate_basis(parameters: torch.Tensor, t: torch.Tensor, config: BasisConfig) -> torch.Tensor:
    """Evaluate basis functions with parameters"""
    t_norm = t / (config.period if config.period is not None else 1.0)
    basis_vals = []
    
    for i in range(config.n_params):
        if config.type == 'fourier':
            basis_val = fourier_basis(t_norm, i, config.degree, config.period)
        elif config.type == 'bspline':
            if 0 <= i < len(config.knots) - config.degree - 1:
                basis_val = bspline_basis(t_norm, i, config.degree, config.knots)
            else:
                raise ValueError(f"Invalid basis function index: {i}")
        basis_vals.append(basis_val)
    
    basis_matrix = torch.stack(basis_vals, dim=1)
    return torch.matmul(basis_matrix, parameters)