from dataclasses import dataclass
from typing import Optional, List
import torch
import numpy as np
from scipy.linalg import logm
import os
from datetime import datetime
import matplotlib.pyplot as plt

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

def matrix_sqrtm(matrix: torch.Tensor) -> torch.Tensor:
    """Compute matrix square root using eigendecomposition"""
    eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
    eigenvalues = eigenvalues.to(dtype=torch.complex128)
    return eigenvectors @ torch.diag(torch.sqrt(eigenvalues)) @ eigenvectors.T

def fidelity(rho: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """Compute quantum fidelity between two density matrices"""
    sqrt_rho = matrix_sqrtm(rho)
    interm_matrix = sqrt_rho @ sigma @ sqrt_rho
    sqrt_interm_matrix = matrix_sqrtm(interm_matrix)
    return torch.abs(torch.norm(torch.trace(sqrt_interm_matrix)))

def fourier_basis(t: torch.Tensor, i: int, period: float = 1.0) -> torch.Tensor:
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
            basis_val = fourier_basis(t_norm, i, 1)
        elif config.type == 'bspline':
            if 0 <= i < len(config.knots) - config.degree - 1:
                basis_val = bspline_basis(t_norm, i, config.knots)
            else:
                raise ValueError(f"Invalid basis function index: {i}")
        basis_vals.append(basis_val)
    
    basis_matrix = torch.stack(basis_vals, dim=1)
    return torch.matmul(basis_matrix, parameters)

def calculate_frob_norm(evo_final: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Calculate Frobenius norm between evolution final state and target state"""
    # Ensure inputs maintain gradients
    if not evo_final.requires_grad:
        print("Warning: evo_final does not require grad")
    
    # Add .to(dtype=torch.complex128) to ensure consistent dtype
    diff = evo_final.to(dtype=torch.complex128) - target.to(dtype=torch.complex128)
    # Use torch.norm instead of matrix_norm for better gradient propagation
    return torch.norm(diff)

def save_plot(fig, name: str, run_folder: str = None) -> str:
    """Save a matplotlib figure to a timestamped results folder
    
    Args:
        fig: matplotlib figure to save
        name: name of the plot file
        run_folder: optional specific run folder name, if None will create new timestamped folder
        
    Returns:
        Path to the saved plot file
    """
    # Create base results directory if it doesn't exist
    base_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
    os.makedirs(base_dir, exist_ok=True)
    
    # Create timestamped folder if run_folder not provided
    if run_folder is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_folder = timestamp
    
    # Create full path for this run
    run_dir = os.path.join(base_dir, run_folder)
    os.makedirs(run_dir, exist_ok=True)
    
    # Save the figure
    filepath = os.path.join(run_dir, f'{name}.png')
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    
    return filepath

def get_run_folder() -> str:
    """Get a timestamped folder name for the current run"""
    return datetime.now().strftime('%Y%m%d_%H%M%S')

def create_run_directories(run_folder: str) -> dict:
    """Create standard directory structure for run outputs
    
    Args:
        run_folder: Base run folder (timestamp-based)
    
    Returns:
        Dictionary containing paths to all subdirectories
    """
    import os
    from pathlib import Path
    
    # Create directory structure
    dirs = {
        'root': run_folder,
        'figures': os.path.join(run_folder, 'figures'),
        'parameters': os.path.join(run_folder, 'parameters'),
        'animations': os.path.join(run_folder, 'animations'),
    }
    
    # Create all directories
    for dir_path in dirs.values():
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    return dirs