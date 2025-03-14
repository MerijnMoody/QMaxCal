import torch
import numpy as np
from qutip import *
from torch.optim.optimizer import Optimizer
from torch import nn
from scipy.linalg import logm
from scipy.optimize import approx_fprime


class BasicGradientDescent(Optimizer):
    def __init__(self, params, lr=1e-3, weight_decay=0):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        
        defaults = dict(lr=lr, weight_decay=weight_decay)
        super(BasicGradientDescent, self).__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step.
        
        Arguments:
        closure (callable, optional): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            weight_decay = group['weight_decay']

            for idx, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                
                grad = p.grad
                
                if weight_decay != 0:
                    grad = grad.add(p, alpha=weight_decay)
                
                if idx == len(group['params']) - 1 or idx == len(group['params']) -2:
                    p.add_(-grad, alpha=-lr)
                else:
                    p.add_(grad, alpha=-lr)
        
        return loss

def matrix_logarithm(A):
    """
    Compute the matrix logarithm of a square matrix A using eigenvalue decomposition.
    
    Parameters:
    A (torch.Tensor): A square matrix of shape (n, n)
    
    Returns:
    torch.Tensor: The matrix logarithm of A
    """
    # Perform eigenvalue decomposition
    eigenvalues, eigenvectors = torch.linalg.eig(A)
    
    # Ensure eigenvalues are real
    if not torch.allclose(eigenvalues.imag, torch.zeros_like(eigenvalues.imag), atol=1e-6):
        raise ValueError("Matrix logarithm is only defined for matrices with real eigenvalues.")
    
    # Take the real part of eigenvalues (since we ensured they are real)
    
    # Compute the logarithm of eigenvalues
    log_eigenvalues = torch.diag(torch.log(eigenvalues))
    
    eigenvectors_inv = torch.linalg.inv(eigenvectors)
    # Reconstruct the matrix logarithm using the eigenvectors and log_eigenvalues
    #raise TypeError(str(eigenvectors.real) + " " + str(log_eigenvalues.real) + " " + str(eigenvectors_inv))
    try:
        log_A = eigenvectors @ log_eigenvalues @ eigenvectors_inv
        return log_A 
    except:
        raise TypeError(str(eigenvectors.real) + " " + str(log_eigenvalues.real) + " " + str(eigenvectors_inv))
    
    return log_A  # Return the real part of the result

def sqrtm(A):
    """
    Compute the matrix square root of a positive semi-definite matrix A.
    
    Parameters:
    A (torch.Tensor): A square matrix of shape (n, n)
    
    Returns:
    torch.Tensor: The matrix square root of A
    """
    # Perform eigenvalue decomposition
    eigenvalues, eigenvectors = torch.linalg.eig(A)
    
    # Ensure the eigenvalues are non-negative (as A should be positive semi-definite)
    #eigenvalues = torch.clamp(eigenvalues, min=0)
    
    # Compute the square root of the eigenvalues
    sqrt_eigenvalues = torch.sqrt(eigenvalues)
    
    # Reconstruct the matrix square root
    #raise TypeError(str(eigenvectors) + " " + str(torch.diag(sqrt_eigenvalues)))
    sqrt_A = eigenvectors @ torch.diag(sqrt_eigenvalues).type(torch.complex128) @ eigenvectors.conj().t()
    
    return sqrt_A

def fidelity(rho, sigma):
    """
    Compute the fidelity between two density matrices rho and sigma.
    
    Parameters:
    rho (torch.Tensor): A density matrix of shape (n, n)
    sigma (torch.Tensor): A density matrix of shape (n, n)
    
    Returns:
    float: The fidelity between rho and sigma
    """
    # Compute the square root of rho
    sqrt_rho = sqrtm(rho)
    
    # Compute the intermediate matrix
    interm_matrix = sqrt_rho @ sigma @ sqrt_rho
    
    # Compute the matrix square root of the intermediate matrix
    sqrt_interm_matrix = sqrtm(interm_matrix)
    
    # Compute the trace and square it to get the fidelity
    fidelity_value = torch.norm(torch.trace(sqrt_interm_matrix))
    
    return fidelity_value.real

def _grad_ent(rho, sigma):
    dim = rho.size()[0]
    dim = round(dim ** (1/2))
    rho = torch.reshape(rho, (dim, dim))
    sigma = torch.reshape(sigma, (dim, dim))
    try:
        out = torch.trace(rho @ (matrix_logarithm(rho) - matrix_logarithm(sigma))).real
        return out
    except:
        return 100

from dataclasses import dataclass
from typing import Optional, Callable

@dataclass
class BasisConfig:
    type: str  # 'fourier' or 'bspline'
    degree: int
    n_params: int
    period: Optional[float] = None  # for Fourier basis
    knots: Optional[torch.Tensor] = None  # for B-splines

def fourier_basis(t: torch.Tensor, i: int, degree: int, period: float = 1.0) -> torch.Tensor:
    """Evaluate Fourier basis function at time t."""
    if i == 0:
        return torch.ones_like(t)
    harmonic = (i + 1) // 2
    if i % 2 == 1:
        return torch.sin(2 * torch.pi * harmonic * t / period)
    else:
        return torch.cos(2 * torch.pi * harmonic * t / period)
    
    # Stack and multiply with parameters
    basis_matrix = torch.stack(basis_vals, dim=1)  # (n_points, n_params)
    return torch.matmul(basis_matrix, parameters)  # (n_points, n_curves)

def bspline_basis(t: torch.Tensor, i: int, degree: int, knots: torch.Tensor) -> torch.Tensor:
    """Evaluate B-spline basis function at time t."""
    n_knots = len(knots)
    if i < 0 or i >= n_knots - degree - 1:
        raise ValueError(f"Invalid basis function index: {i}")
    
    # Initialize basis functions
    basis = torch.zeros(len(t), n_knots - 1, dtype=t.dtype, device=t.device)
    basis[:, i] = 1.0
    
    # Perform Cox-de Boor recursion
    for d in range(1, degree + 1):
        for j in range(n_knots - d - 1):
            left = (t - knots[j]) / (knots[j + d] - knots[j]) if knots[j + d] != knots[j] else 0.0
            right = (knots[j + d + 1] - t) / (knots[j + d + 1] - knots[j + 1]) if knots[j + d + 1] != knots[j + 1] else 0.0
            basis[:, j] = left * basis[:, j] + right * basis[:, j + 1]
    
    return basis[i]

def evaluate_basis(parameters: torch.Tensor, t: torch.Tensor, config: BasisConfig) -> torch.Tensor:
    """
    Evaluate basis functions with parameters.
    
    Args:
        parameters: Tensor of shape (n_params, n_curves)
        t: Tensor of time points (n_points,)
        config: BasisConfig object
    
    Returns:
        Tensor of shape (n_points, n_curves)
    """
    n_params = config.n_params
    
    # Normalize time to [0, 1]
    t_norm = t / (config.period if config.period is not None else 1.0)
    
    # Evaluate basis functions
    basis_vals = []
    for i in range(n_params):
        if config.type == 'fourier':
            basis_val = fourier_basis(t_norm, i, config.degree, config.period)
        elif config.type == 'bspline':
            if 0 <= i < len(config.knots) - config.degree - 1:
                basis_val = bspline_basis(t_norm, i, config.degree, config.knots)
            else:
                raise ValueError(f"Invalid basis function index: {i, len(config.knots) - config.degree - 1}")
        basis_vals.append(basis_val)
    
    # Stack and multiply with parameters
    basis_matrix = torch.stack(basis_vals, dim=1)  # (n_points, n_params)
    return torch.matmul(basis_matrix, parameters)  # (n_points, n_curves)

class LindBladEvolve(torch.nn.Module):
    def __init__(self):
        super(LindBladEvolve, self).__init__()
        self.time_list = None
        self.dt = None
        self.n_ts = None
        
        self.init_ctrls = None
        self.initial = None
        self.target = None
        self.ref_evo = None
        self.ham_list = None
        
        self.onwd_evo = None
        self.evo_list = []
        self.ent_error_list = []
        self.fwd_evo = None
        self.ctrls_real = None
        self.ctrls_list = []
        self.ctrls_im = None
        self.ctrl_gen = None
        self.dyn_gen = None
        self.dim = None
        self.n_ctrls = None
        self.prop = None
        self.prop_grad = None
        
        self.ent_grad = None
        self.fid_grad = []
        
        self.fid_err_list = []
        self.energy_list = []
        self.prop_evo_ref = None
        
        self.fid_err = None
        self.ent_err = None
        
        self.lam = 0
        self.lam2 = 0
        self.n_params = None
        self.params_list = []

        self.PathDist_ref = []
        self.PathDist_Evo = []
        self.PathDistref = None
        self.PathDistEvo = None
        
        self.fid = 0
        self.jump = 0
        self.energy = 0
        self.measurements = None

    def _get_dyn_gen(self, k):
        dyn_gen = self.dyn_gen
        for j in range(self.n_ctrls):
            dyn_gen = dyn_gen + self._get_ctrl_gen(k, j)
        return dyn_gen
    
    def _get_ham(self, k):
        ham = torch.tensor(self.ham_list[0][0] + self.ham_list[0][1])
        for j, hams in enumerate(self.ham_list[1:]):
            ham = ham + self.ctrls_real[k, j]*(torch.tensor(hams[0] + hams[1]))
            ham = ham + 1j*self.ctrls_im[k, j]*(torch.tensor(hams[0] - hams[1]))
        return ham

    def _get_ctrl_gen(self, k, j):
        real_H = self.ctrls_real[k, j] * self.ctrl_gen[j][0] + self.ctrls_real[k, j] * self.ctrl_gen[j][1]
        im_H = 1j*(self.ctrls_im[k, j] * self.ctrl_gen[j][0] - self.ctrls_im[k, j] * self.ctrl_gen[j][1])
        return real_H + im_H

    def _compute_prop(self, k):
        A = self._get_dyn_gen(k) * self.dt
        prop = torch.matrix_exp(A)
        return prop
    
    def _evo(self):
        n_ts = self.n_ts
        self.prop = [self._compute_prop(i) for i in range(n_ts)]

        self.fwd_evo = [None] * (n_ts + 1)
        self.fwd_evo[0] = self.initial
        for k in range(n_ts):
            self.fwd_evo[k + 1] = self.prop[k] @ self.fwd_evo[k]
            self.fwd_evo[k + 1] = self.fwd_evo[k+1]/torch.trace(self.fwd_evo[k+1].view(self.dim, self.dim))

    def _compute_prop_ref_evo(self,k):
        A = self.dyn_gen * self.dt
        prop = torch.matrix_exp(A)
        return prop
    
    def _evo_ref(self):
        n_ts = self.n_ts

        self.prop_evo_ref = [0 for _ in range(n_ts)]

        for k in range(n_ts):
            self.prop_evo_ref[k] = self._compute_prop_ref_evo(k)

    def _trajectory_probability(self, prop):
        n_ts = self.n_ts
        measure_times = self.measurements[1]
        measure_operators = self.measurements[0]
        state = self.initial
        dims = self.dim
        device = state.device  # Get the device (CPU/GPU) from the state tensor
        
        measured_values = []
        
        # Perform measurements at prescribed timesteps
        for i in range(n_ts):
            state = torch.matmul(prop[i], state)
            
            if i in measure_times:
                # Convert state to operator and get diagonal elements
                state = state.reshape(self.dim, self.dim)
                # raise TypeError(state)
                probs = torch.diagonal(state.real, dim1=0, dim2=1)
                state = state.reshape(self.dim * self.dim)
                
                # Normalize probabilities
                probs = probs / torch.sum(probs)
                
                # Sample from categorical distribution
                measurement = torch.multinomial(probs, num_samples=1).item()
                measured_values.append(measurement)
                
                # Apply measurement operator
                state = torch.matmul(torch.from_numpy(measure_operators[measurement]), state)
        
        # Calculate final probability
        probability = torch.trace(state.reshape(self.dim, self.dim)).real
        
        return probability, measured_values

    def _path_prob_given_trajectory(self,prop):
        n_ts = self.n_ts
        measure_times = self.measurements[1]
        measure_operators = self.measurements[0]
        state = self.initial
        dims = self.dim
        # raise TypeError(measure_times)

        probability_distribution = torch.from_numpy(np.array([0 for _ in range(2**3)]))
        for i in range(0,2):
            for j in range(0,2):
                for k in range(0,2):
                    for m in range(n_ts):
                        state = torch.matmul(prop[i], state)
                        if m == 1:
                            state = state.reshape(self.dim, self.dim)
                            # prob_i = torch.diagonal(state.real, dim1=0, dim2=1)[i]
                            state = state.reshape(self.dim * self.dim)
                            state = torch.matmul(torch.from_numpy(measure_operators[i]), state)
                        if m == 2:
                            state = state.reshape(self.dim, self.dim)
                            # prob_j = torch.diagonal(state.real, dim1=0, dim2=1)[j]
                            state = state.reshape(self.dim * self.dim)
                            state = torch.matmul(torch.from_numpy(measure_operators[j]), state)
                        if m == 3:
                            state = state.reshape(self.dim, self.dim)
                            # prob_k = torch.diagonal(state.real, dim1=0, dim2=1)[k]
                            state = state.reshape(self.dim * self.dim)
                            state = torch.matmul(torch.from_numpy(measure_operators[k]), state)
                    # probability = prob_i*prob_j*prob_k*torch.trace(state.reshape(self.dim, self.dim)).real
                    probability = torch.trace(state.reshape(self.dim, self.dim)).real
                    print(probability)
                    probability_distribution[self._binary_array_to_integer(np.array([i,j,k]))] = probability
                    raise TypeError(probability_distribution)
        total_sum = 0
        for i in range(len(probability_distribution)):
            total_sum += probability_distribution[i].item()
        probability_distribution = probability_distribution/total_sum
        raise TypeError(probability_distribution)
        return probability_distribution

    def _path_prob_given_trajectory_claude(self, prop):
        """
        Calculate probability distribution for quantum measurements across multiple time steps.
        
        Args:
            prop: Propagator operators
            
        Returns:
            torch.Tensor: Normalized probability distribution
        """
        n_ts = self.n_ts
        measure_times = self.measurements[1]
        measure_operators = self.measurements[0]
        dims = self.dim
        
        # Initialize probability distribution for all possible measurement outcomes
        probability_distribution = torch.zeros(2**3)
        
        # Iterate through all possible measurement combinations
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    # Start with initial state for each trajectory
                    state = self.initial.clone()
                    
                    # Initialize individual measurement probabilities
                    prob_i = prob_j = prob_k = 1.0
                    
                    # Evolve state through time steps
                    for m in range(n_ts):
                        state = torch.matmul(prop[m], state)
                        
                        # Reshape state and apply measurements at appropriate times
                        if m == 14:
                            state_matrix = state.reshape(dims, dims)
                            prob_i = torch.diagonal(state_matrix.real, dim1=0, dim2=1)[i]
                            state = state.reshape(dims * dims)
                            state = torch.matmul(torch.from_numpy(measure_operators[i]), state)
                        
                        elif m == 30:
                            state_matrix = state.reshape(dims, dims)
                            prob_j = torch.diagonal(state_matrix.real, dim1=0, dim2=1)[j]
                            state = state.reshape(dims * dims)
                            state = torch.matmul(torch.from_numpy(measure_operators[j]), state)
                        
                        elif m == 44:
                            state_matrix = state.reshape(dims, dims)
                            prob_k = torch.diagonal(state_matrix.real, dim1=0, dim2=1)[k]
                            state = state.reshape(dims * dims)
                            state = torch.matmul(torch.from_numpy(measure_operators[k]), state)
                    
                    # Calculate final probability for this trajectory
                    final_state_matrix = state.reshape(dims, dims)
                    trajectory_prob = torch.trace(final_state_matrix).real
                    
                    # Store probability in distribution
                    idx = self._binary_array_to_integer(torch.tensor([i, j, k]))
                    probability_distribution[idx] = trajectory_prob
        
        # Normalize probability distribution
        total_prob = probability_distribution.sum()
        if total_prob > 0:
            probability_distribution = probability_distribution / total_prob
        
        return probability_distribution   
        
    def _binary_array_to_integer(self,binary_array):
    # Convert to numpy array if it's a list
        if isinstance(binary_array, list):
            binary_array = np.array(binary_array)
    
    # Validate input
        if not np.all(np.isin(binary_array, [0, 1])):
            raise ValueError("Input array must contain only 0s and 1s")
    
    # Convert to integer
        integer_value = 0
        for bit in binary_array:
            integer_value = (integer_value << 1) | bit
    
        return integer_value

    def _probability_distribution_estimate(self, prop, eps, iters):
        i = 0
        total_sum = 0
        probability_distribution = torch.zeros(2**len(self.measurements[1]))
        done = []
        while total_sum < 0.95 and i < iters:
            i+=1
            probability, measured_value = self._trajectory_probability(prop)
            index = self._binary_array_to_integer(measured_value)
            if index not in done:
                probability_distribution[index] = probability
                total_sum += probability
                done.append(index)
                
        for i in range(len(probability_distribution)):
            if i not in done:
                probability_distribution[i] = eps
                #probability_distribution[i] = (1 - total_sum)/(len(probability_distribution)-len(done))
                #if (torch.round(1 - total_sum,4))/(len(probability_distribution)-len(done)) < 0:
                #    raise TypeError((1 - total_sum)/(len(probability_distribution)-len(done)))

                #if i >= 100:
                #raise TypeError(total_sum)
        
        total_prob = probability_distribution.sum()
        if total_prob > 0:
            probability_distribution = probability_distribution / total_prob
            
        return probability_distribution

    # Compute KL
    def _get_kl_divergence(self, p1, p2):
        """
        Calculate Kullback-Leibler divergence between two probability distributions.
        Args:
            p1 (array-like): First probability distribution
            p2 (array-like): Second probability distribution
        
        Returns:
            float: KL divergence value
        """
        kl = 0
        for i in range(len(p1)):
            # Check that probabilities are not None and greater than 0
            if p1[i] is not None and p2[i] is not None and p1[i] > 0 and p2[i] > 0:
                kl += p1[i] * torch.log(p1[i] / p2[i])
        return kl

        
    def _evo_ctrls(self, ctrls):
        self.ctrls = ctrls.view(self.n_ts , self.n_ctrls)
        self._evo()

    def _get_error(self):
        ent_error = self._get_ent_err()
        # raise TypeError(ent_error)
        fid_error = self._get_fid()
        return [ent_error, fid_error]

    def _get_ent_err(self):
        ent_error = 0
        ref_evo = self.ref_evo
        n_ts = self.n_ts

        if not self.prop_evo_ref:
            self._evo_ref()
        
        #p1 = self._probability_distribution_estimate(self.prop,1e-8,500)
        p1 = self._path_prob_given_trajectory_claude(self.prop)
        self.PathDistEvo = p1
        #p2 = self._probability_distribution_estimate(self.prop_evo_ref,1e-8,500)
        p2 = self._path_prob_given_trajectory_claude(self.prop_evo_ref)
        value = self._get_kl_divergence(p1,p2)
        self.PathDistref = p2
        # if value < -0.1:
        #     raise TypeError(value, p1, p2)
        return value

    def _get_fid(self):
        evo_final = self.fwd_evo[-1]
        rel_ent = fidelity(torch.reshape(evo_final, (self.dim,self.dim)), 
                           torch.reshape(self.target, (self.dim,self.dim)))
        return (1 - rel_ent.real)
    
    def _get_frob_norm(self):
        evo_final = self.fwd_evo[-1]
        return torch.linalg.matrix_norm(evo_final - self.target)
        
    def _get_jump(self):
        jump_h = torch.tensor(np.array([[0 + 0j, 1 + 0j], [1 + 0j,0 + 0j]]))
        jumps = 0
        for state in self.fwd_evo:
            jumps += torch.trace(jump_h @ state.view(2,2)).real
        return jumps
            
    def _get_energy(self):
        energy = 0
        for k in range(self.n_ts):
            en_H = self._get_ham(k)
            evo_matrix = self.fwd_evo[k+1].view(self.dim, self.dim)
            energy += torch.trace(en_H @ evo_matrix)
        
        #raise TypeError(energy)
        return energy.real

    # def H(self, args, control_parameters):
    #     ctrls_real = args[0]
    #     ctrls_im =  args[1]
        
    #     ctrls_real = ctrls_real.view(self.n_ts, self.n_ctrls)
    #     ctrls_im = ctrls_im.view(self.n_ts, self.n_ctrls)
    #     self.ctrls_real = ctrls_real
    #     self.ctrls_im = ctrls_im
    #     self._evo()

    #     lam = args[-1]
    #     lam2 = args[-2]
        
    #     self.lam = lam
    #     self.lam2 = lam2
        
    #     fid_error = self._get_frob_norm()
    #     ent_error = self._get_ent_err()
    #     self.fid_err = fid_error
    #     self.ent_err = ent_error

    #     # if ent_error < -0.1:
    #     #     raise TypeError('ent_error is negative' + str( print(ent_error)))

        
    #     #jumps = self._get_jump()
    #     #self.jump = jumps
        
    #     energy = self._get_energy()
    #     self.energy = energy
        
    #     reg_error = (torch.linalg.vector_norm(ctrls_real) +
    #                  torch.linalg.vector_norm(ctrls_im))
        
    #     pairwise_diffs_real = ctrls_real[1:] - ctrls_real[:-1]
    #     pairwise_diffs_im = ctrls_im[1:] - ctrls_im[:-1]
    #     cont_error = (torch.linalg.vector_norm(pairwise_diffs_real) 
    #                   + torch.linalg.vector_norm(pairwise_diffs_im))
        
    #     return lam * fid_error * 500 + ent_error * 100 + lam2 * (energy + 5) ** 2 + 10 * (reg_error + 10*cont_error) * 0
    
    def H(self, args):
        params_real = args[0]
        params_im = args[1]
    
        T = self.time_list[-1]
        t_points = torch.linspace(0, T, self.n_ts)
    
        # Create basis config
        fourier_config = BasisConfig(
            type='fourier',
            degree=params_real.shape[0]-1,
            n_params=params_real.shape[0],
            period=T
        )
        bspline_config = BasisConfig(
            type='bspline',
            degree=params_real.shape[0]-1,
            n_params=params_real.shape[0],
            knots=torch.linspace(0, T, params_real.shape[0] + 3).clone().detach()  # Example knots
        )
    
        # Evaluate basis functions
        ctrls_real = evaluate_basis(params_real, t_points, fourier_config)
        ctrls_im = evaluate_basis(params_im, t_points, fourier_config)
        
        self.ctrls_real = ctrls_real
        self.ctrls_im = ctrls_im
        self._evo()
        
        self.lam = args[-1]
        self.lam2 = args[-2]
        
        fid_error = self._get_frob_norm()
        ent_error = self._get_ent_err()
        self.fid_err = fid_error
        self.ent_err = ent_error
        
        #jumps = self._get_jump()
        #self.jump = jumps
        
        energy = self._get_energy()
        self.energy = energy
        
        reg_error = (torch.linalg.vector_norm(ctrls_real) +
                     torch.linalg.vector_norm(ctrls_im))
        
        pairwise_diffs_real = ctrls_real[1:] - ctrls_real[:-1]
        pairwise_diffs_im = ctrls_im[1:] - ctrls_im[:-1]
        cont_error = (torch.linalg.vector_norm(pairwise_diffs_real) 
                      + torch.linalg.vector_norm(pairwise_diffs_im))
        
        return self.lam * fid_error * 1500 + ent_error + self.lam2 * 100 * (energy + 5) ** 2 + 100 * (reg_error + 10*cont_error)

    # def optimize(self, n_iters, learning_rate, constraint, fidelity_target):
    #     lam = [1.000]
    #     lam2 = [1.000]
        
    #     ctrls_real = torch.mul(torch.rand((self.n_ts, self.n_ctrls)), 0.5)
    #     ctrls_im = torch.mul(torch.rand((self.n_ts, self.n_ctrls)), 0.5)
    
        
    #     if self.init_ctrls:
    #         lam = self.init_ctrls[-1]
    #         lam2 = self.init_ctrls[-2]
    #         ctrls_real = self.init_ctrls[0]
    #         ctrls_im = self.init_ctrls[1]
        
    #     lam = torch.tensor(lam, requires_grad=True)
    #     lam2 = torch.tensor(lam2, requires_grad=True)
        
    #     ctrls_real = torch.tensor(ctrls_real, requires_grad=True)
    #     ctrls_im = torch.tensor(ctrls_im, requires_grad=True)
        
    #     optimizer = BasicGradientDescent([ctrls_real, ctrls_im, lam2, lam], lr=learning_rate)
    #     for i in range(n_iters):
    #         optimizer.zero_grad()
    #         loss = self.H([ctrls_real, ctrls_im, lam2, lam])
    #         loss.backward()
    #         optimizer.step()
            
    #         if (i+1) % 1000 == 0 or i == 0:
    #             self.fid_grad.append(ctrls_real.grad)
    #             self.fid_grad.append(ctrls_im.grad)
    #             self.ctrls_list.append(self.ctrls_real.detach())
                
    #             cloned_evo = [evo.clone().detach().numpy() for evo in self.fwd_evo]
    #             self.evo_list.append(cloned_evo)
    #             self.ent_error_list.append(self.ent_err)
    #             self.energy_list.append(self.energy)
    #             self.fid_err_list.append(self.fid_err)
    #             self.PathDist_Evo.append(self.PathDistEvo)
    #             self.PathDist_ref.append(self.PathDistref)
            
    #         if i == 5000:
    #             for g in optimizer.param_groups:
    #                 g['lr'] = g['lr']*2
    #     try:
    #         self.fid = self._get_fid()
    #     except: 
    #         pass
    #     self.energy = self._get_energy()
    #     return [ctrls_real.detach(), ctrls_im.detach(), lam2.detach(), lam.detach()]

    def optimize(self, n_iters, learning_rate, constraint, fidelity_target):
        # Initialize parameters
        lam = torch.tensor([100.0], requires_grad=True)
        lam2 = torch.tensor([50.0], requires_grad=True)
        
        # Initialize Fourier parameters
        params_real = torch.mul(torch.rand((self.n_params, self.n_ctrls)), 1)
        params_im = torch.mul(torch.rand((self.n_params, self.n_ctrls)), 1)
        
        params_real = params_real.clone().detach().requires_grad_(True)
        params_im = params_im.clone().detach().requires_grad_(True)
        
        optimizer = torch.optim.Adam([params_real, params_im, lam, lam2], lr=learning_rate)

        
        for i in range(n_iters):
            optimizer.zero_grad()
            loss = self.H([params_real, params_im, lam2, lam])
            loss.backward()
            optimizer.step()
            
            # Store history
            # with torch.no_grad():
            #     self.params_list.append([params_real.clone(), params_im.clone(), lam2.clone(), lam.clone()])
            #     errors = self._get_error()
            #     self.ent_error_list.append(errors[0])
            #     self.fid_err_list.append(errors[1])
        
            if (i+1) % 1000 == 0 or i == 0:
                cloned_evo = [evo.clone().detach().numpy() for evo in self.fwd_evo]
                self.evo_list.append(cloned_evo)
                self.ent_error_list.append(self.ent_err)
                self.energy_list.append(self.energy)
                self.fid_err_list.append(self.fid_err)
                self.PathDist_Evo.append(self.PathDistEvo)
                self.PathDist_ref.append(self.PathDistref)
        # Final evaluation to set controls
        _ = self.H([params_real, params_im, lam2, lam])
        
        return [self.ctrls_real, self.ctrls_im]
    
    

def create_evolution(dyn_gen, H_con, ham_list, rho0, rhotar, ref_evo, time_list, dim, init_ctrls, measurements,n_params):
    evolution = LindBladEvolve()
    evolution.ham_list = ham_list
    evolution.time_list = time_list
    evolution.n_ts = len(time_list)
    evolution.initial = torch.from_numpy(rho0.full())
    evolution.target = torch.from_numpy(rhotar.full())
    evolution.ref_evo = [torch.from_numpy(ref.full()) for ref in ref_evo]
    evolution.ctrl_gen = [[torch.from_numpy(H[0].full()), torch.from_numpy(H[1].full())] for H in H_con]
    evolution.dyn_gen = torch.from_numpy(dyn_gen.full())
    evolution.n_ctrls = len(H_con)
    evolution.dim = dim
    evolution.dt = time_list[1] - time_list[0]
    evolution.init_ctrls = init_ctrls 
    evolution.measurements = measurements
    evolution.n_params = n_params
    
    return evolution