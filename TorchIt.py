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

def bspline_basis(t, i, degree, knots):
    """
    Compute the value of a B-spline basis function at point t.
    """
    if degree == 0:
        return torch.where((knots[i] <= t) & (t < knots[i + 1]), 
                         torch.ones_like(t), 
                         torch.zeros_like(t))
    
    denom1 = knots[i + degree] - knots[i]
    denom2 = knots[min(i + degree + 1, len(knots) - 1)] - knots[i + 1]
    
    term1 = torch.where(denom1 != 0,
                       (t - knots[i]) / denom1 * bspline_basis(t, i, degree-1, knots),
                       torch.zeros_like(t))
    term2 = torch.where(denom2 != 0,
                       (knots[min(i + degree + 1, len(knots) - 1)] - t) / denom2 * bspline_basis(t, i+1, degree-1, knots),
                       torch.zeros_like(t))
    
    return term1 + term2

def evaluate_bspline_batch(parameters, t, n_ts, T, degree=3):
    """
    Evaluate multiple B-spline functions in batch given control point parameters.
    
    Args:
        parameters (torch.Tensor): Control point parameters (n_params, n_ctrls)
        t (torch.Tensor): Time points at which to evaluate the splines (n_ts,)
        n_ts (int): Number of time steps
        T (float): Total time interval
        degree (int): Degree of the B-spline (default: 3 for cubic splines)
    
    Returns:
        torch.Tensor: Values of the B-spline functions at time t (n_ts, n_ctrls)
    """
    n_params, n_ctrls = parameters.shape
    
    # Create uniform knot vector with appropriate multiplicity at endpoints
    n_knots = n_params + degree + 1
    internal_knots = torch.linspace(0, T, n_knots - 2 * (degree + 1))
    knots = torch.cat([
        torch.zeros(degree),
        internal_knots,
        T * torch.ones(degree + 1)
    ])
    
    # Normalize time to [0, T]
    t_norm = t.unsqueeze(-1).repeat(1, n_ctrls)
    
    # Evaluate B-spline basis functions
    basis_vals = torch.stack([bspline_basis(t_norm, i, degree, knots) for i in range(n_params)], dim=1)
    
    # Compute batch of B-spline values
    result = torch.einsum('ij,ji->ij', parameters, basis_vals)
    
    return result.transpose(0, 1)
    """
    Evaluate multiple B-spline functions in batch given control point parameters.
    
    Args:
        parameters (torch.Tensor): Control point parameters (n_params, n_ctrls)
        t (torch.Tensor): Time points at which to evaluate the splines (n_ts,)
        n_ts (int): Number of time steps
        T (float): Total time interval
        degree (int): Degree of the B-spline (default: 3 for cubic splines)
    
    Returns:
        torch.Tensor: Values of the B-spline functions at time t (n_ctrls, n_ts)
    """
    n_params, n_ctrls = parameters.shape
    
    # Create uniform knot vector with appropriate multiplicity at endpoints
    n_knots = n_params + degree + 1
    internal_knots = torch.linspace(0, T, n_knots - 2 * (degree + 1))
    knots = torch.cat([
        torch.zeros(degree),
        internal_knots,
        T * torch.ones(degree + 1)
    ])
    
    # Normalize time to [0, T]
    t_norm = t.unsqueeze(1).repeat(1, n_ctrls)
    
    # Evaluate each basis function and multiply by corresponding parameters
    result = torch.zeros(n_ctrls, n_ts)
    for i in range(n_params):
        basis_val = bspline_basis(t_norm, i, degree, knots)
        result[i,:] += (parameters[i, :] * basis_val[i,:])
    
    return result
    """
    Evaluate multiple B-spline functions in batch given control point parameters.
    
    Args:
        parameters (torch.Tensor): Control point parameters (n_ctrls, n_params)
        t (torch.Tensor): Time points at which to evaluate the splines (n_ts,)
        n_ts (int): Number of time steps
        T (float): Total time interval
        degree (int): Degree of the B-spline (default: 3 for cubic splines)
    
    Returns:
        torch.Tensor: Values of the B-spline functions at time t (n_ctrls, n_ts)
    """
    n_params, n_ctrls = parameters.shape
    
    # Create uniform knot vector with appropriate multiplicity at endpoints
    n_knots = n_params + degree + 1
    internal_knots = torch.linspace(0, T, n_knots - 2 * (degree + 1))
    knots = torch.cat([
        torch.zeros(degree),
        internal_knots,
        T * torch.ones(degree + 1)
    ])
    
    # Normalize time to [0, T]
    t_norm = t * (T / (n_ts - 1))
    
    # Evaluate each basis function and multiply by corresponding parameters
    result = torch.zeros(n_ctrls, n_ts)
    for i in range(n_params):
        basis_val = bspline_basis(t_norm.unsqueeze(0).repeat(n_ctrls, 1), i, degree, knots)
        result += parameters[:, i].unsqueeze(-1) * basis_val
    
    return result
    """
    Evaluate multiple B-spline functions in batch given control point parameters.
    
    Args:
        parameters (torch.Tensor): Control point parameters (n_params, n_ctrls)
        t (torch.Tensor): Time points at which to evaluate the splines (n_ts,)
        n_ts (int): Number of time steps
        T (float): Total time interval
        degree (int): Degree of the B-spline (default: 3 for cubic splines)
    
    Returns:
        torch.Tensor: Values of the B-spline functions at time t (n_ctrls, n_ts)
    """
    n_params, n_ctrls = parameters.shape
    
    # Create uniform knot vector with appropriate multiplicity at endpoints
    n_knots = n_params + degree + 1
    #raise TypeError(str(n_params))
    internal_knots = torch.linspace(0, T, n_knots - 2 * (degree + 1))
    knots = torch.cat([
        torch.zeros(degree),
        internal_knots,
        T * torch.ones(degree + 1)
    ])
    
    # Normalize time to [0, T]
    t_norm = t * (T / (n_ts - 1))
    
    # Evaluate each basis function and multiply by corresponding parameters
    result = torch.zeros(n_ctrls, n_ts)
    for i in range(n_params):
        basis_val = bspline_basis(t_norm, i, degree, knots)
        result += parameters[:, i].unsqueeze(-1) * basis_val
    
    return result
    """
    Evaluate multiple B-spline functions in batch given control point parameters.
    
    Args:
        parameters (torch.Tensor): Control point parameters (n_params, n_ctrls)
        t (torch.Tensor): Time points at which to evaluate the splines (n_ts,)
        n_ts (int): Number of time steps
        T (float): Total time interval
        degree (int): Degree of the B-spline (default: 3 for cubic splines)
    
    Returns:
        torch.Tensor: Values of the B-spline functions at time t (n_ctrls, n_ts)
    """
    n_params, n_ctrls = parameters.shape
    
    # Create uniform knot vector with appropriate multiplicity at endpoints
    n_knots = n_params + degree + 1
    internal_knots = torch.linspace(0, T, n_knots - 2 * degree)
    knots = torch.cat([
        torch.zeros(degree),
        internal_knots,
        T * torch.ones(degree)
    ])
    
    # Normalize time to [0, T]
    t_norm = t * (T / (n_ts - 1))
    
    # Evaluate each basis function and multiply by corresponding parameters
    result = torch.zeros(n_ctrls, n_ts)
    for i in range(n_params):
        basis_val = bspline_basis(t_norm, i, degree, knots)
        result += parameters[:, i].unsqueeze(-1) * basis_val
    
    return result
    """
    Evaluate a B-spline function at time t given control point parameters.
    
    Args:
        parameters (torch.Tensor): Control point parameters (n_params,)
        t (torch.Tensor): Time point(s) at which to evaluate the spline
        n_ts (int): Number of time steps
        T (float): Total time interval
        degree (int): Degree of the B-spline (default: 3 for cubic splines)
    
    Returns:
        torch.Tensor: Value of the B-spline at time t
    """
    n_params = len(parameters)
    
    # Create uniform knot vector with appropriate multiplicity at endpoints
    n_knots = n_params + degree + 1
    internal_knots = torch.linspace(0, T, n_knots - 2 * degree)
    knots = torch.cat([
        torch.zeros(degree),
        internal_knots,
        T * torch.ones(degree)
    ])
    
    # Normalize time to [0, T]
    t_norm = t * (T / (n_ts - 1))
    
    # Evaluate each basis function and multiply by corresponding parameter
    result = torch.zeros_like(t_norm)
    for i in range(n_params):
        basis_val = bspline_basis(t_norm, i, degree, knots)
        result += parameters[i] * basis_val
    
    return result

class LindBladEvolve(torch.nn.Module):
    def __init__(self):
        super(LindBladEvolve, self).__init__()
        self.time_list = None
        self.dt = None
        self.n_ts = None
        self.n_params = None
        
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
        
        self.fid_err = None
        self.ent_err = None
        
        self.lam = 0
        self.lam2 = 0
        
        self.fid = 0
        self.jump = 0
        self.energy = 0

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
        

    def _evo_ctrls(self, ctrls):
        self.ctrls = ctrls.view(self.n_ts , self.n_ctrls)
        self._evo()

    def _get_error(self):
        ent_error = self._get_ent_err()
        fid_error = self._get_fid()
        return [ent_error, fid_error]

    def _get_ent_err(self):
        ent_error = 0
        ref_evo = self.ref_evo
        n_ts = self.n_ts
        for i in range(n_ts + 1):
            ent_error += _grad_ent(self.fwd_evo[i], ref_evo[i])
        return ent_error

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

    def H(self, args):
        params_real = args[0]
        params_im = args[1]
        
        #ctrls_real = args[0]
        #ctrls_im =  args[1]
        
        T = self.time_list[-1]
        t_points = torch.arange(self.n_ts)
        ctrls_real = evaluate_bspline_batch(params_real, t_points, self.n_ts, T)
        ctrls_real = ctrls_real.view(self.n_ts, self.n_ctrls)
        
        ctrls_im = evaluate_bspline_batch(params_im, t_points, self.n_ts, T)
        ctrls_im = ctrls_im.view(self.n_ts, self.n_ctrls)
        self.ctrls_real = ctrls_real
        self.ctrls_im = ctrls_im
        self._evo()

        lam = args[-1]
        lam2 = args[-2]
        self.lam = lam
        self.lam2 = lam2
        
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
        
        return lam * fid_error * 3000 + ent_error + lam2 * 100 * (energy + 5) ** 2 + 100 * (reg_error + 10*cont_error)

    def optimize(self, n_iters, learning_rate, constraint, fidelity_target):
        lam = [100.0]
        lam2 = [50.0]
        
        #raise TypeError(self.n_params)
        params_real = torch.mul(torch.rand((self.n_params, self.n_ctrls)), 1)
        params_im = torch.mul(torch.rand((self.n_params, self.n_ctrls)), 1)
        
        if self.init_ctrls and False:
            lam = self.init_ctrls[-1]
            lam2 = self.init_ctrls[-2]
            params_real = self.init_ctrls[0]
            params_im = self.init_ctrls[1]
        
        lam = torch.tensor(lam, requires_grad=True)
        lam2 = torch.tensor(lam2, requires_grad=True)
        
        params_real = torch.tensor(params_real, requires_grad=True)
        params_im = torch.tensor(params_im, requires_grad=True)
        
        optimizer = BasicGradientDescent([params_real, params_im, lam2, lam], lr=learning_rate)

        for i in range(n_iters):
            optimizer.zero_grad()
            loss = self.H([params_real, params_im, lam2, lam])
            loss.backward()
            optimizer.step()
            
            if (i+1) % 2500 == 0 or i == 0:
                self.fid_grad.append(params_real.grad)
                self.fid_grad.append(params_im.grad)
                self.params_list.append(self.params_real.detach())
                
                cloned_evo = [evo.clone().detach().numpy() for evo in self.fwd_evo]
                self.evo_list.append(cloned_evo)
                self.ent_error_list.append(self.ent_err)
                self.energy_list.append(self.energy)
                self.fid_err_list.append(self.fid_err)
            
            if i == 5000:
                for g in optimizer.param_groups:
                    g['lr'] = g['lr']*2
        try:
            self.fid = self._get_fid()
        except: 
            pass
        self.energy = self._get_energy()
        return [params_real.detach(), params_im.detach(), lam2.detach(), lam.detach()]


def create_evolution(dyn_gen, H_con, ham_list, rho0, rhotar, ref_evo, time_list, dim, init_ctrls, n_params):
    evolution = LindBladEvolve()
    evolution.ham_list = ham_list
    evolution.time_list = time_list
    evolution.n_ts = len(time_list)
    evolution.degree = 2
    evolution.num_knots = 4
    evolution.initial = torch.from_numpy(rho0.full())
    evolution.target = torch.from_numpy(rhotar.full())
    evolution.ref_evo = [torch.from_numpy(ref.full()) for ref in ref_evo]
    evolution.ctrl_gen = [[torch.from_numpy(H[0].full()), torch.from_numpy(H[1].full())] for H in H_con]
    evolution.dyn_gen = torch.from_numpy(dyn_gen.full())
    evolution.n_ctrls = len(H_con)
    evolution.dim = dim
    evolution.dt = time_list[1] - time_list[0]
    evolution.init_ctrls = init_ctrls 
    evolution.n_params = n_params
    
    return evolution
