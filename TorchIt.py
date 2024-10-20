import torch
import numpy as np
from qutip import *
from torch.optim.optimizer import Optimizer
from torch import nn
from scipy.linalg import logm
from scipy.optimize import approx_fprime

class AdamWithFlippedSign(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(AdamWithFlippedSign, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for idx, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                denom = exp_avg_sq.sqrt().add_(group['eps'])

                step_size = group['lr'] / (1 - beta1 ** state['step'])
                
                if idx == len(group['params']) - 1:
                    # Flip the sign of the last parameter update
                    p.data.addcdiv_(-step_size, abs(exp_avg), abs(denom))
                else:
                    p.data.addcdiv_(step_size, exp_avg, denom)

        return loss

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

def generate_knot_vector(times, degree, num_knots):
    knots = np.linspace(times[0], times[-1], num_knots)
    knots = np.concatenate(([times[0]] * degree, knots, [times[-1]] * degree))
    return torch.tensor(knots, dtype=torch.float32)

def evaluate_bspline_basis(t, degree, knots, i):
    if degree == 0:
        return ((knots[i] <= t) & (t < knots[i + 1])).float()
    else:
        left_den = knots[i + degree] - knots[i]
        right_den = knots[i + degree + 1] - knots[i + 1]
        
        left = ((t - knots[i]) / left_den) * evaluate_bspline_basis(t, degree - 1, knots, i) if left_den != 0 else torch.zeros_like(t)
        right = ((knots[i + degree + 1] - t) / right_den) * evaluate_bspline_basis(t, degree - 1, knots, i + 1) if right_den != 0 else torch.zeros_like(t)

        return left + right
    
class BSplineParameterization(nn.Module):
    def __init__(self, times, num_splines, degree, num_knots):
        super(BSplineParameterization, self).__init__()
        self.times = torch.tensor(times, dtype=torch.float32)
        self.num_splines = num_splines
        self.degree = degree
        self.num_knots = num_knots
        
        # Generate the knot vector
        self.knots = generate_knot_vector(times, degree, num_knots)
        
        # Initialize the weights for the linear combination of B-splines
        self.weights = nn.Parameter(torch.randn(num_knots + degree - 1, num_splines))

    def forward(self):
        # Evaluate B-spline basis functions at each time point
        basis_matrix = []
        for i in range(len(self.knots) - self.degree - 1):
            basis_matrix.append(evaluate_bspline_basis(self.times, self.degree, self.knots, i))
        
        basis_matrix = torch.stack(basis_matrix, dim=-1)  # Shape: (len(times), num_basis)
        
        # Compute the B-spline functions as a linear combination of basis functions
        b_splines = torch.matmul(basis_matrix, self.weights)  # Shape: (len(times), num_splines)
        
        # Sum the B-spline functions along the spline dimension
        b_splines_sum = torch.sum(b_splines, dim=1)  # Shape: (len(times),)
        
        return b_splines, b_splines_sum

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

    def _probability_distribution(self, prop):
        total_sum = 0
        #i = 0 
        probability_distribution = [None for _ in range(2**len(self.measurements[1]))]
        while total_sum < 0.95:
            #i+=1
            probability, measured_value = self._trajectory_probability(prop)
            # raise TypeError(print(probability, str(measured_value), self._binary_array_to_integer(measured_value)))
            index = self._binary_array_to_integer(measured_value)
            if probability_distribution[index] == None:
                probability_distribution[index] = probability
                total_sum += probability
            
            #if i >= 100:
                #raise TypeError(total_sum)
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
        fid_error = self._get_fid()
        return [ent_error, fid_error]

    def _get_ent_err(self):
        ent_error = 0
        ref_evo = self.ref_evo
        n_ts = self.n_ts

        if not self.prop_evo_ref:
            self._evo_ref()

        p1 = self._probability_distribution(self.prop)
        p2 = self._probability_distribution(self.prop_evo_ref)

        return self._get_kl_divergence(p1, p2)

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
        ctrls_real = args[0]
        ctrls_im =  args[1]
        
        ctrls_real = ctrls_real.view(self.n_ts, self.n_ctrls)
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
        
        return 0*lam * fid_error * 3000 - 10 * ent_error + 0*lam2 * 10 * (energy + 5) ** 2 + 0*10 * (reg_error + 10*cont_error)

    def optimize(self, n_iters, learning_rate, constraint, fidelity_target):
        lam = [100.0]
        lam2 = [50.0]
        
        ctrls_real = torch.mul(torch.rand((self.n_ts, self.n_ctrls)), 5)
        ctrls_im = torch.mul(torch.rand((self.n_ts, self.n_ctrls)), 5)
        
        if self.init_ctrls:
            lam = self.init_ctrls[-1]
            lam2 = self.init_ctrls[-2]
            ctrls_real = self.init_ctrls[0]
            ctrls_im = self.init_ctrls[1]
        
        lam = torch.tensor(lam, requires_grad=True)
        lam2 = torch.tensor(lam2, requires_grad=True)
        
        ctrls_real = torch.tensor(ctrls_real, requires_grad=True)
        ctrls_im = torch.tensor(ctrls_im, requires_grad=True)
        
        optimizer = BasicGradientDescent([ctrls_real, ctrls_im, lam2, lam], lr=learning_rate)

        for i in range(n_iters):
            optimizer.zero_grad()
            loss = self.H([ctrls_real, ctrls_im, lam2, lam])
            loss.backward()
            optimizer.step()
            
            if (i+1) % 2500 == 0 or i == 0:
                self.fid_grad.append(ctrls_real.grad)
                self.fid_grad.append(ctrls_im.grad)
                self.ctrls_list.append(self.ctrls_real.detach())
                
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
        return [ctrls_real.detach(), ctrls_im.detach(), lam2.detach(), lam.detach()]


def create_evolution(dyn_gen, H_con, ham_list, rho0, rhotar, ref_evo, time_list, dim, init_ctrls, measurements):
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
    evolution.measurements = measurements
    
    return evolution
