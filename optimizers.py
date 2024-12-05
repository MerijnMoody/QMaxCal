import torch
from collections import defaultdict
from torch.optim import Optimizer
from dataclasses import dataclass
from typing import List, Optional, Tuple

@dataclass
class OptimizationResult:
    """Container for optimization results"""
    controls_real: torch.Tensor
    controls_imag: torch.Tensor
    loss_history: List[float]
    fidelity_history: List[float]
    energy_history: List[float]
    entropy_history: List[float]

class BasicGradientDescent(Optimizer):
    def __init__(self, params, lr=1e-3, weight_decay=0):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        
        defaults = dict(
            lr=lr,
            initial_lr=lr,  # Required for scheduler
            weight_decay=weight_decay,
            max_grad_norm=1.0
        )
        super().__init__(params, defaults)
        self.state = defaultdict(dict)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step with MDMM"""
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            weight_decay = group['weight_decay']

            # Clip gradients group-wise
            torch.nn.utils.clip_grad_norm_(group['params'], group['max_grad_norm'])

            for idx, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                
                grad = p.grad
                state = self.state[p]
                
                if len(state) == 0:
                    state['step'] = 0
                    
                if torch.isnan(grad).any():
                    continue
                    
                if weight_decay != 0:
                    grad = grad.add(p, alpha=weight_decay)
                
                # MDMM: Descent for controls, ascent for multipliers
                if idx < len(group['params']) - 2:  # Control parameters
                    p.add_(grad, alpha=-lr)  # Gradient descent
                else:  # Lambda parameters
                    p.add_(grad, alpha=lr)   # Gradient ascent
                    
                state['step'] += 1

        return loss

class QuantumOptimizer:
    """Handles quantum control optimization"""
    
    def __init__(self, system, learning_rate: float = 1e-3):
        self.system = system
        self.learning_rate = learning_rate
        self.history = defaultdict(list)

    def initialize_parameters(self, n_params: int, n_ctrls: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize control parameters"""
        params_real = torch.mul(torch.rand((n_params, n_ctrls)), 1)
        params_im = torch.mul(torch.rand((n_params, n_ctrls)), 1)
        return params_real.requires_grad_(True), params_im.requires_grad_(True)

    def optimize(self, n_iters: int, constraint: Optional[float] = None, 
                fidelity_target: Optional[float] = None) -> OptimizationResult:
        """Run optimization loop"""
        # Initialize parameters
        params_real, params_im = self.initialize_parameters(
            self.system.n_params, 
            self.system.n_ctrls
        )
        
        # Initialize multipliers
        lam = torch.tensor([1.0], requires_grad=True)
        lam2 = torch.tensor([1.0], requires_grad=True)
        
        # Setup optimizer
        optimizer = BasicGradientDescent(
            [params_real, params_im, lam, lam2], 
            lr=self.learning_rate
        )

        # Optimization loop
        for i in range(n_iters):
            optimizer.zero_grad()
            loss = self.system._loss_function([params_real, params_im, lam, lam2])
            loss.backward()
            optimizer.step()
            
            # Added debug printing
            print(f"\nIteration {i+1}/{n_iters}")
            print(f"Loss: {loss.detach().item():.6f}")
            print(f"Fidelity: {self.system.fidelity.detach().item():.6f}")
            print(f"Energy: {self.system.energy.detach().item():.6f}")
            print(f"Entropy: {self.system.relative_entropy.detach().item():.6f}")
            print(f"Lambda: {lam.item():.6f}, Lambda2: {lam2.item():.6f}")
            
            # Store history
            self.history['loss'].append(loss.detach().item())
            self.history['fidelity'].append(self.system.fidelity.detach().item())
            self.history['energy'].append(self.system.energy.detach().item())
            self.history['entropy'].append(self.system.relative_entropy.detach().item())

        # Final evaluation
        final_controls = self.system._loss_function([params_real, params_im, lam, lam2])
        
        return OptimizationResult(
            controls_real=self.system.ctrls_real.detach(),
            controls_imag=self.system.ctrls_im.detach(),
            loss_history=self.history['loss'],
            fidelity_history=self.history['fidelity'],
            energy_history=self.history['energy'],
            entropy_history=self.history['entropy']
        )