import torch
from collections import defaultdict
from torch.optim import Optimizer
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np

@dataclass
class OptimizationResult:
    """Container for optimization results"""
    controls_real: torch.Tensor
    controls_imag: torch.Tensor
    loss_history: List[float]
    fidelity_history: List[float]
    energy_history: List[float]
    entropy_history: List[float]
    parameter_history_real: List[torch.Tensor]  # Add these new fields
    parameter_history_imag: List[torch.Tensor]

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
            max_grad_norm=1.0  # Added here in defaults instead of constructor
        )
        super().__init__(params, defaults)
        self.state = defaultdict(dict)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for idx, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                
                print(f"\nParameter {idx}:")
                print(f"Value: {p.data}")
                print(f"Gradient: {p.grad}")
                
                # Update step
                if idx < len(group['params']) - 2:  # Control parameters
                    p.add_(p.grad, alpha=-group['lr'])
                else:  # Lambda parameters
                    p.add_(p.grad, alpha=group['lr'])

        return loss

class QuantumOptimizer:
    """Handles quantum control optimization"""
    
    def __init__(self, system, learning_rate: float = 1e-3):
        self.system = system
        self.learning_rate = learning_rate
        self.history = defaultdict(list)
        # Add parameter history dictionary
        self.parameter_history = {
            'real': [],
            'imag': []
        }

    def initialize_parameters(self, n_params: int, n_ctrls: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize control parameters"""
        params_real = torch.mul(torch.rand((n_params, n_ctrls)), 1)
        params_im = torch.mul(torch.rand((n_params, n_ctrls)), 1)
        return params_real.requires_grad_(True), params_im.requires_grad_(True)

    def optimize(self, n_iters: int, constraint: Optional[float] = None, 
                fidelity_target: Optional[float] = None) -> OptimizationResult:
        """Run optimization loop"""
        # Initialize parameters - they already have requires_grad=True from initialize_parameters
        params_real, params_im = self.initialize_parameters(
            self.system.n_params, 
            self.system.n_ctrls
        )
        
        # Initialize multipliers
        lam = torch.tensor([1.0], requires_grad=True)
        lam2 = torch.tensor([1.0], requires_grad=True)
        
        optimizer = BasicGradientDescent(
            [params_real, params_im, lam, lam2], 
            lr=self.learning_rate
        )

        for i in range(n_iters):
            optimizer.zero_grad()
            loss = self.system._loss_function([params_real, params_im, lam, lam2])
            loss.backward()
            optimizer.step()
            
            # Store history
            self.parameter_history['real'].append(params_real.detach().clone())
            self.parameter_history['imag'].append(params_im.detach().clone())
            self.history['loss'].append(loss.detach().item())
            self.history['fidelity'].append(self.system.fidelity.detach().item())
            self.history['energy'].append(self.system.energy.detach().item())
            self.history['entropy'].append(self.system.relative_entropy.detach().item())

        # Final evaluation
        final_controls = self.system._loss_function([params_real, params_im, lam, lam2])
        
        # After optimization loop, plot all trajectories
        self.plot_optimization_trajectories()
        
        # Add parameter evolution plot
        self.plot_parameter_evolution()

        
        return OptimizationResult(
            controls_real=self.system.ctrls_real.detach(),
            controls_imag=self.system.ctrls_im.detach(),
            loss_history=self.history['loss'],
            fidelity_history=self.history['fidelity'],
            energy_history=self.history['energy'],
            entropy_history=self.history['entropy'],
            parameter_history_real=self.parameter_history['real'],
            parameter_history_imag=self.parameter_history['imag']
        )

    def plot_optimization_trajectories(self):
        """Plot trajectories from different optimization steps"""
        import matplotlib.pyplot as plt
        
        # Create figure and axis objects
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Get trajectories list from system
        trajectories = self.system.iteration_trajectories
        n_trajectories = len(trajectories)
        
        # Create color gradient from red to blue
        colors = plt.cm.RdYlBu(np.linspace(0, 1, n_trajectories))
        
        # Plot each trajectory
        for i, traj in enumerate(trajectories):
            label = f'Iteration {i+1}' if i % (n_trajectories // 5) == 0 else None
            alpha = 0.3 + 0.7 * (i / n_trajectories)
            ax.plot(traj['time'], traj['avg'], color=colors[i], alpha=alpha, label=label)
            if i == n_trajectories - 1:
                ax.fill_between(
                    traj['time'],
                    traj['avg'] - traj['std'],
                    traj['avg'] + traj['std'],
                    color=colors[i],
                    alpha=0.2
                )

        ax.set_xlabel('Time')
        ax.set_ylabel('Excited state population')
        ax.set_title('Evolution of Quantum Trajectories During Optimization')
        ax.grid(True)
        ax.legend()

        # Add colorbar properly
        norm = plt.Normalize(1, n_trajectories)
        sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlBu, norm=norm)
        sm.set_array([])  # Needed for matplotlib < 3.3
        plt.colorbar(sm, ax=ax, label='Optimization iteration')
        
        plt.tight_layout()
        plt.show()

    def plot_parameter_evolution(self):
        """Plot the evolution of parameters during optimization"""
        import matplotlib.pyplot as plt
        
        n_iters = len(self.parameter_history['real'])
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot real parameters
        real_params = torch.stack(self.parameter_history['real'])
        for j in range(real_params.shape[1]):
            ax1.plot(range(n_iters), real_params[:, j, 0].numpy(), 
                    label=f'Real param {j+1}')
        ax1.set_title('Evolution of Real Parameters')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Parameter Value')
        ax1.legend()
        ax1.grid(True)
        
        # Plot imaginary parameters
        imag_params = torch.stack(self.parameter_history['imag'])
        for j in range(imag_params.shape[1]):
            ax2.plot(range(n_iters), imag_params[:, j, 0].numpy(), 
                    label=f'Imag param {j+1}')
        ax2.set_title('Evolution of Imaginary Parameters')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Parameter Value')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
        plt.show()