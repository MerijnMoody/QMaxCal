import torch
from collections import defaultdict
from torch.optim import Optimizer
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt

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
    
    def __init__(self, system, learning_rate: float = 1e-3, load_params: str = None):
        self.system = system
        self.learning_rate = learning_rate
        self.history = defaultdict(list)
        # Add parameter history dictionary
        self.parameter_history = {
            'real': [],
            'imag': []
        }
        self.run_folder = None  # Add this line
        self.load_params = load_params  # Path to saved parameters if using previous results
        self.directories = None  # Will store paths to all output directories

    def save_final_parameters(self):
        """Save final parameters to file"""
        import os
        import json
        from datetime import datetime
        
        final_params = {
            'real': self.parameter_history['real'][-1].detach().numpy().tolist(),
            'imag': self.parameter_history['imag'][-1].detach().numpy().tolist(),
            'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'fidelity': self.history['fidelity'][-1],
            'loss': self.history['loss'][-1]
        }
        
        # Save parameters using the parameters directory
        save_path = os.path.join(self.directories['parameters'], 'final_parameters.json')
        with open(save_path, 'w') as f:
            json.dump(final_params, f, indent=4)
        print(f"Final parameters saved to: {save_path}")

    def load_parameters(self, params_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load parameters from file"""
        import json
        
        with open(params_path, 'r') as f:
            params = json.load(f)
        
        params_real = torch.tensor(params['real'], requires_grad=True)
        params_imag = torch.tensor(params['imag'], requires_grad=True)
        
        print(f"Loaded parameters from previous run:")
        print(f"Timestamp: {params['timestamp']}")
        print(f"Final fidelity: {params['fidelity']}")
        print(f"Final loss: {params['loss']}")
        
        return params_real, params_imag

    def initialize_parameters(self, n_params: int, n_ctrls: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize control parameters"""
        if self.load_params:
            try:
                return self.load_parameters(self.load_params)
            except Exception as e:
                print(f"Error loading parameters: {e}")
                print("Falling back to random initialization")
        
        # Original random initialization
        params_real = torch.mul(torch.rand((n_params, n_ctrls)), 0.1)
        params_im = torch.mul(torch.rand((n_params, n_ctrls)), 0.1)
        return params_real.requires_grad_(True), params_im.requires_grad_(True)

    def optimize(self, n_iters: int, constraint: Optional[float] = None, 
                fidelity_target: Optional[float] = None) -> OptimizationResult:
        """Run optimization loop"""
        from Utils import get_run_folder, create_run_directories  # Add this import
        
        # Create run folder and directory structure
        self.run_folder = get_run_folder()
        self.directories = create_run_directories(self.run_folder)
        
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

        self.system._compute_propagator_ref()
        for i in range(n_iters):
            print(f"\nIteration {i+1}/{n_iters}")
            optimizer.zero_grad()
            loss = self.system._loss_function([params_real, params_im, lam, lam2])
            loss.backward()
            optimizer.step()
            
            print(loss)
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

        # Add heatmap visualization after optimization
        self.plot_parameter_heatmaps()
        
        # Save final parameters after optimization
        self.save_final_parameters()
        
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
        from Utils import save_plot  # Add this import
        # Helper function to run quantum master equation
        def run_qutip_mesolve(times, evolution, ref=False):
            """Run quantum master equation using QuTiP's mesolve"""
            from qutip import mesolve, Qobj
            from TorchCleanUpExample import setup_quantum_system
            
            # Get system parameters from setup function
            L0, H_con, Ham_list, rho0, rhotar, times, glob_dim, _, _, c_ops = setup_quantum_system()
            
            # Initial state as density matrix
            rho0 = Qobj([[1,0],[0,0]])
            
            if ref:
                # Reference evolution (no controls)
                H = [Qobj(2*Ham_list[0][0])]
            else:
                # Convert control parameters to numpy arrays
                ctrls_real = evolution.ctrls_real.detach().numpy()
                ctrls_im = evolution.ctrls_im.detach().numpy()
                
                # Fix: Properly scale the time index
                dt = times[1] - times[0]
                
                def get_control_real(t, args):
                    idx = min(int(t/dt), len(ctrls_real)-1)
                    return ctrls_real[idx][0]
                
                def get_control_imag(t, args):
                    idx = min(int(t/dt), len(ctrls_im)-1)
                    return ctrls_im[idx][0]*1j
                
                # Controlled evolution
                H = [Qobj(2*Ham_list[0][0]), 
                     [Qobj(Ham_list[1][0] + Ham_list[1][1]), get_control_real], 
                     [Qobj(Ham_list[1][0] - Ham_list[1][1]), get_control_imag]]
            
            result = mesolve(H, rho0, times, c_ops)
            populations = [state[1][1].real for state in result.states]
            return np.array(populations)

        # Create figure and axis objects
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Get trajectories list from system
        trajectories = self.system.iteration_trajectories
        n_trajectories = len(trajectories)
        
        # Create color gradient from red to blue
        colors = plt.cm.RdYlBu(np.linspace(0, 1, n_trajectories))
        
        times = trajectories[0]['time']  # Use same time points as trajectories
        
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

         # Get reference evolution (no controls)
        ref_populations = run_qutip_mesolve(times, self.system, ref=True)
        ax.plot(times, ref_populations, 'k:', label='Reference (no controls)', linewidth=2)
        
        # Get controlled evolution
        mesolve_populations = run_qutip_mesolve(times, self.system, ref=False)
        ax.plot(times, mesolve_populations, 'k--', label='Controlled evolution', linewidth=2)

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
        
        # After creating the plot, save it
        save_plot(fig, 'optimization_trajectories', self.directories['figures'])
        plt.show()

    def plot_parameter_evolution(self):
        """Plot the evolution of parameters during optimization"""
        from Utils import save_plot  # Add this import
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
        
        # After creating the plot, save it
        save_plot(fig, 'parameter_evolution', self.directories['figures'])
        plt.show()

    def plot_parameter_heatmaps(self):
        """Create animated visualization of control functions"""
        from Utils import save_plot
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
        from Utils import evaluate_basis, BasisConfig
        import os
        from pathlib import Path

        # Create a figures subdirectory in the run folder
        figures_dir = os.path.join(self.run_folder, 'figures')
        Path(figures_dir).mkdir(parents=True, exist_ok=True)

        # Create animation of control functions
        fig_anim, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        fig_anim.suptitle('', fontsize=14, y=0.98)  # Empty title for step counter
        
        # Get time points and setup basis configuration
        T = self.system.time_list[-1]
        t_points = torch.linspace(0, T, 100)
        
        # Setup basis configuration and calculate all control functions
        fourier_config = BasisConfig(
            type='fourier',
            degree=self.parameter_history['real'][0].shape[0]-1,
            n_params=self.parameter_history['real'][0].shape[0],
            period=T
        )

        # Pre-calculate all control functions for better performance
        all_ctrl_real = []
        all_ctrl_imag = []
        for params_real, params_imag in zip(self.parameter_history['real'], 
                                          self.parameter_history['imag']):
            ctrl_real = evaluate_basis(params_real, t_points, fourier_config)
            ctrl_imag = evaluate_basis(params_imag, t_points, fourier_config)
            all_ctrl_real.append(ctrl_real[:, 0])
            all_ctrl_imag.append(ctrl_imag[:, 0])

        # Calculate global y limits
        all_values = torch.cat([torch.cat(all_ctrl_real), torch.cat(all_ctrl_imag)])
        y_min, y_max = all_values.min().item(), all_values.max().item()
        y_padding = (y_max - y_min) * 0.1
        y_min -= y_padding
        y_max += y_padding

        # Initialize plots
        line1, = ax1.plot([], [], 'b-', linewidth=2)
        line2, = ax2.plot([], [], 'r-', linewidth=2)

        # Set up axes
        for ax in [ax1, ax2]:
            ax.set_xlim(0, T)
            ax.set_ylim(y_min, y_max)
            ax.grid(True)
            ax.set_xlabel('Time')
            ax.set_ylabel('Amplitude')

        ax1.set_title('Real Control')
        ax2.set_title('Imaginary Control')
        
        plt.tight_layout()

        def init():
            line1.set_data([], [])
            line2.set_data([], [])
            fig_anim.suptitle('')
            return line1, line2

        def update(frame):
            line1.set_data(t_points, all_ctrl_real[frame*5])
            line2.set_data(t_points, all_ctrl_imag[frame*5])
            fig_anim.suptitle(f'Optimization Step: {frame*5+1}/{len(self.parameter_history["real"])}',
                            fontsize=11, y=1)
            return line1, line2

        # Create and save animation
        anim = animation.FuncAnimation(
            fig_anim, update,
            init_func=init,
            frames=int(len(self.parameter_history['real'])/5),
            interval=200,
            blit=False
        )
        
        try:
            # Save animation to animations directory
            save_path = os.path.join(self.directories['animations'], 'control_evolution.gif')
            anim.save(save_path, writer='pillow', fps=5)
            print(f"Animation saved to: {save_path}")
            plt.show()
        except Exception as e:
            print(f"Error saving animation: {e}")
        finally:
            plt.close(fig_anim)