import matplotlib.pyplot as plt
import numpy as np
import torch
from qutip import *
import os
from datetime import datetime
import json

def setup_quantum_system():
    # System parameters
    glob_dim = 3
    vac = basis(glob_dim, 0)
    a = destroy(glob_dim)
    
    # Hamiltonians
    H_sys = a.dag()*a
    H_con = [[liouvillian(a), liouvillian(a.dag())]]
    Ham_list = [[0.5 * H_sys.full(), 0.5 * H_sys.full()], [a.full(), a.dag().full()]]
    
    # Time evolution parameters
    n_ts = 75
    evo_time = 0.3
    times = np.linspace(0, evo_time, n_ts)
    
    # Collapse operators
    c1, c2, c3 = -1.5, -1.5, 0
    c_ops = [c1*a, c2*a.dag(), c3*(a+a.dag())]
    L0 = liouvillian(H_sys, c_ops=c_ops)
    
    # Initial and target states
    rho0 = operator_to_vector(Qobj([[1,0,0],[0,0,0], [0,0,0]]))
    rhotar = operator_to_vector(Qobj([[0,0,0],[0,0,0],[0,0,1]]))
    
    
    return L0, H_con, Ham_list, rho0, rhotar, times[:-1], glob_dim, None, 10, c_ops

def plot_populations(evolution, save_dir=None):
    """Plot populations with color gradient for iterations"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Only take every 20th iteration
    trajectories = evolution.iteration_trajectories[::10]
    n_trajectories = len(trajectories)
    
    # Create color gradient for the reduced number of trajectories
    colors = plt.cm.RdYlBu(np.linspace(0, 1, n_trajectories))
    line_styles = ['-', '--', '-.']  # Different line styles for each state
    state_labels = ['|0⟩', '|1⟩', '|2⟩']
    
    # Plot each trajectory
    for i, traj in enumerate(trajectories):
        alpha = 0.3 + 0.7 * (i / n_trajectories)
        
        for state_idx in range(3):
            populations = np.array([state[state_idx] for state in traj['avg']])
            # Only add label for the final iteration
            label = state_labels[state_idx] if i == n_trajectories - 1 else None
            ax.plot(traj['time'], populations, 
                   linestyle=line_styles[state_idx],
                   color=colors[i], 
                   alpha=alpha, 
                   label=label)

    ax.set_xlabel('Time')
    ax.set_ylabel('Population')
    ax.set_title('Evolution of State Populations During Optimization')
    ax.grid(True)
    ax.legend()

    # Add colorbar
    norm = plt.Normalize(1, n_trajectories)
    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlBu, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label='Optimization iteration')
    
    plt.tight_layout()
    
    # Save the figure if directory is provided
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'population_evolution.png'), dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_optimization_results(result, save_dir=None):
    """Plot optimization results"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    iterations = range(len(result.loss_history))
    
    # Plot loss
    ax1.plot(iterations, result.loss_history, 'b-')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss Evolution')
    ax1.grid(True)

    # Plot fidelity
    ax2.plot(iterations, result.fidelity_history, 'r-')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Fidelity')
    ax2.set_title('Fidelity Evolution')
    ax2.grid(True)

    # Plot forbidden occupation * 1000 instead of energy
    ax3.plot(iterations, [x*1000 for x in result.forbidden_occupation_history], 'g-')
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Forbidden State Occupation × 1000')
    ax3.set_title('Forbidden State Occupation Evolution')
    ax3.grid(True)

    # Plot entropy
    ax4.plot(iterations, result.entropy_history, 'm-')
    ax4.set_xlabel('Iteration')
    ax4.set_ylabel('Entropy')
    ax4.set_title('Entropy Evolution')
    ax4.grid(True)

    plt.tight_layout()
    
    # Save the figure if directory is provided
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'optimization_metrics.png'), dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_parameter_evolution(result, save_dir=None):
    """Plot the evolution of control parameters"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot real parameters
    for i in range(result.parameter_history_real[0].shape[0]):
        values = [params[i].detach().numpy() for params in result.parameter_history_real]
        ax1.plot(range(len(values)), values, label=f'Real param {i+1}')
    ax1.set_title('Evolution of Real Parameters')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Parameter Value')
    ax1.legend()
    ax1.grid(True)
    
    # Plot imaginary parameters
    for i in range(result.parameter_history_imag[0].shape[0]):
        values = [params[i].detach().numpy() for params in result.parameter_history_imag]
        ax2.plot(range(len(values)), values, label=f'Imag param {i+1}')
    ax2.set_title('Evolution of Imaginary Parameters')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Parameter Value')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    # Save the figure if directory is provided
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'parameter_evolution.png'), dpi=300, bbox_inches='tight')
    
    plt.show()

def save_run_parameters(evolution, result, save_dir):
    """Save run parameters, results, and control parameters to files"""
    # Basic run info for JSON
    params_dict = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'system_dimension': evolution.dim,
        'time_steps': len(evolution.time_list),
        'max_time': evolution.time_list[-1].item(),
        'final_metrics': {
            'loss': result.loss_history[-1],
            'fidelity': result.fidelity_history[-1],
            'entropy': result.entropy_history[-1],
            'forbidden_occupation': result.forbidden_occupation_history[-1]
        }
    }
    
    # Save evolution history
    params_dict['history'] = {
        'loss': result.loss_history,
        'fidelity': result.fidelity_history,
        'entropy': result.entropy_history,
        'forbidden_occupation': result.forbidden_occupation_history
    }
    
    # Save to JSON
    with open(os.path.join(save_dir, 'run_parameters.json'), 'w') as f:
        json.dump(params_dict, f, indent=4)
    
    # Save final control parameters as NumPy arrays
    if hasattr(result, 'parameter_history_real') and len(result.parameter_history_real) > 0:
        # Save Fourier parameters
        final_real_params = result.parameter_history_real[-1].detach().cpu().numpy()
        final_imag_params = result.parameter_history_imag[-1].detach().cpu().numpy()
        
        np.save(os.path.join(save_dir, 'final_real_params.npy'), final_real_params)
        np.save(os.path.join(save_dir, 'final_imag_params.npy'), final_imag_params)
        
        # Also save in human-readable format
        with open(os.path.join(save_dir, 'final_parameters.txt'), 'w') as f:
            f.write("Final Real Parameters:\n")
            f.write(str(final_real_params))
            f.write("\n\nFinal Imaginary Parameters:\n")
            f.write(str(final_imag_params))
        
        print(f"Saved final control parameters to {save_dir}")

def main():
    from TorchCleanUpForbidden import create_evolution
    
    # Create run directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join('/Users/zier/Documents/Projects/QMaxCal/results', timestamp)
    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving run results to: {save_dir}")
    
    # Setup system and create evolution object
    system_params = setup_quantum_system()
    evolution = create_evolution(*system_params)

    # Set up forbidden state |1⟩
    forbidden_state = torch.zeros(evolution.dim, dtype=torch.complex128)
    forbidden_state[1] = 1.0  # Set |1⟩ as forbidden state
    evolution.forbidden_state = forbidden_state

    # Run optimization
    result = evolution.optimize(
        n_iters=10,  # Increased for better results
        learning_rate=0.1,  # Higher learning rate
        constraint=0,
        fidelity_target=0,
        load_params=None
    )

    # Save run parameters and results
    save_run_parameters(evolution, result, save_dir)
    
    # Generate and save plots
    print("Generating and saving plots...")
    
    # Save optimization metrics
    plot_optimization_results(result, save_dir)
    
    # Save parameter evolution
    plot_parameter_evolution(result, save_dir)
    
    # Save population evolution
    plot_populations(evolution, save_dir)
    
    # Print final metrics
    print(f"\nFinal metrics:")
    print(f"Loss: {result.loss_history[-1]:.6f}")
    print(f"Fidelity: {result.fidelity_history[-1]:.6f}")
    print(f"Forbidden occupation: {result.forbidden_occupation_history[-1]:.6f}")
    print(f"Entropy: {result.entropy_history[-1]:.6f}")
    
    print(f"\nAll results saved to: {save_dir}")

if __name__ == "__main__":
    main()