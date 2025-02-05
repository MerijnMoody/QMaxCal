import matplotlib.pyplot as plt
import numpy as np
import torch
from qutip import *

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

def plot_populations(evolution):
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
    plt.show()

def plot_optimization_results(result):
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
    plt.show()

def plot_parameter_evolution(result):
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
    plt.show()

def main():
    from TorchCleanUpForbidden import create_evolution
    # Setup system and create evolution object
    system_params = setup_quantum_system()
    evolution = create_evolution(*system_params)

    # Set up forbidden state |1⟩
    forbidden_state = torch.zeros(evolution.dim, dtype=torch.complex128)
    forbidden_state[1] = 1.0  # Set |1⟩ as forbidden state
    evolution.forbidden_state = forbidden_state

    # Run optimization
    result = evolution.optimize(
        n_iters=100,
        learning_rate=0.05,
        constraint=0,
        fidelity_target=0,
        load_params=None
    )

    # Plot all results
    plot_populations(evolution)
    plot_optimization_results(result)
    plot_parameter_evolution(result)

if __name__ == "__main__":
    main()