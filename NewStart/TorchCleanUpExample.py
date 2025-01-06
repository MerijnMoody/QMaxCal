import matplotlib.pyplot as plt
import numpy as np
from qutip import *
from TorchCleanUp import create_evolution

def setup_quantum_system():
    # System parameters
    glob_dim = 2
    vac = basis(glob_dim, 0)
    a = destroy(glob_dim)
    con2 = Qobj([[0,0], [0, 1]])
    
    # Hamiltonians
    H_sys = a.dag()*a
    H_con = [[liouvillian(a), liouvillian(a.dag())]]
    Ham_list = [[0.5 * H_sys.full(), 0.5 * H_sys.full()], [a.full(), a.dag().full()]]
    
    # Time evolution parameters
    n_ts = 45
    evo_time = 1
    times = np.linspace(0, evo_time, n_ts)
    
    # Collapse operators
    c1, c2, c3 = -1.5, -1.5, 0
    c_ops = [c1*a, c2*a*a.dag(), c3*(a+a.dag())]
    L0 = liouvillian(H_sys, c_ops=c_ops)
    
    # Initial and target states
    rho0 = operator_to_vector(Qobj([[1,0],[0,0]]))
    # rho0 = operator_to_vector(Qobj([[1,0],[0,0]]))
    rhotar = operator_to_vector(Qobj([[0.1,0],[0,0.9]]))
    
    return L0, H_con, Ham_list, rho0, rhotar, times[:-1], glob_dim, None, 10, c_ops

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

    # Plot energy
    ax3.plot(iterations, result.energy_history, 'g-')
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Energy')
    ax3.set_title('Energy Evolution')
    ax3.grid(True)

    # Plot entropy
    ax4.plot(iterations, result.entropy_history, 'm-')
    ax4.set_xlabel('Iteration')
    ax4.set_ylabel('Entropy')
    ax4.set_title('Entropy Evolution')
    ax4.grid(True)

    plt.tight_layout()
    plt.show()

def main():
    # Setup and run optimization
    system_params = setup_quantum_system()
    evolution = create_evolution(*system_params)
    # Increased learning rate and iterations
    result = evolution.optimize(n_iters=200, learning_rate=0.07, constraint=0, fidelity_target=0)
    
    # Plot and print results
    plot_optimization_results(result)

if __name__ == "__main__":
    main()