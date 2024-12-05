import numpy as np
import matplotlib.pyplot as plt
from qutip import mcsolve, basis, destroy, Qobj, mesolve, expect
import torch
from TorchCleanUpExample import setup_quantum_system
from TorchCleanUp import create_evolution

def run_torch_trajectories(evolution, n_samples=1000):
    """Run multiple trajectories using _sample_trajectory_new"""
    all_trajectories = []
    for _ in range(n_samples):
        result = evolution._sample_trajectory_new(ref=False)
        # Convert state trajectory to density matrices and extract populations
        populations = []
        for state in result.state_trajectory:
            density_matrix = torch.outer(state, state.conj())
            populations.append(density_matrix[1,1].real.item())  # Extract |1⟩ population
        all_trajectories.append(populations)
    
    # Calculate average trajectory
    avg_trajectory = np.mean(all_trajectories, axis=0)
    std_trajectory = np.std(all_trajectories, axis=0)
    return avg_trajectory, std_trajectory, result.time_points

def run_qutip_mcsolve(n_trajectories=1000):
    """Run quantum trajectories using QuTiP's mcsolve"""
    # Get system parameters from setup function
    L0, H_con, Ham_list, rho0, rhotar, times, glob_dim, _, _, c_ops = setup_quantum_system()
    
    # Initial state as wave function
    psi0 = np.sqrt(0.7) * basis(glob_dim, 0) + np.sqrt(0.3) * basis(glob_dim, 1)
    
    # Run mcsolve with updated options
    result = mcsolve(L0, psi0, times, c_ops, ntraj=n_trajectories)
    
    # Extract excited state populations
    populations = []
    for state in result.states:
        if isinstance(state, Qobj):  # Single trajectory case
            pop = (state.dag() * destroy(glob_dim).dag() * destroy(glob_dim) * state).tr().real
            populations.append([pop])
        else:  # Multiple trajectories
            pops = [(s.dag() * destroy(glob_dim).dag() * destroy(glob_dim) * s).tr().real 
                   for s in state]
            populations.append(pops)
    
    populations = np.array(populations)
    avg_trajectory = np.mean(populations, axis=1)
    std_trajectory = np.std(populations, axis=1)
    return avg_trajectory, std_trajectory, times

def run_qutip_mesolve(times):
    """Run quantum master equation using QuTiP's mesolve"""
    # Get system parameters from setup function
    L0, H_con, Ham_list, rho0, rhotar, _, glob_dim, _, _, c_ops = setup_quantum_system()
    
    # Initial state as density matrix (matching the state used in mcsolve)
    psi0 = np.sqrt(0.7) * basis(glob_dim, 0) + np.sqrt(0.3) * basis(glob_dim, 1)
    rho0 = psi0 * psi0.dag()
    
    # Run mesolve
    result = mesolve(L0, rho0, times, c_ops)
    
    # Extract excited state populations using number operator
    n_op = destroy(glob_dim).dag() * destroy(glob_dim)
    populations = [expect(n_op, state) for state in result.states]
    
    return np.array(populations)

def compute_trajectory_error(torch_times, torch_avg, qutip_times, qutip_avg):
    """Compute error between torch and qutip trajectories"""
    # Interpolate QuTiP data to match Torch time points
    from scipy.interpolate import interp1d
    qutip_interp = interp1d(qutip_times, qutip_avg)
    qutip_resampled = qutip_interp(torch_times)
    
    # Compute absolute error
    error = np.abs(torch_avg - qutip_resampled)
    return error

def analyze_jumps(evolution, n_samples=1000, time_bins=50):
    """Analyze jump statistics from multiple trajectories"""
    all_jump_times = []
    jump_counts = []
    
    for _ in range(n_samples):
        result = evolution._sample_trajectory_new(ref=False)
        all_jump_times.extend(result.jump_times)
        jump_counts.append(len(result.jump_times))
    
    return all_jump_times, jump_counts

def main():
    # Setup system and create evolution object
    system_params = setup_quantum_system()
    evolution = create_evolution(*system_params)
    
    # Initialize control parameters to zero
    n_ts = len(evolution.time_list)
    evolution.ctrls_real = torch.zeros((n_ts, evolution.n_ctrls))
    evolution.ctrls_im = torch.zeros((n_ts, evolution.n_ctrls))
    
    # Run all methods to compare
    torch_avg, torch_std, torch_times = run_torch_trajectories(evolution)
    qutip_avg, qutip_std, qutip_times = run_qutip_mcsolve()
    mesolve_avg = run_qutip_mesolve(qutip_times)
    
    # Compute errors
    error_mcsolve = compute_trajectory_error(torch_times, torch_avg, qutip_times, qutip_avg)
    error_mesolve = compute_trajectory_error(torch_times, torch_avg, qutip_times, mesolve_avg)
    
    # Analyze jumps
    jump_times, jump_counts = analyze_jumps(evolution)
    
    # Create figure with subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), height_ratios=[2, 1, 1])
    
    # Top subplot: All solutions
    ax1.plot(torch_times, torch_avg, 'b-', label='Torch trajectories')
    ax1.fill_between(torch_times, torch_avg - torch_std, torch_avg + torch_std, color='b', alpha=0.2)
    ax1.plot(qutip_times, qutip_avg, 'r--', label='QuTiP mcsolve')
    ax1.fill_between(qutip_times, qutip_avg - qutip_std, qutip_avg + qutip_std, color='r', alpha=0.2)
    ax1.plot(qutip_times, mesolve_avg, 'g-.', label='QuTiP mesolve')
    ax1.set_ylabel('Excited state population')
    ax1.set_title('Comparison of quantum solutions')
    ax1.legend()
    ax1.grid(True)
    
    # Middle subplot: Errors
    ax2.plot(torch_times, error_mcsolve, 'r-', label='Error vs mcsolve')
    ax2.plot(torch_times, error_mesolve, 'g-', label='Error vs mesolve')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Absolute Error')
    ax2.legend()
    ax2.grid(True)
    
    # Bottom subplot: Jump statistics
    ax3.hist(jump_times, bins=50, density=True, alpha=0.7, color='b', label='Jump distribution')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Jump density')
    ax3.set_title(f'Jump Statistics (avg: {np.mean(jump_counts):.2f} jumps/trajectory)')
    ax3.grid(True)
    ax3.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
    plt.show()  # This ensures the plot window stays open