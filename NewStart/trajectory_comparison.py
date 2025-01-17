import numpy as np
import matplotlib.pyplot as plt
from qutip import mcsolve, basis, destroy, Qobj, mesolve, expect
from qutip import *
import torch
from TorchCleanUpExample import setup_quantum_system
from QMaxCal.NewStart.TorchCleanUp2 import create_evolution
from Utils import fidelity

def run_torch_trajectories(evolution, n_samples=100):
    """Run multiple trajectories using _sample_trajectory_new"""
    all_trajectories = []
    final_states = []  # Add storage for final states
    
    for _ in range(n_samples):
        result = evolution._sample_trajectory_new(ref=False)
        # Convert state trajectory to density matrices and extract populations
        populations = []
        for state in result.state_trajectory:
            norm = (state.conj() @ state).real.item()
            state = state / torch.sqrt(torch.tensor(norm))
            populations.append((state[1]*state[1].conj()).real)
        all_trajectories.append(populations)
        
        # Store final state as density matrix
        final_state = result.state_trajectory[-1]
        norm = (final_state.conj() @ final_state).real.item()
        final_state = final_state / torch.sqrt(torch.tensor(norm))
        final_density = torch.outer(final_state, final_state.conj())
        final_states.append(final_density)
    
    # Calculate average trajectory and final state
    avg_trajectory = np.mean(all_trajectories, axis=0)
    std_trajectory = np.std(all_trajectories, axis=0)
    avg_final_state = sum(final_states) / len(final_states)
    
    return avg_trajectory, std_trajectory, result.time_points, avg_final_state

def run_qutip_mcsolve(evolution, n_trajectories=100,):
    """Run quantum trajectories using QuTiP's mcsolve"""
    # Get system parameters from setup function
    L0, H_con, Ham_list, rho0, rhotar, times, glob_dim, _, _, c_ops = setup_quantum_system()
    
    # Initial state as wave function
    psi0 = basis(glob_dim, 0)
    
    # Run mcsolve with updated options
    #print("resultmc")
    print(evolution.ctrls_real)
    H = [Qobj(2*Ham_list[0][0]), [Qobj(Ham_list[1][0] + Ham_list[1][1]), evolution.ctrls_real], [Qobj(Ham_list[1][0] - Ham_list[1][1]), evolution.ctrls_im*1j]]
    result = mcsolve(H, psi0, times, c_ops, ntraj=n_trajectories)
    print(result.states)
    #print("resultmc")
    
    # Extract excited state populations
    populations = []
    for state in result.states:
        if isinstance(state, Qobj):  # Single trajectory case
            populations.append(state[1][1].real)
    
    populations = np.array(populations)
    #avg_trajectory = np.mean(populations, axis=1)
    #std_trajectory = np.std(populations, axis=1)
    return populations, times

def run_qutip_mesolve(times, evolution):
    """Run quantum master equation using QuTiP's mesolve"""
    # Get system parameters from setup function
    L0, H_con, Ham_list, rho0, rhotar, times, glob_dim, _, _, c_ops = setup_quantum_system()
    
    # Initial state as density matrix (matching the state used in mcsolve)
    #psi0 = basis(glob_dim, 0)
    #rho0 = operator_to_vector(Qobj([[0.7,0],[0,0.3]]))
    rho0 = Qobj([[1,0],[0,0]])
    #psi0 = basis(glob_dim, 0)
    # Run mesolve
    H = [Qobj(2*Ham_list[0][0]), [Qobj(Ham_list[1][0] + Ham_list[1][1]), evolution.ctrls_real], [Qobj(Ham_list[1][0] - Ham_list[1][1]), evolution.ctrls_im*1j]]
    result = mesolve(H, rho0, times, c_ops)
    #result = mesolve(H, psi0, times, c_ops)
    print(result.states)
    # Extract excited state populations using number operator
    # n_op = destroy(glob_dim).dag() * destroy(glob_dim)
    #populations = [expect(n_op, state) for state in result.states]
    #print(result.states)
    populations = [state[1][1].real for state in result.states]
    
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

def compare_fidelities():
    """Compare fidelities for different numbers of samples"""
    # Setup systems
    system_params = setup_quantum_system()
    evolution = create_evolution(*system_params)

    # Initialize control parameters to zero
    n_ts = len(evolution.time_list)
    evolution.ctrls_real = torch.zeros(n_ts, evolution.n_ctrls)
    evolution.ctrls_im = torch.zeros(n_ts, evolution.n_ctrls)
    
    # Get mesolve result for comparison
    times = system_params[5]  # Extract times from system_params
    mesolve_states = run_qutip_mesolve(times)
    final_mesolve_state = Qobj([[1-mesolve_states[-1], 0], [0, mesolve_states[-1]]])
    
    # Sample sizes to test
    n_samples_list = [50, 100, 150, 250, 400, 600, 800, 1000, 1200, 1500, 1700]
    fidelities = []
    
    # Run comparisons
    for n_samples in n_samples_list:
        print(f"Running with {n_samples} samples...")
        _, _, _, avg_final_state = run_torch_trajectories(evolution, n_samples)
        
        # Convert final mesolve state to tensor format
        mesolve_tensor = torch.tensor(final_mesolve_state.full(), dtype=torch.complex128)
        
        # Calculate fidelity using imported function
        fid = float(fidelity(avg_final_state, mesolve_tensor))
        fidelities.append(fid)
        print(f"Fidelity: {fid:.4f}")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(n_samples_list, fidelities, 'bo-')
    plt.xlabel('Number of samples')
    plt.ylabel('Fidelity with mesolve result')
    plt.title('Convergence of Quantum Trajectories')
    plt.grid(True)
    plt.show()

def compare_multiple_runs(evolution, n_runs=5, n_samples=100):
    """Compare multiple runs of trajectory simulations"""
    all_run_results = []
    
    # Run multiple sets of trajectories
    for i in range(n_runs):
        print(f"Running set {i+1}/{n_runs}...")
        avg_traj, std_traj, times, _ = run_torch_trajectories(evolution, n_samples)
        all_run_results.append((avg_traj, std_traj, times))
    
    # Plot results
    plt.figure(figsize=(12, 8))
    colors = plt.cm.viridis(np.linspace(0, 1, n_runs))
    
    for i, (avg, std, times) in enumerate(all_run_results):
        plt.plot(times, avg, color=colors[i], label=f'Run {i+1}', alpha=0.8)
        plt.fill_between(times, avg - std, avg + std, color=colors[i], alpha=0.2)
    
    plt.xlabel('Time')
    plt.ylabel('Excited state population')
    plt.title(f'Comparison of {n_runs} Different Trajectory Sets\n({n_samples} trajectories each)')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    # Add fidelity comparison
    # compare_fidelities()
    
    # Setup system and create evolution object
    system_params = setup_quantum_system()
    evolution = create_evolution(*system_params)
    
    # Initialize control parameters
    n_ts = len(evolution.time_list)
    evolution.ctrls_real = torch.rand(n_ts, evolution.n_ctrls)*10
    evolution.ctrls_im = torch.rand(n_ts, evolution.n_ctrls)*10
    
    # Run all original comparison methods
    torch_avg, torch_std, torch_times,_ = run_torch_trajectories(evolution)
    qutip_avg, qutip_times = run_qutip_mcsolve(evolution)
    mesolve_avg = run_qutip_mesolve(qutip_times, evolution)
    
    # # Run multiple trajectory sets comparison
    # print("\nRunning multiple trajectory sets comparison...")
    # compare_multiple_runs(evolution, n_runs=5, n_samples=100)
    
    # Compute errors
    error_mcsolve = compute_trajectory_error(torch_times, torch_avg, qutip_times, qutip_avg)
    error_mesolve = compute_trajectory_error(torch_times, torch_avg, qutip_times, mesolve_avg)
    
    # Analyze jumps
    jump_times, jump_counts = analyze_jumps(evolution)
    
    # Create original comparison plots
    # Create figure with subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), height_ratios=[2, 1, 1])
    
    # Top subplot: All solutions
    ax1.plot(torch_times, torch_avg, 'b-', label='Torch trajectories')
    ax1.fill_between(torch_times, torch_avg - torch_std, torch_avg + torch_std, color='b', alpha=0.2)
    ax1.plot(qutip_times, qutip_avg, 'r--', label='QuTiP mcsolve')
    #ax1.fill_between(qutip_times, qutip_avg - qutip_std, qutip_avg + qutip_std, color='r', alpha=0.2)
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