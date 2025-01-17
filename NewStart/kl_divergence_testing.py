import torch
from QMaxCal.NewStart.TorchCleanUp2 import LindBladEvolve
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
from TorchCleanUpExample import setup_quantum_system
from QMaxCal.NewStart.TorchCleanUp2 import create_evolution
from random import sample
import numpy as np

def compare_trajectory_probabilities(system: LindBladEvolve, distribution: Dict, n_samples: int = 5):
    """Compare probabilities from distribution with direct calculation"""
    print("\nTrajectory Probability Comparison:")
    
    # Sample n_samples random trajectories from distribution
    trajectories = sample(list(distribution.keys()), min(n_samples, len(distribution)))
    
    for trajectory in trajectories:
        dist_prob = distribution[trajectory].item()  # Convert tensor to float
        calc_prob = get_trajectory_probability(system, list(trajectory)).item()  # Convert tensor to float
        
        print(f"\nTrajectory: {trajectory}")
        print(f"Distribution probability: {dist_prob:.9f}")
        print(f"Calculated probability: {calc_prob:.9f}")
        print(f"Relative difference: {abs(dist_prob - calc_prob)/dist_prob:.2%}")

def get_kl_divergence(p1: Dict, p2: Dict, system: LindBladEvolve) -> torch.Tensor:
    """Calculate Kullback-Leibler divergence between two probability distributions"""
    # Make deep copies of input dictionaries
    p1 = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in p1.items()}
    p2 = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in p2.items()}
    
    all_keys = set(p1.keys()).union(set(p2.keys()))
    overlapping_keys = set(p1.keys()).intersection(set(p2.keys()))
    missing_keys = set(p1.keys()) - set(p2.keys())
    
    print(f"Overlapping keys: {len(overlapping_keys)}")
    print(f"Total keys: {len(all_keys)}")
    print(f"Missing keys: {len(missing_keys)}")
    
    # Calculate probabilities for missing trajectories and track NaN values
    nan_keys = []
    for traj in missing_keys:
        prob = get_trajectory_probability(system, traj, ref=True)
        if torch.isnan(prob):
            nan_keys.append(traj)
            continue
        p2[traj] = prob
    
    print(f"NaN probabilities encountered: {len(nan_keys)}")
     
    # Also remove these keys from p1
    for k in nan_keys:
        if k in p1:
            print(f"Removing key {k} from p1")
            print(f"Value at key {k}: {p1[k]}")
            del p1[k]
    
    # Convert remaining values to tensors
    def to_tensor(val):
        if isinstance(val, torch.Tensor):
            return val
        return torch.tensor(val, dtype=torch.float64, requires_grad=True)
    
    p1 = {k: to_tensor(p1[k]) for k in p1.keys()}
    p2 = {k: to_tensor(p2[k]) for k in p1.keys()}
    
    # Normalize distributions using only valid values
    p1_sum = sum(p1.values())
    p2_sum = sum(p2.values())

    print(f"\nAfter NaN removal:")
    print(f"Sum of p1: {p1_sum}")
    print(f"Sum of p2: {p2_sum}")
    print(f"len(p1): {len(p1)}")
    print(f"len(p2): {len(p2)}")
    
    # Normalize
    p1 = {k: v/p1_sum for k, v in p1.items()}
    p2 = {k: v/p2_sum for k, v in p2.items()}

    # Compute KL divergence term by term
    kl_terms = {}
    for k in p1:
        term = p1[k] * torch.log(p1[k] / p2[k])
        kl_terms[k] = term

    kl_div = sum(kl_terms.values())
    return kl_div

def get_kl_divergence_with_penalty(p1: Dict, p2: Dict, system: LindBladEvolve) -> torch.Tensor:
    """Calculate Kullback-Leibler divergence between two probability distributions"""
    # Make deep copies of input dictionaries
    p1 = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in p1.items()}
    p2 = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in p2.items()}
    
    all_keys = set(p1.keys()).union(set(p2.keys()))
    overlapping_keys = set(p1.keys()).intersection(set(p2.keys()))
    missing_keys = set(p1.keys()) - set(p2.keys())
    
    print(f"Overlapping keys: {len(overlapping_keys)}")
    print(f"Total keys: {len(all_keys)}")
    print(f"Missing keys: {len(missing_keys)}")
    
    # Calculate probabilities for missing trajectories and track NaN values
    nan_count = 0
    for traj in missing_keys:
        prob = get_trajectory_probability(system, traj, ref=True)
        if torch.isnan(prob):
            nan_count += 1
            p2[traj] = prob  # Add NaN keys to p2
            continue
        p2[traj] = prob
    
    print(f"NaN probabilities encountered: {nan_count}")
    
    # Separate NaN and non-NaN trajectories
    p2_filtered = {k: p2[k] for k in p1.keys() if k in p2 and not torch.isnan(p2[k])}
    p2_Nan = {k: p2[k] for k in p2.keys() if k in p2 and torch.isnan(p2[k])}
    p1_filtered = {k: v for k, v in p1.items() if k not in p2_Nan}  # Non-NaN keys in p1
    p1_Nan = {k: v for k, v in p1.items() if k in p2_Nan}  # Keys in p1 that have NaN in p2
    
    # Convert to tensors while maintaining gradients
    def to_tensor(val):
        if isinstance(val, torch.Tensor):
            return val
        return torch.tensor(val, dtype=torch.float64, requires_grad=True)
    
    # Convert filtered dictionaries
    p1_filtered = {k: to_tensor(v) for k, v in p1_filtered.items()}
    p2_filtered = {k: to_tensor(v) for k, v in p2_filtered.items()}
    p1_Nan = {k: to_tensor(v) for k, v in p1_Nan.items()}
    
    # Normalize only the filtered (non-NaN) parts
    p1_sum = sum(p1_filtered.values())
    p2_sum = sum(p2_filtered.values())
    
    print(f"Sum of filtered p1: {p1_sum}")
    print(f"Sum of filtered p2: {p2_sum}")
    print(f"len(filtered p1): {len(p1_filtered)}")
    print(f"len(filtered p2): {len(p2_filtered)}")
    print(f"len(NaN keys): {len(p1_Nan)}")
    
    # Normalize non-NaN parts
    p1_filtered = {k: v/p1_sum for k, v in p1_filtered.items()}
    p2_filtered = {k: v/p2_sum for k, v in p2_filtered.items()}
    
    # add the NaN keys to p2 again
    p2.update(p2_Nan)  # Use update instead of addition

    # Compute KL divergence term by term
    kl_terms = {}
    kl_penalty = torch.tensor(3.0, dtype=torch.float64, requires_grad=True)
    
    # Add terms for non-NaN keys
    for k in p1_filtered:
        term = p1_filtered[k] * torch.log(p1_filtered[k] / p2_filtered[k])
        kl_terms[k] = term
    
    # Add penalty terms for NaN keys
    for k in p1_Nan:
        kl_terms[k] = kl_penalty * p1_Nan[k]
    
    kl_div = sum(kl_terms.values())
    return kl_div

def get_trajectory_probability(system: LindBladEvolve, trajectory_data: Tuple[int, List[int]], ref: bool = False) -> torch.Tensor:
    """Compute probability of a trajectory
    
    Args:
        trajectory_data: Tuple of (initial_position, events_list)
        ref: Whether to use reference Hamiltonian
    """
    init_pos, events = trajectory_data
    dt = system.dt
    
    # Initialize operators
    c_ops_tensors = [torch.tensor(c.full(), dtype=torch.complex128) for c in system.c_ops]
    c_ops_dag_c = [torch.tensor((c.dag() * c).full(), dtype=torch.complex128) for c in system.c_ops]
    sum_c_dag_c = sum(c_ops_dag_c)
    
    # Set up initial state based on initial position
    current_state = torch.zeros(system.dim, dtype=torch.complex128)
    current_state[init_pos] = 1.0
    
    prob = torch.tensor(1.0, dtype=torch.float64)
    
    # Follow trajectory events
    for i, event in enumerate(events):
        # Get Hamiltonian and effective non-Hermitian part
        H = system._get_ham_no_controls() if ref else system._get_ham(i)
        H_eff = H - 0.5j * sum_c_dag_c
        
        # No-jump evolution probability
        old_current_state = current_state
        propagator = torch.matrix_exp(-1j * dt * H_eff)
        current_state = propagator @ current_state
        no_jump_prob = (current_state.conj() @ current_state).real
        
        if event > 0:  # Jump occurred
            prob_for_jump = (old_current_state.conj() @ old_current_state).real
            prob = prob * prob_for_jump
            
            # Calculate jump probabilities
            jump_probs = torch.stack([
                (old_current_state.conj() @ c_ops_dag_c[i] @ old_current_state).real
                for i in range(len(c_ops_dag_c))
            ])
            jump_probs = jump_probs / jump_probs.sum()
            
            # Multiply by probability of specific jump
            prob = prob * jump_probs[event - 1]
            
            # Update state after jump
            current_state = torch.tensor(system.c_ops[event - 1].full(), dtype=torch.complex128) @ old_current_state
            current_state = current_state / torch.norm(current_state)
            
    final_prob = prob * (current_state.conj() @ current_state).real
    return final_prob

def compare_trajectory_distributions(system: LindBladEvolve):
    """Compare trajectory distributions between reference and control"""
    # Get reference distribution
    _, ref_dist, _, _ = system._probability_distribution_estimate(ref=False, max_iters=450)
    
    # Get control distribution 
    _, control_dist, _, _ = system._probability_distribution_estimate(ref=False, max_iters=450)
    
    # Calculate KL divergence
    kl_div = get_kl_divergence(control_dist, ref_dist, system)
    
    print("\nDistribution Comparison:")
    print(f"KL divergence: {kl_div}")
    
    return kl_div

def test_specific_trajectory(system: LindBladEvolve, init_pos: int, trajectory: List[int]):
    """Test a specific trajectory with initial state specified by position"""
    full_trajectory = [init_pos] + trajectory
    
    print(f"\nTesting trajectory")
    print(f"Initial state position: {init_pos}")
    print(f"Trajectory events: {trajectory}")
    
    prob = get_trajectory_probability(system, full_trajectory, ref=False)
    
    print(f"Probability: {prob.item():.9f}")
    return prob

def run_kl_sensitivity_test(n_trials: int = 10, max_iters: int = 100):
    """Test KL divergence sensitivity to initial conditions and sampling for both methods"""
    # Setup system
    system_params = setup_quantum_system()
    evolution = create_evolution(*system_params)
    
    # Initialize control parameters
    n_ts = len(evolution.time_list)
    evolution.ctrls_real = torch.rand((n_ts, evolution.n_ctrls))
    evolution.ctrls_im = torch.rand((n_ts, evolution.n_ctrls))
    
    # Run multiple trials
    kl_values = []
    kl_penalty_values = []
    print(f"\nRunning {n_trials} trials...")
    
    for i in range(n_trials):
        print(f"\nTrial {i+1}/{n_trials}")
        _, ref_dist, _, _ = evolution._probability_distribution_estimate(ref=True, max_iters=max_iters)
        _, control_dist, _, _ = evolution._probability_distribution_estimate(ref=False, max_iters=max_iters)
        
        print("\nTesting KL divergence with penalty:")
        kl_div_penalty = get_kl_divergence_with_penalty(control_dist, ref_dist, evolution)
        print(f"KL with penalty: {kl_div_penalty.item()}")
        kl_penalty_values.append(kl_div_penalty.item())
        
        print("\nTesting regular KL divergence:")
        kl_div = get_kl_divergence(control_dist, ref_dist, evolution)
        print(f"Regular KL: {kl_div.item()}")
        kl_values.append(kl_div.item())
        
    
    # Compute statistics for both methods
    for name, values in [("Regular KL", kl_values), ("KL with penalty", kl_penalty_values)]:
        mean_val = np.mean(values)
        std_val = np.std(values)
        min_val = min(values)
        max_val = max(values)
        
        print(f"\n{name} Statistics:")
        print(f"Mean: {mean_val:.6f}")
        print(f"Std Dev: {std_val:.6f}")
        print(f"Min: {min_val:.6f}")
        print(f"Max: {max_val:.6f}")
        print(f"Range: {max_val - min_val:.6f}")
    
    # Plot distributions side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    ax1.hist(kl_values, bins='auto', alpha=0.7)
    ax1.set_title('Regular KL Divergence')
    ax1.set_xlabel('KL Divergence')
    ax1.set_ylabel('Frequency')
    ax1.grid(True, alpha=0.3)
    
    ax2.hist(kl_penalty_values, bins='auto', alpha=0.7)
    ax2.set_title('KL Divergence with Penalty')
    ax2.set_xlabel('KL Divergence')
    ax2.set_ylabel('Frequency')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return kl_values, kl_penalty_values

def main():
    # Run sensitivity test
    regular_kl, penalty_kl = run_kl_sensitivity_test(n_trials=10, max_iters=100)

if __name__ == "__main__":
    main()