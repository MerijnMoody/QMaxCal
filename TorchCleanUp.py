# Standard library imports
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple
from collections import defaultdict

# Third-party imports
import torch
from torch.optim.optimizer import Optimizer
import numpy as np
from qutip import Qobj

# Local imports
from optimizers import QuantumOptimizer, OptimizationResult
from Utils import BasisConfig, fidelity, evaluate_basis

@dataclass
class TrajectoryResult:
    """Container for trajectory calculation results"""
    final_state: torch.Tensor
    probability: torch.Tensor
    energy: torch.Tensor
    events: List[int]
    state_trajectory: List[torch.Tensor]  # Added this field
    time_points: torch.Tensor  # Added this field
    jump_times: List[float]  # Add this field
    jump_types: List[int]    # Add this field

class LindBladEvolve(torch.nn.Module):
    """Quantum system evolution under Lindblad dynamics"""
    
    def __init__(self):
        super().__init__()
        # Essential system parameters
        self.time_list: torch.Tensor = None
        self.dt: float = None
        self.n_ts: int = None
        self.dim: int = None
        self.n_ctrls: int = None
        self.n_params: int = None
        
        # System operators
        self.ham_list: List = None
        self.dyn_gen: torch.Tensor = None
        self.ctrl_gen: List = None
        self.c_ops: List[Qobj] = []
        
        # States
        self.initial_density: torch.Tensor = None
        self.target: torch.Tensor = None
        
        # Control parameters
        self.ctrls_real: torch.Tensor = None
        self.ctrls_im: torch.Tensor = None
        
        # Evolution results
        self.ref_evo: Optional[List[torch.Tensor]] = None
        self.ref_prob_dist: Dict = None
        self.energy_ref_avg: float = None
        self.control_evo: List[torch.Tensor] = []
        
        # Results tracking
        self.fidelity: torch.Tensor = torch.tensor(0.0)
        self.relative_entropy: torch.Tensor = torch.tensor(0.0)
        self.energy: torch.Tensor = torch.tensor(0.0)
        
        # Optimization parameters
        self.lam: torch.Tensor = torch.tensor(0.0)
        self.lam2: torch.Tensor = torch.tensor(0.0)

    def _get_ham(self, k: int) -> torch.Tensor:
        """Get Hamiltonian at time step k"""
        ham = torch.tensor(self.ham_list[0][0] + self.ham_list[0][1])
        for j, hams in enumerate(self.ham_list[1:]):
            ham = ham + self.ctrls_real[k, j]*(torch.tensor(hams[0] + hams[1]))
            ham = ham + 1j*self.ctrls_im[k, j]*(torch.tensor(hams[0] - hams[1]))
        return ham

    def _get_ham_no_controls(self) -> torch.Tensor:
        """Get base Hamiltonian without controls"""
        return torch.tensor(self.ham_list[0][0] + self.ham_list[0][1])

    def _density_to_vector(self, density_matrix: Qobj) -> Qobj:
        """Convert density matrix to state vector
        
        For a diagonal density matrix with probabilities p0, p1,
        returns state vector [√p0, √p1]
        """
        # Extract diagonal elements (probabilities)
        diag = np.diagonal(density_matrix.full())
        # Convert probabilities to amplitudes by taking sqrt
        amplitudes = np.sqrt(diag)
        return Qobj(amplitudes)

    def _sample_trajectory(self, ref: bool) -> TrajectoryResult:
        """Sample a single quantum trajectory"""
        # Pre-compute tensors
        dt = self.dt
        c_ops_tensors = [torch.tensor(c.full(), dtype=torch.complex128) for c in self.c_ops]
        c_ops_dag_c = [torch.tensor((c.dag() * c).full(), dtype=torch.complex128) for c in self.c_ops]
        kraus_operators = [c * np.sqrt(dt) for c in c_ops_tensors]
        
        # Initialize state
        current_state = torch.tensor(
            self._density_to_vector(self.initial_density).full(), 
            dtype=torch.complex128
        ).view(-1)
        
        # Setup tracking variables
        t_steps = len(self.time_list)
        events = torch.zeros(t_steps-1, dtype=torch.long)
        energy = torch.tensor(0.0, dtype=torch.float64)
        prob_trajectory = torch.tensor(1.0, dtype=torch.float64)
        
        # Pre-compute common matrices
        eye = torch.eye(2, dtype=torch.complex128)
        sum_c_dag_c = sum(c_ops_dag_c)
        
        # Evolution loop
        for i in range(t_steps-1):
            H = self._get_ham_no_controls() if ref else self._get_ham(i)
            
            # Calculate jump probabilities with gradient tracking
            probs = torch.stack([
                (current_state.conj() @ c @ current_state * dt).real
                for c in c_ops_dag_c
            ])
            
            # Debug print for probabilities
            print(f"Step {i}, Probabilities:", probs)
            
            no_jump_prob = 1 - probs.sum()
            probs = torch.cat([torch.tensor([no_jump_prob]), probs])
            probs = probs / probs.sum()
            print(f"After normalization:", probs)

            # Sample without breaking gradient
            with torch.no_grad():  # Only the sampling should be without gradient
                event_index = torch.multinomial(probs, 1)[0]
            print(f"Selected event: {event_index}")
            prob_trajectory = prob_trajectory * probs[event_index]  # Keep gradient here
            
            # Update state while maintaining gradient flow
            if event_index == 0:
                current_state = (eye - (1j * H + 0.5 * sum_c_dag_c) * dt) @ current_state
            else:
                current_state = kraus_operators[event_index - 1] @ current_state
            
            current_state = current_state / torch.norm(current_state)
            energy = energy + (current_state.conj() @ H @ current_state).real

        return TrajectoryResult(
            final_state=current_state,  # Maintains gradient
            probability=prob_trajectory,  # Maintains gradient
            energy=energy,
            events=events.tolist()
        )
    
    def _sample_trajectory_new(self, ref: bool) -> TrajectoryResult:
        """Implement quantum jump algorithm for trajectory sampling"""
        dt = self.dt
        t_steps = len(self.time_list)
        
        # Initialize operators and state
        c_ops_tensors = [torch.tensor(c.full(), dtype=torch.complex128) for c in self.c_ops]
        c_ops_dag_c = [torch.tensor((c.dag() * c).full(), dtype=torch.complex128) for c in self.c_ops]
        current_state = torch.tensor(
            self._density_to_vector(self.initial_density).full(), 
            dtype=torch.complex128
        ).view(-1)
        
        # Setup tracking variables
        events = []
        energy = torch.tensor(0.0, dtype=torch.float64)
        prob_trajectory = torch.tensor(1.0, dtype=torch.float64)
        
        # Calculate effective non-Hermitian Hamiltonian
        sum_c_dag_c = sum(c_ops_dag_c)
        
        # Generate initial random threshold
        with torch.no_grad():
            r = torch.rand(1).item()
        
        # Add trajectory storage
        state_trajectory = [current_state.clone()]
        time_points = self.time_list
        
        # Add jump tracking
        jump_times = []
        jump_types = []
        
        for i in range(t_steps-1):
            # Get current Hamiltonian
            H = self._get_ham_no_controls() if ref else self._get_ham(i)
            H_eff = H - 0.5j * sum_c_dag_c
            
            # Propagate with effective Hamiltonian using matrix exponential
            propagator = torch.matrix_exp(-1j * dt * H_eff)
            current_state = propagator @ current_state
            norm_squared = (current_state.conj() @ current_state).real.item()
            
            if norm_squared <= r or norm_squared < 1e-6:
                # Calculate jump probabilities first
                jump_probs = torch.stack([
                    (current_state.conj() @ c_dag_c @ current_state).real
                    for c_dag_c in c_ops_dag_c
                ])
                jump_probs = jump_probs / jump_probs.sum()
                
                with torch.no_grad():
                    jump_idx = torch.multinomial(jump_probs, 1)[0]
                
                # Record jump time and type after we have jump_idx
                jump_times.append(time_points[i].item())
                jump_types.append(jump_idx.item())
                
                # Apply the jump
                current_state = c_ops_tensors[jump_idx] @ current_state
                norm_squared_after_jump = (current_state.conj() @ current_state).real.item()
                events.append(jump_idx.item() + 1)
                prob_trajectory = prob_trajectory * (1 - norm_squared)
                # normalize state
                current_state = current_state / torch.sqrt(torch.tensor(norm_squared_after_jump))
                
                # Generate new random threshold after jump
                with torch.no_grad():
                    r = torch.rand(1).item()
            else:
                events.append(0)
                prob_trajectory = prob_trajectory * norm_squared

            # Normalize state and store trajectory
            # current_state = current_state / torch.sqrt(torch.tensor(norm_squared))
            state_trajectory.append(current_state.clone())
            
            energy = energy + (current_state.conj() @ H @ current_state).real
        
        return TrajectoryResult(
            final_state=current_state,
            probability=prob_trajectory,
            energy=energy,
            events=events,
            state_trajectory=state_trajectory,
            time_points=time_points,
            jump_times=jump_times,      # Add these
            jump_types=jump_types       # new fields
        )

    def _probability_distribution_estimate(self, ref: bool, max_iters: int) -> Tuple[List[torch.Tensor], Dict, torch.Tensor, torch.Tensor]:
        """Estimate probability distribution from quantum trajectories"""
        num_samples = 0
        energy_sum = torch.zeros(1, dtype=torch.float64)
        fidelity_sum = torch.zeros(1, dtype=torch.float64)
        probability_distribution = defaultdict(lambda: torch.zeros(1, dtype=torch.float64))
        done = set()

        # Sample trajectories
        while num_samples < max_iters:
            num_samples += 1
            result = self._sample_trajectory(ref)
            
            # Update statistics without in-place operations
            current_fidelity = fidelity(
                torch.outer(result.final_state, result.final_state.conj()), 
                self.target.view(self.dim, self.dim)
            )
            fidelity_sum = fidelity_sum + current_fidelity
            energy_sum = energy_sum + result.energy

            # Record unique trajectories
            events_tuple = tuple(result.events)
            if events_tuple not in done:
                probability_distribution[events_tuple] = result.probability
                done.add(events_tuple)

        # Compute averages without in-place operations
        fidelity_avg = fidelity_sum / num_samples
        energy_avg = energy_sum / num_samples

        # Normalize probability distribution
        total_prob = sum(probability_distribution.values())
        if total_prob > 0:
            probability_distribution = {
                k: v/total_prob 
                for k, v in probability_distribution.items()
            }
                
        return [], probability_distribution, energy_avg, fidelity_avg

    def _get_kl_divergence(self, p1: Dict, p2: Dict, epsilon: float = 0.005) -> torch.Tensor:
        """Calculate Kullback-Leibler divergence between two probability distributions"""
        # Debug prints
        print("\nDebugging KL divergence:")
        print("Input distributions:")
        print("p1:", {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in p1.items()})
        print("p2:", {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in p2.items()})
        
        all_keys = set(p1.keys()).union(set(p2.keys()))
        
        # Convert to tensors while maintaining gradients
        def to_tensor(val):
            if isinstance(val, torch.Tensor):
                return val
            return torch.tensor(val, dtype=torch.float64, requires_grad=True)
        
        p1 = {k: to_tensor(p1.get(k, epsilon)) for k in all_keys}
        p2 = {k: to_tensor(p2.get(k, epsilon)) for k in all_keys}
        
        print("\nAfter tensor conversion:")
        print("p1:", {k: v.item() for k, v in p1.items()})
        print("p2:", {k: v.item() for k, v in p2.items()})

        # Normalize distributions
        p1_sum = sum(p1.values())
        p2_sum = sum(p2.values())
        print("\nSums before normalization:")
        print(f"p1_sum: {p1_sum.item()}, p2_sum: {p2_sum.item()}")
        
        p1 = {k: v/p1_sum for k, v in p1.items()}
        p2 = {k: v/p2_sum for k, v in p2.items()}
        
        print("\nAfter normalization:")
        print("p1:", {k: v.item() for k, v in p1.items()})
        print("p2:", {k: v.item() for k, v in p2.items()})

        # Compute KL divergence term by term
        kl_terms = []
        for k in p1:
            term = p1[k] * torch.log(p1[k] / p2[k])
            kl_terms.append(term)
            print(f"\nTerm for key {k}:")
            print(f"p1[k]: {p1[k].item():.6f}")
            print(f"p2[k]: {p2[k].item():.6f}")
            print(f"log term: {torch.log(p1[k] / p2[k]).item():.6f}")
            print(f"full term: {term.item():.6f}")

        kl_div = sum(kl_terms)
        print(f"\nFinal KL divergence: {kl_div.item():.6f}")
        return kl_div

    def _get_err(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Calculate error metrics for current controls"""
       # Get controlled evolution
        evolution, p1, energy_average, fidelity_avg = self._probability_distribution_estimate(
            ref=False, max_iters=250
        )
        self.control_evo = evolution
        self.PathDistEvo = p1

        # Calculate reference evolution if needed
        if self.ref_evo is None:
            ref_evolution, p2, energy_ref_avg, _ = self._probability_distribution_estimate(
                ref=True, max_iters=5
            )
            self.ref_evo = ref_evolution
            self.ref_prob_dist = p2
            self.energy_ref_avg = energy_ref_avg

        relative_entropy = self._get_kl_divergence(p1, self.ref_prob_dist)
        return relative_entropy, energy_average, fidelity_avg

    def _loss_function(self, args: List) -> torch.Tensor:
        """Calculate total loss for optimization
        
        Args:
            args: List containing [params_real, params_im, lam2, lam]
        """
        # Unpack arguments
        params_real, params_im = args[0], args[1]
        self.lam2, self.lam = args[-2], args[-1]

        # Setup time points and basis
        T = self.time_list[-1]
        t_points = torch.linspace(0, T, self.n_ts)
        fourier_config = BasisConfig(
            type='fourier',
            degree=params_real.shape[0]-1,
            n_params=params_real.shape[0],
            period=T
        )

        # Evaluate controls
        self.ctrls_real = evaluate_basis(params_real, t_points, fourier_config)
        self.ctrls_im = evaluate_basis(params_im, t_points, fourier_config)

        # Calculate metrics
        relative_entropy, energy_average, fidelity_avg = self._get_err()
        self.fidelity = fidelity_avg
        self.relative_entropy = relative_entropy
        self.energy = energy_average

        # Calculate regularization terms
        reg_error = torch.linalg.vector_norm(self.ctrls_real) + torch.linalg.vector_norm(self.ctrls_im)
        smoothness_error = (
            torch.linalg.vector_norm(self.ctrls_real[1:] - self.ctrls_real[:-1]) + 
            torch.linalg.vector_norm(self.ctrls_im[1:] - self.ctrls_im[:-1])
        )

        # Compute total loss (removed the zero multipliers)
        return (
            self.lam * fidelity_avg * 5000 + 
            relative_entropy + 
            self.lam2 * (energy_average + 5) ** 2 + 
            100 * reg_error
        )

    def optimize(self, n_iters: int, learning_rate: float, 
                constraint: Optional[float], fidelity_target: Optional[float]):
        optimizer = QuantumOptimizer(self, learning_rate)
        result = optimizer.optimize(n_iters, constraint, fidelity_target)
        return result

def create_evolution(
    dyn_gen: np.ndarray,
    H_con: List[Tuple[np.ndarray, np.ndarray]],
    ham_list: List[Tuple[np.ndarray, np.ndarray]],
    rho0: Qobj,  # This is operator_to_vector(density_matrix)
    rhotar: Qobj,
    time_list: np.ndarray,
    dim: int,
    init_ctrls: np.ndarray,
    n_params: int,
    c_ops: List[Qobj]
) -> LindBladEvolve:
    """Create and initialize a LindBladEvolve instance
    
    Args:
        dyn_gen: Dynamics generator matrix
        H_con: List of control Hamiltonians
        ham_list: List of Hamiltonian components
        rho0: Initial density matrix
        rhotar: Target density matrix
        time_list: List of time points
        dim: System dimension
        init_ctrls: Initial control parameters
        n_params: Number of basis parameters
        c_ops: List of collapse operators
        
    Returns:
        Initialized LindBladEvolve instance
    """
    evolution = LindBladEvolve()
    
    # Time parameters
    evolution.time_list = time_list
    evolution.dt = time_list[1] - time_list[0]
    evolution.n_ts = len(time_list)
    
    # System parameters
    evolution.dim = dim
    evolution.n_params = n_params
    evolution.n_ctrls = len(H_con)
    
    # Convert operators to tensors
    evolution.ham_list = ham_list
    evolution.dyn_gen = torch.from_numpy(dyn_gen.full())
    evolution.ctrl_gen = [
        [torch.from_numpy(H[0].full()), torch.from_numpy(H[1].full())]
        for H in H_con
    ]
    evolution.c_ops = c_ops
    
    # State initialization - convert back from operator_to_vector to density matrix
    rho0_matrix = Qobj(rho0.full().reshape((dim, dim)))  # Reshape vector back to matrix
    evolution.initial_density = rho0_matrix
    evolution.target = torch.from_numpy(rhotar.full())
    evolution.init_ctrls = init_ctrls

    return evolution