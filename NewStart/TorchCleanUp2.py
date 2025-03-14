# Standard library imports
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple
from collections import defaultdict
import time
from statistics import mean

# Third-party imports
import torch

from torch.optim.optimizer import Optimizer
import numpy as np
from qutip import Qobj

# Local imports
from optimizers import QuantumOptimizer, OptimizationResult
from Utils import BasisConfig, fidelity, evaluate_basis, calculate_frob_norm

import torch.multiprocessing as mp
from functools import partial

@dataclass
class TrajectoryResult:
    """Container for trajectory calculation results"""
    final_state: torch.Tensor
    probability: torch.Tensor
    energy: torch.Tensor
    events: List[int]
    state_trajectory: List[torch.Tensor]
    time_points: torch.Tensor
    jump_times: List[float]
    jump_types: List[int]

class LindBladEvolve(torch.nn.Module):
    """Quantum system evolution under Lindblad dynamics"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device('cpu')

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
        self.fidelity: torch.Tensor = torch.tensor(0.0, device=self.device)
        self.relative_entropy: torch.Tensor = torch.tensor(0.0, device=self.device)
        self.energy: torch.Tensor = torch.tensor(0.0, device=self.device)
        
        # Optimization parameters
        self.lam: torch.Tensor = torch.tensor(0.0, device=self.device)
        self.lam2: torch.Tensor = torch.tensor(0.0, device=self.device)

        self.iteration_trajectories = []  # Store trajectories for each iteration

        self.propagator_ref = None
        self.propagator = None

    def _get_ham(self, k: int) -> torch.Tensor:
        """Get Hamiltonian at time step k"""
        ham = torch.tensor(self.ham_list[0][0] + self.ham_list[0][1], device=self.device)
        for j, hams in enumerate(self.ham_list[1:]):
            ham = ham + self.ctrls_real[k, j]*(torch.tensor(hams[0] + hams[1], device=self.device))
            ham = ham + 1j*self.ctrls_im[k, j]*(torch.tensor(hams[0] - hams[1], device=self.device))
        return ham

    def _get_ham_no_controls(self) -> torch.Tensor:
        """Get base Hamiltonian without controls"""
        return torch.tensor(self.ham_list[0][0] + self.ham_list[0][1], device=self.device)

    def _compute_propagator_ref(self):
        t_steps = len(self.time_list)
        dt = self.dt

        c_ops_tensors = [torch.tensor(c.full(), dtype=torch.complex128, device=self.device) for c in self.c_ops]
        c_ops_dag_c = [torch.tensor((c.dag() * c).full(), dtype=torch.complex128, device=self.device) for c in self.c_ops]
        sum_c_dag_c = sum(c_ops_dag_c)

        propagators = torch.zeros(t_steps, 2, 2, dtype=torch.complex128, device=self.device)
        for i in range(t_steps-1):
            # Get Hamiltonian without controls
            H = self._get_ham_no_controls()
            H_eff = H - 0.5j * sum_c_dag_c
            propagators[i] = torch.matrix_exp(-1j * dt * H_eff)
        
        self.propagator_ref = propagators

    def _compute_propagator(self):
        t_steps = len(self.time_list)
        dt = self.dt

        c_ops_tensors = [torch.tensor(c.full(), dtype=torch.complex128, device=self.device) for c in self.c_ops]
        c_ops_dag_c = [torch.tensor((c.dag() * c).full(), dtype=torch.complex128, device=self.device) for c in self.c_ops]
        sum_c_dag_c = sum(c_ops_dag_c)

        propagators = torch.zeros(t_steps, 2, 2, dtype=torch.complex128, device=self.device)
        for i in range(t_steps-1):
            # Get current Hamiltonian
            H = self._get_ham(i)
            H_eff = H - 0.5j * sum_c_dag_c
            propagators[i] = torch.matrix_exp(-1j * dt * H_eff)
        
        self.propagator = propagators

    def _density_to_vector(self, density_matrix: Qobj) -> Qobj:
        """Convert density matrix to state vector
        
        For a diagonal density matrix with probabilities p0, p1,
        returns state vector [√p0, √p1]
        """
        # Extract diagonal elements (probabilities)
        diag = np.diagonal(density_matrix.full().real)
        draw = np.random.choice(self.dim, 1, p=diag)
        purestart = np.zeros(self.dim)
        purestart[draw[0]] = 1
        # Convert probabilities to amplitudes by taking sqrt
        return purestart

    def _sample_trajectory(self, ref: bool) -> TrajectoryResult:
        """Sample a single quantum trajectory"""
        start_time = time.time()
        
        # Pre-compute tensors
        dt = self.dt
        c_ops_tensors = [torch.tensor(c.full(), dtype=torch.complex128, device=self.device) for c in self.c_ops]
        c_ops_dag_c = [torch.tensor((c.dag() * c).full(), dtype=torch.complex128, device=self.device) for c in self.c_ops]
        kraus_operators = [c * np.sqrt(dt) for c in c_ops_tensors]
        
        # Initialize state
        current_state = torch.tensor(
            self._density_to_vector(self.initial_density).full(), 
            dtype=torch.complex128, device=self.device
        ).view(-1)
        
        # Setup tracking variables
        t_steps = len(self.time_list)
        events = torch.zeros(t_steps-1, dtype=torch.long, device=self.device)
        energy = torch.tensor(0.0, dtype=torch.float64, device=self.device)
        prob_trajectory = torch.tensor(1.0, dtype=torch.float64, device=self.device)
        
        # Pre-compute common matrices
        eye = torch.eye(2, dtype=torch.complex128, device=self.device)
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
            probs = torch.cat([torch.tensor([no_jump_prob], device=self.device), probs])
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

        end_time = time.time()
        print(f"_sample_trajectory took {end_time - start_time:.4f} seconds")

        return TrajectoryResult(
            final_state=current_state,  # Maintains gradient
            probability=prob_trajectory,  # Maintains gradient
            energy=energy,
            events=events.tolist()
        )
    
    def _sample_trajectory_new(self, ref: bool = False) -> TrajectoryResult:
        """Implement quantum jump algorithm for trajectory sampling"""
        
        dt = self.dt
        t_steps = len(self.time_list)
        
        # Initialize operators and state
        c_ops_tensors = [torch.tensor(c.full(), dtype=torch.complex128, device=self.device) for c in self.c_ops]
        c_ops_dag_c = [torch.tensor((c.dag() * c).full(), dtype=torch.complex128, device=self.device) for c in self.c_ops]
        sum_c_dag_c = sum(c_ops_dag_c)

        current_state = torch.tensor(
            self._density_to_vector(self.initial_density), 
            dtype=torch.complex128, device=self.device
        ).view(-1)

        # Add initial state position to events list
        initial_pos = torch.where(current_state.abs() > 0.9)[0].item()
        #print(current_state.abs())
        #initial_pos = torch.argmax(current_state.abs()).item()
        events = [] 

        # Setup tracking variables
        energy = torch.tensor(0.0, dtype=torch.float64, device=self.device)
        prob_trajectory = torch.tensor(1.0, dtype=torch.float64, device=self.device)
    
        
        # Generate initial random threshold
        with torch.no_grad():
            r = torch.rand(1, device=self.device).item()
        
        # Add trajectory storage
        state_trajectory = [current_state.clone()]
        time_points = self.time_list
        jump_times = []
        jump_types = []

        for i in range(t_steps-1):
            # Get current Hamiltonian
            H = self._get_ham_no_controls() if ref else self._get_ham(i)
            #H_eff = H - 0.5j * sum_c_dag_c
            propagator = self.propagator_ref[i] if ref else self.propagator[i]
            old_current_state = current_state
            current_state_candidate = propagator @ current_state
            norm_squared = (current_state_candidate.conj() @ current_state_candidate).real.item()

            if norm_squared <= r or norm_squared < 1e-9:
                no_jump_prob = (old_current_state.conj() @ old_current_state).real.item()
                prob_trajectory = prob_trajectory * no_jump_prob
                
                jump_probs = torch.stack([
                    (current_state.conj() @ c_ops_dag_c[i] @ current_state).real
                    for i in range(len(c_ops_dag_c))
                ])
                jump_probs = jump_probs / jump_probs.sum()
                
                with torch.no_grad():
                    jump_idx = torch.multinomial(jump_probs, 1)[0]
                
                prob_trajectory = prob_trajectory * jump_probs[jump_idx]
                jump_times.append(time_points[i].item())
                jump_types.append(jump_idx.item())
                
                current_state = c_ops_tensors[jump_idx] @ old_current_state
                norm_squared_after_jump = (current_state.conj() @ current_state).real.item()
                events.append(jump_idx.item() + 1)
                current_state = current_state / torch.sqrt(torch.tensor(norm_squared_after_jump, device=self.device))
                
                with torch.no_grad():
                    r = torch.rand(1, device=self.device).item()
            else:
                current_state = current_state_candidate
                events.append(0)

            state_trajectory.append(current_state.clone())
            energy = energy + (current_state.conj() @ H @ current_state).real
            
        prob_trajectory = prob_trajectory * (current_state.conj() @ current_state).real
        

        return TrajectoryResult(
            final_state=current_state,
            probability=prob_trajectory,
            energy=energy,
            events=(initial_pos, events),
            state_trajectory=state_trajectory,
            time_points=time_points,
            jump_times=jump_times,
            jump_types=jump_types
        )

    def _sample_trajectory_new_gpu(self, ref: bool = False) -> TrajectoryResult:
        """Optimized quantum jump algorithm for trajectory sampling."""
        start_time = time.time()
        
        dt = self.dt
        t_steps = len(self.time_list)
        
        # Convert collapse operators to GPU tensors once
        c_ops_tensors = torch.stack([torch.tensor(c.full(), dtype=torch.complex128, device=self.device) for c in self.c_ops])
        c_ops_dag_c = torch.stack([torch.tensor((c.dag() * c).full(), dtype=torch.complex128, device=self.device) for c in self.c_ops])
        
        # Initialize state as a vector on GPU
        current_state = torch.tensor(
            self._density_to_vector(self.initial_density), 
            dtype=torch.complex128, device=self.device
        ).view(-1)
        
        # Identify initial state position without using .item()
        initial_pos = torch.argmax(current_state.abs())  # Avoids CPU sync
        
        # Precompute -0.5j * sum_c_dag_c for efficiency
        sum_c_dag_c = torch.sum(c_ops_dag_c, dim=0)
        H_eff_constant = -0.5j * sum_c_dag_c  # Precomputed outside loop
        
        # Initialize trajectory storage
        state_trajectory = torch.empty((t_steps, current_state.size(0)), dtype=torch.complex128, device=self.device)
        state_trajectory[0] = current_state
        
        events = []
        jump_times = []
        jump_types = []
        
        # Probability tracking
        prob_trajectory = torch.tensor(1.0, dtype=torch.float64, device=self.device)
        energy = torch.tensor(0.0, dtype=torch.float64, device=self.device)
        
        # Generate initial random threshold
        r = torch.rand(1, device=self.device)  # Keep as GPU tensor
        
        # Iterate over time steps
        for i in range(t_steps - 1):
            # Get Hamiltonian
            H = self._get_ham_no_controls() if ref else self._get_ham(i)
            H_eff = H + H_eff_constant  # Avoid recomputation of sum
            
            # Compute propagator efficiently
            propagator = torch.linalg.matrix_exp(-1j * dt * H_eff)
            
            # Apply propagator
            old_current_state = current_state
            current_state_candidate = propagator @ current_state
            
            # Compute squared norm without CPU sync
            norm_squared = (current_state_candidate.conj() @ current_state_candidate).real
            
            if norm_squared <= r or norm_squared < 1e-9:
                # Probability of no jump
                no_jump_prob = (old_current_state.conj() @ old_current_state).real
                prob_trajectory *= no_jump_prob
                
                # Compute jump probabilities in a single batched operation
                jump_probs = (current_state.conj().unsqueeze(0) @ c_ops_dag_c @ current_state).real
                jump_probs = jump_probs.squeeze()
                jump_probs /= jump_probs.sum()
                
                # Cumulative probability sampling (avoids multinomial overhead)
                cumsum_probs = torch.cumsum(jump_probs, dim=0)
                jump_idx = torch.searchsorted(cumsum_probs, torch.rand(1, device=self.device)).item()
                
                # Update probability
                prob_trajectory *= jump_probs[jump_idx]
                
                # Record jump event
                jump_times.append(self.time_list[i])
                jump_types.append(jump_idx)
                events.append(jump_idx + 1)
                
                # Apply jump
                current_state = c_ops_tensors[jump_idx] @ old_current_state
                norm_squared_after_jump = (current_state.conj() @ current_state).real
                current_state /= torch.sqrt(norm_squared_after_jump)
                
                # Generate new random threshold
                r = torch.rand(1, device=self.device)
            else:
                current_state = current_state_candidate
                events.append(0)
            
            # Store trajectory
            state_trajectory[i + 1] = current_state
            
            # Update energy
            energy += (current_state.conj() @ H @ current_state).real
        
        # Final probability update
        prob_trajectory *= (current_state.conj() @ current_state).real
        
        end_time = time.time()
        #print(f"Optimized _sample_trajectory_new took {end_time - start_time:.4f} seconds")
        
        return TrajectoryResult(
            final_state=current_state,
            probability=prob_trajectory,
            energy=energy,
            events=(initial_pos.item(), events),
            state_trajectory=state_trajectory,
            time_points=self.time_list,
            jump_times=jump_times,
            jump_types=jump_types
        )

    def _get_trajectory_probability(self, trajectory_data: Tuple[int, List[int]], ref: bool = False) -> torch.Tensor:
        """Compute probability of a trajectory
        
        Args:
            trajectory_data: Tuple of (initial_position, events_list)
            ref: Whether to use reference Hamiltonian
        """
        
        init_pos, events = trajectory_data
        dt = self.dt
        
        # Initialize operators
        c_ops_tensors = [torch.tensor(c.full(), dtype=torch.complex128, device=self.device) for c in self.c_ops]
        c_ops_dag_c = [torch.tensor((c.dag() * c).full(), dtype=torch.complex128, device=self.device) for c in self.c_ops]
        sum_c_dag_c = sum(c_ops_dag_c)
        
        # Set up initial state based on initial position
        current_state = torch.zeros(self.dim, dtype=torch.complex128, device=self.device)
        current_state[init_pos] = 1.0
        
        prob = torch.tensor(1.0, dtype=torch.float64, device=self.device)
        
        # Follow trajectory events
        for i, event in enumerate(events):
            # Get Hamiltonian and effective non-Hermitian part
            H = self._get_ham_no_controls() if ref else self._get_ham(i)
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
                current_state = torch.tensor(self.c_ops[event - 1].full(), dtype=torch.complex128, device=self.device) @ old_current_state
                current_state = current_state / torch.norm(current_state)
                
        final_prob = prob * (current_state.conj() @ current_state).real

        
        return final_prob

    def _parallel_trajectory_sample_profile(self, n_samples: int, n_streams: int = 5) -> List[TrajectoryResult]:
        """Runs parallel sampling on the GPU using CUDA streams."""
        start_time = time.time()
       
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True,  # Capture tensor shapes
            profile_memory=True, # Track memory usage
            with_stack=True      # Capture call stack
        ) as prof:
            streams = [torch.cuda.Stream() for _ in range(n_streams)]  # Create CUDA streams
            results = [None] * n_samples

            # Launch parallel sampling
            for i in range(n_samples):
                #stream = streams[i % n_streams]  # Assign a stream
                with torch.cuda.stream(stream):  # Run in parallel
                    results[i] = self._sample_trajectory_new_gpu()  # Non-blocking operation
            
            # Wait for all streams to complete
            torch.cuda.synchronize()

        end_time = time.time()
        print(f"_parallel_trajectory_sample took {end_time - start_time:.4f} seconds")

        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
        
        return results # Move results to CPU
    
    def _parallel_trajectory_sample(self, n_samples: int, n_streams: int = 5) -> List[TrajectoryResult]:
        """Runs parallel sampling on the GPU using CUDA streams."""
        start_time = time.time()
       
        #streams = [torch.cuda.Stream() for _ in range(n_streams)]  # Create CUDA streams
        results = [None] * n_samples

        # Launch parallel sampling
        for i in range(n_samples):
            #stream = streams[i % n_streams]  # Assign a stream
            #with torch.cuda.stream(stream):  # Run in parallel
            results[i] = self._sample_trajectory_new()  # Non-blocking operation
            
        # Wait for all streams to complete
        #torch.cuda.synchronize()

        end_time = time.time()
        print(f"_parallel_trajectory_sample took {end_time - start_time:.4f} seconds")
        return results  # Move results to CPU

    def _vectorized_trajectory_sample(self, ref: bool, batch_size: int) -> List[TrajectoryResult]:
        """Sample multiple trajectories in parallel using vectorization"""
        start_time = time.time()
        
        dt = self.dt
        t_steps = len(self.time_list)
        
        # Initialize operators with batch dimension
        c_ops_tensors = [torch.tensor(c.full(), dtype=torch.complex128, device=self.device).expand(batch_size, -1, -1) 
                        for c in self.c_ops]
        c_ops_dag_c = [torch.tensor((c.dag() * c).full(), dtype=torch.complex128, device=self.device).expand(batch_size, -1, -1) 
                      for c in self.c_ops]
        
        # Initialize batch of states
        initial_states = []
        for _ in range(batch_size):
            state = torch.tensor(self._density_to_vector(self.initial_density), dtype=torch.complex128, device=self.device)
            initial_states.append(state)
        current_states = torch.stack(initial_states)  # [batch_size, dim]
        
        # Setup tracking variables
        energies = torch.zeros(batch_size, dtype=torch.float64, device=self.device)
        prob_trajectories = torch.ones(batch_size, dtype=torch.float64, device=self.device)
        all_events = [[] for _ in range(batch_size)]
        state_trajectories = [[] for _ in range(batch_size)]
        jump_times = [[] for _ in range(batch_size)]
        jump_types = [[] for _ in range(batch_size)]
        
        # Record initial positions
        #initial_positions = torch.where(current_states.abs() > 0.9)[1]
        initial_positions = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        # Evolution loop
        for i in range(t_steps-1):
            # Get Hamiltonian (batched)
            H = self._get_ham_no_controls() if ref else self._get_ham(i)
            H = H.expand(batch_size, -1, -1)
            H_eff = H - 0.5j * sum(c_ops_dag_c)
            
            # Vectorized evolution
            propagator = torch.matrix_exp(-1j * dt * H_eff)
            old_states = current_states
            current_states = torch.bmm(propagator, current_states.unsqueeze(-1)).squeeze(-1)
            
            # Calculate jump probabilities
            norm_squared = (current_states.conj() * current_states).sum(dim=1).real
            
            # Generate random numbers for all trajectories
            with torch.no_grad():
                r = torch.rand(batch_size, device=self.device)
            
            # Find which trajectories need jumps
            jump_mask = (norm_squared <= r) | (norm_squared < 1e-9)
            
            if jump_mask.any():
                # Process jumps for trajectories that need them
                jump_indices = torch.where(jump_mask)[0]
                
                for idx in jump_indices:
                    # Calculate jump probabilities for this trajectory
                    jump_probs = torch.stack([
                        (old_states[idx].conj() @ c_ops_dag_c[j][idx] @ old_states[idx]).real
                        for j in range(len(c_ops_dag_c))
                    ])
                    jump_probs = jump_probs / jump_probs.sum()
                    
                    with torch.no_grad():
                        jump_type = torch.multinomial(jump_probs, 1)[0]
                    
                    # Update trajectory information
                    prob_trajectories[idx] *= jump_probs[jump_type]
                    all_events[idx].append(jump_type.item() + 1)
                    jump_times[idx].append(self.time_list[i].item())
                    jump_types[idx].append(jump_type.item())
                    
                    # Apply jump
                    current_states[idx] = c_ops_tensors[jump_type][idx] @ old_states[idx]
                    current_states[idx] /= torch.sqrt(torch.tensor(
                        (current_states[idx].conj() @ current_states[idx]).real, device=self.device
                    ))
            
            # No-jump evolution for other trajectories
            no_jump_mask = ~jump_mask
            if no_jump_mask.any():
                for idx in torch.where(no_jump_mask)[0]:
                    all_events[idx].append(0)
            
            # Store states and update energies
            for b in range(batch_size):
                state_trajectories[b].append(current_states[b].clone())
                energies[b] += (current_states[b].conj() @ H[b] @ current_states[b]).real
        
        # Create TrajectoryResult objects
        results = []
        for b in range(batch_size):
            results.append(TrajectoryResult(
                final_state=current_states[b],
                probability=prob_trajectories[b],
                energy=energies[b],
                events=(initial_positions[b].item(), all_events[b]),
                state_trajectory=state_trajectories[b],
                time_points=self.time_list,
                jump_times=jump_times[b],
                jump_types=jump_types[b]
            ))
        
        end_time = time.time()
        print(f"_vectorized_trajectory_sample took {end_time - start_time:.4f} seconds")
        
        return results

    def _probability_distribution_estimate(self, ref: bool, max_iters: int) -> Tuple[List[torch.Tensor], Dict, torch.Tensor, torch.Tensor]:
        """Estimate probability distribution from quantum trajectories"""
        num_samples = 0
        energy_sum = torch.zeros(1, dtype=torch.float64, device=self.device)
        fidelity_sum = torch.zeros(1, dtype=torch.float64, device=self.device)
        probability_distribution = defaultdict(lambda: torch.zeros(1, dtype=torch.float64, device=self.device))
        average_final_state = torch.zeros(self.dim, self.dim, dtype=torch.complex128, device=self.device)
        done = set()
        all_trajectories = []
        
        # Add timing measurements
        trajectory_times = []

        # Sample trajectories
        batch_size = 100  # Adjust based on memory constraints
        while num_samples < max_iters:
            current_batch = min(batch_size, max_iters - num_samples)
            results = self._parallel_trajectory_sample(batch_size)
            
            for result in results:
                num_samples += 1
                # Time the trajectory sampling
                start_time = time.time()
                end_time = time.time()
                trajectory_times.append(end_time - start_time)
                
                # Update statistics without in-place operations
                final_state_not_normalized = torch.outer(result.final_state, result.final_state.conj())
                final_state = final_state_not_normalized/torch.trace(final_state_not_normalized)
                average_final_state += final_state
                energy_sum = energy_sum + result.energy

                # Record unique trajectories with events that already include initial position
                init_pos, events = result.events  # Unpack the tuple
                events_tuple = (init_pos, tuple(events))  # Store as (init_pos, events)
                if events_tuple not in done:
                    probability_distribution[events_tuple] = result.probability
                    done.add(events_tuple)

                populations = []
                for state in result.state_trajectory:
                    norm = (state.conj() @ state).real.item()
                    state = state / torch.sqrt(torch.tensor(norm))
                    # Detach the tensor before converting to numpy
                    pop = (state[1]*state[1].conj()).real.detach().item()
                    populations.append(pop)
                all_trajectories.append(populations)

        # Print timing statistics
        avg_time = mean(trajectory_times)
        max_time = max(trajectory_times)
        min_time = min(trajectory_times)
        # print(f"\nTrajectory timing statistics:")
        # print(f"Average time per trajectory: {avg_time:.4f} seconds")
        # print(f"Min time: {min_time:.4f} seconds")
        # print(f"Max time: {max_time:.4f} seconds")
        # print(f"Total samples: {num_samples}")

        # Compute averages without in-place operations
        average_final_state = average_final_state/max_iters
        
        # Debug gradient information
        print(f"average_final_state requires_grad: {average_final_state.requires_grad}")
        print(f"target requires_grad: {self.target.requires_grad}")
        
        # Ensure proper reshaping and gradient propagation
        target_reshaped = self.target.view(self.dim, self.dim).to(dtype=torch.complex128)
        fidelity_avg = calculate_frob_norm(average_final_state, 
                                           target_reshaped
        )

        print(f"Fidelity calc - input grads: {average_final_state.requires_grad}, {target_reshaped.requires_grad}")
        
        energy_avg = energy_sum / num_samples

        # Normalize probability distribution
        # total_prob = sum(probability_distribution.values())
        # if (total_prob > 0):
        #     probability_distribution = {
        #         k: v/total_prob 
        #         for k, v in probability_distribution.items()
        #     }
        
        # Calculate average trajectory and plot
        avg_trajectory = np.mean(all_trajectories, axis=0)
        std_trajectory = np.std(all_trajectories, axis=0)
        
        # Store the trajectory data if this is not a reference calculation
        if not ref:
            self.iteration_trajectories.append({
                'avg': avg_trajectory,
                'std': std_trajectory,
                'time': self.time_list
            })
        
        return [], probability_distribution, energy_avg, fidelity_avg

    def _get_kl_divergence(self, p1: Dict, p2: Dict, epsilon: float = 0.000005) -> torch.Tensor:
        """Calculate Kullback-Leibler divergence between two probability distributions"""
        all_keys = set(p1.keys()).union(set(p2.keys()))
        overlapping_keys = set(p1.keys()).intersection(set(p2.keys()))
        print(f"Overlapping keys: {len(overlapping_keys)}")
        print(f"Total keys: {len(all_keys)}")
        print(f"fraction of overlapping keys: {len(overlapping_keys)/len(all_keys)}")
        
        # Convert to tensors while maintaining gradients
        def to_tensor(val):
            if isinstance(val, torch.Tensor):
                return val
            return torch.tensor(val, dtype=torch.float64, requires_grad=True)
        
        p1 = {k: to_tensor(p1.get(k, epsilon)) for k in all_keys}
        p2 = {k: to_tensor(p2.get(k, epsilon)) for k in all_keys}

        # Normalize distributions
        p1_sum = sum(p1.values())
        p2_sum = sum(p2.values())
        
        p1 = {k: v/p1_sum for k, v in p1.items()}
        p2 = {k: v/p2_sum for k, v in p2.items()}

        # Compute KL divergence term by term
        kl_terms = {}
        for k in p1:
            term = p1[k] * torch.log(p1[k] / p2[k])
            kl_terms[k] = term

        # Get 5 largest contributions
        sorted_terms = sorted(kl_terms.items(), key=lambda x: abs(x[1].item()), reverse=True)
        print("\nTop 5 KL divergence contributions:")
        for k, v in sorted_terms[:10]:
            print(f"Trajectory {k}:")
            print(f"  p1: {p1[k].item():.7f}")
            print(f"  p2: {p2[k].item():.7f}")
            print(f"  contribution: {v.item():.7f}")

        kl_div = sum(kl_terms.values())
        return kl_div
    
    def _get_kl_divergence_with_penalty(self, p1: Dict, p2: Dict) -> torch.Tensor:
        """Calculate Kullback-Leibler divergence between two probability distributions"""
        start_time = time.time()
        missing_keys = set(p1.keys()) - set(p2.keys())
        
        # Calculate probabilities for missing trajectories and track NaN values
        nan_count = 0
        for traj in missing_keys:
            prob = self._get_trajectory_probability(traj, ref=True)  # Fixed: removed self as argument
            if torch.isnan(prob):
                nan_count += 1
                p2[traj] = prob  # Add NaN keys to p2
                continue
            p2[traj] = prob

        # If I have NaN values, I want to add them to the dictionary just for expanding it.
        # Make deep copies of input dictionaries
        p1 = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in p1.items()}
        p2 = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in p2.items()}
        
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
        
        # Normalize non-NaN parts
        p1_filtered = {k: v/p1_sum for k, v in p1_filtered.items()}
        p2_filtered = {k: v/p2_sum for k, v in p2_filtered.items()}
        
        # add the NaN keys to p2 again
        p2.update(p2_Nan)  # Use update instead of addition

        # Compute KL divergence term by term
        kl_terms = {}
        kl_penalty = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
        
        # Add terms for non-NaN keys
        for k in p1_filtered:
            term = p1_filtered[k] * torch.log(p1_filtered[k] / p2_filtered[k])
            kl_terms[k] = term
        
        # Add penalty terms for NaN keys
        for k in p1_Nan:
            kl_terms[k] = kl_penalty * p1_Nan[k]
        
        kl_div = sum(kl_terms.values())

        end_time = time.time()
        print(f"_get_trajectory_probabilities took {end_time - start_time:.4f} seconds")

        return kl_div

    def _get_err(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Calculate error metrics for current controls"""
        print("\nEntering _get_err")
        # Update to match actual return values
        evolution, p1, energy_average, fidelity_avg = self._probability_distribution_estimate(
            ref=False, max_iters=150
        )
        print(f"After estimate - fidelity requires_grad: {fidelity_avg.requires_grad}")
        
        self.control_evo = evolution
        self.PathDistEvo = p1

        # Calculate reference evolution if needed
        if self.ref_evo is None:
            ref_evolution, p2, energy_ref_avg, _ = self._probability_distribution_estimate(
                ref=True, max_iters=128
            )
            self.ref_evo = ref_evolution
            self.ref_prob_dist = p2
            self.energy_ref_avg = energy_ref_avg

        relative_entropy = self._get_kl_divergence_with_penalty(p1, self.ref_prob_dist)
        return relative_entropy, energy_average, fidelity_avg

    def _loss_function(self, args: List) -> torch.Tensor:
        """Calculate total loss for optimization"""
        # Don't detach parameters, keep original gradient chain
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

        # Evaluate basis without forcing requires_grad
        self.ctrls_real = evaluate_basis(params_real, t_points, fourier_config)
        self.ctrls_im = evaluate_basis(params_im, t_points, fourier_config)

        self._compute_propagator()

        # Calculate metrics
        relative_entropy, energy_average, fidelity_avg = self._get_err()
        
        # Store metrics
        self.fidelity = fidelity_avg
        self.relative_entropy = relative_entropy
        self.energy = energy_average

        # Simple loss
        loss = relative_entropy
        
        return loss

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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
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
    evolution.target = torch.from_numpy(rhotar.full()).to(device)
    evolution.init_ctrls = init_ctrls

    return evolution