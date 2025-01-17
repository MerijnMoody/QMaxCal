import torch
import time

def parallel_trajectory_sample(hamiltonian, initial_state, time_list, n_samples, n_streams = 5):
    """Runs parallel sampling on the GPU using CUDA streams."""
    start_time = time.time()
   
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,  # Capture tensor shapes
        profile_memory=True, # Track memory usage
        with_stack=True      # Capture call stack
    ) as prof:
        streams = [torch.cuda.Stream() for _ in range(n_streams)]  # Create CUDA streams
        #results = [None] * n_samples

        # Launch parallel sampling
        for i in range(int(n_samples/n_streams)):
            print(i)
            #stream = streams[i % n_streams]  # Assign a stream
            #with torch.cuda.stream(stream):  # Run in parallel
            #    results[i] = compute_time_evolution(hamiltonian, initial_state, time_list, n_parallel=1)  # Non-blocking operation
            results = compute_time_evolution(hamiltonian, initial_state, time_list, n_parallel=n_streams)
        
        # Wait for all streams to complete
        torch.cuda.synchronize()

    end_time = time.time()
    print(f"parallel_trajectory_sample took {end_time - start_time:.4f} seconds")

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    
    return torch.stack(results).cpu()  # Move results to CPU

def compute_time_evolution(hamiltonian: torch.Tensor, initial_state: torch.Tensor, time_steps: torch.Tensor, n_parallel: int) -> torch.Tensor:
    """Compute the time evolution of a matrix under a given Hamiltonian with batch processing."""
    dt = time_steps[1] - time_steps[0]
    hamiltonian_batch = hamiltonian.unsqueeze(0).repeat(n_parallel, 1, 1)
    state_batch = initial_state.unsqueeze(0).repeat(n_parallel, 1, 1)
    print(state_batch.device)
    print(hamiltonian_batch.device)
    for t in time_steps:
        propagator_batch = torch.matrix_exp(-1j * dt * hamiltonian_batch)
        state_batch = propagator_batch @ state_batch @ propagator_batch.conj().transpose(-2, -1)
    return state_batch

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hamiltonian = torch.tensor([[0, 0], [1, 0]], dtype=torch.complex128, device=device)
    initial_state = torch.tensor([[1, 0], [0, 0]], dtype=torch.complex128, device=device)
    time_steps = torch.linspace(0, 10, 10, dtype=torch.float64)
    
    final_state = compute_time_evolution(hamiltonian, initial_state, time_steps, n_parallel=1)
    parallel_trajectory_sample(hamiltonian, initial_state, time_steps, 100, 2)
    print("Final state:\n", final_state)

if __name__ == "__main__":
    main()
