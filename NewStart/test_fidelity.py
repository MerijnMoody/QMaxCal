
import torch
from Utils import fidelity
import numpy as np
from qutip import Qobj

def test_pure_states():
    """Test fidelity between pure states"""
    # Same states
    state1 = torch.tensor([[1.0, 0], [0, 0]], dtype=torch.complex128)
    print("\nTest 1 - Same pure states:")
    print(f"Fidelity between |0⟩ and |0⟩: {fidelity(state1, state1)}")  # Should be 1.0
    
    # Orthogonal states
    state2 = torch.tensor([[0.0, 0], [0, 1.0]], dtype=torch.complex128)
    print("\nTest 2 - Orthogonal pure states:")
    print(f"Fidelity between |0⟩ and |1⟩: {fidelity(state1, state2)}")  # Should be 0.0
    
    # Superposition
    state3 = torch.tensor([[0.5, 0], [0, 0.5]], dtype=torch.complex128)
    print("\nTest 3 - Pure state and superposition:")
    print(f"Fidelity between |0⟩ and (|0⟩+|1⟩)/√2: {fidelity(state1, state3)}")  # Should be 0.5

def test_mixed_states():
    """Test fidelity between mixed states"""
    # Completely mixed state
    mixed = torch.tensor([[0.5, 0], [0, 0.5]], dtype=torch.complex128)
    pure = torch.tensor([[1.0, 0], [0, 0]], dtype=torch.complex128)
    
    print("\nTest 4 - Pure and maximally mixed states:")
    print(f"Fidelity between |0⟩ and I/2: {fidelity(pure, mixed)}")  # Should be 0.707... (√0.5)
    
    # Different mixed states
    mixed2 = torch.tensor([[0.7, 0], [0, 0.3]], dtype=torch.complex128)
    print("\nTest 5 - Different mixed states:")
    print(f"Fidelity between mixed states: {fidelity(mixed, mixed2)}")

def sanity():
    """Test fidelity between mixed states"""
    # Completely mixed state
    mixed = torch.tensor([[0.5, 0], [0, 0.5]], dtype=torch.complex128)
    pure = torch.tensor([[0.1, 0], [0, 0.9]], dtype=torch.complex128)
    
    print("\nTest 4 - Pure and maximally mixed states:")
    print(f"Fidelity between |0⟩ and I/2: {fidelity(pure, mixed)}")  # Should be 0.707... (√0.5)
    
    # Different mixed states
    mixed2 = torch.tensor([[0.7, 0], [0, 0.3]], dtype=torch.complex128)
    print("\nTest 5 - Different mixed states:")
    print(f"Fidelity between mixed states: {fidelity(mixed, mixed2)}")

def test_edge_cases():
    """Test edge cases"""
    # Zero state
    zero = torch.zeros((2,2), dtype=torch.complex128)
    pure = torch.tensor([[1.0, 0], [0, 0]], dtype=torch.complex128)
    
    print("\nTest 6 - Edge cases:")
    try:
        print(f"Fidelity with zero state: {fidelity(pure, zero)}")
    except Exception as e:
        print(f"Error with zero state: {e}")

def main():
    print("Running fidelity tests...")
    test_pure_states()
    test_mixed_states()
    test_edge_cases()
    sanity()

if __name__ == "__main__":
    main()