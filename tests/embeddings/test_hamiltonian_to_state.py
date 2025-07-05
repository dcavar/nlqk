#!/usr/bin/env python3
# coding: utf-8


"""
test_hamiltonian_to_state.py

(C) 2025 by [Damir Cavar](http://damir.cavar.me/), James Bryan Graves, and [NLP Lab](https://nlp-lab.org/)

Testing the NLQK Hamiltonian to state conversion functionality.
"""


import sys
sys.path.append('.')
# import os
import unittest
try: # prefer RAPIDS libraries and GPU over numpy and CPU
    import cupy as np  # Try to import cupy and alias it as np
    from cupyx.scipy.linalg import expm
    _USE_GPU = True
except ModuleNotFoundError:
    import numpy as np  # If cupy not found, import numpy and alias it as np
    from scipy.linalg import expm
    _USE_GPU = False
# import GPUtil  # If you're using GPUtil
from nlqk.embeddings import hamiltonian_to_state, check_states_equal


class TestHamiltonianToState(unittest.TestCase):
    """Testing the NLQK Hamiltonian to state conversion functionality."""

    def test_hamiltonian_to_state_identity(self):
        """Test that zero Hamiltonian gives |0⟩ state"""
        # 1-qubit case: H = 0 matrix should give |0⟩ = [1, 0]
        H = np.zeros((2, 2), dtype=complex)
        result = hamiltonian_to_state(H)
        expected = np.array([1.0, 0.0], dtype=complex)
        np.testing.assert_array_almost_equal(result, expected, decimal=7)

    def test_hamiltonian_to_state_pauli_x(self):
        """Test Hamiltonian corresponding to Pauli-X rotation"""
        # H = π/2 * σ_x gives U = exp(i*π/2*σ_x) = cos(π/2)*I + i*sin(π/2)*σ_x = i*σ_x
        # U|0⟩ = i*σ_x|0⟩ = i*|1⟩ = [0, i]
        H = (np.pi / 2) * np.array([[0, 1], [1, 0]], dtype=complex)
        result = hamiltonian_to_state(H)
        expected = np.array([0.0, 1j], dtype=complex)
        np.testing.assert_array_almost_equal(result, expected, decimal=6)

    def test_hamiltonian_to_state_pauli_y(self):
        """Test Hamiltonian corresponding to Pauli-Y rotation"""
        # H = π/2 * σ_y gives U = exp(i*π/2*σ_y) = cos(π/2)*I + i*sin(π/2)*σ_y = i*σ_y
        # U|0⟩ = i*σ_y|0⟩ = i*i*|1⟩ = -|1⟩ = [0, -1]
        H = (np.pi / 2) * np.array([[0, -1j], [1j, 0]], dtype=complex)
        result = hamiltonian_to_state(H)
        expected = np.array([0.0, -1.0], dtype=complex)
        np.testing.assert_array_almost_equal(result, expected, decimal=6)

    def test_hamiltonian_to_state_hadamard_like(self):
        """Test Hamiltonian that creates Hadamard-like superposition"""
        # For H = π/4 * σ_x, we get a rotation that creates superposition
        # This should give approximately equal amplitudes
        H = (np.pi / 4) * np.array([[0, 1], [1, 0]], dtype=complex)
        result = hamiltonian_to_state(H)
        
        # Check normalization
        self.assertAlmostEqual(np.linalg.norm(result), 1.0, places=7)
        
        # For π/4 rotation, both amplitudes should be non-zero and roughly equal in magnitude
        self.assertGreater(abs(result[0]), 0.3)
        self.assertGreater(abs(result[1]), 0.3)

    def test_hamiltonian_to_state_plus_state(self):
        """Test creating the |+⟩ state using correct Hamiltonian"""
        # H = π/2 * (σ_x + σ_z)/√2 creates Hadamard-like transformation
        sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
        sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
        H = (np.pi / (2 * np.sqrt(2))) * (sigma_x + sigma_z)
        
        result = hamiltonian_to_state(H)
        
        # Check normalization
        self.assertAlmostEqual(np.linalg.norm(result), 1.0, places=7)
        
        # Should create some superposition
        self.assertGreater(abs(result[0]), 0.1)
        self.assertGreater(abs(result[1]), 0.1)

    def test_hamiltonian_to_state_pauli_z(self):
        """Test Hamiltonian corresponding to Pauli-Z rotation"""
        # H = π * σ_z should give -|0⟩ = [-1, 0]
        H = np.pi * np.array([[1, 0], [0, -1]], dtype=complex)
        result = hamiltonian_to_state(H)
        expected = np.array([-1.0, 0.0], dtype=complex)
        np.testing.assert_array_almost_equal(result, expected, decimal=6)

    def test_hamiltonian_to_state_two_qubit_zero(self):
        """Test two-qubit zero Hamiltonian gives |00⟩ state"""
        H = np.zeros((4, 4), dtype=complex)
        result = hamiltonian_to_state(H)
        expected = np.array([1.0, 0.0, 0.0, 0.0], dtype=complex)
        np.testing.assert_array_almost_equal(result, expected, decimal=7)

    def test_hamiltonian_to_state_two_qubit_hadamard_like(self):
        """Test two-qubit Hamiltonian that creates superposition"""
        # Simple 2-qubit Hamiltonian that creates equal superposition
        H = (np.pi / 4) * np.array([
            [0, 1, 1, 0],
            [1, 0, 0, 1],
            [1, 0, 0, 1],
            [0, 1, 1, 0]
        ], dtype=complex)
        result = hamiltonian_to_state(H)
        
        # Check that the result is normalized
        self.assertAlmostEqual(np.linalg.norm(result), 1.0, places=7)
        
        # Check that it's not just |00⟩
        self.assertGreater(abs(result[1]), 0.1)

    def test_hamiltonian_to_state_hermitian_property(self):
        """Test that function works with Hermitian Hamiltonians"""
        # Create a random Hermitian matrix
        np.random.seed(42)
        A = np.random.randn(2, 2) + 1j * np.random.randn(2, 2)
        H = (A + A.conj().T) / 2  # Make it Hermitian
        
        result = hamiltonian_to_state(H)
        
        # Check normalization
        self.assertAlmostEqual(np.linalg.norm(result), 1.0, places=7)
        
        # Check that result is a valid 2-element state vector
        self.assertEqual(len(result), 2)

    def test_hamiltonian_to_state_three_qubit(self):
        """Test three-qubit zero Hamiltonian gives |000⟩ state"""
        H = np.zeros((8, 8), dtype=complex)
        result = hamiltonian_to_state(H)
        expected = np.zeros(8, dtype=complex)
        expected[0] = 1.0
        np.testing.assert_array_almost_equal(result, expected, decimal=7)

    def test_hamiltonian_to_state_normalization(self):
        """Test that output state is always normalized"""
        test_cases = [
            np.zeros((2, 2)),  # 1-qubit
            np.zeros((4, 4)),  # 2-qubit
            np.eye(2) * 0.5,   # Small non-zero Hamiltonian
            np.array([[1, 2], [2, 1]]) * 0.1  # Another small Hamiltonian
        ]
        
        for H in test_cases:
            with self.subTest(H_shape=H.shape):
                result = hamiltonian_to_state(H)
                norm = np.linalg.norm(result)
                self.assertAlmostEqual(norm, 1.0, places=7)

    def test_hamiltonian_to_state_unitary_evolution(self):
        """Test that the function implements unitary evolution correctly"""
        # Create a simple Hamiltonian
        H = np.array([[1, 0.5], [0.5, -1]], dtype=complex)
        result = hamiltonian_to_state(H)
        
        # Manually compute U = exp(iH) and apply to |0⟩
        U = expm(1j * H)
        zero_state = np.array([1.0, 0.0], dtype=complex)
        expected = U @ zero_state
        
        np.testing.assert_array_almost_equal(result, expected, decimal=7)

    def test_hamiltonian_to_state_complex_hamiltonian(self):
        """Test with complex Hamiltonian entries"""
        H = np.array([
            [0, 1-1j],
            [1+1j, 0]
        ], dtype=complex)
        
        result = hamiltonian_to_state(H)
        
        # Check normalization
        self.assertAlmostEqual(np.linalg.norm(result), 1.0, places=7)
        
        # Check that result has complex entries
        self.assertTrue(np.iscomplexobj(result))

    def test_hamiltonian_to_state_large_hamiltonian(self):
        """Test with larger Hamiltonian matrix"""
        # 4-qubit system (16x16 matrix)
        H = np.zeros((16, 16), dtype=complex)
        # Add some small random Hermitian perturbation
        np.random.seed(123)
        A = np.random.randn(16, 16) + 1j * np.random.randn(16, 16)
        H = (A + A.conj().T) / 2 * 0.01  # Small Hermitian matrix
        
        result = hamiltonian_to_state(H)
        
        # Check properties
        self.assertEqual(len(result), 16)
        self.assertAlmostEqual(np.linalg.norm(result), 1.0, places=6)
        self.assertAlmostEqual(abs(result[0]), 1.0, places=2)  # Should be close to |0000⟩

    def test_hamiltonian_to_state_power_of_two_dimensions(self):
        """Test that function works only with power-of-2 dimensions"""
        # Valid dimensions (powers of 2)
        valid_dims = [2, 4, 8, 16]
        for dim in valid_dims:
            with self.subTest(dim=dim):
                H = np.zeros((dim, dim), dtype=complex)
                result = hamiltonian_to_state(H)
                self.assertEqual(len(result), dim)
                self.assertAlmostEqual(np.linalg.norm(result), 1.0, places=7)

    def test_hamiltonian_to_state_small_rotation(self):
        """Test small rotation Hamiltonians"""
        # Small rotation around X axis
        theta = 0.1
        H = theta * np.array([[0, 1], [1, 0]], dtype=complex)
        result = hamiltonian_to_state(H)
        
        # For small θ, |ψ⟩ ≈ |0⟩ + iθ|1⟩
        expected_approx = np.array([1.0, 1j * theta], dtype=complex)
        expected_approx = expected_approx / np.linalg.norm(expected_approx)
        
        # Should be close for small theta
        self.assertAlmostEqual(abs(result[0]), abs(expected_approx[0]), places=2)

    def test_hamiltonian_to_state_return_type(self):
        """Test that function returns correct type"""
        H = np.zeros((2, 2), dtype=complex)
        result = hamiltonian_to_state(H)
        
        # Check return type
        self.assertIsInstance(result, np.ndarray)
        self.assertTrue(np.iscomplexobj(result))
        self.assertEqual(result.dtype, complex)

    def test_hamiltonian_to_state_zero_state_property(self):
        """Test that |0⟩ is always the first basis state"""
        for n_qubits in [1, 2, 3]:
            dim = 2 ** n_qubits
            H = np.zeros((dim, dim), dtype=complex)
            result = hamiltonian_to_state(H)
            
            # |0⟩ state should have amplitude 1 in first component, 0 elsewhere
            expected = np.zeros(dim, dtype=complex)
            expected[0] = 1.0
            
            np.testing.assert_array_almost_equal(result, expected, decimal=7)

    def test_hamiltonian_to_state_inverse_relationship(self):
        """Test relationship with check_states_equal function"""
        H = np.array([[0.1, 0.2], [0.2, -0.1]], dtype=complex)
        result1 = hamiltonian_to_state(H)
        result2 = hamiltonian_to_state(H)
        
        # Same Hamiltonian should give identical states
        self.assertTrue(check_states_equal(result1, result2, tol=1e-10))


if __name__ == '__main__':
    unittest.main()
