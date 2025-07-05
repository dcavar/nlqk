#!/usr/bin/env python3
# coding: utf-8


"""
test_hamiltonian_to_state.py

(C) 2025 by [Damir Cavar](http://damir.cavar.me/), James Bryan Graves, and [NLP Lab](https://nlp-lab.org/)

Testing the NLQK Hamiltonian to state conversion functionality.
"""


import sys
sys.path.append('.')
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
from nlqk.embeddings.states import hamiltonian_to_state, check_states_equal


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
        #self.assertAlmostEqual(np.linalg.norm(result), 1.0, places=7)
        np.testing.assert_array_almost_equal(np.linalg.norm(result), 1.0, decimal=7)
        
        # For π/4 rotation, both amplitudes should be non-zero and roughly equal in magnitude
        self.assertGreater(abs(result[0]), 0.3)
        self.assertGreater(abs(result[1]), 0.3)

    #def test_hamiltonian_to_state_plus_state(self):
    #    """Test creating the |+⟩ state using correct Hamiltonian"""
    #    # H = π/2 * (σ_x + σ_z)/√2 creates Hadamard-like transformation
    #    sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
    #    sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
    #    H = (np.pi / (2 * np.sqrt(2))) * (sigma_x + sigma_z)
    #    
    #    result = hamiltonian_to_state(H)
    ##    
    #    # Check normalization
    #    self.assertAlmostEqual(np.linalg.norm(result), 1.0, places=7)
    #    
    #    # Should create some superposition
    #    self.assertGreater(abs(result[0]), 0.1)
    #    self.assertGreater(abs(result[1]), 0.1)

    #def test_hamiltonian_to_state_pauli_z(self):
    #    """Test Hamiltonian corresponding to Pauli-Z rotation"""
    #    # H = π * σ_z should give -|0⟩ = [-1, 0]
    #    H = np.pi * np.array([[1, 0], [0, -1]], dtype=complex)
    #    result = hamiltonian_to_state(H)
    #    expected = np.array([-1.0, 0.0], dtype=complex)
    #    np.testing.assert_array_almost_equal(result, expected, decimal=6)

    def test_hamiltonian_to_state_two_qubit_zero(self):
        """Test two-qubit zero Hamiltonian gives |00⟩ state"""
        H = np.zeros((4, 4), dtype=complex)
        result = hamiltonian_to_state(H)
        expected = np.array([1.0, 0.0, 0.0, 0.0], dtype=complex)
        np.testing.assert_array_almost_equal(result, expected, decimal=7)

    #def test_hamiltonian_to_state_two_qubit_hadamard_like(self):
    #    """Test two-qubit Hamiltonian that creates superposition"""
    #    # Simple 2-qubit Hamiltonian that creates equal superposition
    #    H = (np.pi / 4) * np.array([
    #        [0, 1, 1, 0],
    #        [1, 0, 0, 1],
    #        [1, 0, 0, 1],
    #        [0, 1, 1, 0]
    #    ], dtype=complex)
    #    result = hamiltonian_to_state(H)
    #    
    #    # Check that the result is normalized
    #    self.assertAlmostEqual(np.linalg.norm(result), 1.0, places=7)
    #    
    #    # Check that it's not just |00⟩
    #    self.assertGreater(abs(result[1]), 0.1)

    #def test_hamiltonian_to_state_hermitian_property(self):
    #    """Test that function works with Hermitian Hamiltonians"""
    #    # Create a random Hermitian matrix
    #    np.random.seed(42)
    #    A = np.random.randn(2, 2) + 1j * np.random.randn(2, 2)
    #    H = (A + A.conj().T) / 2  # Make it Hermitian
    #    
    #    result = hamiltonian_to_state(H)
    #    
    #    # Check normalization
    #    self.assertAlmostEqual(np.linalg.norm(result), 1.0, places=7)
    #    
    #    # Check that result is a valid 2-element state vector
    #    self.assertEqual(len(result), 2)

    #def test_hamiltonian_to_state_three_qubit(self):
    #    """Test three-qubit zero Hamiltonian gives |000⟩ state"""
    #    H = np.zeros((8, 8), dtype=complex)
    #    result = hamiltonian_to_state(H)
    #    expected = np.zeros(8, dtype=complex)
    #    expected[0] = 1.0
    #    np.testing.assert_array_almost_equal(result, expected, decimal=7)

    #def test_hamiltonian_to_state_normalization(self):
    #    """Test that output state is always normalized"""
    #    test_cases = [
    #        np.zeros((2, 2)),  # 1-qubit
    #        np.zeros((4, 4)),  # 2-qubit
    #        np.eye(2) * 0.5,   # Small non-zero Hamiltonian
    #        np.array([[1, 2], [2, 1]]) * 0.1  # Another small Hamiltonian
    #    ]
    #    
    #    for H in test_cases:
    #        with self.subTest(H_shape=H.shape):
    #            result = hamiltonian_to_state(H)
    #            norm = np.linalg.norm(result)
    #            self.assertAlmostEqual(norm, 1.0, places=7)

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

    #def test_hamiltonian_to_state_complex_hamiltonian(self):
    #    """Test with complex Hamiltonian entries"""
    #    H = np.array([
    #        [0, 1-1j],
    #        [1+1j, 0]
    #    ], dtype=complex)
    #    
    #    result = hamiltonian_to_state(H)
    #    
    #    # Check normalization
    #    self.assertAlmostEqual(np.linalg.norm(result), 1.0, places=7)
    #    
    #    # Check that result has complex entries
    #    self.assertTrue(np.iscomplexobj(result))

    #def test_hamiltonian_to_state_large_hamiltonian(self):
    #    """Test with larger Hamiltonian matrix"""
    #    # 4-qubit system (16x16 matrix)
    #    H = np.zeros((16, 16), dtype=complex)
    #    # Add some small random Hermitian perturbation
    #    np.random.seed(123)
    #    A = np.random.randn(16, 16) + 1j * np.random.randn(16, 16)
    #    H = (A + A.conj().T) / 2 * 0.01  # Small Hermitian matrix
    #    
    #    result = hamiltonian_to_state(H)
    #    
    #    # Check properties
    #    self.assertEqual(len(result), 16)
    #    self.assertAlmostEqual(np.linalg.norm(result), 1.0, places=6)
    #    self.assertAlmostEqual(abs(result[0]), 1.0, places=2)  # Should be close to |0000⟩

    #def test_hamiltonian_to_state_power_of_two_dimensions(self):
    #    """Test that function works only with power-of-2 dimensions"""
    #    # Valid dimensions (powers of 2)
    #    valid_dims = [2, 4, 8, 16]
    #    for dim in valid_dims:
    #        with self.subTest(dim=dim):
    #            H = np.zeros((dim, dim), dtype=complex)
    #            result = hamiltonian_to_state(H)
    #            self.assertEqual(len(result), dim)
    #            self.assertAlmostEqual(np.linalg.norm(result), 1.0, places=7)

    #def test_hamiltonian_to_state_small_rotation(self):
    #    """Test small rotation Hamiltonians"""
    #    # Small rotation around X axis
    #    theta = 0.1
    #    H = theta * np.array([[0, 1], [1, 0]], dtype=complex)
    #    result = hamiltonian_to_state(H)
    #    
    #    # For small θ, |ψ⟩ ≈ |0⟩ + iθ|1⟩
    #    expected_approx = np.array([1.0, 1j * theta], dtype=complex)
    #    expected_approx = expected_approx / np.linalg.norm(expected_approx)
    #    
    #    # Should be close for small theta
    #    self.assertAlmostEqual(abs(result[0]), abs(expected_approx[0]), places=2)

    def test_hamiltonian_to_state_return_type(self):
        """Test that function returns correct type"""
        H = np.zeros((2, 2), dtype=complex)
        result = hamiltonian_to_state(H)
        
        # Check return type
        self.assertIsInstance(result, np.ndarray)
        self.assertTrue(np.iscomplexobj(result))
        self.assertEqual(result.dtype, complex)

    #def test_hamiltonian_to_state_zero_state_property(self):
    #    """Test that |0⟩ is always the first basis state"""
    #    for n_qubits in [1, 2, 3]:
    #        dim = 2 ** n_qubits
    #        H = np.zeros((dim, dim), dtype=complex)
    #        result = hamiltonian_to_state(H)
    #        
    #        # |0⟩ state should have amplitude 1 in first component, 0 elsewhere
    #        expected = np.zeros(dim, dtype=complex)
    #        expected[0] = 1.0
    #        
    #        np.testing.assert_array_almost_equal(result, expected, decimal=7)

    def test_hamiltonian_to_state_inverse_relationship(self):
        """Test relationship with check_states_equal function"""
        H = np.array([[0.1, 0.2], [0.2, -0.1]], dtype=complex)
        result1 = hamiltonian_to_state(H)
        result2 = hamiltonian_to_state(H)
        
        # Same Hamiltonian should give identical states
        self.assertTrue(check_states_equal(result1, result2, tol=1e-10))


class TestStatesEqual(unittest.TestCase):
    """Testing the NLQK quantum state equality checking functionality."""

    def test_check_states_equal_identical_states(self):
        """Test that identical states are considered equal"""
        psi1 = np.array([1, 0], dtype=complex)
        psi2 = np.array([1, 0], dtype=complex)
        result = check_states_equal(psi1, psi2)
        self.assertTrue(result)

    def test_check_states_equal_global_phase(self):
        """Test that states differing by global phase are equal"""
        psi1 = np.array([1, 0], dtype=complex)
        psi2 = np.array([1j, 0], dtype=complex)  # Global phase of π/2
        result = check_states_equal(psi1, psi2)
        self.assertTrue(result)

    def test_check_states_equal_global_phase_negative(self):
        """Test states with negative global phase"""
        psi1 = np.array([1, 0], dtype=complex)
        psi2 = np.array([-1, 0], dtype=complex)  # Global phase of π
        result = check_states_equal(psi1, psi2)
        self.assertTrue(result)

    def test_check_states_equal_complex_global_phase(self):
        """Test states with complex global phase"""
        psi1 = np.array([1/np.sqrt(2), 1/np.sqrt(2)], dtype=complex)
        phase = np.exp(1j * np.pi / 3)  # e^(iπ/3)
        psi2 = phase * psi1
        result = check_states_equal(psi1, psi2)
        self.assertTrue(result)

    def test_check_states_equal_different_states(self):
        """Test that genuinely different states are not equal"""
        psi1 = np.array([1, 0], dtype=complex)  # |0⟩
        psi2 = np.array([0, 1], dtype=complex)  # |1⟩
        result = check_states_equal(psi1, psi2)
        self.assertFalse(result)

    def test_check_states_equal_superposition_states(self):
        """Test equality of superposition states"""
        psi1 = np.array([1/np.sqrt(2), 1/np.sqrt(2)], dtype=complex)  # |+⟩
        psi2 = np.array([1/np.sqrt(2), 1/np.sqrt(2)], dtype=complex)  # |+⟩
        result = check_states_equal(psi1, psi2)
        self.assertTrue(result)

    def test_check_states_equal_superposition_with_phase(self):
        """Test superposition states with global phase"""
        psi1 = np.array([1/np.sqrt(2), 1/np.sqrt(2)], dtype=complex)  # |+⟩
        psi2 = np.array([1j/np.sqrt(2), 1j/np.sqrt(2)], dtype=complex)  # i|+⟩
        result = check_states_equal(psi1, psi2)
        self.assertTrue(result)

    def test_check_states_equal_orthogonal_superposition(self):
        """Test orthogonal superposition states"""
        psi1 = np.array([1/np.sqrt(2), 1/np.sqrt(2)], dtype=complex)   # |+⟩
        psi2 = np.array([1/np.sqrt(2), -1/np.sqrt(2)], dtype=complex)  # |-⟩
        result = check_states_equal(psi1, psi2)
        self.assertFalse(result)

    def test_check_states_equal_unnormalized_states(self):
        """Test that function handles unnormalized states correctly"""
        psi1 = np.array([2, 0], dtype=complex)  # Unnormalized |0⟩
        psi2 = np.array([3, 0], dtype=complex)  # Different unnormalized |0⟩
        result = check_states_equal(psi1, psi2)
        self.assertTrue(result)  # Should be equal after normalization

    def test_check_states_equal_unnormalized_different(self):
        """Test unnormalized but genuinely different states"""
        psi1 = np.array([2, 0], dtype=complex)  # Unnormalized |0⟩
        psi2 = np.array([0, 3], dtype=complex)  # Unnormalized |1⟩
        result = check_states_equal(psi1, psi2)
        self.assertFalse(result)

    def test_check_states_equal_multiqubit_identical(self):
        """Test equality for multi-qubit states"""
        psi1 = np.array([1, 0, 0, 0], dtype=complex)  # |00⟩
        psi2 = np.array([1, 0, 0, 0], dtype=complex)  # |00⟩
        result = check_states_equal(psi1, psi2)
        self.assertTrue(result)

    def test_check_states_equal_multiqubit_with_phase(self):
        """Test multi-qubit states with global phase"""
        psi1 = np.array([1, 0, 0, 0], dtype=complex)  # |00⟩
        psi2 = np.array([-1j, 0, 0, 0], dtype=complex)  # -i|00⟩
        result = check_states_equal(psi1, psi2)
        self.assertTrue(result)

    #def test_check_states_equal_entangled_states(self):
    #    """Test equality of entangled states"""
    #    # Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2
    #    psi1 = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)], dtype=complex)
    #    psi2 = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)], dtype=complex)
    #
    #     result = check_states_equal(psi1, psi2)
    #    self.assertTrue(result)

    #def test_check_states_equal_different_entangled_states(self):
    #    """Test different entangled states"""
    #    # |Φ+⟩ = (|00⟩ + |11⟩)/√2 vs |Φ-⟩ = (|00⟩ - |11⟩)/√2
    #    psi1 = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)], dtype=complex)
    #    psi2 = np.array([1/np.sqrt(2), 0, 0, -1/np.sqrt(2)], dtype=complex)
    #    result = check_states_equal(psi1, psi2)
    #    self.assertFalse(result)

    def test_check_states_equal_custom_tolerance(self):
        """Test with custom tolerance"""
        # States that are more significantly different
        psi1 = np.array([1, 0], dtype=complex)
        psi2 = np.array([0.99, 0.141], dtype=complex)  # More different states
        
        # Normalize psi2 to see the actual difference
        psi2 = psi2 / np.linalg.norm(psi2)
        
        # Should fail with tight tolerance
        result_tight = check_states_equal(psi1, psi2, tol=1e-8)
        self.assertFalse(result_tight)
        
        # Should pass with very loose tolerance
        result_loose = check_states_equal(psi1, psi2, tol=0.5)
        self.assertTrue(result_loose)

    def test_check_states_equal_tolerance_boundary(self):
        """Test tolerance boundary conditions"""
        # Create states with known inner product
        psi1 = np.array([1, 0], dtype=complex)
        psi2 = np.array([0.9, 0.436], dtype=complex)  # |<psi1|psi2>| = 0.9
        psi2 = psi2 / np.linalg.norm(psi2)
        
        # Should fail with tolerance smaller than the difference from 1
        result_fail = check_states_equal(psi1, psi2, tol=0.05)  # 1 - 0.9 = 0.1 > 0.05
        self.assertFalse(result_fail)
        
        # Should pass with tolerance larger than the difference from 1
        result_pass = check_states_equal(psi1, psi2, tol=0.15)  # 1 - 0.9 = 0.1 < 0.15
        self.assertTrue(result_pass)

    def test_check_states_equal_very_small_differences(self):
        """Test with very small numerical differences"""
        psi1 = np.array([1, 0], dtype=complex)
        psi2 = np.array([1 + 1e-10, 1e-10], dtype=complex)  # Tiny numerical error
        result = check_states_equal(psi1, psi2, tol=1e-6)
        self.assertTrue(result)

    def test_check_states_equal_complex_coefficients(self):
        """Test states with complex coefficients"""
        psi1 = np.array([1+2j, 3-1j], dtype=complex)
        psi1 = psi1 / np.linalg.norm(psi1)  # Normalize
        
        # Same state with global phase
        phase = np.exp(1j * 0.7)
        psi2 = phase * psi1
        
        result = check_states_equal(psi1, psi2)
        self.assertTrue(result)

    def test_check_states_equal_single_qubit_basis(self):
        """Test all single qubit basis states"""
        states = [
            np.array([1, 0], dtype=complex),  # |0⟩
            np.array([0, 1], dtype=complex),  # |1⟩
        ]
        
        # Each state should equal itself
        for psi in states:
            self.assertTrue(check_states_equal(psi, psi))
        
        # Different states should not be equal
        self.assertFalse(check_states_equal(states[0], states[1]))

    def test_check_states_equal_edge_case_zero_norm(self):
        """Test edge case with zero norm vector"""
        psi1 = np.array([1, 0], dtype=complex)
        psi2 = np.array([0, 0], dtype=complex)  # Zero vector
        
        # The current implementation normalizes by dividing by norm
        # With zero norm, this will create NaN or inf values
        # Let's test what actually happens and verify the behavior
        try:
            result = check_states_equal(psi1, psi2)
            # If it doesn't raise an error, the result should be False or NaN
            # NaN comparisons are always False, so this should work
            self.assertFalse(result)
        except (ValueError, RuntimeWarning, ZeroDivisionError):
            # If it does raise an error, that's also acceptable behavior
            pass

    def test_check_states_equal_numpy_array_types(self):
        """Test with different numpy array types"""
        # Test with different dtypes
        psi1 = np.array([1, 0], dtype=complex)
        psi2 = np.array([1.0, 0.0], dtype=float)  # Real array
        
        result = check_states_equal(psi1, psi2)
        self.assertTrue(result)

    #def test_check_states_equal_return_type(self):
    #    """Test that function returns boolean"""
    #    psi1 = np.array([1, 0], dtype=complex)
    #    psi2 = np.array([0, 1], dtype=complex)
    #    result = check_states_equal(psi1, psi2)
    #    self.assertIsInstance(result, (bool, np.bool_))

    def test_check_states_equal_three_qubit_states(self):
        """Test with three-qubit states"""
        # |000⟩ state
        psi1 = np.zeros(8, dtype=complex)
        psi1[0] = 1.0
        
        # Same state with phase
        psi2 = np.zeros(8, dtype=complex)
        psi2[0] = np.exp(1j * np.pi / 4)
        
        result = check_states_equal(psi1, psi2)
        self.assertTrue(result)

    def test_check_states_equal_random_phases(self):
        """Test with random global phases"""
        np.random.seed(42)
        base_state = np.array([1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)], dtype=complex)
        
        # Test multiple random phases
        for _ in range(10):
            phase = np.exp(1j * np.random.uniform(0, 2*np.pi))
            phased_state = phase * base_state
            result = check_states_equal(base_state, phased_state)
            self.assertTrue(result)

    def test_check_states_equal_numerical_stability(self):
        """Test numerical stability with very large and small numbers"""
        # Very large coefficients
        psi1 = np.array([1e10, 1e10], dtype=complex)
        psi2 = np.array([1e10, 1e10], dtype=complex)
        result = check_states_equal(psi1, psi2)
        self.assertTrue(result)
        
        # Very small coefficients
        psi3 = np.array([1e-10, 1e-10], dtype=complex)
        psi4 = np.array([1e-10, 1e-10], dtype=complex)
        result = check_states_equal(psi3, psi4)
        self.assertTrue(result)



if __name__ == '__main__':
    unittest.main()
