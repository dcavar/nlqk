#!/usr/bin/env python3
# coding: utf-8


"""
test_hamiltonian_to_state.py

(C) 2025 by [Damir Cavar](http://damir.cavar.me/), James Bryan Graves, and [NLP Lab](https://nlp-lab.org/)

Testing the NLQK Hamiltonian to state conversion functionality.
"""


import sys
sys.path.append('.')
import os
import unittest
if os.getenv("GITHUB_ACTIONS") == "true":
    import numpy as np
    from scipy.linalg import expm
else:
    try: # prefer RAPIDS libraries and GPU over numpy and CPU
        import cupy as np  # Try to import cupy and alias it as np
        from cupyx.scipy.linalg import expm
        _USE_GPU = True
    except ModuleNotFoundError:
        import numpy as np  # If cupy not found, import numpy and alias it as np
        from scipy.linalg import expm
        _USE_GPU = False
# import GPUtil  # If you're using GPUtil
from nlqk.embeddings.states import hamiltonian_to_state, check_states_equal, householder, state_to_hamiltonian, pad_hamiltonian


class TestHamiltonianToState(unittest.TestCase):
    """Testing the NLQK Hamiltonian to state conversion functionality."""

    def test_hamiltonian_to_state_identity(self):
        """Test that zero Hamiltonian gives |0⟩ state"""
        # 1-qubit case: H = 0 matrix should give |0⟩ = [1, 0]
        H = np.zeros((2, 2), dtype=complex)
        result = hamiltonian_to_state(H, init_state="zero")
        expected = np.array([1.0, 0.0], dtype=complex)
        np.testing.assert_array_almost_equal(result, expected, decimal=7)

    def test_hamiltonian_to_state_pauli_x(self):
        """Test Hamiltonian corresponding to Pauli-X rotation"""
        # H = π/2 * σ_x gives U = exp(i*π/2*σ_x) = cos(π/2)*I + i*sin(π/2)*σ_x = i*σ_x
        # U|0⟩ = i*σ_x|0⟩ = i*|1⟩ = [0, i]
        H = (np.pi / 2) * np.array([[0, 1], [1, 0]], dtype=complex)
        result = hamiltonian_to_state(H, init_state="zero", forward=False)
        expected = np.array([0.0, 1j], dtype=complex)
        np.testing.assert_array_almost_equal(result, expected, decimal=6)

    def test_hamiltonian_to_state_pauli_y(self):
        """Test Hamiltonian corresponding to Pauli-Y rotation"""
        # H = π/2 * σ_y gives U = exp(i*π/2*σ_y) = cos(π/2)*I + i*sin(π/2)*σ_y = i*σ_y
        # U|0⟩ = i*σ_y|0⟩ = i*i*|1⟩ = -|1⟩ = [0, -1]
        H = (np.pi / 2) * np.array([[0, -1j], [1j, 0]], dtype=complex)
        result = hamiltonian_to_state(H, init_state="zero", forward=False)
        expected = np.array([0.0, -1.0], dtype=complex)
        np.testing.assert_array_almost_equal(result, expected, decimal=6)

    def test_hamiltonian_to_state_hadamard_like(self):
        """Test Hamiltonian that creates Hadamard-like superposition"""
        # For H = π/4 * σ_x, we get a rotation that creates superposition
        # This should give approximately equal amplitudes
        H = (np.pi / 4) * np.array([[0, 1], [1, 0]], dtype=complex)
        result = hamiltonian_to_state(H, init_state="zero")
        
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
        result = hamiltonian_to_state(H, init_state="zero")
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
        result = hamiltonian_to_state(H, init_state="zero", forward=False)
        
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
        result = hamiltonian_to_state(H, init_state="zero")
        
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
        result1 = hamiltonian_to_state(H, init_state="zero")
        result2 = hamiltonian_to_state(H, init_state="zero")
        
        # Same Hamiltonian should give identical states
        self.assertTrue(check_states_equal(result1, result2, tol=1e-10))

    def test_hamiltonian_to_state_superposition_init(self):
        """Test that function works with superposition initial state"""
        # Zero Hamiltonian with superposition initial state should give equal superposition
        H = np.zeros((4, 4), dtype=complex)
        result = hamiltonian_to_state(H, init_state="superposition")
        expected = np.array([0.5, 0.5, 0.5, 0.5], dtype=complex)
        np.testing.assert_array_almost_equal(result, expected, decimal=7)

    def test_hamiltonian_to_state_different_init_states(self):
        """Test that different initial states give different results"""
        H = np.array([[0.1, 0.2], [0.2, -0.1]], dtype=complex)
        
        result_zero = hamiltonian_to_state(H, init_state="zero")
        result_super = hamiltonian_to_state(H, init_state="superposition")
        
        # Results should be different (not equal up to global phase)
        self.assertFalse(check_states_equal(result_zero, result_super, tol=1e-6))


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



class TestHouseholder(unittest.TestCase):
    """Testing the NLQK Householder reflection functionality."""

    def test_householder_identical_states(self):
        """Test householder with identical source and target states"""
        s = np.array([1, 0], dtype=complex)
        psi = np.array([1, 0], dtype=complex)
        κ, R = householder(s, psi)
        
        # Should have phase factor κ = 1 and identity-like transformation
        result = R @ s
        expected = κ * psi
        np.testing.assert_array_almost_equal(result, expected, decimal=7)

    def test_householder_orthogonal_states(self):
        """Test householder with orthogonal states"""
        s = np.array([1, 0], dtype=complex)  # |0⟩
        psi = np.array([0, 1], dtype=complex)  # |1⟩
        κ, R = householder(s, psi)
        
        # Check that R|s⟩ = κ|ψ⟩
        result = R @ s
        expected = κ * psi
        np.testing.assert_array_almost_equal(result, expected, decimal=7)
        
        # For orthogonal states, κ should be 1
        self.assertAlmostEqual(κ, 1.0, places=7)

    def test_householder_plus_minus_states(self):
        """Test householder with |+⟩ and |-⟩ states"""
        s = np.array([1/np.sqrt(2), 1/np.sqrt(2)], dtype=complex)    # |+⟩
        psi = np.array([1/np.sqrt(2), -1/np.sqrt(2)], dtype=complex) # |-⟩
        κ, R = householder(s, psi)
        
        # Check that R|s⟩ = κ|ψ⟩
        result = R @ s
        expected = κ * psi
        np.testing.assert_array_almost_equal(result, expected, decimal=7)

    def test_householder_matrix_properties(self):
        """Test that householder matrix R has correct properties"""
        s = np.array([1, 0], dtype=complex)
        psi = np.array([1/np.sqrt(2), 1/np.sqrt(2)], dtype=complex)
        κ, R = householder(s, psi)
        
        # R should be Hermitian
        np.testing.assert_array_almost_equal(R, R.conj().T, decimal=7)
        
        # R should be unitary (R† R = I)
        identity = R.conj().T @ R
        expected_identity = np.eye(R.shape[0], dtype=complex)
        np.testing.assert_array_almost_equal(identity, expected_identity, decimal=7)

    def test_householder_phase_factor(self):
        """Test that phase factor κ has unit magnitude"""
        s = np.array([1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)], dtype=complex)
        psi = np.array([0.5, 0.5, 1/np.sqrt(2)], dtype=complex)
        psi = psi / np.linalg.norm(psi)  # Normalize
        
        κ, R = householder(s, psi)
        
        # κ should have unit magnitude
        self.assertAlmostEqual(abs(κ), 1.0, places=7)

    def test_householder_complex_states(self):
        """Test householder with complex state vectors"""
        s = np.array([1, 0], dtype=complex)
        psi = np.array([1j/np.sqrt(2), 1/np.sqrt(2)], dtype=complex)
        κ, R = householder(s, psi)
        
        # Check transformation
        result = R @ s
        expected = κ * psi
        np.testing.assert_array_almost_equal(result, expected, decimal=7)

    def test_householder_multiqubit_states(self):
        """Test householder with multi-qubit states"""
        # 2-qubit states
        s = np.array([1, 0, 0, 0], dtype=complex)  # |00⟩
        psi = np.array([0, 0, 0, 1], dtype=complex)  # |11⟩
        κ, R = householder(s, psi)
        
        # Check transformation
        result = R @ s
        expected = κ * psi
        np.testing.assert_array_almost_equal(result, expected, decimal=7)
        
        # Check matrix properties
        np.testing.assert_array_almost_equal(R, R.conj().T, decimal=7)  # Hermitian
        identity = R.conj().T @ R
        expected_identity = np.eye(4, dtype=complex)
        np.testing.assert_array_almost_equal(identity, expected_identity, decimal=7)  # Unitary


class TestStateToHamiltonian(unittest.TestCase):
    """Testing the NLQK state-to-Hamiltonian conversion functionality."""

    def test_state_to_hamiltonian_uniform_superposition(self):
        """Test with uniform superposition (should give zero Hamiltonian)"""
        psi = np.array([1/np.sqrt(2), 1/np.sqrt(2)], dtype=complex)  # |+⟩
        H = state_to_hamiltonian(psi)
        
        # Should be approximately zero matrix
        expected = np.zeros((2, 2), dtype=complex)
        np.testing.assert_array_almost_equal(H, expected, decimal=6)

    def test_state_to_hamiltonian_basic_state(self):
        """Test with |0⟩ state"""
        psi = np.array([1, 0], dtype=complex)  # |0⟩
        H = state_to_hamiltonian(psi)
        
        # Check that exp(-iH) applied to uniform superposition gives |0⟩
        s = np.array([1/np.sqrt(2), 1/np.sqrt(2)], dtype=complex)  # uniform superposition
        U = expm(-1j * H)
        result = U @ s
        
        # Result should be proportional to psi
        self.assertTrue(check_states_equal(result, psi, tol=1e-6))

    def test_state_to_hamiltonian_basis_state(self):
        """Test with |1⟩ state"""
        psi = np.array([0, 1], dtype=complex)  # |1⟩
        H = state_to_hamiltonian(psi)
        
        # Check that exp(-iH) applied to uniform superposition gives |1⟩
        s = np.array([1/np.sqrt(2), 1/np.sqrt(2)], dtype=complex)
        U = expm(-1j * H)
        result = U @ s
        
        self.assertTrue(check_states_equal(result, psi, tol=1e-6))

    def test_state_to_hamiltonian_hermitian_property(self):
        """Test that returned Hamiltonian is Hermitian"""
        psi = np.array([0.6, 0.8], dtype=complex)
        H = state_to_hamiltonian(psi)
        
        # H should be Hermitian
        np.testing.assert_array_almost_equal(H, H.conj().T, decimal=7)

    def test_state_to_hamiltonian_complex_state(self):
        """Test with complex target state"""
        psi = np.array([1j/np.sqrt(2), 1/np.sqrt(2)], dtype=complex)
        H = state_to_hamiltonian(psi)
        
        # Check Hermiticity
        np.testing.assert_array_almost_equal(H, H.conj().T, decimal=7)
        
        # Check evolution
        s = np.array([1/np.sqrt(2), 1/np.sqrt(2)], dtype=complex)
        U = expm(-1j * H)
        result = U @ s
        self.assertTrue(check_states_equal(result, psi, tol=1e-6))

    def test_state_to_hamiltonian_multiqubit(self):
        """Test with multi-qubit target states"""
        # 2-qubit Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2
        psi = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)], dtype=complex)
        H = state_to_hamiltonian(psi)
        
        # Check Hermiticity
        np.testing.assert_array_almost_equal(H, H.conj().T, decimal=7)
        
        # Check evolution from uniform superposition
        s = np.ones(4, dtype=complex) / 2.0  # uniform 2-qubit superposition
        U = expm(-1j * H)
        result = U @ s
        self.assertTrue(check_states_equal(result, psi, tol=1e-6))

    def test_state_to_hamiltonian_normalization(self):
        """Test that function handles non-normalized input correctly"""
        psi_unnormalized = np.array([3, 4], dtype=complex)  # |3,4⟩
        H = state_to_hamiltonian(psi_unnormalized)
        
        # Should work the same as normalized version
        psi_normalized = psi_unnormalized / np.linalg.norm(psi_unnormalized)
        H_normalized = state_to_hamiltonian(psi_normalized)
        
        np.testing.assert_array_almost_equal(H, H_normalized, decimal=7)

    def test_state_to_hamiltonian_roundtrip(self):
        """Test roundtrip: state → Hamiltonian → evolved state"""
        original_psi = np.array([0.3, 0.4, 0.5, 0.7], dtype=complex)
        original_psi = original_psi / np.linalg.norm(original_psi)
        
        # Convert to Hamiltonian
        H = state_to_hamiltonian(original_psi)
        
        # Evolve uniform superposition
        s = np.ones(4, dtype=complex) / 2.0
        U = expm(-1j * H)
        evolved_psi = U @ s
        
        # Should get back the original state (up to global phase)
        self.assertTrue(check_states_equal(evolved_psi, original_psi, tol=1e-6))

    def test_state_to_hamiltonian_zero_case(self):
        """Test edge case with uniform superposition input"""
        # For 3-qubit uniform superposition
        psi = np.ones(8, dtype=complex) / np.sqrt(8)
        H = state_to_hamiltonian(psi)
        
        # Should be approximately zero
        expected = np.zeros((8, 8), dtype=complex)
        np.testing.assert_array_almost_equal(H, expected, decimal=6)

    def test_state_to_hamiltonian_return_type(self):
        """Test that function returns correct matrix type"""
        psi = np.array([0.6, 0.8], dtype=complex)
        H = state_to_hamiltonian(psi)
        
        # Check return type and properties
        self.assertIsInstance(H, np.ndarray)
        self.assertEqual(H.ndim, 2)
        self.assertEqual(H.shape[0], H.shape[1])  # Square matrix
        self.assertTrue(np.iscomplexobj(H))


class TestPadHamiltonian(unittest.TestCase):
    """Testing the NLQK Hamiltonian padding functionality."""

    def test_pad_hamiltonian_already_power_of_2(self):
        """Test that matrices that are already power-of-2 size are unchanged"""
        # 2x2 matrix (already power of 2)
        H_2x2 = np.array([[1, 0.5], [0.5, -1]], dtype=complex)
        result = pad_hamiltonian(H_2x2)
        np.testing.assert_array_equal(result, H_2x2)
        
        # 4x4 matrix (already power of 2)
        H_4x4 = np.random.randn(4, 4) + 1j * np.random.randn(4, 4)
        H_4x4 = (H_4x4 + H_4x4.conj().T) / 2  # Make Hermitian
        result = pad_hamiltonian(H_4x4)
        np.testing.assert_array_equal(result, H_4x4)

    def test_pad_hamiltonian_3x3_to_4x4(self):
        """Test padding 3x3 matrix to 4x4"""
        H_3x3 = np.array([
            [1, 0.5, 0.2],
            [0.5, -1, 0.3],
            [0.2, 0.3, 0.5]
        ], dtype=complex)
        
        result = pad_hamiltonian(H_3x3)
        
        # Check dimensions
        self.assertEqual(result.shape, (4, 4))
        
        # Check that original 3x3 block is preserved
        np.testing.assert_array_equal(result[:3, :3], H_3x3)
        
        # Check that padding is zeros
        np.testing.assert_array_equal(result[3, :], [0, 0, 0, 0])
        np.testing.assert_array_equal(result[:, 3], [0, 0, 0, 0])

    def test_pad_hamiltonian_5x5_to_8x8(self):
        """Test padding 5x5 matrix to 8x8"""
        H_5x5 = np.random.randn(5, 5) + 1j * np.random.randn(5, 5)
        H_5x5 = (H_5x5 + H_5x5.conj().T) / 2  # Make Hermitian
        
        result = pad_hamiltonian(H_5x5)
        
        # Check dimensions
        self.assertEqual(result.shape, (8, 8))
        
        # Check that original 5x5 block is preserved
        np.testing.assert_array_almost_equal(result[:5, :5], H_5x5)
        
        # Check that padding rows and columns are zeros
        np.testing.assert_array_equal(result[5:, :], np.zeros((3, 8)))
        np.testing.assert_array_equal(result[:, 5:], np.zeros((8, 3)))

    def test_pad_hamiltonian_preserves_hermiticity(self):
        """Test that padding preserves Hermitian property"""
        # Create a Hermitian 3x3 matrix
        A = np.random.randn(3, 3) + 1j * np.random.randn(3, 3)
        H_3x3 = (A + A.conj().T) / 2
        
        result = pad_hamiltonian(H_3x3)
        
        # Check that result is Hermitian
        np.testing.assert_array_almost_equal(result, result.conj().T)

    def test_pad_hamiltonian_preserves_dtype(self):
        """Test that padding preserves data type"""
        # Complex matrix
        H_complex = np.array([[1+1j, 0.5], [0.5, -1+0.5j]], dtype=complex)
        result_complex = pad_hamiltonian(H_complex)
        self.assertEqual(result_complex.dtype, complex)
        
        # Real matrix
        H_real = np.array([[1, 0.5, 0.2], [0.5, -1, 0.3], [0.2, 0.3, 0.5]], dtype=float)
        result_real = pad_hamiltonian(H_real)
        self.assertEqual(result_real.dtype, float)

    def test_pad_hamiltonian_evolution_equivalence(self):
        """Test that evolution in original vs padded space gives equivalent results"""
        # Create a 3x3 Hermitian matrix
        H_3x3 = np.array([
            [1, 0.5, 0.2],
            [0.5, -1, 0.3],
            [0.2, 0.3, 0.5]
        ], dtype=complex)
        
        # Pad to 4x4
        H_4x4 = pad_hamiltonian(H_3x3)
        
        # Create initial states
        psi_3 = np.array([1, 0, 0], dtype=complex)  # |000⟩ in 3D
        psi_4 = np.array([1, 0, 0, 0], dtype=complex)  # |000⟩ in 4D
        
        # Evolve both
        t = 0.5
        U_3 = expm(-1j * t * H_3x3)
        U_4 = expm(-1j * t * H_4x4)
        
        evolved_3 = U_3 @ psi_3
        evolved_4 = U_4 @ psi_4
        
        # First 3 components should be identical
        np.testing.assert_array_almost_equal(evolved_3, evolved_4[:3])
        
        # 4th component should remain zero
        self.assertAlmostEqual(evolved_4[3], 0.0)

    def test_pad_hamiltonian_single_element(self):
        """Test padding 1x1 matrix to 2x2"""
        H_1x1 = np.array([[2.5]], dtype=complex)
        result = pad_hamiltonian(H_1x1)
        
        expected = np.array([[2.5, 0], [0, 0]], dtype=complex)
        np.testing.assert_array_equal(result, expected)

    def test_pad_hamiltonian_large_matrix(self):
        """Test padding larger matrices"""
        # 7x7 should pad to 8x8
        H_7x7 = np.random.randn(7, 7) + 1j * np.random.randn(7, 7)
        H_7x7 = (H_7x7 + H_7x7.conj().T) / 2
        
        result = pad_hamiltonian(H_7x7)
        
        self.assertEqual(result.shape, (8, 8))
        np.testing.assert_array_almost_equal(result[:7, :7], H_7x7)

    def test_pad_hamiltonian_power_calculation(self):
        """Test that the power-of-2 calculation is correct for various sizes"""
        test_cases = [
            (1, 2),   # 1 -> 2
            (2, 2),   # 2 -> 2 (no change)
            (3, 4),   # 3 -> 4
            (4, 4),   # 4 -> 4 (no change)
            (5, 8),   # 5 -> 8
            (8, 8),   # 8 -> 8 (no change)
            (9, 16),  # 9 -> 16
            (15, 16), # 15 -> 16
            (16, 16), # 16 -> 16 (no change)
        ]
        
        for input_size, expected_size in test_cases:
            with self.subTest(input_size=input_size):
                H = np.random.randn(input_size, input_size)
                result = pad_hamiltonian(H)
                self.assertEqual(result.shape, (expected_size, expected_size))

    def test_pad_hamiltonian_zero_matrix(self):
        """Test padding zero matrices"""
        H_zero = np.zeros((3, 3), dtype=complex)
        result = pad_hamiltonian(H_zero)
        
        expected = np.zeros((4, 4), dtype=complex)
        np.testing.assert_array_equal(result, expected)

    def test_pad_hamiltonian_return_type(self):
        """Test that function returns correct array type"""
        H_3x3 = np.array([[1, 0.5, 0.2], [0.5, -1, 0.3], [0.2, 0.3, 0.5]])
        result = pad_hamiltonian(H_3x3)
        
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.ndim, 2)
        self.assertEqual(result.shape[0], result.shape[1])  # Square matrix

if __name__ == '__main__':
    unittest.main()
