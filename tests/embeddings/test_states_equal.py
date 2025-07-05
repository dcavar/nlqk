#!/usr/bin/env python3

"""
test_states_equal.py

(C) 2025 by [Damir Cavar](https://damir.cavar.me/) and [NLP Lab](https://nlp-lab.org/)

Testing the NLQK quantum state equality checking functionality.
"""

import sys
sys.path.append('.') # ./..')
import os
import unittest
import numpy as np
from nlqk.embeddings import check_states_equal


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

    def test_check_states_equal_entangled_states(self):
        """Test equality of entangled states"""
        # Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2
        psi1 = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)], dtype=complex)
        psi2 = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)], dtype=complex)
        result = check_states_equal(psi1, psi2)
        self.assertTrue(result)

    def test_check_states_equal_different_entangled_states(self):
        """Test different entangled states"""
        # |Φ+⟩ = (|00⟩ + |11⟩)/√2 vs |Φ-⟩ = (|00⟩ - |11⟩)/√2
        psi1 = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)], dtype=complex)
        psi2 = np.array([1/np.sqrt(2), 0, 0, -1/np.sqrt(2)], dtype=complex)
        result = check_states_equal(psi1, psi2)
        self.assertFalse(result)

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

    def test_check_states_equal_return_type(self):
        """Test that function returns boolean"""
        psi1 = np.array([1, 0], dtype=complex)
        psi2 = np.array([0, 1], dtype=complex)
        result = check_states_equal(psi1, psi2)
        self.assertIsInstance(result, (bool, np.bool_))

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
