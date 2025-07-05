#!/usr/bin/env python3

"""
test_vectors_pad.py

(C) 2025 by [Damir Cavar](https://damir.cavar.me/) and [NLP Lab](https://nlp-lab.org/)

Testing the NLQK vector padding functionality.
"""

import sys
sys.path.append('.') # ./..')
import unittest
import numpy as np
from nlqk.embeddings import pad_vector


class TestVectorsPad(unittest.TestCase):
    """Testing the NLQK vector padding functionality."""

    def test_pad_vector_basic(self):
        """Test basic vector padding functionality"""
        vector = [1, 2, 3]
        target_size = 5
        result = pad_vector(vector, target_size)
        expected = np.array([1+0j, 2+0j, 3+0j, 0+0j, 0+0j])
        np.testing.assert_array_equal(result, expected)

    def test_pad_vector_complex_input(self):
        """Test padding with complex input vector"""
        vector = [1+2j, 3+4j]
        target_size = 4
        result = pad_vector(vector, target_size)
        expected = np.array([1+2j, 3+4j, 0+0j, 0+0j])
        np.testing.assert_array_equal(result, expected)

    def test_pad_vector_exact_size(self):
        """Test padding when vector is exactly target size"""
        vector = [1, 2, 3, 4]
        target_size = 4
        result = pad_vector(vector, target_size)
        expected = np.array([1+0j, 2+0j, 3+0j, 4+0j])
        np.testing.assert_array_equal(result, expected)

    def test_pad_vector_power_of_two(self):
        """Test padding to power of 2 sizes"""
        vector = [1, 2, 3]
        target_size = 8  # 2^3
        result = pad_vector(vector, target_size)
        expected = np.array([1+0j, 2+0j, 3+0j, 0+0j, 0+0j, 0+0j, 0+0j, 0+0j])
        np.testing.assert_array_equal(result, expected)

    def test_pad_vector_numpy_array_input(self):
        """Test padding with numpy array as input"""
        vector = np.array([1.5, 2.7, 3.9])
        target_size = 6
        result = pad_vector(vector, target_size)
        expected = np.array([1.5+0j, 2.7+0j, 3.9+0j, 0+0j, 0+0j, 0+0j])
        np.testing.assert_array_almost_equal(result, expected, decimal=7)

    def test_pad_vector_empty_vector(self):
        """Test padding empty vector"""
        vector = []
        target_size = 3
        result = pad_vector(vector, target_size)
        expected = np.array([0+0j, 0+0j, 0+0j])
        np.testing.assert_array_equal(result, expected)

    def test_pad_vector_single_element(self):
        """Test padding single element vector"""
        vector = [42]
        target_size = 4
        result = pad_vector(vector, target_size)
        expected = np.array([42+0j, 0+0j, 0+0j, 0+0j])
        np.testing.assert_array_equal(result, expected)

    def test_pad_vector_negative_values(self):
        """Test padding with negative values"""
        vector = [-1, -2, 3]
        target_size = 5
        result = pad_vector(vector, target_size)
        expected = np.array([-1+0j, -2+0j, 3+0j, 0+0j, 0+0j])
        np.testing.assert_array_equal(result, expected)

    def test_pad_vector_zeros_in_input(self):
        """Test padding vector that already contains zeros"""
        vector = [1, 0, 2, 0]
        target_size = 6
        result = pad_vector(vector, target_size)
        expected = np.array([1+0j, 0+0j, 2+0j, 0+0j, 0+0j, 0+0j])
        np.testing.assert_array_equal(result, expected)

    def test_pad_vector_too_large_raises_error(self):
        """Test that vector larger than target size raises ValueError"""
        vector = [1, 2, 3, 4, 5]
        target_size = 3
        with self.assertRaises(ValueError) as context:
            pad_vector(vector, target_size)
        self.assertIn("Vector is larger than the target size", str(context.exception))

    def test_pad_vector_zero_target_size(self):
        """Test padding with zero target size"""
        vector = []
        target_size = 0
        result = pad_vector(vector, target_size)
        expected = np.array([], dtype=complex)
        np.testing.assert_array_equal(result, expected)

    def test_pad_vector_zero_target_size_with_data_raises_error(self):
        """Test that non-empty vector with zero target size raises error"""
        vector = [1, 2]
        target_size = 0
        with self.assertRaises(ValueError) as context:
            pad_vector(vector, target_size)
        self.assertIn("Vector is larger than the target size", str(context.exception))

    def test_pad_vector_return_type(self):
        """Test that function returns numpy array with complex dtype"""
        vector = [1, 2, 3]
        target_size = 5
        result = pad_vector(vector, target_size)
        
        # Check that result is numpy array
        self.assertIsInstance(result, np.ndarray)
        
        # Check that dtype is complex
        self.assertTrue(np.iscomplexobj(result))
        self.assertEqual(result.dtype, complex)
        
        # Check correct size
        self.assertEqual(len(result), target_size)

    def test_pad_vector_preserves_original_values(self):
        """Test that padding preserves original vector values"""
        vector = [1.1, 2.2, 3.3]
        target_size = 7
        result = pad_vector(vector, target_size)
        
        # Check original values are preserved
        for i, val in enumerate(vector):
            self.assertAlmostEqual(result[i].real, val, places=7)
            self.assertAlmostEqual(result[i].imag, 0.0, places=7)
        
        # Check padding zeros
        for i in range(len(vector), target_size):
            self.assertEqual(result[i], 0+0j)

    def test_pad_vector_very_small_values(self):
        """Test padding with very small values"""
        vector = [1e-15, 2e-15]
        target_size = 4
        result = pad_vector(vector, target_size)
        expected = np.array([1e-15+0j, 2e-15+0j, 0+0j, 0+0j])
        np.testing.assert_array_almost_equal(result, expected, decimal=20)

    def test_pad_vector_very_large_values(self):
        """Test padding with very large values"""
        vector = [1e10, 2e10]
        target_size = 5
        result = pad_vector(vector, target_size)
        expected = np.array([1e10+0j, 2e10+0j, 0+0j, 0+0j, 0+0j])
        np.testing.assert_array_equal(result, expected)

    def test_pad_vector_mixed_complex_real(self):
        """Test padding with mixed complex and real values"""
        vector = [1, 2+3j, 4]
        target_size = 6
        result = pad_vector(vector, target_size)
        expected = np.array([1+0j, 2+3j, 4+0j, 0+0j, 0+0j, 0+0j])
        np.testing.assert_array_equal(result, expected)

    def test_pad_vector_large_target_size(self):
        """Test padding to very large target size"""
        vector = [1, 2]
        target_size = 1000
        result = pad_vector(vector, target_size)
        
        # Check first two elements
        self.assertEqual(result[0], 1+0j)
        self.assertEqual(result[1], 2+0j)
        
        # Check some random positions in the padded region
        self.assertEqual(result[10], 0+0j)
        self.assertEqual(result[500], 0+0j)
        self.assertEqual(result[999], 0+0j)
        
        # Check total length
        self.assertEqual(len(result), target_size)

    def test_pad_vector_quantum_amplitudes_example(self):
        """Test padding for quantum amplitude encoding use case"""
        # Simulate normalized quantum amplitudes
        vector = np.array([0.6, 0.8])  # |ψ⟩ = 0.6|0⟩ + 0.8|1⟩
        target_size = 4  # Pad to 2 qubits (4 amplitudes)
        result = pad_vector(vector, target_size)
        
        expected = np.array([0.6+0j, 0.8+0j, 0+0j, 0+0j])
        np.testing.assert_array_almost_equal(result, expected, decimal=7)
        
        # Check that the first two amplitudes maintain normalization property
        norm_original = np.linalg.norm(vector)
        norm_padded_partial = np.linalg.norm(result[:2])
        self.assertAlmostEqual(norm_original, norm_padded_partial, places=7)

    def test_pad_vector_edge_case_one_element_to_one(self):
        """Test edge case: padding one element to size one"""
        vector = [5]
        target_size = 1
        result = pad_vector(vector, target_size)
        expected = np.array([5+0j])
        np.testing.assert_array_equal(result, expected)


if __name__ == '__main__':
    unittest.main()
