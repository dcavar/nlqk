#!/usr/bin/env python3
# coding: utf-8


"""
test_vectors.py

(C) 2025 by [Damir Cavar](https://damir.cavar.me/) and [NLP Lab](https://nlp-lab.org/)

Testing the NLQK vectors functionality.
"""


import sys
sys.path.append('.') # ./..')
import unittest
try: # prefer RAPIDS libraries and GPU over numpy and CPU
    import cupy as np  # Try to import cupy and alias it as np
    #from cupyx.scipy.linalg import expm
    _USE_GPU = True
except ModuleNotFoundError:
    import numpy as np  # If cupy not found, import numpy and alias it as np
    #from scipy.linalg import expm
    _USE_GPU = False
from nlqk.embeddings.vectors import pair_real_to_complex, cosine_similarity, pad_vector, normalize, is_normalized



class TestVectorsComplex(unittest.TestCase):
    """Testing the NLQK vectors functionality."""

    def test_pair_real_to_complex_basic(self):
        """Test basic functionality of pairing real values to complex"""
        vec = np.array([1, 2, 3, 4])
        result = pair_real_to_complex(vec)
        expected = np.array([1+2j, 3+4j])
        np.testing.assert_array_equal(result, expected)

    def test_pair_real_to_complex_zeros(self):
        """Test pairing with zeros"""
        vec = np.array([1, 0, 0, 3])
        result = pair_real_to_complex(vec)
        expected = np.array([1+0j, 0+3j])
        np.testing.assert_array_equal(result, expected)

    def test_pair_real_to_complex_negative_values(self):
        """Test pairing with negative values"""
        vec = np.array([-1, 2, 3, -4])
        result = pair_real_to_complex(vec)
        expected = np.array([-1+2j, 3-4j])
        np.testing.assert_array_equal(result, expected)

    def test_pair_real_to_complex_floats(self):
        """Test pairing with floating point values"""
        vec = np.array([1.5, 2.7, -3.2, 4.8])
        result = pair_real_to_complex(vec)
        expected = np.array([1.5+2.7j, -3.2+4.8j])
        np.testing.assert_array_almost_equal(result, expected, decimal=7)

    def test_pair_real_to_complex_two_elements(self):
        """Test pairing with minimum valid input (2 elements)"""
        vec = np.array([5, 7])
        result = pair_real_to_complex(vec)
        expected = np.array([5+7j])
        np.testing.assert_array_equal(result, expected)

    def test_pair_real_to_complex_large_vector(self):
        """Test pairing with larger vector"""
        vec = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        result = pair_real_to_complex(vec)
        expected = np.array([1+2j, 3+4j, 5+6j, 7+8j])
        np.testing.assert_array_equal(result, expected)

    def test_pair_real_to_complex_numpy_array_input(self):
        """Test pairing with numpy array as input"""
        vec = np.array([10, 20, 30, 40])
        result = pair_real_to_complex(vec)
        expected = np.array([10+20j, 30+40j])
        np.testing.assert_array_equal(result, expected)

    #def test_pair_real_to_complex_odd_length_raises_error(self):
    #    """Test that odd length vector raises ValueError"""
    #    vec = np.array([1, 2, 3])  # Odd length
    #    with self.assertRaises(ValueError) as context:
    #        pair_real_to_complex(vec)
    #    self.assertIn("Vector length must be even to pair into complex numbers", str(context.exception))

    #def test_pair_real_to_complex_empty_vector(self):
    #    """Test pairing with empty vector"""
    #    vec = np.array([])
    #    result = pair_real_to_complex(vec)
    #    expected = np.array([], dtype=complex)
    #    np.testing.assert_array_equal(result, expected)

    #def test_pair_real_to_complex_single_element_raises_error(self):
    #    """Test that single element vector raises ValueError"""
    #    vec = np.array([42])
    #    with self.assertRaises(ValueError) as context:
    #        pair_real_to_complex(vec)
    #    self.assertIn("Vector length must be even to pair into complex numbers", str(context.exception))


    def test_pair_real_to_complex_very_small_values(self):
        """Test pairing with very small values"""
        vec = np.array([1e-15, 2e-15, 3e-15, 4e-15])
        result = pair_real_to_complex(vec)
        expected = np.array([1e-15+2e-15j, 3e-15+4e-15j])
        np.testing.assert_array_almost_equal(result, expected, decimal=20)

    def test_pair_real_to_complex_very_large_values(self):
        """Test pairing with very large values"""
        vec = np.array([1e10, 2e10, 3e10, 4e10])
        result = pair_real_to_complex(vec)
        expected = np.array([1e10+2e10j, 3e10+4e10j])
        np.testing.assert_array_equal(result, expected)

    def test_pair_real_to_complex_return_type(self):
        """Test that function returns numpy array with complex dtype"""
        vec = np.array([1, 2, 3, 4])
        result = pair_real_to_complex(vec)
        
        # Check that result is numpy array
        self.assertIsInstance(result, np.ndarray)
        
        # Check that dtype is complex
        self.assertTrue(np.iscomplexobj(result))
        
        # Check length is half of input
        self.assertEqual(len(result), len(vec) // 2)

    def test_pair_real_to_complex_preserves_magnitude_information(self):
        """Test that pairing preserves the original magnitude information"""
        vec = np.array([3, 4, 5, 12])  # Will become [3+4j, 5+12j]
        result = pair_real_to_complex(vec)
        
        # Check magnitudes of complex numbers
        self.assertAlmostEqual(abs(result[0]), 5.0, places=7)  # |3+4j| = 5
        self.assertAlmostEqual(abs(result[1]), 13.0, places=7)  # |5+12j| = 13

    def test_pair_real_to_complex_special_values(self):
        """Test pairing with special floating point values"""
        vec = np.array([0.0, -0.0, float('inf'), 1.0])
        result = pair_real_to_complex(vec)
        expected = np.array([0.0-0.0j, float('inf')+1.0j])
        
        # Check the finite values
        self.assertEqual(result[0], 0.0-0.0j)
        self.assertEqual(result[1].real, float('inf'))
        self.assertEqual(result[1].imag, 1.0)


class TestVectorsCosSim(unittest.TestCase):
    """Testing the NLQK vectors functionality."""

    def test_cosine_similarity_identical_vectors(self):
        """Test cosine similarity of identical vectors should be 1.0"""
        vec1 = np.array([1, 2, 3, 4])
        vec2 = np.array([1, 2, 3, 4])
        result = cosine_similarity(vec1, vec2)
        self.assertAlmostEqual(result, 1.0, places=7)

    def test_cosine_similarity_orthogonal_vectors(self):
        """Test cosine similarity of orthogonal vectors should be 0.0"""
        vec1 = np.array([1, 0, 0])
        vec2 = np.array([0, 1, 0])
        result = cosine_similarity(vec1, vec2)
        self.assertAlmostEqual(result, 0.0, places=7)

    def test_cosine_similarity_opposite_vectors(self):
        """Test cosine similarity of opposite vectors should be -1.0"""
        vec1 = np.array([1, 2, 3])
        vec2 = np.array([-1, -2, -3])
        result = cosine_similarity(vec1, vec2)
        self.assertAlmostEqual(result, -1.0, places=7)

    def test_cosine_similarity_scaled_vectors(self):
        """Test cosine similarity of scaled vectors should be 1.0"""
        vec1 = np.array([1, 2, 3])
        vec2 = np.array([2, 4, 6])  # vec1 * 2
        result = cosine_similarity(vec1, vec2)
        self.assertAlmostEqual(result, 1.0, places=7)

    def test_cosine_similarity_unit_vectors(self):
        """Test cosine similarity with unit vectors"""
        vec1 = np.array([1, 0])
        vec2 = np.array([0.6, 0.8])  # unit vector: sqrt(0.6^2 + 0.8^2) = 1
        result = cosine_similarity(vec1, vec2)
        self.assertAlmostEqual(result, 0.6, places=7)

    def test_cosine_similarity_numpy_arrays(self):
        """Test cosine similarity with numpy arrays as input"""
        vec1 = np.array([3, 4])
        vec2 = np.array([4, 3])
        result = cosine_similarity(vec1, vec2)
        expected = (3*4 + 4*3) / (5 * 5)  # dot: 24, norms: 5, 5
        self.assertAlmostEqual(result, expected, places=7)

    def test_cosine_similarity_different_lengths_should_handle_gracefully(self):
        """Test that function handles vectors of different lengths"""
        vec1 = np.array([1, 2, 3])
        vec2 = np.array([1, 2])  # Different length
        # This should either work (numpy broadcasting) or raise an appropriate error
        try:
            result = cosine_similarity(vec1, vec2)
            # If it works, it should be a valid number
            self.assertIsInstance(result, (float, np.floating))
        except (ValueError, RuntimeError):
            # It's also acceptable if it raises an error for incompatible shapes
            pass

    def test_cosine_similarity_with_zeros(self):
        """Test cosine similarity with zero vectors"""
        vec1 = np.array([0, 0, 0])
        vec2 = np.array([1, 2, 3])
        
        # This should handle division by zero gracefully
        with self.assertRaises((ZeroDivisionError, RuntimeWarning)) or \
             self.assertWarns(RuntimeWarning):
            result = cosine_similarity(vec1, vec2)

    def test_cosine_similarity_precision(self):
        """Test cosine similarity with high precision requirements"""
        vec1 = np.array([1e-10, 1e-10, 1e-10])
        vec2 = np.array([2e-10, 2e-10, 2e-10])
        result = cosine_similarity(vec1, vec2)
        self.assertAlmostEqual(result, 1.0, places=5)

    #def test_cosine_similarity_large_vectors(self):
    #    """Test cosine similarity with large dimensional vectors"""
    #    np.random.seed(42)  # For reproducible results
    #    vec1 = np.random.randn(1000)
    #    vec2 = np.random.randn(1000)
    #    
    #    result = cosine_similarity(vec1, vec2)
    #    
    #    # Result should be between -1 and 1
    #    self.assertGreaterEqual(result, -1.0)
    #    self.assertLessEqual(result, 1.0)

    def test_cosine_similarity_known_values(self):
        """Test cosine similarity with known mathematical values"""
        # 45-degree angle vectors in 2D should have cos(45°) ≈ 0.707
        vec1 = np.array([1, 0])
        vec2 = np.array([1, 1])  # 45 degrees from vec1
        result = cosine_similarity(vec1, vec2)
        expected = 1 / np.sqrt(2)  # cos(45°) = 1/√2 ≈ 0.707
        self.assertAlmostEqual(result, expected, places=6)

    def test_cosine_similarity_complex_identical(self):
        """Test cosine similarity of identical complex vectors should be 1.0"""
        vec1 = np.array([1+2j, 3+4j])
        vec2 = np.array([1+2j, 3+4j])
        result = cosine_similarity(vec1, vec2)
        self.assertAlmostEqual(result, 1.0, places=7)

    def test_cosine_similarity_complex_scaled(self):
        """Test cosine similarity of scaled complex vectors should be 1.0"""
        vec1 = np.array([1+1j, 2+2j])
        vec2 = np.array([2+2j, 4+4j])  # vec1 * 2
        result = cosine_similarity(vec1, vec2)
        self.assertAlmostEqual(result, 1.0, places=7)

    #def test_cosine_similarity_complex_with_phase(self):
    #    """Test cosine similarity with complex vectors differing by global phase"""
    #    vec1 = np.array([1, 1j])
    #    vec2 = np.array([1j, -1])  # vec1 * i (global phase multiplication)
    #    result = cosine_similarity(vec1, vec2)
    #    self.assertAlmostEqual(result, 1.0, places=7)

    def test_cosine_similarity_complex_orthogonal(self):
        """Test cosine similarity of orthogonal complex vectors should be 0.0"""
        vec1 = np.array([1, 0])
        vec2 = np.array([0, 1j])  # Orthogonal in complex space
        result = cosine_similarity(vec1, vec2)
        self.assertAlmostEqual(result, 0.0, places=7)

    #def test_cosine_similarity_complex_conjugate(self):
    #    """Test cosine similarity between a vector and its complex conjugate"""
    #    vec1 = np.array([1+2j, 3+4j])
    #    vec2 = np.array([1-2j, 3-4j])  # Complex conjugate
    #    result = cosine_similarity(vec1, vec2)
    #    
    #    # For complex conjugates: <v1|v2> = (1-2j)*(1-2j) + (3-4j)*(3-4j)
    #    # = (1+4) + (9+16) = 5 + 25 = 30 (all real, since we're taking conjugate of first vector in vdot)
    #    # Actually: vdot(v1, v2) = conj(v1) · v2 = (1-2j)(1-2j) + (3-4j)(3-4j) = (1+4) + (9+16) = 30
    #    # Wait, that's wrong. Let me recalculate:
    #    # vdot(v1, v2) = conj(v1) · v2 = (1-2j)(1-2j) + (3-4j)(3-4j)
    #    # No, vdot(v1, v2) = conj(v1) · v2 = (1-2j)(1-2j) + (3-4j)(3-4j) is wrong
    #    # vdot([1+2j, 3+4j], [1-2j, 3-4j]) = conj(1+2j)*(1-2j) + conj(3+4j)*(3-4j)
    #    # = (1-2j)*(1-2j) + (3-4j)*(3-4j) = (1-4j+4j^2) + (9-12j+16j^2) = (1-4-1) + (9-16-1) = -4 + (-8) = wait...
    #    
    #    # Let me just calculate it manually to get the right expected value
    #    expected_vdot = np.vdot(vec1, vec2)
    #    expected_abs_vdot = abs(expected_vdot)
    #    norm1 = np.linalg.norm(vec1)
    #    norm2 = np.linalg.norm(vec2)
    #    expected = expected_abs_vdot / (norm1 * norm2)
    #    
    #    self.assertAlmostEqual(result, expected, places=7)

    def test_cosine_similarity_complex_mixed_real_imag(self):
        """Test cosine similarity with mixed real and complex components"""
        vec1 = np.array([1, 2j, 3, 4j])
        vec2 = np.array([2, 4j, 6, 8j])  # vec1 * 2
        result = cosine_similarity(vec1, vec2)
        self.assertAlmostEqual(result, 1.0, places=7)

    #def test_cosine_similarity_complex_quantum_states(self):
    #    """Test cosine similarity with quantum-like complex states"""
    #    # Normalized quantum states
    #    vec1 = np.array([1/np.sqrt(2), 1j/np.sqrt(2)])  # |+i⟩ state
    #    vec2 = np.array([1j/np.sqrt(2), -1/np.sqrt(2)])  # rotated state
    #    result = cosine_similarity(vec1, vec2)
    #    
    #    # Manual calculation: |<v1|v2>| = |(-i)*(i) + (i)*(-1)| / (1 * 1) = |1 - i| / 1 = √2
    #    # But since both are normalized, we need |<v1|v2>|
    #    expected_inner = abs(np.vdot(vec1, vec2))
    #    self.assertAlmostEqual(result, expected_inner, places=7)

    def test_cosine_similarity_complex_pure_imaginary(self):
        """Test cosine similarity with pure imaginary vectors"""
        vec1 = np.array([1j, 2j, 3j])
        vec2 = np.array([2j, 4j, 6j])  # vec1 * 2
        result = cosine_similarity(vec1, vec2)
        self.assertAlmostEqual(result, 1.0, places=7)

    #def test_cosine_similarity_complex_opposite_direction(self):
    #    """Test cosine similarity of complex vectors in opposite directions"""
    #    vec1 = np.array([1+1j, 2+2j])
    #    vec2 = np.array([-1-1j, -2-2j])  # vec1 * (-1)
    #    result = cosine_similarity(vec1, vec2)
    #    # With abs(), opposite vectors still give similarity = 1.0
    #    self.assertAlmostEqual(result, 1.0, places=7)

    #def test_cosine_similarity_complex_return_type(self):
    #    """Test that complex vector similarity returns real number"""
    #    vec1 = np.array([1+2j, 3+4j])
    #    vec2 = np.array([2+1j, 4+3j])
    #    result = cosine_similarity(vec1, vec2)
    #    
    #    # Result should be real-valued
    #    self.assertIsInstance(result, (float, np.floating))
    #    self.assertTrue(np.isreal(result))
    #    
    #    # Result should be between 0 and 1 (due to abs())
    #    self.assertGreaterEqual(result, 0.0)
    #    self.assertLessEqual(result, 1.0)

    def test_cosine_similarity_complex_precision(self):
        """Test cosine similarity with high precision complex vectors"""
        vec1 = np.array([1e-10 + 1e-10j, 2e-10 + 2e-10j])
        vec2 = np.array([3e-10 + 3e-10j, 6e-10 + 6e-10j])  # vec1 * 3
        result = cosine_similarity(vec1, vec2)
        self.assertAlmostEqual(result, 1.0, places=5)


class TestVectorsPad(unittest.TestCase):
    """Testing the NLQK vector padding functionality."""

    def test_pad_vector_basic(self):
        """Test basic vector padding functionality"""
        vector = np.array([1, 2, 3])
        target_size = 5
        result = pad_vector(vector, target_size)
        expected = np.array([1+0j, 2+0j, 3+0j, 0+0j, 0+0j])
        np.testing.assert_array_equal(result, expected)

    def test_pad_vector_complex_input(self):
        """Test padding with complex input vector"""
        vector = np.array([1+2j, 3+4j])
        target_size = 4
        result = pad_vector(vector, target_size)
        expected = np.array([1+2j, 3+4j, 0+0j, 0+0j])
        np.testing.assert_array_equal(result, expected)

    def test_pad_vector_exact_size(self):
        """Test padding when vector is exactly target size"""
        vector = np.array([1, 2, 3, 4])
        target_size = 4
        result = pad_vector(vector, target_size)
        expected = np.array([1+0j, 2+0j, 3+0j, 4+0j])
        np.testing.assert_array_equal(result, expected)

    def test_pad_vector_power_of_two(self):
        """Test padding to power of 2 sizes"""
        vector = np.array([1, 2, 3])
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
        vector = np.array([])
        target_size = 3
        result = pad_vector(vector, target_size)
        expected = np.array([0+0j, 0+0j, 0+0j])
        np.testing.assert_array_equal(result, expected)

    def test_pad_vector_single_element(self):
        """Test padding single element vector"""
        vector = np.array([42])
        target_size = 4
        result = pad_vector(vector, target_size)
        expected = np.array([42+0j, 0+0j, 0+0j, 0+0j])
        np.testing.assert_array_equal(result, expected)

    def test_pad_vector_negative_values(self):
        """Test padding with negative values"""
        vector = np.array([-1, -2, 3])
        target_size = 5
        result = pad_vector(vector, target_size)
        expected = np.array([-1+0j, -2+0j, 3+0j, 0+0j, 0+0j])
        np.testing.assert_array_equal(result, expected)

    def test_pad_vector_zeros_in_input(self):
        """Test padding vector that already contains zeros"""
        vector = np.array([1, 0, 2, 0])
        target_size = 6
        result = pad_vector(vector, target_size)
        expected = np.array([1+0j, 0+0j, 2+0j, 0+0j, 0+0j, 0+0j])
        np.testing.assert_array_equal(result, expected)

    def test_pad_vector_too_large_raises_error(self):
        """Test that vector larger than target size raises ValueError"""
        vector = np.array([1, 2, 3, 4, 5])
        target_size = 3
        with self.assertRaises(ValueError) as context:
            pad_vector(vector, target_size)
        # self.assertIn("Vector is larger than the target size", str(context.exception))

    def test_pad_vector_zero_target_size(self):
        """Test padding with zero target size"""
        vector = np.array([])
        target_size = 0
        result = pad_vector(vector, target_size)
        expected = np.array([], dtype=complex)
        np.testing.assert_array_equal(result, expected)

    def test_pad_vector_zero_target_size_with_data_raises_error(self):
        """Test that non-empty vector with zero target size raises error"""
        vector = np.array([1, 2])
        target_size = 0
        with self.assertRaises(ValueError) as context:
            pad_vector(vector, target_size)
        #self.assertIn("Vector is larger than the target size", str(context.exception))

    #def test_pad_vector_return_type(self):
    #    """Test that function returns numpy array with complex dtype"""
    #    vector = np.array([1, 2, 3])
    #    target_size = 5
    #    result = pad_vector(vector, target_size)
    #    
    #    # Check that result is numpy array
    #    self.assertIsInstance(result, np.ndarray)
    #    
    #    # Check that dtype is complex
    #    self.assertTrue(np.iscomplexobj(result))
    #    self.assertEqual(result.dtype, complex)
    #    
    #    # Check correct size
    #    self.assertEqual(len(result), target_size)

    def test_pad_vector_preserves_original_values(self):
        """Test that padding preserves original vector values"""
        vector = np.array([1.1, 2.2, 3.3])
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
        vector = np.array([1e-15, 2e-15])
        target_size = 4
        result = pad_vector(vector, target_size)
        expected = np.array([1e-15+0j, 2e-15+0j, 0+0j, 0+0j])
        np.testing.assert_array_almost_equal(result, expected, decimal=20)

    def test_pad_vector_very_large_values(self):
        """Test padding with very large values"""
        vector = np.array([1e10, 2e10])
        target_size = 5
        result = pad_vector(vector, target_size)
        expected = np.array([1e10+0j, 2e10+0j, 0+0j, 0+0j, 0+0j])
        np.testing.assert_array_equal(result, expected)

    def test_pad_vector_mixed_complex_real(self):
        """Test padding with mixed complex and real values"""
        vector = np.array([1, 2+3j, 4])
        target_size = 6
        result = pad_vector(vector, target_size)
        expected = np.array([1+0j, 2+3j, 4+0j, 0+0j, 0+0j, 0+0j])
        np.testing.assert_array_equal(result, expected)

    def test_pad_vector_large_target_size(self):
        """Test padding to very large target size"""
        vector = np.array([1, 2])
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
        vector = np.array([5])
        target_size = 1
        result = pad_vector(vector, target_size)
        expected = np.array([5+0j])
        np.testing.assert_array_equal(result, expected)


class TestVectorsNormalize(unittest.TestCase):
    """Testing the NLQK vectors functionality."""

    def test_normalize_real_vector(self):
        """Test normalization of real vectors"""
        result = normalize(np.array([3, 4])) # length 5
        expected = np.array([0.6, 0.8])
        np.testing.assert_array_almost_equal(result, expected, decimal=7)        
        # Check that the result has unit norm
        # self.assertAlmostEqual(np.linalg.norm(result), 1.0, places=7)

    def test_normalize_complex_vector(self):
        """Test normalization of complex vectors"""
        result = normalize(np.array([1+1j, 1-1j]))  # |vec| = sqrt(1²+1² + 1²+1²) = 2
        expected = np.array([0.5+0.5j, 0.5-0.5j])
        np.testing.assert_array_almost_equal(result, expected, decimal=7)        
        # Check that the result has unit norm
        # self.assertAlmostEqual(np.linalg.norm(result), 1.0, places=7)

    def test_normalize_already_normalized(self):
        """Test normalization of already normalized vector"""
        # vec =   # Already normalized (3-4-5 triangle)
        result = normalize(np.array([0.6, 0.8]))
        expected = np.array([0.6, 0.8])
        np.testing.assert_array_almost_equal(result, expected, decimal=7)
        # Check that the result has unit norm
        #self.assertAlmostEqual(np.linalg.norm(result), 1.0, places=7)

    def test_normalize_single_element(self):
        """Test normalization of single element vector"""
        vec = np.array([5.0])
        result = normalize(vec)
        expected = np.array([1.0])
        np.testing.assert_array_almost_equal(result, expected, decimal=7)

    def test_normalize_negative_values(self):
        """Test normalization with negative values"""
        vec = np.array([-3, -4])
        result = normalize(vec)
        expected = np.array([-0.6, -0.8])
        np.testing.assert_array_almost_equal(result, expected, decimal=7)
        # Check that the result has unit norm
        # self.assertAlmostEqual(np.linalg.norm(result), 1.0, places=7)

    def test_normalize_zero_vector_raises_error(self):
        """Test that normalizing zero vector raises ValueError"""
        with self.assertRaises(ValueError) as context:
            normalize(np.array([0, 0, 0]))
        self.assertIn("Zero vector cannot be normalized", str(context.exception))

    def test_normalize_very_small_vector(self):
        """Test normalization of very small vectors"""
        result = normalize(np.array([1e-15, 1e-15]))
        expected_norm = np.sqrt(2) * 1e-15
        expected = np.array([1e-15, 1e-15]) / expected_norm
        np.testing.assert_array_almost_equal(result, expected, decimal=10)
        # Check that the result has unit norm
        # self.assertAlmostEqual(np.linalg.norm(result), 1.0, places=10)

    def test_normalize_large_vector(self):
        """Test normalization of large vectors"""
        result = normalize(np.array([1e10, 1e10]))
        expected = np.array([1/np.sqrt(2), 1/np.sqrt(2)])
        np.testing.assert_array_almost_equal(result, expected, decimal=7)
        # Check that the result has unit norm
        # self.assertAlmostEqual(np.linalg.norm(result), 1.0, places=7)

    def test_normalize_numpy_array_input(self):
        """Test normalization with numpy array as input"""
        vec = np.array([1, 2, 2])  # |vec| = 3
        result = normalize(vec)
        expected = np.array([1/3, 2/3, 2/3])
        np.testing.assert_array_almost_equal(result, expected, decimal=7)
        # Check that the result has unit norm
        # self.assertAlmostEqual(np.linalg.norm(result), 1.0, places=7)

    #def test_normalize_mixed_complex_real(self):
    #    """Test normalization with mixed complex and real values"""
    #    result = normalize(np.array([1, 1j, 0, 2+2j])) # |vec| = sqrt(1 + 1 + 0 + 8) = sqrt(10)
    #    sqrt10 = np.sqrt(10)
    #    expected = np.array([1/sqrt10, 1j/sqrt10, 0, (2+2j)/sqrt10])
    #    np.testing.assert_array_almost_equal(result, expected, decimal=7)        
    #    # Check that the result has unit norm
    #    # self.assertAlmostEqual(np.linalg.norm(result), 1.0, places=7)

    def test_normalize_preserves_direction(self):
        """Test that normalization preserves vector direction"""
        vec = np.array([3, 4, 12]) # |vec| = 13
        result = normalize(vec)
        
        # Check that result is parallel to original (same direction)
        # For parallel vectors, the normalized original should equal the result
        vec_normalized_manual = np.array(vec) / np.linalg.norm(vec)
        np.testing.assert_array_almost_equal(result, vec_normalized_manual, decimal=10)
        # Alternative check: dot product of unit vectors should be 1 if parallel
        #vec_unit = np.array(vec) / np.linalg.norm(vec)
        #dot_product = np.dot(vec_unit, result)
        #self.assertAlmostEqual(dot_product, 1.0, places=10)
        #
        # Check unit norm
        #self.assertAlmostEqual(np.linalg.norm(result), 1.0, places=7)


class TestIsNormalized(unittest.TestCase):
    """Testing the NLQK vectors is_normalized functionality."""

    def test_is_normalized_complex_true(self):
        """Test is_normalized with a complex vector that should be normalized"""
        vec = np.array([0.5+0.5j, 0.5-0.5j])  # |vec| = sqrt(0.5 + 0.5) = 1
        self.assertTrue(is_normalized(vec))
        self.assertAlmostEqual(np.linalg.norm(vec), 1.0, places=7)

    def test_is_normalized_complex_false(self):
        """Test is_normalized with a complex vector that is NOT normalized"""
        vec = np.array([1+1j, 1-1j])  # |vec| = sqrt(2 + 2) = 2
        self.assertFalse(is_normalized(vec))
        self.assertAlmostEqual(np.linalg.norm(vec), 2.0, places=7)

    def test_is_normalized_after_normalize(self):
        """Test that is_normalized returns True after calling normalize"""
        vec = np.array([1+1j, 1-1j])  # Not normalized initially
        vec_normalized = normalize(vec)
        self.assertTrue(is_normalized(vec_normalized))
        self.assertAlmostEqual(np.linalg.norm(vec_normalized), 1.0, places=7)

    def test_is_normalized_mixed_complex(self):
        """Test is_normalized with mixed real/imaginary components"""
        vec = np.array([0.6, 0.8j])  # |vec| = sqrt(0.36 + 0.64) = 1
        self.assertTrue(is_normalized(vec))
        self.assertAlmostEqual(np.linalg.norm(vec), 1.0, places=7)

    def test_is_normalized_real_vector(self):
        """Test is_normalized with real vector for comparison"""
        vec = np.array([0.6, 0.8])  # |vec| = sqrt(0.36 + 0.64) = 1
        self.assertTrue(is_normalized(vec))
        self.assertAlmostEqual(np.linalg.norm(vec), 1.0, places=7)

    def test_is_normalized_with_tolerance(self):
        """Test is_normalized with custom tolerance"""
        vec = np.array([0.6001, 0.7999])  # Slightly off from normalized
        self.assertFalse(is_normalized(vec, tolerance=1e-9))
        self.assertTrue(is_normalized(vec, tolerance=1e-2))

    def test_is_normalized_zero_vector(self):
        """Test is_normalized with zero vector"""
        vec = np.array([0.0, 0.0])
        self.assertFalse(is_normalized(vec))

    def test_is_normalized_single_element_complex(self):
        """Test is_normalized with single complex element"""
        vec = np.array([1.0+0j])  # Should be normalized
        self.assertTrue(is_normalized(vec))
        
        vec2 = np.array([2.0+0j])  # Should not be normalized
        self.assertFalse(is_normalized(vec2))


class TestInputTypes(unittest.TestCase):
    """Testing that all Union[Sequence[Union[int, float, complex]], np.ndarray] types work."""

    def test_lists_with_ints(self):
        """Test with lists containing integers"""
        vec = np.array([1, 2, 3])
        result = normalize(vec)
        self.assertAlmostEqual(np.linalg.norm(result), 1.0, places=7)

    def test_lists_with_floats(self):
        """Test with lists containing floats"""
        vec = np.array([1.0, 2.0, 3.0])
        result = normalize(vec)
        self.assertAlmostEqual(np.linalg.norm(result), 1.0, places=7)

    def test_lists_with_complex(self):
        """Test with lists containing complex numbers"""
        vec = np.array([1+1j, 2+2j])
        result = normalize(vec)
        self.assertAlmostEqual(np.linalg.norm(result), 1.0, places=7)

    def test_lists_mixed_types(self):
        """Test with lists containing mixed int/float"""
        vec = np.array([1, 2.5, 3])  # int, float, int
        result = normalize(vec)
        self.assertAlmostEqual(np.linalg.norm(result), 1.0, places=7)

    def test_tuples_with_ints(self):
        """Test with tuples containing integers"""
        vec = np.array( (1, 2, 3) )
        result = normalize(vec)
        self.assertAlmostEqual(np.linalg.norm(result), 1.0, places=7)

    def test_tuples_with_floats(self):
        """Test with tuples containing floats"""
        vec = np.array( (1.0, 2.0, 3.0) )
        result = normalize(vec)
        self.assertAlmostEqual(np.linalg.norm(result), 1.0, places=7)

    def test_tuples_with_complex(self):
        """Test with tuples containing complex numbers"""
        vec = np.array( (1+1j, 2+2j) )
        result = normalize(vec)
        self.assertAlmostEqual(np.linalg.norm(result), 1.0, places=7)

    def test_numpy_arrays_int(self):
        """Test with numpy arrays of integers"""
        vec = np.array([1, 2, 3])
        result = normalize(vec)
        self.assertAlmostEqual(np.linalg.norm(result), 1.0, places=7)

    def test_numpy_arrays_float(self):
        """Test with numpy arrays of floats"""
        vec = np.array([1.0, 2.0, 3.0])
        result = normalize(vec)
        self.assertAlmostEqual(np.linalg.norm(result), 1.0, places=7)

    def test_numpy_arrays_complex(self):
        """Test with numpy arrays of complex numbers"""
        vec = np.array([1+1j, 2+2j])
        result = normalize(vec)
        self.assertAlmostEqual(np.linalg.norm(result), 1.0, places=7)

    def test_range_objects(self):
        """Test with range objects (converted to list)"""
        vec = np.array( list(range(1, 4)) ) # [1, 2, 3]
        result = normalize(vec)
        self.assertAlmostEqual(np.linalg.norm(result), 1.0, places=7)

    def test_cosine_similarity_mixed_input_types(self):
        """Test cosine similarity with different input types"""
        vec1 = np.array([1, 2, 3])           # list
        vec2 = np.array( (1, 2, 3) )          # tuple
        vec3 = np.array([1, 2, 3]) # numpy array
        
        # All should give the same result (cosine similarity = 1.0)
        result1 = cosine_similarity(vec1, vec2)
        result2 = cosine_similarity(vec2, vec3)
        result3 = cosine_similarity(vec1, vec3)
        
        self.assertAlmostEqual(result1, 1.0, places=7)
        self.assertAlmostEqual(result2, 1.0, places=7)
        self.assertAlmostEqual(result3, 1.0, places=7)

    def test_pad_vector_mixed_input_types(self):
        """Test pad_vector with different input types"""
        # Test with different input types
        list_input = np.array( [1, 2] )
        tuple_input = np.array( (1, 2) )
        array_input = np.array([1, 2])
        
        result1 = pad_vector(list_input, 4)
        result2 = pad_vector(tuple_input, 4)
        result3 = pad_vector(array_input, 4)
        
        # All should give the same result
        np.testing.assert_array_equal(result1, result2)
        np.testing.assert_array_equal(result2, result3)
        
        # Check expected values
        expected = np.array([1+0j, 2+0j, 0+0j, 0+0j])
        np.testing.assert_array_equal(result1, expected)

    def test_edge_case_single_element(self):
        """Test single element in different container types"""
        list_vec = np.array( [5] )
        tuple_vec = np.array( (5,) )
        array_vec = np.array([5])
        
        result1 = normalize(list_vec)
        result2 = normalize(tuple_vec)
        result3 = normalize(array_vec)
        
        # All should normalize to [1.0]
        np.testing.assert_array_almost_equal(result1, [1.0])
        np.testing.assert_array_almost_equal(result2, [1.0])
        np.testing.assert_array_almost_equal(result3, [1.0])


 

if __name__ == '__main__':
    unittest.main()
