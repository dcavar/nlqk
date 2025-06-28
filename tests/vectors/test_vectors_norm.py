#!/usr/bin/env python3

"""
test_vectors.py

(C) 2025 by [Damir Cavar](https://damir.cavar.me/) and [NLP Lab](https://nlp-lab.org/)

Testing the NLQK vectors functionality.
"""

import unittest
import numpy as np
from nlqk.vectors import normalize

class TestVectorsNormalize(unittest.TestCase):
    """Testing the NLQK vectors functionality."""

    def test_normalize_real_vector(self):
        """Test normalization of real vectors"""
        vec = [3, 4]  # Length is 5
        result = normalize(vec)
        expected = np.array([0.6, 0.8])
        np.testing.assert_array_almost_equal(result, expected, decimal=7)
        
        # Check that the result has unit norm
        self.assertAlmostEqual(np.linalg.norm(result), 1.0, places=7)

    def test_normalize_complex_vector(self):
        """Test normalization of complex vectors"""
        vec = [1+1j, 1-1j]  # |vec| = sqrt(1²+1² + 1²+1²) = 2
        result = normalize(vec)
        expected = np.array([0.5+0.5j, 0.5-0.5j])
        np.testing.assert_array_almost_equal(result, expected, decimal=7)
        
        # Check that the result has unit norm
        self.assertAlmostEqual(np.linalg.norm(result), 1.0, places=7)

    def test_normalize_already_normalized(self):
        """Test normalization of already normalized vector"""
        vec = [0.6, 0.8]  # Already normalized (3-4-5 triangle)
        result = normalize(vec)
        expected = np.array([0.6, 0.8])
        np.testing.assert_array_almost_equal(result, expected, decimal=7)
        
        # Check that the result has unit norm
        self.assertAlmostEqual(np.linalg.norm(result), 1.0, places=7)

    def test_normalize_single_element(self):
        """Test normalization of single element vector"""
        vec = [5.0]
        result = normalize(vec)
        expected = np.array([1.0])
        np.testing.assert_array_almost_equal(result, expected, decimal=7)

    def test_normalize_negative_values(self):
        """Test normalization with negative values"""
        vec = [-3, -4]
        result = normalize(vec)
        expected = np.array([-0.6, -0.8])
        np.testing.assert_array_almost_equal(result, expected, decimal=7)
        
        # Check that the result has unit norm
        self.assertAlmostEqual(np.linalg.norm(result), 1.0, places=7)

    def test_normalize_zero_vector_raises_error(self):
        """Test that normalizing zero vector raises ValueError"""
        vec = [0, 0, 0]
        with self.assertRaises(ValueError) as context:
            normalize(vec)
        self.assertIn("Zero vector cannot be normalized", str(context.exception))

    def test_normalize_very_small_vector(self):
        """Test normalization of very small vectors"""
        vec = [1e-15, 1e-15]
        result = normalize(vec)
        expected_norm = np.sqrt(2) * 1e-15
        expected = np.array([1e-15, 1e-15]) / expected_norm
        np.testing.assert_array_almost_equal(result, expected, decimal=10)
        
        # Check that the result has unit norm
        self.assertAlmostEqual(np.linalg.norm(result), 1.0, places=10)

    def test_normalize_large_vector(self):
        """Test normalization of large vectors"""
        vec = [1e10, 1e10]
        result = normalize(vec)
        expected = np.array([1/np.sqrt(2), 1/np.sqrt(2)])
        np.testing.assert_array_almost_equal(result, expected, decimal=7)
        
        # Check that the result has unit norm
        self.assertAlmostEqual(np.linalg.norm(result), 1.0, places=7)

    def test_normalize_numpy_array_input(self):
        """Test normalization with numpy array as input"""
        vec = np.array([1, 2, 2])  # |vec| = 3
        result = normalize(vec)
        expected = np.array([1/3, 2/3, 2/3])
        np.testing.assert_array_almost_equal(result, expected, decimal=7)
        
        # Check that the result has unit norm
        self.assertAlmostEqual(np.linalg.norm(result), 1.0, places=7)

    def test_normalize_mixed_complex_real(self):
        """Test normalization with mixed complex and real values"""
        vec = [1, 1j, 0, 2+2j]  # |vec| = sqrt(1 + 1 + 0 + 8) = sqrt(10)
        result = normalize(vec)
        sqrt10 = np.sqrt(10)
        expected = np.array([1/sqrt10, 1j/sqrt10, 0, (2+2j)/sqrt10])
        np.testing.assert_array_almost_equal(result, expected, decimal=7)
        
        # Check that the result has unit norm
        self.assertAlmostEqual(np.linalg.norm(result), 1.0, places=7)

    def test_normalize_preserves_direction(self):
        """Test that normalization preserves vector direction"""
        vec = [3, 4, 12]  # |vec| = 13
        result = normalize(vec)
        
        # Check that result is parallel to original (same direction)
        # For parallel vectors, the normalized original should equal the result
        vec_normalized_manual = np.array(vec) / np.linalg.norm(vec)
        np.testing.assert_array_almost_equal(result, vec_normalized_manual, decimal=10)
        
        # Alternative check: dot product of unit vectors should be 1 if parallel
        vec_unit = np.array(vec) / np.linalg.norm(vec)
        dot_product = np.dot(vec_unit, result)
        self.assertAlmostEqual(dot_product, 1.0, places=10)
        
        # Check unit norm
        self.assertAlmostEqual(np.linalg.norm(result), 1.0, places=7)

 