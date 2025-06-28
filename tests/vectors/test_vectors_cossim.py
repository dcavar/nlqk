#!/usr/bin/env python3

"""
test_vectors.py

(C) 2025 by [Damir Cavar](https://damir.cavar.me/) and [NLP Lab](https://nlp-lab.org/)

Testing the NLQK vectors functionality.
"""

import unittest
import numpy as np
from nlqk.vectors import cosine_similarity


class TestVectorsCosSim(unittest.TestCase):
    """Testing the NLQK vectors functionality."""

    def test_cosine_similarity_identical_vectors(self):
        """Test cosine similarity of identical vectors should be 1.0"""
        vec1 = [1, 2, 3, 4]
        vec2 = [1, 2, 3, 4]
        result = cosine_similarity(vec1, vec2)
        self.assertAlmostEqual(result, 1.0, places=7)

    def test_cosine_similarity_orthogonal_vectors(self):
        """Test cosine similarity of orthogonal vectors should be 0.0"""
        vec1 = [1, 0, 0]
        vec2 = [0, 1, 0]
        result = cosine_similarity(vec1, vec2)
        self.assertAlmostEqual(result, 0.0, places=7)

    def test_cosine_similarity_opposite_vectors(self):
        """Test cosine similarity of opposite vectors should be -1.0"""
        vec1 = [1, 2, 3]
        vec2 = [-1, -2, -3]
        result = cosine_similarity(vec1, vec2)
        self.assertAlmostEqual(result, -1.0, places=7)

    def test_cosine_similarity_scaled_vectors(self):
        """Test cosine similarity of scaled vectors should be 1.0"""
        vec1 = [1, 2, 3]
        vec2 = [2, 4, 6]  # vec1 * 2
        result = cosine_similarity(vec1, vec2)
        self.assertAlmostEqual(result, 1.0, places=7)

    def test_cosine_similarity_unit_vectors(self):
        """Test cosine similarity with unit vectors"""
        vec1 = [1, 0]
        vec2 = [0.6, 0.8]  # unit vector: sqrt(0.6^2 + 0.8^2) = 1
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
        vec1 = [1, 2, 3]
        vec2 = [1, 2]  # Different length
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
        vec1 = [0, 0, 0]
        vec2 = [1, 2, 3]
        
        # This should handle division by zero gracefully
        with self.assertRaises((ZeroDivisionError, RuntimeWarning)) or \
             self.assertWarns(RuntimeWarning):
            result = cosine_similarity(vec1, vec2)

    def test_cosine_similarity_precision(self):
        """Test cosine similarity with high precision requirements"""
        vec1 = [1e-10, 1e-10, 1e-10]
        vec2 = [2e-10, 2e-10, 2e-10]
        result = cosine_similarity(vec1, vec2)
        self.assertAlmostEqual(result, 1.0, places=5)

    def test_cosine_similarity_large_vectors(self):
        """Test cosine similarity with large dimensional vectors"""
        np.random.seed(42)  # For reproducible results
        vec1 = np.random.randn(1000)
        vec2 = np.random.randn(1000)
        
        result = cosine_similarity(vec1, vec2)
        
        # Result should be between -1 and 1
        self.assertGreaterEqual(result, -1.0)
        self.assertLessEqual(result, 1.0)

    def test_cosine_similarity_known_values(self):
        """Test cosine similarity with known mathematical values"""
        # 45-degree angle vectors in 2D should have cos(45°) ≈ 0.707
        vec1 = [1, 0]
        vec2 = [1, 1]  # 45 degrees from vec1
        result = cosine_similarity(vec1, vec2)
        expected = 1 / np.sqrt(2)  # cos(45°) = 1/√2 ≈ 0.707
        self.assertAlmostEqual(result, expected, places=6)

 