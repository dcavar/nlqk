#!/usr/bin/env python3

"""
test_is_normalized.py

Testing the NLQK vectors is_normalized functionality.
"""

import unittest
import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from nlqk.embeddings import is_normalized, normalize

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

if __name__ == "__main__":
    unittest.main()
