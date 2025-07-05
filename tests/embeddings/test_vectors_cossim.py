#!/usr/bin/env python3

"""
test_vectors.py

(C) 2025 by [Damir Cavar](https://damir.cavar.me/) and [NLP Lab](https://nlp-lab.org/)

Testing the NLQK vectors functionality.
"""

import sys
sys.path.append('.') # ./..')
import unittest
import numpy as np
from nlqk.embeddings import cosine_similarity


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

    def test_cosine_similarity_complex_identical(self):
        """Test cosine similarity of identical complex vectors should be 1.0"""
        vec1 = [1+2j, 3+4j]
        vec2 = [1+2j, 3+4j]
        result = cosine_similarity(vec1, vec2)
        self.assertAlmostEqual(result, 1.0, places=7)

    def test_cosine_similarity_complex_scaled(self):
        """Test cosine similarity of scaled complex vectors should be 1.0"""
        vec1 = [1+1j, 2+2j]
        vec2 = [2+2j, 4+4j]  # vec1 * 2
        result = cosine_similarity(vec1, vec2)
        self.assertAlmostEqual(result, 1.0, places=7)

    def test_cosine_similarity_complex_with_phase(self):
        """Test cosine similarity with complex vectors differing by global phase"""
        vec1 = [1, 1j]
        vec2 = [1j, -1]  # vec1 * i (global phase multiplication)
        result = cosine_similarity(vec1, vec2)
        self.assertAlmostEqual(result, 1.0, places=7)

    def test_cosine_similarity_complex_orthogonal(self):
        """Test cosine similarity of orthogonal complex vectors should be 0.0"""
        vec1 = [1, 0]
        vec2 = [0, 1j]  # Orthogonal in complex space
        result = cosine_similarity(vec1, vec2)
        self.assertAlmostEqual(result, 0.0, places=7)

    def test_cosine_similarity_complex_conjugate(self):
        """Test cosine similarity between a vector and its complex conjugate"""
        vec1 = [1+2j, 3+4j]
        vec2 = [1-2j, 3-4j]  # Complex conjugate
        result = cosine_similarity(vec1, vec2)
        
        # For complex conjugates: <v1|v2> = (1-2j)*(1-2j) + (3-4j)*(3-4j)
        # = (1+4) + (9+16) = 5 + 25 = 30 (all real, since we're taking conjugate of first vector in vdot)
        # Actually: vdot(v1, v2) = conj(v1) · v2 = (1-2j)(1-2j) + (3-4j)(3-4j) = (1+4) + (9+16) = 30
        # Wait, that's wrong. Let me recalculate:
        # vdot(v1, v2) = conj(v1) · v2 = (1-2j)(1-2j) + (3-4j)(3-4j)
        # No, vdot(v1, v2) = conj(v1) · v2 = (1-2j)(1-2j) + (3-4j)(3-4j) is wrong
        # vdot([1+2j, 3+4j], [1-2j, 3-4j]) = conj(1+2j)*(1-2j) + conj(3+4j)*(3-4j)
        # = (1-2j)*(1-2j) + (3-4j)*(3-4j) = (1-4j+4j^2) + (9-12j+16j^2) = (1-4-1) + (9-16-1) = -4 + (-8) = wait...
        
        # Let me just calculate it manually to get the right expected value
        expected_vdot = np.vdot(vec1, vec2)
        expected_abs_vdot = abs(expected_vdot)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        expected = expected_abs_vdot / (norm1 * norm2)
        
        self.assertAlmostEqual(result, expected, places=7)

    def test_cosine_similarity_complex_mixed_real_imag(self):
        """Test cosine similarity with mixed real and complex components"""
        vec1 = [1, 2j, 3, 4j]
        vec2 = [2, 4j, 6, 8j]  # vec1 * 2
        result = cosine_similarity(vec1, vec2)
        self.assertAlmostEqual(result, 1.0, places=7)

    def test_cosine_similarity_complex_quantum_states(self):
        """Test cosine similarity with quantum-like complex states"""
        # Normalized quantum states
        vec1 = np.array([1/np.sqrt(2), 1j/np.sqrt(2)])  # |+i⟩ state
        vec2 = np.array([1j/np.sqrt(2), -1/np.sqrt(2)])  # rotated state
        result = cosine_similarity(vec1, vec2)
        
        # Manual calculation: |<v1|v2>| = |(-i)*(i) + (i)*(-1)| / (1 * 1) = |1 - i| / 1 = √2
        # But since both are normalized, we need |<v1|v2>|
        expected_inner = abs(np.vdot(vec1, vec2))
        self.assertAlmostEqual(result, expected_inner, places=7)

    def test_cosine_similarity_complex_pure_imaginary(self):
        """Test cosine similarity with pure imaginary vectors"""
        vec1 = [1j, 2j, 3j]
        vec2 = [2j, 4j, 6j]  # vec1 * 2
        result = cosine_similarity(vec1, vec2)
        self.assertAlmostEqual(result, 1.0, places=7)

    def test_cosine_similarity_complex_opposite_direction(self):
        """Test cosine similarity of complex vectors in opposite directions"""
        vec1 = [1+1j, 2+2j]
        vec2 = [-1-1j, -2-2j]  # vec1 * (-1)
        result = cosine_similarity(vec1, vec2)
        # With abs(), opposite vectors still give similarity = 1.0
        self.assertAlmostEqual(result, 1.0, places=7)

    def test_cosine_similarity_complex_return_type(self):
        """Test that complex vector similarity returns real number"""
        vec1 = [1+2j, 3+4j]
        vec2 = [2+1j, 4+3j]
        result = cosine_similarity(vec1, vec2)
        
        # Result should be real-valued
        self.assertIsInstance(result, (float, np.floating))
        self.assertTrue(np.isreal(result))
        
        # Result should be between 0 and 1 (due to abs())
        self.assertGreaterEqual(result, 0.0)
        self.assertLessEqual(result, 1.0)

    def test_cosine_similarity_complex_precision(self):
        """Test cosine similarity with high precision complex vectors"""
        vec1 = [1e-10 + 1e-10j, 2e-10 + 2e-10j]
        vec2 = [3e-10 + 3e-10j, 6e-10 + 6e-10j]  # vec1 * 3
        result = cosine_similarity(vec1, vec2)
        self.assertAlmostEqual(result, 1.0, places=5)

 