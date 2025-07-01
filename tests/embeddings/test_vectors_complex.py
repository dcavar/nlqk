#!/usr/bin/env python3

"""
test_vectors.py

(C) 2025 by [Damir Cavar](https://damir.cavar.me/) and [NLP Lab](https://nlp-lab.org/)

Testing the NLQK vectors functionality.
"""

import unittest
import numpy as np
from nlqk.embeddings import pair_real_to_complex


class TestVectorsComplex(unittest.TestCase):
    """Testing the NLQK vectors functionality."""

    def test_pair_real_to_complex_basic(self):
        """Test basic functionality of pairing real values to complex"""
        vec = [1, 2, 3, 4]
        result = pair_real_to_complex(vec)
        expected = np.array([1+2j, 3+4j])
        np.testing.assert_array_equal(result, expected)

    def test_pair_real_to_complex_zeros(self):
        """Test pairing with zeros"""
        vec = [1, 0, 0, 3]
        result = pair_real_to_complex(vec)
        expected = np.array([1+0j, 0+3j])
        np.testing.assert_array_equal(result, expected)

    def test_pair_real_to_complex_negative_values(self):
        """Test pairing with negative values"""
        vec = [-1, 2, 3, -4]
        result = pair_real_to_complex(vec)
        expected = np.array([-1+2j, 3-4j])
        np.testing.assert_array_equal(result, expected)

    def test_pair_real_to_complex_floats(self):
        """Test pairing with floating point values"""
        vec = [1.5, 2.7, -3.2, 4.8]
        result = pair_real_to_complex(vec)
        expected = np.array([1.5+2.7j, -3.2+4.8j])
        np.testing.assert_array_almost_equal(result, expected, decimal=7)

    def test_pair_real_to_complex_two_elements(self):
        """Test pairing with minimum valid input (2 elements)"""
        vec = [5, 7]
        result = pair_real_to_complex(vec)
        expected = np.array([5+7j])
        np.testing.assert_array_equal(result, expected)

    def test_pair_real_to_complex_large_vector(self):
        """Test pairing with larger vector"""
        vec = [1, 2, 3, 4, 5, 6, 7, 8]
        result = pair_real_to_complex(vec)
        expected = np.array([1+2j, 3+4j, 5+6j, 7+8j])
        np.testing.assert_array_equal(result, expected)

    def test_pair_real_to_complex_numpy_array_input(self):
        """Test pairing with numpy array as input"""
        vec = np.array([10, 20, 30, 40])
        result = pair_real_to_complex(vec)
        expected = np.array([10+20j, 30+40j])
        np.testing.assert_array_equal(result, expected)

    def test_pair_real_to_complex_odd_length_raises_error(self):
        """Test that odd length vector raises ValueError"""
        vec = [1, 2, 3]  # Odd length
        with self.assertRaises(ValueError) as context:
            pair_real_to_complex(vec)
        self.assertIn("Vector length must be even to pair into complex numbers", str(context.exception))

    def test_pair_real_to_complex_empty_vector(self):
        """Test pairing with empty vector"""
        vec = []
        result = pair_real_to_complex(vec)
        expected = np.array([], dtype=complex)
        np.testing.assert_array_equal(result, expected)

    def test_pair_real_to_complex_single_element_raises_error(self):
        """Test that single element vector raises ValueError"""
        vec = [42]
        with self.assertRaises(ValueError) as context:
            pair_real_to_complex(vec)
        self.assertIn("Vector length must be even to pair into complex numbers", str(context.exception))

    def test_pair_real_to_complex_very_small_values(self):
        """Test pairing with very small values"""
        vec = [1e-15, 2e-15, 3e-15, 4e-15]
        result = pair_real_to_complex(vec)
        expected = np.array([1e-15+2e-15j, 3e-15+4e-15j])
        np.testing.assert_array_almost_equal(result, expected, decimal=20)

    def test_pair_real_to_complex_very_large_values(self):
        """Test pairing with very large values"""
        vec = [1e10, 2e10, 3e10, 4e10]
        result = pair_real_to_complex(vec)
        expected = np.array([1e10+2e10j, 3e10+4e10j])
        np.testing.assert_array_equal(result, expected)

    def test_pair_real_to_complex_return_type(self):
        """Test that function returns numpy array with complex dtype"""
        vec = [1, 2, 3, 4]
        result = pair_real_to_complex(vec)
        
        # Check that result is numpy array
        self.assertIsInstance(result, np.ndarray)
        
        # Check that dtype is complex
        self.assertTrue(np.iscomplexobj(result))
        
        # Check length is half of input
        self.assertEqual(len(result), len(vec) // 2)

    def test_pair_real_to_complex_preserves_magnitude_information(self):
        """Test that pairing preserves the original magnitude information"""
        vec = [3, 4, 5, 12]  # Will become [3+4j, 5+12j]
        result = pair_real_to_complex(vec)
        
        # Check magnitudes of complex numbers
        self.assertAlmostEqual(abs(result[0]), 5.0, places=7)  # |3+4j| = 5
        self.assertAlmostEqual(abs(result[1]), 13.0, places=7)  # |5+12j| = 13

    def test_pair_real_to_complex_special_values(self):
        """Test pairing with special floating point values"""
        vec = [0.0, -0.0, float('inf'), 1.0]
        result = pair_real_to_complex(vec)
        expected = np.array([0.0-0.0j, float('inf')+1.0j])
        
        # Check the finite values
        self.assertEqual(result[0], 0.0-0.0j)
        self.assertEqual(result[1].real, float('inf'))
        self.assertEqual(result[1].imag, 1.0)


if __name__ == '__main__':
    unittest.main()
