#!/usr/bin/env python3

"""
test_input_types.py

Testing that all declared input types work correctly with NLQK vector functions.
"""

import sys
sys.path.append('.') # ./..')
import unittest
import numpy as np
import os
# sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from nlqk.embeddings.vectors import cosine_similarity, normalize, pad_vector

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

if __name__ == "__main__":
    unittest.main()
