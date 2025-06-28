#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
embedding.py

(C) 2025 Damir Cavar

"""


import numpy as np


def is_normalized(vector: np.ndarray, tolerance: float=1e-9) -> np.bool_:
    """
    Checks if a NumPy vector is normalized (its L2 norm is approximately 1).

    Args:
        vector (np.ndarray): The input NumPy vector.
        tolerance (float): The allowed tolerance for comparison with 1.

    Returns:
        bool: True if the vector is normalized, False otherwise.
    """
    norm = np.linalg.norm(vector)
    return np.isclose(norm, 1.0, atol=tolerance)


def normalize(v: np.ndarray) -> np.ndarray:
	"""Normalize the vector to have a length of 1."""
	return v / np.linalg.norm(v)


def pad_vectors(vectors: np.ndarray, size: int) -> np.ndarray:
    """Pad rows with zeros to size."""
    return np.pad(vectors, [(0, 0), (0, size - vectors.shape[1])], mode='constant')


def pad_vector(v: np.ndarray, size: int) -> np.ndarray:
    """Pad the vector with zeros to size."""
    return np.pad(v, (0, 9 - v.shape[0]), mode='constant')

