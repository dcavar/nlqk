
"""

"""

import numpy as np


def cosine_similarity(vec1, vec2, tol=1e-12):
    """Computes cosine similarity between two vectors."""
    vec1, vec2 = np.array(vec1), np.array(vec2)
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 < tol or norm2 < tol:
        raise ZeroDivisionError("One of the vectors is zero, cannot compute cosine similarity.")
    return dot_product / (norm1 * norm2)


def normalize(vector):
    """Normalize a complex vector."""
    norm = np.linalg.norm(vector)
    if norm == 0:
        raise ValueError("Zero vector cannot be normalized.")
    return vector / norm


def pair_real_to_complex(vector):
    """Convert a real-valued vector into a complex-valued vector by pairing elements."""
    if len(vector) % 2 != 0:
        raise ValueError("Vector length must be even to pair into complex numbers.")
    complex_vector = np.array([vector[i] + 1j * vector[i+1] for i in range(0, len(vector), 2)])
    return complex_vector


def pad_vector(vector, target_size):
    """Pad the vector with zeros to the nearest power of 2."""
    if len(vector) > target_size:
        raise ValueError("Vector is larger than the target size.")
    padded_vector = np.zeros(target_size, dtype=complex)
    padded_vector[:len(vector)] = vector
    return padded_vector

