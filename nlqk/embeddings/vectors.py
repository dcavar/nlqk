# coding: utf-8

"""
vectors.py

Module: nlqk.embeddings.vectors

(C) 2025 by [Damir Cavar](http://damir.cavar.me/), James Bryan Graves, and [NLP Lab](https://nlp-lab.org/)

Vector functionalities:

"""


# from typing import Union, Sequence
try: # prefer RAPIDS libraries and GPU over numpy and CPU
    import cupy as np  # Try to import cupy and alias it as np
    _USE_GPU = True
except ModuleNotFoundError:
    import numpy as np  # If cupy not found, import numpy and alias it as np
    _USE_GPU = False
# import GPUtil  # If you're using GPUtil


# Example Usage:


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray, tol: float = 1e-12) -> complex:
    """
    Computes the cosine similarity between two vectors.

    Args:
        vec1 np.ndarray: First input vector.
        vec2 np.ndarray: Second input vector.
        tol (float): Threshold below which vector norm is treated as zero. Defaults to 1e-12.

    Returns:
        float: Cosine similarity value. For real inputs: value in [-1, 1].

    Raises:
        ZeroDivisionError: If either vector has zero length (norm below tolerance).
    """
    #if not np.iscomplexobj(vec1):
    #    vec1 = vec1.astype(complex)
    #if not np.iscomplexobj(vec2):
    #    vec2 = vec2.astype(complex) # np.ravel(np.asarray(vec2))
    # if vectors are normalized cosine similarity is equivalent to the dot product
    if is_normalized(vec1) and is_normalized(vec2):
        return np.dot(vec1, vec2)

    n1, n2 = np.linalg.norm(vec1), np.linalg.norm(vec2)
    if n1 < tol or n2 < tol:
        raise ZeroDivisionError("Zero-length vector")

    return complex(np.vdot(vec1, vec2) / (n1 * n2))


def is_normalized(vector: np.ndarray, tolerance: float=1e-9) -> bool:
    """
    Checks if a NumPy vector is normalized (its L2 norm is approximately 1).

    Args:
        vector (Union[Sequence[Union[int, float, complex]], np.ndarray]): The input vector.
        tolerance (float): The allowed tolerance for comparison with 1.

    Returns:
        bool: True if the vector is normalized, False otherwise.
    """
    if np.isclose(np.linalg.norm(vector), 1.0, atol=tolerance):
        return True
    return False


def normalize(vector: np.ndarray) -> np.ndarray:
    """
    Normalizes a vector to unit length.

    Args:
        vector np.ndarray: Input vector to normalize.

    Returns:
        np.ndarray: The normalized vector with unit norm.

    Raises:
        ValueError: If the input vector has zero norm and cannot be normalized.
    """
    norm = np.linalg.norm(vector)
    if norm != 0:
        return vector / norm
    raise ValueError("Zero vector cannot be normalized.")



def pad_vector(vector: np.ndarray, target_size: int) -> np.ndarray:
    """
    Pads a vector with zeros to reach the specified target size.

    Args:
        vector np.ndarray: Input vector to pad.
        target_size (int): The desired size of the output vector.

    Returns:
        np.ndarray: Padded vector of length target_size with complex dtype.

    Raises:
        ValueError: If the input vector is larger than the target size.
    """
    return np.pad(vector, (0, target_size - len(vector)), mode='constant')


def pair_real_to_complex(vector: np.ndarray) -> np.ndarray:
    """
    Converts a real-valued vector into a complex-valued vector by pairing adjacent elements.

    Args:
        vector np.ndarray: Real-valued input vector, if not even length it will be padded to even length.

    Returns:
        np.ndarray: Complex-valued vector where each complex number is formed from consecutive pairs.

    Raises:
        ValueError: If the vector length is odd and cannot be paired into complex numbers.
    """
    if len(vector) > 0:
        if len(vector) % 2 != 0:
            # automatically pad with a 0
            vector = pad_vector(vector, len(vector) + 1)
    else:
        raise ValueError("Vector length must be even to pair into complex numbers.")
    return np.array([vector[i] + 1j * vector[i+1] for i in range(0, len(vector), 2)])



def pad_vectors(vectors: np.ndarray, size: int) -> np.ndarray:
    """Pad rows with zeros to size.

    Args:
        vectors np.ndarray: matrix of vectors to be padded, all the same length.
        size int: target length of vectors.

    Returns:
        np.ndarray: Padded vectors.
    """
    return np.pad(vectors, [(0, 0), (0, size - vectors.shape[1])], mode='constant')

