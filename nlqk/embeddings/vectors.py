
"""

"""

import numpy as np
from typing import Union, Sequence

def cosine_similarity(
    vec1: Union[Sequence[Union[int, float, complex]], np.ndarray], 
    vec2: Union[Sequence[Union[int, float, complex]], np.ndarray], 
    tol: float = 1e-12, 
    *, 
    return_complex: bool = False
) -> Union[float, complex]:
    """
    Computes the cosine similarity between two vectors.

    Args:
        vec1 (Union[Sequence[Union[int, float, complex]], np.ndarray]): First input vector (will be flattened).
        vec2 (Union[Sequence[Union[int, float, complex]], np.ndarray]): Second input vector (will be flattened).
        tol (float): Threshold below which vector norm is treated as zero. Defaults to 1e-12.
        return_complex (bool): If True and inputs are complex, return complex value with phase.
                              Otherwise return magnitude for complex inputs. Defaults to False.

    Returns:
        Union[float, complex]: Cosine similarity value. For real inputs: value in [-1, 1].
                              For complex inputs: magnitude in [0, 1] (default) or full complex value.

    Raises:
        ZeroDivisionError: If either vector has zero length (norm below tolerance).
    """
    v1 = np.ravel(np.asarray(vec1))
    v2 = np.ravel(np.asarray(vec2))

    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < tol or n2 < tol:
        raise ZeroDivisionError("Zero-length vector")

    sim = np.vdot(v1, v2) / (n1 * n2)

    if np.iscomplexobj(v1) or np.iscomplexobj(v2):
        return sim if return_complex else np.clip(np.abs(sim), 0.0, 1.0)
    else:
        return float(np.clip(sim.real, -1.0, 1.0))


def is_normalized(vector: Union[Sequence[Union[int, float, complex]], np.ndarray], tolerance: float=1e-9) -> np.bool_:
    """
    Checks if a NumPy vector is normalized (its L2 norm is approximately 1).

    Args:
        vector (Union[Sequence[Union[int, float, complex]], np.ndarray]): The input vector.
        tolerance (float): The allowed tolerance for comparison with 1.

    Returns:
        bool: True if the vector is normalized, False otherwise.
    """
    norm = np.linalg.norm(vector)
    return np.isclose(norm, 1.0, atol=tolerance)


def normalize(vector: Union[Sequence[Union[int, float, complex]], np.ndarray]) -> np.ndarray:
    """
    Normalizes a vector to unit length.

    Args:
        vector (Union[Sequence[Union[int, float, complex]], np.ndarray]): Input vector to normalize.

    Returns:
        np.ndarray: The normalized vector with unit norm.

    Raises:
        ValueError: If the input vector has zero norm and cannot be normalized.
    """
    norm = np.linalg.norm(vector)
    if norm == 0:
        raise ValueError("Zero vector cannot be normalized.")
    return vector / norm


def pair_real_to_complex(vector: Union[Sequence[Union[int, float]], np.ndarray]) -> np.ndarray:
    """
    Converts a real-valued vector into a complex-valued vector by pairing adjacent elements.

    Args:
        vector (Union[Sequence[Union[int, float]], np.ndarray]): Real-valued input vector with even length.

    Returns:
        np.ndarray: Complex-valued vector where each complex number is formed from consecutive pairs.

    Raises:
        ValueError: If the vector length is odd and cannot be paired into complex numbers.
    """
    if len(vector) % 2 != 0:
        raise ValueError("Vector length must be even to pair into complex numbers.")
    complex_vector = np.array([vector[i] + 1j * vector[i+1] for i in range(0, len(vector), 2)])
    return complex_vector


def pad_vector(vector: Union[Sequence[Union[int, float, complex]], np.ndarray], target_size: int) -> np.ndarray:
    """
    Pads a vector with zeros to reach the specified target size.

    Args:
        vector (Union[Sequence[Union[int, float, complex]], np.ndarray]): Input vector to pad.
        target_size (int): The desired size of the output vector.

    Returns:
        np.ndarray: Padded vector of length target_size with complex dtype.

    Raises:
        ValueError: If the input vector is larger than the target size.
    """
    #if len(vector) > target_size:
    #    raise ValueError("Vector is larger than the target size.")
    #padded_vector = np.zeros(target_size, dtype=complex)
    #padded_vector[:len(vector)] = vector
    #return padded_vector
    return np.pad(vector, (0, 9 - vector.shape[0]), mode='constant')


def pad_vectors(vectors: np.ndarray, size: int) -> np.ndarray:
    """Pad rows with zeros to size."""
    return np.pad(vectors, [(0, 0), (0, size - vectors.shape[1])], mode='constant')

