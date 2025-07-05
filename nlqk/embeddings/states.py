# coding: utf-8

"""
states.py

(C) 2025 by [Damir Cavar](http://damir.cavar.me/), James Bryan Graves, and [NLP Lab](https://nlp-lab.org/)

Module: nlqk.embeddings.states

Quantum state functions and tools.
"""


try: # prefer RAPIDS libraries and GPU over numpy and CPU
    import cupy as np  # Try to import cupy and alias it as np
    from cupyx.scipy.linalg import expm
    _USE_GPU = True
except ModuleNotFoundError:
    import numpy as np  # If cupy not found, import numpy and alias it as np
    from scipy.linalg import expm
    _USE_GPU = False
from typing import Union, Sequence


def hamiltonian_to_state(H: np.ndarray) -> np.ndarray:
    """
    Reconstructs a quantum state |ψ⟩ = e^{iH} |0⟩ from the given Hamiltonian H.

    Args:
        H np.ndarray: 
            Hamiltonian matrix (must be square with dimension 2^n for some integer n).

    Returns:
        np.ndarray: Complex quantum state vector |ψ⟩ obtained by applying e^{iH} to |0⟩.

    Raises:
        ValueError: If H is not a square matrix or dimension is not a power of 2.
    """
    n_qubits = int(np.log2(H.shape[0]))
    dim = 2 ** n_qubits

    zero_state = np.zeros(dim, dtype=complex)
    zero_state[0] = 1.0

    U = expm(1j * H)

    psi = U @ zero_state

    return psi


def check_states_equal(psi1: np.ndarray, psi2: np.ndarray, tol: float = 1e-6) -> bool:
    """
    Checks if two quantum states are equal up to a global phase factor.

    Args:
        psi1 np.ndarray: First quantum state vector.
        psi2 np.ndarray: Second quantum state vector.
        tol (float): Tolerance for the comparison. Defaults to 1e-6.

    Returns:
        bool: True if the states are equal up to global phase, False otherwise.

    Note:
        Two quantum states |ψ₁⟩ and |ψ₂⟩ are considered equal if |⟨ψ₁|ψ₂⟩| = 1,
        meaning they differ only by a global phase factor e^{iθ}.
    """
    psi1 = psi1 / np.linalg.norm(psi1)
    psi2 = psi2 / np.linalg.norm(psi2)
    inner_product = np.abs(np.vdot(psi1, psi2))
    return np.isclose(inner_product, 1.0, atol=tol)
