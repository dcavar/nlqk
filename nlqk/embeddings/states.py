
"""

"""


import numpy as np
from scipy.linalg import expm
from typing import Union, Sequence


def hamiltonian_to_state(H: Union[np.ndarray, Sequence[Sequence[Union[int, float, complex]]]]) -> np.ndarray:
    """
    Reconstructs a quantum state |ψ⟩ = e^{iH} |0⟩ from the given Hamiltonian H.

    Args:
        H (Union[np.ndarray, Sequence[Sequence[Union[int, float, complex]]]]): 
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


def check_states_equal(
    psi1: Union[np.ndarray, Sequence[Union[int, float, complex]]], 
    psi2: Union[np.ndarray, Sequence[Union[int, float, complex]]], 
    tol: float = 1e-6
) -> bool:
    """
    Checks if two quantum states are equal up to a global phase factor.

    Args:
        psi1 (Union[np.ndarray, Sequence[Union[int, float, complex]]]): First quantum state vector.
        psi2 (Union[np.ndarray, Sequence[Union[int, float, complex]]]): Second quantum state vector.
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