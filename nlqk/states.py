import numpy as np
from scipy.linalg import expm

def hamiltonian_to_state(H):
    """
    Reconstruct |ψ⟩ = e^{iH} |0⟩ from the stored Hamiltonian H.
    """
    n_qubits = int(np.log2(H.shape[0]))
    dim = 2 ** n_qubits

    zero_state = np.zeros(dim, dtype=complex)
    zero_state[0] = 1.0

    U = expm(1j * H)

    psi = U @ zero_state

    return psi

def check_states_equal(psi1, psi2, tol=1e-6):
    """Checks if two quantum states are equal up to global phase."""
    psi1 = psi1 / np.linalg.norm(psi1)
    psi2 = psi2 / np.linalg.norm(psi2)
    inner_product = np.abs(np.vdot(psi1, psi2))
    return np.isclose(inner_product, 1.0, atol=tol)