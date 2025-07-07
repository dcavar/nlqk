# coding: utf-8

"""
states.py

(C) 2025 by [Damir Cavar](http://damir.cavar.me/), James Bryan Graves, and [NLP Lab](https://nlp-lab.org/)

Module: nlqk.embeddings.states

Quantum state functions and tools.
"""


import os
if os.getenv("GITHUB_ACTIONS") == "true":
    import numpy as np
    from scipy.linalg import expm
else:
    try: # prefer RAPIDS libraries and GPU over numpy and CPU
        import cupy as np  # Try to import cupy and alias it as np
        from cupyx.scipy.linalg import expm
        _USE_GPU = True
    except ModuleNotFoundError:
        import numpy as np  # If cupy not found, import numpy and alias it as np
        from scipy.linalg import expm
        _USE_GPU = False

def hamiltonian_to_state(H: np.ndarray,
                         t: float = 1.0,
                         init_state: str = "zero",
                         forward: bool = True) -> np.ndarray:
    """
    Evolve an initial state under a Hermitian Hamiltonian H for time t.

    Args:
        H (np.ndarray): Hermitian d×d matrix. d must be a power of 2.
        t (float): Evolution time in units where ħ = 1.
        init_state (str): "superposition" or "zero".
        forward (bool): If True, use e^{-iHt}; else e^{+iHt}.

    Returns:
        np.ndarray: The evolved state vector.
    """
    # structural checks
    if H.shape[0] != H.shape[1]:
        raise ValueError("H must be square")
    if not np.allclose(H, H.conj().T):
        raise ValueError("H must be Hermitian")
    dim = H.shape[0]
    if 2 ** int(np.log2(dim)) != dim:
        raise ValueError("Dimension must be a power of two")

    # choose the initial ket
    if init_state == "zero":
        psi0 = np.zeros(dim, dtype=complex)
        psi0[0] = 1.0
    elif init_state == "superposition":
        psi0 = np.ones(dim, dtype=complex) / np.sqrt(dim)
    else:
        raise ValueError("init_state must be 'zero' or 'superposition'")

    # unitary time evolution
    sign = -1j if forward else 1j
    U = expm(sign * t * H)

    return U @ psi0

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
