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
    from scipy.linalg import expm, logm
else:
    try: # prefer RAPIDS libraries and GPU over numpy and CPU
        import cupy as np  # Try to import cupy and alias it as np
        from cupyx.scipy.linalg import expm, logm
        _USE_GPU = True
    except ModuleNotFoundError:
        import numpy as np  # If cupy not found, import numpy and alias it as np
        from scipy.linalg import expm, logm
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

def householder(s: np.ndarray, psi: np.ndarray):
    """
    Return a Hermitian & unitary matrix R such that R |s⟩ = κ |ψ⟩.

    Args:
        s (np.ndarray): Source quantum state vector.
        psi (np.ndarray): Target quantum state vector.

    Returns:
        tuple: (κ, R) where κ is a pure phase factor ⟨ψ|s⟩ / |⟨ψ|s⟩| and
               R is a Hermitian & unitary reflection matrix.

    Note:
        The reflection matrix R transforms the source state |s⟩ to the target
        state |ψ⟩ up to a global phase factor κ.
    """
    overlap = np.vdot(psi, s)
    if np.isclose(overlap, 0.0):
        # The two states are orthogonal: one reflection already does the trick.
        κ = 1.0
    else:
        κ = overlap / abs(overlap)

    v = s - κ * psi
    v_norm = np.linalg.norm(v)
    
    # Check if states are already equal (up to global phase)
    if np.isclose(v_norm, 0.0):
        # States are already proportional, return identity matrix
        κ = overlap / abs(overlap) if not np.isclose(overlap, 0.0) else 1.0
        R = np.eye(len(s), dtype=complex)
        return κ, R
    
    v /= v_norm
    R = np.eye(len(s), dtype=complex) - 2 * np.outer(v, np.conjugate(v))
    return κ, R

def state_to_hamiltonian(psi: np.ndarray) -> np.ndarray:
    """
    Return a Hermitian H such that exp(-1j*H) |s⟩ = |ψ⟩.

    Args:
        psi (np.ndarray): Target normalized state vector (1-D complex array).

    Returns:
        np.ndarray: Hermitian matrix H (2-D complex array).

    Note:
        This function finds a Hamiltonian that evolves the uniform superposition
        state |s⟩ to the target state |ψ⟩ under the unitary evolution exp(-iH).
    """
    psi = np.asarray(psi, dtype=complex)
    psi /= np.linalg.norm(psi)

    d = psi.size
    s = np.ones(d, dtype=complex) / np.sqrt(d)        # |s⟩

    # Trivial (already-there) case.
    if np.allclose(psi, s * np.vdot(psi, s) / abs(np.vdot(psi, s))):
        return np.zeros((d, d), dtype=complex)

    κ, R = householder(s, psi)
    U = np.conjugate(κ) * R # now U|s⟩ = |ψ⟩

    H = 1j * logm(U)

    # keep numerical noise from breaking Hermiticity
    H = 0.5 * (H + H.conjugate().T)
    return H

def pad_hamiltonian(H_sub):
    """
    Pad a Hamiltonian matrix to the next power of 2 while preserving dynamics.
    
    Args:
        H_sub: Square Hamiltonian matrix of any size
        
    Returns:
        H_padded: Padded Hamiltonian matrix with size = next power of 2
    """
    current_size = H_sub.shape[0]
    
    # Find next power of 2 that is >= current_size
    next_power = 1
    while next_power < current_size:
        next_power *= 2
    
    # If current_size is already a power of 2 and > 1, no padding needed
    if next_power == current_size and current_size > 1:
        return H_sub  
    
    # For size 1, always pad to 2 (smallest valid quantum system)
    if current_size == 1:
        next_power = 2
    
    # Create padded matrix with zeros
    H_padded = np.zeros((next_power, next_power), dtype=H_sub.dtype)
    
    # Copy original Hamiltonian to top-left corner
    H_padded[:current_size, :current_size] = H_sub
    
    return H_padded