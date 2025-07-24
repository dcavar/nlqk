# coding: utf-8

"""
states.py

(C) 2025 by [Damir Cavar](http://damir.cavar.me/), James Bryan Graves, and [NLP Lab](https://nlp-lab.org/)

Module: nlqk.embeddings.states

Quantum state functions and tools.
"""


import os
import itertools
from functools import reduce
import operator

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

# Single-qubit Pauli matrices and labels (commonly used across functions)
PAULI_I = np.array([[1, 0], [0, 1]], dtype=complex)
PAULI_X = np.array([[0, 1], [1, 0]], dtype=complex)
PAULI_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
PAULI_Z = np.array([[1, 0], [0, -1]], dtype=complex)
PAULIS = [PAULI_I, PAULI_X, PAULI_Y, PAULI_Z]
PAULI_LABELS = ["I", "X", "Y", "Z"]

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
    norm1 = np.linalg.norm(psi1)
    norm2 = np.linalg.norm(psi2)
    
    # Handle edge cases with zero norm vectors
    if np.isclose(norm1, 0.0) or np.isclose(norm2, 0.0):
        # Both zero vectors are considered equal
        if np.isclose(norm1, 0.0) and np.isclose(norm2, 0.0):
            return True
        # One zero, one non-zero are not equal
        else:
            return False
    
    # Normalize the vectors
    psi1 = psi1 / norm1
    psi2 = psi2 / norm2
    inner_product = np.abs(np.vdot(psi1, psi2))
    return np.isclose(inner_product, 1.0, atol=tol)

def trace(matrix: np.ndarray) -> complex:
    """
    Compute the trace of a matrix.
    
    Args:
        matrix (np.ndarray): A square matrix.
        
    Returns:
        complex: The trace (sum of diagonal elements) of the matrix.
        
    Note:
        This is a utility wrapper around np.trace that ensures consistent
        behavior across NumPy/CuPy environments and provides clear typing.
    """
    return np.trace(matrix)

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

def pad_state(psi: np.ndarray) -> np.ndarray:
    """
    Pad a quantum state vector to the next power of 2 dimension.
    
    Args:
        psi (np.ndarray): Input quantum state vector of any size.
        
    Returns:
        np.ndarray: Padded state vector with dimension = next power of 2.
        
    Note:
        If the input dimension is already a power of 2, returns the original state.
        Padding is done with zeros, which preserves normalization properties.
        Minimum dimension is 2 (for at least 1 qubit).
    """
    dim = len(psi)
    
    # Find next power of 2 that is >= current dimension
    # Start with 2 as minimum (1 qubit)
    next_power = 2
    while next_power < dim:
        next_power *= 2
    
    # If already a power of 2 and >= 2, return original
    if next_power == dim and dim >= 2:
        return psi
    
    # Create padded state with zeros
    padded_psi = np.zeros(next_power, dtype=psi.dtype)
    padded_psi[:dim] = psi
    
    return padded_psi

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

def tomography(
    rho_true: np.ndarray,
    shots_per_setting: int = 4_000,
    clip_negative_eigvals: bool = True,
    add_noise: bool = True,
    random_seed: int = None,
):
    """
    Full Pauli-basis state tomography for an n-qubit density matrix.

    Args:
        rho_true : np.ndarray (2**n, 2**n)
            The 'ground-truth' density matrix you want to reconstruct.
        shots_per_setting : int
            Number of measurement shots per Pauli setting (≥ 1).
            Larger → smaller statistical error; total shots = shots_per_setting * 4**n.
        clip_negative_eigvals : bool
            If True, enforces positivity by eigenvalue-clipping (simple but common).
        add_noise : bool
            If True, adds shot noise via binomial sampling. If False, uses exact expectations.
        random_seed : int, optional
            Random seed for reproducible noise generation. Only used if add_noise=True.

    Returns:
        rho_est      : np.ndarray
            Reconstructed density matrix, Hermitian and (optionally) positive.
        phi_est      : np.ndarray
            Leading eigenvector (approximate pure state if ρ is nearly pure).
        expect_true  : dict[str, float]
            Exact expectations ⟨P⟩ for every n-qubit Pauli string.
        expect_est   : dict[str, float]
            Shot-based estimates of the same expectations.
    """
    # ------------------ sanity checks & setup ------------------
    dim = rho_true.shape[0]
    if rho_true.shape != (dim, dim):
        raise ValueError("rho_true must be square")
    n_qubits = int(np.log2(dim))
    if 2 ** n_qubits != dim:
        raise ValueError("rho_true dimension must be a power of 2")

    # ------------------ generate n-qubit Paulis ----------------
    ops, names = [], []
    for idx in itertools.product(range(4), repeat=n_qubits):
        op = reduce(np.kron, (PAULIS[i] for i in idx))
        ops.append(op)
        names.append("".join(PAULI_LABELS[i] for i in idx))

    # ------------------ shot-based expectation estimation -------
    if random_seed is not None:
        np.random.seed(random_seed)
    
    expect_true, expect_est = {}, {}
    for name, op in zip(names, ops):
        m_true = float(np.real(np.trace(rho_true @ op)))
        expect_true[name] = m_true

        if add_noise:
            # simulate ±1 measurement outcomes
            p_plus = (1 + m_true) / 2
            n_plus = np.random.binomial(shots_per_setting, p_plus)
            m_est = (n_plus - (shots_per_setting - n_plus)) / shots_per_setting
        else:
            # use exact expectations (no noise)
            m_est = m_true
        expect_est[name] = m_est

    # ------------------ linear inversion tomography -------------
    rho_est = sum(expect_est[name] * op for name, op in zip(names, ops))
    rho_est /= 2 ** n_qubits  # trace normalisation factor

    # ------------------ enforce Hermiticity & positivity --------
    rho_est = (rho_est + rho_est.conj().T) / 2  # Hermitian symmetrisation
    evals, evecs = np.linalg.eigh(rho_est)

    if clip_negative_eigvals:  # simple physical-state projection
        evals_clipped = np.clip(evals, 0, None)
        rho_est = (evecs * evals_clipped) @ evecs.conj().T
        rho_est /= np.trace(rho_est)
    else:
        evals_clipped = evals  # keep as is

    # ------------------ “best-guess” pure state ----------------
    idx_max = int(np.argmax(evals_clipped))
    phi_est = evecs[:, idx_max] * np.sqrt(max(evals_clipped[idx_max], 0))

    return rho_est, phi_est, expect_true, expect_est
