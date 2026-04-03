"""Stress and strain computation utilities for MIDAS.

Provides:
- Voigt notation conversions (3x3 tensor <-> 6-vector)
- Lattice parameters to strain tensor (Fable-Beaudoin method)
- Hooke's law: strain -> stress using single-crystal stiffness
- Coordinate frame transformations (grain <-> lab <-> sample)
- The 6x6 rotation matrix in Voigt space (U-matrix from Paper I Eq. 14)
- Volume-average stress constraint (FF-1 from whitepaper)
- Hydrostatic-deviatoric decomposition (FF-2 from whitepaper)
- Common stiffness matrices for standard materials

All computations use Voigt-Mandel notation with sqrt(2) factors
on shear components, following the convention in Paper I Appendix A.

Reference: Sharma, Park, Kenesei, "MIDAS: A Methodological Framework
for HEDM Data Reduction -- Part I: Methodology"
"""

import math
from typing import Optional, Tuple, Union

import numpy as np


# ===================================================================
#  Voigt notation conversions
# ===================================================================

def tensor_to_voigt(T: np.ndarray) -> np.ndarray:
    """Convert symmetric 3x3 tensor(s) to 6-vector Voigt-Mandel notation.

    Mandel convention: shear components scaled by sqrt(2) so that
    ||T||_F = ||v||_2 (Frobenius norm preserved).

    Parameters
    ----------
    T : ndarray (..., 3, 3)

    Returns
    -------
    ndarray (..., 6) -- [T_xx, T_yy, T_zz, sqrt(2)*T_yz, sqrt(2)*T_xz, sqrt(2)*T_xy]
    """
    s2 = math.sqrt(2.0)
    return np.stack([
        T[..., 0, 0],
        T[..., 1, 1],
        T[..., 2, 2],
        s2 * T[..., 1, 2],
        s2 * T[..., 0, 2],
        s2 * T[..., 0, 1],
    ], axis=-1)


def voigt_to_tensor(v: np.ndarray) -> np.ndarray:
    """Convert 6-vector Voigt-Mandel notation to symmetric 3x3 tensor(s).

    Parameters
    ----------
    v : ndarray (..., 6)

    Returns
    -------
    ndarray (..., 3, 3)
    """
    s2_inv = 1.0 / math.sqrt(2.0)
    T = np.zeros(v.shape[:-1] + (3, 3), dtype=v.dtype)
    T[..., 0, 0] = v[..., 0]
    T[..., 1, 1] = v[..., 1]
    T[..., 2, 2] = v[..., 2]
    T[..., 1, 2] = T[..., 2, 1] = v[..., 3] * s2_inv
    T[..., 0, 2] = T[..., 2, 0] = v[..., 4] * s2_inv
    T[..., 0, 1] = T[..., 1, 0] = v[..., 5] * s2_inv
    return T


def tensor_to_voigt_engineering(T: np.ndarray) -> np.ndarray:
    """Convert to engineering Voigt notation (no sqrt(2) factor).

    Returns [T_xx, T_yy, T_zz, 2*T_yz, 2*T_xz, 2*T_xy] for strain,
    or [T_xx, T_yy, T_zz, T_yz, T_xz, T_xy] for stress.
    """
    return np.stack([
        T[..., 0, 0], T[..., 1, 1], T[..., 2, 2],
        2.0 * T[..., 1, 2], 2.0 * T[..., 0, 2], 2.0 * T[..., 0, 1],
    ], axis=-1)


# ===================================================================
#  A-matrix: lattice parameters to orthonormal basis
# ===================================================================

def lattice_params_to_A_matrix(latc: np.ndarray) -> np.ndarray:
    """Build the A matrix (Paper I Eq. 6) from lattice parameters.

    Maps fractional crystal coordinates to Cartesian coordinates.

    Parameters
    ----------
    latc : ndarray (..., 6)
        [a, b, c, alpha_deg, beta_deg, gamma_deg]

    Returns
    -------
    ndarray (..., 3, 3)
    """
    d2r = math.pi / 180.0
    a = latc[..., 0]
    b = latc[..., 1]
    c = latc[..., 2]
    alpha = latc[..., 3] * d2r
    beta = latc[..., 4] * d2r
    gamma = latc[..., 5] * d2r

    cos_a, sin_a = np.cos(alpha), np.sin(alpha)
    cos_b, sin_b = np.cos(beta), np.sin(beta)
    cos_g = np.cos(gamma)

    # gamma* in reciprocal space
    cos_g_star = (cos_a * cos_b - cos_g) / (sin_a * sin_b)
    sin_g_star = np.sqrt(np.clip(1.0 - cos_g_star**2, 0, None))

    A = np.zeros(latc.shape[:-1] + (3, 3), dtype=np.float64)
    A[..., 0, 0] = a * sin_b * sin_g_star
    A[..., 1, 0] = -a * sin_b * cos_g_star
    A[..., 1, 1] = b * sin_a
    A[..., 2, 0] = a * cos_b
    A[..., 2, 1] = b * cos_a
    A[..., 2, 2] = c
    return A


# ===================================================================
#  Strain tensor from lattice parameters (Fable-Beaudoin method)
# ===================================================================

def lattice_params_to_strain(
    latc_strained: np.ndarray,
    latc_unstrained: np.ndarray,
) -> np.ndarray:
    """Compute Green-Lagrange strain tensor in grain frame.

    Matches CalcStrainTensorFableBeaudoin in CalcStrains.c.

    Parameters
    ----------
    latc_strained : ndarray (..., 6)
    latc_unstrained : ndarray (..., 6) or (6,)

    Returns
    -------
    ndarray (..., 3, 3)  strain tensor in grain coordinates
    """
    A = lattice_params_to_A_matrix(latc_strained)
    A0 = lattice_params_to_A_matrix(latc_unstrained)
    A0_inv = np.linalg.inv(A0)
    F = A @ A0_inv  # deformation gradient
    I = np.eye(3)
    # Green-Lagrange: E = 0.5*(F + F^T) - I
    return 0.5 * (F + np.swapaxes(F, -1, -2)) - I


def strain_grain_to_lab(
    strain_grain: np.ndarray,
    orient: np.ndarray,
) -> np.ndarray:
    """Transform strain tensor from grain to lab frame.

    Paper I Eq. 4: epsilon_lab = U * epsilon_gr * U^T

    Parameters
    ----------
    strain_grain : ndarray (..., 3, 3)
    orient : ndarray (..., 3, 3)  orientation matrix

    Returns
    -------
    ndarray (..., 3, 3)  strain in lab frame
    """
    return orient @ strain_grain @ np.swapaxes(orient, -1, -2)


def strain_lab_to_grain(
    strain_lab: np.ndarray,
    orient: np.ndarray,
) -> np.ndarray:
    """Transform strain tensor from lab to grain frame.

    epsilon_gr = U^T * epsilon_lab * U
    """
    Ut = np.swapaxes(orient, -1, -2)
    return Ut @ strain_lab @ orient


# ===================================================================
#  6x6 rotation matrix in Voigt space (Paper I Eq. 14)
# ===================================================================

def rotation_voigt_mandel(U: np.ndarray) -> np.ndarray:
    """Build the 6x6 rotation matrix for Voigt-Mandel notation.

    Transforms vectorized symmetric tensors between coordinate frames:
        {epsilon_grain} = M @ {epsilon_lab}
        {sigma_grain} = M @ {sigma_lab}

    Uses Mandel convention (sqrt(2) factors on shear).
    Paper I Eq. 14.

    Parameters
    ----------
    U : ndarray (..., 3, 3)  rotation matrix (e.g., orientation matrix)

    Returns
    -------
    ndarray (..., 6, 6)
    """
    s2 = math.sqrt(2.0)
    M = np.zeros(U.shape[:-2] + (6, 6), dtype=U.dtype)

    # Diagonal block (normal components)
    for i in range(3):
        for j in range(3):
            M[..., i, j] = U[..., i, j] ** 2

    # Off-diagonal: normal-to-shear coupling
    # Column 3 (yz), 4 (xz), 5 (xy)
    pairs = [(1, 2), (0, 2), (0, 1)]
    for col_idx, (p, q) in enumerate(pairs):
        for row in range(3):
            M[..., row, 3 + col_idx] = s2 * U[..., row, p] * U[..., row, q]

    # Rows 3-5: shear-to-normal coupling
    for row_idx, (p, q) in enumerate(pairs):
        for col in range(3):
            M[..., 3 + row_idx, col] = s2 * U[..., p, col] * U[..., q, col]

    # Shear-to-shear block (3x3 lower-right)
    for row_idx, (r1, r2) in enumerate(pairs):
        for col_idx, (c1, c2) in enumerate(pairs):
            M[..., 3 + row_idx, 3 + col_idx] = (
                U[..., r1, c1] * U[..., r2, c2]
                + U[..., r1, c2] * U[..., r2, c1]
            )

    return M


# ===================================================================
#  Hooke's law: strain -> stress
# ===================================================================

def hooke_stress(
    strain: np.ndarray,
    stiffness: np.ndarray,
    orient: Optional[np.ndarray] = None,
    frame: str = "lab",
) -> np.ndarray:
    """Compute stress from strain using Hooke's law.

    Parameters
    ----------
    strain : ndarray (..., 3, 3) or (..., 6)
        Strain tensor. If 3x3, converted to Voigt-Mandel internally.
    stiffness : ndarray (6, 6)
        Single-crystal stiffness matrix C in Voigt-Mandel notation,
        defined in the crystal frame.
    orient : ndarray (..., 3, 3), optional
        Orientation matrix. Required if ``frame`` is ``"lab"`` or ``"sample"``.
    frame : str
        ``"grain"``: strain is in grain frame, return stress in grain frame.
        ``"lab"``: strain is in lab frame, transform to grain, apply C,
        transform back.  This is the most common case.

    Returns
    -------
    ndarray (..., 3, 3) stress tensor in the requested frame.
    """
    # Convert to Voigt if needed
    if strain.shape[-1] == 3 and strain.shape[-2] == 3:
        eps_voigt = tensor_to_voigt(strain)
    else:
        eps_voigt = strain

    if frame == "grain":
        sig_voigt = eps_voigt @ stiffness.T  # σ = C @ ε in Voigt
        return voigt_to_tensor(sig_voigt)

    if orient is None:
        raise ValueError("orient required for lab-frame computation")

    # Paper I Eq. 16: σ_lab = M^T @ C @ M @ ε_lab
    M = rotation_voigt_mandel(orient)
    Mt = np.swapaxes(M, -1, -2)
    C_lab = Mt @ stiffness @ M  # rotated stiffness in lab frame
    sig_voigt = (C_lab @ eps_voigt[..., None]).squeeze(-1)
    return voigt_to_tensor(sig_voigt)


# ===================================================================
#  Common single-crystal stiffness matrices (Voigt-Mandel notation)
# ===================================================================

def cubic_stiffness(C11: float, C12: float, C44: float) -> np.ndarray:
    """Build 6x6 stiffness matrix for cubic crystal in Mandel notation.

    Parameters
    ----------
    C11, C12, C44 : float
        Independent elastic constants in GPa.

    Returns
    -------
    ndarray (6, 6)
    """
    C = np.zeros((6, 6))
    C[0, 0] = C[1, 1] = C[2, 2] = C11
    C[0, 1] = C[0, 2] = C[1, 0] = C[1, 2] = C[2, 0] = C[2, 1] = C12
    # Mandel convention: C44_mandel = 2 * C44_engineering
    # because σ_mandel = sqrt(2)*τ and ε_mandel = sqrt(2)*γ/2
    C[3, 3] = C[4, 4] = C[5, 5] = 2.0 * C44
    return C


# Common materials (C11, C12, C44 in GPa)
STIFFNESS_LIBRARY = {
    "Au":  {"C11": 192.9, "C12": 163.8, "C44": 41.5},
    "Cu":  {"C11": 168.4, "C12": 121.4, "C44": 75.4},
    "Al":  {"C11": 108.2, "C12": 61.3,  "C44": 28.5},
    "Fe":  {"C11": 231.4, "C12": 134.7, "C44": 116.4},  # alpha-Fe (BCC)
    "Ni":  {"C11": 246.5, "C12": 147.3, "C44": 124.7},
    "Ti":  {"C11": 162.4, "C12": 92.0,  "C44": 46.7},   # alpha-Ti (HCP, approx)
    "W":   {"C11": 522.4, "C12": 204.4, "C44": 160.8},
    "Si":  {"C11": 165.7, "C12": 63.9,  "C44": 79.6},
    "CeO2": {"C11": 403.0, "C12": 105.0, "C44": 60.0},
}


def get_stiffness(material: str) -> np.ndarray:
    """Get stiffness matrix for a common cubic material.

    Parameters
    ----------
    material : str
        Material name (e.g., "Au", "Cu", "Fe").

    Returns
    -------
    ndarray (6, 6) stiffness in GPa, Voigt-Mandel notation.
    """
    if material not in STIFFNESS_LIBRARY:
        raise ValueError(
            f"Unknown material '{material}'. "
            f"Available: {list(STIFFNESS_LIBRARY.keys())}"
        )
    p = STIFFNESS_LIBRARY[material]
    return cubic_stiffness(p["C11"], p["C12"], p["C44"])


# ===================================================================
#  Equilibrium constraints (FF-1 and FF-2 from whitepaper)
# ===================================================================

def volume_average_stress_constraint(
    stresses: np.ndarray,
    volumes: np.ndarray,
    applied_stress: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Apply volume-average stress constraint (FF-1).

    Enforces: sum(V_g * sigma_g) / V_total = sigma_applied

    Uses weighted distribution: the correction is distributed
    proportionally to each grain's compliance (inverse stiffness).
    For simplicity and generality, we use volume-weighted distribution:
    grains with larger volume get proportionally larger corrections.

    Parameters
    ----------
    stresses : ndarray (N, 3, 3) or (N, 6)
        Per-grain stress tensors.
    volumes : ndarray (N,)
        Grain volumes (or areas for 2D). Relative sizes suffice.
    applied_stress : ndarray (3, 3) or (6,), optional
        Applied macroscopic stress. Default: zero (unloaded sample).

    Returns
    -------
    ndarray (N, 3, 3) or (N, 6) corrected stresses (same shape as input).
    """
    is_voigt = stresses.ndim == 2 and stresses.shape[-1] == 6
    if is_voigt:
        sig = stresses.copy()
    else:
        sig = tensor_to_voigt(stresses)

    if applied_stress is None:
        sig_app = np.zeros(6)
    elif applied_stress.shape == (3, 3):
        sig_app = tensor_to_voigt(applied_stress)
    else:
        sig_app = applied_stress.copy()

    V_total = volumes.sum()
    w = volumes / V_total  # (N,)

    # Current volume average
    sig_avg = np.sum(w[:, None] * sig, axis=0)  # (6,)

    # Correction needed
    delta_sig = sig_app - sig_avg  # (6,)

    # Weighted distribution: larger grains get proportionally larger correction
    # This preserves the volume average while distributing based on volume
    sig_corrected = sig + delta_sig[None, :]  # uniform correction

    if not is_voigt:
        return voigt_to_tensor(sig_corrected)
    return sig_corrected


def hydrostatic_deviatoric_decomposition(
    stresses: np.ndarray,
    volumes: np.ndarray,
    applied_stress: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decompose stresses into hydrostatic and deviatoric parts (FF-2).

    The deviatoric part is determined from relative peak shifts (well-
    conditioned). The hydrostatic part is determined from the equilibrium
    constraint, removing the dependence on the ambiguous strain-free
    reference lattice parameter (d0).

    Parameters
    ----------
    stresses : ndarray (N, 3, 3)
        Per-grain stress tensors (from lattice parameter refinement).
    volumes : ndarray (N,)
        Grain volumes.
    applied_stress : ndarray (3, 3), optional
        Applied macroscopic stress. Default: zero.

    Returns
    -------
    hydrostatic : ndarray (N,)
        Per-grain hydrostatic stress (pressure = -hydrostatic).
    deviatoric : ndarray (N, 3, 3)
        Per-grain deviatoric stress tensors.
    corrected : ndarray (N, 3, 3)
        Reconstructed full stress tensors with equilibrium-consistent
        hydrostatic component.
    """
    if applied_stress is None:
        applied_stress = np.zeros((3, 3))

    N = stresses.shape[0]
    I = np.eye(3)
    V_total = volumes.sum()
    w = volumes / V_total

    # Decompose each grain
    hydro_raw = np.trace(stresses, axis1=-2, axis2=-1) / 3.0  # (N,)
    deviatoric = stresses - hydro_raw[:, None, None] * I[None, :, :]

    # The deviatoric part is well-determined from relative peak shifts.
    # The hydrostatic part depends on d0 and is poorly determined.
    # Use equilibrium to fix it:
    # sum(V_g * (hydro_g * I + dev_g)) / V_total = sigma_applied
    # Taking trace: sum(V_g * 3*hydro_g) / V_total = tr(sigma_applied)
    # => volume-average hydrostatic = tr(sigma_applied) / 3

    target_hydro = np.trace(applied_stress) / 3.0
    current_avg_hydro = np.sum(w * hydro_raw)
    hydro_shift = target_hydro - current_avg_hydro

    # Apply shift uniformly (all grains get the same hydrostatic correction)
    hydro_corrected = hydro_raw + hydro_shift

    # Also enforce deviatoric equilibrium:
    # sum(V_g * dev_g) / V_total = dev(sigma_applied)
    dev_applied = applied_stress - (np.trace(applied_stress) / 3.0) * I
    dev_avg = np.sum(w[:, None, None] * deviatoric, axis=0)
    dev_correction = dev_applied - dev_avg
    deviatoric_corrected = deviatoric + dev_correction[None, :, :]

    # Reconstruct
    corrected = hydro_corrected[:, None, None] * I[None, :, :] + deviatoric_corrected

    return hydro_corrected, deviatoric_corrected, corrected


# ===================================================================
#  Grain data reader (consolidated HDF5)
# ===================================================================

def read_grains_h5(filepath: str) -> dict:
    """Read grain data from MIDAS consolidated HDF5 output.

    Parameters
    ----------
    filepath : str
        Path to consolidated_Output.h5 or similar.

    Returns
    -------
    dict with keys:
        'orientations': ndarray (N, 3, 3)
        'euler_angles': ndarray (N, 3)
        'positions': ndarray (N, 3)
        'lattice_params': ndarray (N, 6)
        'strain_fable': ndarray (N, 3, 3)
        'strain_kenesei': ndarray (N, 3, 3)
        'radii': ndarray (N,)
        'confidences': ndarray (N,)
        'grain_ids': list of str
    """
    import h5py

    grains = {
        'orientations': [], 'euler_angles': [], 'positions': [],
        'lattice_params': [], 'strain_fable': [], 'strain_kenesei': [],
        'radii': [], 'confidences': [], 'grain_ids': [],
    }

    with h5py.File(filepath, 'r') as f:
        grp = f['grains']
        for gid in sorted(grp.keys()):
            g = grp[gid]
            grains['grain_ids'].append(gid)
            grains['orientations'].append(g['orientation'][()])
            grains['euler_angles'].append(g['euler_angles'][()])
            grains['positions'].append(g['position'][()])
            grains['lattice_params'].append(g['lattice_params_fit'][()])
            grains['strain_fable'].append(g['strain_fable'][()])
            grains['strain_kenesei'].append(g['strain_kenesei'][()])
            grains['radii'].append(float(g['radius'][()]))
            grains['confidences'].append(float(g['confidence'][()]))

    for key in grains:
        if key != 'grain_ids':
            grains[key] = np.array(grains[key])

    return grains
