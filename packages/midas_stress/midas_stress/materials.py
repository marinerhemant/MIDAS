"""Built-in single-crystal stiffness matrices for common materials.

All values in GPa, Voigt-Mandel notation.
"""

import numpy as np


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
    C[3, 3] = C[4, 4] = C[5, 5] = 2.0 * C44
    return C


def hexagonal_stiffness(
    C11: float, C12: float, C13: float, C33: float, C44: float,
) -> np.ndarray:
    """Build 6x6 stiffness matrix for hexagonal crystal in Mandel notation.

    Parameters
    ----------
    C11, C12, C13, C33, C44 : float
        Independent elastic constants in GPa.

    Returns
    -------
    ndarray (6, 6)
    """
    C = np.zeros((6, 6))
    C[0, 0] = C[1, 1] = C11
    C[2, 2] = C33
    C[0, 1] = C[1, 0] = C12
    C[0, 2] = C[2, 0] = C[1, 2] = C[2, 1] = C13
    # Mandel shear: 2 * engineering
    C[3, 3] = C[4, 4] = 2.0 * C44
    C[5, 5] = C11 - C12  # 2 * C66_eng = 2 * (C11 - C12)/2
    return C


# Library of common materials
STIFFNESS_LIBRARY = {
    "Au":   {"C11": 192.9, "C12": 163.8, "C44": 41.5,  "symmetry": "cubic"},
    "Cu":   {"C11": 168.4, "C12": 121.4, "C44": 75.4,  "symmetry": "cubic"},
    "Al":   {"C11": 108.2, "C12": 61.3,  "C44": 28.5,  "symmetry": "cubic"},
    "Fe":   {"C11": 231.4, "C12": 134.7, "C44": 116.4, "symmetry": "cubic"},
    "Ni":   {"C11": 246.5, "C12": 147.3, "C44": 124.7, "symmetry": "cubic"},
    "W":    {"C11": 522.4, "C12": 204.4, "C44": 160.8, "symmetry": "cubic"},
    "Si":   {"C11": 165.7, "C12": 63.9,  "C44": 79.6,  "symmetry": "cubic"},
    "CeO2": {"C11": 403.0, "C12": 105.0, "C44": 60.0,  "symmetry": "cubic"},
    "Ti":   {"C11": 162.4, "C12": 92.0,  "C13": 69.0,
             "C33": 180.7, "C44": 46.7, "symmetry": "hexagonal"},
}


def get_stiffness(material: str) -> np.ndarray:
    """Get stiffness matrix for a material from the built-in library.

    Parameters
    ----------
    material : str
        Material name (e.g., "Au", "Cu", "Fe", "Ti").

    Returns
    -------
    ndarray (6, 6) stiffness in GPa, Voigt-Mandel notation.
    """
    if material not in STIFFNESS_LIBRARY:
        raise ValueError(
            f"Unknown material '{material}'. "
            f"Available: {sorted(STIFFNESS_LIBRARY.keys())}"
        )
    p = STIFFNESS_LIBRARY[material]
    if p["symmetry"] == "cubic":
        return cubic_stiffness(p["C11"], p["C12"], p["C44"])
    elif p["symmetry"] == "hexagonal":
        return hexagonal_stiffness(p["C11"], p["C12"], p["C13"],
                                   p["C33"], p["C44"])
    raise ValueError(f"Unsupported symmetry '{p['symmetry']}' for {material}")


def list_materials() -> list:
    """Return sorted list of available material names."""
    return sorted(STIFFNESS_LIBRARY.keys())


# -------------------------------------------------------------------
#  d0 sensitivity analysis
# -------------------------------------------------------------------

def d0_sensitivity(material: str = None, stiffness: np.ndarray = None,
                   C11: float = None, C12: float = None) -> dict:
    """Compute stress sensitivity to a d0 (isotropic strain) error.

    A fractional d0 error acts as an isotropic strain perturbation
    eps_iso * I. The resulting stress error in the crystal frame is
    sigma_err = C @ {I} * eps_iso, where {I} = [1,1,1,0,0,0]^T.

    For cubic crystals, this produces a pure hydrostatic stress error:
    delta_sigma = 3K * eps_iso (K = Voigt bulk modulus).

    For non-cubic crystals (e.g., hexagonal), the stress error has
    BOTH hydrostatic and deviatoric components in the crystal frame.
    After rotation to the lab frame, the artifact becomes
    orientation-dependent. This function reports the full crystal-frame
    response and flags non-cubic materials.

    Parameters
    ----------
    material : str, optional
        Material name from built-in library.
    stiffness : ndarray (6, 6), optional
        Stiffness matrix (used if material not given).
    C11, C12 : float, optional
        Cubic elastic constants (used if neither material nor
        stiffness given).

    Returns
    -------
    dict with keys:
        'bulk_modulus_GPa': Voigt bulk modulus K
        'sensitivity_MPa_per_ppm': total stress error norm (MPa)
            per ppm of fractional d0 error
        'sensitivity_MPa_per_100ppm': ... per 100 ppm
        'sensitivity_MPa_per_1000ppm': ... per 1000 ppm
        'is_pure_hydrostatic': bool — True for cubic, False otherwise
        'crystal_frame_response': ndarray (6,) — C @ {I} (Voigt)
        'hydrostatic_fraction': float — fraction of the error that
            is hydrostatic (1.0 for cubic, < 1.0 for non-cubic)
    """
    if material is not None:
        C = get_stiffness(material)
    elif stiffness is not None:
        C = stiffness
    elif C11 is not None and C12 is not None:
        C = cubic_stiffness(C11, C12, 0)  # C44 irrelevant for {I}
    else:
        raise ValueError("Provide material, stiffness, or C11+C12.")

    # Isotropic strain in Mandel notation: {I} = [1, 1, 1, 0, 0, 0]
    I_voigt = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    response = C @ I_voigt  # stress per unit isotropic strain

    # Bulk modulus from the response
    K = np.mean(response[:3]) / 3.0

    # Check if response is pure hydrostatic
    # Hydrostatic stress p*I has Voigt form [p, p, p, 0, 0, 0]
    # where p = (sigma_xx + sigma_yy + sigma_zz) / 3
    hydro_part = np.mean(response[:3])  # = p
    response_hydro = np.array([hydro_part]*3 + [0.0, 0.0, 0.0])
    response_dev = response - response_hydro
    norm_total = np.linalg.norm(response)
    norm_hydro = np.linalg.norm(response_hydro)
    is_pure = np.linalg.norm(response_dev) < 1e-10 * norm_total

    hydro_frac = (norm_hydro / norm_total) if norm_total > 0 else 1.0

    # Sensitivity: norm of stress error per ppm of eps_iso
    # eps_iso = 1 ppm = 1e-6 → stress_err = response * 1e-6 GPa
    # In MPa: norm * 1e-6 * 1e3 = norm * 1e-3
    sens_per_ppm = norm_total * 1e-3  # MPa per ppm

    return {
        'bulk_modulus_GPa': K,
        'sensitivity_MPa_per_ppm': sens_per_ppm,
        'sensitivity_MPa_per_100ppm': sens_per_ppm * 100.0,
        'sensitivity_MPa_per_1000ppm': sens_per_ppm * 1000.0,
        'is_pure_hydrostatic': bool(is_pure),
        'crystal_frame_response': response,
        'hydrostatic_fraction': float(hydro_frac),
    }


def d0_sensitivity_table() -> dict:
    """Compute d0 sensitivity for all materials in the library.

    Returns
    -------
    dict mapping material name -> d0_sensitivity result dict.
    """
    table = {}
    for mat in sorted(STIFFNESS_LIBRARY.keys()):
        table[mat] = d0_sensitivity(material=mat)
    return table
