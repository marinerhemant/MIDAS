"""Single-crystal elastic constant recovery from HEDM load stages.

Inverts the macroscopic equilibrium constraint across one or more load
stages to fit the independent components of the single-crystal
stiffness tensor :math:`\\mathbf{C}_{\\mathrm{crystal}}`.

Three homogenisation assumptions are supported (``method`` keyword):

``"hill"`` (default, the Paper III method)
    No intra-aggregate homogenisation.  Uses each grain's *measured*
    lab-frame strain and enforces only the volume-average stress
    identity (Hill's lemma):

    .. math::

       \\{\\boldsymbol{\\sigma}_{\\mathrm{app}}^{(i)}\\}
       = \\sum_g w_g\\,\\mathfrak{U}_g^{\\top}\\,
           \\mathbf{C}_{\\mathrm{crystal}}\\,\\mathfrak{U}_g\\,
           \\{\\boldsymbol{\\varepsilon}_g^{(i)}\\}.

``"voigt"`` (iso-strain)
    Replaces every grain's strain by the common value
    :math:`\\bar{\\boldsymbol{\\varepsilon}}^{(i)}` (volume-weighted
    average of measured strains, unless a macroscopic strain is
    supplied via the stage dict):

    .. math::

       \\{\\boldsymbol{\\sigma}_{\\mathrm{app}}^{(i)}\\}
       = \\Bigl(\\sum_g w_g\\,\\mathfrak{U}_g^{\\top}\\,
              \\mathbf{C}_{\\mathrm{crystal}}\\,\\mathfrak{U}_g\\Bigr)\\,
           \\{\\bar{\\boldsymbol{\\varepsilon}}^{(i)}\\}.

``"reuss"`` (iso-stress)
    Replaces every grain's stress by :math:`\\boldsymbol{\\sigma}_{
    \\mathrm{app}}^{(i)}` and works with the single-crystal compliance
    :math:`\\mathbf{S}_{\\mathrm{crystal}}` (same symmetry basis in
    Mandel form), then inverts at the end:

    .. math::

       \\{\\bar{\\boldsymbol{\\varepsilon}}^{(i)}\\}
       = \\Bigl(\\sum_g w_g\\,\\mathfrak{U}_g^{\\top}\\,
              \\mathbf{S}_{\\mathrm{crystal}}\\,\\mathfrak{U}_g\\Bigr)\\,
           \\{\\boldsymbol{\\sigma}_{\\mathrm{app}}^{(i)}\\},\\qquad
       \\mathbf{C}_{\\mathrm{crystal}} = \\mathbf{S}_{\\mathrm{crystal}}^{-1}.

Parameterising :math:`\\mathbf{C}_{\\mathrm{crystal}} = \\sum_k c_k P_k`
(and likewise :math:`\\mathbf{S}_{\\mathrm{crystal}} = \\sum_k s_k P_k`,
valid in Mandel form since the zero/equality pattern is shared) gives
a linear system :math:`\\mathbf{A}^{(i)}\\,\\mathbf{c} =
\\mathbf{b}^{(i)}` for each method.  Stacking stages yields a weighted
least-squares problem.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from .tensor import tensor_to_voigt, rotation_voigt_mandel
from .equilibrium import d0_correction_strain_level, effective_weights


_SQRT2 = math.sqrt(2.0)


# -------------------------------------------------------------------
#  Standard-Voigt (1-indexed) -> midas-Mandel index translation
# -------------------------------------------------------------------

# Standard Voigt ordering:  1=xx, 2=yy, 3=zz, 4=yz, 5=xz, 6=xy
# Midas Mandel ordering:    0=xx, 1=yy, 2=zz, 3=xy, 4=xz, 5=yz
_VOIGT_TO_MIDAS = {1: 0, 2: 1, 3: 2, 4: 5, 5: 4, 6: 3}


def _mandel_factor(alpha: int, beta: int) -> float:
    """Mandel scaling for the (alpha, beta) entry of a stiffness matrix.

    Entries are scaled by ``k_alpha * e_beta / k_beta`` where
    ``k=1`` for a normal index (0,1,2) and ``k=sqrt(2)`` for a shear
    index (3,4,5); ``e=1`` for normal and ``e=2`` for shear (the
    engineering strain doubling).  Equivalently:
    normal-normal -> 1, normal-shear and shear-normal -> sqrt(2),
    shear-shear -> 2.
    """
    a_shear = alpha >= 3
    b_shear = beta >= 3
    if not a_shear and not b_shear:
        return 1.0
    if a_shear and b_shear:
        return 2.0
    return _SQRT2


def _make_P_from_voigt_entries(entries: dict) -> np.ndarray:
    """Build a 6x6 symmetric Mandel matrix from standard-Voigt entries.

    Parameters
    ----------
    entries : dict
        Maps ``(i, j)`` tuples in 1-indexed standard-Voigt ordering to
        a coefficient.  Symmetry ``(j, i) = (i, j)`` is applied
        automatically; do not list both.

    Returns
    -------
    ndarray (6, 6)
        Symmetric Mandel-notation matrix with correct Mandel factors
        applied.  Indices follow midas ordering
        [xx, yy, zz, xy, xz, yz].
    """
    P = np.zeros((6, 6), dtype=np.float64)
    for (i, j), coeff in entries.items():
        a = _VOIGT_TO_MIDAS[i]
        b = _VOIGT_TO_MIDAS[j]
        val = coeff * _mandel_factor(a, b)
        P[a, b] = val
        P[b, a] = val
    return P


# -------------------------------------------------------------------
#  Symmetry basis matrices
# -------------------------------------------------------------------
#
# For each crystal system we list the canonical independent elastic
# constants and their contribution to the 6x6 stiffness matrix in
# standard-Voigt (1-indexed) notation.  The helper
# ``_make_P_from_voigt_entries`` converts to midas-Mandel form.
#
# Conventions used:
#   - Hexagonal, trigonal, tetragonal: c-axis = z (sample index 3).
#   - Monoclinic: unique axis b = y (sample index 2), standard setting.
#   - Triclinic: all 21 upper-triangular entries independent.
#
# References: Nye (1957) "Physical Properties of Crystals", Ch. VIII;
# Simmons & Wang (1971) "Single Crystal Elastic Constants and
# Calculated Aggregate Properties".

def _cubic_basis() -> tuple[list[str], np.ndarray]:
    """Cubic (m-3m, 432, 43m): 3 independent constants."""
    defs = [
        ("C11", {(1, 1): 1.0, (2, 2): 1.0, (3, 3): 1.0}),
        ("C12", {(1, 2): 1.0, (1, 3): 1.0, (2, 3): 1.0}),
        ("C44", {(4, 4): 1.0, (5, 5): 1.0, (6, 6): 1.0}),
    ]
    return [n for n, _ in defs], np.stack(
        [_make_P_from_voigt_entries(e) for _, e in defs], axis=0
    )


def _hexagonal_basis() -> tuple[list[str], np.ndarray]:
    """Hexagonal (6/mmm etc.): 5 independent constants.

    ``C66 = (C11 - C12) / 2`` is enforced automatically by building
    the (6,6) entry from the ``C11`` and ``C12`` basis matrices.
    """
    defs = [
        # C11 enters at (1,1), (2,2), and contributes +1/2 to (6,6)
        ("C11", {(1, 1): 1.0, (2, 2): 1.0, (6, 6): 0.5}),
        # C12 enters at (1,2), and contributes -1/2 to (6,6)
        ("C12", {(1, 2): 1.0, (6, 6): -0.5}),
        ("C13", {(1, 3): 1.0, (2, 3): 1.0}),
        ("C33", {(3, 3): 1.0}),
        ("C44", {(4, 4): 1.0, (5, 5): 1.0}),
    ]
    return [n for n, _ in defs], np.stack(
        [_make_P_from_voigt_entries(e) for _, e in defs], axis=0
    )


def _trigonal_basis() -> tuple[list[str], np.ndarray]:
    """Trigonal (-3m, 32, 3m): 6 independent constants.

    Includes the C14 shear-coupling term characteristic of trigonal
    symmetry.  ``C66 = (C11 - C12) / 2`` as in hexagonal.
    Trigonal low-symmetry classes (3, -3) add a seventh constant
    ``C15``; use :func:`_trigonal_low_basis` for those.
    """
    defs = [
        ("C11", {(1, 1): 1.0, (2, 2): 1.0, (6, 6): 0.5}),
        ("C12", {(1, 2): 1.0, (6, 6): -0.5}),
        ("C13", {(1, 3): 1.0, (2, 3): 1.0}),
        # C14 couples (1,4) = -(2,4) and enters shear block as (5,6) = C14
        ("C14", {(1, 4): 1.0, (2, 4): -1.0, (5, 6): 1.0}),
        ("C33", {(3, 3): 1.0}),
        ("C44", {(4, 4): 1.0, (5, 5): 1.0}),
    ]
    return [n for n, _ in defs], np.stack(
        [_make_P_from_voigt_entries(e) for _, e in defs], axis=0
    )


def _tetragonal_basis() -> tuple[list[str], np.ndarray]:
    """Tetragonal (4/mmm, 4mm, 422, -42m): 6 independent constants."""
    defs = [
        ("C11", {(1, 1): 1.0, (2, 2): 1.0}),
        ("C12", {(1, 2): 1.0}),
        ("C13", {(1, 3): 1.0, (2, 3): 1.0}),
        ("C33", {(3, 3): 1.0}),
        ("C44", {(4, 4): 1.0, (5, 5): 1.0}),
        ("C66", {(6, 6): 1.0}),
    ]
    return [n for n, _ in defs], np.stack(
        [_make_P_from_voigt_entries(e) for _, e in defs], axis=0
    )


def _orthorhombic_basis() -> tuple[list[str], np.ndarray]:
    """Orthorhombic (mmm, 222, mm2): 9 independent constants."""
    defs = [
        ("C11", {(1, 1): 1.0}),
        ("C22", {(2, 2): 1.0}),
        ("C33", {(3, 3): 1.0}),
        ("C12", {(1, 2): 1.0}),
        ("C13", {(1, 3): 1.0}),
        ("C23", {(2, 3): 1.0}),
        ("C44", {(4, 4): 1.0}),
        ("C55", {(5, 5): 1.0}),
        ("C66", {(6, 6): 1.0}),
    ]
    return [n for n, _ in defs], np.stack(
        [_make_P_from_voigt_entries(e) for _, e in defs], axis=0
    )


def _monoclinic_basis() -> tuple[list[str], np.ndarray]:
    """Monoclinic (2/m, 2, m), unique-axis-b setting: 13 independent.

    Non-zero engineering-Voigt entries split into a 4x4 block coupling
    (xx, yy, zz, xz) and a 2x2 block coupling (xy, yz), with
    additionally the (4,6) coupling ``C46``.
    """
    defs = [
        ("C11", {(1, 1): 1.0}),
        ("C22", {(2, 2): 1.0}),
        ("C33", {(3, 3): 1.0}),
        ("C44", {(4, 4): 1.0}),
        ("C55", {(5, 5): 1.0}),
        ("C66", {(6, 6): 1.0}),
        ("C12", {(1, 2): 1.0}),
        ("C13", {(1, 3): 1.0}),
        ("C23", {(2, 3): 1.0}),
        ("C15", {(1, 5): 1.0}),
        ("C25", {(2, 5): 1.0}),
        ("C35", {(3, 5): 1.0}),
        ("C46", {(4, 6): 1.0}),
    ]
    return [n for n, _ in defs], np.stack(
        [_make_P_from_voigt_entries(e) for _, e in defs], axis=0
    )


def _triclinic_basis() -> tuple[list[str], np.ndarray]:
    """Triclinic (-1, 1): all 21 upper-triangular entries independent."""
    names = []
    mats = []
    for i in range(1, 7):
        for j in range(i, 7):
            names.append(f"C{i}{j}")
            mats.append(_make_P_from_voigt_entries({(i, j): 1.0}))
    return names, np.stack(mats, axis=0)


_SYMMETRY_DISPATCH = {
    "cubic":        _cubic_basis,
    "hexagonal":    _hexagonal_basis,
    "trigonal":     _trigonal_basis,
    "tetragonal":   _tetragonal_basis,
    "orthorhombic": _orthorhombic_basis,
    "monoclinic":   _monoclinic_basis,
    "triclinic":    _triclinic_basis,
}


def symmetry_parameterisation(symmetry: str) -> tuple[list[str], np.ndarray]:
    """Return the independent elastic constants and basis matrices.

    Parameters
    ----------
    symmetry : str
        One of ``"cubic"``, ``"hexagonal"``, ``"trigonal"``,
        ``"tetragonal"``, ``"orthorhombic"``, ``"monoclinic"``,
        ``"triclinic"``.

    Returns
    -------
    names : list[str]
        Canonical names ``"C11"``, ``"C12"``, ... of the independent
        constants, length ``N_c``.
    P_stack : ndarray (N_c, 6, 6)
        Mandel-notation basis matrices such that
        ``C = sum_k c_k * P_stack[k]`` reconstructs the full 6x6
        stiffness for any independent-constant vector ``c``.
    """
    key = symmetry.lower()
    if key not in _SYMMETRY_DISPATCH:
        raise ValueError(
            f"Unknown symmetry '{symmetry}'. Available: "
            f"{sorted(_SYMMETRY_DISPATCH.keys())}"
        )
    return _SYMMETRY_DISPATCH[key]()


def stiffness_from_cij(cij: np.ndarray | dict, symmetry: str) -> np.ndarray:
    """Reconstruct the 6x6 Mandel stiffness from independent constants.

    Parameters
    ----------
    cij : ndarray (N_c,) or dict
        Independent-constant vector or a dict keyed by constant name
        (``"C11"`` etc.).
    symmetry : str

    Returns
    -------
    ndarray (6, 6) in Mandel notation.
    """
    names, P = symmetry_parameterisation(symmetry)
    if isinstance(cij, dict):
        missing = [n for n in names if n not in cij]
        if missing:
            raise ValueError(
                f"Missing constants for {symmetry}: {missing}. "
                f"Required: {names}"
            )
        vec = np.array([cij[n] for n in names], dtype=np.float64)
    else:
        vec = np.asarray(cij, dtype=np.float64)
        if vec.shape != (len(names),):
            raise ValueError(
                f"Expected cij of shape ({len(names)},) for {symmetry}; "
                f"got {vec.shape}"
            )
    return np.einsum("k,kij->ij", vec, P)


# -------------------------------------------------------------------
#  Stage matrix construction
# -------------------------------------------------------------------

def build_stage_matrix(
    orientations: np.ndarray,
    strains_lab: np.ndarray,
    weights: np.ndarray,
    P_stack: np.ndarray,
) -> np.ndarray:
    """Build the Hill-method stage design matrix :math:`\\mathbf{A}^{(i)}`.

    Column ``k`` of the returned matrix is

    .. math::

       \\mathbf{A}^{(i)}[:, k]
       = \\sum_g w_g\\,\\mathfrak{U}_g^{\\top}\\,P_k\\,\\mathfrak{U}_g\\,
           \\{\\boldsymbol{\\varepsilon}_g^{\\mathrm{lab}}\\}.

    Parameters
    ----------
    orientations : ndarray (N, 3, 3)
        Per-grain orientation matrices (crystal -> lab).
    strains_lab : ndarray (N, 3, 3)
        Per-grain strain tensors in the lab frame.
    weights : ndarray (N,)
        Normalised weights (sum to 1).  Typically
        ``volumes * confidences`` normalised.
    P_stack : ndarray (N_c, 6, 6)
        Symmetry basis matrices in Mandel notation.

    Returns
    -------
    A : ndarray (6, N_c)
    """
    M = rotation_voigt_mandel(orientations)      # (N, 6, 6) lab->grain
    Mt = np.swapaxes(M, -1, -2)                  # (N, 6, 6) grain->lab
    eps_voigt = tensor_to_voigt(strains_lab)     # (N, 6) lab Mandel

    # Per-grain strain rotated to the grain frame.
    eps_grain = np.einsum("nij,nj->ni", M, eps_voigt)          # (N, 6)
    # Apply each basis matrix to get a per-basis crystal-frame "stress".
    sigma_grain = np.einsum("kij,nj->nki", P_stack, eps_grain)  # (N, Nc, 6)
    # Rotate back to lab.
    sigma_lab = np.einsum("nij,nkj->nki", Mt, sigma_grain)      # (N, Nc, 6)
    # Volume-weighted sum.
    A = np.einsum("n,nki->ik", weights, sigma_lab)              # (6, Nc)
    return A


def _aggregate_basis_rotated(
    orientations: np.ndarray,
    weights: np.ndarray,
    P_stack: np.ndarray,
) -> np.ndarray:
    """Volume-weighted lab-frame basis tensors
    :math:`\\bar{P}_k = \\sum_g w_g\\,\\mathfrak{U}_g^{\\top} P_k \\mathfrak{U}_g`.

    Returns
    -------
    Pbar : ndarray (N_c, 6, 6)
        One matrix per basis element, in Mandel form.
    """
    M = rotation_voigt_mandel(orientations)      # (N, 6, 6) lab->grain
    Mt = np.swapaxes(M, -1, -2)                  # (N, 6, 6) grain->lab
    # Compute Mt @ P_k @ M per grain and average: P_lab_g,k = Mt_g P_k M_g.
    PM = np.einsum("kij,njm->nkim", P_stack, M)          # (N, Nc, 6, 6)
    P_lab = np.einsum("nij,nkjm->nkim", Mt, PM)          # (N, Nc, 6, 6)
    return np.einsum("n,nkij->kij", weights, P_lab)      # (Nc, 6, 6)


def build_stage_matrix_voigt(
    orientations: np.ndarray,
    mean_strain_lab: np.ndarray,
    weights: np.ndarray,
    P_stack: np.ndarray,
) -> np.ndarray:
    """Build the Voigt (iso-strain) stage design matrix.

    Assumes every grain carries the same lab-frame strain
    :math:`\\bar{\\boldsymbol{\\varepsilon}}`.  Column ``k`` is

    .. math::

       \\mathbf{A}^{(i)}_{\\mathrm{Voigt}}[:, k]
       = \\Bigl(\\sum_g w_g\\,\\mathfrak{U}_g^{\\top}\\,P_k\\,
                \\mathfrak{U}_g\\Bigr)\\,\\{\\bar{\\boldsymbol{\\varepsilon}}\\}.

    Parameters
    ----------
    orientations : ndarray (N, 3, 3)
    mean_strain_lab : ndarray (3, 3)
        Macroscopic strain assumed uniform across all grains.
    weights : ndarray (N,)
    P_stack : ndarray (N_c, 6, 6)

    Returns
    -------
    A : ndarray (6, N_c)
    """
    Pbar = _aggregate_basis_rotated(orientations, weights, P_stack)   # (Nc,6,6)
    eps_v = tensor_to_voigt(mean_strain_lab)                          # (6,)
    return np.einsum("kij,j->ik", Pbar, eps_v)                        # (6, Nc)


def build_stage_matrix_reuss(
    orientations: np.ndarray,
    applied_stress: np.ndarray,
    weights: np.ndarray,
    P_stack: np.ndarray,
) -> np.ndarray:
    """Build the Reuss (iso-stress) stage design matrix in compliance space.

    Assumes every grain carries :math:`\\boldsymbol{\\sigma}_{\\mathrm{app}}`.
    Unknown is the compliance :math:`\\mathbf{S}_{\\mathrm{crystal}}
    = \\sum_k s_k P_k` (same Mandel basis as stiffness).  Column ``k``:

    .. math::

       \\mathbf{A}^{(i)}_{\\mathrm{Reuss}}[:, k]
       = \\Bigl(\\sum_g w_g\\,\\mathfrak{U}_g^{\\top}\\,P_k\\,
                \\mathfrak{U}_g\\Bigr)\\,\\{\\boldsymbol{\\sigma}_{\\mathrm{app}}\\}.

    Parameters
    ----------
    orientations : ndarray (N, 3, 3)
    applied_stress : ndarray (3, 3)
    weights : ndarray (N,)
    P_stack : ndarray (N_c, 6, 6)

    Returns
    -------
    A : ndarray (6, N_c)
        Column k gives the contribution of compliance component
        :math:`s_k` to the volume-averaged lab-frame strain.
    """
    Pbar = _aggregate_basis_rotated(orientations, weights, P_stack)   # (Nc,6,6)
    sig_v = tensor_to_voigt(applied_stress)                           # (6,)
    return np.einsum("kij,j->ik", Pbar, sig_v)                        # (6, Nc)


# -------------------------------------------------------------------
#  Public API
# -------------------------------------------------------------------

_VALID_METHODS = ("hill", "voigt", "reuss")


def fit_single_crystal_stiffness(
    stages: list[dict],
    symmetry: str,
    fit_eps_iso: bool = True,
    initial_stiffness: Optional[np.ndarray] = None,
    material_hint: Optional[str] = None,
    min_confidence: float = 0.0,
    cond_threshold: float = 1e3,
    max_iter: int = 20,
    tol: float = 1e-8,
    method: str = "hill",
) -> dict:
    """Fit single-crystal elastic constants from HEDM load stages.

    Each stage is a ``dict`` with keys:

    - ``orient``        : ndarray (N, 3, 3) crystal->lab orientations
    - ``strain``        : ndarray (N, 3, 3) lab-frame strain tensors
    - ``volumes``       : ndarray (N,) grain volumes
    - ``applied_stress``: ndarray (3, 3) macroscopic applied stress
      (set to zeros for an unloaded stage)
    - ``confidences``   : ndarray (N,), optional, for
      confidence-weighted averaging
    - ``is_unloaded``   : bool, optional; if True, the stage pins
      :math:`\\varepsilon_{\\mathrm{iso}}` during the coupled
      :math:`d_0`/:math:`\\mathbf{C}` iteration
    - ``macro_strain``  : ndarray (3, 3), optional; macroscopic strain
      used by the Voigt variant in place of the volume-weighted mean
      of the measured strains.  Ignored unless ``method="voigt"``.

    Parameters
    ----------
    stages : list[dict]
    symmetry : str
        One of the keys of :func:`symmetry_parameterisation`.
    fit_eps_iso : bool
        If True and at least one stage is marked ``is_unloaded``,
        alternate between fitting :math:`\\varepsilon_{\\mathrm{iso}}`
        from the unloaded stage and refitting :math:`\\mathbf{C}`
        from the corrected strains until convergence.  Only supported
        when ``method="hill"``; for Voigt/Reuss ``eps_iso`` is
        estimated in closed form from the unloaded-stage mean strain
        (one-shot, no iteration).
    initial_stiffness : ndarray (6, 6), optional
        Initial guess for the coupled iteration.  Defaults to a
        literature value from ``material_hint`` if given, or to a
        Voigt-average isotropic proxy otherwise.
    material_hint : str, optional
        Name in the built-in stiffness library for initialisation.
    min_confidence : float
        Threshold on per-grain confidence for inclusion.
    cond_threshold : float
        Flag the fit as ill-conditioned if the stacked design matrix
        condition number exceeds this value.  Default 1e3.
    max_iter : int
        Maximum iterations for the coupled :math:`d_0`/:math:`\\mathbf{C}`
        fit.  Ignored when ``fit_eps_iso=False``.
    tol : float
        Relative change in :math:`\\mathbf{c}` below which the coupled
        iteration is considered converged.
    method : str
        Homogenisation assumption: ``"hill"`` (default, measured
        per-grain strains + Hill's lemma), ``"voigt"`` (iso-strain),
        or ``"reuss"`` (iso-stress, solved in compliance space then
        inverted).  See the module docstring for the equations.

    Returns
    -------
    dict with keys:

    - ``stiffness`` : (6, 6) fitted stiffness in Mandel notation
    - ``cij``       : dict of independent constants by name
    - ``cij_se``    : dict of per-constant standard errors.  For
      ``method="reuss"`` propagated from the compliance covariance via
      the first-order delta method.
    - ``covariance``: (N_c, N_c) covariance of the fitted *stiffness*
      vector (delta-method-propagated for Reuss).
    - ``compliance``: (6, 6) fitted compliance (only for Reuss; else
      ``None``).
    - ``sij``       : dict of compliance constants (Reuss only).
    - ``cij_names`` : list[str]
    - ``condition_number`` : condition number of the stacked A
    - ``well_conditioned`` : bool
    - ``eps_iso``   : float (0.0 if ``fit_eps_iso=False`` or no unloaded stage)
    - ``residual_norm`` : stacked fit residual norm
    - ``n_iter``    : int, iterations used in the coupled fit
    - ``symmetry``  : str
    - ``method``    : str, one of ``"hill" | "voigt" | "reuss"``
    """
    method_key = method.lower()
    if method_key not in _VALID_METHODS:
        raise ValueError(
            f"Unknown method '{method}'. Valid: {_VALID_METHODS}"
        )

    names, P_stack = symmetry_parameterisation(symmetry)
    N_c = len(names)

    if not stages:
        raise ValueError("stages must contain at least one entry")

    # Normalise inputs and compute per-stage weights
    prepped = _prep_stages(stages, min_confidence)

    # Identify the unloaded stage (if any)
    unloaded_idx = [i for i, s in enumerate(prepped) if s["is_unloaded"]]
    if fit_eps_iso and not unloaded_idx:
        fit_eps_iso = False  # nothing to pin against

    # The coupled d0/C iteration is only defined for the Hill method
    # because it uses per-grain strains (Paper II §3.3).  For Voigt
    # and Reuss we take the one-shot closed form below.
    coupled = fit_eps_iso and method_key == "hill"

    eps_iso = 0.0
    n_iter = 1
    compliance = None
    sij = None

    if not coupled:
        # For Voigt/Reuss with an unloaded stage, estimate eps_iso
        # directly from the volume-averaged trace of the unloaded-stage
        # strain (Paper II §3.2 collapsed to one shot).
        if fit_eps_iso and method_key in ("voigt", "reuss"):
            s0 = prepped[unloaded_idx[0]]
            eps_iso = _eps_iso_from_mean_strain(s0)
            # Apply the isotropic correction to every stage's strain.
            I3 = np.eye(3)
            prepped_use = []
            for s in prepped:
                s2 = dict(s)
                s2["strain"] = s["strain"] - eps_iso * I3[None, :, :]
                prepped_use.append(s2)
        else:
            prepped_use = prepped

        c_vec, cov, cond_num, residual, compliance, sij = _solve_stacked(
            prepped_use, P_stack, symmetry_names=names, method=method_key,
        )
    else:
        # Coupled d0 + C fit (Hill only).  Start from strains_corrected = strains.
        C_current = _initial_stiffness(
            initial_stiffness, material_hint, symmetry, names, P_stack,
        )
        c_prev = None
        for n_iter in range(1, max_iter + 1):
            # (a) Fit eps_iso using the current C on the unloaded stage
            stage0 = prepped[unloaded_idx[0]]
            d0_res = d0_correction_strain_level(
                stage0["strain"], C_current,
                stage0["orient"], stage0["volumes"],
                confidences=stage0.get("confidences"),
                applied_stress=stage0["applied_stress"],
                min_confidence=min_confidence,
            )
            eps_iso = d0_res["eps_iso"]

            # (b) Subtract eps_iso from every stage's strains
            prepped_corrected = []
            I3 = np.eye(3)
            for s in prepped:
                s2 = dict(s)
                s2["strain"] = s["strain"] - eps_iso * I3[None, :, :]
                prepped_corrected.append(s2)

            # (c) Refit C from corrected strains (Hill)
            c_vec, cov, cond_num, residual, _, _ = _solve_stacked(
                prepped_corrected, P_stack, symmetry_names=names,
                method="hill",
            )
            C_current = np.einsum("k,kij->ij", c_vec, P_stack)

            if c_prev is not None:
                denom = max(np.linalg.norm(c_prev), 1e-12)
                if np.linalg.norm(c_vec - c_prev) / denom < tol:
                    break
            c_prev = c_vec

    # Final stiffness
    C_fit = np.einsum("k,kij->ij", c_vec, P_stack)

    # Standard errors on the diagonal of the covariance
    cij = {n: float(c_vec[i]) for i, n in enumerate(names)}
    cij_se = {n: float(np.sqrt(max(cov[i, i], 0.0))) for i, n in enumerate(names)}

    return {
        "stiffness":        C_fit,
        "cij":              cij,
        "cij_se":           cij_se,
        "cij_names":        names,
        "covariance":       cov,
        "compliance":       compliance,
        "sij":              sij,
        "condition_number": float(cond_num),
        "well_conditioned": bool(cond_num < cond_threshold),
        "eps_iso":          float(eps_iso),
        "residual_norm":    float(residual),
        "n_iter":           int(n_iter),
        "symmetry":         symmetry.lower(),
        "method":           method_key,
    }


# -------------------------------------------------------------------
#  Internal helpers
# -------------------------------------------------------------------

def _prep_stages(stages: list[dict], min_confidence: float) -> list[dict]:
    """Validate and normalise each stage dict; compute weights."""
    out = []
    for i, s in enumerate(stages):
        needed = ("orient", "strain", "volumes", "applied_stress")
        for key in needed:
            if key not in s:
                raise ValueError(f"stage {i} missing key '{key}'")
        orient   = np.asarray(s["orient"], dtype=np.float64)
        strain   = np.asarray(s["strain"], dtype=np.float64)
        volumes  = np.asarray(s["volumes"], dtype=np.float64)
        applied  = np.asarray(s["applied_stress"], dtype=np.float64)
        confidences = s.get("confidences")
        if confidences is not None:
            confidences = np.asarray(confidences, dtype=np.float64)

        N = orient.shape[0]
        if strain.shape != (N, 3, 3):
            raise ValueError(
                f"stage {i}: strain shape {strain.shape} "
                f"inconsistent with orient shape {orient.shape}"
            )
        if volumes.shape != (N,):
            raise ValueError(
                f"stage {i}: volumes shape {volumes.shape} inconsistent"
            )
        if applied.shape != (3, 3):
            raise ValueError(
                f"stage {i}: applied_stress must be (3, 3); got {applied.shape}"
            )

        if confidences is not None and min_confidence > 0:
            mask = confidences >= min_confidence
        else:
            mask = np.ones(N, dtype=bool)

        w = effective_weights(
            volumes[mask],
            confidences[mask] if confidences is not None else None,
        )

        macro_strain = s.get("macro_strain")
        if macro_strain is not None:
            macro_strain = np.asarray(macro_strain, dtype=np.float64)
            if macro_strain.shape != (3, 3):
                raise ValueError(
                    f"stage {i}: macro_strain must be (3, 3); "
                    f"got {macro_strain.shape}"
                )

        out.append({
            "orient":         orient[mask],
            "strain":         strain[mask],
            "volumes":        volumes[mask],
            "applied_stress": applied,
            "confidences":    confidences[mask] if confidences is not None else None,
            "weights":        w,
            "is_unloaded":    bool(s.get("is_unloaded", False)),
            "macro_strain":   macro_strain,
        })
    return out


def _solve_stacked(
    prepped: list[dict],
    P_stack: np.ndarray,
    symmetry_names: list[str],
    method: str = "hill",
) -> tuple[np.ndarray, np.ndarray, float, float, Optional[np.ndarray], Optional[dict]]:
    """Stack per-stage equations and solve in the least-squares sense.

    Parameters
    ----------
    prepped : list[dict]
        Output of :func:`_prep_stages`.
    P_stack : ndarray (N_c, 6, 6)
    symmetry_names : list[str]
    method : str
        ``"hill"``, ``"voigt"``, or ``"reuss"``.

    Returns
    -------
    c_vec : ndarray (N_c,)
        Stiffness coefficients (already inverted from compliance for
        Reuss).
    cov : ndarray (N_c, N_c)
        Covariance of ``c_vec`` (delta-method-propagated for Reuss).
    cond_num : float
    residual_norm : float
    compliance : ndarray (6, 6) or None
        Fitted compliance matrix (Reuss only).
    sij : dict or None
        Compliance coefficients by name (Reuss only).
    """
    rows_A = []
    rows_b = []
    for s in prepped:
        if method == "hill":
            A = build_stage_matrix(
                s["orient"], s["strain"], s["weights"], P_stack,
            )
            b = tensor_to_voigt(s["applied_stress"])
        elif method == "voigt":
            mean_eps = _voigt_mean_strain(s)
            A = build_stage_matrix_voigt(
                s["orient"], mean_eps, s["weights"], P_stack,
            )
            b = tensor_to_voigt(s["applied_stress"])
        elif method == "reuss":
            # Volume-weighted mean measured strain is the observable.
            mean_eps = _volume_weighted_strain(s["strain"], s["weights"])
            A = build_stage_matrix_reuss(
                s["orient"], s["applied_stress"], s["weights"], P_stack,
            )
            b = tensor_to_voigt(mean_eps)
        else:
            raise ValueError(f"Unknown method '{method}'")
        rows_A.append(A)
        rows_b.append(b)

    A_stacked = np.vstack(rows_A)            # (6 * Nstage, Nc)
    b_stacked = np.concatenate(rows_b)       # (6 * Nstage,)

    # Rank check — fail fast on under-specified systems
    if A_stacked.shape[0] < A_stacked.shape[1]:
        raise ValueError(
            f"Under-determined: {A_stacked.shape[0]} equations for "
            f"{A_stacked.shape[1]} unknowns ({symmetry_names}). "
            f"Add more load stages."
        )

    # Least-squares fit (in compliance space for Reuss, stiffness otherwise)
    x_vec, _, _, _ = np.linalg.lstsq(A_stacked, b_stacked, rcond=None)

    cond_num = np.linalg.cond(A_stacked)

    try:
        ATA_inv = np.linalg.inv(A_stacked.T @ A_stacked)
    except np.linalg.LinAlgError:
        ATA_inv = np.linalg.pinv(A_stacked.T @ A_stacked)

    pred = A_stacked @ x_vec
    r = b_stacked - pred
    dof = max(A_stacked.shape[0] - A_stacked.shape[1], 1)
    sigma_sq = float(r @ r) / dof
    cov_x = ATA_inv * sigma_sq
    residual_norm = float(np.linalg.norm(r))

    if method != "reuss":
        return x_vec, cov_x, float(cond_num), residual_norm, None, None

    # Reuss: x_vec holds the compliance coefficients s_k.  Convert to
    # stiffness coefficients c_k by building S, inverting, and reading
    # off the components along the P_k basis.  Propagate covariance
    # via the first-order delta method:
    #
    #     c = f(s),   df/ds_k = -S^{-1} P_k S^{-1}   (at fixed s)
    #
    # Reading the k-th entry of ``c`` uses the inverse basis projector
    # ``proj_k`` with tr(proj_k P_l) = delta_kl (Mandel inner product).
    S = np.einsum("k,kij->ij", x_vec, P_stack)
    try:
        C_from_S = np.linalg.inv(S)
    except np.linalg.LinAlgError as err:
        raise RuntimeError(
            "Reuss fit produced a singular compliance matrix; "
            "check stage conditioning and applied-stress diversity."
        ) from err

    # Project C_from_S onto the P_k basis: c_k = <proj_k, C>
    proj = _basis_dual(P_stack)                                # (N_c, 6, 6)
    c_vec = np.einsum("kij,ij->k", proj, C_from_S)             # (N_c,)

    # Jacobian J[k, l] = dc_k / ds_l.
    #   dC/ds_l = -C P_l C          (matrix Mandel form, both in Mandel)
    #   dc_k/ds_l = <proj_k, dC/ds_l> = -<proj_k, C P_l C>
    CP = np.einsum("ij,ljk->lik", C_from_S, P_stack)           # (N_c, 6, 6)
    CPC = np.einsum("lij,jk->lik", CP, C_from_S)               # (N_c, 6, 6)
    J = -np.einsum("kij,lij->kl", proj, CPC)                   # (N_c, N_c)

    cov_c = J @ cov_x @ J.T

    sij = {n: float(x_vec[i]) for i, n in enumerate(symmetry_names)}
    return c_vec, cov_c, float(cond_num), residual_norm, S, sij


def _volume_weighted_strain(
    strain_lab: np.ndarray, weights: np.ndarray,
) -> np.ndarray:
    """Volume-weighted mean of per-grain lab-frame strain tensors."""
    return np.einsum("n,nij->ij", weights, strain_lab)


def _voigt_mean_strain(stage: dict) -> np.ndarray:
    """Mean strain used by the Voigt variant.

    Priority: stage-provided ``macro_strain`` override (e.g. a
    load-frame extensometer value) if present, else the volume-weighted
    mean of measured per-grain strains.
    """
    macro = stage.get("macro_strain")
    if macro is not None:
        arr = np.asarray(macro, dtype=np.float64)
        if arr.shape != (3, 3):
            raise ValueError(
                f"macro_strain must be (3, 3); got {arr.shape}"
            )
        return arr
    return _volume_weighted_strain(stage["strain"], stage["weights"])


def _eps_iso_from_mean_strain(stage: dict) -> float:
    """Closed-form eps_iso for Voigt/Reuss from an unloaded stage.

    Under zero applied stress the Voigt and Reuss aggregates both
    predict zero mean strain.  Any measured mean trace is therefore
    attributed to a spherical :math:`d_0` bias.  Returns
    :math:`\\varepsilon_{\\mathrm{iso}} = \\tfrac{1}{3}\\,\\mathrm{tr}\\,
    \\bar{\\boldsymbol{\\varepsilon}}`.
    """
    mean_eps = _volume_weighted_strain(stage["strain"], stage["weights"])
    return float(np.trace(mean_eps) / 3.0)


def _basis_dual(P_stack: np.ndarray) -> np.ndarray:
    """Dual projectors onto the symmetry basis using the Mandel inner product.

    Returns ``proj`` such that ``sum_ij proj_k[i,j] * P_l[i,j]`` equals
    the Kronecker delta.  Built by taking the Gram matrix
    ``G[k,l] = <P_k, P_l>_F`` (Frobenius / Mandel trace inner product)
    and forming ``proj_k = sum_l G^{-1}[k,l] P_l``.

    Used to read off the independent-constant coefficients of a given
    6x6 Mandel tensor when it is known to lie in the span of
    ``P_stack``.
    """
    G = np.einsum("kij,lij->kl", P_stack, P_stack)
    G_inv = np.linalg.inv(G)
    return np.einsum("kl,lij->kij", G_inv, P_stack)


def _initial_stiffness(
    initial_stiffness: Optional[np.ndarray],
    material_hint: Optional[str],
    symmetry: str,
    names: list[str],
    P_stack: np.ndarray,
) -> np.ndarray:
    """Resolve the initial stiffness used by the coupled iteration.

    Preference order: explicit ``initial_stiffness``, library lookup
    by ``material_hint``, isotropic proxy.  The initial guess only
    seeds the ``eps_iso`` fit; for free-standing unloaded stages the
    stiffness magnitude cancels (Paper II, scale-invariance), so
    rough initialisation is acceptable.
    """
    if initial_stiffness is not None:
        C0 = np.asarray(initial_stiffness, dtype=np.float64)
        if C0.shape != (6, 6):
            raise ValueError(
                f"initial_stiffness must be (6, 6); got {C0.shape}"
            )
        return C0

    if material_hint is not None:
        from .materials import get_stiffness
        try:
            return get_stiffness(material_hint)
        except ValueError:
            pass  # fall through to proxy

    # Isotropic proxy: C11 = 200, C12 = 80, C44 = 60 (GPa-scale); the
    # scale cancels for the unloaded-stage eps_iso fit.
    C_iso = np.zeros((6, 6))
    C_iso[0, 0] = C_iso[1, 1] = C_iso[2, 2] = 200.0
    C_iso[0, 1] = C_iso[1, 0] = C_iso[0, 2] = C_iso[2, 0] = 80.0
    C_iso[1, 2] = C_iso[2, 1] = 80.0
    C_iso[3, 3] = C_iso[4, 4] = C_iso[5, 5] = 120.0  # 2 * C44 Mandel
    return C_iso
