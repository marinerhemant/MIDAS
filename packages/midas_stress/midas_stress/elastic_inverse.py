"""Single-crystal elastic constant recovery from HEDM load stages.

Inverts the macroscopic equilibrium constraint across one or more load
stages to fit the independent components of the single-crystal
stiffness tensor :math:`\\mathbf{C}_{\\mathrm{crystal}}`.

For each load stage :math:`i` with applied macroscopic stress
:math:`\\boldsymbol{\\sigma}_{\\mathrm{app}}^{(i)}` and per-grain
lab-frame strains :math:`\\{\\boldsymbol{\\varepsilon}_g^{(i)}\\}`,
mechanical equilibrium requires

.. math::

   \\{\\boldsymbol{\\sigma}_{\\mathrm{app}}^{(i)}\\}
   = \\sum_g w_g\\,\\mathfrak{U}_g^{\\top}\\,
        \\mathbf{C}_{\\mathrm{crystal}}\\,\\mathfrak{U}_g\\,
        \\{\\boldsymbol{\\varepsilon}_g^{(i)}\\},

where :math:`\\mathfrak{U}_g` is the 6x6 Mandel rotation matrix
(Paper I Eq. 14) and :math:`w_g` the volume weights.

Parameterising :math:`\\mathbf{C}_{\\mathrm{crystal}} = \\sum_k c_k P_k`
by symmetry-specific basis matrices :math:`P_k` gives a linear system

.. math::

   \\mathbf{A}^{(i)}\\,\\mathbf{c}
   = \\{\\boldsymbol{\\sigma}_{\\mathrm{app}}^{(i)}\\},

with :math:`\\mathbf{A}^{(i)} \\in \\mathbb{R}^{6\\times N_c}`.
Stacking stages yields a weighted least-squares problem for
:math:`\\mathbf{c}`.
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
    """Build the stage design matrix :math:`\\mathbf{A}^{(i)}`.

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


# -------------------------------------------------------------------
#  Public API
# -------------------------------------------------------------------

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

    Parameters
    ----------
    stages : list[dict]
    symmetry : str
        One of the keys of :func:`symmetry_parameterisation`.
    fit_eps_iso : bool
        If True and at least one stage is marked ``is_unloaded``,
        alternate between fitting :math:`\\varepsilon_{\\mathrm{iso}}`
        from the unloaded stage and refitting :math:`\\mathbf{C}`
        from the corrected strains until convergence.
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

    Returns
    -------
    dict with keys:

    - ``stiffness`` : (6, 6) fitted stiffness in Mandel notation
    - ``cij``       : dict of independent constants by name
    - ``cij_se``    : dict of per-constant standard errors
    - ``covariance``: (N_c, N_c) covariance of the fitted vector
    - ``cij_names`` : list[str]
    - ``condition_number`` : condition number of the stacked A
    - ``well_conditioned`` : bool
    - ``eps_iso``   : float (0.0 if ``fit_eps_iso=False`` or no unloaded stage)
    - ``residual_norm`` : ``||A c - sigma_app||`` (stacked)
    - ``n_iter``    : int, iterations used in the coupled fit
    - ``symmetry``  : str
    """
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

    # Initial stiffness for the coupled iteration
    C_current = _initial_stiffness(
        initial_stiffness, material_hint, symmetry, names, P_stack,
    )

    eps_iso = 0.0
    n_iter = 1

    if not fit_eps_iso:
        c_vec, cov, cond_num, residual = _solve_stacked(
            prepped, P_stack, symmetry_names=names
        )
    else:
        # Coupled d0 + C fit.  Start from strains_corrected = strains.
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

            # (c) Refit C from corrected strains
            c_vec, cov, cond_num, residual = _solve_stacked(
                prepped_corrected, P_stack, symmetry_names=names,
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
        "condition_number": float(cond_num),
        "well_conditioned": bool(cond_num < cond_threshold),
        "eps_iso":          float(eps_iso),
        "residual_norm":    float(residual),
        "n_iter":           int(n_iter),
        "symmetry":         symmetry.lower(),
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

        out.append({
            "orient":         orient[mask],
            "strain":         strain[mask],
            "volumes":        volumes[mask],
            "applied_stress": applied,
            "confidences":    confidences[mask] if confidences is not None else None,
            "weights":        w,
            "is_unloaded":    bool(s.get("is_unloaded", False)),
        })
    return out


def _solve_stacked(
    prepped: list[dict],
    P_stack: np.ndarray,
    symmetry_names: list[str],
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Stack per-stage equations and solve in the least-squares sense."""
    rows_A = []
    rows_b = []
    for s in prepped:
        A = build_stage_matrix(
            s["orient"], s["strain"], s["weights"], P_stack,
        )
        b = tensor_to_voigt(s["applied_stress"])  # (6,) Mandel
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

    # Least-squares fit
    c_vec, residuals, rank, _ = np.linalg.lstsq(A_stacked, b_stacked, rcond=None)

    # Condition number of the design matrix
    cond_num = np.linalg.cond(A_stacked)

    # Covariance: (A^T A)^{-1} * sigma^2 where sigma^2 estimated
    # from the residual.  When the system is well-determined, the
    # residual is the projection of noise onto the null-space of A.
    try:
        ATA_inv = np.linalg.inv(A_stacked.T @ A_stacked)
    except np.linalg.LinAlgError:
        ATA_inv = np.linalg.pinv(A_stacked.T @ A_stacked)

    # Residual norm & dof
    pred = A_stacked @ c_vec
    r = b_stacked - pred
    dof = max(A_stacked.shape[0] - A_stacked.shape[1], 1)
    sigma_sq = float(r @ r) / dof
    cov = ATA_inv * sigma_sq

    residual_norm = float(np.linalg.norm(r))
    return c_vec, cov, float(cond_num), residual_norm


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
