"""Mechanical equilibrium constraints for polycrystalline stress analysis.

Implements:
- Volume-average stress constraint (stress-level correction)
- Strain-level d0 correction (works for all crystal symmetries)
- Confidence-weighted variants for incomplete grain populations
- Uncertainty estimation for the equilibrium correction

The strain-level d0 correction (``d0_correction_strain_level``) is the
physically correct approach for all crystal symmetries.  For non-cubic
materials, a d0 error produces orientation-dependent stress artifacts
with both hydrostatic and deviatoric components; the strain-level
correction handles this correctly by solving for the scalar isotropic
strain error before applying Hooke's law.

As of 0.6.0, the two core constraint functions
(``volume_average_stress_constraint``, ``hydrostatic_deviatoric_decomposition``)
accept torch.Tensor inputs transparently. The iterative scipy-driven
utilities (``recover_d0``, ``equilibrium_correction_uncertainty``, ...)
remain NumPy-only.
"""


from __future__ import annotations
from typing import Optional, Tuple

import numpy as np
from ._optional import torch

from .tensor import tensor_to_voigt, voigt_to_tensor, rotation_voigt_mandel


def _is_torch(*args) -> bool:
    return any(isinstance(a, torch.Tensor) for a in args)


def volume_average_stress_constraint(
    stresses,
    volumes,
    applied_stress=None,
):
    """Apply volume-average stress constraint (FF-1).

    Enforces: sum(V_g * sigma_g) / V_total = sigma_applied

    Parameters
    ----------
    stresses : ndarray or torch.Tensor (N, 3, 3) or (N, 6)
        Per-grain stress tensors.
    volumes : ndarray or torch.Tensor (N,)
        Grain volumes (relative sizes suffice).
    applied_stress : ndarray, torch.Tensor, or None, optional
        Applied macroscopic stress. Default: zero (unloaded sample).

    Returns
    -------
    Same backend as input, same shape as input.
    """
    is_voigt = stresses.ndim == 2 and stresses.shape[-1] == 6
    sig = stresses.clone() if isinstance(stresses, torch.Tensor) else stresses.copy()
    if not is_voigt:
        sig = tensor_to_voigt(sig)

    if applied_stress is None:
        if isinstance(sig, torch.Tensor):
            sig_app = torch.zeros(6, dtype=sig.dtype, device=sig.device)
        else:
            sig_app = np.zeros(6)
    elif applied_stress.shape == (3, 3):
        sig_app = tensor_to_voigt(applied_stress)
    else:
        sig_app = applied_stress.clone() if isinstance(applied_stress, torch.Tensor) \
            else applied_stress.copy()

    V_total = volumes.sum()
    w = volumes / V_total

    if isinstance(sig, torch.Tensor):
        sig_avg = (w.unsqueeze(-1) * sig).sum(dim=0)
    else:
        sig_avg = np.sum(w[:, None] * sig, axis=0)
    delta_sig = sig_app - sig_avg

    if isinstance(sig, torch.Tensor):
        sig_corrected = sig + delta_sig.unsqueeze(0)
    else:
        sig_corrected = sig + delta_sig[None, :]

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
    conditioned). The hydrostatic part is fixed via the equilibrium
    constraint, removing dependence on the ambiguous d0.

    Parameters
    ----------
    stresses : ndarray (N, 3, 3)
        Per-grain stress tensors.
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
        Full stress tensors with equilibrium-consistent hydrostatic part.
    """
    if isinstance(stresses, torch.Tensor):
        return _hydrostatic_deviatoric_decomposition_torch(
            stresses, volumes, applied_stress,
        )

    if applied_stress is None:
        applied_stress = np.zeros((3, 3))

    I = np.eye(3)
    V_total = volumes.sum()
    w = volumes / V_total

    hydro_raw = np.trace(stresses, axis1=-2, axis2=-1) / 3.0
    deviatoric = stresses - hydro_raw[:, None, None] * I[None, :, :]

    # Fix hydrostatic via equilibrium
    target_hydro = np.trace(applied_stress) / 3.0
    current_avg_hydro = np.sum(w * hydro_raw)
    hydro_shift = target_hydro - current_avg_hydro
    hydro_corrected = hydro_raw + hydro_shift

    # Fix deviatoric equilibrium
    dev_applied = applied_stress - (np.trace(applied_stress) / 3.0) * I
    dev_avg = np.sum(w[:, None, None] * deviatoric, axis=0)
    dev_correction = dev_applied - dev_avg
    deviatoric_corrected = deviatoric + dev_correction[None, :, :]

    corrected = hydro_corrected[:, None, None] * I[None, :, :] + deviatoric_corrected
    return hydro_corrected, deviatoric_corrected, corrected


def _hydrostatic_deviatoric_decomposition_torch(stresses, volumes, applied_stress):
    """Torch path for `hydrostatic_deviatoric_decomposition`."""
    dtype, device = stresses.dtype, stresses.device
    if applied_stress is None:
        applied_stress = torch.zeros((3, 3), dtype=dtype, device=device)
    elif not isinstance(applied_stress, torch.Tensor):
        applied_stress = torch.as_tensor(applied_stress, dtype=dtype, device=device)
    if not isinstance(volumes, torch.Tensor):
        volumes = torch.as_tensor(volumes, dtype=dtype, device=device)

    I = torch.eye(3, dtype=dtype, device=device)
    V_total = volumes.sum()
    w = volumes / V_total

    hydro_raw = torch.diagonal(stresses, dim1=-2, dim2=-1).sum(dim=-1) / 3.0
    deviatoric = stresses - hydro_raw[:, None, None] * I

    target_hydro = torch.diagonal(applied_stress, dim1=-2, dim2=-1).sum() / 3.0
    current_avg_hydro = (w * hydro_raw).sum()
    hydro_shift = target_hydro - current_avg_hydro
    hydro_corrected = hydro_raw + hydro_shift

    dev_applied = applied_stress - target_hydro * I
    dev_avg = (w[:, None, None] * deviatoric).sum(dim=0)
    dev_correction = dev_applied - dev_avg
    deviatoric_corrected = deviatoric + dev_correction[None, :, :]

    corrected = hydro_corrected[:, None, None] * I + deviatoric_corrected
    return hydro_corrected, deviatoric_corrected, corrected


# -------------------------------------------------------------------
#  Strain-level d0 correction (all crystal symmetries)
# -------------------------------------------------------------------

def d0_correction_strain_level(
    strains: np.ndarray,
    stiffness: np.ndarray,
    orientations: np.ndarray,
    volumes: np.ndarray,
    confidences: Optional[np.ndarray] = None,
    applied_stress: Optional[np.ndarray] = None,
    min_confidence: float = 0.0,
) -> dict:
    """Correct d0 error at the strain level (all crystal symmetries).

    A d0 error acts as an isotropic strain perturbation eps_iso * I
    added to every grain's strain tensor.  This function finds the
    scalar eps_iso that best satisfies the macroscopic equilibrium
    condition, subtracts it from all strains, and recomputes stresses.

    Unlike the stress-level hydrostatic shift, this approach correctly
    handles non-cubic materials where a d0 error produces
    orientation-dependent stress artifacts with both hydrostatic and
    deviatoric components.

    Algorithm:
        1. Compute per-grain lab-frame stiffness:
           C_lab_g = M_g^T @ C_crystal @ M_g
        2. Compute volume-averaged lab-frame stiffness:
           <C_lab> = sum(w_g * C_lab_g)
        3. Compute the "d0 response vector":
           a = <C_lab> @ {I}, where {I} = [1,1,1,0,0,0]^T
        4. Compute the stress residual:
           b = <sigma_measured> - sigma_applied (in Voigt)
        5. Solve for eps_iso via least squares:
           eps_iso = (a^T @ b) / (a^T @ a)
        6. Correct strains: eps_corrected = eps - eps_iso * I
        7. Recompute stresses from corrected strains.

    Parameters
    ----------
    strains : ndarray (N, 3, 3)
        Per-grain strain tensors in the lab frame.
    stiffness : ndarray (6, 6)
        Single-crystal stiffness in Voigt-Mandel notation (crystal frame).
    orientations : ndarray (N, 3, 3)
        Orientation matrices (crystal -> lab).
    volumes : ndarray (N,)
        Grain volumes.
    confidences : ndarray (N,), optional
        Per-grain confidence (0 to 1).
    applied_stress : ndarray (3, 3), optional
        Applied macroscopic stress. Default: zero (free-standing).
    min_confidence : float
        Minimum confidence for contributing to the average.

    Returns
    -------
    dict with keys:
        'eps_iso': float — the fitted isotropic strain correction
        'strains_corrected': ndarray (N, 3, 3) — corrected strains
        'stresses_corrected': ndarray (N, 3, 3) — stresses from
            corrected strains
        'stresses_raw': ndarray (N, 3, 3) — stresses before correction
        'residual_norm_before': float — ||<sigma> - sigma_app|| before
        'residual_norm_after': float — ... after correction
        'uncertainty': dict — uncertainty information
    """
    from .hooke import hooke_stress

    N = strains.shape[0]
    if applied_stress is None:
        applied_stress = np.zeros((3, 3))

    # Build mask
    if confidences is not None and min_confidence > 0:
        mask = confidences >= min_confidence
    else:
        mask = np.ones(N, dtype=bool)

    w = effective_weights(
        volumes[mask],
        confidences[mask] if confidences is not None else None,
    )

    # Step 1: Compute per-grain lab-frame stiffness C_lab_g
    # M maps lab->grain, so C_lab = M^T @ C @ M
    M_all = rotation_voigt_mandel(orientations)  # (N, 6, 6) lab->grain
    Mt_all = np.swapaxes(M_all, -1, -2)         # grain->lab
    C_lab_all = Mt_all @ stiffness @ M_all       # (N, 6, 6)

    # Step 2: Volume-averaged lab-frame stiffness (from masked grains)
    C_lab_avg = np.sum(w[:, None, None] * C_lab_all[mask], axis=0)  # (6, 6)

    # Step 3: d0 response vector
    I_voigt = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    a = C_lab_avg @ I_voigt  # (6,)

    # Step 4: Compute raw stresses and volume-average residual
    stresses_raw = hooke_stress(strains, stiffness, orient=orientations,
                                frame="lab")
    sig_voigt_all = tensor_to_voigt(stresses_raw)  # (N, 6)
    sig_avg = np.sum(w[:, None] * sig_voigt_all[mask], axis=0)  # (6,)
    sig_app_voigt = tensor_to_voigt(applied_stress)
    b = sig_avg - sig_app_voigt  # (6,) residual

    # Step 5: Least-squares fit for eps_iso
    eps_iso = float(np.dot(a, b) / np.dot(a, a))

    # Step 6: Correct strains
    I_3x3 = np.eye(3)
    strains_corrected = strains - eps_iso * I_3x3[None, :, :]

    # Step 7: Recompute stresses
    stresses_corrected = hooke_stress(strains_corrected, stiffness,
                                       orient=orientations, frame="lab")

    # Residual after correction
    sig_corr_voigt = tensor_to_voigt(stresses_corrected)
    sig_avg_after = np.sum(w[:, None] * sig_corr_voigt[mask], axis=0)
    residual_after = sig_avg_after - sig_app_voigt

    # Uncertainty
    info = equilibrium_correction_uncertainty(
        stresses_corrected[mask], volumes[mask],
        confidences[mask] if confidences is not None else None,
    )
    info['n_grains_used'] = int(mask.sum())
    info['n_grains_total'] = N

    return {
        'eps_iso': eps_iso,
        'strains_corrected': strains_corrected,
        'stresses_corrected': stresses_corrected,
        'stresses_raw': stresses_raw,
        'residual_norm_before': float(np.linalg.norm(b)),
        'residual_norm_after': float(np.linalg.norm(residual_after)),
        'uncertainty': info,
    }


def correct_d0(
    strains: np.ndarray,
    stiffness: np.ndarray,
    orientations: np.ndarray,
    volumes: np.ndarray,
    confidences: Optional[np.ndarray] = None,
    applied_stress: Optional[np.ndarray] = None,
    min_confidence: float = 0.0,
) -> dict:
    """Two-step d0 correction: strain-level then stress-level.

    Step 1 fits the scalar isotropic strain error
    $\\varepsilon_{\\mathrm{iso}}$ from equilibrium and subtracts it
    at the strain level (before Hooke's law).  This correctly removes
    the dominant d0 artifact for all crystal symmetries.

    Step 2 applies a uniform stress-level shift to enforce exact
    macroscopic equilibrium on the corrected stresses.  This removes
    any residual imbalance from anisotropic d0 errors (where the
    error in different lattice parameters is not the same fraction)
    or other non-d0 systematic effects.

    The two-step approach is never worse than either step alone:
    for isotropic d0 errors it equals the strain-level result;
    for anisotropic errors it equals the stress-level result.

    Parameters
    ----------
    strains : ndarray (N, 3, 3)
        Per-grain strain tensors in the lab frame.
    stiffness : ndarray (6, 6)
        Single-crystal stiffness in Voigt-Mandel notation.
    orientations : ndarray (N, 3, 3)
        Orientation matrices (crystal -> lab).
    volumes : ndarray (N,)
        Grain volumes.
    confidences : ndarray (N,), optional
    applied_stress : ndarray (3, 3), optional
    min_confidence : float

    Returns
    -------
    dict with all keys from ``d0_correction_strain_level`` plus:
        'stresses_2step': ndarray (N, 3, 3) — final corrected stresses
        'residual_norm_2step': float — residual after both steps
    """
    # Step 1: strain-level
    result = d0_correction_strain_level(
        strains, stiffness, orientations, volumes,
        confidences=confidences,
        applied_stress=applied_stress,
        min_confidence=min_confidence,
    )

    # Step 2: stress-level on the residual
    if applied_stress is None:
        applied_stress = np.zeros((3, 3))

    stresses_2step = volume_average_stress_constraint(
        result['stresses_corrected'], volumes, applied_stress)

    # Final residual
    N_grains = strains.shape[0]
    w = effective_weights(
        volumes, confidences)
    sig_voigt_2step = tensor_to_voigt(stresses_2step)
    sig_avg_2step = np.sum(w[:, None] * sig_voigt_2step, axis=0)
    sig_app_voigt = tensor_to_voigt(applied_stress)
    residual_2step = sig_avg_2step - sig_app_voigt

    result['stresses_2step'] = stresses_2step
    result['residual_norm_2step'] = float(np.linalg.norm(residual_2step))

    return result


def recover_d0_cubic_free_standing(
    lattice_params: np.ndarray,
    assumed_reference: np.ndarray,
    volumes: Optional[np.ndarray] = None,
    confidences: Optional[np.ndarray] = None,
    min_confidence: float = 0.0,
) -> dict:
    """Stiffness-free d0 recovery for cubic, free-standing polycrystals.

    For a cubic material under zero applied macroscopic stress, the
    equilibrium correction simplifies exactly to the volume-averaged
    hydrostatic strain:

    .. math::

       \\varepsilon_{\\mathrm{iso}}
       = \\tfrac{1}{3}\\,\\mathrm{tr}\\langle
         \\boldsymbol{\\varepsilon}\\rangle_V.

    The proof is short: the stiffness response vector ``C{I} = 3K{I}``
    for any cubic crystal is purely hydrostatic and rotation-invariant,
    so ``<C_lab>{I} = 3K{I}`` regardless of texture. Substituting into
    Eq.~5 of the paper, the bulk modulus ``K`` cancels from
    numerator and denominator, and the fit collapses to the
    hydrostatic mean of the per-grain strain tensors. Neither ``K``
    nor the per-grain orientations enter.

    Use this helper when the single-crystal elastic constants are
    unknown or poorly characterised (battery electrolytes like LLZO,
    obscure cubic intermetallics, newly synthesised phases). The
    full :func:`recover_d0` returns the same number for the same
    data — this function just removes the ``stiffness`` and
    ``orientations`` arguments from the API so the user does not
    have to invent placeholder values.

    Parameters
    ----------
    lattice_params : ndarray (N, 6)
        Per-grain ``[a, b, c, alpha, beta, gamma]``. Must be cubic
        per grain (a=b=c, angles=90°) within reasonable tolerance.
    assumed_reference : ndarray (6,)
        Assumed strain-free lattice parameters (possibly wrong).
        Must describe a cubic cell.
    volumes : ndarray (N,), optional
        Per-grain volumes. Default: uniform weights.
    confidences : ndarray (N,), optional
    min_confidence : float

    Returns
    -------
    dict with keys:
        'reference_recovered' : ndarray (6,) — corrected lattice params
        'reference_assumed'   : ndarray (6,) — input assumed
        'eps_iso'             : float — fitted hydrostatic strain error
        'scale_factor'        : float — a0_true = a0_assumed / (1 - eps_iso)
        'n_grains_used'       : int
        'n_grains_total'      : int

    See Also
    --------
    recover_d0 : Full version, works for all symmetries and any applied
        stress. Requires stiffness and per-grain orientations.
    """
    from .tensor import lattice_params_to_strain

    assumed_reference = np.asarray(assumed_reference, dtype=np.float64)
    lattice_params = np.asarray(lattice_params, dtype=np.float64)
    N = lattice_params.shape[0]

    # Sanity-check cubic symmetry of the assumed reference
    a_ref, b_ref, c_ref = assumed_reference[:3]
    if not (np.isclose(a_ref, b_ref, rtol=1e-6)
            and np.isclose(a_ref, c_ref, rtol=1e-6)
            and np.allclose(assumed_reference[3:], 90.0, atol=1e-3)):
        raise ValueError(
            "assumed_reference must be cubic (a=b=c, angles=90°); "
            f"got {assumed_reference}. For non-cubic materials use "
            "recover_d0() with a stiffness tensor."
        )

    if volumes is None:
        volumes = np.ones(N, dtype=np.float64)
    else:
        volumes = np.asarray(volumes, dtype=np.float64)

    if confidences is not None and min_confidence > 0:
        mask = confidences >= min_confidence
    else:
        mask = np.ones(N, dtype=bool)

    w = effective_weights(
        volumes[mask],
        confidences[mask] if confidences is not None else None,
    )

    strains = lattice_params_to_strain(lattice_params, assumed_reference)
    # Per-grain hydrostatic strain = trace / 3
    hydro_per_grain = np.trace(strains, axis1=-2, axis2=-1) / 3.0  # (N,)
    eps_iso = float(np.sum(w * hydro_per_grain[mask]))

    scale = 1.0 / (1.0 - eps_iso)
    ref_recovered = assumed_reference.copy()
    ref_recovered[:3] = assumed_reference[:3] * scale

    return {
        'reference_recovered': ref_recovered,
        'reference_assumed':   assumed_reference.copy(),
        'eps_iso':             eps_iso,
        'scale_factor':        float(scale),
        'n_grains_used':       int(mask.sum()),
        'n_grains_total':      N,
    }


def recover_d0(
    lattice_params: np.ndarray,
    assumed_reference: np.ndarray,
    stiffness: np.ndarray,
    orientations: np.ndarray,
    volumes: np.ndarray,
    confidences: Optional[np.ndarray] = None,
    applied_stress: Optional[np.ndarray] = None,
    min_confidence: float = 0.0,
) -> dict:
    """Recover the strain-free lattice parameters from equilibrium.

    Given per-grain fitted lattice parameters and an assumed (possibly
    wrong) reference, this function determines the true strain-free
    lattice parameters by finding the isotropic strain error that
    satisfies macroscopic equilibrium.

    Works for all crystal symmetries.  The d0 error is assumed to
    scale all lattice lengths (a, b, c) by the same factor while
    leaving angles unchanged.

    Parameters
    ----------
    lattice_params : ndarray (N, 6)
        Per-grain fitted lattice parameters [a, b, c, alpha, beta, gamma].
        Lengths in Angstroms, angles in degrees.
    assumed_reference : ndarray (6,)
        The assumed strain-free lattice parameters used for strain
        computation (the possibly wrong d0).
    stiffness : ndarray (6, 6)
        Single-crystal stiffness in Voigt-Mandel notation (crystal frame).
    orientations : ndarray (N, 3, 3)
        Orientation matrices (crystal -> lab).
    volumes : ndarray (N,)
        Grain volumes.
    confidences : ndarray (N,), optional
        Per-grain confidence (0 to 1).
    applied_stress : ndarray (3, 3), optional
        Applied macroscopic stress. Default: zero (free-standing).
    min_confidence : float
        Minimum confidence for contributing to the average.

    Returns
    -------
    dict with keys:
        'reference_recovered': ndarray (6,) — corrected strain-free
            lattice parameters [a, b, c, alpha, beta, gamma]
        'reference_assumed': ndarray (6,) — the input assumed reference
        'eps_iso': float — the fitted isotropic strain error
        'scale_factor': float — multiplicative correction:
            a0_true = a0_assumed / (1 + eps_iso)
        'strains_corrected': ndarray (N, 3, 3)
        'stresses_corrected': ndarray (N, 3, 3)
        'residual_norm_before': float
        'residual_norm_after': float
        'uncertainty': dict
    """
    from .tensor import lattice_params_to_strain

    # Accept list/tuple input for convenience
    assumed_reference = np.asarray(assumed_reference, dtype=np.float64)
    lattice_params = np.asarray(lattice_params, dtype=np.float64)

    # Compute strains with the assumed (wrong) reference
    strains = lattice_params_to_strain(lattice_params, assumed_reference)

    # Run the strain-level d0 correction
    result = d0_correction_strain_level(
        strains, stiffness, orientations, volumes,
        confidences=confidences,
        applied_stress=applied_stress,
        min_confidence=min_confidence,
    )

    eps_iso = result['eps_iso']

    # Recover the true reference lattice parameters.
    # The d0 error scales all lengths by the same factor:
    #   a0_assumed = a0_true * (1 + delta)
    # where delta ≈ -eps_iso (the strain error is the negative of
    # the reference error).
    # Therefore: a0_true = a0_assumed / (1 - eps_iso)
    #
    # This is exact to first order in strain and accurate to O(eps^2).
    scale = 1.0 / (1.0 - eps_iso)

    ref_recovered = assumed_reference.copy()
    ref_recovered[:3] = assumed_reference[:3] * scale  # scale a, b, c
    # angles are unchanged by an isotropic d0 error

    result['reference_recovered'] = ref_recovered
    result['reference_assumed'] = assumed_reference.copy()
    result['scale_factor'] = float(scale)

    return result


# -------------------------------------------------------------------
#  Confidence-weighted variants with uncertainty
# -------------------------------------------------------------------

def effective_weights(
    volumes: np.ndarray,
    confidences: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute effective weights from volumes and optional confidences.

    Parameters
    ----------
    volumes : ndarray (N,)
        Grain volumes.
    confidences : ndarray (N,), optional
        Per-grain confidence / completeness (0 to 1).
        If provided, weights are volume * confidence.

    Returns
    -------
    ndarray (N,) normalized weights summing to 1.
    """
    if confidences is not None:
        w = volumes * confidences
    else:
        w = volumes.copy()
    return w / w.sum()


def equilibrium_correction_uncertainty(
    stresses: np.ndarray,
    volumes: np.ndarray,
    confidences: Optional[np.ndarray] = None,
) -> dict:
    """Estimate uncertainty of the equilibrium correction.

    When only a subset of grains is indexed (incomplete sampling),
    the volume-average stress is a sample estimate of the true
    population mean.  The standard error of this estimate quantifies
    the uncertainty of the FF-1/FF-2 corrections.

    Parameters
    ----------
    stresses : ndarray (N, 3, 3)
        Per-grain stress tensors.
    volumes : ndarray (N,)
        Grain volumes.
    confidences : ndarray (N,), optional
        Per-grain confidence (0 to 1).

    Returns
    -------
    dict with keys:
        'n_grains': int — number of grains used
        'hydrostatic_mean_MPa': float — weighted mean hydrostatic stress
        'hydrostatic_std_MPa': float — weighted std of hydrostatic stress
        'hydrostatic_se_MPa': float — standard error of the mean
            (uncertainty of the FF-2 correction)
        'stress_mean_voigt_MPa': ndarray (6,) — weighted mean stress (Voigt)
        'stress_se_voigt_MPa': ndarray (6,) — standard error per component
            (uncertainty of the FF-1 correction)
        'effective_n': float — effective sample size accounting for
            weight concentration (Kish's formula)
    """
    N = stresses.shape[0]
    w = effective_weights(volumes, confidences)

    # Hydrostatic component
    hydro = np.trace(stresses, axis1=-2, axis2=-1) / 3.0  # (N,)
    hydro_mean = np.sum(w * hydro)
    hydro_var = np.sum(w * (hydro - hydro_mean)**2)
    hydro_std = np.sqrt(hydro_var)

    # Effective sample size (Kish, 1965): accounts for unequal weights
    n_eff = 1.0 / np.sum(w**2)

    # Standard error of the weighted mean
    hydro_se = hydro_std / np.sqrt(n_eff) if n_eff > 1 else hydro_std

    # Full stress tensor (Voigt)
    sig_voigt = tensor_to_voigt(stresses)  # (N, 6)
    sig_mean = np.sum(w[:, None] * sig_voigt, axis=0)  # (6,)
    sig_var = np.sum(w[:, None] * (sig_voigt - sig_mean)**2, axis=0)  # (6,)
    sig_se = np.sqrt(sig_var) / np.sqrt(n_eff) if n_eff > 1 else np.sqrt(sig_var)

    return {
        'n_grains': N,
        'hydrostatic_mean_MPa': float(hydro_mean),
        'hydrostatic_std_MPa': float(hydro_std),
        'hydrostatic_se_MPa': float(hydro_se),
        'stress_mean_voigt_MPa': sig_mean,
        'stress_se_voigt_MPa': sig_se,
        'effective_n': float(n_eff),
    }


def hydrostatic_deviatoric_decomposition_weighted(
    stresses: np.ndarray,
    volumes: np.ndarray,
    confidences: Optional[np.ndarray] = None,
    applied_stress: Optional[np.ndarray] = None,
    min_confidence: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """FF-2 with confidence weighting and uncertainty estimation.

    Like ``hydrostatic_deviatoric_decomposition`` but:
    - Weights the average by volume * confidence
    - Optionally filters out low-confidence grains
    - Returns uncertainty estimates for the correction

    Parameters
    ----------
    stresses : ndarray (N, 3, 3)
    volumes : ndarray (N,)
    confidences : ndarray (N,), optional
        Per-grain confidence (0 to 1). Default: uniform.
    applied_stress : ndarray (3, 3), optional
        Default: zero.
    min_confidence : float
        Grains below this threshold are excluded from the average
        but still receive the correction.

    Returns
    -------
    hydrostatic : ndarray (N,)
    deviatoric : ndarray (N, 3, 3)
    corrected : ndarray (N, 3, 3)
    info : dict
        Uncertainty information (see ``equilibrium_correction_uncertainty``).
    """
    if applied_stress is None:
        applied_stress = np.zeros((3, 3))

    N = stresses.shape[0]
    I = np.eye(3)

    # Build mask for grains used in the average
    if confidences is not None and min_confidence > 0:
        mask = confidences >= min_confidence
    else:
        mask = np.ones(N, dtype=bool)

    w = effective_weights(
        volumes[mask],
        confidences[mask] if confidences is not None else None,
    )

    # Decompose ALL grains
    hydro_raw = np.trace(stresses, axis1=-2, axis2=-1) / 3.0
    deviatoric = stresses - hydro_raw[:, None, None] * I[None, :, :]

    # Compute correction from masked subset
    target_hydro = np.trace(applied_stress) / 3.0
    current_avg_hydro = np.sum(w * hydro_raw[mask])
    hydro_shift = target_hydro - current_avg_hydro

    # Apply correction to ALL grains (including low-confidence ones)
    hydro_corrected = hydro_raw + hydro_shift

    dev_applied = applied_stress - (np.trace(applied_stress) / 3.0) * I
    dev_avg = np.sum(w[:, None, None] * deviatoric[mask], axis=0)
    dev_correction = dev_applied - dev_avg
    deviatoric_corrected = deviatoric + dev_correction[None, :, :]

    corrected = hydro_corrected[:, None, None] * I[None, :, :] + deviatoric_corrected

    # Uncertainty from the masked subset
    info = equilibrium_correction_uncertainty(
        stresses[mask], volumes[mask],
        confidences[mask] if confidences is not None else None,
    )
    info['n_grains_used'] = int(mask.sum())
    info['n_grains_total'] = N

    return hydro_corrected, deviatoric_corrected, corrected, info


# -------------------------------------------------------------------
#  Anisotropic (symmetry-aware) d0 recovery
# -------------------------------------------------------------------

#: Independent reference-lattice length degrees of freedom per crystal
#: system, as (name, basis tensor diagonal, which of a/b/c it scales).
#: The reference cell of a non-cubic phase has more than one free
#: length, so a single isotropic scale cannot represent its error.
_D0_BASES = {
    "cubic":        [("a", (1.0, 1.0, 1.0), (0, 1, 2))],
    "tetragonal":   [("a", (1.0, 1.0, 0.0), (0, 1)), ("c", (0.0, 0.0, 1.0), (2,))],
    "hexagonal":    [("a", (1.0, 1.0, 0.0), (0, 1)), ("c", (0.0, 0.0, 1.0), (2,))],
    "trigonal":     [("a", (1.0, 1.0, 0.0), (0, 1)), ("c", (0.0, 0.0, 1.0), (2,))],
    "rhombohedral": [("a", (1.0, 1.0, 0.0), (0, 1)), ("c", (0.0, 0.0, 1.0), (2,))],
    "orthorhombic": [("a", (1.0, 0.0, 0.0), (0,)), ("b", (0.0, 1.0, 0.0), (1,)),
                     ("c", (0.0, 0.0, 1.0), (2,))],
    "monoclinic":   [("a", (1.0, 0.0, 0.0), (0,)), ("b", (0.0, 1.0, 0.0), (1,)),
                     ("c", (0.0, 0.0, 1.0), (2,))],
    "triclinic":    [("a", (1.0, 0.0, 0.0), (0,)), ("b", (0.0, 1.0, 0.0), (1,)),
                     ("c", (0.0, 0.0, 1.0), (2,))],
}


def recover_d0_anisotropic(
    lattice_params: np.ndarray,
    assumed_reference: np.ndarray,
    stiffness: np.ndarray,
    orientations: np.ndarray,
    volumes: Optional[np.ndarray] = None,
    crystal_system: str = "hexagonal",
    confidences: Optional[np.ndarray] = None,
    applied_stress: Optional[np.ndarray] = None,
    min_confidence: float = 0.0,
    cond_warn: float = 1e3,
) -> dict:
    """Recover an ANISOTROPIC strain-free reference cell from equilibrium.

    :func:`recover_d0` assumes the reference error scales ``a``, ``b`` and
    ``c`` by one common factor.  That is exact only for cubic phases.  A
    hexagonal/trigonal cell has **two** independent reference lengths
    (``a`` and ``c``), an orthorhombic cell three; a wrong reference in
    those systems is generally *not* an isotropic dilatation — ``a`` can
    be too small while ``c`` is too large — and no single scale can
    absorb it.

    This routine solves the same macroscopic-equilibrium condition, but
    for one unknown per symmetry-allowed reference length.

    Algorithm
    ---------
    A perturbation of reference length *k* adds, in the **grain** frame,
    a fixed strain ``delta_k * B_k`` to every grain, where ``B_k`` is a
    diagonal indicator tensor (e.g. ``diag(1,1,0)`` for hexagonal ``a``).
    Because the Mandel rotation ``M`` is orthogonal, the resulting
    lab-frame stress of grain *g* is

    .. math::

        \\{\\Delta\\sigma_g\\} = M_g^{T} C \\{B_k\\},

    so each column of the design matrix is one crystal-frame stiffness
    response, rotated and volume-averaged:

    .. math::

        A_{:,k} = \\Big\\langle M_g^{T} \\Big\\rangle_V \\, C \\{B_k\\}.

    The reference error then solves the 6-equation least-squares problem
    ``A delta = <sigma> - sigma_applied``.

    Identifiability
    ---------------
    Counter-intuitively, a **weak** texture is the bad case, not a sharp
    one.  Averaging ``M_g^T`` over a uniform orientation distribution
    projects onto the isotropic subspace, so every column tends to the
    same direction ``C\\{I\\}`` and the split between ``a`` and ``c``
    washes out; the condition number grows with the number of grains as
    the average converges.  A sharply textured or single-orientation
    population is *well* conditioned, because one crystal's anisotropic
    stiffness gives an ``a`` error and a ``c`` error visibly different
    stress signatures.

    Measured on a synthetic hexagonal aggregate (C11/C12/C13/C33/C44 =
    242/76/48/196/46 GPa): single orientation and a 10 deg fibre both
    give ``cond`` 2.8, while uniform random texture gives 23 at N=100
    and 142 at N=1000.  Even then the recovery stays usable — with
    uniform texture, N=1000 and 500 ue of per-grain scatter the
    recovered lengths land within ~200-270 ue.

    ``condition_number`` is the diagnostic to read; a large value means
    the ``a``/``c`` split is weakly determined however tight the
    residual looks.  Check it before trusting the answer.

    Parameters
    ----------
    lattice_params : ndarray (N, 6)
        Per-grain/voxel ``[a, b, c, alpha, beta, gamma]``, Angstrom/degrees.
    assumed_reference : ndarray (6,)
        The assumed (possibly wrong) strain-free cell.
    stiffness : ndarray (6, 6)
        Single-crystal stiffness, Voigt-Mandel, crystal frame.
    orientations : ndarray (N, 3, 3)
        Orientation matrices, crystal -> lab.
    volumes : ndarray (N,), optional
        Per-grain volumes. Default uniform.
    crystal_system : str
        One of ``cubic``, ``tetragonal``, ``hexagonal``, ``trigonal``,
        ``rhombohedral``, ``orthorhombic``, ``monoclinic``, ``triclinic``.
        Only the reference *lengths* are recovered; reference angles are
        never fitted.
    confidences, min_confidence, applied_stress
        As :func:`recover_d0`.  ``applied_stress`` defaults to zero,
        i.e. a free-standing (unloaded) sample.
    cond_warn : float
        Condition number above which ``well_conditioned`` is set False.

    Returns
    -------
    dict with keys:
        ``reference_recovered`` (6,), ``reference_assumed`` (6,),
        ``deltas`` {name: float}, ``eps_iso_equivalent`` float,
        ``strains_corrected`` (N,3,3) grain frame,
        ``residual_norm_before`` / ``residual_norm_after``,
        ``condition_number``, ``singular_values``, ``well_conditioned``,
        ``n_grains_used``, ``n_grains_total``, ``crystal_system``.

    See Also
    --------
    recover_d0 : isotropic single-scale version (exact for cubic).
    """
    from .tensor import lattice_params_to_strain
    from .hooke import hooke_stress

    key = str(crystal_system).strip().lower()
    if key not in _D0_BASES:
        raise ValueError(
            f"unknown crystal_system {crystal_system!r}; "
            f"expected one of {sorted(_D0_BASES)}")
    bases = _D0_BASES[key]

    lattice_params = np.asarray(lattice_params, dtype=np.float64)
    assumed_reference = np.asarray(assumed_reference, dtype=np.float64)
    orientations = np.asarray(orientations, dtype=np.float64)
    stiffness = np.asarray(stiffness, dtype=np.float64)
    N = lattice_params.shape[0]
    if volumes is None:
        volumes = np.ones(N, dtype=np.float64)
    volumes = np.asarray(volumes, dtype=np.float64)
    if applied_stress is None:
        applied_stress = np.zeros((3, 3))

    if confidences is not None and min_confidence > 0:
        mask = np.asarray(confidences) >= min_confidence
    else:
        mask = np.ones(N, dtype=bool)
    if mask.sum() < len(bases):
        raise ValueError(
            f"only {int(mask.sum())} grains pass min_confidence but "
            f"{len(bases)} reference parameters must be determined")

    w = effective_weights(
        volumes[mask],
        np.asarray(confidences)[mask] if confidences is not None else None,
    )

    # strains with the assumed reference, grain frame -> lab frame
    eps_grain = lattice_params_to_strain(lattice_params, assumed_reference)
    eps_lab = orientations @ eps_grain @ np.swapaxes(orientations, -1, -2)

    stresses_raw = hooke_stress(eps_lab, stiffness, orient=orientations,
                                frame="lab")
    sig_avg = np.sum(w[:, None] * tensor_to_voigt(stresses_raw)[mask], axis=0)
    b = sig_avg - tensor_to_voigt(applied_stress)

    # design matrix: one column per reference length
    M_all = rotation_voigt_mandel(orientations)          # lab -> grain
    Mt_avg = np.sum(w[:, None, None] * np.swapaxes(M_all, -1, -2)[mask], axis=0)
    A = np.zeros((6, len(bases)))
    for k, (_name, diag, _idx) in enumerate(bases):
        B_k = np.diag(np.asarray(diag, dtype=np.float64))
        A[:, k] = Mt_avg @ (stiffness @ tensor_to_voigt(B_k))

    delta, *_ = np.linalg.lstsq(A, b, rcond=None)
    sv = np.linalg.svd(A, compute_uv=False)
    cond = float(sv[0] / sv[-1]) if sv[-1] > 0 else np.inf

    # corrected strains (grain frame) and the recovered reference
    corr = np.zeros((3, 3))
    reference_recovered = assumed_reference.copy()
    deltas = {}
    for k, (name, diag, idx) in enumerate(bases):
        d = float(delta[k])
        deltas[name] = d
        corr += d * np.diag(np.asarray(diag, dtype=np.float64))
        for i in idx:
            reference_recovered[i] = assumed_reference[i] / (1.0 - d)

    eps_corr_grain = eps_grain - corr[None, :, :]
    eps_corr_lab = (orientations @ eps_corr_grain
                    @ np.swapaxes(orientations, -1, -2))
    stresses_corr = hooke_stress(eps_corr_lab, stiffness,
                                 orient=orientations, frame="lab")
    sig_after = np.sum(
        w[:, None] * tensor_to_voigt(stresses_corr)[mask], axis=0)
    residual_after = sig_after - tensor_to_voigt(applied_stress)

    return {
        "reference_recovered": reference_recovered,
        "reference_assumed": assumed_reference,
        "deltas": deltas,
        "eps_iso_equivalent": float(np.mean(list(deltas.values()))),
        "strains_corrected": eps_corr_grain,
        "stresses_corrected": stresses_corr,
        "stresses_raw": stresses_raw,
        "residual_norm_before": float(np.linalg.norm(b)),
        "residual_norm_after": float(np.linalg.norm(residual_after)),
        "condition_number": cond,
        "singular_values": sv,
        "well_conditioned": bool(cond < cond_warn),
        "n_grains_used": int(mask.sum()),
        "n_grains_total": int(N),
        "crystal_system": key,
    }
