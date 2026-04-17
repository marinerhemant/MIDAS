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
"""

from typing import Optional, Tuple

import numpy as np

from .tensor import tensor_to_voigt, voigt_to_tensor, rotation_voigt_mandel


def volume_average_stress_constraint(
    stresses: np.ndarray,
    volumes: np.ndarray,
    applied_stress: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Apply volume-average stress constraint (FF-1).

    Enforces: sum(V_g * sigma_g) / V_total = sigma_applied

    Parameters
    ----------
    stresses : ndarray (N, 3, 3) or (N, 6)
        Per-grain stress tensors.
    volumes : ndarray (N,)
        Grain volumes (relative sizes suffice).
    applied_stress : ndarray (3, 3) or (6,), optional
        Applied macroscopic stress. Default: zero (unloaded sample).

    Returns
    -------
    ndarray same shape as input, corrected stresses.
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
    w = volumes / V_total

    sig_avg = np.sum(w[:, None] * sig, axis=0)
    delta_sig = sig_app - sig_avg

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
