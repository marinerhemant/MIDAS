"""High-level stress analysis pipeline for use with any HEDM code.

Provides ``compute_stress()`` — a single function that takes numpy arrays
of orientations, strains, and volumes (from any source: MIDAS, hexrd,
DAXM, ImageD11, custom scripts) and returns the complete stress analysis
with equilibrium corrections and uncertainty.
"""

from typing import Optional

import numpy as np

from .hooke import hooke_stress
from .tensor import hydrostatic, deviatoric, von_mises
from .equilibrium import (
    hydrostatic_deviatoric_decomposition_weighted,
    equilibrium_correction_uncertainty,
)


def compute_stress(
    strain: np.ndarray,
    stiffness: np.ndarray,
    orient: np.ndarray,
    volumes: np.ndarray,
    confidences: Optional[np.ndarray] = None,
    applied_stress: Optional[np.ndarray] = None,
    min_confidence: float = 0.0,
    frame: str = "lab",
) -> dict:
    """Complete stress analysis pipeline from arrays.

    Computes per-grain stress via Hooke's law, applies equilibrium
    constraints, and returns the full decomposition with uncertainty.

    Works with output from any HEDM code — just provide numpy arrays.

    Parameters
    ----------
    strain : ndarray (N, 3, 3)
        Per-grain strain tensors in the lab (or sample) frame.
    stiffness : ndarray (6, 6)
        Single-crystal stiffness matrix in Voigt-Mandel notation (GPa).
        Use ``midas_stress.get_stiffness("Cu")`` or build with
        ``cubic_stiffness()`` / ``hexagonal_stiffness()``.
    orient : ndarray (N, 3, 3)
        Per-grain orientation matrices (crystal -> lab/sample frame).
    volumes : ndarray (N,)
        Per-grain volumes (relative sizes suffice; e.g., grain radii
        cubed, or voxel counts from NF-HEDM).
    confidences : ndarray (N,), optional
        Per-grain confidence / completeness (0 to 1).
        If provided, the equilibrium correction is weighted by
        volume * confidence.
    applied_stress : ndarray (3, 3), optional
        Applied macroscopic stress (GPa). Default: zero (free-standing).
    min_confidence : float
        Minimum confidence for a grain to contribute to the equilibrium
        average. Grains below this threshold still receive the correction.
    frame : str
        ``"lab"`` (default) or ``"grain"``.

    Returns
    -------
    dict with keys:
        'stress_raw': ndarray (N, 3, 3)
            Per-grain stress from Hooke's law (before equilibrium).
        'stress_corrected': ndarray (N, 3, 3)
            Per-grain stress after equilibrium correction.
        'hydrostatic_raw': ndarray (N,)
            Hydrostatic stress before correction (MPa-scale if C in GPa).
        'hydrostatic_corrected': ndarray (N,)
            Hydrostatic stress after FF-2 correction.
        'deviatoric': ndarray (N, 3, 3)
            Deviatoric stress tensors (equilibrium-corrected).
        'von_mises': ndarray (N,)
            Von Mises equivalent stress (from corrected stress).
        'hydrostatic_shift': float
            The uniform hydrostatic correction applied (d0 proxy).
        'uncertainty': dict
            Uncertainty information from the equilibrium correction.
            See ``equilibrium_correction_uncertainty()`` for keys.

    Example
    -------
    >>> import numpy as np
    >>> import midas_stress as ms
    >>> # From hexrd or any other code:
    >>> orientations = np.array([...])  # (N, 3, 3)
    >>> strains = np.array([...])       # (N, 3, 3)
    >>> volumes = np.array([...])       # (N,)
    >>> result = ms.compute_stress(
    ...     strain=strains,
    ...     stiffness=ms.get_stiffness("Cu"),
    ...     orient=orientations,
    ...     volumes=volumes,
    ... )
    >>> print(result['von_mises'].mean())
    >>> print(result['hydrostatic_shift'])
    >>> print(result['uncertainty']['hydrostatic_se_MPa'])
    """
    strain = np.asarray(strain, dtype=np.float64)
    orient = np.asarray(orient, dtype=np.float64)
    volumes = np.asarray(volumes, dtype=np.float64)

    # Step 1: Hooke's law
    stress_raw = hooke_stress(strain, stiffness, orient=orient, frame=frame)

    # Step 2: Equilibrium correction (FF-1 + FF-2)
    hydro_corr, dev_corr, stress_corr, info = \
        hydrostatic_deviatoric_decomposition_weighted(
            stress_raw, volumes,
            confidences=confidences,
            applied_stress=applied_stress,
            min_confidence=min_confidence,
        )

    # Step 3: Compute derived quantities
    hydro_raw = hydrostatic(stress_raw)
    vm = von_mises(stress_corr)
    hydro_shift = float(hydro_corr.mean() - hydro_raw.mean())

    return {
        'stress_raw': stress_raw,
        'stress_corrected': stress_corr,
        'hydrostatic_raw': hydro_raw,
        'hydrostatic_corrected': hydro_corr,
        'deviatoric': dev_corr,
        'von_mises': vm,
        'hydrostatic_shift': hydro_shift,
        'uncertainty': info,
    }
