"""Joint spec construction: powder geometry + HEDM grain nuisance blocks.

The output is a single :class:`midas_peakfit.ParameterSpec` whose dict contains
all parameters from both modalities under canonical names. Pack/unpack/LM
operate on the merged spec uniformly.

Canonical names (paper-3 §3.2 + paper-4 §3.3 unification):

    Geometry      — Lsd, BC_y, BC_z, tx, ty, tz, Wedge, Wavelength
    Distortion    — iso_R2, iso_R4, iso_R6, a1, phi1, ..., a6, phi6
    Pixel scale   — pxY, pxZ, RhoD, Parallax
    Per-panel     — panel_delta_yz [N, 2], panel_delta_theta [N],
                    panel_delta_lsd [N], panel_delta_p2 [N]
    HEDM grains   — grain_euler [N_g, 3], grain_pos [N_g, 3],
                    grain_lattice [N_g, 6]

Multi-detector variants (per-detector replicas, used when the user wants
each detector to have its own Lsd / BC):

    Lsd_per_det [D], BC_y_per_det [D], BC_z_per_det [D], tilts_per_det [D, 3]
"""
from __future__ import annotations

from typing import Optional, Sequence

import torch

from midas_peakfit import Parameter, ParameterSpec
from midas_calibrate_v2.parameters.spec import CalibrationSpec


def build_joint_spec(
    *,
    powder_spec: CalibrationSpec,
    grain_eulers_init: torch.Tensor,        # (N_g, 3) radians
    grain_positions_init: torch.Tensor,     # (N_g, 3) microns
    grain_lattices_init: torch.Tensor,      # (N_g, 6) Voigt: a, b, c, α, β, γ
    refine_grain_orientation: bool = True,
    refine_grain_position: bool = True,
    refine_grain_strain: bool = False,
) -> CalibrationSpec:
    """Extend a powder ``CalibrationSpec`` with HEDM grain nuisance blocks.

    The powder spec retains its existing parameters (Lsd, BC_y, BC_z,
    distortion, panel shifts, ...).  We append three new vector parameters:
    grain Eulers, grain positions, and grain lattice constants.  Default
    refinement flags follow the alternating-driver convention: orientations
    + positions on, strains off (refined in a separate pass).

    The returned object is the same ``CalibrationSpec`` (now a
    ``ParameterSpec`` subclass), so it slots into ``lm_minimise`` and
    ``laplace_at_map`` directly.

    Parameters
    ----------
    powder_spec
        Output of e.g. :func:`midas_calibrate_v2.compat.from_v1.spec_from_v1_file`,
        possibly with :func:`add_panel_parameters` already applied.
    grain_eulers_init, grain_positions_init, grain_lattices_init
        Initial values from a prior MIDAS grain-fit (e.g. ``Grains.csv``).
        Shapes must match.
    """
    if grain_eulers_init.dim() != 2 or grain_eulers_init.shape[1] != 3:
        raise ValueError(
            f"grain_eulers_init must be (N_g, 3); got {tuple(grain_eulers_init.shape)}")
    if grain_positions_init.shape != grain_eulers_init.shape:
        raise ValueError("grain_positions_init must have the same shape as grain_eulers_init")
    if grain_lattices_init.dim() != 2 or grain_lattices_init.shape[1] != 6:
        raise ValueError(
            f"grain_lattices_init must be (N_g, 6); got {tuple(grain_lattices_init.shape)}")

    n_g = grain_eulers_init.shape[0]
    if grain_lattices_init.shape[0] != n_g:
        raise ValueError(
            f"grain_lattices_init has {grain_lattices_init.shape[0]} rows but "
            f"grain_eulers_init has {n_g}")

    powder_spec.add(Parameter(
        "grain_euler",
        init=grain_eulers_init.to(torch.float64),
        refined=refine_grain_orientation,
    ))
    powder_spec.add(Parameter(
        "grain_pos",
        init=grain_positions_init.to(torch.float64),
        refined=refine_grain_position,
    ))
    powder_spec.add(Parameter(
        "grain_lattice",
        init=grain_lattices_init.to(torch.float64),
        refined=refine_grain_strain,
    ))
    return powder_spec


__all__ = ["build_joint_spec"]
