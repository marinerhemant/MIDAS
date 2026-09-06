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
                    grain_lattice [N_g, 6] (seed, frozen),
                    grain_strain [N_g, 6] (dimensionless, refinable)

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
    strain_bound: float = 0.02,
) -> CalibrationSpec:
    """Extend a powder ``CalibrationSpec`` with HEDM grain nuisance blocks.

    The powder spec retains its existing parameters (Lsd, BC_y, BC_z,
    distortion, panel shifts, ...).  We append four new vector parameters:
    grain Eulers, grain positions, grain lattice constants, and per-grain
    strain.  Default refinement flags follow the alternating-driver
    convention: orientations + positions on, strains off (refined in a
    separate pass).

    **Strain is refined through ``grain_strain``, never by thawing
    ``grain_lattice``.** ``grain_lattice`` is the per-grain SEED (a, b, c in
    Å and α, β, γ in degrees) and is always frozen; ``grain_strain`` is a
    dimensionless crystal-frame symmetric strain, init 0, that the forward
    model applies as ``B = (I + eps)^-1 B0`` on top of that seed.

    The reason is not stylistic. ``lm_minimise`` boxes every refined
    parameter with ONE ``(lo, hi)`` pair for the whole tensor and, when a
    parameter declares no bounds, fabricates that box as
    ``init.flatten()[0] ± fallback_span`` — the FIRST element only. For a
    (N, 6) lattice that box is built around ``a`` (e.g. 3.585 ± 2), and the
    logit transform then clamps every angle silently:

        in  [3.585, 3.585, 3.585, 90.0, 90.0, 90.0]
        out [3.585, 3.585, 3.585,  5.585,  5.585,  5.585]

    i.e. thawing ``grain_lattice`` rewrites α, β, γ from 90° to the top of
    the a-box before the first residual evaluation, and returns a degenerate
    cell with no error raised. Lengths and angles cannot share one box; a
    dimensionless strain needs only one, centred on zero.

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
        Shapes must match. ``grain_lattices_init`` should be each grain's OWN
        fitted lattice where the file carries one — see
        :func:`midas_joint_ff_calibrate.grain_observations.grain_lattices_for_fit`.
    refine_grain_strain
        Thaw ``grain_strain``. ``grain_lattice`` stays frozen either way.
    strain_bound
        Half-width of the ``grain_strain`` box, dimensionless. The default
        0.02 (= ±20 000 µε) is ~20× the largest per-grain RMS strain seen on
        real FF data and keeps ``(I + eps)`` far from singular.
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
    # The per-grain SEED lattice. Always frozen (see the docstring): it is
    # pinnable with ``--fix grain_lattice=...`` but never refined directly.
    powder_spec.add(Parameter(
        "grain_lattice",
        init=grain_lattices_init.to(torch.float64),
        refined=False,
    ))
    # Crystal-frame symmetric strain, PLAIN-Voigt [e11, e12, e13, e22, e23,
    # e33] to match HEDMForwardModel.strain_as_voigt. Dimensionless, init 0,
    # so one box bounds all six components.
    if not (strain_bound > 0.0):
        raise ValueError(f"strain_bound must be > 0; got {strain_bound}")
    powder_spec.add(Parameter(
        "grain_strain",
        init=torch.zeros((n_g, 6), dtype=torch.float64),
        refined=refine_grain_strain,
        bounds=(-float(strain_bound), float(strain_bound)),
    ))
    return powder_spec


__all__ = ["build_joint_spec"]
