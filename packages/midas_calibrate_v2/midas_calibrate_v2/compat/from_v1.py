"""Build a v2 :class:`CalibrationSpec` from a v1 :class:`CalibrationParams`.

This lets v2 read v1 parameter files unchanged, then opt into v2 features by
mutating the resulting spec (e.g., refinement flags, priors, panel layout).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import torch

from midas_calibrate.params import CalibrationParams as V1Params

from ..parameters.parameter import Parameter
from ..parameters.spec import CalibrationSpec


def _add(spec: CalibrationSpec, name: str, init, refined: bool,
         tol: Optional[float] = None) -> None:
    bounds = None
    if tol is not None and tol > 0:
        if isinstance(init, (int, float)):
            bounds = (float(init) - tol, float(init) + tol)
    spec.add(Parameter(name=name, init=init, refined=refined, bounds=bounds))


def spec_from_v1_params(v1: V1Params) -> CalibrationSpec:
    """Convert a v1 CalibrationParams into a v2 CalibrationSpec.

    The mapping preserves v1's refinement flags, tolerances (as bounds), and
    fixed/refined status.  Per-panel parameters are NOT created here — call
    :func:`add_panel_parameters` after this if you have a multi-panel
    detector.
    """
    s = CalibrationSpec()
    s.NrPixelsY = v1.NrPixelsY
    s.NrPixelsZ = v1.NrPixelsZ
    s.SpaceGroup = v1.SpaceGroup
    s.LatticeConstant = v1.LatticeConstant
    s.MaxRingRad = v1.MaxRingRad
    s.MinRingRad = v1.MinRingRad
    s.fix_panel_id = v1.FixedPanelID

    refine = v1.Refine
    _add(s, "Lsd", v1.Lsd, refined=refine.get("Lsd", True), tol=v1.tolLsd)
    _add(s, "BC_y", v1.BC_y, refined=refine.get("BC", True), tol=v1.tolBC)
    _add(s, "BC_z", v1.BC_z, refined=refine.get("BC", True), tol=v1.tolBC)
    _add(s, "tx", v1.tx, refined=False, tol=1e-6)   # v1: tx never refined
    _add(s, "ty", v1.ty, refined=refine.get("ty", True), tol=v1.tolTilts)
    _add(s, "tz", v1.tz, refined=refine.get("tz", True), tol=v1.tolTilts)
    # Distortion: translate v1's 15 chaotic p₀..p₁₄ into v2's canonical
    # names (3 isotropic + 6 amp/phase pairs).  The v1→v2 mapping is:
    #
    #   v1 p₀  (amp,  ρ²,  fold=2)  →  v2 a2
    #   v1 p₁  (amp,  ρ⁴,  fold=4)  →  v2 a4
    #   v1 p₂  (amp,  ρ²,  fold=0)  →  v2 iso_R2
    #   v1 p₃  (phase for p₁)       →  v2 phi4
    #   v1 p₄  (amp,  ρ⁶,  fold=0)  →  v2 iso_R6
    #   v1 p₅  (amp,  ρ⁴,  fold=0)  →  v2 iso_R4
    #   v1 p₆  (phase for p₀)       →  v2 phi2
    #   v1 p₇  (amp,  ρ⁴,  fold=1)  →  v2 a1
    #   v1 p₈  (phase for p₇)       →  v2 phi1
    #   v1 p₉  (amp,  ρ³,  fold=3)  →  v2 a3
    #   v1 p₁₀ (phase for p₉)       →  v2 phi3
    #   v1 p₁₁ (amp,  ρ⁵,  fold=5)  →  v2 a5
    #   v1 p₁₂ (phase for p₁₁)      →  v2 phi5
    #   v1 p₁₃ (amp,  ρ⁶,  fold=6)  →  v2 a6
    #   v1 p₁₄ (phase for p₁₃)      →  v2 phi6
    V1_TO_V2_DISTORTION = {
        0: "a2",     1: "a4",     2: "iso_R2", 3: "phi4",
        4: "iso_R6", 5: "iso_R4", 6: "phi2",   7: "a1",
        8: "phi1",   9: "a3",     10: "phi3",  11: "a5",
        12: "phi5",  13: "a6",    14: "phi6",
    }
    PHASE_NAMES = {"phi1", "phi2", "phi3", "phi4", "phi5", "phi6"}
    for i in range(15):
        v2_name = V1_TO_V2_DISTORTION[i]
        tol_i = 90.0 if v2_name in PHASE_NAMES else v1.tolDistortion
        _add(s, v2_name, getattr(v1, f"p{i}"),
             refined=refine.get(f"p{i}", True), tol=tol_i)
    _add(s, "Parallax", v1.Parallax, refined=refine.get("Parallax", False),
         tol=v1.tolParallax)
    _add(s, "Wavelength", v1.Wavelength, refined=refine.get("Wavelength", False),
         tol=v1.tolWavelength)
    # v2 promotes pxY, pxZ to first-class refinable parameters (v1 had them
    # fixed).  Default to refined=False so v1 parity is preserved; the user
    # opts in via spec.thaw("pxY", "pxZ") for multi-image fitting.
    _add(s, "pxY", v1.pxY, refined=False, tol=0.5)   # μm
    pxZ = v1.pxZ if v1.pxZ > 0 else v1.pxY
    _add(s, "pxZ", pxZ, refined=False, tol=0.5)
    rho_d = v1.RhoD if v1.RhoD > 0 else v1.MaxRingRad
    _add(s, "RhoD", rho_d, refined=False, tol=10.0)
    # Hex-lattice parameters.  Always added so unpack_spec produces
    # consistent dict keys; only consulted when CalibrationSpec.lattice
    # != "cartesian".  Apothem default 0.0 is intentional — refusing to
    # initialise the hex path until the caller supplies a real apothem.
    _add(s, "Apothem", 0.0, refined=False, tol=0.5)
    _add(s, "LatticeOrientation", 0.0, refined=False, tol=0.5)
    return s


def _parse_ring_filters(path: Path) -> Tuple[list, int]:
    """Extract ``RingsToExclude`` and ``MaxRingNumber`` from a v1 paramstest.

    These keys exist in the v1 C parameter file format but aren't carried
    by ``midas_calibrate.params.CalibrationParams``.  We re-parse the raw
    text to recover them.
    """
    rings_to_exclude: list = []
    max_ring_number = 0
    try:
        for line in Path(path).read_text().splitlines():
            line = line.split("#")[0].strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            if parts[0] == "RingsToExclude":
                try:
                    rings_to_exclude.append(int(parts[1]))
                except ValueError:
                    pass
            elif parts[0] == "MaxRingNumber":
                try:
                    max_ring_number = int(parts[1])
                except ValueError:
                    pass
    except OSError:
        pass
    return rings_to_exclude, max_ring_number


def _parse_lattice_keys(path: Path) -> dict:
    """Extract optional hex-lattice keys from a v1-style paramstest.

    Recognised keys (case-sensitive, matching MIDAS convention):
      ``PixelLattice``         — ``cartesian`` (default) or ``hex_offset_y``
      ``Apothem``              — float, μm
      ``LatticeOrientation``   — float, deg

    These keys are not present in v1's :class:`CalibrationParams`; we
    re-parse the raw paramstest text to recover them.  Returns an empty
    dict if the file doesn't exist or contains no relevant keys.
    """
    out: dict = {}
    try:
        for line in Path(path).read_text().splitlines():
            line = line.split("#")[0].strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            key = parts[0]
            if key == "PixelLattice":
                val = parts[1].strip()
                if val:
                    out["lattice"] = val
            elif key == "Apothem":
                try:
                    out["Apothem"] = float(parts[1])
                except ValueError:
                    pass
            elif key == "LatticeOrientation":
                try:
                    out["LatticeOrientation"] = float(parts[1])
                except ValueError:
                    pass
    except OSError:
        pass
    return out


def spec_from_v1_file(path) -> CalibrationSpec:
    """Convenience: read a v1 paramstest.txt and produce a v2 spec.

    Also extracts v1's ``RingsToExclude`` / ``MaxRingNumber`` (which the v1
    Python params dataclass doesn't carry) and stores them on the spec for
    downstream pipelines to honour.  The hex-lattice keys
    ``PixelLattice`` / ``Apothem`` / ``LatticeOrientation`` (if present)
    are parsed and written onto ``spec.lattice`` and the matching
    Parameters.
    """
    p = Path(path)
    v1 = V1Params.from_file(p)
    spec = spec_from_v1_params(v1)
    rings, max_ring = _parse_ring_filters(p)
    if rings:
        spec.rings_to_exclude = sorted(set(rings))
    if max_ring > 0:
        spec.max_ring_number = max_ring
    lattice_keys = _parse_lattice_keys(p)
    if lattice_keys:
        if "lattice" in lattice_keys:
            spec.lattice = lattice_keys["lattice"]
        if "Apothem" in lattice_keys and "Apothem" in spec.parameters:
            spec.parameters["Apothem"].init = float(lattice_keys["Apothem"])
        if "LatticeOrientation" in lattice_keys and "LatticeOrientation" in spec.parameters:
            spec.parameters["LatticeOrientation"].init = float(
                lattice_keys["LatticeOrientation"])
    return spec


def add_panel_parameters(spec: CalibrationSpec, n_panels: int,
                         tol_shift_px: float = 1.0,
                         tol_rot_deg: float = 0.5,
                         tol_lsd_um: float = 200.0,
                         tol_p2: float = 1e-3,
                         enable_lsd: bool = False,
                         enable_p2: bool = False) -> None:
    """Inject 5-DOF per-panel parameters into the spec.

    The four parameter blocks are:
      - ``panel_delta_yz``    [N, 2]
      - ``panel_delta_theta`` [N]
      - ``panel_delta_lsd``   [N]   (refined only if enable_lsd)
      - ``panel_delta_p2``    [N]   (refined only if enable_p2)
    """
    spec.add(Parameter(
        name="panel_delta_yz",
        init=torch.zeros(n_panels, 2, dtype=torch.float64),
        refined=True,
        bounds=(-tol_shift_px, tol_shift_px),
    ))
    spec.add(Parameter(
        name="panel_delta_theta",
        init=torch.zeros(n_panels, dtype=torch.float64),
        refined=tol_rot_deg > 0,
        bounds=(-tol_rot_deg, tol_rot_deg) if tol_rot_deg > 0 else None,
    ))
    spec.add(Parameter(
        name="panel_delta_lsd",
        init=torch.zeros(n_panels, dtype=torch.float64),
        refined=enable_lsd,
        bounds=(-tol_lsd_um, tol_lsd_um) if enable_lsd else None,
    ))
    spec.add(Parameter(
        name="panel_delta_p2",
        init=torch.zeros(n_panels, dtype=torch.float64),
        refined=enable_p2,
        bounds=(-tol_p2, tol_p2) if enable_p2 else None,
    ))


def add_panel_zero_sum_constraint(
    spec: CalibrationSpec,
    *,
    lambda_zs: float = 1e6,
) -> None:
    """Replace the ``fix_panel_id`` gauge with a soft Σ panel = 0 penalty.

    Per Wright, Giacobbe & Lawrence Bright (2022, *Crystals* 12(2),
    255 §3.2): the natural gauge for per-panel shifts is to require
    ``Σ panel = 0`` rather than fixing one specific panel to zero.
    This is symmetric across panels and numerically more robust:

      * The constraint adds curvature exactly along the nullspace
        direction (uniform shift of all panels), so the data-determined
        directions are untouched.
      * No "reference panel" is special — useful when a detector
        replacement happens (Wright 2022 §3.2 explicitly mentions
        re-calibration after a module swap).

    Implementation: the in-forward ``fix_panel_id`` mask is disabled
    (``spec.fix_panel_id = -1``), and a flag is set on the spec.  The
    pipeline residual closure must then concatenate
    :func:`midas_calibrate_v2.loss.constraints.zero_sum_residual` onto
    the data residual.  ``single_pv`` does this automatically when
    ``spec.zero_sum_panels`` is True.

    Parameters
    ----------
    spec : :class:`CalibrationSpec`
        Spec that already has panel parameters added (call
        :func:`add_panel_parameters` first).
    lambda_zs : float
        Penalty weight.  Default ``1e6`` makes the constraint
        effectively hard.

    See also
    --------
    :func:`midas_calibrate_v2.loss.constraints.zero_sum_residual`
    """
    spec.fix_panel_id = -1   # disable per-id mask in forward
    spec.zero_sum_panels = True
    spec.zero_sum_lambda = float(lambda_zs)


def add_per_ring_offset(
    spec: CalibrationSpec, n_rings: int,
    *,
    tol_px: float = 2.0,
    lambda_zs: float = 1e6,
) -> None:
    """Inject the per-ring radial offset δr_k (basis-fix F2).

    Adds a refined parameter ``delta_r_k`` of shape ``[n_rings]``,
    initially zero, with bounds ``±tol_px`` (per-ring shift in
    pixels).  Also flips ``spec.zero_sum_dr_k = True`` so the residual
    closure in ``pipelines/single_pv.py`` concatenates the
    ``Σ δr_k = 0`` gauge row from
    :func:`midas_calibrate_v2.loss.constraints.zero_sum_residual`.

    The gauge breaks the per-ring offset's degeneracy with the global
    ``Lsd``: a uniform shift ``δr_k = c`` is mathematically equivalent
    to a small change in ``Lsd``, so without ``Σ = 0`` the LM has a
    zero-eigenvalue mode in that direction.

    Pair this with the F1 fix (refining ``Parallax``) for the F1+F2
    combined entry of ``tab:basis_fixes`` in the paper.
    """
    spec.add(Parameter(
        name="delta_r_k",
        init=torch.zeros(n_rings, dtype=torch.float64),
        refined=True,
        bounds=(-tol_px, tol_px),
    ))
    spec.zero_sum_dr_k = True
    spec.zero_sum_dr_k_lambda = float(lambda_zs)


__all__ = ["spec_from_v1_params", "spec_from_v1_file", "add_panel_parameters",
            "add_panel_zero_sum_constraint", "add_per_ring_offset"]
