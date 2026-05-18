"""CalibrationSpec — a powder-calibration ``ParameterSpec`` subclass.

The generic registry-of-parameters is :class:`midas_peakfit.spec.ParameterSpec`.
``CalibrationSpec`` extends it with powder-specific metadata (lattice, ring
table, panel layout, ring filtering) used by the v2 powder forward model.

A spec lists every input the forward model needs.  Refined parameters
participate in autograd; fixed ones are held constant.  Pack/unpack converts
between this dict-of-Parameters and a single torch tensor.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch

from midas_peakfit.parameter import Parameter, GaussianPrior
from midas_peakfit.spec import ParameterSpec


# ----------------------------------------------------------- standard names
#
# The canonical layout of a powder-calibration parameter vector.  See also
# ``midas_calibrate_v2.forward.distortion.P_COEF_NAMES`` for the harmonic
# distortion coefficient slots (``iso_R2/R4/R6`` and ``a_k``/``phi_k`` for
# k=1..6).  Unifies on the names in the J. Appl. Cryst. paper:
#
#   - Lsd, BC_y, BC_z, tx, ty, tz                 — geometry block
#   - iso_R2, iso_R4, iso_R6                      — isotropic radial distortion
#   - a_k, phi_k  (k=1..6)                        — harmonic distortion
#   - Parallax, Wavelength                        — single-value scalars
#   - pxY, pxZ, RhoD                              — pixel-size / detector ρ
#
# Multi-panel detectors add per-panel blocks via
# ``midas_calibrate_v2.compat.from_v1.add_panel_parameters``:
#
#   - panel_delta_yz [N, 2]
#   - panel_delta_theta [N]
#   - panel_delta_lsd [N]
#   - panel_delta_p2 [N]
#
STANDARD_GEOMETRY = (
    "Lsd", "BC_y", "BC_z", "tx", "ty", "tz",
    "iso_R2", "iso_R4", "iso_R6",
    "a1", "phi1", "a2", "phi2", "a3", "phi3",
    "a4", "phi4", "a5", "phi5", "a6", "phi6",
    "Parallax", "Wavelength",
    "pxY", "pxZ", "RhoD",
    # Hex-lattice parameters (only meaningful when CalibrationSpec.lattice
    # != "cartesian"; frozen by default, opt-in to refine via spec.thaw).
    "Apothem", "LatticeOrientation",
)


@dataclass
class CalibrationSpec(ParameterSpec):
    """Powder-calibration spec — a :class:`ParameterSpec` plus crystallography
    and detector metadata.

    Construct via :func:`midas_calibrate_v2.compat.from_v1.spec_from_v1_params`
    (recommended) or by directly populating ``parameters``.
    """

    # crystallography (not parameters by default but available for refinement)
    SpaceGroup: int = 0
    LatticeConstant: Tuple[float, float, float, float, float, float] = (0, 0, 0, 90, 90, 90)
    NrPixelsY: int = 0
    NrPixelsZ: int = 0
    MaxRingRad: float = 0.0
    MinRingRad: float = 0.0

    # ring-table residency: filled at runtime
    ring_d_spacing: Optional[torch.Tensor] = None
    ring_two_theta: Optional[torch.Tensor] = None

    # panel layout (only if multi-panel; absent for monolithic detectors)
    panel_grid: Optional[Tuple[int, int]] = None
    panel_size: Optional[Tuple[int, int]] = None
    panel_gaps: Optional[Tuple[List[int], List[int]]] = None
    fix_panel_id: int = 0

    # Pixel lattice.  "cartesian" (default) is the historical regular
    # rectangular grid; "hex_offset_y" enables the PIXIRAD-style
    # honeycomb pixel arrangement (apothem ``a`` along Y, pitch 2a × a√3,
    # odd-Z rows shifted by +a in Y).  When non-cartesian the spec must
    # carry refinable ``Apothem`` (μm) and ``LatticeOrientation`` (deg)
    # Parameters; refinement defaults to frozen.
    lattice: str = "cartesian"

    # Ring filtering (mirrors v1 C's RingsToExclude / MaxRingNumber).
    rings_to_exclude: List[int] = field(default_factory=list)
    max_ring_number: int = 0   # 0 = no limit


# ----------------------------------------------------------- multi-image spec

@dataclass
class MultiImageSpec:
    """Multi-image / multi-distance calibration spec.

    ``shared`` parameters are common across all images (e.g., distortion
    harmonics, panel shifts, pxY, pxZ).
    ``per_image`` parameters are duplicated per image with independent values
    (e.g., Lsd, BC, tilts).
    """

    shared: Dict[str, Parameter] = field(default_factory=dict)
    per_image: List[Dict[str, Parameter]] = field(default_factory=list)
    crystallography: Optional[CalibrationSpec] = None  # SpaceGroup, lattice, etc.

    def n_images(self) -> int:
        return len(self.per_image)

    def add_shared(self, p: Parameter) -> None:
        if p.name in self.shared:
            raise ValueError(f"Shared parameter {p.name!r} already exists")
        self.shared[p.name] = p

    def add_image(self, params_for_image: Dict[str, Parameter]) -> None:
        self.per_image.append(params_for_image)

    @classmethod
    def from_calibration_specs(cls, specs: List[CalibrationSpec],
                               shared_names: List[str]) -> "MultiImageSpec":
        """Build a multi-image spec from a list of single-image specs.

        Parameters whose name is in ``shared_names`` are taken from the first
        spec and shared across all images; the remaining are made per-image.
        """
        if not specs:
            raise ValueError("Need at least one CalibrationSpec")
        shared: Dict[str, Parameter] = {}
        for n in shared_names:
            if n in specs[0].parameters:
                shared[n] = specs[0].parameters[n]
        per_image: List[Dict[str, Parameter]] = []
        for s in specs:
            local: Dict[str, Parameter] = {}
            for n, p in s.parameters.items():
                if n in shared_names:
                    continue
                local[n] = p
            per_image.append(local)
        out = cls(shared=shared, per_image=per_image, crystallography=specs[0])
        return out


__all__ = ["CalibrationSpec", "MultiImageSpec", "STANDARD_GEOMETRY"]
