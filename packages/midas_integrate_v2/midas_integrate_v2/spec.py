"""IntegrationSpec — v2-native integration parameters.

Mirrors the role of ``midas_integrate.params.IntegrationParams`` but holds
``torch.Tensor`` fields (any of which may have ``requires_grad=True``) and
uses the canonical ``midas_calibrate_v2`` distortion naming
(``iso_R2``/``iso_R4``/``iso_R6``, ``a1``/``phi1``..``a6``/``phi6``)
instead of v1's ``p0``..``p14``. The 15-parameter remap mirrors
:data:`midas_calibrate_v2.compat.to_v1._V2_TO_V1_DISTORTION` so the two
namings are 1-to-1 invertible.

The non-refinable bits — detector geometry (``NrPixelsY/Z``, ``RhoD``,
``pxY/pxZ``), binning, panel layout, file paths — sit alongside the
tensor fields as plain Python scalars / strings, in the same spirit as
``CalibrationSpec`` in calibrate-v2.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import torch


def _t(value, *, dtype=torch.float64, device=None) -> torch.Tensor:
    """Coerce a Python scalar / numpy / tensor into a torch tensor."""
    if isinstance(value, torch.Tensor):
        return value.to(dtype=dtype, device=device) if device else value.to(dtype=dtype)
    return torch.tensor(value, dtype=dtype, device=device)


# v2 canonical distortion-coefficient ordering. Index in this tuple is the
# same as the v1 p-index after applying the v2 → v1 remap.
DISTORTION_NAMES = (
    "iso_R2", "iso_R4", "iso_R6",
    "a1", "phi1", "a2", "phi2", "a3", "phi3",
    "a4", "phi4", "a5", "phi5", "a6", "phi6",
)


@dataclass
class IntegrationSpec:
    """v2-native integration parameter set.

    Tensor fields default to scalar fp64 zeros; users assign torch tensors
    directly (``spec.Lsd = torch.tensor(895_900.0, requires_grad=True)``)
    when joint refinement is desired.

    Geometry, binning and I/O fields match :class:`IntegrationParams`'s
    semantics one-to-one — the only differences are name-canonicalisation
    of the distortion coefficients and the tensor backing for refinable
    quantities.
    """
    # ── Detector geometry ────────────────────────────────────────────────
    NrPixelsY: int = 0
    NrPixelsZ: int = 0
    pxY: float = 0.0
    pxZ: float = 0.0
    Lsd: torch.Tensor = field(default_factory=lambda: _t(0.0))
    BC_y: torch.Tensor = field(default_factory=lambda: _t(0.0))
    BC_z: torch.Tensor = field(default_factory=lambda: _t(0.0))
    RhoD: float = 0.0

    # Pixel-lattice descriptor.  Default "cartesian" preserves prior
    # behaviour; "hex_offset_y" enables PIXIRAD-style honeycomb pixel
    # centroids.  Apothem (μm) is only consulted in the hex branch and
    # then drives pxY/pxZ via pxY=2a, pxZ=a√3.  LatticeOrientation is an
    # in-plane rotation of lattice axes vs detector (Y, Z).
    lattice: str = "cartesian"
    Apothem: torch.Tensor = field(default_factory=lambda: _t(0.0))
    LatticeOrientation: torch.Tensor = field(default_factory=lambda: _t(0.0))

    # Tilts (degrees)
    tx: torch.Tensor = field(default_factory=lambda: _t(0.0))
    ty: torch.Tensor = field(default_factory=lambda: _t(0.0))
    tz: torch.Tensor = field(default_factory=lambda: _t(0.0))

    # 15-parameter harmonic distortion (v2 names)
    iso_R2: torch.Tensor = field(default_factory=lambda: _t(0.0))
    iso_R4: torch.Tensor = field(default_factory=lambda: _t(0.0))
    iso_R6: torch.Tensor = field(default_factory=lambda: _t(0.0))
    a1:  torch.Tensor = field(default_factory=lambda: _t(0.0))
    phi1: torch.Tensor = field(default_factory=lambda: _t(0.0))
    a2:  torch.Tensor = field(default_factory=lambda: _t(0.0))
    phi2: torch.Tensor = field(default_factory=lambda: _t(0.0))
    a3:  torch.Tensor = field(default_factory=lambda: _t(0.0))
    phi3: torch.Tensor = field(default_factory=lambda: _t(0.0))
    a4:  torch.Tensor = field(default_factory=lambda: _t(0.0))
    phi4: torch.Tensor = field(default_factory=lambda: _t(0.0))
    a5:  torch.Tensor = field(default_factory=lambda: _t(0.0))
    phi5: torch.Tensor = field(default_factory=lambda: _t(0.0))
    a6:  torch.Tensor = field(default_factory=lambda: _t(0.0))
    phi6: torch.Tensor = field(default_factory=lambda: _t(0.0))

    # ── Binning ──────────────────────────────────────────────────────────
    RMin: float = 0.0
    RMax: float = 0.0
    RBinSize: float = 0.0
    EtaMin: float = -180.0
    EtaMax: float = 180.0
    EtaBinSize: float = 5.0

    # Q-mode
    QMin: float = 0.0
    QMax: float = 0.0
    QBinSize: float = 0.0
    Wavelength: torch.Tensor = field(default_factory=lambda: _t(0.0))   # Å

    # ── Mapping options ──────────────────────────────────────────────────
    Normalize: int = 1
    SumImages: int = 0
    Write2D: int = 0
    DoBinSort: int = 1
    SubPixelLevel: int = 1
    SubPixelCardinalWidth: float = 5.0
    Parallax: torch.Tensor = field(default_factory=lambda: _t(0.0))
    SolidAngleCorrection: int = 0
    PolarizationCorrection: int = 0
    PolarizationFraction: float = 0.99
    PolarizationPlaneEtaDeg: float = 0.0
    GradientCorrection: int = 0
    NrTransOpt: int = 0
    TransOpt: List[int] = field(default_factory=list)

    GapIntensity: int = 0
    BadPxIntensity: int = 0

    # ── Panels / distortion / residual correction ────────────────────────
    DistortionFile: str = ""
    NPanelsY: int = 0
    NPanelsZ: int = 0
    PanelSizeY: int = 0
    PanelSizeZ: int = 0
    PanelGapsY: List[int] = field(default_factory=list)
    PanelGapsZ: List[int] = field(default_factory=list)
    PanelShiftsFile: str = ""
    ResidualCorrectionMap: str = ""

    # ── Misc ─────────────────────────────────────────────────────────────
    DataDirectory: str = "."
    SkipFrame: int = 0
    FlatFile: str = ""
    MaskFile: str = ""

    # Per-ring offsets (F2 fix): refinable as a 1-D tensor of shape (n_rings,)
    delta_r_k: Optional[torch.Tensor] = None
    ring_d_spacing_A: Optional[torch.Tensor] = None
    ring_two_theta_deg: Optional[torch.Tensor] = None

    # ── Derived ──────────────────────────────────────────────────────────
    @property
    def n_r_bins(self) -> int:
        if self.q_mode_active:
            import math
            return int(math.ceil((self.QMax - self.QMin) / self.QBinSize))
        import math
        return int(math.ceil((self.RMax - self.RMin) / self.RBinSize))

    @property
    def n_eta_bins(self) -> int:
        import math
        return int(math.ceil((self.EtaMax - self.EtaMin) / self.EtaBinSize))

    @property
    def n_bins(self) -> int:
        return self.n_r_bins * self.n_eta_bins

    @property
    def q_mode_active(self) -> bool:
        wl = float(self.Wavelength.detach()) if isinstance(self.Wavelength, torch.Tensor) else self.Wavelength
        return (self.QBinSize > 0 and wl > 0
                and self.QMin > 0 and self.QMax > 0)

    @property
    def n_pixels(self) -> int:
        return self.NrPixelsY * self.NrPixelsZ

    def device(self) -> torch.device:
        return self.Lsd.device

    def dtype(self) -> torch.dtype:
        return self.Lsd.dtype

    def effective_pxYZ(self) -> "tuple[float, float]":
        """Return the (pxY, pxZ) the radial scale should use, in μm.

        For ``lattice='cartesian'`` this is the spec's stored ``pxY/pxZ``
        unchanged.  For ``lattice='hex_offset_y'`` it's derived from the
        apothem: ``pxY=2a, pxZ=a√3`` — the column / row pitches of the
        hex honeycomb.
        """
        if self.lattice == "hex_offset_y":
            import math
            a_val = (float(self.Apothem.detach())
                     if isinstance(self.Apothem, torch.Tensor)
                     else float(self.Apothem))
            return 2.0 * a_val, a_val * math.sqrt(3.0)
        return float(self.pxY), float(self.pxZ)

    def refinable_tensors(self) -> dict:
        """Return the dict of all torch.Tensor fields. Convenience for
        wiring an :class:`torch.optim.Optimizer` over a chosen subset."""
        out = {}
        for name in ("Lsd", "BC_y", "BC_z", "tx", "ty", "tz",
                     "Parallax", "Wavelength", *DISTORTION_NAMES):
            out[name] = getattr(self, name)
        if self.delta_r_k is not None:
            out["delta_r_k"] = self.delta_r_k
        return out

    def validate(self) -> None:
        if self.NrPixelsY <= 0 or self.NrPixelsZ <= 0:
            raise ValueError(
                f"NrPixelsY/NrPixelsZ invalid ({self.NrPixelsY}, {self.NrPixelsZ})"
            )
        if float(self.Lsd) <= 0:
            raise ValueError(f"Lsd invalid ({float(self.Lsd)})")
        if self.lattice == "cartesian":
            if self.pxY <= 0:
                raise ValueError(f"pxY invalid ({self.pxY}) for lattice=cartesian")
        elif self.lattice == "hex_offset_y":
            if float(self.Apothem) <= 0:
                raise ValueError(
                    f"Apothem invalid ({float(self.Apothem)}) for lattice=hex_offset_y"
                )
        else:
            raise ValueError(
                f"Unknown lattice {self.lattice!r}; expected "
                f"'cartesian' or 'hex_offset_y'"
            )
        if self.n_r_bins <= 0 or self.n_eta_bins <= 0:
            raise ValueError(
                f"Invalid bins: nR={self.n_r_bins}, nEta={self.n_eta_bins}"
            )
        # RhoD units self-consistency: RhoD is the distortion normalisation
        # radius and MUST be in micrometres (ρ = R_µm / RhoD). A pixel-valued
        # RhoD silently corrupts the distortion (rings wash out). Auto-correct
        # to µm with a loud warning when it looks like a unit mistake. Only
        # acts on cartesian specs with a positive RhoD (RhoD<=0 = distortion
        # off, a valid choice).
        if (self.lattice == "cartesian" and float(self.RhoD) > 0
                and self.pxY > 0 and self.NrPixelsY > 0 and self.NrPixelsZ > 0):
            from midas_distortion.rhod import resolve_rho_d_um_warn
            self.RhoD = float(resolve_rho_d_um_warn(
                float(self.RhoD),
                int(self.NrPixelsY), int(self.NrPixelsZ),
                float(self.BC_y), float(self.BC_z),
                float(self.pxY),
                float(self.pxZ) if self.pxZ > 0 else None,
                where="IntegrationSpec",
            ))


__all__ = ["IntegrationSpec", "DISTORTION_NAMES"]
