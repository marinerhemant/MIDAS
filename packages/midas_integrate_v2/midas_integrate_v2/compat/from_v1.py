"""Build a v2 :class:`IntegrationSpec` from a v1 :class:`IntegrationParams`.

The 15-coefficient v1 → v2 distortion remap is the inverse of
:data:`midas_calibrate_v2.compat.to_v1._V2_TO_V1_DISTORTION` and matches
:data:`midas_integrate.compat.from_v2._V2_TO_V1_DISTORTION` so a
``v1 → v2 → v1`` round-trip is the identity on every field.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch

from midas_integrate.params import IntegrationParams, parse_params

from ..spec import IntegrationSpec, _t


def _parse_lattice_keys(path: Path) -> dict:
    """Extract optional hex-lattice keys from a v1-style paramstest.

    Mirrors :func:`midas_calibrate_v2.compat.from_v1._parse_lattice_keys`
    so calibrate-v2 and integrate-v2 read the same paramstest dialect.
    Recognised keys: ``PixelLattice``, ``Apothem``, ``LatticeOrientation``.
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


# v1 p-index → v2 canonical distortion name. Inverse of the v2-to-v1 map.
_V1_TO_V2_DISTORTION = {
    "p2": "iso_R2", "p5": "iso_R4", "p4": "iso_R6",
    "p7": "a1",  "p8": "phi1",
    "p0": "a2",  "p6": "phi2",
    "p9": "a3",  "p10": "phi3",
    "p1": "a4",  "p3": "phi4",
    "p11": "a5", "p12": "phi5",
    "p13": "a6", "p14": "phi6",
}


def spec_from_v1_params(
    p: IntegrationParams,
    *,
    dtype: torch.dtype = torch.float64,
    device: Optional[torch.device] = None,
    requires_grad: bool = False,
) -> IntegrationSpec:
    """Convert a v1 :class:`IntegrationParams` to a v2 :class:`IntegrationSpec`.

    Refinable fields (geometry, tilts, distortion, Parallax, Wavelength)
    become torch tensors of the requested dtype/device with the requested
    ``requires_grad`` flag. Non-refinable fields (binning, panels, file
    paths, NrPixelsY/Z, RhoD) pass through as Python scalars / lists /
    strings unchanged.
    """
    s = IntegrationSpec()

    # Detector geometry
    s.NrPixelsY = int(p.NrPixelsY)
    s.NrPixelsZ = int(p.NrPixelsZ)
    s.pxY = float(p.pxY)
    s.pxZ = float(p.pxZ)
    s.RhoD = float(p.RhoD)

    def _mk(value: float) -> torch.Tensor:
        t = _t(value, dtype=dtype, device=device)
        if requires_grad:
            t = t.detach().clone().requires_grad_(True)
        return t

    s.Lsd = _mk(p.Lsd)
    s.BC_y = _mk(p.BC_y)
    s.BC_z = _mk(p.BC_z)
    s.tx = _mk(p.tx)
    s.ty = _mk(p.ty)
    s.tz = _mk(p.tz)
    s.Parallax = _mk(p.Parallax)
    s.Wavelength = _mk(p.Wavelength)

    # Distortion remap p0..p14 → iso_R*/a*/phi*
    for v1_key, v2_name in _V1_TO_V2_DISTORTION.items():
        setattr(s, v2_name, _mk(getattr(p, v1_key)))

    # Binning
    s.RMin = float(p.RMin); s.RMax = float(p.RMax)
    s.RBinSize = float(p.RBinSize)
    s.EtaMin = float(p.EtaMin); s.EtaMax = float(p.EtaMax)
    s.EtaBinSize = float(p.EtaBinSize)
    s.QMin = float(p.QMin); s.QMax = float(p.QMax)
    s.QBinSize = float(p.QBinSize)

    # Mapping options
    s.Normalize = int(p.Normalize)
    s.SumImages = int(p.SumImages)
    s.Write2D = int(p.Write2D)
    s.DoBinSort = int(p.DoBinSort)
    s.SubPixelLevel = int(p.SubPixelLevel)
    s.SubPixelCardinalWidth = float(p.SubPixelCardinalWidth)
    s.SolidAngleCorrection = int(p.SolidAngleCorrection)
    s.PolarizationCorrection = int(p.PolarizationCorrection)
    s.PolarizationFraction = float(p.PolarizationFraction)
    s.PolarizationPlaneEtaDeg = float(p.PolarizationPlaneEtaDeg)
    s.GradientCorrection = int(p.GradientCorrection)
    s.NrTransOpt = int(p.NrTransOpt)
    s.TransOpt = list(p.TransOpt)
    s.GapIntensity = int(p.GapIntensity)
    s.BadPxIntensity = int(p.BadPxIntensity)

    # Panels & residual correction
    s.DistortionFile = str(p.DistortionFile)
    s.NPanelsY = int(p.NPanelsY)
    s.NPanelsZ = int(p.NPanelsZ)
    s.PanelSizeY = int(p.PanelSizeY)
    s.PanelSizeZ = int(p.PanelSizeZ)
    s.PanelGapsY = list(p.PanelGapsY)
    s.PanelGapsZ = list(p.PanelGapsZ)
    s.PanelShiftsFile = str(p.PanelShiftsFile)
    s.ResidualCorrectionMap = str(p.ResidualCorrectionMap)

    # Misc
    s.DataDirectory = str(p.DataDirectory)
    s.SkipFrame = int(p.SkipFrame)
    s.FlatFile = str(p.FlatFile)
    s.MaskFile = str(p.MaskFile)

    return s


def spec_from_v1_paramstest(
    path: str | Path,
    *,
    dtype: torch.dtype = torch.float64,
    device: Optional[torch.device] = None,
    requires_grad: bool = False,
) -> IntegrationSpec:
    """Read a v1 paramstest file and return a v2 :class:`IntegrationSpec`.

    Also parses the v2-introduced hex-lattice keys ``PixelLattice``,
    ``Apothem``, ``LatticeOrientation`` and writes them onto the spec
    (defaults: ``lattice='cartesian'``, ``Apothem=0``,
    ``LatticeOrientation=0``).
    """
    p = parse_params(path)
    spec = spec_from_v1_params(p, dtype=dtype, device=device,
                                requires_grad=requires_grad)
    keys = _parse_lattice_keys(Path(path))
    if "lattice" in keys:
        spec.lattice = keys["lattice"]
    if "Apothem" in keys:
        spec.Apothem = _t(keys["Apothem"], dtype=dtype, device=device)
        if requires_grad:
            spec.Apothem = spec.Apothem.detach().clone().requires_grad_(True)
    if "LatticeOrientation" in keys:
        spec.LatticeOrientation = _t(keys["LatticeOrientation"],
                                      dtype=dtype, device=device)
        if requires_grad:
            spec.LatticeOrientation = (
                spec.LatticeOrientation.detach().clone().requires_grad_(True)
            )
    return spec


__all__ = ["spec_from_v1_params", "spec_from_v1_paramstest",
           "_V1_TO_V2_DISTORTION"]
