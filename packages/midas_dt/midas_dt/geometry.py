"""Detector geometry, from wherever it happens to live.

Three sources, one destination:

* a ``midas_calibrate_v2.AutoCalibrationResult`` (a fresh calibration)
* a legacy MIDAS parameter file (``ps_dt.txt`` and friends)
* explicit values

all become a :class:`DTGeometry`, which converts to whatever the integrator
wants. Keeping the parsing here means the 2022 parameter files load without
anyone hand-transcribing eight numbers, which is exactly where a sign error
would enter.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = ["DTGeometry", "from_calibration", "parse_legacy_params"]

log = logging.getLogger(__name__)

#: Legacy keys read as plain floats.
_FLOAT_KEYS = (
    "Lsd", "px", "RhoD", "Wavelength", "ty", "tz", "tx",
    "p0", "p1", "p2", "p3", "Width",
    "RMin", "RMax", "RBinSize", "EtaMin", "EtaMax", "EtaBinSize",
    "Rwidth", "etaWidth", "startOme", "omeStep",
)
_INT_KEYS = (
    "NrPixelsY", "NrPixelsZ", "ImTransOpt", "HeadSize", "BytesPerPx",
    "Padding", "startNr", "endNr", "nFrames", "pad", "BadRotation",
    "multipeak", "numProcs", "ExtraPadForTomo", "filt", "DataType",
    "doIntegration", "doLineout", "nFrames_dark",
)
#: Keys that may repeat or carry several values.
_LIST_KEYS = ("rads", "etas", "Rcenters", "BC")
_STR_KEYS = ("Ext", "ext", "FileStem", "fStem", "RawFolder", "SeedFolder",
             "OutFolder", "Dark", "darkFN")


@dataclass
class DTGeometry:
    """Detector geometry and beam energy for an XRD-CT scan."""

    lsd_um: float                 # sample-to-detector distance
    bc_y_px: float                # beam centre, horizontal
    bc_z_px: float                # beam centre, vertical
    px_um: float                  # pixel size (square)
    n_pixels_y: int
    n_pixels_z: int
    wavelength_a: float
    tx_deg: float = 0.0
    ty_deg: float = 0.0
    tz_deg: float = 0.0
    distortion: dict[str, float] = field(default_factory=dict)
    rho_d_um: float = 150000.0

    @property
    def energy_kev(self) -> float:
        """Beam energy from the wavelength. 12.398 keV.Å / λ."""
        return 12.398419843320026 / self.wavelength_a

    def to_integration_spec(self, channel=None):
        """Convert to a ``midas_integrate_v2.IntegrationSpec``.

        ``channel`` optionally supplies the R/eta binning; without it the
        spec carries geometry only and the caller sets the ranges.

        **Distortion.** The legacy files give ``p0..p3``; v2 wants named terms
        (``iso_R2``, ``a1..a6``, ``phi1..phi6``). The mapping is a permutation,
        NOT positional -- ``p0 -> a2``, ``p1 -> a4``, ``p2 -> iso_R2``,
        ``p3 -> phi4`` -- so it is taken from ``midas_distortion``'s canonical
        table rather than written out here. Guessing it (``p0 -> iso_R2`` is
        the natural wrong guess) silently distorts every radius.

        The terms are defined relative to ``RhoD``, so that value travels with
        them; changing one without the other rescales the correction.
        """
        try:
            from midas_integrate_v2 import IntegrationSpec
        except ImportError as exc:
            raise ImportError(
                "converting to an IntegrationSpec needs midas-integrate-v2. "
                "Install with `pip install midas-dt[full]`."
            ) from exc

        import torch

        def t(v: float):
            """IntegrationSpec's geometry fields are torch.Tensor, not float.

            They are declared that way so the whole spec is differentiable for
            joint refinement. Passing a plain float gets as far as
            ``spec.device()``, which then fails on ``float.device`` -- a long
            way from the call site.
            """
            return torch.as_tensor(float(v), dtype=torch.float64)

        kw = dict(
            NrPixelsY=int(self.n_pixels_y), NrPixelsZ=int(self.n_pixels_z),
            pxY=float(self.px_um), pxZ=float(self.px_um),
            Lsd=t(self.lsd_um), BC_y=t(self.bc_y_px), BC_z=t(self.bc_z_px),
            RhoD=float(self.rho_d_um), Wavelength=t(self.wavelength_a),
            tx=t(self.tx_deg), ty=t(self.ty_deg), tz=t(self.tz_deg),
        )
        kw.update({k: t(v) for k, v in self.v2_distortion().items()})
        if channel is not None:
            kw.update(
                RMin=channel.r_min, RMax=channel.r_max, RBinSize=channel.r_bin,
                EtaMin=channel.eta_min, EtaMax=channel.eta_max,
                EtaBinSize=channel.eta_bin,
            )
        return IntegrationSpec(**kw)

    def v2_distortion(self) -> dict[str, float]:
        """Legacy ``p0..p3`` as v2 named distortion terms.

        Uses ``midas_distortion.V1_TO_V2_DISTORTION``, the canonical table.
        Returns an empty dict when no distortion was supplied.
        """
        if not self.distortion:
            return {}
        try:
            from midas_distortion import V1_TO_V2_DISTORTION
        except ImportError as exc:
            raise ImportError(
                "mapping legacy p0..p3 to v2 distortion terms needs "
                "midas-distortion. Install with `pip install midas-dt[full]`."
            ) from exc

        out: dict[str, float] = {}
        for i, name in V1_TO_V2_DISTORTION.items():
            key = f"p{i}"
            if key in self.distortion:
                out[name] = float(self.distortion[key])
        unknown = {k for k in self.distortion
                   if not (k.startswith("p") and k[1:].isdigit())}
        if unknown:
            log.warning(
                "distortion keys %s are not legacy p<n> names and were dropped; "
                "pass them as v2 names if they are already v2 terms", sorted(unknown)
            )
        return out

    def describe(self) -> str:
        return (
            f"Lsd {self.lsd_um:.1f} um, BC ({self.bc_y_px:.3f}, {self.bc_z_px:.3f}) px, "
            f"px {self.px_um:g} um, {self.n_pixels_y}x{self.n_pixels_z}, "
            f"lambda {self.wavelength_a:.6f} A = {self.energy_kev:.2f} keV, "
            f"tilts ({self.tx_deg:g}, {self.ty_deg:g}, {self.tz_deg:g}) deg"
        )


def from_calibration(result: Any) -> DTGeometry:
    """Build from a ``midas_calibrate_v2.AutoCalibrationResult``."""
    missing = [f for f in ("Lsd", "BC_y", "BC_z", "pxY", "wavelength_A")
               if not hasattr(result, f)]
    if missing:
        raise TypeError(
            f"expected an AutoCalibrationResult; the object given lacks {missing}"
        )
    return DTGeometry(
        lsd_um=float(result.Lsd),
        bc_y_px=float(result.BC_y), bc_z_px=float(result.BC_z),
        px_um=float(result.pxY),
        n_pixels_y=int(result.NrPixelsY), n_pixels_z=int(result.NrPixelsZ),
        wavelength_a=float(result.wavelength_A),
        tx_deg=float(getattr(result, "tx", 0.0)),
        ty_deg=float(getattr(result, "ty", 0.0)),
        tz_deg=float(getattr(result, "tz", 0.0)),
        distortion=dict(getattr(result, "distortion", {}) or {}),
    )


def parse_legacy_params(path: str | Path) -> dict:
    """Parse a legacy MIDAS/DT parameter file into a dict.

    Tolerant in the same way the C parser is: unknown keys are kept as raw
    strings, inline ``#`` comments are stripped, and repeated keys accumulate.
    ``BC`` and the various list keys come back as lists of floats.
    """
    out: dict = {}
    for raw in Path(path).read_text(errors="replace").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        key, vals = parts[0], parts[1:]
        if not vals:
            continue
        try:
            if key in _INT_KEYS:
                out[key] = int(float(vals[0]))
            elif key in _FLOAT_KEYS:
                out[key] = float(vals[0])
            elif key in _LIST_KEYS:
                out.setdefault(key, []).extend(float(v) for v in vals)
            elif key in _STR_KEYS:
                out[key] = vals[0]
            elif key == "RadiusToFit":
                out.setdefault("rads", []).append(float(vals[0]))
                out["Rwidth"] = float(vals[-1])
            elif key == "EtaToFit":
                out.setdefault("etas", []).append(float(vals[0]))
                out["etaWidth"] = float(vals[-1])
            else:
                out.setdefault(key, vals[0] if len(vals) == 1 else vals)
        except ValueError:
            out[key] = vals[0] if len(vals) == 1 else vals
    return out


def geometry_from_legacy_params(path: str | Path) -> DTGeometry:
    """Build a :class:`DTGeometry` from a legacy parameter file.

    Reads the 2022 U3O8 files directly. The distortion terms ``p0..p3`` are
    carried through as-is; they are defined relative to ``RhoD``, so that value
    travels with them and must not be changed independently.
    """
    p = parse_legacy_params(path)
    required = ("Lsd", "BC", "px", "NrPixelsY", "NrPixelsZ", "Wavelength")
    missing = [k for k in required if k not in p]
    if missing:
        raise KeyError(f"{path} is missing required keys: {missing}")

    bc = p["BC"]
    if len(bc) < 2:
        raise ValueError(f"{path}: BC needs two values, got {bc}")

    distortion = {k: p[k] for k in ("p0", "p1", "p2", "p3") if k in p}
    geo = DTGeometry(
        lsd_um=p["Lsd"], bc_y_px=bc[0], bc_z_px=bc[1], px_um=p["px"],
        n_pixels_y=p["NrPixelsY"], n_pixels_z=p["NrPixelsZ"],
        wavelength_a=p["Wavelength"],
        tx_deg=p.get("tx", 0.0), ty_deg=p.get("ty", 0.0), tz_deg=p.get("tz", 0.0),
        distortion=distortion, rho_d_um=p.get("RhoD", 150000.0),
    )
    log.info("loaded geometry from %s: %s", Path(path).name, geo.describe())
    return geo
