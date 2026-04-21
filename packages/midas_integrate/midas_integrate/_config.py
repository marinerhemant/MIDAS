"""Parameters.txt generation for MIDASDetectorMapper + MIDASIntegrator.

MIDAS's C binaries read a plaintext ``key value`` file (one key per line,
space-separated list values). We reuse the low-level writer from
midas_auto_calibrate so both packages emit identical files, and expose a
user-facing ``IntegrationConfig`` dataclass that captures the geometry +
binning knobs the mapper/integrator need.

Public API:
    IntegrationConfig          — geometry + binning + I/O parameters
    IntegrationConfig.from_geometry(geom, ...)  — seed from a refined
                                                   midas_auto_calibrate
                                                   DetectorGeometry.
    IntegrationConfig.to_params(...)            — dict for write_params_file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence, TYPE_CHECKING

# Reuse the low-level writer so sibling packages stay byte-for-byte consistent
# on the MIDAS Parameters.txt format. No duplication of format handling.
from midas_auto_calibrate import write_params_file  # noqa: F401 (re-export)

if TYPE_CHECKING:
    from midas_auto_calibrate import DetectorGeometry


@dataclass
class IntegrationConfig:
    """Inputs to :class:`midas_integrate.Mapper` / :class:`Integrator`.

    Distances are micrometers, angles are degrees, wavelength is angstroms —
    same conventions as ``midas_auto_calibrate.CalibrationConfig``.

    The geometry half of this dataclass mirrors the calibration side so the
    handoff ``auto_calibrate → build_map → integrate`` is a simple
    ``IntegrationConfig.from_geometry(result.geometry)``.
    """

    # ---- Geometry (from a calibration result) ----
    lsd: float = 1_000_000.0
    ybc: float = 1024.0
    zbc: float = 1024.0
    tx: float = 0.0
    ty: float = 0.0
    tz: float = 0.0
    # 15-parameter distortion model
    p0: float = 0.0
    p1: float = 0.0
    p2: float = 0.0
    p3: float = 0.0
    p4: float = 0.0
    p5: float = 0.0
    p6: float = 0.0
    p7: float = 0.0
    p8: float = 0.0
    p9: float = 0.0
    p10: float = 0.0
    p11: float = 0.0
    p12: float = 0.0
    p13: float = 0.0
    p14: float = 0.0

    # ---- Detector ----
    wavelength: float = 0.0
    pixel_size: float = 200.0
    nr_pixels_y: int = 2048
    nr_pixels_z: int = 2048
    rho_d: float | None = None                   # auto-filled when None

    # ---- Binning (mapper + integrator share these) ----
    r_min: float = 10.0                          # pixels
    r_max: float | None = None                   # None → auto = shorter side/2 in px
    r_bin_size: float = 0.25                     # pixels
    eta_min: float = -180.0
    eta_max: float = 180.0
    eta_bin_size: float = 1.0
    q_min: float | None = None                   # Å⁻¹; optional Q-spacing alternative
    q_max: float | None = None
    q_bin_size: float | None = None

    # ---- Corrections ----
    polarization_correction: int = 0             # 0 off, 1 on
    polarization_fraction: float = 0.99
    solid_angle_correction: int = 1
    sub_pixel_level: int = 1
    sub_pixel_cardinal_width: float = 0.0        # 0 off; >0 enables the paper's
                                                 # cardinal-angle aliasing fix
    parallax: float = 0.0

    # ---- Per-panel corrections (Pilatus / Eiger) ----
    n_panels_y: int = 0
    n_panels_z: int = 0
    panel_size_y: int = 0
    panel_size_z: int = 0
    panel_gaps_y: Sequence[int] = field(default_factory=list)
    panel_gaps_z: Sequence[int] = field(default_factory=list)
    panel_shifts_file: str = ""

    # ---- Image transforms + masks ----
    im_trans_opt: Sequence[int] = field(default_factory=list)
    mask_file: str = ""
    residual_correction_map: str = ""

    # ---- Escape hatch ----
    extra_params: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Convenience constructors
    # ------------------------------------------------------------------
    @classmethod
    def from_geometry(
        cls,
        geometry: "DetectorGeometry",
        *,
        nr_pixels_y: int | None = None,
        nr_pixels_z: int | None = None,
        **overrides: Any,
    ) -> "IntegrationConfig":
        """Build an IntegrationConfig seeded from a refined ``DetectorGeometry``.

        Copies every geometry field; binning defaults stay as-is unless
        passed via ``overrides``. ``nr_pixels_y``/``nr_pixels_z`` fall back
        to ``geometry.nr_pixels_y``/``_z`` which may be zero if the geometry
        came from an older JSON — pass them explicitly in that case.
        """
        kwargs: dict[str, Any] = {
            "lsd": geometry.lsd, "ybc": geometry.ybc, "zbc": geometry.zbc,
            "tx": geometry.tx, "ty": geometry.ty, "tz": geometry.tz,
            "wavelength": geometry.wavelength,
            "pixel_size": geometry.px,
            "nr_pixels_y": nr_pixels_y or geometry.nr_pixels_y,
            "nr_pixels_z": nr_pixels_z or geometry.nr_pixels_z,
            "rho_d": geometry.rhod or None,
        }
        for i in range(15):
            kwargs[f"p{i}"] = getattr(geometry, f"p{i}")
        kwargs.update(overrides)
        return cls(**kwargs)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------
    def to_params(self, *, extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Render to a key→value dict ready for :func:`write_params_file`."""
        params: dict[str, Any] = {
            "Lsd": self.lsd,
            "BC": [self.ybc, self.zbc],
            "tx": self.tx, "ty": self.ty, "tz": self.tz,
            "Wavelength": self.wavelength,
            "px": self.pixel_size,
            "NrPixelsY": self.nr_pixels_y, "NrPixelsZ": self.nr_pixels_z,
            "RhoD": self._resolve_rho_d(),
            "RMin": self.r_min,
            "RBinSize": self.r_bin_size,
            "EtaMin": self.eta_min, "EtaMax": self.eta_max,
            "EtaBinSize": self.eta_bin_size,
            "PolarizationCorrection": self.polarization_correction,
            "PolarizationFraction": self.polarization_fraction,
            "SolidAngleCorrection": self.solid_angle_correction,
            "SubPixelLevel": self.sub_pixel_level,
        }

        # Distortion: emit all 15 for unambiguous round-trip even when zero.
        for i in range(15):
            params[f"p{i}"] = getattr(self, f"p{i}")

        resolved_r_max = self.r_max if self.r_max is not None else self._auto_r_max()
        if resolved_r_max is not None:
            params["RMax"] = resolved_r_max

        # Q-space binning (alternative to R-space) — emit only when set.
        if self.q_min is not None:
            params["QMin"] = self.q_min
        if self.q_max is not None:
            params["QMax"] = self.q_max
        if self.q_bin_size is not None:
            params["QBinSize"] = self.q_bin_size

        if self.sub_pixel_cardinal_width:
            params["SubPixelCardinalWidth"] = self.sub_pixel_cardinal_width
        if self.parallax:
            params["Parallax"] = self.parallax

        if self.n_panels_y and self.n_panels_z:
            params["NPanelsY"] = self.n_panels_y
            params["NPanelsZ"] = self.n_panels_z
        if self.panel_size_y and self.panel_size_z:
            params["PanelSizeY"] = self.panel_size_y
            params["PanelSizeZ"] = self.panel_size_z
        if self.panel_gaps_y:
            params["PanelGapsY"] = list(self.panel_gaps_y)
        if self.panel_gaps_z:
            params["PanelGapsZ"] = list(self.panel_gaps_z)
        if self.panel_shifts_file:
            params["PanelShiftsFile"] = self.panel_shifts_file

        if self.im_trans_opt:
            params["ImTransOpt"] = list(self.im_trans_opt)
        if self.mask_file:
            params["MaskFile"] = self.mask_file
        if self.residual_correction_map:
            params["ResidualCorrectionMap"] = self.residual_correction_map

        # Priority: auto < extra_params (config) < extra (per-call).
        if self.extra_params:
            params.update(self.extra_params)
        if extra:
            params.update(extra)

        return params

    def _resolve_rho_d(self) -> float:
        if self.rho_d is not None and self.rho_d > 0:
            return self.rho_d
        return min(self.nr_pixels_y, self.nr_pixels_z) / 2.0 * self.pixel_size

    def _auto_r_max(self) -> float | None:
        if not self.nr_pixels_y or not self.nr_pixels_z:
            return None
        return min(self.nr_pixels_y, self.nr_pixels_z) / 2.0
