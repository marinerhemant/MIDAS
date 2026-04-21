"""Parameters.txt generation for MIDASCalibrant.

MIDAS's C binaries read a plaintext ``key value`` file (one key per line,
space-separated list values). This module centralises that format so all
higher-level callers emit identical files.

Public API:
    write_params_file(path, params)   — low-level dict → Parameters.txt
    CalibrationConfig                  — user-facing input dataclass
    CalibrationConfig.to_params(...)   — render to a params dict
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence


def write_params_file(path: str | Path, params: Mapping[str, Any]) -> Path:
    """Write a MIDAS-style Parameters.txt from a flat mapping.

    - Scalars (int/float/str/bool) → ``key value\\n``.
    - Sequences (list/tuple/np.ndarray) → ``key v1 v2 …\\n``.
    - Any entry whose value is None is skipped.
    - Any entry whose value is a list-of-lines (multiline string) emits one
      ``key value\\n`` line per element — used for repeated keys like
      ``ImTransOpt`` or ``PeakLocation``.
    """
    p = Path(path)
    with p.open("w") as f:
        for key, value in params.items():
            if value is None:
                continue
            if isinstance(value, bool):
                f.write(f"{key} {1 if value else 0}\n")
            elif isinstance(value, (list, tuple)) and value and isinstance(value[0], (list, tuple)):
                # Repeated key emission, e.g. PeakLocation appears many times.
                for row in value:
                    f.write(f"{key} {_fmt_seq(row)}\n")
            elif isinstance(value, (list, tuple)):
                f.write(f"{key} {_fmt_seq(value)}\n")
            else:
                f.write(f"{key} {value}\n")
    return p


def _fmt_seq(seq: Sequence[Any]) -> str:
    return " ".join(str(v) for v in seq)


@dataclass
class CalibrationConfig:
    """User-facing inputs to :func:`midas_auto_calibrate.auto_calibrate`.

    Distances are micrometers, wavelength is angstroms. Matches MIDAS
    conventions so numbers from existing parameter files carry over unchanged.
    """

    # Required: what are we calibrating against?
    material: str | None = None                  # e.g. "CeO2" — resolves to lattice params if builtin
    lattice_params: Sequence[float] | None = None  # (a, b, c, α, β, γ) in Å/degrees
    wavelength: float = 0.0                      # Å
    pixel_size: float = 200.0                    # μm

    # Starting-point geometry (refined during calibration)
    lsd: float = 1_000_000.0                     # μm, sample–detector distance
    ybc: float = 1024.0
    zbc: float = 1024.0
    tx: float = 0.0
    ty: float = 0.0
    tz: float = 0.0
    rho_d: float | None = None                   # μm, max ring radius on detector
                                                 # (GetHKLList reads as RhoD; auto-
                                                 # filled from detector geometry
                                                 # if not explicitly set)

    # Detector
    nr_pixels_y: int = 2048
    nr_pixels_z: int = 2048
    space_group: int = 225                       # FCC for CeO2; FCC=225, BCC=229, diamond=227
    im_trans_opt: Sequence[int] = field(default_factory=list)
    mask_file: str = ""
    dark_file: str = ""                          # subtracted from each frame; strongly
                                                 # recommended — without it the E-step
                                                 # often fails to match any rings
    data_type: int = 6                           # 6 = TIFF; see MIDAS manual for other formats

    # Integration extents (overridable; sensible defaults match AutoCalibrateZarr)
    r_min: float = 10.0                          # pixels
    r_max: float | None = None                   # None → auto-fill from max ring + 50 px
    r_bin_size: float = 0.25                     # pixels
    eta_min: float = -180.0
    eta_max: float = 180.0
    eta_bin_size: float = 1.0
    normalize: int = 1
    gradient_correction: int = 1                 # cardinal-angle aliasing fix

    # Fit control
    n_iterations: int = 30
    fit_p_models: str = "tilt,spherical,dipole,trefoil,octupole"
    peak_fit_mode: int = 0                       # 0=pV default, 1=TCH GSAS-II
    fit_parallax: int = 0
    parallax_in: float = 0.0

    # Paths
    folder: str = ""
    residual_corr_map_fn: str = ""

    # Escape hatch — raw MIDAS Parameters.txt key/value pairs that aren't
    # exposed as structured fields yet (panel geometry, ring exclusions,
    # objective weights). See `FF_HEDM/Example/Calibration/parameters.txt`
    # for the full key catalogue. These are emitted verbatim by
    # :meth:`to_params`; structured-field overrides win on conflict.
    extra_params: "dict[str, Any]" = field(default_factory=dict)

    def to_params(
        self,
        *,
        max_r_px: float | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Render this config to a key→value dict ready for ``write_params_file``.

        `max_r_px` — if provided, overrides `r_max` (AutoCalibrateZarr uses
        ``max_r_px + 50`` from the farthest detected ring). If both are None,
        no ``RMax`` is emitted and MIDASCalibrant uses its own default.
        """
        params: dict[str, Any] = {
            "Lsd": self.lsd,
            "BC": [self.ybc, self.zbc],
            "tx": self.tx, "ty": self.ty, "tz": self.tz,
            "Wavelength": self.wavelength,
            "px": self.pixel_size,
            "NrPixelsY": self.nr_pixels_y,
            "NrPixelsZ": self.nr_pixels_z,
            "SpaceGroup": self.space_group,
            "RMin": self.r_min,
            "RBinSize": self.r_bin_size,
            "EtaMin": self.eta_min,
            "EtaMax": self.eta_max,
            "EtaBinSize": self.eta_bin_size,
            "Normalize": self.normalize,
            "GradientCorrection": self.gradient_correction,
            "RhoD": self._resolve_rho_d(),
        }

        # If the caller passed `max_r_px` (the farthest detected ring radius
        # in pixels), apply AutoCalibrateZarr's 50-px padding heuristic.
        # Otherwise use `self.r_max` as-is — it's already the intended RMax.
        if max_r_px is not None:
            params["RMax"] = math.ceil(max_r_px + 50)
        elif self.r_max is not None:
            params["RMax"] = self.r_max

        if self.lattice_params is not None:
            # MIDAS C code reads the key as "LatticeConstant" (see
            # MIDAS_ParamParser.c). Don't confuse with "LatticeParameter".
            params["LatticeConstant"] = list(self.lattice_params)

        if self.im_trans_opt:
            params["ImTransOpt"] = list(self.im_trans_opt)
        if self.mask_file:
            params["MaskFile"] = self.mask_file
        if self.dark_file:
            params["Dark"] = self.dark_file
        if self.data_type:
            params["DataType"] = self.data_type
        if self.folder:
            params["Folder"] = self.folder
        if self.parallax_in != 0.0:
            params["Parallax"] = self.parallax_in
        if self.residual_corr_map_fn:
            params["ResidualCorrectionMap"] = self.residual_corr_map_fn

        # Priority (lowest to highest): auto-filled defaults from structured
        # fields < `extra_params` (config) < `extra` (per-call override).
        # extra_params overrides auto-defaults so users can set RhoD,
        # tolerances, etc. explicitly without touching structured fields.
        if self.extra_params:
            params.update(self.extra_params)

        if extra:
            params.update(extra)

        return params

    def _resolve_rho_d(self) -> float:
        """Max ring radius in μm. User override wins; else half the shorter
        detector side in pixels times pixel pitch (a safe overestimate that
        GetHKLList uses as an upper bound, not an expected ring position)."""
        if self.rho_d is not None:
            return self.rho_d
        half_side_px = min(self.nr_pixels_y, self.nr_pixels_z) / 2.0
        return half_side_px * self.pixel_size
