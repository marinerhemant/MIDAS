"""MIDAS parameter file (.txt) parser.

Reads the same one-key-per-line format consumed by ``DetectorMapper``,
``IntegratorZarrOMP``, and ``IntegratorFitPeaksGPUStream``. Comments
start with ``#`` and blank lines are ignored.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class IntegrationParams:
    # ── Detector geometry ────────────────────────────────────────────────
    NrPixelsY: int = 0
    NrPixelsZ: int = 0
    pxY: float = 0.0           # pixel size in µm (Y)
    pxZ: float = 0.0           # pixel size in µm (Z)
    Lsd: float = 0.0           # sample-to-detector distance (µm)
    BC_y: float = 0.0          # beam center Y (pixels)
    BC_z: float = 0.0          # beam center Z (pixels)
    RhoD: float = 0.0          # distortion normalisation radius (MICROMETRES; ρ = Rad_µm/RhoD)

    # Tilt angles (degrees)
    tx: float = 0.0
    ty: float = 0.0
    tz: float = 0.0

    # 15-parameter harmonic distortion model
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

    # ── Binning ──────────────────────────────────────────────────────────
    RMin: float = 0.0
    RMax: float = 0.0
    RBinSize: float = 0.0
    EtaMin: float = -180.0
    EtaMax: float = 180.0
    EtaBinSize: float = 5.0

    # Q-mode (overrides R bins if all of these are set)
    QMin: float = 0.0
    QMax: float = 0.0
    QBinSize: float = 0.0
    Wavelength: float = 0.0    # Å

    # 2θ-mode (overrides R bins if all of these are set; mutually
    # exclusive with Q-mode).  Units: degrees.
    TthMin: float = 0.0
    TthMax: float = 0.0
    TthBinSize: float = 0.0

    # ── Mapping options ──────────────────────────────────────────────────
    Normalize: int = 1
    SumImages: int = 0
    Write2D: int = 0
    DoBinSort: int = 1
    SubPixelLevel: int = 1
    SubPixelCardinalWidth: float = 5.0
    Parallax: float = 0.0
    SolidAngleCorrection: int = 0
    PolarizationCorrection: int = 0
    PolarizationFraction: float = 0.99
    # Azimuthal angle (degrees) of the polarization plane with respect to η = 0.
    # 0° matches the legacy "horizontal polarization at η = 0" convention used
    # by pyFAI; 90° rotates the plane to vertical. For unpolarized beams set
    # PolarizationFraction = 0 (the η-dependence vanishes).
    PolarizationPlaneEtaDeg: float = 0.0
    GradientCorrection: int = 0
    NrTransOpt: int = 0
    TransOpt: List[int] = field(default_factory=list)

    # Mask sentinels (read by DetectorMapper)
    GapIntensity: int = 0
    BadPxIntensity: int = 0

    # ── Peak fitting ─────────────────────────────────────────────────────
    DoPeakFit: int = 0
    DoSmoothing: int = 0
    MultiplePeaks: int = 0
    AutoDetectPeaks: int = 0
    SNIPIterations: int = 50
    FitROIPadding: int = 20
    FitROIAuto: int = 0
    PeakLocations: List[float] = field(default_factory=list)

    # ── Per-panel + distortion + residual correction ─────────────────────
    DistortionFile: str = ""       # binary file with two NY*NZ float64 arrays:
                                   #   dY-map then dZ-map (matches DetectorMapper.c)
    NPanelsY: int = 0
    NPanelsZ: int = 0
    PanelSizeY: int = 0
    PanelSizeZ: int = 0
    PanelGapsY: List[int] = field(default_factory=list)   # length NPanelsY-1
    PanelGapsZ: List[int] = field(default_factory=list)   # length NPanelsZ-1
    PanelShiftsFile: str = ""      # text file with `id dY dZ [dTheta dLsd dP2]`
    ResidualCorrectionMap: str = ""   # binary file: NY*NZ float64, ΔR per pixel

    # ── Misc ─────────────────────────────────────────────────────────────
    DataDirectory: str = "."
    SkipFrame: int = 0
    FlatFile: str = ""             # optional path to a per-pixel sensitivity map
    MaskFile: str = ""             # optional path to a binary pixel mask

    # ── Derived ──────────────────────────────────────────────────────────
    @property
    def n_r_bins(self) -> int:
        if self.q_mode_active:
            return int(math.ceil((self.QMax - self.QMin) / self.QBinSize))
        if self.tth_mode_active:
            return int(math.ceil(
                (self.TthMax - self.TthMin) / self.TthBinSize
            ))
        return int(math.ceil((self.RMax - self.RMin) / self.RBinSize))

    @property
    def n_eta_bins(self) -> int:
        return int(math.ceil((self.EtaMax - self.EtaMin) / self.EtaBinSize))

    @property
    def n_bins(self) -> int:
        return self.n_r_bins * self.n_eta_bins

    @property
    def q_mode_active(self) -> bool:
        return (self.QBinSize > 0 and self.Wavelength > 0
                and self.QMin > 0 and self.QMax > 0)

    @property
    def tth_mode_active(self) -> bool:
        return (self.TthBinSize > 0
                and self.TthMax > self.TthMin >= 0.0)

    @property
    def bin_axis(self) -> str:
        """``"R"`` (default), ``"Q"``, or ``"tth"`` --- the radial axis on
        which bin edges are equi-spaced.  Q-mode and 2theta-mode are
        opt-in via ``QBinSize`` / ``TthBinSize`` respectively, and
        mutually exclusive (enforced by :meth:`validate`)."""
        if self.q_mode_active:
            return "Q"
        if self.tth_mode_active:
            return "tth"
        return "R"

    @property
    def n_pixels(self) -> int:
        return self.NrPixelsY * self.NrPixelsZ

    def validate(self) -> None:
        if self.NrPixelsY <= 0 or self.NrPixelsZ <= 0:
            raise ValueError(
                f"NrPixelsY/NrPixelsZ invalid ({self.NrPixelsY}, {self.NrPixelsZ})"
            )
        if self.Lsd <= 0 or self.pxY <= 0:
            raise ValueError(f"Lsd/pxY invalid ({self.Lsd}, {self.pxY})")
        # RhoD units self-consistency: the distortion uses ρ = Rad/RhoD where
        # Rad is in micrometres (see geometry: Yc = (Ycen-Y)*px), so RhoD MUST
        # be µm. A pixel-valued RhoD silently corrupts the distortion. Auto-
        # correct to µm with a loud warning when it looks like a unit mistake.
        if self.RhoD > 0 and self.pxY > 0:
            try:
                from midas_distortion.rhod import resolve_rho_d_um_warn
                self.RhoD = float(resolve_rho_d_um_warn(
                    float(self.RhoD),
                    int(self.NrPixelsY), int(self.NrPixelsZ),
                    float(self.BC_y), float(self.BC_z),
                    float(self.pxY),
                    float(self.pxZ) if self.pxZ > 0 else None,
                    where="IntegrationParams",
                ))
            except ImportError:
                pass   # guard is best-effort if midas-distortion absent
        if self.q_mode_active and self.tth_mode_active:
            raise ValueError(
                "Q-mode (QBinSize>0) and 2theta-mode (TthBinSize>0) are "
                "mutually exclusive; set exactly one (or neither for "
                "default equi-R binning)."
            )
        if self.n_r_bins <= 0 or self.n_eta_bins <= 0:
            raise ValueError(
                f"Invalid bins: nR={self.n_r_bins}, nEta={self.n_eta_bins}"
            )


# Map of parameter-name → (attribute, parser).
# Parsers take the value-string and return the typed value.
def _list_int(s: str) -> List[int]:
    return [int(x) for x in s.split() if x]


def _two_floats(s: str) -> tuple[float, float]:
    a, b = s.split()[:2]
    return float(a), float(b)


_KEY_HANDLERS = {
    # geometry
    "NrPixelsY":        ("NrPixelsY",        int),
    "NrPixelsZ":        ("NrPixelsZ",        int),
    "px":               ("pxY",              float),       # also sets pxZ below
    "pxY":              ("pxY",              float),
    "pxZ":              ("pxZ",              float),
    "Lsd":              ("Lsd",              float),
    "RhoD":             ("RhoD",             float),
    "tx":               ("tx",               float),
    "ty":               ("ty",               float),
    "tz":               ("tz",               float),
    # distortion
    **{f"p{i}": (f"p{i}", float) for i in range(15)},
    # binning
    "RMin":             ("RMin",             float),
    "RMax":             ("RMax",             float),
    "RBinSize":         ("RBinSize",         float),
    "EtaMin":           ("EtaMin",           float),
    "EtaMax":           ("EtaMax",           float),
    "EtaBinSize":       ("EtaBinSize",       float),
    "QMin":             ("QMin",             float),
    "QMax":             ("QMax",             float),
    "QBinSize":         ("QBinSize",         float),
    "Wavelength":       ("Wavelength",       float),
    "TthMin":           ("TthMin",           float),
    "TthMax":           ("TthMax",           float),
    "TthBinSize":       ("TthBinSize",       float),
    # mapping options
    "Normalize":        ("Normalize",        int),
    "SumImages":        ("SumImages",        int),
    "Write2D":          ("Write2D",          int),
    "DoBinSort":        ("DoBinSort",        int),
    "SubPixelLevel":    ("SubPixelLevel",    int),
    "SubPixelCardinalWidth": ("SubPixelCardinalWidth", float),
    "Parallax":         ("Parallax",         float),
    "SolidAngleCorrection": ("SolidAngleCorrection", int),
    "PolarizationCorrection": ("PolarizationCorrection", int),
    "PolarizationFraction":   ("PolarizationFraction", float),
    "PolarizationPlaneEtaDeg": ("PolarizationPlaneEtaDeg", float),
    "GradientCorrection": ("GradientCorrection", int),
    "GapIntensity":     ("GapIntensity",     int),
    "BadPxIntensity":   ("BadPxIntensity",   int),
    # peak fit
    "DoPeakFit":        ("DoPeakFit",        int),
    "DoSmoothing":      ("DoSmoothing",      int),
    "MultiplePeaks":    ("MultiplePeaks",    int),
    "AutoDetectPeaks":  ("AutoDetectPeaks",  int),
    "SNIPIterations":   ("SNIPIterations",   int),
    "FitROIPadding":    ("FitROIPadding",    int),
    "FitROIAuto":       ("FitROIAuto",       int),
    # panels / distortion / residual correction
    "DistortionFile":   ("DistortionFile",   str),
    "NPanelsY":         ("NPanelsY",         int),
    "NPanelsZ":         ("NPanelsZ",         int),
    "PanelSizeY":       ("PanelSizeY",       int),
    "PanelSizeZ":       ("PanelSizeZ",       int),
    "PanelShiftsFile":  ("PanelShiftsFile",  str),
    "ResidualCorrectionMap": ("ResidualCorrectionMap", str),
    # misc
    "DataDirectory":    ("DataDirectory",    str),
    "SkipFrame":        ("SkipFrame",        int),
    "FlatFile":         ("FlatFile",         str),
    "MaskFile":         ("MaskFile",         str),
}


def parse_params(path: str | Path) -> IntegrationParams:
    """Parse a MIDAS parameter file."""
    params = IntegrationParams()
    with open(path, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            tokens = line.split(None, 1)
            if len(tokens) < 2:
                continue
            key, value = tokens[0], tokens[1].strip()

            # Special multi-value keys
            if key == "BC":
                params.BC_y, params.BC_z = _two_floats(value)
                continue
            if key == "NrPixels":          # shortcut: square detector
                n = int(value.split()[0])
                params.NrPixelsY = n
                params.NrPixelsZ = n
                continue
            if key == "ImTransOpt":
                params.TransOpt.append(int(value.split()[0]))
                params.NrTransOpt = len(params.TransOpt)
                continue
            if key == "PeakLocation":
                params.PeakLocations.append(float(value.split()[0]))
                params.MultiplePeaks = 1
                params.DoPeakFit = 1
                params.DoSmoothing = 0
                continue
            if key == "PanelGapsY":
                params.PanelGapsY = _list_int(value)
                continue
            if key == "PanelGapsZ":
                params.PanelGapsZ = _list_int(value)
                continue

            handler = _KEY_HANDLERS.get(key)
            if handler is None:
                continue          # silently ignore unknown keys (matches C behavior)
            attr, parser = handler
            try:
                parsed = parser(value.split()[0]) if parser is not str else value
            except (ValueError, IndexError):
                continue
            setattr(params, attr, parsed)
            # ``px`` is shorthand for both pxY and pxZ if pxZ wasn't set
            if key == "px" and params.pxZ == 0.0:
                params.pxZ = params.pxY

    if params.pxZ == 0.0 and params.pxY != 0.0:
        params.pxZ = params.pxY

    if params.AutoDetectPeaks > 0:
        params.DoPeakFit = 1

    return params
