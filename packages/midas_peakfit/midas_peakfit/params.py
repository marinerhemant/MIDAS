"""Dataclass mirror of C ImageMetadata + AnalysisParams."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np


# Defaults from PeaksFittingOMPZarrRefactor.c (DEFAULT_* constants).
DEFAULT_NR_PIXELS = 2048
DEFAULT_BC = 1.0
DEFAULT_INT_SAT = 14000.0
DEFAULT_LSD = 1_000_000.0
DEFAULT_PIXEL_SIZE = 200.0
DEFAULT_WIDTH = 1000.0
DEFAULT_WAVELENGTH = 0.189714
MAX_OVERLAPS_PER_IMAGE = 10000


@dataclass
class ZarrParams:
    """All parameters parsed from the Zarr archive.

    Mirrors C ``ImageMetadata`` and ``AnalysisParams`` combined; field names
    intentionally match the C struct names (case-preserving) where possible.
    """

    # ── Frame/detector metadata ─────────────────────────────────────
    nFrames: int = 0
    NrPixelsY: int = 0
    NrPixelsZ: int = 0
    NrPixels: int = 0  # max(Y, Z), set after parsing
    nDarks: int = 0
    nFloods: int = 0
    nMasks: int = 0
    bytesPerPx: int = 2
    pixelType: str = "uint16"  # "uint16", "int32", "uint32", "float32", "float64"

    # ── Scan parameters ──────────────────────────────────────────────
    omegaStart: float = 0.0
    omegaStep: float = 1.0
    omegaCenter: Optional[np.ndarray] = None  # (nFrames,) double or None
    skipFrame: int = 0
    doPeakFit: int = 1
    localMaximaOnly: int = 0

    # ── Detector geometry ────────────────────────────────────────────
    Ycen: float = float(DEFAULT_NR_PIXELS) / 2
    Zcen: float = float(DEFAULT_NR_PIXELS) / 2
    px: float = DEFAULT_PIXEL_SIZE  # pixel size, µm
    Lsd: float = DEFAULT_LSD
    Width: float = DEFAULT_WIDTH  # µm at parse, divided by px after parsing
    RhoD: float = float(DEFAULT_NR_PIXELS) * DEFAULT_PIXEL_SIZE
    Wavelength: float = DEFAULT_WAVELENGTH

    # ── Distortion polynomial ────────────────────────────────────────
    tx: float = 0.0
    ty: float = 0.0
    tz: float = 0.0
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
    # Canonical distortion coefficients in the v2 harmonic basis
    # (midas_distortion.P_COEF_NAMES order). Populated by the zarr loader from
    # the v2 names (iso_R2/4/6, a1..a6, phi1..phi6) when present, else derived
    # from the legacy p0..p14. This is the source of truth the geometry uses.
    dist_coeffs_v2: Optional["np.ndarray"] = None

    # ── Ring / threshold configuration ───────────────────────────────
    RingNrs: List[int] = field(default_factory=list)
    Thresholds: List[float] = field(default_factory=list)
    nRingsThresh: int = 0
    DoFullImage: int = 0

    # ── Peak-fitting controls ────────────────────────────────────────
    minNrPx: int = 1
    maxNrPx: int = 10000
    maxNPeaks: int = 400
    IntSat: float = DEFAULT_INT_SAT
    zDiffThresh: float = 0.0
    BadPxIntensity: float = 0.0
    makeMap: int = 0  # set to 1 if BadPxIntensity key was present
    bc: float = DEFAULT_BC

    # ── Image transformation (applied in order) ──────────────────────
    TransOpt: List[int] = field(default_factory=list)
    nImTransOpt: int = 0

    # ── Multi-panel ──────────────────────────────────────────────────
    NPanelsY: int = 0
    NPanelsZ: int = 0
    PanelSizeY: int = 0
    PanelSizeZ: int = 0
    PanelGapsY: List[int] = field(default_factory=list)
    PanelGapsZ: List[int] = field(default_factory=list)
    PanelShiftsFile: Optional[str] = None

    # ── Output paths ─────────────────────────────────────────────────
    ResultFolder: str = "./"
    LayerNr: int = 1

    # ── Optional residual map ────────────────────────────────────────
    ResidualCorrectionMap: Optional[str] = None
    residualMap: Optional[np.ndarray] = None  # (Y, Z) double if loaded

    # ── Loaded calibration arrays (populated by zarr_io reader) ──────
    dark: Optional[np.ndarray] = None  # averaged dark, (Y, Z) float64
    flood: Optional[np.ndarray] = None  # (Y, Z) float64
    mask: Optional[np.ndarray] = None  # (Y, Z) float64 (0=good, >0=bad)

    def finalize(self) -> None:
        """Apply the C tool's post-parse transformations.

        Replicates the tail of ``parseZarrMetadata`` in the C code:
        - ``NrPixels = max(NrPixelsY, NrPixelsZ)``
        - subtract ``skipFrame`` from ``nFrames`` and ``nDarks``
        - convert ``Width`` from µm to pixels (``Width /= px``)
        """
        self.NrPixels = max(self.NrPixelsY, self.NrPixelsZ)
        if self.skipFrame > 0:
            self.nFrames -= self.skipFrame
            self.nDarks = max(0, self.nDarks - self.skipFrame)
        self.Width = self.Width / self.px

    def block_frame_range(self, blockNr: int, nBlocks: int) -> tuple[int, int]:
        """Compute frame range for a block, matching C semantics.

        ``startFileNr = ceil(nFrames/nBlocks) * blockNr`` clamped to ``[0, nFrames)``.
        """
        chunk = (self.nFrames + nBlocks - 1) // nBlocks
        start = chunk * blockNr
        end = min(start + chunk, self.nFrames)
        return start, end

    def block_frame_indices(
        self, blockNr: int, nBlocks: int, *, interleave: bool = False
    ) -> list[int]:
        """List of frame indices this block owns.

        ``interleave=False`` (default): contiguous slice — block N gets
        ``[N×chunk, (N+1)×chunk)``. Matches C semantics, easy to reason about.

        ``interleave=True``: round-robin striping — block N gets every
        ``nBlocks``-th frame starting at N. Useful for load-balancing when
        peak density correlates with omega: contiguous blocks can land on
        omega ranges full of dense overlapping spots, while striped blocks
        spread that variance across all GPUs.
        """
        if interleave:
            return list(range(blockNr, self.nFrames, nBlocks))
        start, end = self.block_frame_range(blockNr, nBlocks)
        return list(range(start, end))

    def result_folder_temp(self) -> Path:
        """Resolve the {ResultFolder}/Temp directory (creating if absent)."""
        out = Path(self.ResultFolder) / "Temp"
        out.mkdir(parents=True, exist_ok=True)
        return out

    def __repr__(self) -> str:  # pragma: no cover — diagnostic
        head = (
            f"<ZarrParams nFrames={self.nFrames} "
            f"NrPixels={self.NrPixels} ({self.NrPixelsY}x{self.NrPixelsZ}) "
            f"pixelType={self.pixelType} "
            f"nRingsThresh={self.nRingsThresh} "
            f"DoFullImage={self.DoFullImage} "
            f"ResultFolder={self.ResultFolder!r}>"
        )
        return head

    def dump(self) -> str:
        """Pretty-printed parameter dump matching C ``printAllParameters``."""
        lines = [
            "===========================================================",
            "            Parameters Read from Zarr Archive            ",
            "===========================================================",
            "",
            "--- Image Metadata ---",
            f"  nFrames            : {self.nFrames}",
            f"  NrPixelsY          : {self.NrPixelsY}",
            f"  NrPixelsZ          : {self.NrPixelsZ}",
            f"  NrPixels           : {self.NrPixels}",
            f"  nDarks             : {self.nDarks}",
            f"  nFloods            : {self.nFloods}",
            f"  nMasks             : {self.nMasks}",
            f"  pixelType          : {self.pixelType}",
            f"  bytesPerPx         : {self.bytesPerPx}",
            f"  omegaStart         : {self.omegaStart}",
            f"  omegaStep          : {self.omegaStep}",
            f"  skipFrame          : {self.skipFrame}",
            f"  doPeakFit          : {self.doPeakFit}",
            f"  LocalMaximaOnly    : {self.localMaximaOnly}",
            "",
            "--- Analysis Parameters ---",
            f"  bc                 : {self.bc}",
            f"  Ycen               : {self.Ycen}",
            f"  Zcen               : {self.Zcen}",
            f"  IntSat             : {self.IntSat}",
            f"  Lsd                : {self.Lsd}",
            f"  px                 : {self.px}",
            f"  Width (px)         : {self.Width}",
            f"  RhoD               : {self.RhoD}",
            f"  Wavelength         : {self.Wavelength}",
            f"  tx,ty,tz           : {self.tx}, {self.ty}, {self.tz}",
            f"  p0..p5             : {self.p0} {self.p1} {self.p2} {self.p3} {self.p4} {self.p5}",
            f"  p6..p14            : {self.p6} {self.p7} {self.p8} {self.p9} {self.p10} {self.p11} {self.p12} {self.p13} {self.p14}",
            f"  zDiffThresh        : {self.zDiffThresh}",
            f"  minNrPx, maxNrPx   : {self.minNrPx}, {self.maxNrPx}",
            f"  DoFullImage        : {self.DoFullImage}",
            f"  LayerNr            : {self.LayerNr}",
            f"  maxNPeaks          : {self.maxNPeaks}",
            f"  BadPxIntensity     : {self.BadPxIntensity} (makeMap={self.makeMap})",
            f"  nImTransOpt        : {self.nImTransOpt} → {self.TransOpt}",
            f"  nRingsThresh       : {self.nRingsThresh} → {list(zip(self.RingNrs, self.Thresholds))}",
            f"  ResultFolder       : {self.ResultFolder}",
            f"  Panels             : {self.NPanelsY} x {self.NPanelsZ}, "
            f"size={self.PanelSizeY}x{self.PanelSizeZ}, "
            f"gapsY={self.PanelGapsY} gapsZ={self.PanelGapsZ}",
            f"  PanelShiftsFile    : {self.PanelShiftsFile}",
            f"  ResidualCorrMap    : {self.ResidualCorrectionMap}",
            "===========================================================",
        ]
        return "\n".join(lines)


def resolve_result_folder(
    cli_arg: Optional[str], zarr_value: str, fallback: str = "./"
) -> str:
    """Resolution rule: CLI arg > Zarr-stored ResultFolder > './'."""
    if cli_arg is not None and cli_arg.strip():
        return cli_arg
    if zarr_value and zarr_value.strip():
        return zarr_value
    return fallback


def resolve_do_peak_fit(cli_arg: Optional[int], zarr_value: int) -> int:
    """Resolution rule: CLI arg > Zarr-stored doPeakFit."""
    if cli_arg is not None:
        return int(cli_arg)
    return int(zarr_value)


__all__ = [
    "ZarrParams",
    "MAX_OVERLAPS_PER_IMAGE",
    "DEFAULT_NR_PIXELS",
    "DEFAULT_INT_SAT",
    "resolve_result_folder",
    "resolve_do_peak_fit",
]
