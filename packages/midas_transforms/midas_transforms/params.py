"""Parameter parsers for midas-transforms.

Two input formats are supported:

1. **paramstest.txt** (used by ``SaveBinData`` and downstream by ``IndexerOMP``).
   Plain-text key-value pairs; ``RingNumbers`` / ``RingRadii`` repeat. Same
   format as ``midas-index``.

2. **Zarr archive** (used by ``MergeOverlappingPeaksAllZarr``,
   ``CalcRadiusAllZarr``, ``FitSetupZarr``).  Parameters live under
   ``analysis/process/analysis_parameters/<key>``; image data under
   ``exchange/data``. The full key inventory is in ``REQUIRED_FITSETUP_KEYS``
   and ``OPTIONAL_FITSETUP_KEYS`` below.

Per the implementation plan (§8 risk #6), missing required keys raise
``KeyError`` with the full list of required keys, never default silently.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# ---------------------------------------------------------------------------
# paramstest.txt
# ---------------------------------------------------------------------------


@dataclass
class ParamsTest:
    """A typed view of paramstest.txt that ``SaveBinData``/``IndexerOMP`` read.

    The field set mirrors what ``FitSetupParamsAllZarr.c:1579-1634`` emits, so
    ``write_paramstest`` can produce a byte-equivalent file. Older callers that
    only set the indexing-relevant fields (``Lsd``, ``RingNumbers``, …) still
    work — every C-only field has a benign default.
    """

    Wavelength: float = 0.0
    Lsd: float = 0.0
    px: float = 200.0
    StepSizeOrient: float = 0.2
    StepsizePos: float = 100.0
    MarginOme: float = 0.5
    MarginEta: float = 500.0
    MarginRad: float = 500.0          # legacy alias used internally
    MarginRadius: float = 500.0       # value emitted under "MarginRadius"
    MarginRadial: float = 500.0
    EtaBinSize: float = 0.1
    OmeBinSize: float = 0.1
    NoSaveAll: int = 0

    # Sample / illumination geometry (C "FitSetupParamsAllZarr" emits these)
    MaxRingRad: float = 0.0
    Rsample: float = 0.0
    Hbeam: float = 0.0
    BeamSize: float = 0.0
    ExcludePoleAngle: float = 0.0     # = ZarrParams.MinEta
    Wedge: float = 0.0
    MargABC: float = 0.0
    MargABG: float = 0.0
    MinMatchesToAcceptFrac: float = 0.0

    # Refined fit geometry (FitSetupZarr post-refine values; default to the
    # raw geometry when DoFit=0).
    # NB: txFit was historically MISSING (the C FitSetupZarr never emitted
    # tx) — any raw-frame consumer (midas_pf_odf, midas_grain_odf) that
    # built geometry from paramstest got tx=0, a ~0.27° in-plane rotation
    # ≈ 3-4e3 µε fake strain on real Varex data. Emit it always.
    LsdFit: float = 0.0
    YBCFit: float = 0.0
    ZBCFit: float = 0.0
    txFit: float = 0.0
    tyFit: float = 0.0
    tzFit: float = 0.0
    # Full v1-ordered distortion polynomial (the C FitSetupZarr truncated
    # to p0..p3, silently dropping p4..p14 — meter-scale errors for large
    # p3/phi4 detectors when a consumer applied the truncated set).
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
    # Canonical v2 harmonic distortion (midas_distortion basis), carried
    # through from ZarrParams when the archive is calibrate-v2 native.
    # Written as the named keys (iso_R2.., a1.., phi1..) so raw-frame
    # consumers get the authoritative coefficients even when p0..p14 are 0.
    dist_coeffs_v2: Optional[Any] = None

    # Scan description. OmegaStep == 0 means "unknown" and the pair is NOT
    # written (a false 0.0 is worse than absence); consumers must not infer
    # the step from shadow-gapped OmegaRange spans (P0-3).
    OmegaStart: float = 0.0
    OmegaStep: float = 0.0

    # N8: fit_setup intensity spot filter; recorded so reruns see it.
    # Written only when non-zero (no FF behaviour change).
    MinIntegratedIntensity: float = 0.0

    # Refinement weights
    WeightMask: float = 1.0
    WeightFitRMSE: float = 0.0

    RingNumbers: List[int] = field(default_factory=list)
    RingRadii: List[float] = field(default_factory=list)
    OmegaRanges: List[Tuple[float, float]] = field(default_factory=list)
    BoxSizes: List[Tuple[float, float, float, float]] = field(default_factory=list)
    # Multi-detector additions (no-op for single-detector runs):
    #   DetParams[det_id] = {"lsd","y_bc","z_bc","tx","ty","tz","p_distortion"}
    #   RingRadiiPerDet[det_id][ring_nr] = radius
    # Populated when the merged paramstest carries DetParams + RingRadii_DetN
    # rows from midas_ff_pipeline's cross_det_merge stage.
    DetParams: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    RingRadiiPerDet: Dict[int, Dict[int, float]] = field(default_factory=dict)
    LatticeConstant: Tuple[float, float, float, float, float, float] = (
        0.0, 0.0, 0.0, 90.0, 90.0, 90.0,
    )

    SpaceGroup: int = 225
    UseFriedelPairs: int = 1
    # Seed ring for candidate-orientation generation. The scanning C indexer
    # (IndexerScanningOMP.c:1689) only uses observed spots whose ring number
    # equals RingToIndex as seeds; if this is absent/0 it finds zero seeds and
    # silently returns zero grains. Sourced from ZarrParams.OverallRingToIndex.
    RingToIndex: int = 0

    SpotsFileName: str = "InputAll.csv"
    RefinementFileName: str = "InputAllExtraInfoFittingAll.csv"
    IDsFileName: str = "SpotsToIndex.csv"
    OutputFolder: str = ""
    ResultFolder: str = ""
    GrainsFileName: str = ""

    raw: Dict[str, Any] = field(default_factory=dict)

    def get_ring_radius(self, ring_number: int) -> float:
        """Return the radius for ``ring_number`` (or 0.0 if not configured)."""
        for r, rad in zip(self.RingNumbers, self.RingRadii):
            if r == ring_number:
                return rad
        return 0.0

    @property
    def highest_ring_no(self) -> int:
        return max(self.RingNumbers) if self.RingNumbers else 0


_FLOAT_KEYS = {
    "px", "Wavelength", "Lsd", "Distance",
    "StepSizeOrient", "StepsizeOrient", "StepSizePos", "StepsizePos",
    "MarginOme", "MarginEta", "MarginRad", "MarginRadius", "MarginRadial",
    "EtaBinSize", "OmeBinSize",
    "MaxRingRad", "Rsample", "Hbeam", "BeamSize",
    "ExcludePoleAngle", "Wedge",
    "MargABC", "MargABG", "MinMatchesToAcceptFrac",
    "LsdFit", "YBCFit", "ZBCFit", "txFit", "tyFit", "tzFit",
    "p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7",
    "p8", "p9", "p10", "p11", "p12", "p13", "p14",
    "OmegaStart", "OmegaStep", "MinIntegratedIntensity",
    "WeightMask", "WeightFitRMSE",
}
_INT_KEYS = {"NoSaveAll", "SpaceGroup", "UseFriedelPairs", "RingToIndex"}
_STR_KEYS = {
    "SpotsFileName", "RefinementFileName", "IDsFileName",
    "OutputFolder", "ResultFolder", "GrainsFile",
}


def read_paramstest(path: Union[str, Path]) -> ParamsTest:
    """Parse a paramstest.txt file into a ``ParamsTest`` dataclass."""
    p = ParamsTest()
    with open(path, "r") as fp:
        for raw in fp:
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            # The C FitSetupZarr writes each line with a trailing ';'
            # (FitSetupParamsAllZarr.c:1591-1631). Strip it before tokenising.
            line = line.rstrip(";")
            tokens = [t.rstrip(";") for t in line.split()]
            key, args = tokens[0], tokens[1:]
            p.raw[key] = args

            if key == "RingNumbers":
                p.RingNumbers.append(int(args[0]))
            elif key == "RingRadii":
                p.RingRadii.append(float(args[0]))
            elif key == "DetParams":
                # DetParams det_id Lsd y_bc z_bc tx ty tz p0..p10
                if len(args) >= 7:
                    det_id = int(float(args[0]))
                    p.DetParams[det_id] = {
                        "lsd": float(args[1]),
                        "y_bc": float(args[2]),
                        "z_bc": float(args[3]),
                        "tx": float(args[4]),
                        "ty": float(args[5]),
                        "tz": float(args[6]),
                        "p_distortion": [float(v) for v in args[7:7 + 11]],
                    }
            elif key.startswith("RingRadii_Det"):
                # RingRadii_Det<det_id> ring_nr radius
                try:
                    det_id = int(key[len("RingRadii_Det"):])
                except ValueError:
                    continue
                if len(args) >= 2:
                    p.RingRadiiPerDet.setdefault(det_id, {})[int(float(args[0]))] = float(args[1])
            elif key == "OmegaRange":
                p.OmegaRanges.append((float(args[0]), float(args[1])))
            elif key == "BoxSize":
                p.BoxSizes.append(
                    (float(args[0]), float(args[1]), float(args[2]), float(args[3]))
                )
            elif key in ("LatticeParameter", "LatticeConstant"):
                if len(args) >= 6:
                    p.LatticeConstant = (
                        float(args[0]), float(args[1]), float(args[2]),
                        float(args[3]), float(args[4]), float(args[5]),
                    )
                else:
                    a = float(args[0])
                    p.LatticeConstant = (a, a, a, 90.0, 90.0, 90.0)
            elif key == "GrainsFile":
                p.GrainsFileName = args[0]
            elif key == "OverAllRingToIndex":
                # Master-param spelling of the seed ring; the C indexer reads
                # "RingToIndex" so collapse the alias onto that field.
                p.RingToIndex = int(args[0])
            elif key in _FLOAT_KEYS:
                if key in ("Lsd", "Distance"):
                    attr = "Lsd"
                elif key in ("StepSizeOrient", "StepsizeOrient"):
                    attr = "StepSizeOrient"
                elif key in ("StepSizePos", "StepsizePos"):
                    attr = "StepsizePos"
                else:
                    attr = key
                setattr(p, attr, float(args[0]))
                # Mirror MarginRadius -> MarginRad (legacy alias) so old callers
                # that read .MarginRad still see the right value.
                if key == "MarginRadius":
                    p.MarginRad = float(args[0])
            elif key in _INT_KEYS:
                setattr(p, key, int(args[0]))
            elif key in _STR_KEYS:
                attr = "GrainsFileName" if key == "GrainsFile" else key
                setattr(p, attr, args[0])
            # unknown keys are recorded in p.raw and otherwise ignored

    # Rebuild the canonical v2 distortion vector from whatever the file
    # carried: v2 names (calibrate-v2 native) win over the v1-ordered
    # p0..p14 on any collision (v2_coeffs_from_named semantics).
    try:
        from midas_distortion import P_COEF_NAMES, v2_coeffs_from_named
        named: Dict[str, float] = {}
        for i in range(15):
            v = getattr(p, f"p{i}")
            if v:
                named[f"p{i}"] = v
        for nm in P_COEF_NAMES:
            if nm in p.raw and p.raw[nm]:
                named[nm] = float(p.raw[nm][0])
        if named:
            p.dist_coeffs_v2 = v2_coeffs_from_named(named)
    except ImportError:
        pass
    return p


def write_paramstest(p: ParamsTest, path: Union[str, Path]) -> None:
    """Write a ``ParamsTest`` as ``paramstest.txt``.

    Mirrors ``FitSetupParamsAllZarr.c:1579-1634`` exactly: same key order,
    same trailing ``;`` on numeric lines, same %f formatting (default ``%f``
    is 6 fractional digits in C, which we replicate via Python ``%f``).
    """
    def f6(v: float) -> str:
        return f"{v:f}"

    # Guard the three silent-zero-grain traps the scanning C indexer falls into
    # (it crashes loudly on missing files but degrades silently on these):
    #   * RingToIndex == 0      → no seed spots → zero grains.
    #   * RingRadii all zero     → theoretical spots collapse to radius 0.
    #   * no scan tolerance      → every spot attributed to every voxel.
    if p.RingToIndex <= 0:
        warnings.warn(
            "write_paramstest: RingToIndex is unset (0); the scanning C indexer "
            "will find zero seed spots and return zero grains. Set "
            "ZarrParams.OverallRingToIndex before to_paramstest().",
            stacklevel=2,
        )
    if p.RingNumbers and not any(r > 0 for r in p.RingRadii):
        warnings.warn(
            "write_paramstest: RingRadii are all zero/empty while RingNumbers "
            "are set; the C indexer places theoretical spots at radius 0 and "
            "matches nothing. Populate ParamsTest.RingRadii from hkls.csv.",
            stacklevel=2,
        )
    if p.BeamSize <= 0.0:
        warnings.warn(
            "write_paramstest: BeamSize is 0; the scanning indexer's scan-position "
            "filter (ScanPosTol = BeamSize/2) is disabled, so every spot is "
            "attributed to every voxel. Set BeamSize (um) for PF runs.",
            stacklevel=2,
        )

    with open(path, "w") as fp:
        # 1. Lattice + crystal block
        fp.write(
            "LatticeParameter "
            + " ".join(f6(v) for v in p.LatticeConstant) + ";\n"
        )
        fp.write(f"MaxRingRad {f6(p.MaxRingRad)};\n")
        fp.write(f"SpaceGroup {p.SpaceGroup};\n")
        fp.write(f"Wavelength {f6(p.Wavelength)};\n")
        fp.write(f"Distance {f6(p.LsdFit if p.LsdFit else p.Lsd)};\n")
        fp.write(f"Rsample {f6(p.Rsample)};\n")
        fp.write(f"Hbeam {f6(p.Hbeam)};\n")
        fp.write(f"px {f6(p.px)};\n")
        fp.write(f"BeamSize {f6(p.BeamSize)};\n")
        fp.write(f"StepsizePos {f6(p.StepsizePos)};\n")
        fp.write(f"StepsizeOrient {f6(p.StepSizeOrient)};\n")
        fp.write(f"MarginRadius {f6(p.MarginRadius if p.MarginRadius else p.MarginRad)};\n")
        fp.write(f"OmeBinSize {f6(p.OmeBinSize)};\n")
        fp.write(f"EtaBinSize {f6(p.EtaBinSize)};\n")
        fp.write(f"ExcludePoleAngle {f6(p.ExcludePoleAngle)};\n")
        for r in p.RingNumbers:
            fp.write(f"RingNumbers {r};\n")
        for r in p.RingRadii:
            fp.write(f"RingRadii {f6(r)};\n")
        fp.write(f"UseFriedelPairs {p.UseFriedelPairs};\n")
        # Seed ring for the scanning indexer (only emitted when set; a 0 value
        # means no seed ring → zero grains, so warn rather than silently write).
        if p.RingToIndex > 0:
            fp.write(f"RingToIndex {p.RingToIndex};\n")
        fp.write(f"Wedge {f6(p.Wedge)};\n")
        for omr in p.OmegaRanges:
            fp.write(f"OmegaRange {f6(omr[0])} {f6(omr[1])};\n")
        for bx in p.BoxSizes:
            fp.write(
                f"BoxSize {f6(bx[0])} {f6(bx[1])} {f6(bx[2])} {f6(bx[3])};\n"
            )
        fp.write(f"MarginEta {f6(p.MarginEta)};\n")
        fp.write(f"MarginOme {f6(p.MarginOme)};\n")
        fp.write(f"MargABC {f6(p.MargABC)};\n")
        fp.write(f"MargABG {f6(p.MargABG)};\n")
        fp.write(
            f"MarginRadial {f6(p.MarginRadial if p.MarginRadial else p.MarginRad)};\n"
        )
        fp.write(f"MinMatchesToAcceptFrac {f6(p.MinMatchesToAcceptFrac)};\n")
        # NoSaveAll is a user-set switch read by SaveBinData; the C FitSetup
        # never emits it, so only write it when non-default to preserve
        # byte-for-byte parity with the C output for the default case.
        if p.NoSaveAll:
            fp.write(f"NoSaveAll {int(p.NoSaveAll)};\n")
        # 2. File-name + path block (no trailing semicolons in C)
        fp.write(f"SpotsFileName {p.SpotsFileName}\n")
        fp.write(f"RefinementFileName {p.RefinementFileName}\n")
        if p.OutputFolder:
            fp.write(f"OutputFolder {p.OutputFolder}\n")
        if p.ResultFolder:
            fp.write(f"ResultFolder {p.ResultFolder}\n")
        fp.write(f"IDsFileName {p.IDsFileName}\n")
        # 3. Refined-geometry block (no trailing semicolons in C)
        fp.write(f"LsdFit {f6(p.LsdFit if p.LsdFit else p.Lsd)}\n")
        fp.write(f"YBCFit {f6(p.YBCFit)}\n")
        fp.write(f"ZBCFit {f6(p.ZBCFit)}\n")
        # txFit: the C FitSetupZarr never emitted tx — raw-frame consumers
        # then rebuilt geometry with tx=0 (~0.27° in-plane rotation ≈ 3-4e3
        # µε fake strain on SOH/emerson Varex data). Always write it.
        fp.write(f"txFit {f6(p.txFit)}\n")
        fp.write(f"tyFit {f6(p.tyFit)}\n")
        fp.write(f"tzFit {f6(p.tzFit)}\n")
        # Full v1-ordered distortion polynomial (C truncated to p0..p3).
        # %.12g: distortion amplitudes span 1e-9 .. 1e2 — the C %f (6
        # fractional digits) would flush small harmonics to 0.
        for i in range(15):
            fp.write(f"p{i} {getattr(p, f'p{i}'):.12g}\n")
        # Canonical v2 harmonic coefficients when the source archive was
        # calibrate-v2 native (p0..p14 may legitimately be all-zero then).
        # C parsers keyword-match and skip unknown keys.
        if p.dist_coeffs_v2 is not None:
            try:
                from midas_distortion import P_COEF_NAMES
                vals = np.asarray(p.dist_coeffs_v2).flatten()
                if vals.shape[0] == len(P_COEF_NAMES) and np.any(vals != 0.0):
                    for nm, val in zip(P_COEF_NAMES, vals):
                        fp.write(f"{nm} {float(val):.12g}\n")
            except ImportError:
                pass
        # Scan description — omitted when the step is unknown (a false
        # "OmegaStep 0.0" is worse than absence; see P0-3: consumers must
        # never infer the step from shadow-gapped OmegaRange spans).
        if p.OmegaStep:
            fp.write(f"OmegaStart {p.OmegaStart:.12g}\n")
            fp.write(f"OmegaStep {p.OmegaStep:.12g}\n")
        # N8: record the applied intensity filter so reruns see it.
        if p.MinIntegratedIntensity:
            fp.write(f"MinIntegratedIntensity {p.MinIntegratedIntensity:.12g}\n")
        fp.write(f"WeightMask {f6(p.WeightMask)}\n")
        fp.write(f"WeightFitRMSE {f6(p.WeightFitRMSE)}\n")


# ---------------------------------------------------------------------------
# Zarr archive
# ---------------------------------------------------------------------------

# Per dev/implementation_plan.md §8 risk #6: full key inventory locked here.

REQUIRED_FITSETUP_KEYS = (
    "Lsd", "Wavelength", "PixelSize",
    "YCen", "ZCen",
    "RingThresh",
    "LatticeParameter",
    "ty", "tz",
)
# ``tx`` defaults to 0 when absent — older datasets (Wenxi pre-2025)
# only wrote ty/tz to the zarr because tx wasn't a refined tilt at the
# time. Modern datasets write all three; if tx is absent we treat it as 0.0.

OPTIONAL_FITSETUP_KEYS_FLOAT = (
    "Width", "WidthTthPx", "Hbeam", "Rsample", "BeamThickness", "BeamSize",
    "RhoD", "MaxRingRad",
    "MarginRadius", "MarginRadial", "MarginEta", "MarginOme",
    "EtaBinSize", "OmeBinSize",
    "StepSizeOrient", "StepSizePos",
    "MargABG", "MargABC",
    "tolTilts", "tolBC", "tolLsd",
    "p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7",
    "p8", "p9", "p10", "p11", "p12", "p13", "p14",
    "Wedge",
    "MinEta", "MinMatchesToAcceptFrac",
    "MaxOmeSpotIDsToIndex", "MinOmeSpotIDsToIndex",
    "WeightFitRMSE", "WeightMask",
    "tInt", "tGap",
    "Vsample", "DiscArea",
    "OmegaStart", "OmegaStep",
    "MinIntegratedIntensity",
)
OPTIONAL_FITSETUP_KEYS_INT = (
    "DoFit", "UseFriedelPairs", "OverallRingToIndex",
    "MaxNFrames", "SkipFrame", "LayerNr",
    "NPanelsY", "NPanelsZ", "PanelSizeY", "PanelSizeZ",
    "SpaceGroup",
    "DiscModel",
)
OPTIONAL_FITSETUP_KEYS_STR = (
    "PanelShiftsFile", "ResidualCorrectionMap", "ResultFolder",
)
OPTIONAL_FITSETUP_KEYS_ARRAY = (
    "BoxSizes", "OmegaRanges", "PanelGapsY", "PanelGapsZ",
)


@dataclass
class ZarrParams:
    """Geometry + bookkeeping parameters parsed from a Zarr archive.

    Defaults match the C code's initial values
    (``FitSetupParamsAllZarr.c:644-664``).
    """

    # geometry (required)
    Lsd: float = 0.0
    Wavelength: float = 0.0
    PixelSize: float = 200.0
    YCen: float = 0.0
    ZCen: float = 0.0
    tx: float = 0.0
    ty: float = 0.0
    tz: float = 0.0
    LatticeConstant: Tuple[float, float, float, float, float, float] = (
        0.0, 0.0, 0.0, 90.0, 90.0, 90.0,
    )

    # detector / sample
    Hbeam: float = 0.0
    Rsample: float = 0.0
    BeamThickness: float = 0.0
    BeamSize: float = 0.0
    Vsample: float = 0.0
    DiscModel: int = 0
    DiscArea: float = 0.0
    RhoD: float = 0.0
    MaxRingRad: float = 0.0
    NrPixelsY: int = 0
    NrPixelsZ: int = 0
    NrPixels: int = 0
    EndNr: int = 0

    # ring filter
    RingThresh: List[Tuple[int, float]] = field(default_factory=list)
    Width: float = -1.0
    WidthOrig: float = -1.0

    # margins / bin sizes
    MarginRadius: float = 500.0
    MarginRadial: float = 500.0
    MarginEta: float = 500.0
    MarginOme: float = 0.5
    EtaBinSize: float = 0.1
    OmeBinSize: float = 0.1
    StepSizeOrient: float = 0.2
    StepSizePos: float = 5.0
    MargABG: float = 2.0
    MargABC: float = 2.0

    # tolerances (NLopt-era; carried through for paramstest.txt parity)
    tolTilts: float = 1.0
    tolBC: float = 1.0
    tolLsd: float = 5000.0

    # distortion polynomial
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
    # Canonical distortion in the v2 harmonic basis (midas_distortion.
    # P_COEF_NAMES order). Built by read_zarr_params from the v2 names
    # (iso_R2/4/6, a1..a6, phi1..phi6) when present, else from p0..p14.
    dist_coeffs_v2: Optional[Any] = None

    # other
    Wedge: float = 0.0
    MinEta: float = 0.0
    MinMatchesToAcceptFrac: float = 0.0
    MaxOmeSpotIDsToIndex: float = 0.0
    MinOmeSpotIDsToIndex: float = 0.0
    WeightFitRMSE: float = 0.0
    WeightMask: float = 1.0
    tInt: float = 1.0
    tGap: float = 0.0
    # N8: fit_setup intensity spot filter (0 = off — the FF default).
    MinIntegratedIntensity: float = 0.0

    DoFit: int = 0
    UseFriedelPairs: int = 1
    OverallRingToIndex: int = 0
    MaxNFrames: int = 100000
    SkipFrame: int = 0
    LayerNr: int = 0
    SpaceGroup: int = 225

    # panels
    NPanelsY: int = 0
    NPanelsZ: int = 0
    PanelSizeY: int = 0
    PanelSizeZ: int = 0
    PanelGapsY: List[int] = field(default_factory=list)
    PanelGapsZ: List[int] = field(default_factory=list)
    PanelShiftsFile: str = ""

    # files
    ResidualCorrectionMap: str = ""
    ResultFolder: str = ""

    # arrays
    BoxSizes: List[Tuple[float, float, float, float]] = field(default_factory=list)
    OmegaRanges: List[Tuple[float, float]] = field(default_factory=list)

    # scan
    OmegaStart: float = 0.0
    OmegaStep: float = 0.0

    # merge-specific (also in some Zarr archives)
    # C default ``MarginOmegaOverlap = sqrt(4) = 2.0``
    # (MergeOverlappingPeaksAllZarr.c:524).
    OverlapLength: float = 2.0
    UsePixelOverlap: int = 0
    UseMaximaPositions: int = 0

    # raw passthrough for any keys we didn't enumerate
    raw: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.RhoD and not self.MaxRingRad:
            self.MaxRingRad = self.RhoD
        if self.MaxRingRad and not self.RhoD:
            self.RhoD = self.MaxRingRad
        if self.Width == -1.0 and self.WidthOrig != -1.0:
            self.Width = self.WidthOrig

    def to_paramstest(self) -> ParamsTest:
        """Convert to a ``ParamsTest`` view (for downstream binning / indexing).

        Populates the full C-equivalent paramstest.txt schema (Lattice / sample
        / margins / refinement) so ``write_paramstest`` can produce a
        byte-equivalent file. The refined fit fields (``LsdFit`` etc.) default
        to the raw geometry values; ``fit_setup`` overwrites them after a
        DoFit=1 refinement.
        """
        pt = ParamsTest()
        # Geometry / crystal
        pt.Wavelength = self.Wavelength
        pt.Lsd = self.Lsd
        pt.px = self.PixelSize
        pt.LatticeConstant = self.LatticeConstant
        pt.SpaceGroup = self.SpaceGroup
        pt.MaxRingRad = self.MaxRingRad
        # Sample / illumination
        pt.Rsample = self.Rsample
        pt.Hbeam = self.Hbeam
        # C alias: paramstest's "BeamSize" is read from Zarr key "BeamThickness"
        # (FitSetupParamsAllZarr.c:751-754). Prefer explicit BeamSize when set,
        # else fall back to BeamThickness (matches the C aliasing).
        pt.BeamSize = self.BeamSize if self.BeamSize else self.BeamThickness
        pt.ExcludePoleAngle = self.MinEta
        pt.Wedge = self.Wedge
        # Search / refinement margins + steps
        pt.StepSizeOrient = self.StepSizeOrient
        pt.StepsizePos = self.StepSizePos
        pt.MarginOme = self.MarginOme
        pt.MarginEta = self.MarginEta
        pt.MarginRad = self.MarginRadius
        pt.MarginRadius = self.MarginRadius
        pt.MarginRadial = self.MarginRadial
        pt.MargABC = self.MargABC
        pt.MargABG = self.MargABG
        pt.MinMatchesToAcceptFrac = self.MinMatchesToAcceptFrac
        pt.EtaBinSize = self.EtaBinSize
        pt.OmeBinSize = self.OmeBinSize
        pt.UseFriedelPairs = self.UseFriedelPairs
        # Seed ring — without this the scanning C indexer finds zero seeds.
        pt.RingToIndex = self.OverallRingToIndex
        # Distortion polynomial — the FULL v1-ordered set (the C FitSetupZarr
        # truncated to p0..p3, dropping p4..p14), plus the canonical v2
        # harmonic coefficients when the archive is calibrate-v2 native.
        for i in range(15):
            setattr(pt, f"p{i}", getattr(self, f"p{i}"))
        pt.dist_coeffs_v2 = self.dist_coeffs_v2
        # Scan description (write_paramstest omits the pair when step==0).
        pt.OmegaStart = self.OmegaStart
        pt.OmegaStep = self.OmegaStep
        pt.MinIntegratedIntensity = self.MinIntegratedIntensity
        # Refinement weights
        pt.WeightMask = self.WeightMask
        pt.WeightFitRMSE = self.WeightFitRMSE
        # Refined fit defaults: when DoFit=0 the C code writes the raw geometry
        # values into the *Fit fields. Match that. txFit is never refined
        # (refine_5param holds tx fixed, matching C FitTiltBCLsd) so the raw
        # tx IS the fit value.
        pt.LsdFit = self.Lsd
        pt.YBCFit = self.YCen
        pt.ZBCFit = self.ZCen
        pt.txFit = self.tx
        pt.tyFit = self.ty
        pt.tzFit = self.tz
        # Repeated keys
        pt.OmegaRanges = list(self.OmegaRanges)
        pt.BoxSizes = list(self.BoxSizes)
        pt.RingNumbers = [int(rn) for (rn, _) in self.RingThresh]
        # File / output paths (C also emits ResultFolder + RefinementFileName)
        pt.OutputFolder = self.ResultFolder
        pt.ResultFolder = self.ResultFolder
        return pt


def read_zarr_params(zarr_path: Union[str, Path]) -> ZarrParams:
    """Read a MIDAS Zarr archive (typically a .zip) into a ``ZarrParams``.

    Validates required keys and raises ``KeyError`` (per implementation plan
    §8 risk #6) with the full list of required keys when any are missing.
    """
    import zarr

    store = zarr.ZipStore(str(zarr_path), mode="r")
    try:
        root = zarr.group(store=store)
    finally:
        # we keep store open for the lifetime of the read (we don't close
        # until after reading); zarr.group returns a view that needs the store
        pass

    p = ZarrParams()
    seen_required: set = set()

    # Walk all known keys.
    ap_path = "analysis/process/analysis_parameters"

    def _read(key: str, dtype: type, allow_missing: bool = True):
        full = f"{ap_path}/{key}"
        try:
            arr = root[full]
        except KeyError:
            if not allow_missing:
                raise
            return None
        if dtype is str:
            data = arr[...]
            if data.dtype.kind in ("S", "U"):
                val = data.flat[0]
                if isinstance(val, bytes):
                    val = val.decode("utf-8", errors="replace")
                return val
            return None
        return np.asarray(arr[...]).flatten()

    # Required scalar floats / ints
    for key in ("Lsd", "Wavelength", "PixelSize", "YCen", "ZCen", "tx", "ty", "tz"):
        v = _read(key, float, allow_missing=True)
        if v is not None and len(v) > 0:
            setattr(p, key, float(v[0]))
            seen_required.add(key)

    # LatticeParameter (6-vector)
    v = _read("LatticeParameter", float, allow_missing=True)
    if v is not None and len(v) >= 6:
        p.LatticeConstant = tuple(float(x) for x in v[:6])  # type: ignore[assignment]
        seen_required.add("LatticeParameter")

    # RingThresh: shape (N, 2) — (ring_number, threshold)
    try:
        rt = root[f"{ap_path}/RingThresh"][...]
        rt = np.asarray(rt).reshape(-1, 2)
        p.RingThresh = [(int(r[0]), float(r[1])) for r in rt]
        seen_required.add("RingThresh")
    except KeyError:
        pass

    # Optional scalar floats
    for key in OPTIONAL_FITSETUP_KEYS_FLOAT:
        v = _read(key, float)
        if v is not None and len(v) > 0:
            setattr(p, key, float(v[0]))

    # Optional scalar ints
    for key in OPTIONAL_FITSETUP_KEYS_INT:
        v = _read(key, int)
        if v is not None and len(v) > 0:
            setattr(p, key, int(v[0]))

    # Canonical distortion in the v2 harmonic basis. Prefer the v2 names
    # written by calibrate-v2 (iso_R2/4/6, a1..a6, phi1..phi6); fall back to
    # the legacy p0..p14 for old archives.
    from midas_distortion import P_COEF_NAMES, v2_coeffs_from_named
    named: dict = {}
    for nm in P_COEF_NAMES:
        v = _read(nm, float)
        if v is not None and len(v) > 0:
            named[nm] = float(v[0])
    for i in range(15):
        v = _read(f"p{i}", float)
        if v is not None and len(v) > 0:
            named[f"p{i}"] = float(v[0])
    p.dist_coeffs_v2 = v2_coeffs_from_named(named)

    # Optional strings
    for key in OPTIONAL_FITSETUP_KEYS_STR:
        s = _read(key, str)
        if s is not None:
            setattr(p, key, s)

    # Optional 2-D arrays
    try:
        bs = np.asarray(root[f"{ap_path}/BoxSizes"][...]).reshape(-1, 4)
        p.BoxSizes = [(float(r[0]), float(r[1]), float(r[2]), float(r[3])) for r in bs]
    except KeyError:
        pass
    try:
        oms = np.asarray(root[f"{ap_path}/OmegaRanges"][...]).reshape(-1, 2)
        p.OmegaRanges = [(float(r[0]), float(r[1])) for r in oms]
    except KeyError:
        pass
    try:
        gy = np.asarray(root[f"{ap_path}/PanelGapsY"][...]).flatten()
        p.PanelGapsY = [int(x) for x in gy]
    except KeyError:
        pass
    try:
        gz = np.asarray(root[f"{ap_path}/PanelGapsZ"][...]).flatten()
        p.PanelGapsZ = [int(x) for x in gz]
    except KeyError:
        pass

    # Image stack shape: exchange/data is (N_frames, NrPixelsZ, NrPixelsY)
    try:
        shape = root["exchange/data"].shape
        p.EndNr = int(shape[0])
        p.NrPixelsZ = int(shape[1])
        p.NrPixelsY = int(shape[2])
        p.NrPixels = max(p.NrPixelsY, p.NrPixelsZ)
    except KeyError:
        pass

    # OmegaStart / OmegaStep: the analysis_parameters keys (read above via
    # OPTIONAL_FITSETUP_KEYS_FLOAT) are authoritative when non-zero; fall
    # back to the measurement scan_parameters. NB midas_zipper < E5-fix
    # writes a literal 0.0 into analysis_parameters/omegaStep when the CLI
    # flag was absent — the zero check routes around that.
    if not p.OmegaStep:
        try:
            v = root["measurement/process/scan_parameters/step"][...]
            p.OmegaStep = float(np.asarray(v).flat[0])
        except KeyError:
            pass
    if not p.OmegaStart:
        try:
            v = root["measurement/process/scan_parameters/start"][...]
            p.OmegaStart = float(np.asarray(v).flat[0])
        except KeyError:
            pass

    # Validate required keys.
    missing = [k for k in REQUIRED_FITSETUP_KEYS if k not in seen_required]
    if missing:
        raise KeyError(
            f"Required Zarr keys missing from {zarr_path}: {missing}. "
            f"Full required list: {list(REQUIRED_FITSETUP_KEYS)}"
        )

    p.__post_init__()  # propagate RhoD<->MaxRingRad, Width<->WidthOrig
    return p
