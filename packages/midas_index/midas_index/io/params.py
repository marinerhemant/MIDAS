"""paramstest.txt parser.

Mirrors `ReadParams` from `FF_HEDM/src/IndexerOMP.c:1281-1547`.

Format: one key per line, "Key value [value...]". Repeated keys
(RingNumbers, RingsToExcludeFraction, RingRadii, OmegaRange, BoxSize)
accumulate. Aliases collapse to the canonical field.

Keys the indexer doesn't consume but are valid MIDAS paramstest entries
(detector distortion, refinement-fit echoes, calc-radius / process-grains
knobs, etc.) are listed in ``_IGNORED_KEYS`` and skipped silently. Any
other unrecognized key warns ONCE per key process-wide (the scanning
indexer re-parses paramstest per voxel, so a per-call warning would spam
the log thousands of times).
"""

from __future__ import annotations

import warnings
from pathlib import Path

from ..params import IndexerParams

# Canonical alias map: source key -> field on IndexerParams.
# The order in C uses startswith on a trailing space, so longer
# names are checked before substrings (e.g. "Distance " before any
# would-be "Dist*" prefix).  We replicate that with a single dict
# keyed by the *exact prefix* the C parser uses.
_FLOAT_KEYS: dict[str, str] = {
    "px": "px",
    "Wavelength": "Wavelength",
    "Distance": "Distance",
    "Lsd": "Distance",
    "Rsample": "Rsample",
    "Hbeam": "Hbeam",
    "StepsizePos": "StepsizePos",
    "StepSizePos": "StepsizePos",     # alias used by midas-fit-setup writer
    "StepsizeOrient": "StepsizeOrient",
    "StepSizeOrient": "StepsizeOrient",
    "MarginOme": "MarginOme",
    "MarginRadius": "MarginRad",
    "MarginRadial": "MarginRadial",
    "MarginEta": "MarginEta",
    "EtaBinSize": "EtaBinSize",
    "OmeBinSize": "OmeBinSize",
    "MinMatchesToAcceptFrac": "MinMatchesToAcceptFrac",
    "Completeness": "MinMatchesToAcceptFrac",
    "ExcludePoleAngle": "ExcludePoleAngle",
    "MinEta": "ExcludePoleAngle",   # IndexerOMP.c:1454 — aliases ExcludePoleAngle
    # --- pf-HEDM scan-aware keys (P5) ---
    "ScanPosTol": "scan_pos_tol_um",
    "BeamSize": "_beam_size_for_default_scan_tol",   # post-processed below
}

_INT_KEYS: dict[str, str] = {
    "SpaceGroup": "SpaceGroup",
    "UseFriedelPairs": "UseFriedelPairs",
    "RingToIndex": "RingToIndex",
}

_STR_KEYS: dict[str, str] = {
    "SpotsFileName": "SpotsFileName",
    "IDsFileName": "IDsFileName",
    "OutputFolder": "OutputFolder",
}

# Keys that are valid MIDAS paramstest entries consumed by *other* stages
# (calibration, peak fitting, fit-setup, refinement, reconstruction) but
# which the indexer legitimately does not read. These are NOT typos, so we
# skip them silently rather than warning. Anything outside both the
# recognized sets above AND this set is a genuine unknown and warns once.
_IGNORED_KEYS: frozenset[str] = frozenset({
    # Detector geometry / distortion (used by transforms + calibration).
    # The distortion polynomial exists in BOTH namings: the v1 p0..p14
    # (midas-transforms >= 0.8.0 writes the full set; C wrote p0..p3) and
    # the v2 harmonic names (iso_R2/4/6, a1..a6, phi1..phi6) written by
    # calibrate-v2-native archives. The indexer applies neither (it works
    # in the DetCor ideal frame) but must recognise both so a v2-named or
    # full-v1 paramstest doesn't spam unknown-key warnings.
    "BC", "tx", "ty", "tz", "RhoD", "MaxRingRad",
    "p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7",
    "p8", "p9", "p10", "p11", "p12", "p13", "p14",
    "iso_R2", "iso_R4", "iso_R6",
    "a1", "a2", "a3", "a4", "a5", "a6",
    "phi1", "phi2", "phi3", "phi4", "phi5", "phi6",
    "DetParams", "Width", "WidthTthPx",
    # Detector pixel dims + scan description (midas-transforms >= 0.8.x).
    "NrPixelsY", "NrPixelsZ", "NrFrames", "OmegaStart",
    "MinIntegratedIntensity",
    # Refinement-fit echoes written by midas-fit-setup
    "LsdFit", "YBCFit", "ZBCFit", "txFit", "tyFit", "tzFit",
    "p0Fit", "p1Fit", "p2Fit", "p3Fit",
    # Lattice / strain refinement margins (used by midas-fit-grain)
    "MargABC", "MargABG", "MargD", "MargAng", "MargPos", "MargPos2",
    # Weighting (refinement)
    "WeightMask", "WeightFitRMSE",
    # Geometry / acquisition (used by peakfit / transforms / forward sim)
    "Wedge", "OmegaStep", "OmegaFirstFile", "GapIntensity", "BadPxIntensity",
    "UpperBoundThreshold", "DoFullImage", "numPxY", "numPxZ", "NrPixels",
    "ImTransOpt", "Padding", "Ext", "FileStem", "StartFileNr",
    "StartFileNrFirstLayer", "NrFilesPerSweep", "numFilesPerScan", "StartNr",
    "EndNr", "Lsd2", "BeamCurrent",
    # I/O paths used by sibling stages
    "RefinementFileName", "ResultFolder", "OutputFolderRecon", "DataFolder",
    # Grain-size / volume (calc-radius, process-grains)
    "Vsample", "BeamThickness", "DiscModel", "DiscArea",
    # Phase bookkeeping
    "NumPhases", "PhaseNr", "MinNrSpots", "Twins", "TakeGrainMax",
    "OverAllRingToIndex",
})

# Process-wide dedup so a genuinely-unknown key warns once, not once per
# read_params() call (the scanning indexer parses paramstest repeatedly).
_WARNED_UNKNOWN_KEYS: set[str] = set()


def read_params(path: str | Path) -> IndexerParams:
    """Parse a paramstest.txt and return an IndexerParams.

    Mirrors `ReadParams` semantics line-for-line.
    """
    p = IndexerParams()
    ring_radii_user: list[float] = []  # parallel to ring_numbers_in_order
    beam_size_um = 0.0                  # captured for default scan_pos_tol

    with open(path, "r") as fp:
        for raw in fp:
            line = raw.rstrip("\n").rstrip("\r")
            if not line.strip():
                continue
            stripped = line.lstrip()
            if stripped.startswith("#"):
                continue
            # The C ``FitSetupZarr`` writer emits ``key value;`` lines (trailing
            # semicolons on numerics). Strip ``;`` from each token so the float
            # / int conversions below work transparently for both writers.
            tokens = [t.rstrip(";") for t in stripped.split()]
            tokens = [t for t in tokens if t]
            if not tokens:
                continue
            key = tokens[0]
            args = tokens[1:]

            # --- repeated/structured keys first ---
            if key == "RingNumbers":
                p.RingNumbers.append(int(args[0]))
                continue
            if key == "RingsToExcludeFraction":
                p.RingsToReject.append(int(args[0]))
                continue
            if key == "RingRadii":
                ring_radii_user.append(float(args[0]))
                continue
            if key == "OmegaRange":
                p.OmegaRanges.append((float(args[0]), float(args[1])))
                continue
            if key == "BoxSize":
                p.BoxSizes.append(
                    (float(args[0]), float(args[1]), float(args[2]), float(args[3]))
                )
                continue
            if key == "GrainsFile":
                p.isGrainsInput = True
                p.GrainsFileName = args[0]
                continue
            if key in ("LatticeParameter", "LatticeConstant"):
                # 6 floats expected; if fewer present, fall back to scalar a (rare).
                if len(args) >= 6:
                    p.LatticeConstant = (
                        float(args[0]), float(args[1]), float(args[2]),
                        float(args[3]), float(args[4]), float(args[5]),
                    )
                else:
                    a = float(args[0])
                    p.LatticeConstant = (a, a, a, 90.0, 90.0, 90.0)
                continue
            if key == "BigDetSize":
                # Deprecated; parsed for backward compat then ignored.
                continue

            # --- multi-detector (pinwheel) per-panel blocks ---
            if key == "DetParams":
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
                continue
            if key.startswith("RingRadii_Det"):
                # RingRadii_Det<det_id> ring_nr radius_um
                try:
                    det_id = int(key[len("RingRadii_Det"):])
                except ValueError:
                    continue
                if len(args) >= 2:
                    p.RingRadiiPerDet.setdefault(det_id, {})[
                        int(float(args[0]))
                    ] = float(args[1])
                continue
            if key.startswith("EtaCoverage_Det"):
                # EtaCoverage_Det<det_id> ring_nr eta_lo_deg eta_hi_deg
                try:
                    det_id = int(key[len("EtaCoverage_Det"):])
                except ValueError:
                    continue
                if len(args) >= 3:
                    p.EtaCoverage.setdefault(det_id, []).append((
                        int(float(args[0])),
                        float(args[1]),
                        float(args[2]),
                    ))
                continue

            # --- scalar typed keys ---
            if key in _FLOAT_KEYS:
                target = _FLOAT_KEYS[key]
                if target == "_beam_size_for_default_scan_tol":
                    # Capture BeamSize for the post-fold default. The C
                    # indexer uses ScanPosTol > 0 ? ScanPosTol : BeamSize/2;
                    # we mirror that below after the full file is parsed.
                    beam_size_um = float(args[0])
                else:
                    setattr(p, target, float(args[0]))
                continue
            if key in _INT_KEYS:
                setattr(p, _INT_KEYS[key], int(args[0]))
                continue
            if key in _STR_KEYS:
                setattr(p, _STR_KEYS[key], args[0])
                continue

            # Valid sibling-stage keys: skip silently (not typos).
            if key in _IGNORED_KEYS:
                continue
            # Genuine unknown: warn once per key, process-wide.
            if key not in _WARNED_UNKNOWN_KEYS:
                _WARNED_UNKNOWN_KEYS.add(key)
                warnings.warn(
                    f"Unknown key '{key}' in paramstest at line: {line!r}",
                    stacklevel=2,
                )

    # IndexerOMP.c:1535-1538 — fold parallel (RingNumbers, RingRadiiUser) into sparse map
    for ring_nr, radius in zip(p.RingNumbers, ring_radii_user):
        p.RingRadii[ring_nr] = radius

    # IndexerScanningOMP.c:107-110 (scan tolerance default).
    # Production C code: ``scanTol = ScanPosTol > 0 ? ScanPosTol : BeamSize / 2``.
    # If neither was set in the file, leave at 0 ⇒ scan filter disabled
    # (FF default — preserves regression on every existing FF test).
    if p.scan_pos_tol_um <= 0.0 and beam_size_um > 0.0:
        p.scan_pos_tol_um = beam_size_um / 2.0

    return p
