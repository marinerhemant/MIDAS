"""Re-read ``Parameters.txt`` into an existing ``.MIDAS.zip``, safely.

``zip_convert`` reuses any archive it finds, and every downstream stage reads
its geometry and thresholds from the **zarr**, not from ``Parameters.txt``. So
editing a parameter and re-running into the same result folder used to keep the
old value with no warning and an exit code of 0.

The rewrite itself lives in ``midas_zipper.param_refresh`` (it owns the zarr
schema and the type coercion). This module adds the part only the pipeline
knows: **which stage outputs a given parameter invalidates.** Refreshing
``RingThresh`` while ``Temp/AllPeaks_PS.bin`` still sits on disk would just move
the staleness one stage downstream -- peakfit skips on that file, so the zarr
would be right and the peaks would still be wrong. That is refused, not warned
about.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .._logging import LOG

#: Pipeline order, earliest first. A changed key invalidates its own stage and
#: every stage after it.
STAGE_ORDER: Tuple[str, ...] = (
    "hkl", "peakfit", "transforms", "binning", "indexing", "refinement",
    "process_grains",
)

#: Files whose presence makes a stage skip (or which are simply its result).
#: Paths are relative to the layer dir (FF) or the scan dir (PF).
STAGE_SENTINELS: Dict[str, Tuple[str, ...]] = {
    "hkl": ("hkls.csv",),
    "peakfit": ("Temp/AllPeaks_PS.bin", "Temp/AllPeaks_PX.bin"),
    "transforms": ("InputAll.csv", "InputAllExtraInfoFittingAll.csv",
                   "paramstest.txt"),
    "binning": ("Spots.bin", "ExtraInfo.bin", "Data.bin", "nData.bin"),
    "indexing": ("Output/IndexBest_all.bin", "Output/IndexKey_all.bin"),
    "refinement": ("Results",),
    "process_grains": ("Grains.csv",),
}

#: Earliest stage each parameter feeds. Anything not listed falls back to
#: ``hkl`` -- an unmapped key has an unknown blast radius, and the safe
#: assumption is that it invalidates everything.
KEY_EARLIEST_STAGE: Dict[str, str] = {
    # ── hkl: ring list / d-spacings / ring radii ──
    "LatticeConstant": "hkl", "LatticeParameter": "hkl", "SpaceGroup": "hkl",
    "Wavelength": "hkl", "MaxRingRad": "hkl", "RhoD": "hkl", "px": "hkl",
    "PixelSize": "hkl", "Lsd": "hkl", "RingsToExclude": "hkl",
    "OverAllRingToIndex": "hkl", "OverallRingToIndex": "hkl",
    "PhaseNr": "hkl", "NumPhases": "hkl",
    # ── peakfit: the peak search itself ──
    "RingThresh": "peakfit", "MinPeakSNR": "peakfit", "BgSubtract": "peakfit",
    "BgNSectors": "peakfit", "RMin": "peakfit", "RMax": "peakfit",
    "MaxNrPx": "peakfit", "MinNrPx": "peakfit", "MaxNPeaks": "peakfit",
    "DoFullImage": "peakfit", "LocalMaximaOnly": "peakfit",
    "UseMaximaPositions": "peakfit", "doPeakFit": "peakfit",
    "OverlapLength": "peakfit", "DiscArea": "peakfit", "DiscModel": "peakfit",
    "IntensityThresh": "peakfit", "BadPxIntensity": "peakfit",
    "GapIntensity": "peakfit", "GapFile": "peakfit", "BadPxFile": "peakfit",
    "MaskFile": "peakfit", "Width": "peakfit", "WidthTthPx": "peakfit",
    "UpperBoundThreshold": "peakfit", "DoubletSeparation": "peakfit",
    "ImTransOpt": "peakfit", "Normalize": "peakfit", "SumImages": "peakfit",
    "OmegaSumFrames": "peakfit", "SaveIndividualFrames": "peakfit",
    "UsePixelOverlap": "peakfit", "MinS_N": "peakfit", "OverArea": "peakfit",
    "ReferenceRingCurrent": "peakfit", "zDiffThresh": "peakfit",
    # ── transforms: detector geometry applied to the spots ──
    "tx": "transforms", "ty": "transforms", "tz": "transforms",
    "BC": "transforms", "YCen": "transforms", "ZCen": "transforms",
    "Wedge": "transforms", "MinEta": "transforms", "EtaMin": "transforms",
    "EtaMax": "transforms", "OmegaStart": "transforms",
    "OmegaFirstFile": "transforms", "OmegaStep": "transforms",
    "Parallax": "transforms", "ResidualCorrectionMap": "transforms",
    "PanelShiftsFile": "transforms", "DoFit": "transforms",
    "tolTilts": "transforms", "tolBC": "transforms", "tolLsd": "transforms",
    "GlobalPosition": "transforms", "BeamStopY": "transforms",
    "BeamStopZ": "transforms",
    # ── indexing: search envelope and matching windows ──
    "Rsample": "indexing", "Hbeam": "indexing", "Vsample": "indexing",
    "MinNrSpots": "indexing", "Completeness": "indexing",
    "MinMatchesToAcceptFrac": "indexing", "MarginRadius": "indexing",
    "MarginRadial": "indexing", "MarginEta": "indexing",
    "MarginOme": "indexing", "MargABG": "indexing", "MargABC": "indexing",
    "StepSizePos": "indexing", "StepSizeOrient": "indexing",
    "OmeBinSize": "indexing", "EtaBinSize": "indexing",
    "BeamThickness": "indexing", "UseFriedelPairs": "indexing",
    "MaxOmeSpotIDsToIndex": "indexing", "MinOmeSpotIDsToIndex": "indexing",
    "OmegaRange": "indexing", "OmegaRanges": "indexing",
    "BoxSize": "indexing", "BoxSizes": "indexing",
    # ── refinement / grain assembly ──
    "Twins": "refinement", "nIterations": "refinement",
    "WeightMask": "refinement", "WeightFitRMSE": "refinement",
    "L2Objective": "refinement", "OutlierIterations": "refinement",
}

#: Distortion coefficients all enter at transforms.
for _i in range(15):
    KEY_EARLIEST_STAGE.setdefault(f"p{_i}", "transforms")
for _n in ("iso_R2", "iso_R4", "iso_R6"):
    KEY_EARLIEST_STAGE.setdefault(_n, "transforms")
for _i in range(1, 7):
    KEY_EARLIEST_STAGE.setdefault(f"a{_i}", "transforms")
    KEY_EARLIEST_STAGE.setdefault(f"phi{_i}", "transforms")

_DEFAULT_STAGE = "hkl"


def earliest_stage(keys: Sequence[str]) -> Optional[str]:
    """The earliest pipeline stage any of ``keys`` invalidates."""
    if not keys:
        return None
    idx = min(
        STAGE_ORDER.index(KEY_EARLIEST_STAGE.get(k, _DEFAULT_STAGE))
        for k in keys
    )
    return STAGE_ORDER[idx]


def stale_outputs(keys: Sequence[str], work_dir: Path) -> List[Tuple[str, str, Path]]:
    """Existing outputs that a refresh of ``keys`` would leave stale.

    Returns ``(key, stage, path)`` triples. Only the *earliest* affected stage
    is used per key, since every later stage is invalidated transitively.
    """
    found: List[Tuple[str, str, Path]] = []
    for key in keys:
        stage = KEY_EARLIEST_STAGE.get(key, _DEFAULT_STAGE)
        start = STAGE_ORDER.index(stage)
        for later in STAGE_ORDER[start:]:
            for rel in STAGE_SENTINELS.get(later, ()):
                p = work_dir / rel
                if p.exists():
                    found.append((key, later, p))
    return found


def refresh_zip_params(
    *,
    zip_path: Path,
    param_file: str,
    work_dir: Path,
    force: bool = False,
    label: str = "zip_convert",
) -> Optional[object]:
    """Refresh one archive's analysis parameters; return the report (or None).

    Raises ``RuntimeError`` when a changed parameter invalidates a stage output
    that already exists, unless ``force``. Baked-in changes raise out of
    ``midas_zipper.param_refresh`` with their own message.
    """
    try:
        from midas_zipper.param_refresh import (diff_analysis_params,
                                                refresh_analysis_params)
    except ImportError:                                  # pragma: no cover
        LOG.warning("%s: midas_zipper.param_refresh unavailable; "
                    "parameters NOT refreshed from %s", label, param_file)
        return None

    refreshable, _baked, warnings = diff_analysis_params(zip_path, param_file)
    for w in warnings:
        LOG.warning("%s: %s", label, w)
    if not refreshable:
        LOG.info("%s: analysis parameters already match %s", label,
                 Path(param_file).name)
        # Still call through so a changed baked-in key is reported.
        return refresh_analysis_params(zip_path, param_file)

    changed_keys = [c.source_key for c in refreshable]
    stale = stale_outputs(changed_keys, work_dir)
    if stale and not force:
        by_key: Dict[str, List[Tuple[str, Path]]] = {}
        for key, stage, path in stale:
            by_key.setdefault(key, []).append((stage, path))
        listing = "\n".join(
            f"    {key} -> invalidates {items[0][0]} and later; "
            f"found {len(items)} existing output(s), e.g. {items[0][1]}"
            for key, items in by_key.items()
        )
        raise RuntimeError(
            f"{label}: {len(by_key)} parameter(s) changed in "
            f"{Path(param_file).name}, but stage outputs built with the OLD "
            f"values are still on disk:\n{listing}\n"
            "  Those stages skip when their output exists, so refreshing the "
            "zarr alone would leave the run half on the old parameters -- the "
            "same silent staleness this refresh exists to prevent.\n"
            "  Delete the affected outputs (or the whole result folder) and "
            "re-run, or pass --force-param-refresh to refresh anyway and "
            "accept that those outputs are stale."
        )

    report = refresh_analysis_params(zip_path, param_file)
    if report.applied:
        LOG.info("%s: %s", label, report.summary())
        if stale:
            LOG.warning("%s: --force-param-refresh given; %d existing output(s) "
                        "were built with the old values and are now stale",
                        label, len(stale))
    return report
