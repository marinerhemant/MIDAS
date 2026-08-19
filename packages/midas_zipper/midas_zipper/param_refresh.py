"""Refresh a built ``.MIDAS.zip``'s analysis parameters from ``Parameters.txt``.

``zip_convert`` reuses any existing ``*.MIDAS.zip`` unconditionally, and every
downstream stage reads its geometry from the **zarr**, not from
``Parameters.txt``. So editing ``tx`` / ``Lsd`` / ``BC`` and re-running into the
same result folder used to keep the *old* value silently, while the run reported
success. That reads exactly like "changing ``tx`` does nothing", and it has sent
people hunting through the refiner's C source for a ``tx`` that was never
supposed to be there — the roll is applied in ``transforms``
(``midas_transforms.fit_setup``), upstream of every indexing and refinement
backend.

This module closes that. It reuses the *same* coercion the create path uses
(:func:`coerce_analysis_params`, called by ``ff_zip.write_analysis_parameters``),
so the two cannot drift, and replaces entries in place with ``zip -u`` — the
mechanism ``update_zarr`` already established for this archive format.

Two things it deliberately refuses to do:

* **Baked-in keys are never patched.** ``SkipFrame`` and friends were *consumed*
  when the frames were written — ``ff_zip`` drops those frames from
  ``exchange/data`` — so rewriting the metadata afterwards would leave the data
  and the parameters disagreeing, which shifts every omega by one step with
  nothing downstream able to detect it. A changed baked-in key raises
  :class:`BakedInParamChanged`; the fix is to rebuild the zip, not to patch it.
* **A no-op ``zip -u`` is an error, not a success.** ``zip -u`` replaces an entry
  only when the staged file is *newer* than the one in the archive, and exits 12
  ("nothing to do") otherwise, having changed nothing. Measured: a refresh
  issued in the same second as the build silently kept the old value. Every
  refresh here forces the staged mtimes past the archive's own and then
  re-reads the archive to confirm, so a silent no-op becomes a hard error.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "ANALYSIS_PATH",
    "MEASUREMENT_PATH",
    "BAKED_IN_KEYS",
    "CoercedParam",
    "ParamChange",
    "RefreshReport",
    "BakedInParamChanged",
    "ParamRefreshError",
    "coerce_analysis_params",
    "diff_analysis_params",
    "refresh_analysis_params",
]

# ── zarr layout ──────────────────────────────────────────────────────────────
ANALYSIS_PATH = "analysis/process/analysis_parameters"
MEASUREMENT_PATH = "measurement/process/scan_parameters"

#: Config keys that land in the *measurement* group rather than the analysis one.
MEASUREMENT_KEYS = frozenset({"OmegaStep", "start", "datatype", "doPeakFit"})

# ── type coercion, shared with the create path ───────────────────────────────
# These sets lived inside ``ff_zip.write_analysis_parameters``. They are module
# scope here so the refresh path coerces a value *identically* to the way the
# create path wrote it -- otherwise a refresh would report spurious differences
# (int32 1 vs float64 1.0) on keys nobody touched.
FORCE_DOUBLE_PARAMS = {
    "MinPeakSNR", "RMin", "RMax", "px", "PixelSize", "Completeness",
    "MinMatchesToAcceptFrac", "OverArea", "IntensityThresh", "MinS_N",
    "YPixelSize", "ZPixelSize", "BeamStopY", "BeamStopZ", "DetDist", "MaxDev",
    "OmegaStart", "OmegaFirstFile", "OmegaStep", "step", "BadPxIntensity",
    "GapIntensity", "FitWeightMean", "PixelSplittingRBin", "tolTilts", "tolBC",
    "tolLsd", "DiscArea", "OverlapLength", "ReferenceRingCurrent",
    "zDiffThresh", "GlobalPosition", "tolPanelFit", "tolP", "tolP0", "tolP1",
    "tolP2", "tolP3", "tolP4", "tolShifts", "tolRotation", "tolLsdPanel",
    "tolP2Panel", "DoubletSeparation", "MultFactor", "StepSizePos", "tInt",
    "tGap", "StepSizeOrient", "MarginRadius", "MarginRadial", "MarginEta",
    "MarginOme", "MargABG", "MargABC", "OmeBinSize", "EtaBinSize", "RBinSize",
    "EtaMin", "MinEta", "EtaMax", "X", "Y", "Z", "U", "V", "W", "SHpL",
    "Polariz", "MaxOmeSpotIDsToIndex", "MinOmeSpotIDsToIndex", "BeamThickness",
    "Wedge", "Rsample", "Hbeam", "Vsample", "RhoD", "MaxRingRad", "Lsd",
    "Wavelength", "Width", "WidthTthPx", "UpperBoundThreshold", "p4", "p5",
    "Parallax", "p3", "p2", "p1", "p0", "p6", "p7", "p8", "p9", "p10", "p11",
    "p12", "p13", "p14", "tz", "ty", "tx", "tolP5", "tolP6", "tolP7", "tolP8",
    "tolP9", "tolP10", "tolP11", "tolP12", "tolP13", "tolP14", "tolParallax",
    "WeightMask", "WeightFitRMSE", "QBinSize", "QMin", "QMax",
    # v2 distortion harmonics (carried natively by calibrate-v2; peakfit +
    # transforms read these instead of p0..p14 when present).
    "iso_R2", "iso_R4", "iso_R6", "a1", "phi1", "a2", "phi2", "a3", "phi3",
    "a4", "phi4", "a5", "phi5", "a6", "phi6",
}
FORCE_INT_PARAMS = {
    "Twins", "MaxNFrames", "DoFit", "DiscModel", "UseMaximaPositions",
    "UsePixelOverlap", "MaxNrPx", "MinNrPx", "MaxNPeaks", "PhaseNr",
    "NumPhases", "MinNrSpots", "UseFriedelPairs", "OverallRingToIndex",
    "SpaceGroup", "LayerNr", "DoFullImage", "SkipFrame", "SumImages",
    "Normalize", "SaveIndividualFrames", "OmegaSumFrames", "NrFilesPerSweep",
    "NPanelsY", "NPanelsZ", "Padding", "PanelSizeY", "PanelSizeZ",
    "PanelGapsY", "PanelGapsZ", "doPeakFit", "nIterations",
    "NormalizeRingWeights", "OutlierIterations", "WeightByRadius",
    "WeightByFitSNR", "L2Objective", "PerPanelLsd", "PerPanelDistortion",
    "FixPanelID", "MinIndicesForFit", "LocalMaximaOnly", "FitParallax",
    # Opt-in local background subtraction in the peak search
    # (midas_peakfit.background). BgSubtract 0 = legacy/C behaviour.
    "BgSubtract", "BgNSectors",
}
FORCE_STRING_PARAMS = {
    "GapFile", "BadPxFile", "ResultFolder", "PanelShiftsFile", "MaskFile",
    "GrainsFile", "ResidualCorrectionMap",
}
RENAME_MAP = {
    "OmegaStep": "step", "Completeness": "MinMatchesToAcceptFrac",
    "px": "PixelSize", "LatticeConstant": "LatticeParameter",
    "OverAllRingToIndex": "OverallRingToIndex", "resultFolder": "ResultFolder",
    "OmegaRange": "OmegaRanges", "BoxSize": "BoxSizes",
}

#: Keys whose value the **conversion itself** consumed, so the frames already
#: stored in ``exchange/data`` depend on them. Patching these in an existing
#: archive would leave the data and the metadata describing different things.
#:
#: ``SkipFrame`` is the sharp one: ``ff_zip`` drops that many leading frames
#: while writing ``exchange/data`` (``ff_zip.py:311``, ``:351``, ``:561``), so a
#: later edit cannot be honoured by rewriting the number -- the frames are gone.
#: The rest select *which raw file* was read, how it was addressed, or its
#: pixel layout.
BAKED_IN_KEYS = frozenset({
    "SkipFrame",
    "NrFilesPerSweep", "numFilesPerScan", "ScanStep",
    "Padding", "StartFileNrFirstLayer", "FileStem", "RawFolder", "Ext",
    "HeadSize", "PixelValue", "BytesPerPixel",
    "NrPixelsY", "NrPixelsZ", "NrPixels", "numPxY", "numPxZ",
    "LayerNr", "dataFN", "darkFN", "DarkFN", "Dark",
    "OrigFileName", "dataLoc", "darkLoc",
})


class ParamRefreshError(RuntimeError):
    """A refresh was attempted and could not be completed safely."""


class BakedInParamChanged(ParamRefreshError):
    """A parameter the stored frames depend on differs from the archive.

    Carries :attr:`keys` so a caller can name them without re-parsing the text.
    """

    def __init__(self, message: str, keys: Sequence[str]):
        super().__init__(message)
        self.keys = list(keys)


@dataclass(frozen=True)
class CoercedParam:
    """One ``Parameters.txt`` entry, typed and placed as the zarr stores it."""

    source_key: str
    path: str
    key: str
    value: np.ndarray

    @property
    def full_key(self) -> str:
        return f"{self.path}/{self.key}"


@dataclass(frozen=True)
class ParamChange:
    """A single key whose archived value differs from the parameter file."""

    source_key: str
    full_key: str
    old: Optional[np.ndarray]
    new: np.ndarray

    def describe(self) -> str:
        old = "<absent>" if self.old is None else _fmt(self.old)
        return f"{self.source_key} ({self.key_tail}): {old} -> {_fmt(self.new)}"

    @property
    def key_tail(self) -> str:
        return self.full_key.rsplit("/", 1)[-1]


@dataclass
class RefreshReport:
    """What a refresh did, for the log and for stage provenance."""

    zip_path: str
    applied: List[ParamChange] = field(default_factory=list)
    baked_in: List[ParamChange] = field(default_factory=list)
    dry_run: bool = False

    @property
    def changed_keys(self) -> List[str]:
        return [c.source_key for c in self.applied]

    def to_metrics(self) -> Dict[str, Any]:
        return {
            "n_params_refreshed": len(self.applied),
            "params_refreshed": ",".join(sorted(self.changed_keys)),
        }

    def summary(self) -> str:
        if not self.applied:
            return "analysis parameters already match the parameter file"
        verb = "would refresh" if self.dry_run else "refreshed"
        return f"{verb} {len(self.applied)} parameter(s): " + "; ".join(
            c.describe() for c in self.applied
        )


def _fmt(arr: np.ndarray) -> str:
    """Compact one-line rendering of a parameter value for the log."""
    a = np.asarray(arr)
    if a.dtype.kind == "S":
        return repr(b"".join(a.flatten().tolist()).decode("utf-8", "replace"))
    flat = a.flatten()
    if flat.size == 1:
        return f"{flat[0]:g}" if a.dtype.kind in "fiu" else str(flat[0])
    if flat.size > 8:
        head = ", ".join(f"{v:g}" for v in flat[:8])
        return f"[{head}, ... {flat.size} values]"
    return "[" + ", ".join(f"{v:g}" for v in flat) + "]"


# ── coercion ─────────────────────────────────────────────────────────────────
def _coerce_one(key: str, value: Any) -> List[CoercedParam]:
    """Type one config entry exactly as ``write_analysis_parameters`` would."""
    target_key = RENAME_MAP.get(key, key)
    path = MEASUREMENT_PATH if key in MEASUREMENT_KEYS else ANALYSIS_PATH

    if key == "BC":
        # BC is the one key that fans out into two datasets.
        return [
            CoercedParam(key, ANALYSIS_PATH, "YCen",
                         np.array([value[0]], dtype=np.double)),
            CoercedParam(key, ANALYSIS_PATH, "ZCen",
                         np.array([value[1]], dtype=np.double)),
        ]
    if key in ("LatticeConstant", "LatticeParameter"):
        values = value if isinstance(value, list) else [value]
        padded = np.zeros(6, dtype=np.double)
        padded[:len(values)] = values
        return [CoercedParam(key, path, target_key, padded)]
    if key == "ImTransOpt":
        values = value if isinstance(value, list) else [value]
        return [CoercedParam(key, path, target_key,
                             np.array(values, dtype=np.int32).flatten())]
    if key in ("RingThresh", "RingsToExclude", "OmegaRange", "BoxSize"):
        temp = value if isinstance(value, list) else [value]
        rows = temp if (temp and isinstance(temp[0], list)) else [temp]
        arr = np.array(rows)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        # NB: the create path computes ``np.int32 if key == 'ImTransOpt'`` here,
        # which is unreachable (ImTransOpt is handled above) -- these four are
        # always double. Kept explicit so the two paths agree.
        return [CoercedParam(key, path, target_key, arr.astype(np.double))]

    if key in FORCE_STRING_PARAMS or target_key in FORCE_STRING_PARAMS:
        arr = np.array([np.bytes_(str(value).encode("UTF-8"))])
    elif key in FORCE_DOUBLE_PARAMS or target_key in FORCE_DOUBLE_PARAMS:
        arr = np.array(value if isinstance(value, list) else [value],
                       dtype=np.double)
    elif key in FORCE_INT_PARAMS or target_key in FORCE_INT_PARAMS:
        arr = np.array(value if isinstance(value, list) else [value],
                       dtype=np.int32)
    elif isinstance(value, str):
        arr = np.array([np.bytes_(value.encode("UTF-8"))])
    else:
        arr = np.array(value if isinstance(value, list) else [value])
    return [CoercedParam(key, path, target_key, arr)]


def coerce_analysis_params(config: Dict[str, Any]) -> Tuple[List[CoercedParam],
                                                            List[str]]:
    """Type every config entry the way the zarr stores it.

    Returns ``(params, warnings)``. A key that cannot be coerced becomes a
    warning string rather than an exception, matching the create path's
    per-key ``try/except`` -- one malformed entry must not lose the rest.
    """
    out: List[CoercedParam] = []
    warnings: List[str] = []
    for key, value in config.items():
        try:
            out.extend(_coerce_one(key, value))
        except Exception as e:                      # noqa: BLE001 - see docstring
            warnings.append(f"Could not write parameter '{key}'. Reason: {e}")
    return out, warnings


# ── diffing ──────────────────────────────────────────────────────────────────
def _read_stored(root, full_key: str) -> Optional[np.ndarray]:
    try:
        return np.asarray(root[full_key][...])
    except Exception:                               # noqa: BLE001 - absent key
        return None


def _differs(old: Optional[np.ndarray], new: np.ndarray) -> bool:
    if old is None:
        return True
    if old.shape != new.shape:
        return True
    if old.dtype.kind == "S" or new.dtype.kind == "S":
        return [bytes(x) for x in old.flatten()] != [bytes(x) for x in new.flatten()]
    # Exact comparison on purpose: the question is "did the user edit this",
    # not "are these close". A tolerance here would swallow a real 1e-9 edit
    # and, worse, make the refresh non-idempotent in the other direction.
    return not np.array_equal(old, new)


def diff_analysis_params(
    zip_path: os.PathLike | str,
    param_file: os.PathLike | str,
) -> Tuple[List[ParamChange], List[ParamChange], List[str]]:
    """Compare an archive's stored parameters against ``Parameters.txt``.

    Returns ``(refreshable, baked_in, warnings)``. Neither list is applied;
    this is the read-only half and is what ``--dry-run`` reports.
    """
    import zarr
    from .ff_zip import parse_parameter_file

    config = parse_parameter_file(str(param_file))
    params, warnings = coerce_analysis_params(config)

    root = zarr.open(str(zip_path), "r")
    refreshable: List[ParamChange] = []
    baked_in: List[ParamChange] = []
    for p in params:
        old = _read_stored(root, p.full_key)
        if not _differs(old, p.value):
            continue
        change = ParamChange(p.source_key, p.full_key, old, p.value)
        if p.source_key in BAKED_IN_KEYS:
            baked_in.append(change)
        else:
            refreshable.append(change)
    return refreshable, baked_in, warnings


# ── applying ─────────────────────────────────────────────────────────────────
def _chunk_entries(stage_dir: Path, full_key: str) -> List[str]:
    """Archive-relative paths zarr wrote for one staged dataset."""
    base = stage_dir / full_key
    return sorted(
        str(p.relative_to(stage_dir))
        for p in base.rglob("*") if p.is_file()
    )


def _archive_mtime_floor(zip_path: Path, entries: Sequence[str]) -> float:
    """Newest stored timestamp among the entries we are about to replace.

    ``zip -u`` compares the on-disk file against the *entry's* stored date, not
    against the archive file's mtime, so this is the number the staged files
    have to beat.
    """
    floor = 0.0
    try:
        with zipfile.ZipFile(zip_path) as zf:
            stored = {i.filename: i.date_time for i in zf.infolist()}
    except (OSError, zipfile.BadZipFile):
        return time.time()
    for name in entries:
        dt = stored.get(name)
        if dt is None:
            continue
        try:
            floor = max(floor, time.mktime((*dt, 0, 0, -1)))
        except (ValueError, OverflowError):
            continue
    return floor


def refresh_analysis_params(
    zip_path: os.PathLike | str,
    param_file: os.PathLike | str,
    *,
    dry_run: bool = False,
    allow_baked_in: bool = False,
) -> RefreshReport:
    """Rewrite an existing ``.MIDAS.zip``'s analysis parameters from the params file.

    Raises :class:`BakedInParamChanged` when a key the stored frames depend on
    has changed (unless ``allow_baked_in``), and :class:`ParamRefreshError` if
    the rewrite did not take. Returns a :class:`RefreshReport` describing what
    changed; an empty ``applied`` list means the archive already agreed and no
    ``zip`` call was made.
    """
    import zarr

    zip_path = Path(zip_path).resolve()
    refreshable, baked_in, warnings = diff_analysis_params(zip_path, param_file)

    if baked_in and not allow_baked_in:
        listing = "\n".join(f"    {c.describe()}" for c in baked_in)
        raise BakedInParamChanged(
            f"{zip_path.name}: {len(baked_in)} parameter(s) that the stored "
            f"frames depend on differ from {Path(param_file).name}:\n{listing}\n"
            "  These were consumed when the frames were written -- SkipFrame "
            "drops leading frames from exchange/data, and the file-addressing "
            "keys select which raw file was read -- so patching the metadata "
            "would leave the data and the parameters describing different "
            "things (a silently shifted omega, in SkipFrame's case).\n"
            "  Rebuild instead: delete this .MIDAS.zip (or the whole result "
            "folder) and re-run so the conversion re-reads the raw data.",
            keys=[c.source_key for c in baked_in],
        )

    # With allow_baked_in the caller has taken responsibility for the frames
    # no longer matching; apply those too rather than silently reporting them.
    to_apply = list(refreshable) + (list(baked_in) if allow_baked_in else [])
    report = RefreshReport(zip_path=str(zip_path), applied=to_apply,
                           baked_in=list(baked_in), dry_run=dry_run)
    if dry_run or not to_apply:
        return report

    if shutil.which("zip") is None:
        raise ParamRefreshError(
            "the 'zip' executable is required to rewrite parameters inside an "
            "existing .MIDAS.zip and was not found on PATH. Install Info-ZIP, "
            "or delete the archive so it is rebuilt from the parameter file."
        )

    with tempfile.TemporaryDirectory(prefix="midas_param_refresh_") as tmp:
        stage_dir = Path(tmp) / "stage"
        staged = zarr.open(str(stage_dir), "w")
        for change in to_apply:
            value = change.new
            # write_empty_chunks: an all-zero value equals zarr's fill value, and
            # without this zarr omits the chunk file entirely -- `zip -u` would
            # then refresh .zarray while the OLD chunk stayed in the archive,
            # i.e. a parameter that reads back unchanged. Setting a value to 0
            # is exactly what someone does to disable a correction.
            ds = staged.create_dataset(
                change.full_key, shape=value.shape, dtype=value.dtype,
                chunks=value.shape or (1,), write_empty_chunks=True,
            )
            ds[...] = value

        entries: List[str] = []
        for change in to_apply:
            entries.extend(_chunk_entries(stage_dir, change.full_key))
        if not entries:
            raise ParamRefreshError(
                "staging produced no files to update -- refusing to report a "
                "refresh that cannot have happened."
            )

        # Beat the archive's own timestamps (see module docstring): `zip -u`
        # exits 12 and changes nothing when the staged file is not newer.
        stamp = max(time.time(), _archive_mtime_floor(zip_path, entries) + 2.0)
        for rel in entries:
            os.utime(stage_dir / rel, (stamp, stamp))

        # ONE batched call: `zip -u` rewrites the archive, which is O(archive
        # size). Per-key calls would multiply that by the number of parameters.
        proc = subprocess.run(
            ["zip", "-u", str(zip_path), *entries],
            cwd=str(stage_dir), capture_output=True, text=True, check=False,
        )
        # rc 12 == "nothing to do". Both it and a hard failure mean the archive
        # still holds the old values, so neither may be reported as success.
        if proc.returncode != 0:
            raise ParamRefreshError(
                f"zip -u failed on {zip_path} (exit {proc.returncode}"
                f"{'; nothing to do' if proc.returncode == 12 else ''}): "
                f"{(proc.stderr or proc.stdout or '').strip()[:400]}"
            )

    _verify(zip_path, to_apply)
    return report


def _verify(zip_path: Path, changes: Sequence[ParamChange]) -> None:
    """Re-read the archive and confirm every value actually took.

    Rule 11 of the FF handbook: suspect success. ``zip -u`` has a silent no-op
    mode, and a parameter that reads back stale is the exact failure this module
    exists to prevent -- so it is checked, not assumed.
    """
    import zarr

    root = zarr.open(str(zip_path), "r")
    stale = []
    for change in changes:
        got = _read_stored(root, change.full_key)
        if got is None or _differs(got, change.new):
            stale.append(
                f"    {change.source_key}: wanted {_fmt(change.new)}, "
                f"archive still reads {'<absent>' if got is None else _fmt(got)}"
            )
    if stale:
        raise ParamRefreshError(
            f"{zip_path.name}: parameter refresh did not take for "
            f"{len(stale)} key(s):\n" + "\n".join(stale)
        )


# ── CLI ──────────────────────────────────────────────────────────────────────
def main(argv: Optional[Sequence[str]] = None) -> int:
    """``midas-refresh-zarr-params`` — fix a stale archive outside a pipeline run."""
    import argparse

    ap = argparse.ArgumentParser(
        prog="midas-refresh-zarr-params",
        description=(
            "Re-read a parameter file into an existing .MIDAS.zip. Downstream "
            "MIDAS stages read geometry from the zarr, not from Parameters.txt, "
            "so an archive built before you edited tx/Lsd/BC still carries the "
            "old values."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("-fn", "--zip", dest="zip_path", required=True,
                    help="the .MIDAS.zip to update")
    ap.add_argument("-paramFN", "--params", dest="param_file", required=True,
                    help="parameter file to read the new values from")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would change; write nothing")
    ap.add_argument("--allow-baked-in", action="store_true",
                    help=("also patch parameters the stored frames depend on "
                          "(SkipFrame, file addressing, pixel layout). The "
                          "frames are NOT re-read, so the archive will "
                          "describe data it does not contain -- rebuild "
                          "instead unless you know exactly why you want this"))
    args = ap.parse_args(argv)

    try:
        report = refresh_analysis_params(
            args.zip_path, args.param_file,
            dry_run=args.dry_run, allow_baked_in=args.allow_baked_in,
        )
    except BakedInParamChanged as e:
        print(f"error: {e}", file=__import__("sys").stderr)
        return 2
    except ParamRefreshError as e:
        print(f"error: {e}", file=__import__("sys").stderr)
        return 1

    print(report.summary())
    if report.baked_in:
        print(f"warning: {len(report.baked_in)} baked-in parameter(s) patched "
              "without re-reading the frames:")
        for c in report.baked_in:
            print(f"    {c.describe()}")
    return 0


if __name__ == "__main__":                              # pragma: no cover
    raise SystemExit(main())
