"""Orchestrator: run MIDASCalibrant on an image, parse its output.

MVP scope (v0.1.0): one invocation of the binary, one set of input parameters,
one refined geometry out. The multi-stage orchestration that
``utils/AutoCalibrateZarr.py`` performs (progressive distortion unlocking,
ring outlier rejection, panel refinement) sits above this primitive and
lands later.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ._binaries import midas_bin
from ._config import CalibrationConfig, write_params_file
from .geometry import DetectorGeometry

# Regex for the "key value" lines that CalibrantIntegratorOMP emits after
# the "Mean Values" marker. We match the whole line and strip the key.
_GEOMETRY_KEYS = (
    "Lsd", "ty", "tz", "RhoD",
    *(f"p{i}" for i in range(15)),
)
_LINE = re.compile(r"^(?P<k>[A-Za-z_][\w]*)\s+(?P<vals>.+)$")


@dataclass
class CalibrationResult:
    """Output of a calibration run.

    Attributes
    ----------
    geometry : DetectorGeometry
        Refined geometry + distortion.
    pseudo_strain : float
        Microstrain (ppm) after the last iteration — lower = better fit.
        Mirrors ``DetectorGeometry.mean_strain`` for convenience.
    pseudo_strain_std : float
        Standard deviation of per-ring residual strain (ppm).
    convergence_history : list[dict] | None
        Parsed ``<rawFN>.convergence_history.csv`` as a list of row dicts.
        ``None`` if the file was not produced (single-iteration runs).
    corr_csv_path : Path | None
        Path to ``<rawFN>.corr.csv`` if produced.
    stdout : str
        Full captured stdout from MIDASCalibrant — retained for debugging.
    work_dir : Path
        Directory where Parameters.txt and outputs live.
    """

    geometry: DetectorGeometry
    pseudo_strain: float
    pseudo_strain_std: float
    convergence_history: Optional[list[dict]] = None
    corr_csv_path: Optional[Path] = None
    stdout: str = ""
    work_dir: Path = field(default_factory=Path)


def run_calibration(
    config: CalibrationConfig,
    data_file: str | Path,
    *,
    work_dir: str | Path | None = None,
    n_cpus: int = 8,
    bin_dir: str | Path | None = None,
    n_iterations: int | None = None,
    check: bool = True,
) -> CalibrationResult:
    """Invoke ``MIDASCalibrant`` on ``data_file`` using ``config``.

    Parameters
    ----------
    config : CalibrationConfig
        User-facing inputs (material, wavelength, starting geometry, …).
    data_file : path-like
        Path to a single frame, e.g. ``CeO2_00001.h5`` / ``.tif`` / ``.ge3``.
        ``FileStem``, ``StartNr``, ``Padding`` and ``Ext`` are derived from
        the name so ``MIDASCalibrant`` writes its outputs next to it.
    work_dir : path-like, optional
        Directory to place Parameters.txt + outputs in. Defaults to the
        parent directory of ``data_file``.
    n_cpus : int, default 8
        Passed to ``MIDASCalibrant`` as its second positional argument.
    bin_dir : path-like, optional
        Explicit override for the binary search directory; forwarded to
        :func:`midas_bin`.
    n_iterations : int, optional
        Override ``config.n_iterations`` for this call. Useful for fast
        smoke tests with a single iteration.
    check : bool, default True
        Raise on non-zero exit; when ``False``, the returned result has
        whatever state could be recovered plus ``stdout`` for debugging.
    """
    data_path = Path(data_file).resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"data_file not found: {data_path}")

    work_path = Path(work_dir).resolve() if work_dir is not None else data_path.parent
    work_path.mkdir(parents=True, exist_ok=True)

    # Derive FileStem/StartNr/Ext from the data file name. The C binary
    # assembles ``rawFN = FileStem + _ + zero-padded StartNr + . + Ext``.
    stem, start_nr, padding, ext = _split_numbered_filename(data_path)

    # MIDAS's TIFF-read path in CalibrantIntegratorOMP.c:1038 builds the
    # filename as ``"%s/%s_%0*d%s"`` (no literal dot between number and Ext),
    # so Ext MUST be dot-prefixed for the binary to find the image.
    ext_with_dot = ext if ext.startswith(".") else f".{ext}"
    extra = {
        "Folder": str(work_path),
        "FileStem": stem,
        "StartNr": start_nr,
        "EndNr": start_nr,                    # MVP: single-frame processing
        "Padding": padding,
        "Ext": ext_with_dot,
    }
    # MIDAS_ParamParser.c reads the key as `nIterations` (lowercase n).
    # Using the wrong case silently falls back to the default (1), which
    # cripples convergence.
    if n_iterations is not None:
        extra["nIterations"] = n_iterations
    elif config.n_iterations is not None:
        extra["nIterations"] = config.n_iterations

    params = config.to_params(extra=extra)
    params_path = work_path / "Parameters.txt"
    write_params_file(params_path, params)

    exe = midas_bin("MIDASCalibrant", bin_dir=bin_dir)
    cmd = [str(exe), str(params_path), str(n_cpus)]
    proc = subprocess.run(
        cmd, capture_output=True, text=True, cwd=work_path, check=False,
    )
    stdout = proc.stdout + ("\n" + proc.stderr if proc.stderr else "")

    # Persist stdout so reruns can inspect without re-executing.
    (work_path / "calibrant.stdout").write_text(stdout)

    if check and proc.returncode != 0:
        raise RuntimeError(
            f"MIDASCalibrant exited {proc.returncode}.\n"
            f"Last stderr lines:\n  " +
            "\n  ".join(proc.stderr.strip().splitlines()[-10:])
        )

    geometry = _parse_final_geometry(stdout, config)
    raw_fn = work_path / f"{stem}_{str(start_nr).zfill(padding)}.{ext}"
    conv_hist = _load_convergence_history(Path(f"{raw_fn}.convergence_history.csv"))
    corr_csv = Path(f"{raw_fn}.corr.csv")
    corr_csv_path: Optional[Path] = corr_csv if corr_csv.exists() else None

    return CalibrationResult(
        geometry=geometry,
        pseudo_strain=geometry.mean_strain,
        pseudo_strain_std=geometry.std_strain,
        convergence_history=conv_hist,
        corr_csv_path=corr_csv_path,
        stdout=stdout,
        work_dir=work_path,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FILENAME_RE = re.compile(r"^(?P<stem>.+?)_(?P<num>\d+)\.(?P<ext>[^.]+)$")


def _split_numbered_filename(path: Path) -> tuple[str, int, int, str]:
    """Parse ``CeO2_00001.h5`` → (``CeO2``, 1, 5, ``h5``)."""
    m = _FILENAME_RE.match(path.name)
    if not m:
        raise ValueError(
            f"Cannot parse {path.name!r} as <FileStem>_<NN...>.<ext>; "
            "rename the file or pass FileStem/StartNr/Padding/Ext via config.extra."
        )
    num_str = m.group("num")
    return m.group("stem"), int(num_str), len(num_str), m.group("ext")


def _parse_final_geometry(
    stdout: str, config: CalibrationConfig,
) -> DetectorGeometry:
    """Extract the last ``Mean Values`` block as a ``DetectorGeometry``.

    Defaults (px, wavelength, NrPixelsY/Z, initial BC) come from the config
    so the returned geometry is self-contained even when a key isn't printed.
    """
    geom = DetectorGeometry(
        lsd=config.lsd, ybc=config.ybc, zbc=config.zbc,
        tx=config.tx, ty=config.ty, tz=config.tz,
        px=config.pixel_size,
        wavelength=config.wavelength,
        nr_pixels_y=config.nr_pixels_y,
        nr_pixels_z=config.nr_pixels_z,
    )

    lines = stdout.splitlines()
    try:
        last_marker = max(i for i, line in enumerate(lines) if "Mean Values" in line)
    except ValueError:
        return geom  # binary didn't print one — return config-defaults

    for line in lines[last_marker + 1:]:
        line = line.strip()
        if not line or line.startswith("Copy to par"):
            continue
        m = _LINE.match(line)
        if not m:
            continue
        key, vals = m.group("k"), m.group("vals").split()
        if key == "BC" and len(vals) == 2:
            geom.ybc, geom.zbc = float(vals[0]), float(vals[1])
        elif key in ("Lsd", "tx", "ty", "tz", "RhoD") and vals:
            attr = "rhod" if key == "RhoD" else key.lower()
            setattr(geom, attr, float(vals[0]))
        elif re.fullmatch(r"p\d{1,2}", key) and vals:
            idx = int(key[1:])
            if 0 <= idx <= 14:
                setattr(geom, f"p{idx}", float(vals[0]))
        elif key == "MeanStrain" and vals:
            # Binary emits ppm; convert to dimensionless for DetectorGeometry
            # storage (matches AutoCalibrateZarr: state.mean_strain = val / 1e6).
            geom.mean_strain = float(vals[0])
        elif key == "StdStrain" and vals:
            geom.std_strain = float(vals[0])
    return geom


def _load_convergence_history(path: Path) -> Optional[list[dict]]:
    """Parse the CSV written by CalibrantIntegratorOMP into row dicts.

    Falls back to stdlib ``csv`` so this stays pandas-optional at runtime.
    """
    if not path.exists():
        return None
    import csv
    with path.open() as f:
        reader = csv.DictReader(f)
        rows: list[dict] = []
        for row in reader:
            typed: dict[str, float | int | str] = {}
            for k, v in row.items():
                if v is None or v == "":
                    typed[k] = v
                    continue
                try:
                    typed[k] = float(v)
                except ValueError:
                    typed[k] = v
            rows.append(typed)
    return rows
