"""Writing results, with their provenance attached.

The legacy pipeline wrote bare ``.bin`` files whose shape lived only in the
filename, and whose channel labels were wrong from index 5 on. Everything here
writes the array together with what it is, how it was made, and what is not
yet corrected -- so a map cannot be separated from its caveats by copying a
file.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .branches import BranchResult

__all__ = ["write_result", "write_maps_hdf5", "read_legacy_reconstruction"]

log = logging.getLogger(__name__)


def _provenance(result: BranchResult, extra: dict | None = None) -> dict:
    from . import __version__

    prov: dict[str, Any] = {
        "midas_dt_version": __version__,
        "written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "branch": result.branch,
        "channel": {
            "label": result.channel.label,
            "r_min": result.channel.r_min, "r_max": result.channel.r_max,
            "eta_min": result.channel.eta_min, "eta_max": result.channel.eta_max,
            "r_bin": result.channel.r_bin, "eta_bin": result.channel.eta_bin,
            "n_peaks": result.channel.n_peaks,
        },
        "linearity": dict(result.linearity),
        "approximate_outputs": result.approximate_outputs(),
        "known_limits": result.limits.warnings(),
        "snake_corrected": result.limits.snake_corrected,
        "omega_negated": result.limits.omega_negated,
    }
    if extra:
        prov.update(extra)
    return prov


def write_result(result: BranchResult, directory: str | Path, *,
                 extra: dict | None = None) -> Path:
    """Write every map as ``.npy`` plus a ``provenance.json``.

    Chosen over a bare ``.bin`` because ``.npy`` carries its own shape and
    dtype: the legacy format encoded those in the filename, so renaming or
    copying a file lost them.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name, arr in result.maps.items():
        np.save(directory / f"{name}.npy", np.asarray(arr, dtype=np.float32))
    prov = _provenance(result, extra)
    (directory / "provenance.json").write_text(json.dumps(prov, indent=2) + "\n")
    log.info("wrote %d maps + provenance to %s", len(result.maps), directory)
    return directory


def write_maps_hdf5(result: BranchResult, path: str | Path, *,
                    extra: dict | None = None) -> Path:
    """Write all maps into one HDF5 file, provenance as attributes.

    Each dataset carries a ``linearity`` attribute, so a map that rests on the
    weighted-moment approximation says so wherever it is opened.
    """
    try:
        import h5py
    except ImportError as exc:
        raise ImportError(
            "HDF5 output needs h5py. Install with `pip install h5py`, or use "
            "write_result() for .npy output."
        ) from exc

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    prov = _provenance(result, extra)
    with h5py.File(path, "w") as hf:
        g = hf.create_group("maps")
        for name, arr in result.maps.items():
            ds = g.create_dataset(name, data=np.asarray(arr, dtype=np.float32),
                                  compression="gzip")
            ds.attrs["linearity"] = result.linearity.get(name, "unknown")
        hf.attrs["provenance_json"] = json.dumps(prov)
        for k in ("branch", "snake_corrected", "omega_negated"):
            hf.attrs[k] = prov[k]
        hf.attrs["known_limits"] = "\n".join(prov["known_limits"])
    log.info("wrote %s", path)
    return path


def read_legacy_reconstruction(path: str | Path, size: int, *,
                               n_eta: int = 1, n_rad: int = 1,
                               n_outputs: int = 12) -> np.ndarray:
    """Read a 2023-era ``PeakFitResult.bin``.

    Shape ``(size, size, n_eta, n_rad, n_outputs)``, float64, as written by
    ``PeakFit``.

    **Index the output axis; do not trust filenames.** The scripts that wrote
    these omitted ``MaxIntensityObs`` from slot 5, so every label from index 5
    on is shifted by one -- a file named ``*_BGFit_*`` holds
    ``MaxIntensityObs``. Use
    :data:`~midas_dt.conventions.FIT_OUTPUT_NAMES` for the true meaning of
    each index.
    """
    path = Path(path)
    count = size * size * n_eta * n_rad * n_outputs
    actual = path.stat().st_size // 8
    if actual != count:
        raise ValueError(
            f"{path.name} holds {actual} float64 values but the given shape "
            f"implies {count} ({size}x{size}x{n_eta}x{n_rad}x{n_outputs}). "
            f"Check size / n_eta / n_rad against the run's parameter file."
        )
    arr = np.fromfile(path, dtype=np.float64, count=count)
    return arr.reshape(size, size, n_eta, n_rad, n_outputs)
