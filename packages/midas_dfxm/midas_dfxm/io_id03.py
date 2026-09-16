"""Read ESRF ID03 DFXM rocking scans into a :class:`midas_dfxm.rocking.RockingScan`.

One measurement campaign, one master HDF5 file (Bliss/Nexus convention): scan entries named
``"<plane>.1"``, one plane per fixed value of the objective's own rocking angle (``obpitch`` in
this campaign -- an *objective* 2theta, not the sample's, so it plays no part in
:func:`midas_dfxm.rocking.classify_motors`). Each plane holds a full ``mu`` x ``chi`` mesh: mu
(theta-like rock) and chi (roll) both step across the plane's frames, giving a ``tilt2d`` scan
once loaded -- the same scan type :mod:`midas_dfxm.io_6idc` produces for a two-axis 6-ID-C mesh,
so it goes through the identical :func:`midas_dfxm.rocking.reduce_rocking` path.

Traps met on the real Mg-4Al campaign this was built against:

* The image dataset is compressed with a filter only :mod:`hdf5plugin` registers. Opening the
  file without importing it first does not fail at ``h5py.File(...)`` -- it fails later, on the
  first frame read, with an opaque ``OSError``. Imported here before ``h5py``, and the error
  is caught and re-raised with what it means.
* One plane is ``(650, 2048, 2048)`` uint16, ~2.7 GB. Reading it whole before cropping is the
  same trap :mod:`midas_dfxm.io_6idc` refuses for an uncropped 6-ID-C scan: a ROI must be set
  and is applied frame by frame as they are read, never after.
* ``obpitch`` lives at ``positioners_start/obpitch`` in some entries and
  ``instrument/positioners_start/obpitch`` in others (both seen in the same file); both are
  tried before giving up.
* mu and chi are recorded **per frame**, already paired correctly in acquisition order (the
  scan is point-major with a zigzag mu direction across chi rows). No un-zigzagging is needed
  or done here: :class:`RockingScan` only needs each frame matched to its own motor reading,
  which the file already gives directly. Do not "fix" the order against a sorted grid --
  that is a self-consistency check the original reduction ran once, not a loading step.
* A plane with only one moving tilt motor (mu OR chi held fixed to a constant, seen on
  malformed scans) is refused by :func:`~midas_dfxm.rocking.classify_motors` itself, with a
  clear "no rocking angle moves" or "1 tilt motor moves" message -- nothing special done here.
"""
from __future__ import annotations

import os
from typing import Optional, Union

import numpy as np

from .rocking import RockingScan

__all__ = ["load_id03_scan", "list_id03_planes"]

_OBPITCH_PATHS = ("positioners_start/obpitch", "instrument/positioners_start/obpitch")


def _h5py():
    try:
        import hdf5plugin  # noqa: F401  (registers the compression filter before h5py opens)
        import h5py
    except ImportError as e:                                        # pragma: no cover
        raise ImportError("reading ID03 frames needs `h5py` and `hdf5plugin` "
                          "(pip install h5py hdf5plugin)") from e
    return h5py


def list_id03_planes(h5_path: str) -> list:
    """Scan entries in this file, in file order (e.g. ``['1.1', '2.1', ..., '8.1']``)."""
    h5py = _h5py()
    with h5py.File(h5_path, "r") as f:
        return [k for k in f.keys() if k.endswith(".1")]


def _crop(a, roi):
    if roi is None:
        return a
    r0, r1, c0, c1 = roi
    return a[..., r0:r1, c0:c1]


def _read_obpitch(entry) -> Optional[float]:
    for path in _OBPITCH_PATHS:
        if path in entry:
            return float(entry[path][()])
    return None


def load_id03_scan(h5_path: str, plane: Union[int, str] = 1, *, roi=None,
                   dark: Optional[Union[float, np.ndarray]] = None) -> RockingScan:
    """Load one ESRF ID03 rocking-scan plane (fixed objective angle) into a ``RockingScan``.

    Parameters
    ----------
    h5_path : path to the campaign's master HDF5 file.
    plane : plane number (``1`` -> entry ``"1.1"``) or an explicit entry name (``"5.1"``).
    roi : ``(row0, row1, col0, col1)`` crop applied as frames are read. Required in practice:
        a full plane is ~2.7 GB uint16; leaving ``roi=None`` reads all of it into memory.
    dark : ``None``, a number, or a 2-D array subtracted from every frame (cropped to ``roi``
        first). ID03 frames have no per-scan dark folder convention here; pass one only if you
        have a measured one.

    Returns
    -------
    RockingScan
        ``scan_type`` is ``"tilt2d"`` (mu and chi both move). ``scan.meta["obpitch"]`` carries
        the plane's fixed objective angle, read but not treated as a scan axis -- comparing
        planes (a strain/d-spacing measurement) is a different, harder analysis than reducing
        one plane's orientation mesh, and is out of scope here. ``scan.notes`` records what was
        found and any path that had to be guessed.
    """
    h5py = _h5py()
    if not os.path.isfile(h5_path):
        raise FileNotFoundError(h5_path)
    entry_name = f"{int(plane)}.1" if isinstance(plane, (int, np.integer)) else str(plane)
    notes = []
    with h5py.File(h5_path, "r") as f:
        if entry_name not in f:
            found = [k for k in f.keys() if k.endswith(".1")]
            raise KeyError(f"no entry {entry_name!r} in {h5_path}; planes present: {found}")
        entry = f[entry_name]
        if "instrument" not in entry or "pco_ff" not in entry["instrument"] \
                or "image" not in entry["instrument/pco_ff"]:
            raise KeyError(f"{entry_name}/instrument/pco_ff/image not found in {h5_path}")
        img = entry["instrument/pco_ff/image"]
        if "mu" not in entry["instrument"] or "data" not in entry["instrument/mu"]:
            raise KeyError(f"{entry_name}/instrument/mu/data not found in {h5_path}")
        if "chi" not in entry["instrument"] or "value" not in entry["instrument/chi"]:
            raise KeyError(f"{entry_name}/instrument/chi/value not found in {h5_path}")
        mu = np.asarray(entry["instrument/mu/data"][()], dtype=float)
        chi = np.asarray(entry["instrument/chi/value"][()], dtype=float)
        M = img.shape[0]
        if mu.size != M or chi.size != M:
            raise ValueError(f"{entry_name}: {M} frames but mu has {mu.size}, chi has "
                             f"{chi.size} values. Refusing to guess the frame/angle pairing.")
        try:
            frames = np.empty((M,) + _crop(np.empty(img.shape[1:], img.dtype), roi).shape,
                              dtype=np.float32)
            for i in range(M):
                frames[i] = _crop(img[i], roi)
        except OSError as e:                                          # pragma: no cover
            raise OSError(
                f"could not read {entry_name}/instrument/pco_ff/image: {e}. If this says the "
                "filter is unavailable, `hdf5plugin` failed to register its codec -- check it "
                "is installed in this environment.") from e
        obpitch = _read_obpitch(entry)
        if obpitch is None:
            notes.append("WARNING: obpitch not found at either known path; scan.meta['obpitch'] "
                         "is None")
        else:
            notes.append(f"plane {entry_name}: obpitch = {obpitch:.4f} deg (fixed for this "
                         "plane, not a scan axis)")
    if dark is not None:
        dark_arr = np.asarray(dark, dtype=np.float32)
        if dark_arr.ndim == 2:
            dark_arr = _crop(dark_arr, roi)
        frames = frames - dark_arr
        dark_level = dark_arr
    else:
        dark_level = 0.0
    return RockingScan.from_arrays(
        frames, {"mu": mu, "chi": chi}, dark_level=dark_level,
        source=f"{h5_path}::{entry_name}", notes=notes,
        meta={"h5_path": os.path.abspath(h5_path), "entry": entry_name, "obpitch": obpitch,
             "roi": roi})
