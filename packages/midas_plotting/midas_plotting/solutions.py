"""Reading Laue indexing output.

The Laue analogue of :mod:`midas_plotting.grains`: one place that knows the
column layout so analysis scripts stop re-deriving it.

``LaueMatchingGPUStream`` writes two text files per run into its ``ResultDir``:

``solutions.txt``
    one row per accepted orientation per frame -- 35 columns, ending
    ``... OrientMatrix0..8, CoarseNMatches*sqrt(Intensity),
    misOrientationPostRefinement, orientationRowNr``.

``spots.txt``
    one row per assigned reflection -- ``ImageNr GrainNr SpotNr h k l X Y
    Qhat[0..2] Intensity``.

Columns are looked up **by name** from the ``%ImageNr ...`` header, never by
position. The failure this prevents is not hypothetical: ``orientationRowNr`` is
column 34 and ``misOrientationPostRefinement`` is column 33, and reading 33 for
34 does not raise -- it returns a float near zero for every row, so every
"distinct orientations" count silently collapses to single digits and the scan
looks like it found one crystal.

Nothing here imports the indexer. These are stable text formats, and a plotting
package that could not open a results directory without the GPU pipeline
installed would not be much use at a beamline.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

__all__ = [
    "LaueSolutions", "LaueSpots", "read_solutions", "read_spots",
    "read_validated", "COS45",
]

#: The out-of-plane stage axis at 34-ID-E sits at 45 deg to the beam, so a
#: recorded Z step is ``1/cos45`` shorter than the distance travelled across the
#: sample surface. Divide recorded Z by this to get true in-sample micrometres.
#: Quoting the raw stage extent as a map size understates it by 1.41x -- a
#: 200 x 100 um map reads as 200 x 71.
COS45 = float(np.sqrt(0.5))

_OM_NAMES = [f"OrientMatrix{i}" for i in range(9)]
_LATTICE_NAMES = ["LatticeParameterFit[a]", "LatticeParameterFit[b]",
                  "LatticeParameterFit[c]", "LatticeParameterFit[alpha]",
                  "LatticeParameterFit[beta]", "LatticeParameterFit[gamma]"]

#: An orientation matrix should be a proper rotation. Text output at ~7
#: significant figures round-trips to well under this, so anything larger means
#: the row is being sliced wrong rather than that the fit was poor.
ORTHONORMAL_TOL = 1e-3


@dataclass
class LaueSolutions:
    """Accepted orientations from a Laue run.

    One row per *orientation per frame*, not per grain: a crystal spanning
    twenty raster positions appears twenty times. Grains come from clustering
    these, which :func:`midas_plotting.laue.cluster` does.

    Attributes
    ----------
    image : (N,) int
        Frame number within the shard, as written by the indexer. This indexes
        into that run's ``frame_mapping.json``; it is **not** a position.
    grain : (N,) int
        Solution index within the frame (0, 1, 2 ... for multiple orientations
        on one frame). Not a grain identity across frames.
    n_matches : (N,) int
        Reflections the orientation matched. This is the quantity every
        acceptance gate in the MIDAS Laue work is applied to.
    orient_mat : (N, 3, 3) float
    lattice : (N, 6) float or None
        a, b, c (nm as written by the indexer) and alpha, beta, gamma (degrees).
    row_nr : (N,) int or None
        ``orientationRowNr`` -- the row in the orientation library. Two rows
        with the same value are the same orientation to the library's spacing,
        which makes this the cheap way to count *distinct* orientations without
        clustering.
    intensity : (N,) float or None
    misorientation : (N,) float or None
        ``misOrientationPostRefinement``, degrees.
    pos : (N, 2) float or None
        Sample-frame x, y in micrometres, when positions were supplied. The
        out-of-plane axis is already divided by :data:`COS45`.
    columns : list of str
    raw : (N, C) float
    path : Path or None
    """

    image: np.ndarray
    grain: np.ndarray
    n_matches: np.ndarray
    orient_mat: np.ndarray
    lattice: Optional[np.ndarray] = None
    row_nr: Optional[np.ndarray] = None
    intensity: Optional[np.ndarray] = None
    misorientation: Optional[np.ndarray] = None
    pos: Optional[np.ndarray] = None
    columns: Optional[list] = None
    raw: Optional[np.ndarray] = None
    path: Optional[Path] = None
    frame_name: Optional[np.ndarray] = None

    def __len__(self) -> int:
        return int(self.image.shape[0])

    @property
    def n_distinct(self) -> Optional[int]:
        """Distinct library rows, or None if ``orientationRowNr`` was absent.

        A useful sanity number on its own: a scan whose thousands of solutions
        use a handful of library rows has found one object many times, not many
        objects -- the signature of a substrate being indexed as the deposit.
        """
        if self.row_nr is None:
            return None
        return int(np.unique(self.row_nr).size)

    def gate(self, n_matches: int) -> "LaueSolutions":
        """Keep solutions matching **more than** ``n_matches`` reflections.

        The threshold is a property of the scan, not of this package: it should
        be the largest number of matches a *randomly oriented* crystal achieves
        on these frames. There is no default here on purpose.
        """
        return self[self.n_matches > int(n_matches)]

    def __getitem__(self, m) -> "LaueSolutions":
        def sel(a):
            return None if a is None else a[m]
        return LaueSolutions(
            image=self.image[m], grain=self.grain[m],
            n_matches=self.n_matches[m], orient_mat=self.orient_mat[m],
            lattice=sel(self.lattice), row_nr=sel(self.row_nr),
            intensity=sel(self.intensity), misorientation=sel(self.misorientation),
            pos=sel(self.pos), columns=self.columns, raw=sel(self.raw),
            path=self.path, frame_name=sel(self.frame_name))

    def summary(self) -> str:
        parts = [f"{len(self)} solutions",
                 f"{np.unique(self.image).size} frames"]
        if self.n_distinct is not None:
            parts.append(f"{self.n_distinct} distinct orientations")
        parts.append(f"matches {int(self.n_matches.min())}-"
                     f"{int(self.n_matches.max())} (median "
                     f"{int(np.median(self.n_matches))})")
        if self.pos is not None and len(self):
            parts.append(f"map {np.ptp(self.pos[:, 0]):.0f} x "
                         f"{np.ptp(self.pos[:, 1]):.0f} um")
        return "; ".join(parts)


@dataclass
class LaueSpots:
    """Reflections assigned to accepted orientations (``spots.txt``)."""

    image: np.ndarray
    grain: np.ndarray
    hkl: np.ndarray
    xy: np.ndarray
    qhat: Optional[np.ndarray] = None
    intensity: Optional[np.ndarray] = None
    columns: Optional[list] = None
    path: Optional[Path] = None

    def __len__(self) -> int:
        return int(self.image.shape[0])

    def for_frame(self, image: int) -> "LaueSpots":
        m = self.image == int(image)
        def sel(a):
            return None if a is None else a[m]
        return LaueSpots(image=self.image[m], grain=self.grain[m],
                         hkl=self.hkl[m], xy=self.xy[m], qhat=sel(self.qhat),
                         intensity=sel(self.intensity), columns=self.columns,
                         path=self.path)


def _read_named(path, expect: str):
    """Header-named table -> (dict name->index, columns, data)."""
    path = Path(path)
    with open(path) as fh:
        header = fh.readline()
    if not header.startswith("%"):
        raise ValueError(
            f"{path} does not start with a '%'-prefixed header line; this does "
            f"not look like a MIDAS Laue {expect} file")
    cols = header.lstrip("%").split()
    idx = {c: i for i, c in enumerate(cols)}
    data = np.atleast_2d(np.loadtxt(path, skiprows=1, ndmin=2))
    if data.size and data.shape[1] != len(cols):
        raise ValueError(
            f"{path}: header names {len(cols)} columns but rows have "
            f"{data.shape[1]}")
    return idx, cols, data


def _need(idx, name, path):
    if name not in idx:
        raise KeyError(
            f"{path}: column {name!r} not found. Present: {sorted(idx)}. "
            f"Reading Laue output positionally is what this reader exists to "
            f"prevent, so it will not guess an index.")
    return idx[name]


def read_solutions(path, positions=None, *, check: bool = True) -> LaueSolutions:
    """Parse a ``solutions.txt``.

    Parameters
    ----------
    path : str or Path
    positions : (M, 2) array, dict, or None
        Optional sample-frame coordinates. An ``(M, 2)`` array is indexed by
        ``image - 1``; a dict maps image number to ``(x, y)``. Supply these in
        **micrometres already corrected** for the 45 deg stage axis (see
        :data:`COS45`), or pass ``raw_z=True`` style corrected values yourself.
    check : bool
        Verify each orientation matrix is a proper rotation and warn if not.

    Notes
    -----
    Rows are *orientation per frame*. Counting grains from ``len(sol)`` counts
    one crystal once per position it was seen at.
    """
    path = Path(path)
    idx, cols, data = _read_named(path, "solutions.txt")
    if data.size == 0:
        return LaueSolutions(
            image=np.zeros(0, int), grain=np.zeros(0, int),
            n_matches=np.zeros(0, int), orient_mat=np.zeros((0, 3, 3)),
            columns=cols, raw=data, path=path)

    image = data[:, _need(idx, "ImageNr", path)].astype(int)
    grain = data[:, _need(idx, "GrainNr", path)].astype(int)
    nmat = data[:, _need(idx, "NMatches", path)].astype(int)
    om = data[:, [_need(idx, n, path) for n in _OM_NAMES]].reshape(-1, 3, 3)

    def opt(name, cast=float):
        return data[:, idx[name]].astype(cast) if name in idx else None

    lattice = (data[:, [idx[n] for n in _LATTICE_NAMES]]
               if all(n in idx for n in _LATTICE_NAMES) else None)
    row_nr = opt("orientationRowNr", np.int64)
    if row_nr is None:
        warnings.warn(
            f"{path}: no 'orientationRowNr' column, so distinct-orientation "
            f"counts are unavailable. Do NOT substitute "
            f"'misOrientationPostRefinement' -- it is the adjacent column and "
            f"reading it instead returns near-zero for every row.",
            RuntimeWarning, stacklevel=2)

    if check:
        _check_rotations(om, path)

    pos = _resolve_positions(positions, image, path)
    return LaueSolutions(
        image=image, grain=grain, n_matches=nmat, orient_mat=om,
        lattice=lattice, row_nr=row_nr, intensity=opt("Intensity"),
        misorientation=opt("misOrientationPostRefinement"), pos=pos,
        columns=cols, raw=data, path=path)


def _check_rotations(om: np.ndarray, path) -> None:
    """Warn if the matrices are not proper rotations.

    The FF reader cross-checks Euler angles against the matrix; Laue output
    carries no Euler column, so the available invariant is that a valid
    orientation matrix satisfies ``R R^T = I`` and ``det R = +1``. A row read
    with the wrong column offset fails both, which turns a silent
    mis-slice into a message.
    """
    if om.size == 0:
        return
    eye = np.einsum("nij,nkj->nik", om, om)
    off = np.abs(eye - np.eye(3)).reshape(len(om), -1).max(axis=1)
    det = np.linalg.det(om)
    bad = (off > ORTHONORMAL_TOL) | (np.abs(det - 1.0) > ORTHONORMAL_TOL)
    if bad.any():
        warnings.warn(
            f"{path}: {int(bad.sum())} of {len(om)} orientation matrices are "
            f"not proper rotations (max |RR^T-I| = {off.max():.2e}, det range "
            f"{det.min():.4f}..{det.max():.4f}). The columns are probably being "
            f"sliced wrong.", RuntimeWarning, stacklevel=3)


def _resolve_positions(positions, image, path):
    if positions is None:
        return None
    if isinstance(positions, dict):
        miss = [int(i) for i in np.unique(image) if int(i) not in positions]
        if miss:
            warnings.warn(f"{path}: no position for {len(miss)} image numbers "
                          f"(e.g. {miss[:5]}); those rows get NaN",
                          RuntimeWarning, stacklevel=3)
        out = np.full((len(image), 2), np.nan)
        for k, im in enumerate(image):
            p = positions.get(int(im))
            if p is not None:
                out[k] = p
        return out
    arr = np.asarray(positions, dtype=float).reshape(-1, 2)
    j = image - 1
    if j.min() < 0 or j.max() >= len(arr):
        raise IndexError(
            f"{path}: image numbers run {image.min()}..{image.max()} but "
            f"positions has {len(arr)} rows. Image numbers are 1-based and "
            f"per-shard; a whole-scan position table does not line up with a "
            f"single shard's solutions.txt.")
    return arr[j]


def read_spots(path) -> LaueSpots:
    """Parse a ``spots.txt``.

    ``X`` and ``Y`` are detector pixels. These are the indexer's own spot
    positions -- **not** interchangeable with those from an analysis-side peak
    finder run over the same frame. The two detect differently and their
    coordinates do not coincide; a spot list built from the wrong one silently
    selects nothing.
    """
    path = Path(path)
    idx, cols, data = _read_named(path, "spots.txt")
    if data.size == 0:
        return LaueSpots(image=np.zeros(0, int), grain=np.zeros(0, int),
                         hkl=np.zeros((0, 3)), xy=np.zeros((0, 2)),
                         columns=cols, path=path)
    q = [f"Qhat[{i}]" for i in range(3)]
    return LaueSpots(
        image=data[:, _need(idx, "ImageNr", path)].astype(int),
        grain=data[:, _need(idx, "GrainNr", path)].astype(int),
        hkl=data[:, [_need(idx, k, path) for k in ("h", "k", "l")]],
        xy=data[:, [_need(idx, k, path) for k in ("X", "Y")]],
        qhat=data[:, [idx[k] for k in q]] if all(k in idx for k in q) else None,
        intensity=data[:, idx["Intensity"]] if "Intensity" in idx else None,
        columns=cols, path=path)


def read_validated(path, *, raw_z: bool = True) -> LaueSolutions:
    """Load validated instances from an analysis ``.npz``.

    Accepts the arrays the MIDAS Laue analysis pipeline writes after per-frame
    validation: ``oms`` ``(N, 3, 3)``, ``X`` and ``Z`` (stage micrometres),
    ``nhit``, and optionally ``frames`` and ``labels``. Several shards can be
    concatenated by passing a sequence of paths.

    Parameters
    ----------
    raw_z : bool
        ``Z`` in these files is the **stage** coordinate, so it is divided by
        :data:`COS45` to give true in-sample distance. Pass ``False`` only if
        the file already holds corrected values.
    """
    paths = [Path(path)] if isinstance(path, (str, Path)) else [Path(p) for p in path]
    oms, X, Z, nh, fr = [], [], [], [], []
    for p in paths:
        d = np.load(p, allow_pickle=True)
        for key in ("oms", "X", "Z", "nhit"):
            if key not in d:
                raise KeyError(f"{p}: expected array {key!r}; found {list(d)}")
        oms.append(np.asarray(d["oms"]).reshape(-1, 3, 3))
        X.append(np.asarray(d["X"], float).ravel())
        Z.append(np.asarray(d["Z"], float).ravel())
        nh.append(np.asarray(d["nhit"]).ravel())
        fr.append(np.asarray(d["frames"]).ravel() if "frames" in d
                  else np.arange(len(oms[-1])))
    om = np.concatenate(oms)
    x = np.concatenate(X)
    z = np.concatenate(Z)
    if raw_z:
        z = z / COS45
    n = np.concatenate(nh).astype(int)
    _check_rotations(om, paths[0])

    # ``frames`` is written as frame FILENAMES by some versions of the analysis
    # and as integers by others, so it cannot be cast blindly. Names are kept as
    # ``frame_name`` and ``image`` falls back to a running index -- silently
    # int()-ing 'scan100Cu_1.h5' is a crash at best and a wrong join at worst.
    raw_fr = np.concatenate(fr)
    try:
        image = raw_fr.astype(int)
        names = None
    except (ValueError, TypeError):
        names = raw_fr.astype(str)
        _, image = np.unique(names, return_inverse=True)
        image = image.astype(int) + 1

    return LaueSolutions(
        image=image, grain=np.zeros(len(om), int), n_matches=n, orient_mat=om,
        pos=np.stack([x, z], axis=1), path=paths[0], frame_name=names)
