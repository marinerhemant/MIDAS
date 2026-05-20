"""Binary file I/O for the NF-HEDM fit pipeline.

Provides readers for every binary the C executables consume
(``SpotsInfo.bin``, ``OrientMat.bin``, ``Key.bin``,
``DiffractionSpots.bin``, ``grid.txt``, ``hkls.csv``) and writers for
``MicFileBinary`` and ``MicFileBinary.AllMatches`` that match the C
``pwrite``-at-fixed-offset layout byte-for-byte.

All files are read via ``numpy.memmap`` where possible. The C code uses
``mmap`` and reads from ``DataDirectory`` directly.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np


# ---------------------------------------------------------------------------
#  Orientations
# ---------------------------------------------------------------------------

@dataclass
class OrientationData:
    """Precomputed candidate orientations and their predicted spots.

    Mirrors the bundle the C code reads from ``OrientMat.bin``,
    ``Key.bin``, and ``DiffractionSpots.bin``.

    Attributes
    ----------
    matrices : np.ndarray (N, 3, 3) float64
        Orientation matrices in row-major convention (the C code's
        ``OrientationMatrix[i*9 + 0..8]`` layout).
    n_spots : np.ndarray (N,) int32
        Number of predicted spots for each orientation.
    starts : np.ndarray (N,) int32
        Starting row of each orientation's spots in ``spots``.
    spots : np.ndarray (T, 3) float64
        Concatenated predicted spots, ``[2theta, eta, omega]`` (radians)
        per the convention written by ``MakeDiffrSpots``.
    """
    matrices: np.ndarray
    n_spots: np.ndarray
    starts: np.ndarray
    spots: np.ndarray

    @property
    def n_orientations(self) -> int:
        return int(self.matrices.shape[0])


def read_orientations(out_dir: str | Path) -> OrientationData:
    """Read ``OrientMat.bin`` + ``Key.bin`` + ``DiffractionSpots.bin``.

    Layout matches the C writers in ``MakeDiffrSpots.c``:
    - ``OrientMat.bin``: ``N * 9 * sizeof(double)``
    - ``Key.bin``: ``N * 2 * sizeof(int32)`` (NrSpots, StartingRowNr)
    - ``DiffractionSpots.bin``: ``T * 3 * sizeof(double)``
    """
    out = Path(out_dir)

    om_bytes = (out / "OrientMat.bin").stat().st_size
    if om_bytes % (9 * 8) != 0:
        raise ValueError(f"OrientMat.bin size {om_bytes} not a multiple of 72")
    n_or = om_bytes // (9 * 8)

    matrices = np.memmap(out / "OrientMat.bin", dtype=np.float64,
                         mode="r", shape=(n_or, 3, 3))

    key = np.memmap(out / "Key.bin", dtype=np.int32, mode="r",
                    shape=(n_or, 2))
    n_spots = np.ascontiguousarray(key[:, 0])
    starts = np.ascontiguousarray(key[:, 1])

    ds_bytes = (out / "DiffractionSpots.bin").stat().st_size
    if ds_bytes % (3 * 8) != 0:
        raise ValueError(
            f"DiffractionSpots.bin size {ds_bytes} not a multiple of 24"
        )
    n_ds = ds_bytes // (3 * 8)
    spots = np.memmap(out / "DiffractionSpots.bin", dtype=np.float64,
                      mode="r", shape=(n_ds, 3))

    # Force a real ndarray copy of matrices/spots so downstream torch
    # ingestion is straightforward (memmap doesn't always survive
    # reshape/torch-from-numpy on stale handles).
    return OrientationData(
        matrices=np.array(matrices, copy=True),
        n_spots=n_spots.astype(np.int64),
        starts=starts.astype(np.int64),
        spots=np.array(spots, copy=True),
    )


# ---------------------------------------------------------------------------
#  HKLs
# ---------------------------------------------------------------------------

@dataclass
class HKLTable:
    """HKL list parsed from ``hkls.csv`` written by ``GetHKLList``.

    The columns of ``hkls.csv`` (header: ``h k l D-spacing RingNr g1
    g2 g3 Theta 2Theta Radius``) carry both the integer Miller indices
    and the Cartesian reciprocal-space G-vectors. The C consumers
    (``MakeDiffrSpots``, ``FitOrientationOMP``) read columns 5–7
    (``g1, g2, g3``) — these are **already** Cartesian G-vectors with
    the B matrix applied, in 1/Å. Re-applying B to the integer
    Miller indices (as :func:`midas_diffract.hkls.hkls_for_forward_model`
    does for inputs that lack the precomputed Cartesian columns) would
    be a double conversion and produce wrong omegas. We therefore
    expose both:

    Attributes
    ----------
    hkls_int : np.ndarray (M, 3) float64
        Integer Miller indices (h, k, l). Used by
        :meth:`HEDMForwardModel.correct_hkls_latc` for strain
        refinement.
    hkls_cart : np.ndarray (M, 3) float64
        Cartesian G-vectors in 1/Å. Used by
        :meth:`HEDMForwardModel.calc_bragg_geometry` (the omega/eta
        solver). This is what the C ``MakeDiffrSpots`` actually
        consumes.
    rings : np.ndarray (M,) int64
        Ring number per row (the ``RingNr`` column).
    thetas_deg : np.ndarray (M,)
        Bragg angles in degrees as written by ``GetHKLList``.
    """
    hkls_int: np.ndarray
    hkls_cart: np.ndarray
    rings: np.ndarray
    thetas_deg: np.ndarray

    @property
    def n(self) -> int:
        return int(self.hkls_int.shape[0])

    def filter_rings(self, rings_to_use: List[int]) -> "HKLTable":
        """Return a new table keeping only rows whose ``ring`` is in
        ``rings_to_use``. Empty ``rings_to_use`` is a no-op.
        """
        if not rings_to_use:
            return self
        keep_set = set(int(r) for r in rings_to_use)
        mask = np.array([int(r) in keep_set for r in self.rings])
        return HKLTable(
            hkls_int=self.hkls_int[mask],
            hkls_cart=self.hkls_cart[mask],
            rings=self.rings[mask],
            thetas_deg=self.thetas_deg[mask],
        )


def read_hkls(out_dir: str | Path) -> HKLTable:
    """Parse ``hkls.csv``.

    The file has 11 whitespace-separated columns. The first line is a
    header we skip. We read both the integer Miller indices (cols
    0–2) and the precomputed Cartesian G-vectors (cols 5–7); see
    :class:`HKLTable` for why both matter.
    """
    path = Path(out_dir) / "hkls.csv"
    int_rows: List[Tuple[float, float, float]] = []
    cart_rows: List[Tuple[float, float, float]] = []
    rings_l: List[float] = []
    thetas_l: List[float] = []
    with open(path, "r") as f:
        # skip header
        f.readline()
        for line in f:
            tokens = line.split()
            if len(tokens) < 9:
                continue
            int_rows.append((
                float(tokens[0]), float(tokens[1]), float(tokens[2]),
            ))
            rings_l.append(float(tokens[4]))
            cart_rows.append((
                float(tokens[5]), float(tokens[6]), float(tokens[7]),
            ))
            thetas_l.append(float(tokens[8]))

    if not int_rows:
        raise ValueError(f"hkls.csv at {path} contained no parseable rows")

    return HKLTable(
        hkls_int=np.asarray(int_rows, dtype=np.float64),
        hkls_cart=np.asarray(cart_rows, dtype=np.float64),
        rings=np.asarray(rings_l, dtype=np.int64),
        thetas_deg=np.asarray(thetas_l, dtype=np.float64),
    )


# ---------------------------------------------------------------------------
#  Grid (voxel positions)
# ---------------------------------------------------------------------------

@dataclass
class GridTable:
    """Voxel positions from ``grid.txt``.

    The C grid file has a single header line giving ``TotalNrSpots``,
    followed by one line per voxel with five whitespace-separated floats
    ``y1 y2 xs ys gs`` (the asymmetric triangle parameters).

    Attributes
    ----------
    y1, y2, xs, ys, gs : np.ndarray (N,) float64
        Per-voxel triangle parameters as written by ``ParseMic``.
    ud : np.ndarray (N,) int8
        +1 for an "upward-pointing" triangle (``y1 ≤ y2``), -1 for
        downward (``y1 > y2``). Computed from ``y1`` and ``y2``.
    """
    y1: np.ndarray
    y2: np.ndarray
    xs: np.ndarray
    ys: np.ndarray
    gs: np.ndarray
    ud: np.ndarray

    @property
    def n_voxels(self) -> int:
        return int(self.xs.shape[0])

    def slice_block(self, block_nr: int, n_blocks: int) -> Tuple[int, int]:
        """Return ``(start, end_inclusive)`` voxel indices for a block.

        Mirrors the C block-decomposition::

            startRowNr = ceil(N / nBlocks) * blockNr
            endRowNr   = min(ceil(N / nBlocks) * (blockNr + 1), N - 1)
        """
        N = self.n_voxels
        chunk = -(-N // n_blocks)  # ceil
        start = chunk * block_nr
        end = min(chunk * (block_nr + 1), N - 1)
        return start, end

    def triangle_vertices(self, vi: int) -> np.ndarray:
        """Return the three voxel-triangle vertex (x, y) positions in
        sample-frame microns.

        Mirrors the C inline block at FitOrientationOMP.c:1413-1431.
        ``y1, y2`` are the two asymmetric height parameters from
        ``grid.txt``; their relative ordering encodes the up/down flag.
        Returns an array of shape ``(3, 2)`` with the three (x, y)
        rows in the same order the C code uses (``XY[0..2]``).
        """
        xs = self.xs[vi]
        ys = self.ys[vi]
        gs = self.gs[vi]
        y1 = self.y1[vi]
        y2 = self.y2[vi]
        if y1 > y2:           # downward-pointing (ud = -1)
            return np.array([
                [xs,       ys - y1],
                [xs - gs,  ys + y2],
                [xs + gs,  ys + y2],
            ], dtype=np.float64)
        # upward-pointing (ud = +1)
        return np.array([
            [xs,       ys + y2],
            [xs - gs,  ys - y1],
            [xs + gs,  ys - y1],
        ], dtype=np.float64)


def read_grid(out_dir: str | Path, grid_file_name: str | None = None) -> GridTable:
    """Read ``grid.txt`` (or the override path)."""
    if grid_file_name is None:
        grid_file_name = "grid.txt"
    path = Path(out_dir) / grid_file_name
    with open(path, "r") as f:
        header = f.readline().strip()
        n_total = int(header.split()[0])
        rows: List[Tuple[float, float, float, float, float]] = []
        for line in f:
            tokens = line.split()
            if len(tokens) < 5:
                continue
            rows.append((
                float(tokens[0]), float(tokens[1]),
                float(tokens[2]), float(tokens[3]),
                float(tokens[4]),
            ))
    if len(rows) != n_total:
        raise ValueError(
            f"{path}: header says {n_total} voxels but found {len(rows)}"
        )
    arr = np.asarray(rows, dtype=np.float64)
    y1 = arr[:, 0]
    y2 = arr[:, 1]
    ud = np.where(y1 > y2, -1, 1).astype(np.int8)
    return GridTable(
        y1=y1.copy(), y2=y2.copy(),
        xs=arr[:, 2].copy(), ys=arr[:, 3].copy(),
        gs=arr[:, 4].copy(),
        ud=ud,
    )


# ---------------------------------------------------------------------------
#  Reconstructed microstructure (.mic text file)
# ---------------------------------------------------------------------------

def read_mic_gridpoints(
    path: str | Path,
    *,
    min_confidence: float = 0.0,
    max_points: int | None = None,
) -> List[Tuple[float, float, float, float, float, float]]:
    """Derive multipoint ``GridPoints`` from a reconstructed text ``.mic``.

    The C ``FitOrientationParametersMultiPoint`` consumes ``GridPoints``
    lines whose columns are exactly the columns of a reconstructed
    ``.mic`` row (see ``FitOrientationParametersMultiPoint.c:697``)::

        OrientationRowNr  OrientationID  RunTime  X  Y  TriEdgeSize
        UpDown  Eul1  Eul2  Eul3  Confidence  PhaseNr

    so each grid point is ``(X, Y, UpDown, Eul1, Eul2, Eul3)`` and the C
    code drops everything else. When the param file carries no explicit
    ``GridPoints`` block, the multipoint driver derives them from the
    reconstruction this way instead of failing.

    Rows whose ``Confidence`` is below ``min_confidence`` are skipped.
    If ``max_points`` is given, the highest-confidence rows are kept
    (the C code caps the multipoint set at 200 voxels).

    Returns a list of ``(xc, yc, ud, eul1, eul2, eul3)`` tuples in the
    same shape :attr:`FitParams.grid_points` holds.
    """
    path = Path(path)
    rows: List[Tuple[float, float, float, float, float, float, float]] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("%"):
                continue
            tok = line.split()
            if len(tok) < 11:
                continue
            try:
                x = float(tok[3])
                y = float(tok[4])
                ud = float(tok[6])
                e1 = float(tok[7])
                e2 = float(tok[8])
                e3 = float(tok[9])
                conf = float(tok[10])
            except ValueError:
                continue
            if conf < min_confidence:
                continue
            rows.append((x, y, ud, e1, e2, e3, conf))

    # Highest-confidence first when capping the count.
    rows.sort(key=lambda r: r[6], reverse=True)
    if max_points is not None:
        rows = rows[:max_points]
    return [(r[0], r[1], r[2], r[3], r[4], r[5]) for r in rows]
