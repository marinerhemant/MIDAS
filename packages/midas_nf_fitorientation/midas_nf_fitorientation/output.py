"""Binary output writers compatible with the C ``pwrite`` layout.

The C drivers produce two binary files per run:

- ``MicFileBinary`` — fixed 11 doubles per voxel at offset
  ``voxel_idx * 88`` bytes:

  ``[bestRowNr, nWinners, fitTime, xs, ys, gridSize, ud,
   eulerA, eulerB, eulerC, fracOverlap]``

- ``MicFileBinary.AllMatches`` — ``7 + 4 * nSaves`` doubles per voxel
  at offset ``voxel_idx * 8 * (7 + 4*nSaves)`` bytes:

  ``[blockNr, nWinners, _reserved, xs, ys, gridSize, ud,
   eulA_1, eulB_1, eulC_1, frac_1,
   eulA_2, eulB_2, eulC_2, frac_2, ...]``

This module exposes a thin :class:`MicWriter` that pre-allocates both
files via ``numpy.memmap`` and writes per-voxel records by index,
which is the natural shape for the OpenMP-style block fan-out.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


# Number of doubles per record in MicFileBinary.
MIC_RECORD_DOUBLES = 11
MIC_RECORD_BYTES = MIC_RECORD_DOUBLES * 8


@dataclass
class MicRecord:
    """One row of ``MicFileBinary``."""
    best_row_nr: float
    n_winners: float
    fit_time_s: float
    xs: float
    ys: float
    grid_size: float
    ud: float
    euler_a: float
    euler_b: float
    euler_c: float
    frac_overlap: float

    def to_array(self) -> np.ndarray:
        return np.array([
            self.best_row_nr, self.n_winners, self.fit_time_s,
            self.xs, self.ys, self.grid_size, self.ud,
            self.euler_a, self.euler_b, self.euler_c, self.frac_overlap,
        ], dtype=np.float64)


class MicWriter:
    """Writer for ``MicFileBinary`` and ``MicFileBinary.AllMatches``.

    Pre-allocates both files with ``np.memmap`` (zero-filled) so that
    callers can write per-voxel records by index without worrying about
    file growth or seek correctness.

    Parameters
    ----------
    mic_path : str | Path
        Path to the primary output file.
    n_voxels : int
        Total number of voxels (sets the file size).
    n_saves : int
        Top-N solutions kept per voxel; sets the AllMatches record width.
    block_nr : int
        Used as the first column of each AllMatches record (the C code
        stores ``argv[2]`` here, the 0-based block index).
    """

    def __init__(
        self,
        mic_path: str | Path,
        n_voxels: int,
        n_saves: int = 1,
        block_nr: int = 0,
        create: bool = True,
    ):
        self.mic_path = Path(mic_path)
        self.n_voxels = n_voxels
        self.n_saves = max(1, n_saves)
        self.block_nr = block_nr

        self.am_path = Path(str(mic_path) + ".AllMatches")
        self.am_record_doubles = 7 + 4 * self.n_saves
        self.am_record_bytes = self.am_record_doubles * 8

        # ``create=True`` allocates and ZEROES the whole file. That is right for
        # a single-process run, but it must happen exactly once when the voxel
        # range is split across concurrent block processes -- otherwise every
        # worker zeroes the entire file and wipes the others' records. Parallel
        # workers pass ``create=False`` and open r+ instead, writing only the
        # rows in their own block (write_mic indexes by absolute voxel_idx, so
        # the regions are disjoint).
        mode = "w+" if create else "r+"
        self._mic = np.memmap(
            self.mic_path, dtype=np.float64, mode=mode,
            shape=(n_voxels, MIC_RECORD_DOUBLES),
        )
        self._am = np.memmap(
            self.am_path, dtype=np.float64, mode=mode,
            shape=(n_voxels, self.am_record_doubles),
        )
        if create:
            self._mic[:] = 0.0
            self._am[:] = 0.0

    # ------------------------------------------------------------------
    @staticmethod
    def allocate(mic_path: str | Path, n_voxels: int, n_saves: int = 1) -> None:
        """Create both output files zero-filled, without writing any records.

        Call this ONCE in the parent before fanning block workers out, so each
        worker can open the shared files with ``create=False``.
        """
        mic_path = Path(mic_path)
        am_path = Path(str(mic_path) + ".AllMatches")
        am_doubles = 7 + 4 * max(1, n_saves)
        for path, width in ((mic_path, MIC_RECORD_DOUBLES), (am_path, am_doubles)):
            arr = np.memmap(path, dtype=np.float64, mode="w+",
                            shape=(n_voxels, width))
            arr[:] = 0.0
            arr.flush()
            del arr

    # ------------------------------------------------------------------
    def write_mic(self, voxel_idx: int, rec: MicRecord) -> None:
        self._mic[voxel_idx] = rec.to_array()

    # ------------------------------------------------------------------
    def write_all_matches(
        self,
        voxel_idx: int,
        n_winners: int,
        xs: float,
        ys: float,
        grid_size: float,
        ud: float,
        sols: np.ndarray,   # (k, 4): (eulA, eulB, eulC, frac), sorted desc by frac
    ) -> None:
        """Write the AllMatches row for one voxel.

        ``sols`` may have fewer than ``n_saves`` rows; remaining slots
        are zero-filled (matching the C behaviour).
        """
        row = np.zeros(self.am_record_doubles, dtype=np.float64)
        row[0] = float(self.block_nr)
        row[1] = float(n_winners)
        row[2] = 0.0
        row[3] = xs
        row[4] = ys
        row[5] = grid_size
        row[6] = ud
        k = min(self.n_saves, sols.shape[0])
        for i in range(k):
            row[7 + 4 * i + 0] = sols[i, 0]
            row[7 + 4 * i + 1] = sols[i, 1]
            row[7 + 4 * i + 2] = sols[i, 2]
            row[7 + 4 * i + 3] = sols[i, 3]
        self._am[voxel_idx] = row

    # ------------------------------------------------------------------
    def flush(self) -> None:
        self._mic.flush()
        self._am.flush()

    def close(self) -> None:
        self.flush()
        # numpy.memmap doesn't have an explicit close in older versions;
        # deleting the reference triggers cleanup.
        del self._mic
        del self._am

    def __enter__(self) -> "MicWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
