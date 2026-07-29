"""Generate the 5-column ``SpotsToIndex.csv`` the c-omp PF refiner
(``FitUnified.c``) reads in scanning mode.

Background
----------
The unified C refiner's PF path (``FitUnified.c`` ~line 1513) parses
``SpotsToIndex.csv`` as **five columns per voxel**::

    voxNr  SpId  nSpotsBest  (unused=0)  bestSolIdx

and uses ``nSpotsBest``/``bestSolIdx`` to pick each voxel's seed solution out
of the consolidated ``IndexBest_all.bin``.  The **python** indexer
(``Indexer.run_scanning``) writes ``IndexBest_all.bin`` directly and never
emits that 5-column file, so a subsequent ``--refine-backend c-omp`` run reads
a malformed (or single-column) ``SpotsToIndex.csv``, computes ``nSpotsBest<=0``
for every voxel, and silently refines **nothing**.

:func:`write_pf_seed_file` closes that gap: it derives the per-voxel seed rows
straight from ``IndexBest_all.bin`` using the *same* highest-completeness rule
the python refiner uses (:func:`scan_driver._top_candidate_index`,
``block[:,15] / max(block[:,14], 1)``), so the C refiner refines the identical
seed solution per voxel.  Reads are memory-light (``np.memmap`` — the file is
tens of GB).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# IndexBest_all.bin layout (see scan_driver._read_index_best_all):
#   int32   n_voxels
#   int32   n_sol[n_voxels]              — solutions per voxel
#   int64   off[n_voxels]                — offsets (recomputable; skipped)
#   float64 vals[total_solutions, 16]    — per-solution records
_VALS_COLS = 16
_N_EXPECTED_COL = 14        # NrExpected
_N_MATCHED_COL = 15         # NrObserved / nMatched


def write_pf_seed_file(index_best_all: str | Path, out_path: str | Path) -> int:
    """Write the 5-column PF ``SpotsToIndex.csv`` for the c-omp refiner.

    Parameters
    ----------
    index_best_all : path-like
        Consolidated ``IndexBest_all.bin`` from the (python or c-omp) scanning
        indexer.
    out_path : path-like
        Destination ``SpotsToIndex.csv`` (the C binary opens this name in cwd).

    Returns
    -------
    int
        Number of non-empty voxels written (= ``nSpotsToIndex`` to pass to the
        refiner as ``n_work``).
    """
    index_best_all = Path(index_best_all)
    out_path = Path(out_path)

    n_voxels = int(np.memmap(index_best_all, dtype=np.int32, mode="r",
                             shape=(1,))[0])
    n_sol = np.asarray(
        np.memmap(index_best_all, dtype=np.int32, mode="r", offset=4,
                  shape=(n_voxels,)),
        dtype=np.int64,
    )
    vals_off = 4 + 4 * n_voxels + 8 * n_voxels
    vals = np.memmap(index_best_all, dtype=np.float64, mode="r",
                     offset=vals_off).reshape(-1, _VALS_COLS)

    offsets = np.zeros(n_voxels + 1, dtype=np.int64)
    np.cumsum(n_sol, out=offsets[1:])

    lines = []
    for v in range(n_voxels):
        n = int(n_sol[v])
        if n <= 0:
            continue
        block = np.asarray(vals[offsets[v]:offsets[v + 1]])   # small per-voxel copy
        n_expected = np.maximum(block[:, _N_EXPECTED_COL], 1.0)
        best = int(np.argmax(block[:, _N_MATCHED_COL] / n_expected))
        n_spots = int(block[best, _N_MATCHED_COL])
        if n_spots <= 0:
            continue
        # voxNr  SpId(=voxNr label)  nSpotsBest  unused  bestSolIdx
        lines.append(f"{v} {v} {n_spots} 0 {best}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    return len(lines)
