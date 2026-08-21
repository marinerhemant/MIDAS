"""``SpotDiagnostics.bin`` — every theoretical spot, matched or not.

This is the only artifact that records the reflections a grain/voxel was
*predicted* to produce but which were **not found**. `Grains.csv`,
`SpotMatrix.csv` and `FitBest.bin` all describe matched spots only, so
completeness can be read off them as a number but never explained. Written by
``FitUnified.c`` (``DoSpotDiag = 1``, on by default) for **both** FF and PF —
the comment at ``FitUnified.c:1596`` claiming PF-only is wrong; a 55,593-seed
FF layer produced a 977 MB file.

File layout (little-endian, matching ``FitUnified.c:2309-2358``)::

    header    64 B : magic 0x47414944 "DIAG", uint32 version, int32 nVoxels,
                     int32 nCols, float64 sentinel, 40 B reserved
    directory      : nVoxels x 3 int32   (voxNr, nTheor, nMatched)
    metadata       : nVoxels x 13 float64
                     (voxNr, pos[3], euler[3], a,b,c,alpha,beta,gamma)
    spotdata       : per voxel, nTheor x nCols float64, matched rows first

Per-spot columns (19)::

     0 theorY        3 theorEta      6 theorGx      9  theorScanNr
     1 theorZ        4 ringNr        7 theorGy     10  matched (1/0)
     2 theorOmega    5 theorSpotID   8 theorGz
    11 obsY         13 obsOmega     15 obsScanNr   17  diffLen
    12 obsZ         14 obsSpotID    16 IA          18  diffOme

Unmatched rows carry the prediction in 0-10 and the **sentinel** (-999.0) in
every observed column 11-18.

Version history
---------------
**v1** — ``col 5`` is ``theorSpotID`` on unmatched rows but ``theorGx`` on
matched rows (the writer's own comment said "repurpose as hklIdx proxy"), so
the column means two different things depending on ``matched``. Measured on a
55,593-voxel FF layer: ``col5 == col6`` on 41,118/41,118 matched rows. The
original reader (``utils/spot_diagnostics.py``) labelled it ``hklIndex`` for
every row, i.e. wrong for exactly the matched half. Also ``obsScanNr``
(col 15) was read one row past the spot, because SpotIDs are 1-based rows of
Spots.bin (verified: ``Spots.bin`` col 4 == row+1 for all 617,505 rows) and
the lookup indexed by SpotID directly; the last spot was sentinelled outright.
Both are invisible in FF, where the scan column is identically 0.

**v2** (2026-08-21) — both fixed. Layout and column count unchanged, so a v1
reader still parses a v2 file; it would simply mislabel col 5 as before.
:attr:`SpotDiag.col5_is_theor_spot_id` reports whether col 5 is trustworthy on
matched rows.

``theorSpotID`` is the stable per-reflection id
``ih * 2 + 1 + within`` from ``CalcDiffractionSpots.c:112`` — ``ih`` indexes
the hkl table and ``within`` distinguishes the two omega solutions. It is NOT
a bare hkl row index, which is why it is not called ``hklIndex`` here.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np

__all__ = ["SpotDiag", "SPOT_DIAG_COLS", "SPOT_DIAG_META", "SPOT_DIAG_MAGIC",
           "SPOT_DIAG_SENTINEL", "load_spot_diag",
           "PF_SPOT_MATRIX_COLS", "PF_SPOT_MATRIX_HEADER", "write_pf_spot_matrix"]

SPOT_DIAG_MAGIC = 0x47414944          # "DIAG"
SPOT_DIAG_SENTINEL = -999.0
SPOT_DIAG_SUPPORTED_VERSIONS = (1, 2)

SPOT_DIAG_COLS = (
    "theorY", "theorZ", "theorOmega", "theorEta", "ringNr", "theorSpotID",
    "theorGx", "theorGy", "theorGz", "theorScanNr",
    "matched", "obsY", "obsZ", "obsOmega", "obsSpotID", "obsScanNr",
    "IA", "diffLen", "diffOme",
)
SPOT_DIAG_META = (
    "voxelNr", "posX", "posY", "posZ", "euler1", "euler2", "euler3",
    "a", "b", "c", "alpha", "beta", "gamma",
)

_COL = {n: i for i, n in enumerate(SPOT_DIAG_COLS)}


class SpotDiag:
    """Reader for ``SpotDiagnostics.bin``.

    Lifted from ``utils/spot_diagnostics.py`` (which remains as a shim, along
    with its interactive plotter) so pipeline code can depend on it. Adds
    version awareness and named-column access.

    The whole spot table is read into RAM: 977 MB of float64 on a 55,593-voxel
    layer. That is the file's total size, not a multiple of it — unlike
    ``FitBest.bin`` there is no per-seed padding, so there is nothing to gain
    from memmapping it lazily.
    """

    def __init__(self, path: Union[str, Path]):
        self.path = Path(path)
        with open(self.path, "rb") as f:
            magic, version, n_vox, n_cols = struct.unpack("<IIii", f.read(16))
            if magic != SPOT_DIAG_MAGIC:
                raise ValueError(
                    f"{self.path}: bad magic 0x{magic:08X} "
                    f"(expected 0x{SPOT_DIAG_MAGIC:08X}) — not a "
                    f"SpotDiagnostics.bin"
                )
            if version not in SPOT_DIAG_SUPPORTED_VERSIONS:
                raise ValueError(
                    f"{self.path}: SpotDiagnostics version {version} is not "
                    f"supported (known: {SPOT_DIAG_SUPPORTED_VERSIONS}). A "
                    f"newer writer may have changed the column layout — "
                    f"refusing to guess."
                )
            self.version = int(version)
            self.sentinel = struct.unpack("<d", f.read(8))[0]
            f.read(40)                                    # reserved
            self.n_voxels = int(n_vox)
            self.n_cols = int(n_cols)
            if self.n_cols != len(SPOT_DIAG_COLS):
                raise ValueError(
                    f"{self.path}: header says {self.n_cols} columns, this "
                    f"reader knows {len(SPOT_DIAG_COLS)} "
                    f"({', '.join(SPOT_DIAG_COLS)})"
                )

            d = np.frombuffer(f.read(12 * self.n_voxels),
                              dtype=np.int32).reshape(self.n_voxels, 3)
            self.voxel_nrs = d[:, 0].copy()
            self.n_theor = d[:, 1].copy()
            self.n_matched = d[:, 2].copy()

            self.metadata = np.frombuffer(
                f.read(13 * 8 * self.n_voxels), dtype=np.float64
            ).reshape(self.n_voxels, 13).copy()

            self._offsets = np.zeros(self.n_voxels + 1, dtype=np.int64)
            np.cumsum(self.n_theor, out=self._offsets[1:])
            total = int(self._offsets[-1])
            self.spots = np.frombuffer(
                f.read(total * self.n_cols * 8), dtype=np.float64
            ).reshape(total, self.n_cols).copy()
            if self.spots.shape[0] != total:
                raise ValueError(
                    f"{self.path}: directory promises {total} spot rows, file "
                    f"holds {self.spots.shape[0]} — truncated."
                )
        self._nr_to_idx = {int(n): i for i, n in enumerate(self.voxel_nrs)}

    # -- provenance --------------------------------------------------------
    @property
    def col5_is_theor_spot_id(self) -> bool:
        """True when ``theorSpotID`` is valid on matched rows too (v2+).

        On v1 files col 5 holds ``theorGx`` for matched rows; use it only
        where ``matched == 0``, or re-run the refiner.
        """
        return self.version >= 2

    def __repr__(self) -> str:
        return (f"SpotDiag({self.path.name}, v{self.version}, "
                f"{self.n_voxels} voxels, {self.spots.shape[0]:,} spots, "
                f"{self.n_matched.sum():,} matched)")

    # -- access ------------------------------------------------------------
    def col(self, name: str) -> np.ndarray:
        """Whole-file view of one named column."""
        try:
            return self.spots[:, _COL[name]]
        except KeyError:
            raise KeyError(
                f"unknown column {name!r}; known: {', '.join(SPOT_DIAG_COLS)}"
            ) from None

    def voxel(self, idx: int) -> Dict[str, object]:
        """One voxel by array index: metadata + its ``(nTheor, 19)`` block."""
        if not (0 <= idx < self.n_voxels):
            raise IndexError(
                f"voxel index {idx} out of range (0..{self.n_voxels - 1})")
        s0, s1 = int(self._offsets[idx]), int(self._offsets[idx + 1])
        m = self.metadata[idx]
        return {
            "voxelNr": int(self.voxel_nrs[idx]),
            "position": m[1:4], "euler": m[4:7], "latc": m[7:13],
            "nTheor": int(self.n_theor[idx]),
            "nMatched": int(self.n_matched[idx]),
            "spots": self.spots[s0:s1],
        }

    def voxel_by_nr(self, voxel_nr: int) -> Dict[str, object]:
        idx = self._nr_to_idx.get(int(voxel_nr))
        if idx is None:
            raise KeyError(f"voxel {voxel_nr} not in {self.path.name}")
        return self.voxel(idx)

    def matched_mask(self) -> np.ndarray:
        """Boolean mask over the whole spot table."""
        return self.spots[:, _COL["matched"]] > 0.5

    def unmatched(self, voxel_nr: Optional[int] = None) -> np.ndarray:
        """The un-found expected spots — the completeness deficit itself."""
        s = (self.voxel_by_nr(voxel_nr)["spots"] if voxel_nr is not None
             else self.spots)
        return s[s[:, _COL["matched"]] <= 0.5]

    def completeness(self) -> np.ndarray:
        """Per-voxel ``nMatched / nTheor`` (NaN where nothing was predicted)."""
        out = np.full(self.n_voxels, np.nan)
        nz = self.n_theor > 0
        out[nz] = self.n_matched[nz] / self.n_theor[nz]
        return out

    def completeness_by_ring(
        self, voxel_nr: Optional[int] = None,
    ) -> Dict[int, Dict[str, float]]:
        """Matched / predicted per ring — which rings the fit is losing."""
        s = (self.voxel_by_nr(voxel_nr)["spots"] if voxel_nr is not None
             else self.spots)
        ring = s[:, _COL["ringNr"]]
        matched = s[:, _COL["matched"]] > 0.5
        out: Dict[int, Dict[str, float]] = {}
        for r in np.unique(ring[ring > 0]).astype(int):
            sel = ring == r
            tot = int(sel.sum())
            hit = int(matched[sel].sum())
            out[int(r)] = {"total": tot, "matched": hit,
                           "frac": hit / tot if tot else 0.0}
        return out

    def summary(self) -> str:
        tot = int(self.n_theor.sum())
        hit = int(self.n_matched.sum())
        lines = [
            f"{self.path.name}: v{self.version}, {self.n_voxels:,} voxels, "
            f"{tot:,} predicted spots, {hit:,} matched "
            f"({tot - hit:,} un-found)",
            f"  mean completeness {hit / tot if tot else float('nan'):.4f}",
        ]
        if not self.col5_is_theor_spot_id:
            lines.append(
                "  WARNING: v1 file — col 5 (theorSpotID) is theorGx on "
                "MATCHED rows; trust it only where matched == 0. Col 15 "
                "(obsScanNr) is also off by one row on this version."
            )
        return "\n".join(lines)


def load_spot_diag(run_dir: Union[str, Path]) -> SpotDiag:
    """Load ``SpotDiagnostics.bin`` from a run dir (``Results/`` or bare)."""
    rd = Path(run_dir)
    for cand in (rd / "Results" / "SpotDiagnostics.bin",
                 rd / "SpotDiagnostics.bin"):
        if cand.exists():
            return SpotDiag(cand)
    raise FileNotFoundError(
        f"SpotDiagnostics.bin not found under {rd} (looked in Results/ and "
        f"the run dir itself). It is written by the c-omp refiner with "
        f"DoSpotDiag on, which is the default."
    )


# ---------------------------------------------------------------------------
# PF SpotMatrix
# ---------------------------------------------------------------------------

#: Column layout of the PF ``SpotMatrix.csv``.
#:
#: PF never had one. FF's is keyed by grain and joins observed spot properties
#: out of ``InputAllExtraInfoFittingAll.csv``; PF has one such file per scan, so
#: that join needs the scan number to be correct. ``SpotDiagnostics.bin``
#: already carries observed position, predicted position, the residuals, the
#: matched flag AND the scan, so this file is built from it alone and is
#: self-consistent by construction.
#:
#: Keyed by **voxel**, not grain — that is PF's unit, and a voxel is not a
#: grain. Un-found rows use -1 in the integer columns (they are printed %d and
#: cannot hold NaN) and NaN in the observed floats, matching FF's convention.
PF_SPOT_MATRIX_COLS = (
    "VoxelNr",        # 0  PF's analogue of FF's GrainID
    "SpotID",         # 1  observed; -1 when the spot was never found
    "Omega",          # 2  observed
    "YLab",           # 3  observed
    "ZLab",           # 4  observed
    "ScanNr",         # 5  observed scan; -1 when un-found
    "RingNr",         # 6  PREDICTED ring — known even for un-found spots
    "Matched",        # 7  1 = observed and matched, 0 = predicted, never found
    "theorSpotID",    # 8
    "theorEta",       # 9
    "YExp",           # 10 prediction
    "ZExp",           # 11
    "OmegaExp",       # 12
    "theorScanNr",    # 13 the scan the model expects to light this reflection
    "DiffLen",        # 14 per-spot residuals (NaN on un-found rows)
    "DiffOme",        # 15
    "InternalAngle",  # 16
)
PF_SPOT_MATRIX_HEADER = "%" + "\t".join(PF_SPOT_MATRIX_COLS) + "\n"


def write_pf_spot_matrix(diag: "SpotDiag", out_path: Union[str, Path]) -> int:
    """Write a per-voxel ``SpotMatrix.csv`` for PF from a ``SpotDiag``.

    One row per **predicted** reflection per voxel, matched or not. The
    un-found rows are the point: they are the completeness deficit, and PF
    records it nowhere else — ``Result_OrientPos_voxel_*.csv`` carries only a
    completeness *number*.

    Returns the number of data rows written.
    """
    d = diag
    sd = d.spots
    n = sd.shape[0]
    out = np.full((n, len(PF_SPOT_MATRIX_COLS)), np.nan)

    # voxel id per spot row, expanded from the directory
    vox = np.repeat(d.voxel_nrs.astype(np.float64), d.n_theor.astype(np.int64))
    matched = sd[:, _COL["matched"]] > 0.5

    out[:, 0] = vox
    out[:, 1] = np.where(matched, sd[:, _COL["obsSpotID"]], -1.0)
    out[:, 2] = np.where(matched, sd[:, _COL["obsOmega"]], np.nan)
    out[:, 3] = np.where(matched, sd[:, _COL["obsY"]], np.nan)
    out[:, 4] = np.where(matched, sd[:, _COL["obsZ"]], np.nan)
    scan = sd[:, _COL["obsScanNr"]]
    out[:, 5] = np.where(matched & (scan != d.sentinel), scan, -1.0)
    out[:, 6] = sd[:, _COL["ringNr"]]
    out[:, 7] = np.where(matched, 1.0, 0.0)
    out[:, 8] = sd[:, _COL["theorSpotID"]]
    out[:, 9] = sd[:, _COL["theorEta"]]
    out[:, 10] = sd[:, _COL["theorY"]]
    out[:, 11] = sd[:, _COL["theorZ"]]
    out[:, 12] = sd[:, _COL["theorOmega"]]
    ts = sd[:, _COL["theorScanNr"]]
    out[:, 13] = np.where(ts != d.sentinel, ts, -1.0)
    for c, name in ((14, "diffLen"), (15, "diffOme"), (16, "IA")):
        out[:, c] = np.where(matched, sd[:, _COL[name]], np.nan)

    if not d.col5_is_theor_spot_id:
        # v1 wrote theorGx into col 5 on matched rows. Emitting it as
        # "theorSpotID" would be wrong for exactly half the file, so blank it
        # there rather than ship a column that is right only sometimes.
        out[matched, 8] = np.nan

    fmt = ("%d\t%d\t%.6f\t%.6f\t%.6f\t%d\t%d\t%d\t%.0f"
           + "\t%.6f" * 4 + "\t%d" + "\t%.6f" * 3)
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        f.write(PF_SPOT_MATRIX_HEADER)
        np.savetxt(f, out, fmt=fmt)
    return n
