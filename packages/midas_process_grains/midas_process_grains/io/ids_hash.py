"""``IDsHash.csv`` — SpotID-range-per-ring lookup.

Each line of ``IDsHash.csv`` is::

    <ring_nr> <id_min> <id_max> <d_spacing_A>

The C code uses this table to look up the reference d-spacing for a matched
SpotID inside the strain solver — see ``CalcStrains.c::StrainTensorKenesei``
and ``ProcessGrains.c:797-832``. We reuse the same convention so our Phase-4
strain has identical reference d-values.

Note: ``id_max`` is **exclusive** in the C convention (the next ring starts
where this one ends — e.g. ``3 1 738333 ...`` then ``4 738333 ...``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np


@dataclass(frozen=True)
class IDsHash:
    """Sorted SpotID-range table.

    Attributes
    ----------
    ring_nrs : np.ndarray
        ``(n_rings,)`` int — ring numbers in ascending order.
    id_starts : np.ndarray
        ``(n_rings,)`` int64 — inclusive lower SpotID bound for each ring.
    id_ends : np.ndarray
        ``(n_rings,)`` int64 — exclusive upper SpotID bound.
    d_spacings : np.ndarray
        ``(n_rings,)`` float64 — d-spacing in Å for each ring.
    """

    ring_nrs: np.ndarray
    id_starts: np.ndarray
    id_ends: np.ndarray
    d_spacings: np.ndarray

    def ring_for_spot_id(self, spot_id: int) -> int:
        """Return the ring number for a SpotID, or ``-1`` if out of range."""
        idx = np.searchsorted(self.id_starts, spot_id, side="right") - 1
        if idx < 0 or idx >= self.ring_nrs.size:
            return -1
        if spot_id >= self.id_ends[idx]:
            return -1
        return int(self.ring_nrs[idx])

    def d_for_spot_id(self, spot_id: int) -> float:
        """Return the reference d-spacing for a SpotID, or ``0.0`` if missing."""
        idx = np.searchsorted(self.id_starts, spot_id, side="right") - 1
        if idx < 0 or idx >= self.ring_nrs.size:
            return 0.0
        if spot_id >= self.id_ends[idx]:
            return 0.0
        return float(self.d_spacings[idx])

    def d_for_spot_ids(self, spot_ids: np.ndarray) -> np.ndarray:
        """Vectorised lookup; returns ``0.0`` for out-of-range SpotIDs."""
        out = np.zeros(spot_ids.shape, dtype=np.float64)
        # binary-search per element using sorted starts
        idx = np.searchsorted(self.id_starts, spot_ids, side="right") - 1
        valid = (idx >= 0) & (idx < self.ring_nrs.size)
        if valid.any():
            v = idx[valid]
            in_range = spot_ids[valid] < self.id_ends[v]
            sub = np.where(valid)[0][in_range]
            out[sub] = self.d_spacings[idx[sub]]
        return out


def load_ids_hash(path: Union[str, Path]) -> IDsHash:
    """Parse ``IDsHash.csv`` into an :class:`IDsHash`.

    Lines of the form ``ring_nr id_min id_max d_spacing`` (whitespace-separated).
    """
    rings: List[int] = []
    starts: List[int] = []
    ends: List[int] = []
    ds: List[float] = []
    with open(path, "r") as f:
        for raw in f:
            tokens = raw.split()
            if len(tokens) < 4:
                continue
            try:
                rings.append(int(tokens[0]))
                starts.append(int(tokens[1]))
                ends.append(int(tokens[2]))
                ds.append(float(tokens[3]))
            except (ValueError, IndexError):
                continue
    if not rings:
        raise ValueError(f"{path} contained no parseable rows")
    rings_a = np.asarray(rings, dtype=np.int64)
    starts_a = np.asarray(starts, dtype=np.int64)
    ends_a = np.asarray(ends, dtype=np.int64)
    ds_a = np.asarray(ds, dtype=np.float64)
    order = np.argsort(starts_a)
    return IDsHash(
        ring_nrs=rings_a[order],
        id_starts=starts_a[order],
        id_ends=ends_a[order],
        d_spacings=ds_a[order],
    )


def build_ids_hash_from_inputall(
    run_dir: Union[str, Path],
    wavelength_A: float,
    ring_numbers: List[int],
    *,
    inputall_name: str = "InputAllExtraInfoFittingAll.csv",
    write: bool = True,
) -> Optional["IDsHash"]:
    """Synthesize ``IDsHash.csv`` from the per-spot ``InputAll`` table.

    The legacy C FitSetup writes ``IDsHash.csv`` (per-ring SpotID ranges +
    d-spacing); the c-omp pipeline does not. Each diffraction ring occupies a
    contiguous SpotID block, so we recover ``<ring> <id_min> <id_max> <d_A>``
    from ``InputAll`` columns SpotID(4), RingNumber(5), Ttheta(7), keeping only
    the real ``ring_numbers`` (ring 0 / 2θ=0 background is ignored). Returns the
    table (and writes ``run_dir/IDsHash.csv`` when ``write``), or ``None`` if the
    InputAll file is absent.
    """
    run_dir = Path(run_dir)
    src = run_dir / inputall_name
    if not src.exists():
        return None
    arr = np.loadtxt(src, skiprows=1, usecols=(4, 5, 7))
    if arr.ndim == 1:
        arr = arr[None, :]
    sid = arr[:, 0].astype(np.int64)
    ring = arr[:, 1].astype(np.int64)
    tth = arr[:, 2]
    keep = set(int(r) for r in ring_numbers)

    # Reference d-spacing per ring comes from the THEORETICAL ring positions
    # (hkls.csv, computed from the strain-free reference lattice), NOT the
    # observed median 2θ — which carries the sample's mean isotropic strain and
    # would zero out the diagonal of every per-spot strain tensor. This mirrors
    # the C FitSetup, whose IDsHash.csv d-spacings derive from RingRadii (the
    # reference-lattice ring radii). Columns: D-spacing(3), RingNr(4).
    ref_d_by_ring: dict = {}
    hkls_path = run_dir / "hkls.csv"
    if hkls_path.exists():
        try:
            hk = np.loadtxt(hkls_path, skiprows=1, usecols=(3, 4))
            if hk.ndim == 1:
                hk = hk[None, :]
            for d_ref, rn in hk:
                ref_d_by_ring.setdefault(int(rn), float(d_ref))
        except Exception:
            ref_d_by_ring = {}

    rings, starts, ends, ds = [], [], [], []
    deg2rad = np.pi / 180.0
    for r in sorted(keep):
        m = ring == r
        if not m.any():
            continue
        if r in ref_d_by_ring and ref_d_by_ring[r] > 0:
            d = ref_d_by_ring[r]                          # reference (strain-free)
        else:
            # Fallback only when hkls.csv is unavailable: observed median 2θ.
            tth_r = float(np.median(tth[m]))
            if tth_r <= 0:
                continue
            d = wavelength_A / (2.0 * np.sin(0.5 * tth_r * deg2rad))
        rings.append(r)
        starts.append(int(sid[m].min()))
        ends.append(int(sid[m].max()) + 1)   # exclusive upper bound
        ds.append(float(d))
    if not rings:
        return None
    order = np.argsort(starts)
    rings_a = np.asarray(rings, np.int64)[order]
    starts_a = np.asarray(starts, np.int64)[order]
    ends_a = np.asarray(ends, np.int64)[order]
    ds_a = np.asarray(ds, np.float64)[order]
    if write:
        with open(run_dir / "IDsHash.csv", "w") as f:
            for rn, s, e, d in zip(rings_a, starts_a, ends_a, ds_a):
                f.write(f"{int(rn)} {int(s)} {int(e)} {d:.6f}\n")
    return IDsHash(ring_nrs=rings_a, id_starts=starts_a, id_ends=ends_a,
                   d_spacings=ds_a)
