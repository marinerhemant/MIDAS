"""Reconstruction quality, per method: what you have, and how far to trust it.

For each reconstruction (FBP, MLEM, the direct per-voxel PBP map, ...) of one layer:

- **half_split**: reflections of each grain alternately into two halves, each reconstructed alone; the
  fraction of voxels given the same grain by both halves. A reconstruction that is reproducing data, not
  noise, gives the same map from half the data.
- **agreement_vs_pbp**: the fraction of voxels given the same grain as the per-voxel (PBP) map. PBP is a
  reference, not truth.
- **majority_null**: the fraction the most common label covers: what agreement a constant map would get.
- **orientation check**: agreement with PBP under each of the 8 flips/transposes of the grid; a
  non-identity one winning means a grid-convention error (MLEM's transpose on alumina was one).

Every number is taken on a stated **support**: the sample mask if there is one, else the voxels the PBP
map solved. Never the whole grid: the reconstructions are > 0 in vacuum, and on 20-ID-E Fe9Cr a whole-grid
half-split measured mostly vacuum (FBP 0.33 all-grid vs 0.56 on the sample).

Which reconstruction wins depends on the data: FBP beat MLEM on ESRF ma5608 alumina (half-split 0.864 vs
0.789; dense, 180 deg, 0.3 um beam); MLEM beat FBP on 20-ID-E Fe9Cr (0.77 vs 0.56 and 0.73 vs 0.57;
sparse, 360 deg with a 16-deg missing wedge, 10 um beam). So produce them all and read this report.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import numpy as np

__all__ = ["labels_from_stack", "half_rows", "agreement", "quality_entry", "DIHEDRAL"]

DIHEDRAL = [lambda a: a, lambda a: a[::-1], lambda a: a[:, ::-1], lambda a: a[::-1, ::-1],
            lambda a: a.T, lambda a: a.T[::-1], lambda a: a.T[:, ::-1], lambda a: a.T[::-1, ::-1]]


def labels_from_stack(R: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Grain label per voxel = argmax over the (n_grains, n, n) stack; -1 where no grain is > 0 or outside mask."""
    R = np.asarray(R)
    lab = np.where(R.max(0) > 0, np.argmax(R, 0), -1).astype(np.int32)
    if mask is not None:
        lab[~np.asarray(mask, bool)] = -1
    return lab


def half_rows(sinos: np.ndarray, omegas: np.ndarray, nr: np.ndarray, parity: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Keep every other reflection row of each grain (parity 0 or 1), packed to the front."""
    s2 = np.zeros_like(sinos); o2 = np.zeros_like(omegas); n2 = np.zeros_like(nr)
    for g in range(sinos.shape[0]):
        idx = np.arange(int(nr[g]))[np.arange(int(nr[g])) % 2 == parity]
        s2[g, :len(idx)] = sinos[g, idx]; o2[g, :len(idx)] = omegas[g, idx]; n2[g] = len(idx)
    return s2, o2, n2


def agreement(a: np.ndarray, b: np.ndarray, support: np.ndarray) -> float:
    """Fraction of support voxels where b is assigned and a gives the same label (a unassigned = disagree)."""
    m = np.asarray(support, bool) & (b >= 0)
    return float(np.mean(a[m] == b[m])) if m.any() else float("nan")


def _majority(lab: np.ndarray, support: np.ndarray) -> float:
    v = lab[np.asarray(support, bool) & (lab >= 0)]
    return float(np.bincount(v).max() / v.size) if v.size else float("nan")


def quality_entry(full: np.ndarray, support: np.ndarray, *, pbp: Optional[np.ndarray] = None,
                  halves: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> Dict:
    """One method's row of the report, from its label maps (full, and optionally the two halves)."""
    sup = np.asarray(support, bool)
    e = {"assigned_in_support": int((sup & (full >= 0)).sum()), "support_voxels": int(sup.sum()),
         "majority_null": _majority(full, sup)}
    if halves is not None:
        le, lo = halves; m = sup & (le >= 0) & (lo >= 0)
        e["half_split"] = float(np.mean(le[m] == lo[m])) if m.any() else float("nan")
        e["half_split_voxels"] = int(m.sum())
    if pbp is not None:
        e["agreement_vs_pbp"] = agreement(full, pbp, sup)
        per = [agreement(D(full), pbp, sup) if D(full).shape == pbp.shape else float("nan") for D in DIHEDRAL]
        e["agreement_by_transform"] = per
        k = int(np.nanargmax(per))
        e["best_transform"] = k
        e["grid_convention_ok"] = bool(k == 0 or per[k] - per[0] < 0.05)
    return e
