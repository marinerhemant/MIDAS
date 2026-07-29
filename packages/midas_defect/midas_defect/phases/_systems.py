"""Slip-system enumeration helpers shared by cubic phase tables.

A "system family" {hkl}<uvw> in a cubic crystal is the set of (n_hat, b_hat)
unit-vector pairs (in the crystal Cartesian frame) where:

  * n is a member of the {hkl} permutation+sign family,
  * b is a member of the <uvw> permutation+sign family,
  * n . b = 0,
  * n is identified with -n and b with -b (Burgers vector sense is not part
    of the slip-system definition; flipped pairs are duplicates).

This module generates those pairs from the two Miller-index families.
"""

from __future__ import annotations

from itertools import product

import numpy as np
from numpy.typing import NDArray


def _normalize_and_dedup(rows: list[tuple[tuple[int, ...], tuple[int, ...]]]):
    seen: set[tuple[tuple[float, ...], tuple[float, ...]]] = set()
    out: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    for n_int, b_int in rows:
        n = np.array(n_int, dtype=float)
        b = np.array(b_int, dtype=float)
        n /= np.linalg.norm(n)
        b /= np.linalg.norm(b)

        # canonical sign: leading nonzero positive
        for v in (n, b):
            for x in v:
                if abs(x) > 1e-12:
                    if x < 0:
                        v *= -1.0
                    break

        key = (tuple(np.round(n, 6)), tuple(np.round(b, 6)))
        if key in seen:
            continue
        seen.add(key)
        out.append((n_int, b_int))
    return out


def _family_members(base: tuple[int, ...]) -> list[tuple[int, ...]]:
    """All distinct permutations+sign-flips of ``base`` Miller indices."""
    from itertools import permutations

    members: set[tuple[int, ...]] = set()
    for perm in permutations(base):
        for signs in product((1, -1), repeat=len(perm)):
            v = tuple(s * p for s, p in zip(signs, perm))
            members.add(v)
    return sorted(members)


def cubic_systems(
    plane_family: tuple[int, ...],
    direction_family: tuple[int, ...],
) -> NDArray[np.floating]:
    """Generate {hkl}<uvw> slip systems for a cubic crystal.

    Parameters
    ----------
    plane_family, direction_family
        Miller-index "type" tuples, e.g. ``(1, 1, 1)`` for {111} and
        ``(1, 1, 0)`` for <110>. Order within the tuple is irrelevant.

    Returns
    -------
    (n_sys, 2, 3) float64 array
        ``out[i, 0]`` is the unit plane-normal of system *i*,
        ``out[i, 1]`` is the unit slip-direction (or twin-shear direction).
        ``n_hat . b_hat = 0`` for every row.

    Notes
    -----
    Per-system sign is canonicalized so flipped duplicates (n, b) <-> (-n, -b)
    are removed. Counts:
      * cubic_systems((1,1,1), (1,1,0))  -> 12  (FCC {111}<110>)
      * cubic_systems((1,1,1), (1,1,2))  -> 12  (FCC twin / partial)
      * cubic_systems((1,1,0), (1,1,1))  -> 12  (BCC {110}<111> primary)
      * cubic_systems((1,1,2), (1,1,1))  -> 12  (BCC {112}<111> + BCC twin)
    """
    raw: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    for n_int in _family_members(plane_family):
        for b_int in _family_members(direction_family):
            if sum(n * b for n, b in zip(n_int, b_int)) == 0:
                raw.append((n_int, b_int))
    rows = _normalize_and_dedup(raw)

    out = np.empty((len(rows), 2, 3), dtype=float)
    for i, (n_int, b_int) in enumerate(rows):
        n = np.array(n_int, dtype=float)
        b = np.array(b_int, dtype=float)
        out[i, 0] = n / np.linalg.norm(n)
        out[i, 1] = b / np.linalg.norm(b)
    return out


__all__ = ["cubic_systems"]
