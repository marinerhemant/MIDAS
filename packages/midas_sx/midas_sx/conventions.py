"""Image-orientation and rotation-sense conventions for a frame series whose header does not fix them.

A detector array can sit in the MIDAS lab frame (rows along +Z = the rotation axis, columns along -Y, beam along +X, ``q_lab = R_z(+omega) q_sample``)
in eight ways (the dihedral group of the square) and the scan can turn either way, so 16 candidates. :func:`candidate_conventions` lists them;
:func:`transform_rc` maps raw array coordinates (pixel centres at integers) to the MIDAS array coordinates of one candidate. The choice is made by the
INDEXING score, not by the header (:func:`score_conventions` ranks them and refuses a pick whose margin is too small).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Sequence, Tuple

import numpy as np

__all__ = ["Convention", "DIHEDRAL_NAMES", "candidate_conventions", "transform_rc", "transform_frame", "rank_conventions"]

DIHEDRAL_NAMES = ("identity", "flip_cols", "flip_rows", "rot180", "transpose", "transpose_flip_cols", "transpose_flip_rows", "anti_transpose")


@dataclass(frozen=True)
class Convention:
    dihedral: int          # index into DIHEDRAL_NAMES
    omega_sign: int        # +1: omega = +phi, -1: omega = -phi

    @property
    def name(self) -> str:
        return f"{DIHEDRAL_NAMES[self.dihedral]}|omega{'+' if self.omega_sign > 0 else '-'}"


def candidate_conventions() -> List[Convention]:
    return [Convention(k, s) for k in range(8) for s in (+1, -1)]


def transform_rc(row, col, n_rows: int, n_cols: int, dihedral: int):
    """Raw (row, col) -> (row', col') in the transformed array. For the transposing members the output array has shape (n_cols, n_rows)."""
    r, c = np.asarray(row, float), np.asarray(col, float)
    R, C = n_rows - 1.0, n_cols - 1.0
    table = (lambda: (r, c), lambda: (r, C - c), lambda: (R - r, c), lambda: (R - r, C - c),
             lambda: (c, r), lambda: (c, R - r), lambda: (C - c, r), lambda: (C - c, R - r))
    return table[dihedral]()


def transform_frame(a: np.ndarray, dihedral: int) -> np.ndarray:
    """The same transform applied to an array (last two axes): consistent with :func:`transform_rc` by construction (tested)."""
    a = np.asarray(a)
    ops = (lambda x: x, lambda x: x[..., ::-1], lambda x: x[..., ::-1, :], lambda x: x[..., ::-1, ::-1],
           lambda x: np.swapaxes(x, -1, -2), lambda x: np.swapaxes(x, -1, -2)[..., :, ::-1],
           lambda x: np.swapaxes(x, -1, -2)[..., ::-1, :], lambda x: np.swapaxes(x, -1, -2)[..., ::-1, ::-1])
    return ops[dihedral](a)


def rank_conventions(scores: Sequence[Tuple[Convention, float]], *, min_margin: float = 0.20):
    """Rank (convention, score) pairs, higher is better. Returns ``(best, runner_up_score, resolved)``; resolved is True only when the best beats the
    runner-up by at least ``min_margin`` (relative). A tie is reported as not resolved, never broken."""
    order = sorted(scores, key=lambda t: -t[1])
    best, s0 = order[0]
    s1 = order[1][1] if len(order) > 1 else 0.0
    resolved = bool(s0 > 0 and (s0 - s1) >= min_margin * s0)
    return best, s1, resolved
