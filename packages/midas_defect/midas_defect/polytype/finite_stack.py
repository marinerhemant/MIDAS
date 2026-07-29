"""3-D finite close-packed slab diffraction -- the proof behind the on-axis ladder.

:func:`~midas_defect.polytype.cell_index.structure_factor_intensity` evaluates the
*infinite*-crystal structure factor of one close-packed cell. This module builds a
**real, finite Cartesian atom slab** (``n_inplane x n_inplane`` atoms per layer,
``n_layers`` close-packed layers along the c = <111> axis) and evaluates the full
3-D scattered intensity ``I(q) = |sum_j w_j exp(i q.r_j)|^2 / N^2`` at arbitrary
``q``. It is the independent, assumption-light confirmation of the two structural
claims the paper rests on:

1. **On-axis extinction.** For an *ideal* 9R (equal layer spacing, one element) the
   on-axis ``n*G/3`` satellites with ``n`` not a multiple of 3 are numerically zero
   (~1e-30) -- only the FCC fundamentals 111 (n=3) and 222 (n=6) survive on the
   axis. A finite slab adds Laue *fringes at the fundamentals*, never satellites, so
   the observed n*G/3 ladder cannot be a finite-size effect.
2. **A period-3 modulation turns the ladder ON.** Adding a period-3 interlayer-
   spacing modulation (``spacing_modulation``, fault relaxation) and/or a period-3
   composition/scattering-power modulation (``comp_modulation``) makes the on-axis
   satellites appear with intensity **proportional to (amplitude)^2** (second
   order). This is the mechanism that reconciles the extinct ideal structure with
   the measured ladder.

Because the on-axis intensity depends only on the layer *z*-positions and per-layer
weights (the in-plane offsets contribute no phase when ``q`` has no in-plane
component), the extinction result is independent of the in-plane cluster size and
of obverse/reverse choice -- it is a pure stacking-sequence property.

All lengths in angstrom, ``q`` in inverse angstrom (``q = 2*pi/d``). Pure NumPy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .cell_index import NINE_R_SEQUENCE

__all__ = [
    "FiniteStack",
    "build_close_packed_slab",
    "slab_intensity",
    "on_axis_ladder",
    "g_111",
]

#: in-plane A/B/C stacking offsets as integer multiples of the shift (a1+a2)/3.
_ABC_OFFSET = {"A": 0, "B": 1, "C": 2}


def g_111(a_fcc: float) -> float:
    """Magnitude of the FCC 111 reciprocal vector, ``|G_111| = 2*pi*sqrt(3)/a``."""
    return 2.0 * math.pi * math.sqrt(3.0) / a_fcc


@dataclass
class FiniteStack:
    """A finite close-packed atom slab.

    positions : (N, 3) Cartesian atom coordinates in angstrom (c-axis along +z).
    weights   : (N,) per-atom scattering weight (1 for a single element; modulated
                by ``comp_modulation``).
    a_fcc     : parent FCC lattice parameter (angstrom).
    axis      : unit c = <111> stacking axis in this frame (always +z here).
    """

    positions: NDArray[np.floating]
    weights: NDArray[np.floating]
    a_fcc: float
    axis: NDArray[np.floating]


def build_close_packed_slab(
    sequence: str = NINE_R_SEQUENCE,
    *,
    n_inplane: int = 26,
    n_layers: int = 36,
    a_fcc: float = 3.6356,
    spacing_modulation: float = 0.0,
    comp_modulation: float = 0.0,
) -> FiniteStack:
    """Build a finite close-packed slab, c = <111> along +z.

    ``sequence`` is repeated (cyclically) to fill ``n_layers`` layers, each an
    ``n_inplane x n_inplane`` patch of the triangular close-packed net. The
    in-plane nearest-neighbour spacing is ``a_fcc/sqrt(2)`` and the mean layer
    spacing is ``d_111 = a_fcc/sqrt(3)``.

    ``spacing_modulation`` (units of d_111) adds a period-3 height modulation
    ``z += spacing_modulation*d_111*cos(2*pi*layer/3)`` (fault relaxation).
    ``comp_modulation`` (in (-1, 1)) sets per-layer weight
    ``1 + comp_modulation*cos(2*pi*layer/3)`` (period-3 chemical ordering).
    """
    seq = sequence.upper()
    for ch in seq:
        if ch not in _ABC_OFFSET:
            raise ValueError(f"stacking letter {ch!r} not in A/B/C")
    if n_inplane < 1 or n_layers < 1:
        raise ValueError("n_inplane and n_layers must be >= 1")

    a_ip = a_fcc / math.sqrt(2.0)          # in-plane nearest-neighbour spacing
    d = a_fcc / math.sqrt(3.0)             # mean close-packed layer spacing
    a1 = np.array([a_ip, 0.0, 0.0])
    a2 = np.array([a_ip / 2.0, a_ip * math.sqrt(3.0) / 2.0, 0.0])
    shift = (a1 + a2) / 3.0

    ii, jj = np.meshgrid(np.arange(n_inplane), np.arange(n_inplane))
    ii = ii.ravel()
    jj = jj.ravel()
    inplane = ii[:, None] * a1 + jj[:, None] * a2   # (Nip, 3)

    pos = []
    wts = []
    for k in range(n_layers):
        letter = seq[k % len(seq)]
        offset = _ABC_OFFSET[letter] * shift
        z = k * d + spacing_modulation * d * math.cos(2.0 * math.pi * k / 3.0)
        layer = inplane + offset
        layer[:, 2] = z
        pos.append(layer)
        w = 1.0 + comp_modulation * math.cos(2.0 * math.pi * k / 3.0)
        wts.append(np.full(len(layer), w))

    return FiniteStack(
        positions=np.concatenate(pos, axis=0),
        weights=np.concatenate(wts, axis=0),
        a_fcc=a_fcc,
        axis=np.array([0.0, 0.0, 1.0]),
    )


def slab_intensity(stack: FiniteStack, q) -> NDArray[np.floating] | float:
    """Per-atom-normalized scattered intensity ``|sum w exp(i q.r)|^2 / N^2``.

    ``q`` is a single ``(3,)`` vector or an ``(M, 3)`` array of inverse-angstrom
    scattering vectors; returns a float or ``(M,)`` array. Normalization by ``N^2``
    makes a fundamental Bragg peak approach 1 (all atoms in phase) and an extinct
    reflection approach 0, independent of slab size.
    """
    R = stack.positions
    w = stack.weights
    q = np.asarray(q, dtype=np.float64)
    single = q.ndim == 1
    q = np.atleast_2d(q)                    # (M, 3)
    phase = R @ q.T                         # (N, M)
    amp = (w[:, None] * np.exp(1j * phase)).sum(axis=0)   # (M,)
    inten = (np.abs(amp) ** 2) / (len(R) ** 2)
    return float(inten[0]) if single else inten


def on_axis_ladder(
    orders,
    *,
    sequence: str = NINE_R_SEQUENCE,
    n_inplane: int = 26,
    n_layers: int = 36,
    a_fcc: float = 3.6356,
    spacing_modulation: float = 0.0,
    comp_modulation: float = 0.0,
) -> dict[float, float]:
    """On-axis intensity at ``q = (G_111/3) * n`` for each ``n`` in ``orders``.

    Returns ``{n: I_normalized}``. With the defaults (an ideal slab) only the
    integer-triple orders n=3 (111), n=6 (222), n=9 ... are nonzero; the
    ``n mod 3 != 0`` satellites are ~1e-30. A nonzero ``spacing_modulation`` or
    ``comp_modulation`` turns the satellites on with I ~ (amplitude)^2.
    """
    stack = build_close_packed_slab(
        sequence,
        n_inplane=n_inplane,
        n_layers=n_layers,
        a_fcc=a_fcc,
        spacing_modulation=spacing_modulation,
        comp_modulation=comp_modulation,
    )
    G = g_111(a_fcc)
    axis = stack.axis
    qs = np.array([axis * (n * G / 3.0) for n in orders])
    inten = np.atleast_1d(slab_intensity(stack, qs))
    return {float(n): float(v) for n, v in zip(orders, inten)}
