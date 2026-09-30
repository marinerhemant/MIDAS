"""Rotation utilities — thin shims over `midas-stress.orientation`.

The bulk of orientation math (axis-angle -> R, Euler <-> R, quaternions,
symmetry, FZ reduction) lives in `midas_stress.orientation`. After
midas-stress 0.5.0 (see dev/implementation_plan.md §17), those calls accept
torch tensors and return them on the active device.

The only indexer-specific item that stays here is `calc_rotation_angle`.
"""

from __future__ import annotations

import functools

import torch

from midas_stress.orientation import axis_angle_to_orient_mat


# ---------------------------------------------------------------------------
# Symmetry-scaled angular sweep for the orientation grid.
# Mirrors `CalcRotationAngle` / `LaueRotationsInt` in c_src/IndexerUnified.c;
# tests/test_rotation_angle_laue.py pins both to midas_hkls for all 230 SGs.
# ---------------------------------------------------------------------------

_ROT_ANGLE_TOL = 1e-3

_C2A = ((1, 0, 0), (0, -1, 0), (0, 0, -1))
_C2B = ((-1, 0, 0), (0, 1, 0), (0, 0, -1))
_C2C = ((-1, 0, 0), (0, -1, 0), (0, 0, 1))
_C4C = ((0, -1, 0), (1, 0, 0), (0, 0, 1))        # -y, x, z
_C3C = ((0, -1, 0), (1, -1, 0), (0, 0, 1))       # -y, x-y, z   (hex axes)
_C6C = ((1, -1, 0), (1, 0, 0), (0, 0, 1))        # x-y, x, z    (hex axes)
_C2_110 = ((0, 1, 0), (1, 0, 0), (0, 0, -1))     # y, x, -z
_C2_1M10 = ((0, -1, 0), (-1, 0, 0), (0, 0, -1))  # -y, -x, -z
_C3_111 = ((0, 0, 1), (1, 0, 0), (0, 1, 0))      # z, x, y

# -31m groups: 2-folds along <1-10>. The rest of 149-167 (incl. every R group)
# are -3m1, 2-folds along <110>.
_TRIGONAL_TYPE2 = frozenset({149, 151, 153, 157, 159, 162, 163})


def _mat_mul(a, b):
    return tuple(tuple(sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3))
                 for i in range(3))


def _close_group(gens):
    eye = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
    group = [eye]
    seen = {eye}
    grew = True
    while grew:
        grew = False
        for g in list(group):
            for s in gens:
                p = _mat_mul(g, s)
                if p not in seen:
                    seen.add(p)
                    group.append(p)
                    grew = True
                    if len(group) > 24:
                        raise AssertionError("Laue rotation group exceeds order 24")
    return group


@functools.lru_cache(maxsize=512)
def laue_rotations_int(
    space_group: int,
    abcabg: tuple[float, float, float, float, float, float] | None = None,
) -> list:
    """Proper half of the Laue group as integer matrices on direct coordinates.

    ``x' = M x``. Settings follow the lattice angles: monoclinic unique axis is
    the one non-90 angle (default b); trigonal uses hexagonal axes unless
    a = b = c and alpha = beta = gamma != 90 (rhombohedral axes). With
    ``abcabg=None`` the default settings (b-unique, hexagonal axes) are used.
    """
    sg = int(space_group)
    if not 1 <= sg <= 230:
        raise ValueError(f"invalid space group {space_group}")
    if abcabg is None:
        a = b = c = 1.0
        al, be, ga = 90.0, 90.0, (120.0 if 143 <= sg <= 194 else 90.0)
    else:
        a, b, c, al, be, ga = (float(x) for x in abcabg)
    close = lambda x, y: abs(x - y) < _ROT_ANGLE_TOL  # noqa: E731
    gens: list = []
    if sg <= 2:
        pass
    elif sg <= 15:
        if not close(al, 90) and close(be, 90) and close(ga, 90):
            gens = [_C2A]
        elif close(al, 90) and close(be, 90) and not close(ga, 90):
            gens = [_C2C]
        else:
            gens = [_C2B]
    elif sg <= 74:
        gens = [_C2C, _C2A]
    elif sg <= 88:
        gens = [_C4C]
    elif sg <= 142:
        gens = [_C4C, _C2A]
    elif sg <= 167:
        rhomb = (not close(ga, 120) and abs(a - b) <= 1e-6 * a
                 and abs(b - c) <= 1e-6 * a and close(al, be) and close(be, ga)
                 and not close(al, 90))
        gens = [_C3_111 if rhomb else _C3C]
        if sg >= 149:
            if rhomb or sg in _TRIGONAL_TYPE2:
                gens.append(_C2_1M10)
            else:
                gens.append(_C2_110)
    elif sg <= 176:
        gens = [_C6C]
    elif sg <= 194:
        gens = [_C6C, _C2_110]
    elif sg <= 206:
        gens = [_C2C, _C2A, _C3_111]
    else:
        gens = [_C4C, _C3_111]
    return _close_group(gens)


def calc_rotation_angle(
    ring_nr: int,
    space_group: int,
    hkl_int: tuple[int, int, int],
    abcabg: tuple[float, float, float, float, float, float] | None = None,
) -> float:
    """Sweep angle (degrees) about the seed plane normal: ``360 / n``.

    ``n`` is the order of the stabilizer of the reciprocal vector (hkl) in the
    proper half of the Laue group -- the number of symmetry rotations about G
    that map the crystal's predicted spots onto themselves. The orientation grid
    runs ``[0, angle)`` in steps of ``IndexerParams.StepsizeOrient``. Too small
    an angle skips orientations silently; too large only costs time.

    Parameters
    ----------
    ring_nr : int
        Unused; kept for signature compatibility with the C port.
    space_group : int
        Space group number (1-230).
    hkl_int : tuple[int, int, int]
        Integer Miller indices of the ring's ALIGNED reflection (the same row
        as the Cartesian ``hkl`` passed to the grid).
    abcabg : optional 6-tuple
        Lattice (a, b, c, alpha, beta, gamma); selects the monoclinic unique
        axis and hexagonal vs rhombohedral trigonal axes.
    """
    del ring_nr
    h = tuple(int(x) for x in hkl_int)
    if h == (0, 0, 0):
        return 0.0
    lat = None if abcabg is None else tuple(float(x) for x in abcabg)
    n = sum(1 for m in laue_rotations_int(int(space_group), lat)
            if all(sum(h[i] * m[i][j] for i in range(3)) == h[j] for j in range(3)))
    return 360.0 / max(n, 1)


# ---------------------------------------------------------------------------
# Axis-angle -> rotation matrix (batched, torch-native via midas-stress)
# ---------------------------------------------------------------------------


def axis_angle_batch(
    axes: torch.Tensor,
    angles_deg: torch.Tensor,
) -> torch.Tensor:
    """Stacked axis-angle -> rotation matrices.

    Parameters
    ----------
    axes : torch.Tensor, shape (..., 3)
    angles_deg : torch.Tensor, shape broadcastable to leading dims of `axes`

    Returns
    -------
    R : torch.Tensor, shape (..., 3, 3) on the same device/dtype as `axes`.
    """
    return axis_angle_to_orient_mat(axes, angles_deg)
