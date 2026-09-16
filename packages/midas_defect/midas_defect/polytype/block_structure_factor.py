"""``|F_block(L)|**2`` along ``(0, 0, L)`` for one repeating STACKING BLOCK, at continuous L.

For a rod along ``(0, 0, L)`` built from identical, randomly-stacked repeating blocks (a
Hendricks-Teller model, see :mod:`.gap_fraction`), the diffuse density is
``|F_block(L)|**2 * HT(L)``, and the Bragg nodes carry the same ``|F_block|**2`` at every even L.
That block structure factor is NOT flat across a period -- it can vary by orders of magnitude
between neighbouring even nodes, since it is sampling the SAME continuous function
``|F_block(L)|**2`` that ordinarily only shows through its values at integer/allowed L. A gap
fraction computed from raw counts therefore mixes this real crystallographic variation in with
whatever the faulting itself does; :func:`gap_fraction.g_from_bins` divides each L-bin by
``block_f2`` at that bin's centre to remove it before comparing to a Hendricks-Teller model.

The block is the slab of unit-cell atoms (all symmetry and centring images, from
``midas_hkls.Crystal.unit_cell_atoms``) with fractional z in ``[slab_lo, slab_lo + 1/2)`` --
i.e. HALF the cell along c, split at ``slab_lo`` and ``slab_lo + 1/2``. In-plane coordinates give
no phase for ``(0, 0, L)``. Form factors: ``midas_hkls.form_factors.form_factor`` (Cromer-Mann,
``s**2 = L**2 / (4*c**2)``); isotropic Debye-Waller ``exp(-B*s**2)`` with ``B = 8*pi**2*U_iso``
when a CIF gives anisotropic ``U`` instead of ``B_iso`` directly.

Only the SHAPE of ``|F_block(L)|**2`` across a period matters for a gap fraction -- the absolute
scale drops out of a ratio.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from midas_hkls.form_factors import form_factor
from midas_hkls.io.cif import read_cif

__all__ = ["slab_atoms", "block_f2", "cell_f2_00L"]


def slab_atoms(cif_path, slab_lo: float = -0.25):
    """The unit-cell atoms with fractional z in ``[slab_lo, slab_lo + 1/2)``, as
    ``(element, z, occupancy, B_iso)`` tuples. Returns ``(crystal, atoms)``."""
    cr = read_cif(Path(cif_path))
    out = []
    for at in cr.unit_cell_atoms():
        z = (float(at.fract[2]) - slab_lo) % 1.0 + slab_lo
        if slab_lo <= z < slab_lo + 0.5 - 1e-9:
            B = float(at.B_iso)
            if B == 0.0 and at.U_aniso is not None:
                B = 8.0 * math.pi ** 2 * float(np.mean(at.U_aniso[:3]))
            out.append((at.element, z, float(at.occupancy), B))
    return cr, out


def block_f2(cif_path, L, *, c_A: float, slab_lo: float = -0.25):
    """``|F_block(L)|**2`` at the given L values, for a cell with c = ``c_A`` (the fractional z
    values come from the CIF; only ``s**2`` uses ``c_A``, so a caller can supply a refined c
    without re-refining the CIF's own atomic positions)."""
    _, atoms = slab_atoms(cif_path, slab_lo)
    L = np.asarray(L, float)
    s2 = L ** 2 / (4.0 * c_A ** 2)
    F = np.zeros(L.shape, complex)
    for el, z, occ, B in atoms:
        F += occ * form_factor(s2, el) * np.exp(-B * s2) * np.exp(2j * math.pi * L * z)
    return np.abs(F) ** 2


def cell_f2_00L(cif_path, L_even, *, c_A: float):
    """``|F_cell(0,0,L)|**2`` from ALL unit-cell atoms (no slab split), for cross-checks against
    an independent structure-factor calculator at even (allowed) L."""
    cr = read_cif(Path(cif_path))
    L = np.asarray(L_even, float)
    s2 = L ** 2 / (4.0 * c_A ** 2)
    F = np.zeros(L.shape, complex)
    for at in cr.unit_cell_atoms():
        B = float(at.B_iso)
        if B == 0.0 and at.U_aniso is not None:
            B = 8.0 * math.pi ** 2 * float(np.mean(at.U_aniso[:3]))
        F += float(at.occupancy) * form_factor(s2, at.element) * np.exp(-B * s2) \
            * np.exp(2j * math.pi * L * float(at.fract[2]))
    return np.abs(F) ** 2
