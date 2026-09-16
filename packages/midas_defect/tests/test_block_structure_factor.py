"""Tests for polytype/block_structure_factor.py, on a small SYNTHETIC I4/mmm cell (not a real
deposited structure -- avoids bundling third-party crystallographic data with the package).
Two Wyckoff sites, chosen so the slab (half the cell) and its I-centring image reconstruct the
whole cell exactly, the same body-centred-lattice property real I4/mmm La3Ni2O7 has.

A wrong slab (atoms missing or double-counted), a wrong s**2, or a wrong phase sign fails here:
the slab and its I-centring image must rebuild the full cell, and the odd-L (0,0,L) must vanish
for the whole cell while the block itself has no such symmetry restriction.
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pytest
import torch

from midas_defect.polytype.block_structure_factor import block_f2, cell_f2_00L, slab_atoms
from midas_hkls.crystal_torch import crystal_to_tensor
from midas_hkls.io.cif import read_cif
from midas_hkls.structure_factor import structure_factor_intensity, structure_factors

CIF_TEXT = """\
data_synthetic_I4mmm
_symmetry_space_group_name_H-M   'I 4/m m m'
_cell_length_a   4.0000
_cell_length_b   4.0000
_cell_length_c   12.0000
_cell_angle_alpha   90.00000000
_cell_angle_beta   90.00000000
_cell_angle_gamma   90.00000000
loop_
 _atom_site_label
 _atom_site_type_symbol
 _atom_site_fract_x
 _atom_site_fract_y
 _atom_site_fract_z
 _atom_site_occupancy
 _atom_site_adp_type
 _atom_site_U_iso_or_equiv
 _atom_site_site_symmetry_multiplicity
A1 Fe 0.00000 0.00000 0.10000 1.0000 Uiso 0.0100 4
A2 O 0.00000 0.00000 0.00000 1.0000 Uiso 0.0100 2
"""


@pytest.fixture(scope="module")
def cif_path(tmp_path_factory):
    p = tmp_path_factory.mktemp("cif") / "synthetic_I4mmm.cif"
    p.write_text(CIF_TEXT)
    return p


def test_cell_f2_matches_midas_hkls_structure_factors(cif_path):
    cr = read_cif(cif_path)
    ct = crystal_to_tensor(cr, dtype=torch.float64)
    L = np.arange(2, 21, 2)
    ref = structure_factor_intensity(structure_factors(ct, [[0, 0, int(l)] for l in L])).numpy()
    ours = cell_f2_00L(cif_path, L, c_A=cr.lattice.c)
    assert ours == pytest.approx(ref, rel=2e-3)


def test_slab_plus_centring_image_is_the_whole_cell(cif_path):
    cr = read_cif(cif_path)
    _, atoms = slab_atoms(cif_path)
    assert 2 * len(atoms) == len(cr.unit_cell_atoms())
    L = np.arange(2, 21, 2)
    assert 4 * block_f2(cif_path, L, c_A=cr.lattice.c) == pytest.approx(cell_f2_00L(cif_path, L, c_A=cr.lattice.c),
                                                                        rel=1e-9)


def test_odd_L_is_forbidden_in_the_cell_not_in_the_block(cif_path):
    cr = read_cif(cif_path)
    L = np.arange(1, 20, 2)
    cell = cell_f2_00L(cif_path, L, c_A=cr.lattice.c)
    block = block_f2(cif_path, L, c_A=cr.lattice.c)
    assert np.all(cell < 1e-6 * block.max())
    assert np.all(block > 0)
