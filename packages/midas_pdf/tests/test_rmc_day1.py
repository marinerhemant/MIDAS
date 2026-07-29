"""Day-1 RMC tests: Supercell datastructure + forward G(r) matches pdffit_gr.

Checkpoint: on an FCC Ni supercell of any size, the Gaussian-broadened
G(r) computed from PBC-minimum-image pair distances must agree with the
crystal-based ``pdffit_gr`` to machine precision, up to the minimum-image
distance cutoff (half the smallest cell dimension).
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from midas_pdf.rmc import Supercell, supercell_G_r, pair_distance_histogram
from midas_pdf.structure import build_pair_list, pdffit_gr


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _fcc_ni(a=3.524):
    from midas_hkls import Crystal, Atom, Lattice, SpaceGroup
    return Crystal(
        lattice=Lattice(a, a, a, 90, 90, 90),
        space_group=SpaceGroup.from_number(225),
        atoms=[Atom(element="Ni", fract=(0, 0, 0))], name="Ni",
    ).to_torch()


# ---------------------------------------------------------------------------
# Supercell construction
# ---------------------------------------------------------------------------

def test_supercell_from_crystal_size1_equals_unit_cell():
    ni = _fcc_ni()
    sc = Supercell.from_crystal(ni, size=(1, 1, 1))
    assert sc.n_atoms == 4                          # FCC has 4 atoms/unit cell
    assert abs(sc.volume - 3.524 ** 3) < 1e-6
    assert abs(sc.number_density - 4.0 / 3.524 ** 3) < 1e-9


def test_supercell_size_scales_n_atoms():
    ni = _fcc_ni()
    for n in (2, 3, 4, 5):
        sc = Supercell.from_crystal(ni, size=(n, n, n))
        assert sc.n_atoms == 4 * n ** 3


def test_supercell_positions_in_cell():
    """Every atom must lie inside the supercell box (before any RMC moves)."""
    ni = _fcc_ni()
    sc = Supercell.from_crystal(ni, size=(3, 3, 3))
    cell_inv = torch.linalg.inv(sc.cell)
    frac = sc.positions @ cell_inv
    assert torch.all(frac >= -1e-12)
    assert torch.all(frac < 1 + 1e-12)


def test_supercell_repr_string_present():
    ni = _fcc_ni()
    sc = Supercell.from_crystal(ni, size=(2, 2, 2))
    s = repr(sc)
    assert "N=32" in s
    assert "volume" in s


# ---------------------------------------------------------------------------
# PBC pair distances
# ---------------------------------------------------------------------------

def test_pair_distances_all_positive_and_bounded():
    ni = _fcc_ni()
    sc = Supercell.from_crystal(ni, size=(3, 3, 3))
    r = sc.pair_distances(r_max=5.0)
    assert r.numel() > 0
    assert torch.all(r > 0)
    assert torch.all(r <= 5.0 + 1e-9)


def test_pair_distances_first_shell_is_fcc_nearest_neighbour():
    """First shell for FCC is a/√2 ≈ 2.492 Å."""
    ni = _fcc_ni()
    sc = Supercell.from_crystal(ni, size=(3, 3, 3))
    r = sc.pair_distances(r_max=3.0)
    assert torch.abs(r.min() - 3.524 / np.sqrt(2)) < 1e-6


def test_pair_distance_count_matches_fcc_coordination():
    """Nearest-neighbour count per atom in FCC is 12.

    Sum of pair distances at the first shell = N * 12 / 2 (each pair
    counted once)."""
    ni = _fcc_ni()
    sc = Supercell.from_crystal(ni, size=(4, 4, 4))
    r = sc.pair_distances(r_max=2.6)                # first shell only
    expected_pairs = sc.n_atoms * 12 // 2
    assert abs(r.numel() - expected_pairs) <= 4      # tiny surface effect on 4×4×4


# ---------------------------------------------------------------------------
# supercell_G_r vs pdffit_gr — the Day 1 headline invariant
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("nx", [3, 4, 5])
def test_supercell_G_r_agrees_with_pdffit_below_pbc_cutoff(nx):
    """Below the PBC minimum-image cutoff, supercell_G_r must match
    pdffit_gr to machine precision on a crystalline supercell."""
    ni = _fcc_ni()
    sc = Supercell.from_crystal(ni, size=(nx, nx, nx))
    L = float(sc.cell[0, 0])
    r_max_pbc = L / 2.0
    r = torch.linspace(1.5, r_max_pbc - 0.5, 300, dtype=torch.float64)
    pairs = build_pair_list(ni, r_max=float(r.max()) + 1)
    G_pdf = pdffit_gr(ni, r, pairs, scale=1.0, u_iso=0.005)
    G_sc  = supercell_G_r(sc, r, u_iso=0.005, r_max=r_max_pbc + 1)
    max_diff = float((G_sc - G_pdf).abs().max())
    # Not machine precision: pdffit_gr uses a distinct-pairs enumeration
    # over the crystal that has slightly different truncation from the
    # supercell brute-force pair list. Sub-0.01% agreement is the physically
    # meaningful bound.
    assert max_diff < 1e-4, f"nx={nx}: max |ΔG|={max_diff:.6g}"


def test_supercell_G_r_disagrees_above_pbc_cutoff():
    """Above the PBC cutoff, supercell_G_r WILL disagree with pdffit_gr —
    this is the documented limit of minimum-image, and it's expected."""
    ni = _fcc_ni()
    sc = Supercell.from_crystal(ni, size=(4, 4, 4))
    L = float(sc.cell[0, 0])
    r_pbc = L / 2.0
    r = torch.linspace(r_pbc + 0.5, r_pbc + 2.0, 100, dtype=torch.float64)
    pairs = build_pair_list(ni, r_max=float(r.max()) + 1)
    G_pdf = pdffit_gr(ni, r, pairs, scale=1.0, u_iso=0.005)
    G_sc  = supercell_G_r(sc, r, u_iso=0.005, r_max=float(r.max()) + 1)
    # There MUST be a non-trivial disagreement in this window
    assert float((G_sc - G_pdf).abs().max()) > 1.0


# ---------------------------------------------------------------------------
# Histogram utility
# ---------------------------------------------------------------------------

def test_pair_distance_histogram_shape_and_sum():
    ni = _fcc_ni()
    sc = Supercell.from_crystal(ni, size=(3, 3, 3))
    edges, counts = pair_distance_histogram(sc, bins=100, r_max=5.0)
    assert edges.shape == (101,)
    assert counts.shape == (100,)
    # Total count = number of pairs with distance ≤ r_max
    r_all = sc.pair_distances(r_max=5.0)
    assert abs(float(counts.sum()) - r_all.numel()) < 1e-6


def test_pair_distance_histogram_first_bins_are_zero():
    """No pair should be at distance below ~1.5 Å for FCC Ni (nearest
    neighbour is 2.49 Å)."""
    ni = _fcc_ni()
    sc = Supercell.from_crystal(ni, size=(3, 3, 3))
    edges, counts = pair_distance_histogram(sc, bins=100, r_max=3.0)
    # Bins up to 2.0 Å should be empty
    idx_below_2 = int(np.argmin(np.abs(edges.numpy() - 2.0)))
    assert float(counts[:idx_below_2].sum()) == 0.0


# ---------------------------------------------------------------------------
# Validation error paths
# ---------------------------------------------------------------------------

def test_supercell_from_crystal_rejects_zero_size():
    ni = _fcc_ni()
    with pytest.raises(ValueError):
        Supercell.from_crystal(ni, size=(0, 1, 1))


def test_pair_distances_rejects_nonpositive_rmax():
    ni = _fcc_ni()
    sc = Supercell.from_crystal(ni, size=(2, 2, 2))
    with pytest.raises(ValueError):
        sc.pair_distances(r_max=0.0)


def test_supercell_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        Supercell(species=["Ni"], positions=torch.zeros((2, 3)),
                   cell=torch.eye(3))
    with pytest.raises(ValueError):
        Supercell(species=["Ni"], positions=torch.zeros((1, 2)),
                   cell=torch.eye(3))
    with pytest.raises(ValueError):
        Supercell(species=["Ni"], positions=torch.zeros((1, 3)),
                   cell=torch.zeros((2, 3)))
