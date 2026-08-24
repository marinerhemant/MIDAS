"""|F|^2 reaches FF/PF the same way it reaches NF, from one implementation.

Three properties, each of which has already caused a real problem somewhere:

1. **One definition.** NF grew an ``F2`` column first; FF/PF is getting the
   same one. Two copies of "normalised |F|^2" would drift, and the drift would
   show up as two techniques disagreeing about a phase's achievable
   completeness rather than as an error.

2. **Silence by default.** With no atom basis the FF/PF ``hkls.csv`` must be
   byte-identical to the historical 11-column file. The column is opt-in
   because completeness gates which grains exist.

3. **The weight has to be non-uniform to prove anything.** An all-ones weight
   reproduces the unweighted answer exactly, so a test that only ever sees a
   monatomic cubic cell cannot distinguish "working" from "not wired up" --
   the exact failure recorded at
   ``midas_nf_fitorientation/screen.py:330-334``.
"""
from __future__ import annotations

import numpy as np
import pytest

from midas_hkls import Atom, Lattice, SpaceGroup, parse_phase_atoms
from midas_hkls.structure_factor import f2_normalised


# --------------------------------------------------------------- the parser

def test_parse_phase_atoms_defaults():
    atoms = parse_phase_atoms([
        "Ni 0 0 0",
        "Ni 0.5 0.5 0.0 0.8",
        "Ni 0.5 0.0 0.5 0.9 0.42",
    ])
    assert [a.element for a in atoms] == ["Ni", "Ni", "Ni"]
    assert atoms[0].occupancy == 1.0 and atoms[0].B_iso == 0.0
    assert atoms[1].occupancy == 0.8 and atoms[1].B_iso == 0.0
    assert atoms[2].occupancy == 0.9 and atoms[2].B_iso == 0.42
    assert atoms[2].fract == (0.5, 0.0, 0.5)


def test_parse_phase_atoms_none_when_empty():
    """None, not [] — it is what keeps hkls.csv byte-identical."""
    assert parse_phase_atoms([]) is None
    assert parse_phase_atoms(None) is None


def test_parse_phase_atoms_rejects_short_line():
    """Defaulting a malformed row would put the atom at the origin and produce
    a perfectly plausible F2 column for the wrong structure."""
    with pytest.raises(ValueError, match="PhaseAtom needs"):
        parse_phase_atoms(["Ni 0 0"])


# ------------------------------------------------------------------ the |F|^2

def test_fcc_has_no_basis_forbidden_reflections():
    """One atom at the origin: nothing extinct beyond the space group's own
    rules, so the ceiling is 1.000."""
    lat = Lattice(3.6, 3.6, 3.6, 90, 90, 90)
    sg = SpaceGroup.from_number(225)
    hkl = np.array([[1, 1, 1], [2, 0, 0], [2, 2, 0], [3, 1, 1]])
    f2 = f2_normalised(lat, sg, [Atom("Ni", (0.0, 0.0, 0.0))], hkl)
    assert f2.shape == (4,)
    assert f2.max() == pytest.approx(1.0)
    assert (f2 > 1e-6).all(), "fcc should have no basis-forbidden reflections"


def test_a_two_site_basis_extinguishes_some_reflections():
    """The case the whole feature exists for.

    A diamond-structure cell (two sites, 000 and 1/4 1/4 1/4) extinguishes
    (200) by destructive interference between the two sites -- a *basis*
    extinction the F-centring rules alone do not predict. If this returns all
    non-zero, the atom basis is not reaching the structure factor.
    """
    lat = Lattice(5.431, 5.431, 5.431, 90, 90, 90)
    sg = SpaceGroup.from_number(227)
    atoms = [Atom("Si", (0.0, 0.0, 0.0)), Atom("Si", (0.25, 0.25, 0.25))]
    hkl = np.array([[1, 1, 1], [2, 2, 0], [2, 0, 0]])
    f2 = f2_normalised(lat, sg, atoms, hkl)
    assert f2[0] > 1e-6 and f2[1] > 1e-6, "111 and 220 are allowed"
    assert f2[2] < 1e-6, (
        f"(200) must be extinguished by the two-site basis; got {f2[2]:.3e}"
    )


def test_normalisation_is_scale_free():
    """Doubling the cell content must not move the weights."""
    lat = Lattice(3.6, 3.6, 3.6, 90, 90, 90)
    sg = SpaceGroup.from_number(225)
    hkl = np.array([[1, 1, 1], [2, 0, 0], [3, 1, 1]])
    a = f2_normalised(lat, sg, [Atom("Ni", (0.0, 0.0, 0.0), occupancy=1.0)], hkl)
    b = f2_normalised(lat, sg, [Atom("Ni", (0.0, 0.0, 0.0), occupancy=0.5)], hkl)
    np.testing.assert_allclose(a, b, rtol=1e-12)


def test_weights_are_non_uniform_for_a_real_structure():
    """Guard against the all-ones trap: if every weight is 1 the weighted and
    unweighted metrics agree and the test proves nothing."""
    lat = Lattice(3.6, 3.6, 3.6, 90, 90, 90)
    sg = SpaceGroup.from_number(225)
    hkl = np.array([[1, 1, 1], [2, 0, 0], [2, 2, 0], [3, 1, 1], [4, 2, 0]])
    f2 = f2_normalised(lat, sg, [Atom("Ni", (0.0, 0.0, 0.0))], hkl)
    assert f2.std() > 0.01, (
        f"weights are effectively uniform (std {f2.std():.2e}); a weighted "
        "metric built on these cannot be distinguished from the unweighted one"
    )


# ------------------------------------------------- NF and FF share the code

def test_nf_writer_uses_the_shared_function(tmp_path):
    """The NF emitter's F2 column must equal a direct f2_normalised call."""
    from midas_hkls import write_nf_hkls_csv

    lat = Lattice(5.431, 5.431, 5.431, 90, 90, 90)
    sg = SpaceGroup.from_number(227)
    atoms = [Atom("Si", (0.0, 0.0, 0.0)), Atom("Si", (0.25, 0.25, 0.25))]
    out = tmp_path / "hkls.csv"
    write_nf_hkls_csv(out, sg, lat, wavelength_A=0.2066,
                      lsd_um=1_000_000.0, max_ring_rad_um=200_000.0,
                      atoms=atoms)
    lines = out.read_text().splitlines()
    assert lines[0].split()[-1] == "F2"
    body = [l.split() for l in lines[1:] if l.strip()]
    hkl = np.array([[int(float(t[0])), int(float(t[1])), int(float(t[2]))]
                    for t in body])
    got = np.array([float(t[11]) for t in body])
    want = f2_normalised(lat, sg, atoms, hkl)
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=0)
    assert (got < 1e-6).any(), "this structure must have extinct reflections"


def test_nf_writer_omits_the_column_without_a_basis(tmp_path):
    from midas_hkls import write_nf_hkls_csv

    lat = Lattice(3.6, 3.6, 3.6, 90, 90, 90)
    out = tmp_path / "hkls.csv"
    write_nf_hkls_csv(out, SpaceGroup.from_number(225), lat,
                      wavelength_A=0.2066, lsd_um=1_000_000.0,
                      max_ring_rad_um=200_000.0, atoms=None)
    lines = out.read_text().splitlines()
    assert lines[0].split()[-1] == "Radius"
    assert all(len(l.split()) == 11 for l in lines[1:] if l.strip())
