"""Rev-9 tests: minimal CIF reader (parse + hydrate to Crystal)."""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
import torch

from midas_pdf.cif import parse_cif, read_cif_to_crystal, _strip_esd


CIF_FCC_NI = textwrap.dedent("""\
    # FCC Ni test CIF
    data_FCC_Ni
    _cell_length_a       3.524(1)
    _cell_length_b       3.524
    _cell_length_c       3.524
    _cell_angle_alpha    90
    _cell_angle_beta     90
    _cell_angle_gamma    90
    _symmetry_Int_Tables_number   225

    loop_
    _atom_site_label
    _atom_site_type_symbol
    _atom_site_fract_x
    _atom_site_fract_y
    _atom_site_fract_z
    Ni1  Ni  0.0  0.0  0.0
""")


CIF_CEO2 = textwrap.dedent("""\
    data_CeO2
    _cell_length_a       5.41165
    _cell_length_b       5.41165
    _cell_length_c       5.41165
    _cell_angle_alpha    90.000
    _cell_angle_beta     90.000
    _cell_angle_gamma    90.000
    _space_group_IT_number   225

    loop_
    _atom_site_type_symbol
    _atom_site_fract_x
    _atom_site_fract_y
    _atom_site_fract_z
    Ce   0.000   0.000   0.000
    O    0.250   0.250   0.250
""")


CIF_MISSING_SG = textwrap.dedent("""\
    data_bad
    _cell_length_a       3.5
    _cell_length_b       3.5
    _cell_length_c       3.5
    _cell_angle_alpha    90
    _cell_angle_beta     90
    _cell_angle_gamma    90

    loop_
    _atom_site_type_symbol
    _atom_site_fract_x
    _atom_site_fract_y
    _atom_site_fract_z
    Ni   0.0   0.0   0.0
""")


# ---------------------------------------------------------------------------
# ESD stripping
# ---------------------------------------------------------------------------

def test_strip_esd_basic():
    assert _strip_esd("3.524(1)") == "3.524"
    assert _strip_esd("3.524") == "3.524"
    assert _strip_esd("-3.5(2)") == "-3.5"
    assert _strip_esd("1.2e-5") == "1.2e-5"
    assert _strip_esd("1.2e-5(3)") == "1.2e-5"


# ---------------------------------------------------------------------------
# parse_cif
# ---------------------------------------------------------------------------

def _write_cif(tmp_path: Path, contents: str) -> Path:
    path = tmp_path / "test.cif"
    path.write_text(contents)
    return path


def test_parse_cif_returns_all_keys(tmp_path):
    path = _write_cif(tmp_path, CIF_FCC_NI)
    cif = parse_cif(path)
    for k in ("_cell_length_a", "_cell_length_b", "_cell_length_c",
              "_cell_angle_alpha", "_cell_angle_beta", "_cell_angle_gamma",
              "_symmetry_int_tables_number"):
        assert k in cif.keys


def test_parse_cif_finds_atom_loop(tmp_path):
    path = _write_cif(tmp_path, CIF_FCC_NI)
    cif = parse_cif(path)
    assert len(cif.loops) == 1
    loop_keys, rows = cif.loops[0]
    assert "_atom_site_fract_x" in loop_keys
    assert len(rows) == 1
    assert rows[0][0] == "Ni1"


def test_parse_cif_ignores_comments(tmp_path):
    path = _write_cif(tmp_path, CIF_FCC_NI)
    cif = parse_cif(path)
    # The "# FCC Ni test CIF" comment should not appear anywhere
    for k, v in cif.keys.items():
        assert "#" not in v


# ---------------------------------------------------------------------------
# read_cif_to_crystal
# ---------------------------------------------------------------------------

def test_read_cif_fcc_ni(tmp_path):
    path = _write_cif(tmp_path, CIF_FCC_NI)
    ni = read_cif_to_crystal(path)
    lat = ni.lattice
    assert abs(lat.a - 3.524) < 1e-6
    assert abs(lat.b - 3.524) < 1e-6
    assert abs(lat.c - 3.524) < 1e-6
    assert lat.alpha == pytest.approx(90.0)
    assert ni.space_group.number == 225
    assert len(ni.atoms) == 1
    assert ni.atoms[0].element == "Ni"


def test_read_cif_ceo2(tmp_path):
    path = _write_cif(tmp_path, CIF_CEO2)
    ce = read_cif_to_crystal(path)
    lat = ce.lattice
    assert abs(lat.a - 5.41165) < 1e-6
    assert ce.space_group.number == 225
    assert len(ce.atoms) == 2
    elements = sorted(a.element for a in ce.atoms)
    assert elements == ["Ce", "O"]


def test_read_cif_missing_space_group_raises(tmp_path):
    path = _write_cif(tmp_path, CIF_MISSING_SG)
    with pytest.raises(ValueError, match="space-group"):
        read_cif_to_crystal(path)


def test_read_cif_missing_cell_raises(tmp_path):
    truncated = "\n".join([ln for ln in CIF_FCC_NI.splitlines()
                            if not ln.startswith("_cell_length_a")])
    path = _write_cif(tmp_path, truncated)
    with pytest.raises(ValueError, match="_cell_length_a"):
        read_cif_to_crystal(path)


CIF_HM_STANDARD = textwrap.dedent("""\
    data_HM_std
    _cell_length_a       3.524
    _cell_length_b       3.524
    _cell_length_c       3.524
    _cell_angle_alpha    90
    _cell_angle_beta     90
    _cell_angle_gamma    90
    _symmetry_space_group_name_H-M   'Fm-3m'

    loop_
    _atom_site_type_symbol
    _atom_site_fract_x
    _atom_site_fract_y
    _atom_site_fract_z
    Ni 0.0 0.0 0.0
""")


CIF_HM_SHORTHAND = textwrap.dedent("""\
    data_HM_short
    _cell_length_a       3.524
    _cell_length_b       3.524
    _cell_length_c       3.524
    _cell_angle_alpha    90
    _cell_angle_beta     90
    _cell_angle_gamma    90
    _symmetry_space_group_name_H-M   'Fm3m'

    loop_
    _atom_site_type_symbol
    _atom_site_fract_x
    _atom_site_fract_y
    _atom_site_fract_z
    Ni 0.0 0.0 0.0
""")


CIF_OCC_BISO = textwrap.dedent("""\
    data_occ_biso
    _cell_length_a       5.41165
    _cell_length_b       5.41165
    _cell_length_c       5.41165
    _cell_angle_alpha    90
    _cell_angle_beta     90
    _cell_angle_gamma    90
    _symmetry_Int_Tables_number   225

    loop_
    _atom_site_type_symbol
    _atom_site_fract_x
    _atom_site_fract_y
    _atom_site_fract_z
    _atom_site_occupancy
    _atom_site_B_iso_or_equiv
    Ce 0.0 0.0 0.0 0.95 0.5
    O  0.25 0.25 0.25 0.98 0.8
""")


# ---------------------------------------------------------------------------
# H-M symbol resolution via midas-hkls (Rev 9 refactor)
# ---------------------------------------------------------------------------

def test_read_cif_via_hm_symbol_standard(tmp_path):
    """CIF with H-M symbol (no IT number) must resolve via midas-hkls."""
    path = _write_cif(tmp_path, CIF_HM_STANDARD)
    ni = read_cif_to_crystal(path)
    assert ni.space_group.number == 225


def test_read_cif_via_hm_symbol_shorthand(tmp_path):
    """Common shorthand 'Fm3m' (no bar) must fall through to 'Fm-3m'."""
    path = _write_cif(tmp_path, CIF_HM_SHORTHAND)
    ni = read_cif_to_crystal(path)
    assert ni.space_group.number == 225


def test_normalise_hm_shorthand_inserts_bars():
    from midas_pdf.cif import _normalise_hm_shorthand
    assert _normalise_hm_shorthand("Fm3m") == "Fm-3m"
    assert _normalise_hm_shorthand("Pm3n") == "Pm-3n"
    assert _normalise_hm_shorthand("Fd3") == "Fd-3"
    # Already has a bar - unchanged
    assert _normalise_hm_shorthand("Fm-3m") == "Fm-3m"


def test_normalise_hm_shorthand_covers_all_centro_cubic_letters():
    """Widened alphabet: the letter before 3 in centrosymmetric cubic H-M
    can be m / n / d / a / b — all should get an inserted bar."""
    from midas_pdf.cif import _normalise_hm_shorthand
    # SG 205 Pa-3, SG 206 Ia-3, SG 230 Ia-3d all use 'a'
    assert _normalise_hm_shorthand("Pa3") == "Pa-3"
    assert _normalise_hm_shorthand("Ia3") == "Ia-3"
    assert _normalise_hm_shorthand("Ia3d") == "Ia-3d"


def test_normalise_hm_shorthand_space_separated():
    """Space-separated shorthand (F m 3 m) must also get a bar."""
    from midas_pdf.cif import _normalise_hm_shorthand
    assert _normalise_hm_shorthand("F m 3 m") == "F m -3 m"
    assert _normalise_hm_shorthand("I m 3 m") == "I m -3 m"


@pytest.mark.parametrize("shorthand,expected_sg", [
    # Every centrosymmetric cubic space group (SG 200-230, -3-containing)
    ("Pm3", 200), ("Pn3", 201), ("Fm3", 202), ("Fd3", 203),
    ("Im3", 204), ("Pa3", 205), ("Ia3", 206),
    ("Pm3m", 221), ("Pn3n", 222), ("Pm3n", 223), ("Pn3m", 224),
    ("Fm3m", 225), ("Fm3c", 226), ("Fd3m", 227), ("Fd3c", 228),
    ("Im3m", 229), ("Ia3d", 230),
])
def test_cif_resolves_every_centro_cubic_shorthand(tmp_path, shorthand, expected_sg):
    """Every centrosymmetric cubic SG (200-230) resolves via the shorthand
    normaliser + midas-hkls.from_hm."""
    cif = textwrap.dedent(f"""\
        data_test
        _cell_length_a       5.0
        _cell_length_b       5.0
        _cell_length_c       5.0
        _cell_angle_alpha    90
        _cell_angle_beta     90
        _cell_angle_gamma    90
        _symmetry_space_group_name_H-M   '{shorthand}'

        loop_
        _atom_site_type_symbol
        _atom_site_fract_x
        _atom_site_fract_y
        _atom_site_fract_z
        X 0.0 0.0 0.0
    """)
    path = _write_cif(tmp_path, cif)
    crystal = read_cif_to_crystal(path)
    assert crystal.space_group.number == expected_sg


# ---------------------------------------------------------------------------
# Occupancy + B_iso pass-through
# ---------------------------------------------------------------------------

def test_read_cif_partial_occupancy_and_B_iso(tmp_path):
    path = _write_cif(tmp_path, CIF_OCC_BISO)
    ce = read_cif_to_crystal(path)
    assert len(ce.atoms) == 2
    ce_atom = next(a for a in ce.atoms if a.element == "Ce")
    o_atom = next(a for a in ce.atoms if a.element == "O")
    assert abs(ce_atom.occupancy - 0.95) < 1e-9
    assert abs(o_atom.occupancy - 0.98) < 1e-9
    assert abs(ce_atom.B_iso - 0.5) < 1e-9
    assert abs(o_atom.B_iso - 0.8) < 1e-9


def test_read_cif_default_occupancy_when_missing(tmp_path):
    """When _atom_site_occupancy is absent, Atom.occupancy defaults to 1.0."""
    path = _write_cif(tmp_path, CIF_FCC_NI)
    ni = read_cif_to_crystal(path)
    assert ni.atoms[0].occupancy == 1.0
    assert ni.atoms[0].B_iso == 0.0


# ---------------------------------------------------------------------------
# CIF writer round-trip (Rev 10)
# ---------------------------------------------------------------------------

def test_write_crystal_round_trip_fcc_ni(tmp_path):
    """Crystal → CIF → Crystal recovers cell + space group + atoms."""
    from midas_hkls import Crystal, Atom, Lattice, SpaceGroup
    from midas_pdf.cif import write_crystal_to_cif, read_cif_to_crystal
    ni = Crystal(lattice=Lattice(3.524, 3.524, 3.524, 90, 90, 90),
                  space_group=SpaceGroup.from_number(225),
                  atoms=[Atom(element="Ni", fract=(0, 0, 0),
                                occupancy=1.0, B_iso=0.5)],
                  name="FCC_Ni")
    path = tmp_path / "ni_out.cif"
    write_crystal_to_cif(ni, path)
    back = read_cif_to_crystal(path)
    assert back.space_group.number == 225
    assert abs(back.lattice.a - 3.524) < 1e-6
    assert len(back.atoms) == 1
    assert back.atoms[0].element == "Ni"
    assert abs(back.atoms[0].B_iso - 0.5) < 1e-6


def test_write_crystal_round_trip_partial_occupancy(tmp_path):
    """Partial occupancy + B_iso round-trip through write → read."""
    from midas_hkls import Crystal, Atom, Lattice, SpaceGroup
    from midas_pdf.cif import write_crystal_to_cif, read_cif_to_crystal
    ce = Crystal(
        lattice=Lattice(5.41165, 5.41165, 5.41165, 90, 90, 90),
        space_group=SpaceGroup.from_number(225),
        atoms=[Atom(element="Ce", fract=(0, 0, 0), occupancy=0.95, B_iso=0.5),
               Atom(element="O",  fract=(0.25, 0.25, 0.25),
                     occupancy=0.98, B_iso=0.8)],
        name="CeO2")
    path = tmp_path / "ceo2_out.cif"
    write_crystal_to_cif(ce, path)
    back = read_cif_to_crystal(path)
    ce_atom = next(a for a in back.atoms if a.element == "Ce")
    o_atom = next(a for a in back.atoms if a.element == "O")
    assert abs(ce_atom.occupancy - 0.95) < 1e-6
    assert abs(o_atom.occupancy - 0.98) < 1e-6
    assert abs(ce_atom.B_iso - 0.5) < 1e-6
    assert abs(o_atom.B_iso - 0.8) < 1e-6


def test_write_supercell_round_trip_P1(tmp_path):
    """Supercell → CIF (P1) → Crystal recovers all atoms explicitly."""
    from midas_hkls import Crystal, Atom, Lattice, SpaceGroup
    from midas_pdf.cif import write_supercell_to_cif, read_cif_to_crystal
    from midas_pdf.rmc import Supercell
    ni = Crystal(lattice=Lattice(3.524, 3.524, 3.524, 90, 90, 90),
                  space_group=SpaceGroup.from_number(225),
                  atoms=[Atom(element="Ni", fract=(0, 0, 0))],
                  name="Ni").to_torch()
    sc = Supercell.from_crystal(ni, size=(3, 3, 3))
    path = tmp_path / "sc_out.cif"
    write_supercell_to_cif(sc, path)
    back = read_cif_to_crystal(path)
    assert back.space_group.number == 1        # P1
    assert len(back.atoms) == sc.n_atoms       # every atom explicit
    assert abs(back.lattice.a - float(torch.linalg.norm(sc.cell[0]))) < 1e-5


def test_write_crystal_creates_readable_file(tmp_path):
    from midas_hkls import Crystal, Atom, Lattice, SpaceGroup
    from midas_pdf.cif import write_crystal_to_cif
    ni = Crystal(lattice=Lattice(3.524, 3.524, 3.524, 90, 90, 90),
                  space_group=SpaceGroup.from_number(225),
                  atoms=[Atom(element="Ni", fract=(0, 0, 0))], name="Ni")
    path = tmp_path / "check.cif"
    written = write_crystal_to_cif(ni, path)
    assert written.exists()
    content = written.read_text()
    assert content.startswith("data_")
    assert "_cell_length_a" in content
    assert "loop_" in content


def test_read_cif_hydrated_crystal_produces_G_r(tmp_path):
    """End-to-end: CIF → Crystal → pair list → pdffit_gr must work
    without raising."""
    from midas_pdf.structure import build_pair_list, pdffit_gr
    path = _write_cif(tmp_path, CIF_FCC_NI)
    ni = read_cif_to_crystal(path).to_torch()
    r = torch.linspace(0.5, 8.0, 200, dtype=torch.float64)
    pairs = build_pair_list(ni, r_max=9.0)
    G = pdffit_gr(ni, r, pairs, scale=1.0, u_iso=0.005)
    assert G.shape == r.shape
    # FCC Ni first shell should give a peak near 2.49 Å
    peak_r = float(r[int(G[:100].argmax())])
    assert abs(peak_r - 2.492) < 0.1
