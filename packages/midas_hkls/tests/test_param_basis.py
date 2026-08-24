"""Reading an atom basis out of a MIDAS parameter file.

The point of this reader is that FF and PF get the basis the same way NF does,
without a third parse of what a ``PhaseAtom`` line means. It also has to be
inert: a parameter file that declares nothing must produce ``{}`` so
``hkls.csv`` stays byte-identical to every file ever written before.
"""
from __future__ import annotations

import pytest

from midas_hkls import read_phase_basis


def _write(tmp_path, text, name="Parameters.txt"):
    p = tmp_path / name
    p.write_text(text)
    return p


def test_no_basis_gives_empty_dict(tmp_path):
    p = _write(tmp_path, "Wavelength 0.2066\nLsd 1000000\n")
    assert read_phase_basis(p) == {}


def test_missing_file_is_not_an_error(tmp_path):
    """Stages call this unconditionally; an absent file just means no basis."""
    assert read_phase_basis(tmp_path / "nope.txt") == {}


def test_phase_atoms_are_collected_in_order(tmp_path):
    p = _write(tmp_path, """
Wavelength 0.2066
PhaseAtom Si 0 0 0
PhaseAtom Si 0.25 0.25 0.25 0.9 0.42
Lsd 1000000
""")
    got = read_phase_basis(p)
    atoms = got["atoms"]
    assert [a.element for a in atoms] == ["Si", "Si"]
    assert atoms[1].fract == (0.25, 0.25, 0.25)
    assert atoms[1].occupancy == 0.9
    assert atoms[1].B_iso == 0.42
    assert "drop_forbidden" not in got


def test_comments_and_trailing_semicolons_are_handled(tmp_path):
    """paramstest.txt writes ``Key value;``; Parameters.txt does not."""
    p = _write(tmp_path, """
# a comment
PhaseAtom Ni 0 0 0;      # inline comment
Wavelength 0.2066;
""")
    atoms = read_phase_basis(p)["atoms"]
    assert len(atoms) == 1 and atoms[0].element == "Ni"
    assert atoms[0].fract == (0.0, 0.0, 0.0)


def test_cif_path_is_picked_up(tmp_path):
    p = _write(tmp_path, "PhaseCIF /some/where/ceo2.cif\n")
    got = read_phase_basis(p)
    assert got == {"cif_path": "/some/where/ceo2.cif"}


def test_atoms_and_cif_together_are_refused(tmp_path):
    """Silently preferring one would let a stale block shadow the other."""
    p = _write(tmp_path, "PhaseAtom Si 0 0 0\nPhaseCIF x.cif\n")
    with pytest.raises(ValueError, match="mutually exclusive"):
        read_phase_basis(p)


def test_drop_forbidden_rides_along_with_its_threshold(tmp_path):
    p = _write(tmp_path, """
PhaseAtom Si 0 0 0
PhaseAtom Si 0.25 0.25 0.25
DropForbiddenReflections 1
ForbiddenF2Threshold 1e-4
""")
    got = read_phase_basis(p)
    assert got["drop_forbidden"] is True
    assert got["forbidden_f2_threshold"] == 1e-4


def test_drop_forbidden_without_a_basis_is_refused(tmp_path):
    """Asking for filtering and silently not getting it is the failure mode."""
    p = _write(tmp_path, "DropForbiddenReflections 1\nWavelength 0.2066\n")
    with pytest.raises(ValueError, match="no PhaseAtom or PhaseCIF"):
        read_phase_basis(p)


def test_drop_forbidden_zero_is_inert(tmp_path):
    p = _write(tmp_path, "DropForbiddenReflections 0\nWavelength 0.2066\n")
    assert read_phase_basis(p) == {}


def test_result_is_directly_usable_as_generator_kwargs(tmp_path):
    """The contract: whatever comes back is **kwargs for the hkls generator."""
    import inspect

    from midas_hkls.zarr_compat import generate_hkls_from_zarr

    p = _write(tmp_path, """
PhaseAtom Si 0 0 0
DropForbiddenReflections 1
""")
    got = read_phase_basis(p)
    sig = inspect.signature(generate_hkls_from_zarr).parameters
    for k in got:
        assert k in sig, f"{k} is not a generate_hkls_from_zarr parameter"
