"""FF/PF ``hkls.csv`` gains the F2 column, and dropping must not renumber rings.

The ring-number property is the one with teeth. Ring numbers are positional in
the d-sorted reflection list, and ``RingThresh`` / ``RingNumbers`` /
``OverAllRingToIndex`` in the parameter file all reference them **by value**.
Filtering forbidden reflections before the numbering would shift every ring
above the first gap, silently re-pointing every one of those keys at a
different ring — the run would complete and index against the wrong rings.

So: numbers are assigned on the full list, then rows are removed. A surviving
row keeps the number it had.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

zarr = pytest.importorskip("zarr")

from midas_hkls import Atom, Lattice, SpaceGroup  # noqa: E402
from midas_hkls.zarr_compat import generate_hkls_from_zarr  # noqa: E402


def _make_zarr(tmp_path: Path, *, lat, sg_num, wl=0.2066, lsd=1_000_000.0,
               rho_d=200_000.0) -> Path:
    """Minimal archive carrying just the keys _read_zarr_minimal wants."""
    p = tmp_path / "t.MIDAS.zip"
    store = zarr.ZipStore(str(p), mode="w")
    root = zarr.group(store=store)
    ap = root.require_group("analysis").require_group("process").require_group(
        "analysis_parameters")
    ap.create_dataset("LatticeParameter", data=np.array(lat, dtype=np.float64))
    ap.create_dataset("Wavelength", data=np.array([wl]))
    ap.create_dataset("Lsd", data=np.array([lsd]))
    ap.create_dataset("RhoD", data=np.array([rho_d]))
    ap.create_dataset("SpaceGroup", data=np.array([sg_num]))
    store.close()
    return p


SI = dict(lat=(5.431, 5.431, 5.431, 90.0, 90.0, 90.0), sg_num=227)
SI_ATOMS = [Atom("Si", (0.0, 0.0, 0.0)), Atom("Si", (0.25, 0.25, 0.25))]


def _read(path):
    lines = Path(path).read_text().splitlines()
    return lines[0].split(), [l.split() for l in lines[1:] if l.strip()]


def test_no_basis_is_byte_identical_to_the_historical_file(tmp_path):
    """The column is opt-in; without a basis nothing may change at all."""
    z = _make_zarr(tmp_path, **SI)
    a = generate_hkls_from_zarr(z, result_folder=tmp_path / "a")
    b = generate_hkls_from_zarr(z, result_folder=tmp_path / "b", atoms=None)
    assert Path(a).read_bytes() == Path(b).read_bytes()
    hdr, rows = _read(a)
    assert hdr[-1] == "Radius" and len(hdr) == 11
    assert all(len(r) == 11 for r in rows)


def test_basis_appends_exactly_one_column_on_every_row(tmp_path):
    """Ragged output would break numpy.loadtxt in radius/theoretical.py."""
    z = _make_zarr(tmp_path, **SI)
    out = generate_hkls_from_zarr(z, result_folder=tmp_path / "o",
                                  atoms=SI_ATOMS)
    hdr, rows = _read(out)
    assert hdr[-1] == "F2" and len(hdr) == 12
    assert rows and all(len(r) == 12 for r in rows), "every row or none"
    f2 = np.array([float(r[11]) for r in rows])
    assert f2.max() == pytest.approx(1.0)
    assert (f2 >= 0.0).all()


def test_the_first_eleven_columns_are_unchanged_by_adding_f2(tmp_path):
    z = _make_zarr(tmp_path, **SI)
    plain = generate_hkls_from_zarr(z, result_folder=tmp_path / "p")
    withf2 = generate_hkls_from_zarr(z, result_folder=tmp_path / "w",
                                     atoms=SI_ATOMS)
    _, rp = _read(plain)
    _, rw = _read(withf2)
    assert len(rp) == len(rw)
    for a, b in zip(rp, rw):
        assert a == b[:11], "adding F2 must not perturb any existing column"


def test_silicon_has_forbidden_reflections(tmp_path):
    """If this comes back empty the basis never reached the structure factor."""
    z = _make_zarr(tmp_path, **SI)
    out = generate_hkls_from_zarr(z, result_folder=tmp_path / "o",
                                  atoms=SI_ATOMS)
    _, rows = _read(out)
    f2 = np.array([float(r[11]) for r in rows])
    assert (f2 <= 1e-6).any(), "diamond-structure Si must extinguish (200) etc."


def test_drop_forbidden_preserves_ring_numbers(tmp_path):
    """The load-bearing property: surviving rows keep their ring numbers."""
    z = _make_zarr(tmp_path, **SI)
    full = generate_hkls_from_zarr(z, result_folder=tmp_path / "f",
                                   atoms=SI_ATOMS)
    _, rows_full = _read(full)
    # (h,k,l) -> RingNr before any filtering
    ring_of = {(r[0], r[1], r[2]): r[4] for r in rows_full}

    dropped = generate_hkls_from_zarr(z, result_folder=tmp_path / "d",
                                      atoms=SI_ATOMS, drop_forbidden=True)
    _, rows_kept = _read(dropped)
    assert 0 < len(rows_kept) < len(rows_full), "the filter must remove something"
    for r in rows_kept:
        key = (r[0], r[1], r[2])
        assert r[4] == ring_of[key], (
            f"reflection {key} moved from ring {ring_of[key]} to {r[4]} — "
            "filtering renumbered the rings, which silently re-points "
            "RingThresh / OverAllRingToIndex at different rings"
        )


def test_drop_forbidden_removes_only_forbidden_rows(tmp_path):
    z = _make_zarr(tmp_path, **SI)
    full = generate_hkls_from_zarr(z, result_folder=tmp_path / "f",
                                   atoms=SI_ATOMS)
    dropped = generate_hkls_from_zarr(z, result_folder=tmp_path / "d",
                                      atoms=SI_ATOMS, drop_forbidden=True)
    _, rows_full = _read(full)
    _, rows_kept = _read(dropped)
    allowed = {tuple(r[:3]) for r in rows_full if float(r[11]) > 1e-6}
    assert {tuple(r[:3]) for r in rows_kept} == allowed
    assert all(float(r[11]) > 1e-6 for r in rows_kept)


def test_drop_forbidden_without_a_basis_is_an_error(tmp_path):
    z = _make_zarr(tmp_path, **SI)
    with pytest.raises(ValueError, match="requires an atom basis"):
        generate_hkls_from_zarr(z, result_folder=tmp_path / "x",
                                drop_forbidden=True)


def test_atoms_and_cif_together_are_rejected(tmp_path):
    z = _make_zarr(tmp_path, **SI)
    with pytest.raises(ValueError, match="not both"):
        generate_hkls_from_zarr(z, result_folder=tmp_path / "x",
                                atoms=SI_ATOMS, cif_path="anything.cif")


def test_fcc_ceiling_is_one(tmp_path):
    """A monatomic fcc cell must lose nothing — the contrast that makes the
    silicon result meaningful rather than an artefact of the machinery."""
    z = _make_zarr(tmp_path, lat=(3.6, 3.6, 3.6, 90.0, 90.0, 90.0), sg_num=225)
    out = generate_hkls_from_zarr(z, result_folder=tmp_path / "o",
                                  atoms=[Atom("Ni", (0.0, 0.0, 0.0))])
    _, rows = _read(out)
    f2 = np.array([float(r[11]) for r in rows])
    assert (f2 > 1e-6).all(), "fcc has no basis-forbidden reflections"
