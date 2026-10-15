"""Gate G0 of PREREGISTER_atomistic.md: the MD readers round-trip exactly and refuse to guess.

Synthetic dumps only. Each test plants a known configuration, writes it, reads it back,
and compares to the planted values, so a reader that silently drops a triclinic tilt, a
type map or a frame fails here rather than producing a plausible wrong SAXS frame.
"""
import gzip

import numpy as np
import pytest

from midas_saxs.md_io import (
    AtomicFrame,
    iter_lammps_dump,
    read_lammps_dump,
    read_lammps_trajectory,
    write_lammps_dump,
)

pytestmark = pytest.mark.unit

TYPES = {1: "Cu", 2: "He"}


def _frame(n=40, seed=0, cell=None, origin=(0.0, 0.0, 0.0), timestep=100):
    rng = np.random.default_rng(seed)
    cell = np.diag([18.0, 21.0, 24.0]) if cell is None else np.asarray(cell, float)
    s = rng.random((n, 3))
    pos = np.asarray(origin) + s @ cell
    species = np.where(rng.random(n) < 0.7, "Cu", "He")
    return AtomicFrame(positions=pos, species=species, cell=cell,
                       origin=np.asarray(origin, float), timestep=timestep,
                       ids=np.arange(1, n + 1))


def _trajectory(n_frames, **kw):
    """Same atoms and species every frame; only the positions move (as in a real NVT run)."""
    base = _frame(seed=0, **kw)
    out = []
    for k in range(n_frames):
        rng = np.random.default_rng(100 + k)
        out.append(AtomicFrame(positions=base.positions + 0.1 * rng.standard_normal(base.positions.shape),
                               species=base.species, cell=base.cell, origin=base.origin,
                               timestep=100 * k, ids=base.ids))
    return out


def _same(a: AtomicFrame, b: AtomicFrame, tol=0.0):
    assert np.array_equal(a.species, b.species)
    np.testing.assert_allclose(a.cell, b.cell, rtol=0, atol=tol)
    np.testing.assert_allclose(a.origin, b.origin, rtol=0, atol=tol)
    np.testing.assert_allclose(a.positions, b.positions, rtol=0, atol=tol)
    assert a.timestep == b.timestep


def test_orthogonal_round_trip_is_exact(tmp_path):
    f = _frame(origin=(-3.5, 1.25, 7.0))
    p = tmp_path / "o.dump"
    write_lammps_dump(p, f, type_map=TYPES)
    g = read_lammps_dump(p, type_map=TYPES)
    _same(f, g)                              # repr() floats: bit exact
    assert g.pbc == (True, True, True)


def test_triclinic_round_trip_recovers_all_three_tilts(tmp_path):
    cell = [[17.0, 0, 0], [3.1, 19.0, 0], [-2.2, 4.4, 23.0]]
    f = _frame(cell=cell, origin=(0.5, -1.0, 2.0))
    p = tmp_path / "t.dump"
    write_lammps_dump(p, f, type_map=TYPES)
    g = read_lammps_dump(p, type_map=TYPES)
    _same(f, g, tol=1e-12)
    np.testing.assert_allclose(g.cell[1, 0], 3.1)       # xy
    np.testing.assert_allclose(g.cell[2, 0], -2.2)      # xz
    np.testing.assert_allclose(g.cell[2, 1], 4.4)       # yz


def test_triclinic_bounding_box_is_converted_with_the_lammps_rule(tmp_path):
    """Hand-written triclinic header: bounds are the bounding box, not the cell."""
    xy, xz, yz = 3.0, -2.0, 4.0
    xlo, xhi, ylo, yhi, zlo, zhi = 0.0, 10.0, 0.0, 12.0, 0.0, 14.0
    xlo_b = xlo + min(0, xy, xz, xy + xz)
    xhi_b = xhi + max(0, xy, xz, xy + xz)
    ylo_b = ylo + min(0, yz)
    yhi_b = yhi + max(0, yz)
    p = tmp_path / "hand.dump"
    p.write_text(
        "ITEM: TIMESTEP\n0\nITEM: NUMBER OF ATOMS\n1\n"
        f"ITEM: BOX BOUNDS xy xz yz pp pp pp\n{xlo_b} {xhi_b} {xy}\n{ylo_b} {yhi_b} {xz}\n"
        f"{zlo} {zhi} {yz}\nITEM: ATOMS id type xs ys zs\n1 1 0.5 0.5 0.5\n")
    g = read_lammps_dump(p, type_map={1: "Cu"})
    np.testing.assert_allclose(g.cell, [[10, 0, 0], [3, 12, 0], [-2, 4, 14]])
    np.testing.assert_allclose(g.origin, [xlo, ylo, zlo])
    np.testing.assert_allclose(g.positions[0], 0.5 * np.array([10 + 3 - 2, 12 + 4, 14]))


def test_scaled_coordinates_match_cartesian(tmp_path):
    cell = [[17.0, 0, 0], [3.1, 19.0, 0], [-2.2, 4.4, 23.0]]
    f = _frame(cell=cell, origin=(1.0, 2.0, 3.0))
    pc, ps = tmp_path / "c.dump", tmp_path / "s.dump"
    write_lammps_dump(pc, f, type_map=TYPES, scaled=False)
    write_lammps_dump(ps, f, type_map=TYPES, scaled=True)
    a, b = read_lammps_dump(pc, type_map=TYPES), read_lammps_dump(ps, type_map=TYPES)
    np.testing.assert_allclose(a.positions, b.positions, atol=1e-12)


def test_element_column_needs_no_type_map(tmp_path):
    f = _frame()
    p = tmp_path / "e.dump"
    write_lammps_dump(p, f, element_column=True)
    g = read_lammps_dump(p)
    _same(f, g)


def test_integer_types_without_a_type_map_are_refused(tmp_path):
    p = tmp_path / "x.dump"
    write_lammps_dump(p, _frame(), type_map=TYPES)
    with pytest.raises(ValueError, match="no default element"):
        read_lammps_dump(p)


def test_missing_type_in_the_map_is_refused(tmp_path):
    p = tmp_path / "x.dump"
    write_lammps_dump(p, _frame(), type_map=TYPES)
    with pytest.raises(ValueError, match="no entry for type"):
        read_lammps_dump(p, type_map={1: "Cu"})


def test_type_map_as_a_sequence_starts_at_type_one(tmp_path):
    f = _frame()
    p = tmp_path / "x.dump"
    write_lammps_dump(p, f, type_map=TYPES)
    g = read_lammps_dump(p, type_map=["Cu", "He"])
    assert np.array_equal(f.species, g.species)


def test_unknown_element_is_refused(tmp_path):
    p = tmp_path / "x.dump"
    p.write_text("ITEM: TIMESTEP\n0\nITEM: NUMBER OF ATOMS\n1\nITEM: BOX BOUNDS pp pp pp\n"
                 "0 5\n0 5\n0 5\nITEM: ATOMS id element x y z\n1 Zz 1 1 1\n")
    with pytest.raises(ValueError, match="not an element"):
        read_lammps_dump(p)


def test_non_angstrom_units_are_refused(tmp_path):
    p = tmp_path / "lj.dump"
    write_lammps_dump(p, _frame(), type_map=TYPES, units="lj")
    with pytest.raises(ValueError, match="Angstrom"):
        read_lammps_dump(p, type_map=TYPES)


def test_real_and_missing_units_are_accepted(tmp_path):
    f = _frame()
    for units in ("real", None):
        p = tmp_path / f"{units}.dump"
        write_lammps_dump(p, f, type_map=TYPES, units=units)
        _same(f, read_lammps_dump(p, type_map=TYPES))


def test_gzip(tmp_path):
    f = _frame()
    p = tmp_path / "z.dump.gz"
    write_lammps_dump(p, f, type_map=TYPES)
    with gzip.open(p, "rt") as fh:                       # really gzipped
        assert fh.readline().startswith("ITEM")
    _same(f, read_lammps_dump(p, type_map=TYPES))


def test_multi_frame_and_selection(tmp_path):
    frames = _trajectory(6)
    p = tmp_path / "m.dump"
    write_lammps_dump(p, frames, type_map=TYPES)
    for k in (0, 3, 5):
        _same(frames[k], read_lammps_dump(p, type_map=TYPES, frame=k))
    traj = read_lammps_trajectory(p, type_map=TYPES)
    assert [t.timestep for t in traj] == [0, 100, 200, 300, 400, 500]
    sub = read_lammps_trajectory(p, type_map=TYPES, frames=slice(1, 6, 2))
    assert [t.timestep for t in sub] == [100, 300, 500]
    with pytest.raises(ValueError, match="only 6 frames"):
        read_lammps_dump(p, type_map=TYPES, frame=6)
    with pytest.raises(ValueError):
        read_lammps_dump(p, type_map=TYPES, frame=-1)


def test_iterator_is_lazy(tmp_path):
    frames = _trajectory(3)
    p = tmp_path / "m.dump"
    write_lammps_dump(p, frames, type_map=TYPES)
    it = iter_lammps_dump(p, type_map=TYPES)
    assert next(it).timestep == 0
    assert next(it).timestep == 100


def test_changing_cell_between_frames_is_refused(tmp_path):
    a = _frame(seed=0, timestep=0)
    b = AtomicFrame(positions=a.positions, species=a.species, cell=np.diag([18.5, 21.0, 24.0]),
                    timestep=1, ids=a.ids)
    p = tmp_path / "npt.dump"
    write_lammps_dump(p, [a, b], type_map=TYPES)
    with pytest.raises(ValueError, match="cell changes"):
        read_lammps_trajectory(p, type_map=TYPES)


def test_truncated_file_is_refused(tmp_path):
    p = tmp_path / "t.dump"
    write_lammps_dump(p, _frame(n=40), type_map=TYPES)
    lines = p.read_text().splitlines()
    p.write_text("\n".join(lines[:-5]) + "\n")
    with pytest.raises(ValueError, match="truncated"):
        read_lammps_dump(p, type_map=TYPES)


def test_not_a_lammps_dump_is_refused(tmp_path):
    p = tmp_path / "x.xyz"
    p.write_text("3\ncomment\nCu 0 0 0\n")
    with pytest.raises(ValueError, match="ITEM"):
        read_lammps_dump(p, type_map=TYPES)


def test_extra_columns_and_ids_are_kept(tmp_path):
    p = tmp_path / "c.dump"
    p.write_text("ITEM: TIMESTEP\n7\nITEM: NUMBER OF ATOMS\n2\nITEM: BOX BOUNDS pp pp pp\n"
                 "0 5\n0 5\n0 5\nITEM: ATOMS id type x y z c_epot\n"
                 "9 1 1 1 1 -3.5\n4 1 2 2 2 -3.4\n")
    g = read_lammps_dump(p, type_map={1: "Cu"})
    assert g.timestep == 7
    np.testing.assert_array_equal(g.ids, [9, 4])             # file order is kept
    np.testing.assert_allclose(g.columns["c_epot"], [-3.5, -3.4])


def test_non_periodic_flags_are_recorded(tmp_path):
    p = tmp_path / "f.dump"
    p.write_text("ITEM: TIMESTEP\n0\nITEM: NUMBER OF ATOMS\n1\nITEM: BOX BOUNDS pp pp ff\n"
                 "0 5\n0 5\n0 5\nITEM: ATOMS id type x y z\n1 1 1 1 1\n")
    assert read_lammps_dump(p, type_map={1: "Cu"}).pbc == (True, True, False)


def test_wrapped_and_fractional(tmp_path):
    cell = [[10.0, 0, 0], [2.0, 10.0, 0], [0, 0, 10.0]]
    f = AtomicFrame(positions=[[-1.0, 12.0, 25.0]], species=["Cu"], cell=cell)
    s = f.fractional()
    np.testing.assert_allclose(f.origin + s @ f.cell, f.positions, atol=1e-12)
    w = f.wrapped()
    sw = f.__class__(positions=w, species=["Cu"], cell=cell).fractional()
    assert ((sw >= -1e-12) & (sw < 1 + 1e-12)).all()
    # wrapping moves an atom by a lattice vector, nothing else
    d = np.linalg.solve(f.cell.T, (f.positions - w).T).T
    np.testing.assert_allclose(d, np.round(d), atol=1e-12)


def test_frame_validation():
    with pytest.raises(ValueError, match="singular"):
        AtomicFrame(positions=[[0, 0, 0]], species=["Cu"], cell=np.zeros((3, 3)))
    with pytest.raises(ValueError, match="species"):
        AtomicFrame(positions=[[0, 0, 0]], species=["Cu", "Cu"], cell=np.eye(3))
    with pytest.raises(ValueError, match="NaN"):
        AtomicFrame(positions=[[np.nan, 0, 0]], species=["Cu"], cell=np.eye(3))


def test_changing_species_between_frames_is_refused(tmp_path):
    a = _frame(seed=0, timestep=0)
    sp = a.species.copy()
    sp[0] = "He" if sp[0] == "Cu" else "Cu"
    b = AtomicFrame(positions=a.positions, species=sp, cell=a.cell, timestep=1, ids=a.ids)
    p = tmp_path / "sp.dump"
    write_lammps_dump(p, [a, b], type_map=TYPES)
    with pytest.raises(ValueError, match="species order"):
        read_lammps_trajectory(p, type_map=TYPES)
