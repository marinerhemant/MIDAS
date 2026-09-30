"""ESRF Bliss master -> MIDAS zips (midas_zipper.bliss) and the in-place array writer (midas_zipper.zipwrite).

The synthetic master reproduces the ID11 layout seen on ma5608: entries "N.1" with measurement/eiger as a VIRTUAL
dataset over a Lima file scanNNNN/eiger_0000.h5 (relative path), measurement/rot_center per frame, and
instrument/positioners/dty; plus a counting entry with no detector, which must be skipped.
"""

from __future__ import annotations

import shutil

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")
zarr = pytest.importorskip("zarr")

from midas_zipper.bliss import (DTY_KEY, MASK_KEY, OMEGA_CENTER_KEY, convert_master,
                                omega_map_from_reference, survey_master)
from midas_zipper.param_refresh import ParamRefreshError
from midas_zipper.zipwrite import write_arrays

pytestmark = pytest.mark.skipif(shutil.which("zip") is None, reason="Info-ZIP 'zip' not on PATH")

NF, NZ, NY = 5, 16, 12
PARAMS = """\
Lsd 1000000.0
BC 6.0 8.0
Wavelength 0.123
Padding 6
SpaceGroup 167
OmegaStep 0.125
OmegaFirstFile -90
"""


def _master(tmp_path, dtys=(-0.3, 0.0), snake=True, moving_dty=False):
    """Bliss-like master + one Lima file per scan. Scan k's frames are k*1000 + frame index."""
    m = tmp_path / "sample" / "sample_Z860.h5"; m.parent.mkdir(parents=True)
    rots = []
    with h5py.File(m, "w") as f:
        for k, y in enumerate(dtys, start=1):
            lima = m.parent / f"scan{k:04d}" / "eiger_0000.h5"; lima.parent.mkdir()
            data = (np.arange(NF)[:, None, None] + 1000 * k + np.zeros((NF, NZ, NY))).astype(np.uint16)
            with h5py.File(lima, "w") as L:
                L.create_dataset("entry_0000/measurement/data", data=data)
            layout = h5py.VirtualLayout(shape=data.shape, dtype=data.dtype)
            layout[:] = h5py.VirtualSource(f"scan{k:04d}/eiger_0000.h5", "entry_0000/measurement/data", shape=data.shape)
            g = f.create_group(f"{k}.1")
            g["title"] = f"fscan rot {k}"
            g.create_virtual_dataset("measurement/eiger", layout)
            r = np.linspace(-89.94, 90.94, NF) + np.random.default_rng(k).normal(0, 1e-4, NF)
            if snake and k % 2 == 0:
                r = r[::-1]
            g["measurement/rot_center"] = r; rots.append(r)
            g["instrument/positioners/dty"] = np.array([y, y + 0.1]) if moving_dty else y
        c = f.create_group(f"{len(dtys) + 1}.1")
        c["title"] = "ct 0.1"; c["measurement/fpico6"] = np.ones(3)
    return m, rots


def _ref_zip(tmp_path, omegas, name="ref.MIDAS.zip", mask=None):
    fn = tmp_path / name
    store = zarr.ZipStore(str(fn), mode="w"); root = zarr.group(store=store, overwrite=True)
    root.create_dataset(OMEGA_CENTER_KEY, data=np.asarray(omegas, float))
    if mask is not None:
        root.create_dataset(MASK_KEY, data=mask)
    store.close()
    return fn


# ── zipwrite ────────────────────────────────────────────────────────────────
def test_write_arrays_adds_replaces_and_writes_zeros(tmp_path):
    fn = _ref_zip(tmp_path, [1.0, 2.0])
    write_arrays(fn, {OMEGA_CENTER_KEY: np.array([5.0, 6.0, 7.0]),          # replace, new shape
                      "brand/new/group/x": np.zeros((2, 3), np.uint16)})     # new groups, all-zero value
    r = zarr.open(str(fn), "r")
    assert np.array_equal(r[OMEGA_CENTER_KEY][...], [5.0, 6.0, 7.0])
    assert r["brand/new/group/x"].shape == (2, 3) and not r["brand/new/group/x"][...].any()


def test_write_arrays_reports_a_write_that_did_not_take(tmp_path, monkeypatch):
    import subprocess
    fn = _ref_zip(tmp_path, [1.0])
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0, "", ""))
    with pytest.raises(ParamRefreshError, match="did not take"):
        write_arrays(fn, {OMEGA_CENTER_KEY: np.array([9.0])})


# ── survey ──────────────────────────────────────────────────────────────────
def test_survey_finds_scans_resolves_the_lima_file_and_skips_counts(tmp_path):
    m, rots = _master(tmp_path)
    sc = survey_master(m)
    assert [s.entry for s in sc] == ["1.1", "2.1"]
    assert sc[0].frames_file.endswith("scan0001/eiger_0000.h5") and sc[0].frames_path.endswith("entry_0000/measurement/data")
    assert sc[0].n_frames == NF and np.allclose(sc[1].rot_center, rots[1]) and sc[0].dty == -0.3


def test_a_partial_vds_is_read_through_the_vds_not_the_whole_lima_file(tmp_path):
    """Two scans sharing one Lima file (frames 0-4 and 5-9): each must get its own frames."""
    m = tmp_path / "shared.h5"; lima = tmp_path / "eiger_0000.h5"
    data = np.arange(2 * NF)[:, None, None].repeat(NZ, 1).repeat(NY, 2).astype(np.uint16)
    with h5py.File(lima, "w") as L:
        L.create_dataset("entry_0000/measurement/data", data=data)
    with h5py.File(m, "w") as f:
        for k in (1, 2):
            src = h5py.VirtualSource("eiger_0000.h5", "entry_0000/measurement/data", shape=data.shape)
            lay = h5py.VirtualLayout(shape=(NF, NZ, NY), dtype=data.dtype)
            lay[:] = src[(k - 1) * NF:k * NF]
            g = f.create_group(f"{k}.1"); g.create_virtual_dataset("measurement/eiger", lay)
            g["measurement/rot_center"] = np.arange(NF, dtype=float)
    sc = survey_master(m)
    assert all(s.frames_file == str(m.resolve()) for s in sc)       # not the shared Lima file
    p = tmp_path / "Parameters.txt"; p.write_text(PARAMS)
    out = convert_master(m, tmp_path / "raw", "S", p, omega_sign=1)
    assert np.array_equal(zarr.open(str(out[1]), "r")["exchange/data"][:, 0, 0], np.arange(NF, 2 * NF))


def test_survey_refuses_positioner_values_it_cannot_assign_to_frames(tmp_path):
    m, _ = _master(tmp_path, moving_dty=True)          # 2 differing dty values for 5 frames
    with pytest.raises(ValueError, match="cannot tell which frames"):
        survey_master(m)


def _fscan2d(tmp_path, lines=3, per_line=NF, continuous=False):
    """One entry holding every line: frames k*1000 + i, dty per frame in measurement/dty."""
    m = tmp_path / "f2d.h5"; lima = tmp_path / "eiger_0000.h5"; n = lines * per_line
    data = (np.arange(n)[:, None, None] + np.zeros((n, NZ, NY))).astype(np.uint16)
    with h5py.File(lima, "w") as L:
        L.create_dataset("entry_0000/measurement/data", data=data)
    y = np.linspace(-0.3, 0.3, n) if continuous else np.repeat(np.arange(lines) * 0.3 - 0.3, per_line)
    rot = np.concatenate([np.linspace(-90, 91, per_line)[::(1 if k % 2 == 0 else -1)] for k in range(lines)])
    with h5py.File(m, "w") as f:
        lay = h5py.VirtualLayout(shape=data.shape, dtype=data.dtype)
        lay[:] = h5py.VirtualSource("eiger_0000.h5", "entry_0000/measurement/data", shape=data.shape)
        g = f.create_group("1.1"); g["title"] = "fscan2d dty -0.3 0.3 3 rot -90 181 5"
        g.create_virtual_dataset("measurement/eiger", lay)
        g["measurement/rot_center"] = rot; g["measurement/dty"] = y
    return m, rot


def test_fscan2d_entry_is_split_into_one_scan_per_line(tmp_path):
    m, rot = _fscan2d(tmp_path)
    sc = survey_master(m, min_frames_per_line=3)
    assert [s.n_frames for s in sc] == [NF] * 3 and [s.frame_start for s in sc] == [0, NF, 2 * NF]
    assert np.allclose([s.dty for s in sc], [-0.3, 0.0, 0.3])
    p = tmp_path / "Parameters.txt"; p.write_text(PARAMS)
    out = convert_master(m, tmp_path / "raw", "S", p, omega_sign=1, scans=sc)
    for k, o in enumerate(out):
        z = zarr.open(str(o), "r")
        assert np.array_equal(z["exchange/data"][:, 0, 0], np.arange(k * NF, (k + 1) * NF))   # this line's frames only
        assert np.allclose(z[OMEGA_CENTER_KEY][...], rot[k * NF:(k + 1) * NF])


def test_continuously_moving_dty_is_refused(tmp_path):
    m, _ = _fscan2d(tmp_path, continuous=True)
    with pytest.raises(ValueError, match="continuously moving"):
        survey_master(m)


# ── omega mapping ───────────────────────────────────────────────────────────
def test_omega_map_picks_the_sign_that_matches_frame_by_frame():
    rng = np.random.default_rng(0)
    rots = [np.linspace(-89.94, 90.94, 1448) + rng.normal(0, 1e-3, 1448) for _ in range(3)]
    rots[1] = rots[1][::-1]                                         # snake
    refs = [-r + 0.5 + rng.normal(0, 1e-3, r.size) for r in rots]   # reference made with -rot + 0.5
    m = omega_map_from_reference(rots, refs, max_abs_offset=None)
    assert m["sign"] == -1.0 and abs(m["offset"] - 0.5) < 1e-3 and m["residual_p99"] < 0.01
    assert m["other_residual_p99"] > 1.0


def test_omega_map_refuses_when_nothing_matches():
    rng = np.random.default_rng(1)
    rots = [np.linspace(0, 180, 100)]; refs = [rng.uniform(-90, 90, 100)]
    with pytest.raises(ValueError, match="not determined"):
        omega_map_from_reference(rots, refs)


# ── end to end ──────────────────────────────────────────────────────────────
def test_convert_master_writes_frames_omega_dty_and_mask(tmp_path):
    m, rots = _master(tmp_path)
    p = tmp_path / "Parameters.txt"; p.write_text(PARAMS)
    mask = np.zeros((1, NZ, NY), np.uint16); mask[0, 3, 4] = 1
    ref = _ref_zip(tmp_path, np.zeros(NF), name="maskref.MIDAS.zip", mask=mask)
    out = convert_master(m, tmp_path / "raw", "S_Z860_dataset", p, omega_sign=-1, omega_offset=0.25, mask_from=ref)
    assert [o.name for o in out] == ["S_Z860_dataset_000001.MIDAS.zip", "S_Z860_dataset_000002.MIDAS.zip"]
    for k, o in enumerate(out, start=1):
        z = zarr.open(str(o), "r")
        assert z["exchange/data"].shape == (NF, NZ, NY)
        assert np.array_equal(z["exchange/data"][:, 0, 0], np.arange(NF) + 1000 * k)      # the right scan's frames
        assert np.allclose(z[OMEGA_CENTER_KEY][...], -rots[k - 1] + 0.25)
        assert np.array_equal(z[MASK_KEY][...], mask)
        assert np.isclose(z[DTY_KEY][0], (-0.3, 0.0)[k - 1])
    oc2 = zarr.open(str(out[1]), "r")[OMEGA_CENTER_KEY][...]
    assert oc2[0] < oc2[-1]                     # scan 2's rot runs down; with sign -1 its omega runs up, frame order kept


def test_convert_master_requires_an_explicit_sign(tmp_path):
    m, _ = _master(tmp_path)
    p = tmp_path / "Parameters.txt"; p.write_text(PARAMS)
    with pytest.raises(ValueError, match="omega_sign"):
        convert_master(m, tmp_path / "raw", "S", p, omega_sign=0)


def test_a_snake_starting_the_other_way_is_refused_not_pinned_to_the_wrong_sign():
    """Reference scans run up, down, up...; the new data run down, up, down (same encoder values reversed in time).
    -rot + (first + last) matches frame by frame - the wrong sign - so a nonzero offset must be refused by default."""
    rng = np.random.default_rng(2)
    ref = [np.linspace(-89.94, 90.94, 1448) + rng.normal(0, 1e-3, 1448) for _ in range(4)]
    ref = [r if k % 2 == 0 else r[::-1] for k, r in enumerate(ref)]
    new = [r[::-1] + rng.normal(0, 1e-3, r.size) for r in ref]
    with pytest.raises(ValueError, match="WRONG sign"):
        omega_map_from_reference(new, ref)
    m = omega_map_from_reference([r + rng.normal(0, 1e-3, r.size) for r in ref], ref)   # same direction: pure +rot
    assert m["sign"] == 1.0 and abs(m["offset"]) < 0.01
