"""ESRF Bliss master -> per-scan zips inside midas_pipeline (zip_convert --bliss-master).

Synthetic master with the ESRF ID11 layout (see midas_zipper's test_bliss): scan k's frames are
k*1000 + frame index, rot_center runs up on odd scans and down on even ones (a snake), dty steps by 0.3 um.
"""

from __future__ import annotations

import shutil
import types

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")
zarr = pytest.importorskip("zarr")
pytest.importorskip("midas_zipper.bliss")

from midas_pipeline._pf_scans import PFScanInfo
from midas_pipeline.bliss_ingest import BlissIngestError, build_zips_from_bliss
from midas_zipper.bliss import OMEGA_CENTER_KEY

pytestmark = pytest.mark.skipif(shutil.which("zip") is None, reason="Info-ZIP 'zip' not on PATH")

NF, NZ, NY = 5, 16, 12
PARAMS = "Lsd 1000000.0\nBC 6.0 8.0\nWavelength 0.123\nPadding 6\nSpaceGroup 167\nOmegaStep 0.125\nOmegaFirstFile -90\n"
DTY = [-0.3, 0.0, 0.3]


def _master(tmp_path, dtys=DTY, rot_lo=-89.94, rot_hi=90.94):
    m = tmp_path / "sample" / "sample_layer.h5"; m.parent.mkdir(parents=True)
    rots = []
    with h5py.File(m, "w") as f:
        for k, y in enumerate(dtys, start=1):
            lima = m.parent / f"scan{k:04d}" / "eiger_0000.h5"; lima.parent.mkdir()
            data = (np.arange(NF)[:, None, None] + 1000 * k + np.zeros((NF, NZ, NY))).astype(np.uint16)
            with h5py.File(lima, "w") as L:
                L.create_dataset("entry_0000/measurement/data", data=data)
            lay = h5py.VirtualLayout(shape=data.shape, dtype=data.dtype)
            lay[:] = h5py.VirtualSource(f"scan{k:04d}/eiger_0000.h5", "entry_0000/measurement/data", shape=data.shape)
            g = f.create_group(f"{k}.1"); g["title"] = f"fscan rot {k}"
            g.create_virtual_dataset("measurement/eiger", lay)
            r = np.linspace(rot_lo, rot_hi, NF) + np.random.default_rng(k).normal(0, 1e-4, NF)
            g["measurement/rot_center"] = r[::-1] if k % 2 == 0 else r
            rots.append(r[::-1] if k % 2 == 0 else r)
            g["instrument/positioners/dty"] = y
    return m, rots


def _scans(tmp_path, positions):
    out = []
    for k, y in enumerate(positions, start=1):
        d = tmp_path / "work" / str(k)
        out.append(PFScanInfo(layer_nr=1, scan_nr=k, abs_scan_nr=k, scan_dir=d,
                              zip_path=d / f"S_{k:06d}.MIDAS.zip", y_position_um=float(y)))
    return out


def _no_zips(tmp_path):
    return not list((tmp_path / "work").rglob("*.zip"))


def _run(tmp_path, master, positions=(0.3, 0.0, -0.3), step=0.3, **kw):
    p = tmp_path / "params.txt"; p.write_text(PARAMS)
    return build_zips_from_bliss(_scans(tmp_path, positions), master=master, param_file=p, layer_dir=tmp_path / "layer",
                                 scan_step_um=step, **kw)


def test_the_omega_sign_is_never_guessed(tmp_path):
    m, _ = _master(tmp_path)
    with pytest.raises(BlissIngestError, match="not guessed"):
        _run(tmp_path, m)
    assert _no_zips(tmp_path)


def test_builds_every_zip_with_per_frame_omega_for_a_snake_and_records_the_decisions(tmp_path):
    m, rots = _master(tmp_path)
    rec = _run(tmp_path, m, omega_sign=1)
    assert rec["n_built"] == 3 and rec["n_failed"] == 0
    assert rec["positions_vs_dty"] == "-dty" and rec["snake_odd_rising_even_falling"] is True
    for k, r in enumerate(rots, start=1):
        z = zarr.open(str(tmp_path / "work" / str(k) / f"S_{k:06d}.MIDAS.zip"), "r")
        assert np.allclose(z[OMEGA_CENTER_KEY][...], r)                # each frame carries its OWN angle
        assert np.array_equal(z["exchange/data"][:, 0, 0], np.arange(NF) + 1000 * k)    # frames unchanged, in order
    assert rec["n_scans_rising"] == 2 and rec["n_scans_falling"] == 1
    assert (tmp_path / "layer" / "midas_log" / "bliss_ingest.json").is_file()


def test_pins_the_sign_from_reference_zips_and_refuses_a_disagreeing_flag(tmp_path):
    m, rots = _master(tmp_path)
    ref = tmp_path / "ref"
    for k, r in enumerate(rots, start=1):
        d = ref / str(k); d.mkdir(parents=True)
        s = zarr.ZipStore(str(d / f"R_{k:06d}.MIDAS.zip"), mode="w"); g = zarr.group(store=s, overwrite=True)
        g.create_dataset(OMEGA_CENTER_KEY, data=-r); s.close()           # the reference says omega = -rot
    with pytest.raises(BlissIngestError, match="disagrees"):
        _run(tmp_path, m, omega_sign=1, omega_ref_raw=str(ref))
    rec = _run(tmp_path, m, omega_ref_raw=str(ref))
    assert rec["omega_sign"] == -1.0 and rec["omega_pin"]["other_residual_p99"] > 1.0


def test_refuses_wrong_scan_count_and_wrong_dty_step(tmp_path):
    m, _ = _master(tmp_path)
    with pytest.raises(BlissIngestError, match="scan positions"):
        _run(tmp_path, m, positions=(0.3, 0.0), omega_sign=1)
    with pytest.raises(BlissIngestError, match="dty steps"):
        _run(tmp_path, m, step=0.5, omega_sign=1)
    assert _no_zips(tmp_path)


def test_a_scan_through_180_is_refused_before_anything_is_written(tmp_path):
    m, _ = _master(tmp_path)
    with pytest.raises(BlissIngestError, match="through"):
        _run(tmp_path, m, omega_sign=1, omega_offset=120.0)
    assert _no_zips(tmp_path)


def test_existing_zips_are_kept_and_only_the_missing_ones_are_built(tmp_path):
    m, _ = _master(tmp_path)
    _run(tmp_path, m, omega_sign=1)
    gone = tmp_path / "work" / "2" / "S_000002.MIDAS.zip"; gone.unlink()
    keep = tmp_path / "work" / "1" / "S_000001.MIDAS.zip"; before = keep.stat().st_mtime_ns
    rec = _run(tmp_path, m, omega_sign=1)
    assert rec["n_built"] == 1 and rec["n_present"] == 2 and gone.is_file() and keep.stat().st_mtime_ns == before


def test_cli_flags_reach_the_config():
    from midas_pipeline.cli import _build_parser
    run = _build_parser()._subparsers._group_actions[0].choices["run"]
    names = {o for act in run._actions for o in act.option_strings}
    assert {"--bliss-master", "--bliss-omega-sign", "--bliss-omega-ref-raw", "--bliss-omega-offset",
            "--bliss-allow-offset", "--bliss-mask-from", "--bliss-detector"} <= names


def test_parallel_workers_each_get_their_own_scan(tmp_path):
    """Regression: generate_ff_zip goes through sys.argv, so threads in one process all built scan 1."""
    m, rots = _master(tmp_path)
    _run(tmp_path, m, omega_sign=1, zip_workers=3)
    for k in (1, 2, 3):
        z = zarr.open(str(tmp_path / "work" / str(k) / f"S_{k:06d}.MIDAS.zip"), "r")
        assert np.array_equal(z["exchange/data"][:, 0, 0], np.arange(NF) + 1000 * k)
        assert np.allclose(z[OMEGA_CENTER_KEY][...], rots[k - 1])


def test_a_missing_mask_is_recorded_and_warned(tmp_path, caplog):
    """A conversion with no exchange/mask leaves the gaps unmasked in the peak search (nMasks 0). The record must say so."""
    import logging
    m, _ = _master(tmp_path)
    with caplog.at_level(logging.WARNING):
        rec = _run(tmp_path, m, omega_sign=1)
    assert rec["mask_from"] is None
    assert any("exchange/mask" in r.getMessage() for r in caplog.records)
    z = zarr.open(str(tmp_path / "work" / "1" / "S_000001.MIDAS.zip"), "r")
    assert "exchange/mask" not in z
