"""The ID03 reader must recover a planted tilt field and refuse every malformed file.

The fixture is written exactly as the real Mg-4Al master HDF5 stores it: Bliss-style scan
entries named ``"<plane>.1"``, ``instrument/pco_ff/image`` (M, H, W), ``instrument/mu/data`` and
``instrument/chi/value`` (M,), and ``obpitch`` at either of the two paths seen on real data.
"""
import os

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")
pytest.importorskip("hdf5plugin")

from midas_dfxm import reduce_rocking  # noqa: E402
from midas_dfxm.io_id03 import list_id03_planes, load_id03_scan  # noqa: E402

pytestmark = pytest.mark.unit


def _mesh(n_mu=5, n_chi=4, H=24, W=24, amp=3000.0, ped=400.0, seed=0):
    """A zigzag mu x chi mesh (like the real scan) with a per-column mu-centroid gradient."""
    rng = np.random.default_rng(seed)
    mu_grid = 20.10 + 0.04 * np.arange(n_mu)
    chi_grid = -5.60 + 0.10 * np.arange(n_chi)
    xx = np.arange(W)
    centre_by_col = mu_grid[n_mu // 2] + 0.03 * (xx / (W - 1) - 0.5)   # planted gradient
    sig = 0.04 / 2.3548
    frames = np.empty((n_mu * n_chi, H, W), dtype=np.uint16)
    mu = np.empty(n_mu * n_chi)
    chi = np.empty(n_mu * n_chi)
    k = 0
    for r in range(n_chi):
        order = range(n_mu) if r % 2 == 0 else range(n_mu - 1, -1, -1)   # zigzag, like the beamline
        for j in order:
            lam = amp * np.exp(-0.5 * ((mu_grid[j] - centre_by_col[None, :]) / sig) ** 2) + ped
            lam = np.broadcast_to(lam, (H, W))
            frames[k] = rng.poisson(lam).astype(np.uint16)
            mu[k] = mu_grid[j]
            chi[k] = chi_grid[r]
            k += 1
    return frames, mu, chi


def _write_h5(path, entries):
    """``entries``: {entry_name: (frames, mu, chi, obpitch, obpitch_under_instrument)}."""
    with h5py.File(path, "w") as f:
        for name, (frames, mu, chi, obpitch, nest) in entries.items():
            g = f.create_group(f"{name}/instrument/pco_ff")
            g.create_dataset("image", data=frames)
            f.create_dataset(f"{name}/instrument/mu/data", data=mu)
            f.create_dataset(f"{name}/instrument/chi/value", data=chi)
            if obpitch is not None:
                base = f"{name}/instrument/positioners_start" if nest else f"{name}/positioners_start"
                f.create_dataset(f"{base}/obpitch", data=obpitch)


def test_recovers_the_planted_tilt_gradient(tmp_path):
    frames, mu, chi = _mesh()
    path = str(tmp_path / "campaign.h5")
    _write_h5(path, {"5.1": (frames, mu, chi, 16.1991, False)})

    scan = load_id03_scan(path, plane=5)
    assert scan.scan_type == "tilt2d" and set(scan.axes) == {"mu", "chi"}
    np.testing.assert_array_equal(scan.frames, frames.astype(np.float32))
    np.testing.assert_array_equal(scan.motors["mu"], mu)
    assert scan.meta["obpitch"] == pytest.approx(16.1991)
    assert scan.meta["entry"] == "5.1"
    assert any("obpitch = 16.1991" in n for n in scan.notes)

    maps = reduce_rocking(scan)
    # planted: mu centroid rises 0.03 deg end to end across 24 columns
    assert maps.scan_type == "tilt2d"
    mu_axis = maps.axes.index("mu")
    mu_centre = maps.centre_deg[..., mu_axis]
    lit = maps.lit
    col = np.broadcast_to(np.arange(24), (24, 24))
    fit = np.polyfit(col[lit], mu_centre[lit], 1)
    assert fit[0] * 23 == pytest.approx(0.03, abs=0.01)


def test_obpitch_found_under_instrument_path_too(tmp_path):
    frames, mu, chi = _mesh(seed=1)
    path = str(tmp_path / "campaign.h5")
    _write_h5(path, {"1.1": (frames, mu, chi, 10.0, True)})
    scan = load_id03_scan(path, plane=1)
    assert scan.meta["obpitch"] == pytest.approx(10.0)


def test_missing_obpitch_is_a_warning_not_a_failure(tmp_path):
    frames, mu, chi = _mesh(seed=2)
    path = str(tmp_path / "campaign.h5")
    _write_h5(path, {"1.1": (frames, mu, chi, None, False)})
    scan = load_id03_scan(path, plane=1)
    assert scan.meta["obpitch"] is None
    assert any("obpitch not found" in n for n in scan.notes)


def test_unknown_plane_lists_what_is_present(tmp_path):
    frames, mu, chi = _mesh(seed=3)
    path = str(tmp_path / "campaign.h5")
    _write_h5(path, {"1.1": (frames, mu, chi, 10.0, False),
                     "2.1": (frames, mu, chi, 12.0, False)})
    with pytest.raises(KeyError, match=r"\['1\.1', '2\.1'\]"):
        load_id03_scan(path, plane=9)
    assert list_id03_planes(path) == ["1.1", "2.1"]


def test_mismatched_motor_length_refused(tmp_path):
    frames, mu, chi = _mesh(seed=4)
    path = str(tmp_path / "campaign.h5")
    _write_h5(path, {"1.1": (frames, mu[:-1], chi[:-1], 10.0, False)})
    with pytest.raises(ValueError, match="Refusing to guess"):
        load_id03_scan(path, plane=1)


def test_roi_crops_before_any_full_frame_is_kept(tmp_path):
    frames, mu, chi = _mesh(seed=5)
    path = str(tmp_path / "campaign.h5")
    _write_h5(path, {"1.1": (frames, mu, chi, 10.0, False)})
    scan = load_id03_scan(path, plane=1, roi=(4, 20, 4, 20))
    assert scan.frames.shape == (len(mu), 16, 16)
    np.testing.assert_array_equal(scan.frames, frames[:, 4:20, 4:20].astype(np.float32))


def test_dark_is_subtracted_and_recorded(tmp_path):
    frames, mu, chi = _mesh(seed=6)
    path = str(tmp_path / "campaign.h5")
    _write_h5(path, {"1.1": (frames, mu, chi, 10.0, False)})
    scan = load_id03_scan(path, plane=1, dark=50.0)
    np.testing.assert_allclose(scan.frames, frames.astype(np.float32) - 50.0)
    assert scan.dark_level == 50.0
