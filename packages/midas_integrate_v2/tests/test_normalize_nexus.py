"""Item 17 — FrameNormalizer.from_nexus_h5 auto-detect."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from midas_integrate_v2.streaming import FrameNormalizer


def _write_synthetic_nexus(path: Path, n_frames: int, *,
                             monitor_path="entry/instrument/monitor/data",
                             exposure_path="entry/instrument/detector/count_time",
                             transmission_path=None,
                             monitor_values=None,
                             exposure_values=None,
                             transmission_values=None):
    monitor_values = (np.linspace(1000.0, 1100.0, n_frames)
                       if monitor_values is None else monitor_values)
    exposure_values = (np.full(n_frames, 0.5)
                        if exposure_values is None else exposure_values)
    with h5py.File(path, "w") as f:
        f.create_dataset(monitor_path, data=monitor_values)
        f.create_dataset(exposure_path, data=exposure_values)
        if transmission_path is not None and transmission_values is not None:
            f.create_dataset(transmission_path, data=transmission_values)


def test_from_nexus_basic_normalisation(tmp_path: Path):
    path = tmp_path / "scan.nx"
    n = 5
    monitor = np.array([100.0, 110.0, 95.0, 105.0, 102.0])
    exposure = np.full(n, 0.2)
    _write_synthetic_nexus(path, n,
                            monitor_values=monitor, exposure_values=exposure)
    nrm = FrameNormalizer.from_nexus_h5(path)
    img = np.full((4, 4), 1000.0)
    out0 = nrm("frame_00000", img)
    expected0 = 1000.0 / (monitor[0] * exposure[0] * 1.0)
    np.testing.assert_allclose(out0, expected0, rtol=1e-12)
    out2 = nrm("frame_00002", img)
    expected2 = 1000.0 / (monitor[2] * exposure[2] * 1.0)
    np.testing.assert_allclose(out2, expected2, rtol=1e-12)


def test_from_nexus_with_transmission(tmp_path: Path):
    path = tmp_path / "scan.nx"
    monitor = np.full(3, 100.0)
    exposure = np.full(3, 1.0)
    transmission = np.array([1.0, 0.5, 0.25])
    _write_synthetic_nexus(
        path, 3, monitor_values=monitor, exposure_values=exposure,
        transmission_path="entry/sample/transmission",
        transmission_values=transmission,
    )
    nrm = FrameNormalizer.from_nexus_h5(
        path, transmission_path="entry/sample/transmission",
    )
    img = np.full((4, 4), 100.0)
    out2 = nrm("frame_00002", img)
    expected2 = 100.0 / (100.0 * 1.0 * 0.25)
    np.testing.assert_allclose(out2, expected2, rtol=1e-12)


def test_from_nexus_missing_monitor_raises(tmp_path: Path):
    path = tmp_path / "bad.nx"
    with h5py.File(path, "w") as f:
        f.create_dataset("entry/instrument/detector/count_time", data=[1.0])
    with pytest.raises(KeyError):
        FrameNormalizer.from_nexus_h5(path)


def test_from_nexus_singleton_exposure_broadcasts(tmp_path: Path):
    path = tmp_path / "scan.nx"
    monitor = np.array([100.0, 200.0, 300.0])
    exposure = np.array([0.7])  # singleton broadcast
    _write_synthetic_nexus(path, 3,
                            monitor_values=monitor, exposure_values=exposure)
    nrm = FrameNormalizer.from_nexus_h5(path)
    img = np.full((4, 4), 1.0)
    out1 = nrm("frame_00001", img)
    expected1 = 1.0 / (200.0 * 0.7 * 1.0)
    np.testing.assert_allclose(out1, expected1, rtol=1e-12)
