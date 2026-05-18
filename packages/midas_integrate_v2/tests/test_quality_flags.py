"""Item 23 — Per-frame quality flag sidecar."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from midas_integrate_v2.streaming import compute_quality_flags


def _write_synthetic_integrated_h5(path: Path, n=10, n_r=64, *,
                                     bad_frames=(3,), with_eta_cov=True):
    profiles = np.tile(
        100.0 + 50.0 * np.sin(np.linspace(0, 4 * np.pi, n_r)),
        (n, 1),
    ) + np.random.default_rng(0).normal(0, 1, size=(n, n_r))
    for k in bad_frames:
        profiles[k] = profiles[k] + 200.0  # planted RMS jump
    frame_ids = np.array([f"frame_{i:05d}" for i in range(n)], dtype="S")
    with h5py.File(path, "w") as f:
        f.create_dataset("profiles", data=profiles)
        f.create_dataset("r_axis_px", data=np.arange(n_r, dtype=np.float64))
        f.create_dataset("frame_ids", data=frame_ids)
        if with_eta_cov:
            meta = {
                "package": "midas_integrate_v2",
                "version": "test",
                "extra": {"eta_coverage_per_ring": [0.95, 0.4, 0.92]},
            }
        else:
            meta = {"package": "midas_integrate_v2"}
        f.attrs["metadata_json"] = json.dumps(meta)


def test_rms_jump_flagged(tmp_path: Path):
    h5_path = tmp_path / "scan.h5"
    _write_synthetic_integrated_h5(h5_path, bad_frames=(3, 7))
    # rms/norm scaling: 200-amplitude shift on a ~800-norm reference
    # produces rms_to_ref ~ 0.03; threshold tuned accordingly.
    out = compute_quality_flags(h5_path, rms_threshold=0.02)
    flagged = [
        rec for rec in out["per_frame"] if "rms_jump" in rec["flags"]
    ]
    assert len(flagged) >= 2  # both planted bad frames flagged
    assert {rec["frame_id"] for rec in flagged}.issuperset(
        {"frame_00003", "frame_00007"}
    )


def test_eta_coverage_low_flagged(tmp_path: Path):
    h5_path = tmp_path / "scan_dac.h5"
    _write_synthetic_integrated_h5(
        h5_path, n=4, with_eta_cov=True, bad_frames=(),
    )
    out = compute_quality_flags(
        h5_path, rms_threshold=10.0, eta_coverage_min=0.5,
    )
    # η-coverage entry includes 0.4 < 0.5 threshold; flagged on every frame
    flagged = [
        rec for rec in out["per_frame"] if "eta_coverage_low" in rec["flags"]
    ]
    assert len(flagged) == 4


def test_sidecar_written(tmp_path: Path):
    h5_path = tmp_path / "scan.h5"
    sidecar = tmp_path / "scan_quality.json"
    _write_synthetic_integrated_h5(h5_path)
    compute_quality_flags(h5_path, sidecar_path=sidecar)
    assert sidecar.exists()
    parsed = json.loads(sidecar.read_text())
    assert "global" in parsed and "per_frame" in parsed
    assert parsed["global"]["n_frames"] == 10
