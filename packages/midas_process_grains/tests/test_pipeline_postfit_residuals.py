"""The spot-aware ('adaptive') sidecar: /residuals is POST-fit (FitBestFinal),
/residuals_prefit is the FitBest seed, and E7 reads the pre-fit table.

The shared ``tiny_run_dir`` fixture writes FitBest rows with zero positions,
which the decomposition drops (the radial basis is undefined at the beam
centre); here both tables get real positions with a known radial offset.
"""
from pathlib import Path

import numpy as np
import pytest

from midas_process_grains.compute.residual_decomposition import RESIDUAL_SOURCES

PRE_BIAS_UM, POST_BIAS_UM, R_UM = 30.0, 1.5, 150_000.0


def _fill(fb, bias):
    out = fb.copy()
    for s in range(out.shape[0]):
        v = np.flatnonzero(out[s, :, 0] > 0)
        for j, row in enumerate(v):
            eta = -np.pi + 2 * np.pi * (j + 0.5) / max(len(v), 1)
            y_e, z_e = -R_UM * np.sin(eta), R_UM * np.cos(eta)
            k = (R_UM + bias) / R_UM
            out[s, row, 1:4] = (y_e * k, z_e * k, 10.0)
            out[s, row, 7:10] = (y_e, z_e, 10.0)
            out[s, row, 19] = 0.1
    return out


def _run(rd: Path, with_final: bool):
    from midas_process_grains.pipeline import ProcessGrains
    fbp = rd / "Output" / "FitBest.bin"
    fb = np.fromfile(fbp, dtype=np.float64).reshape(-1, 5000, 22)
    _fill(fb, PRE_BIAS_UM).tofile(fbp)
    if with_final:
        _fill(fb, POST_BIAS_UM)[:, ::-1].copy().tofile(rd / "Output" / "FitBestFinal.bin")
    pg = ProcessGrains.from_param_file(rd / "paramstest.txt", device="cpu")
    return pg.run(mode="adaptive")


def test_adaptive_sidecar_post_and_pre(tiny_run_dir: Path, tmp_path: Path):
    h5py = pytest.importorskip("h5py")
    result = _run(tiny_run_dir, with_final=True)
    out = tmp_path / "out"
    result.write(out, h5=False, diagnostics_h5=True)
    with h5py.File(out / "processgrains_diagnostics.h5", "r") as f:
        assert f["residuals"].attrs["source"] == RESIDUAL_SOURCES["residuals"]
        assert f["residuals_prefit"].attrs["source"] == RESIDUAL_SOURCES["residuals_prefit"]
        post = f["residuals/spot_table"][:, 6]
        pre = f["residuals_prefit/spot_table"][:, 6]
        assert post.size and pre.size
        np.testing.assert_allclose(post, POST_BIAS_UM, atol=1e-3)
        np.testing.assert_allclose(pre, PRE_BIAS_UM, atol=1e-3)
        # post-fit table covers the same attributed spots as the pre-fit one
        np.testing.assert_array_equal(np.sort(f["residuals/spot_table"][:, 1]),
                                      np.sort(f["residuals_prefit/spot_table"][:, 1]))


def test_adaptive_sidecar_without_final_has_no_postfit(tiny_run_dir: Path, tmp_path: Path):
    h5py = pytest.importorskip("h5py")
    result = _run(tiny_run_dir, with_final=False)
    out = tmp_path / "out"
    result.write(out, h5=False, diagnostics_h5=True)
    with h5py.File(out / "processgrains_diagnostics.h5", "r") as f:
        assert "residuals" not in f
        assert "residuals_prefit" in f
