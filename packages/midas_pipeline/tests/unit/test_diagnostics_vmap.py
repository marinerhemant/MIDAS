"""Tests for ``midas_pipeline.diagnostics.vmap``.

Covers:
- All five plotters render a non-empty PNG without raising.
- ``write_v_map_tif`` writes either a real TIFF (when ``tifffile`` is
  installed) or a ``.npy`` fallback, with the right dtype + shape.
- ``write_k_per_ring_table`` writes the 7-column schema and round-trips
  via ``np.loadtxt``.
- Empty-input edge cases:
    * ``plot_per_grain_v_histograms`` with no valid grains still writes
      a placeholder PNG.
    * ``plot_spot_residuals`` with all-zero intensities renders the
      "no valid spots" placeholder.
- ``refine_vmap`` stage's ``emit_diagnostics`` flag controls whether
  the ``diag/`` directory and ``Recons/v_map.tif`` appear.
"""

from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from pathlib import Path

import numpy as np
import pytest

from midas_pipeline.diagnostics.vmap import (
    plot_loss_history,
    plot_per_grain_v_histograms,
    plot_spot_residuals,
    plot_v_map_overlay,
    write_k_per_ring_table,
    write_v_map_tif,
)


def _png_nonempty(p: Path) -> bool:
    if not p.exists():
        return False
    b = p.read_bytes()
    # PNG signature
    return b.startswith(b"\x89PNG\r\n\x1a\n") and len(b) > 200


# ---------------------------------------------------------- write_v_map_tif


def test_write_v_map_tif_writes_image(tmp_path):
    """4×3 grid in xy → 2-D image of shape (3, 4)."""
    voxel_pos = np.array(
        [[x, y, 0.0] for y in (0.0, 5.0, 10.0) for x in (0.0, 5.0, 10.0, 15.0)],
        dtype=np.float64,
    )
    V = np.arange(12, dtype=np.float64)
    out = tmp_path / "v_map.tif"
    written = write_v_map_tif(voxel_pos, V, out)
    assert written.exists()
    if written.suffix == ".tif":
        try:
            import tifffile
            img = tifffile.imread(written)
            assert img.dtype == np.float32
            assert img.shape == (3, 4)
        except ImportError:
            pytest.skip("tifffile available branch — but skip if loader missing")
    else:
        img = np.load(written)
        assert img.dtype == np.float32
        assert img.shape == (3, 4)


# ---------------------------------------------------------- write_k_per_ring_table


def test_k_per_ring_table_round_trip(tmp_path):
    K = np.array([1.0, 4.2, 0.7])
    I_th = np.array([10.0, 22.0, 5.0])
    out = tmp_path / "k_per_ring.csv"
    write_k_per_ring_table(
        K, I_th, out,
        ring_numbers=np.array([1, 2, 3]),
        residual_stats={0: {"mean": 0.01, "std": 0.05, "n": 10},
                        1: {"mean": -0.02, "std": 0.04, "n": 25}},
    )
    arr = np.loadtxt(out, comments="#", skiprows=1)
    assert arr.shape == (3, 7)
    # columns: ring_idx, ring_number, K, I_theory, mean_log_resid, std_log_resid, n_spots
    assert (arr[:, 0] == [0, 1, 2]).all()
    assert (arr[:, 1] == [1, 2, 3]).all()
    np.testing.assert_allclose(arr[:, 2], K)
    np.testing.assert_allclose(arr[:, 3], I_th)
    np.testing.assert_allclose(arr[:, 4], [0.01, -0.02, 0.0])
    np.testing.assert_allclose(arr[:, 6], [10, 25, 0])


def test_k_per_ring_table_handles_empty_stats(tmp_path):
    K = np.array([2.0]); I_th = np.array([5.0])
    out = tmp_path / "k.csv"
    write_k_per_ring_table(K, I_th, out)
    arr = np.loadtxt(out, comments="#", skiprows=1)
    assert arr.ndim == 1 and arr.shape == (7,)


# ---------------------------------------------------------- plotters


def test_plot_loss_history(tmp_path):
    out = tmp_path / "loss.png"
    plot_loss_history(np.array([1.0, 0.1, 0.01, 1e-3, 1e-5]), out)
    assert _png_nonempty(out)


def test_plot_loss_history_handles_all_zeros(tmp_path):
    out = tmp_path / "loss_zero.png"
    plot_loss_history(np.array([0.0, 0.0]), out)
    assert _png_nonempty(out)


def test_plot_spot_residuals(tmp_path):
    obs = np.array([100.0, 200.0, 50.0, 30.0])
    pred = obs * (1.0 + 0.1 * np.random.RandomState(0).randn(4))
    out = tmp_path / "resid.png"
    plot_spot_residuals(obs, pred, out)
    assert _png_nonempty(out)


def test_plot_spot_residuals_no_valid_spots(tmp_path):
    out = tmp_path / "resid_empty.png"
    plot_spot_residuals(np.zeros(3), np.zeros(3), out)
    assert _png_nonempty(out)


def test_plot_v_map_overlay(tmp_path):
    voxel_pos = np.array(
        [[x, y, 0.0] for y in (0.0, 5.0, 10.0) for x in (0.0, 5.0, 10.0)],
        dtype=np.float64,
    )
    V = np.arange(9, dtype=np.float64)
    grain_map = np.array([0]*3 + [1]*3 + [2]*3, dtype=np.int64)
    out = tmp_path / "overlay.png"
    plot_v_map_overlay(voxel_pos, V, grain_map, out)
    assert _png_nonempty(out)


def test_plot_per_grain_v_histograms(tmp_path):
    V = np.concatenate([
        np.random.RandomState(0).normal(1.0, 0.1, 30),
        np.random.RandomState(1).normal(2.0, 0.2, 50),
        np.random.RandomState(2).normal(0.5, 0.05, 20),
    ])
    gm = np.array([0]*30 + [1]*50 + [2]*20, dtype=np.int64)
    out = tmp_path / "hist.png"
    plot_per_grain_v_histograms(V, gm, out)
    assert _png_nonempty(out)


def test_plot_per_grain_v_histograms_no_grains(tmp_path):
    out = tmp_path / "hist_empty.png"
    plot_per_grain_v_histograms(
        np.array([]), np.array([-1, -1, -1], dtype=np.int64), out,
    )
    assert _png_nonempty(out)


def test_plot_per_grain_v_histograms_caps_panels(tmp_path):
    """``max_panels`` caps the number of subplots — file still renders."""
    V = np.arange(100, dtype=np.float64)
    gm = (np.arange(100) // 5).astype(np.int64)   # 20 grains
    out = tmp_path / "hist_capped.png"
    plot_per_grain_v_histograms(V, gm, out, max_panels=6)
    assert _png_nonempty(out)


# ---------------------------------------------------------- stage integration


def test_refine_vmap_stage_emits_diagnostics(tmp_path):
    """End-to-end check that the stage actually writes the diag artifacts."""
    # Reuse the synthetic layout helper from the existing vmap-stages test.
    from tests.unit.test_vmap_stages import _setup_refine_inputs, _make_ctx, _make_pipeline_config
    from midas_pipeline.stages import refine_vmap

    V_true, K_true = _setup_refine_inputs(tmp_path)
    cfg = _make_pipeline_config(
        tmp_path, run=True, refine_V=True, refine_K=False,
        max_iter=40, tolerance=1e-9,
        emit_diagnostics=True,
    )
    ctx = _make_ctx(tmp_path, cfg)
    res = refine_vmap.run(ctx)
    assert not res.skipped
    assert res.metrics.get("n_diag_artifacts", 0) >= 4

    diag = tmp_path / "diag"
    assert _png_nonempty(diag / "v_map_overlay.png")
    assert _png_nonempty(diag / "spot_residuals.png")
    assert _png_nonempty(diag / "refine_loss_history.png")
    assert _png_nonempty(diag / "per_grain_v_histograms.png")
    assert (diag / "k_per_ring.csv").exists()


def test_refine_vmap_stage_skips_diagnostics_when_disabled(tmp_path):
    from tests.unit.test_vmap_stages import _setup_refine_inputs, _make_ctx, _make_pipeline_config
    from midas_pipeline.stages import refine_vmap

    _setup_refine_inputs(tmp_path)
    cfg = _make_pipeline_config(
        tmp_path, run=True, refine_V=True, refine_K=False,
        max_iter=10, emit_diagnostics=False,
    )
    ctx = _make_ctx(tmp_path, cfg)
    res = refine_vmap.run(ctx)
    assert res.metrics.get("n_diag_artifacts", 0) == 0
    # diag dir should not be created
    assert not (tmp_path / "diag").exists()


# ---------------------------------------------------------- compare_modes


def _write_npz_v_map(path: Path, V, K, positions, grain_map):
    np.savez(
        path,
        V=V.astype(np.float64), K=K.astype(np.float64),
        positions_um=positions.astype(np.float64),
        grain_map=grain_map.astype(np.int64),
        I_theory=np.ones_like(K),
    )


def test_compare_modes_diff_report(tmp_path):
    """Build two synthetic V-maps + verify the diff report fields."""
    from midas_pipeline.diagnostics import run_compare_modes

    # Tiny shared grid: 3x3 in xy.
    pos = np.array(
        [[x, y, 0.0] for y in range(3) for x in range(3)],
        dtype=np.float64,
    )
    grain_map = np.array([0]*5 + [1]*4, dtype=np.int64)

    layer = tmp_path
    Vb = np.array([1.0, 1.1, 0.9, 1.2, 0.8, 2.0, 2.1, 1.9, 2.2])
    Vs = Vb * 1.05                 # soft mode "shifts" V by 5%
    Kb = np.array([3.0]); Ks = np.array([3.0 * 0.95])
    bin_npz = layer / "v_binary.npz"
    sft_npz = layer / "v_soft.npz"
    _write_npz_v_map(bin_npz, Vb, Kb, pos, grain_map)
    _write_npz_v_map(sft_npz, Vs, Ks, pos, grain_map)

    # Synthetic loss history files
    bin_loss = layer / "binloss.csv"
    sft_loss = layer / "sftloss.csv"
    np.savetxt(bin_loss, np.column_stack([np.arange(5), 10.0**-np.arange(5)]),
               header="iter loss", fmt=["%d", "%.6e"], comments="")
    np.savetxt(sft_loss, np.column_stack([np.arange(5), 10.0**-(np.arange(5)+1)]),
               header="iter loss", fmt=["%d", "%.6e"], comments="")

    res = run_compare_modes(
        layer, binary_v_map=bin_npz, soft_v_map=sft_npz,
        binary_loss_csv=bin_loss, soft_loss_csv=sft_loss,
    )
    assert res.n_voxels == 9
    assert res.n_rings == 1
    # mean |dV| = 0.05 * mean(V)
    assert abs(res.mean_abs_dV - 0.05 * Vb.mean()) < 1e-9
    # rel = 0.05 everywhere
    assert abs(res.mean_rel_dV - 0.05) < 1e-9
    assert res.final_loss_binary == pytest.approx(1e-4)
    assert res.final_loss_soft == pytest.approx(1e-5)
    # Artifacts: v_binary npz copy, v_soft npz copy, summary, k_compare, diff png, loss png
    out = Path(res.out_dir)
    assert (out / "compare_summary.csv").exists()
    assert (out / "k_per_ring_compare.csv").exists()
    assert (out / "v_map_diff.png").exists()
    assert (out / "loss_compare.png").exists()
    summary = np.loadtxt(out / "compare_summary.csv", comments="#", skiprows=1)
    # 2 grains
    assert summary.shape == (2, 5)
    # k_per_ring_compare schema
    k_arr = np.loadtxt(out / "k_per_ring_compare.csv", comments="#", skiprows=1)
    if k_arr.ndim == 1:
        k_arr = k_arr.reshape(1, -1)
    assert k_arr.shape == (1, 4)
    assert abs(k_arr[0, 3] - (-5.0)) < 1e-6  # delta_pct = (Ks - Kb)/Kb*100 = -5


def test_compare_modes_shape_mismatch_raises(tmp_path):
    from midas_pipeline.diagnostics import run_compare_modes
    pos1 = np.zeros((3, 3))
    pos2 = np.zeros((5, 3))
    _write_npz_v_map(tmp_path / "a.npz", np.ones(3), np.ones(1), pos1,
                     np.zeros(3, dtype=np.int64))
    _write_npz_v_map(tmp_path / "b.npz", np.ones(5), np.ones(1), pos2,
                     np.zeros(5, dtype=np.int64))
    with pytest.raises(ValueError, match="V-map shapes differ"):
        run_compare_modes(
            tmp_path,
            binary_v_map=tmp_path / "a.npz",
            soft_v_map=tmp_path / "b.npz",
        )
