"""Smoke tests for viz/static.py — Agg backend, no display required.

These use a synthetic CalibrationResult built from in-memory data so
they don't need the MIDASCalibrant binary to have run.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

# Force headless backend before any matplotlib import inside viz.
matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from midas_auto_calibrate import CalibrationResult, DetectorGeometry
from midas_auto_calibrate.viz import static as viz


_CORR_HEADER_SUMMARY = (
    "Lsd,ybcFit,zbcFit,ty,tz,p0,p1,p2,p3,MeanStrain,StdStrain\n"
    "657436.9,685.5,921.0,0.193,0.447,1.2e-4,1.2e-4,-6.9e-4,-13.7,4.5,1.2\n"
)
_CORR_TABLE_HEADER = (
    "%Eta Strain RadFit EtaCalc DiffCalc RadCalc Ideal2Theta Outlier "
    "YRawCorr ZRawCorr RingNr RadGlobal IdealR Fit2Theta IdealA FitA DeltaR DeltaA\n"
)


def _write_fake_corr(path: Path, n_rings: int = 4, n_eta: int = 360) -> None:
    """Synthetic corr.csv with deterministic ΔR(η) structure per ring."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write(_CORR_HEADER_SUMMARY)
        f.write(_CORR_TABLE_HEADER)
        eta_grid = np.linspace(-180, 180, n_eta, endpoint=False)
        for ring in range(1, n_rings + 1):
            r_ideal = 200.0 * ring  # pixels
            # k=2 harmonic with ring-dependent amplitude
            delta_r_um = 5.0 * ring * np.cos(np.radians(2 * eta_grid))
            for eta, dr in zip(eta_grid, delta_r_um):
                r_fit = r_ideal + dr
                y = 1000 + r_fit * np.cos(np.radians(eta))
                z = 1000 + r_fit * np.sin(np.radians(eta))
                f.write(
                    f"{eta:.3f} 1e-4 {r_fit * 172:.3f} {eta:.3f} 0.0 "
                    f"{r_ideal * 172:.3f} 3.17 0 {y:.3f} {z:.3f} {ring} "
                    f"{r_fit * 172:.3f} {r_ideal * 172:.3f} 3.17 5.4 5.4 "
                    f"{dr:.3f} 0.0\n"
                )


def _synthetic_result(tmp_path: Path) -> CalibrationResult:
    corr_path = tmp_path / "fake.corr.csv"
    _write_fake_corr(corr_path)
    history = [
        {"Iter": 0.0, "MeanStrain_ppm": 25.0, "StdStrain_ppm": 3.0},
        {"Iter": 1.0, "MeanStrain_ppm": 12.4, "StdStrain_ppm": 2.1},
        {"Iter": 2.0, "MeanStrain_ppm": 5.1, "StdStrain_ppm": 1.3},
        {"Iter": 3.0, "MeanStrain_ppm": 4.6, "StdStrain_ppm": 1.1},
    ]
    return CalibrationResult(
        geometry=DetectorGeometry(
            lsd=657_000, ybc=1000, zbc=1000, px=172.0,
            mean_strain=4.6, std_strain=1.1,
        ),
        pseudo_strain=4.6,
        pseudo_strain_std=1.1,
        convergence_history=history,
        corr_csv_path=corr_path,
        work_dir=tmp_path,
    )


def test_convergence_renders(tmp_path):
    result = _synthetic_result(tmp_path)
    out = tmp_path / "convergence.png"
    fig = viz.convergence(result, save=out)
    assert out.exists()
    assert out.stat().st_size > 1000
    fig.clf()


def test_convergence_requires_history(tmp_path):
    result = CalibrationResult(
        geometry=DetectorGeometry(),
        pseudo_strain=0.0,
        pseudo_strain_std=0.0,
        convergence_history=None,
    )
    with pytest.raises(ValueError, match="convergence history"):
        viz.convergence(result)


def test_rings_overlay_with_synthetic_image(tmp_path):
    result = _synthetic_result(tmp_path)
    img = np.random.default_rng(0).integers(0, 1000, size=(2000, 2000))
    out = tmp_path / "rings.png"
    fig = viz.rings_overlay(result, img, save=out)
    assert out.exists()
    fig.clf()


def test_residual_heatmap_renders(tmp_path):
    result = _synthetic_result(tmp_path)
    out = tmp_path / "residual.png"
    fig = viz.residual_heatmap(result, save=out)
    assert out.exists()
    fig.clf()


def test_fourier_harmonics_finds_k2(tmp_path):
    """Synthetic ΔR = 5·ring·cos(2η) → amplitude peak at k=2."""
    result = _synthetic_result(tmp_path)
    fig = viz.fourier_harmonics(result)
    ax = fig.axes[0]
    # Heatmap data lives on the AxesImage.
    amps = ax.images[0].get_array()
    # For each ring, k=2 (column index 1) should dominate.
    k_dominant = np.argmax(amps, axis=1)
    assert np.all(k_dominant == 1), (
        f"k=2 did not dominate (got indices {k_dominant})"
    )
    fig.clf()


def test_distortion_field_renders(tmp_path):
    result = _synthetic_result(tmp_path)
    out = tmp_path / "distortion.png"
    fig = viz.distortion_field(result, save=out)
    assert out.exists()
    fig.clf()


def test_inspect_bundle_writes_multiple_plots(tmp_path):
    result = _synthetic_result(tmp_path)
    img = np.random.default_rng(0).integers(0, 1000, size=(2000, 2000))
    written = viz.inspect(result, image=img, out_dir=tmp_path / "out")
    # Convergence + rings + residual + fourier + distortion
    assert set(written.keys()) == {
        "convergence", "rings", "residual", "fourier", "distortion",
    }
    for path in written.values():
        assert path.exists()


def test_inspect_skips_missing_pieces(tmp_path):
    """No image + no corr.csv → only convergence (if history present)."""
    result = CalibrationResult(
        geometry=DetectorGeometry(),
        pseudo_strain=0.0,
        pseudo_strain_std=0.0,
        convergence_history=[{"Iter": 0.0, "MeanStrain_ppm": 1.0, "StdStrain_ppm": 0.1}],
        corr_csv_path=None,
    )
    written = viz.inspect(result, image=None, out_dir=tmp_path / "out")
    assert set(written.keys()) == {"convergence"}
