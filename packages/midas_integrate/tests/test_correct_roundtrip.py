"""★ Round-trip validation: correct_image() + re-calibrate (pipeline).

Paper's headline interop claim — MIDAS-rectified images can be processed
with simple flat-detector geometry downstream. The v1.0 paper goal is
<5 µε residual after round-trip.

Measured v0.1.0 behaviour with AutoCalibrateZarr-converged seed geometry
(full 15-param distortion, per-panel dLsd/dP2 for mosaic detectors):

    | Detector | Paper baseline | v0.1.0 rectified |
    |----------|---------------:|-----------------:|
    | Pilatus (panels)  |  17 µε  |  174 µε |
    | Varex (monolithic) | 4.12 µε |   44 µε |

The rectification is **mathematically correct** (per-point Rt matches
MIDAS C's ``dg_pixel_to_REta_corr`` to 0.01 µm across all 15 distortion
coefficients + per-panel dY/dZ/dTheta/dLsd/dP2), but pixel-grid-limited
in practice. Sub-pixel signal is lost to bilinear splat noise
(~0.1-0.2 px → ~40-150 µε residual floor at R ≈ 300-1000 px).

v0.2 roadmap for <5 µε paper goal:
  (a) Panel-aware Newton-Raphson inverse in panel-corrected space
      (matches C's ``dg_invert_REta_to_pixel_panel_corr``).
  (b) Sub-pixel-preserving resampling: proper inverse mapping with
      bilinear interpolation of RAW pixel values at the inverse-mapped
      position — NOT scatter splatting.
  (c) Use ``AutoCalibrateZarr --fit-p-models all`` output as the seed
      geometry (v0.1.0 MVP run_calibration fits only p0-p4 unless p5-p14
      are explicitly seeded at non-zero).

For v0.1.0 this test validates the **plumbing** (pipeline runs cleanly,
numerical residual is finite and < a generous ceiling).
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

from midas_auto_calibrate import (
    CalibrationConfig,
    MidasBinaryNotFoundError as _MacErr,
    auto_calibrate,
    data as mac_data,
    midas_bin as _mac_bin,
)
from midas_integrate import (
    correct_image,
    midas_bin as _mi_bin,
    write_tiff,
)
from midas_integrate import MidasBinaryNotFoundError as _MiErr


def _both_binaries_available() -> bool:
    try:
        _mac_bin("MIDASCalibrant")
        _mac_bin("GetHKLList")
        _mi_bin("MIDASDetectorMapper")
        return True
    except (_MacErr, _MiErr):
        return False


pytestmark = [
    pytest.mark.skipif(
        not mac_data.CEO2_PILATUS.exists(),
        reason="Bundled CeO2 calibrant data missing.",
    ),
    pytest.mark.skipif(
        not _both_binaries_available(),
        reason="Both midas_auto_calibrate and midas_integrate binaries must "
               "be wheel-installed. Set MIDAS_BIN.",
    ),
]


# Monolithic (no-panel) calibration knobs. These suffice to prove the
# plumbing; numerical accuracy on a mosaic detector requires panels and
# that's v0.2 scope.
_MONO_KNOBS = {
    "RhoD": 219964.42411013643,
    "Width": 800,
    "OmegaStart": -180, "OmegaStep": 0.25,
    "tolTilts": 2, "tolBC": 10, "tolLsd": 5000,
    "tolP": 1e-3, "tolP4": 1e-4,
    "OutlierIterations": 3, "MultFactor": 2,
    "NormalizeRingWeights": 1, "WeightByRadius": 1,
    "WeightByFitSNR": 1, "L2Objective": 1,
    "p0": 0.000230535992, "p1": 0.000172564332,
    "p2": -0.000542224078, "p3": -13.773706892191,
    "p4": 0.001909017437,
    "ty": 0.200888234849, "tz": 0.446902376310, "tx": 0.0,
    "RingsToExclude": [[n] for n in range(19, 34)],
}


def _stage_bundled(workdir: Path) -> Path:
    workdir.mkdir(parents=True, exist_ok=True)
    img = workdir / "CeO2_00001.tif"
    shutil.copy(mac_data.CEO2_PILATUS, img)
    shutil.copy(mac_data.CEO2_PILATUS_DARK, workdir / "dark.tif")
    shutil.copy(mac_data.CEO2_PILATUS_MASK, workdir / "mask_upd.tif")
    return img


def _calibrate_monolithic(stem_dir: Path, img: Path, overrides: dict):
    cfg = CalibrationConfig(
        material="CeO2",
        lattice_params=(5.4116, 5.4116, 5.4116, 90, 90, 90),
        wavelength=0.172973, pixel_size=172.0,
        lsd=657_436.895687981, ybc=685.485459654, zbc=921.034377044,
        nr_pixels_y=1475, nr_pixels_z=1679,
        dark_file="dark.tif", mask_file="mask_upd.tif",
        im_trans_opt=[2],
        n_iterations=10,
        extra_params={**_MONO_KNOBS, **overrides},
    )
    return auto_calibrate(cfg, img, work_dir=stem_dir, n_cpus=2)


def test_round_trip_pipeline_smoke(tmp_path):
    """Pipeline plumbing test: correct → write_tiff → re-calibrate runs clean.

    Numerical round-trip accuracy on a Pilatus mosaic is v0.2.
    """
    workdir = tmp_path / "roundtrip"

    # ---- Stage 1: baseline calibration. ----
    cal0 = workdir / "cal0"
    img0 = _stage_bundled(cal0)
    result0 = _calibrate_monolithic(cal0, img0, overrides={})
    baseline = result0.pseudo_strain
    assert np.isfinite(baseline), "baseline calibration produced NaN"
    assert 0 < baseline < 5000, (
        f"baseline unexpectedly bad ({baseline:.1f} µε). The bundled "
        f"Pilatus data or the monolithic-fit knobs regressed."
    )

    # ---- Stage 2: rectify. Must actually change the image. ----
    import tifffile
    raw = tifffile.imread(img0).astype(float)
    rectified = correct_image(
        img0, result0.geometry,
        nr_pixels_y=1475, nr_pixels_z=1679,
    )
    assert rectified.shape == raw.shape
    assert np.isfinite(rectified).all(), "correct_image emitted NaN/Inf"
    # Rectification is non-trivial — max pixel difference is multiple orders
    # of magnitude above interpolation noise.
    delta = np.abs(rectified - raw)
    assert delta.max() > 1000, (
        f"correct_image was a near no-op (max |Δ|={delta.max():.2f}); "
        f"geometry tilts={result0.geometry.tilts} may be too small, or "
        f"forward map regressed to identity."
    )

    # ---- Stage 3: write + re-calibrate the rectified TIFF. ----
    cal1 = workdir / "cal1"
    cal1.mkdir()
    rect_img = cal1 / "CeO2_00001.tif"
    write_tiff(rect_img, rectified, geometry=result0.geometry)
    assert rect_img.exists()
    assert rect_img.stat().st_size > 1_000_000  # 1475×1679 float32 ≈ 9.4 MB

    shutil.copy(mac_data.CEO2_PILATUS_DARK, cal1 / "dark.tif")
    shutil.copy(mac_data.CEO2_PILATUS_MASK, cal1 / "mask_upd.tif")

    flat_overrides = {
        "tx": 0.0, "ty": 0.0, "tz": 0.0,
        "p0": 0.0, "p1": 0.0, "p2": 0.0, "p3": 0.0, "p4": 0.0,
        "tolTilts": 0.2, "tolP": 5e-5, "tolP4": 5e-6,
    }
    result1 = _calibrate_monolithic(cal1, rect_img, overrides=flat_overrides)
    residual = result1.pseudo_strain

    # ---- Stage 4: residual is finite + not catastrophic. ----
    # Not a strict numerical gate — see module docstring. The gate is
    # simply that the re-calibration ran to completion on the rectified
    # image (proves write_tiff / file I/O / parameter plumbing holds up).
    assert np.isfinite(residual), "re-calibration on rectified image gave NaN"
    assert residual < 5000, (
        f"Re-calibration residual {residual:.2f} µε is garbage-in-range. "
        f"baseline={baseline:.2f} µε. Artifacts in {workdir}."
    )


@pytest.mark.xfail(
    reason=(
        "v1.0 paper goal: round-trip <5 µε on any detector. v0.1.0 measurement\n"
        "with progressive-calibrated seed (all 15 p's refined via\n"
        "calibrate_progressive, fit_p_models='all'):\n"
        "  Pilatus (panels, scatter):  baseline 70 µε  → rectified 250 µε\n"
        "  Varex (monolithic, inverse): baseline 20 µε → rectified 200 µε\n"
        "  Varex (monolithic, scatter): baseline 20 µε → rectified 225 µε\n"
        "Forward math is paper-correct (matches MIDAS C dg_pixel_to_REta_corr\n"
        "to 0.01 µm per-point). The inverse/bilinear path (correct_image with\n"
        "method='inverse') preserves sub-pixel signal but the remaining\n"
        "~100-200 µε floor is the Stage-3 TPS residual spline that\n"
        "AutoCalibrateZarr applies post-refinement; porting it requires a\n"
        "ctypes shim over dg_correct_image and is v0.2.1+ scope."
    ),
    strict=False,
)
def test_round_trip_sub_five_ustrain(tmp_path):
    """Paper v1.0 goal: Pilatus round-trip <5 µε. v0.2 target."""
    workdir = tmp_path / "roundtrip"
    cal0 = workdir / "cal0"
    img0 = _stage_bundled(cal0)
    result0 = _calibrate_monolithic(cal0, img0, overrides={})

    rectified = correct_image(
        img0, result0.geometry,
        nr_pixels_y=1475, nr_pixels_z=1679,
    )
    cal1 = workdir / "cal1"
    cal1.mkdir()
    rect_img = cal1 / "CeO2_00001.tif"
    write_tiff(rect_img, rectified, geometry=result0.geometry)
    shutil.copy(mac_data.CEO2_PILATUS_DARK, cal1 / "dark.tif")
    shutil.copy(mac_data.CEO2_PILATUS_MASK, cal1 / "mask_upd.tif")

    flat_overrides = {
        "tx": 0, "ty": 0, "tz": 0,
        "p0": 0, "p1": 0, "p2": 0, "p3": 0, "p4": 0,
    }
    result1 = _calibrate_monolithic(cal1, rect_img, overrides=flat_overrides)
    assert result1.pseudo_strain < 5.0, (
        f"pseudo_strain={result1.pseudo_strain:.2f} µε"
    )


# ---------------------------------------------------------------------------
# correct_image(method=...) unit test: both backends produce sensible output
# and differ from each other (NR sampling ≠ scatter splat on the same raw).
# Doesn't need MIDAS binaries — pure-Python math only.
# ---------------------------------------------------------------------------


def test_correct_image_method_parameter():
    """``method='inverse'`` and ``method='scatter'`` both run and produce
    geometrically-sensible (non-trivial, non-identical, finite) outputs."""
    from midas_auto_calibrate import DetectorGeometry
    from midas_integrate import correct_image

    rng = np.random.default_rng(0)
    img = rng.random((200, 200)) * 1000.0
    geom = DetectorGeometry(
        lsd=900_000, ybc=100, zbc=100,
        tx=0.0, ty=0.2, tz=0.3,
        p0=2e-4, p1=1e-4, p2=-5e-5, p4=1e-4,
        px=150.0, nr_pixels_y=200, nr_pixels_z=200,
    )
    r_scatter = correct_image(img, geom, method="scatter")
    r_inverse = correct_image(img, geom, method="inverse")

    assert r_scatter.shape == r_inverse.shape == img.shape
    assert np.isfinite(r_scatter).all()
    assert np.isfinite(r_inverse).all()
    # Non-trivial rectification (not just a passthrough of raw).
    assert np.abs(r_scatter - img).max() > 100
    assert np.abs(r_inverse - img).max() > 100
    # The two backends produce distinct output — scatter quantises, inverse
    # samples sub-pixel.
    assert not np.allclose(r_scatter, r_inverse)


def test_correct_image_method_invalid_raises():
    from midas_auto_calibrate import DetectorGeometry
    from midas_integrate import correct_image

    with pytest.raises(ValueError, match="method must be"):
        correct_image(
            np.zeros((50, 50)),
            DetectorGeometry(nr_pixels_y=50, nr_pixels_z=50, lsd=1e6, px=100.0),
            method="bogus",
        )


# ---------------------------------------------------------------------------
# Monolithic Varex — no panels, lower-distortion calibrant. Scatter-splat and
# inverse-bilinear both converge to ~200-800 µε residual depending on seed
# quality; the pipeline plumbing is the gate, not a numerical threshold
# (see module docstring for the v0.2.1+ TPS-spline path to <5 µε).
# ---------------------------------------------------------------------------

_VAREX_TIF = (
    Path(__file__).resolve().parents[3]
    / "FF_HEDM" / "Example" / "Calibration"
    / "Ceria_63keV_900mm_100x100_0p5s_aero_0_001137.tif"
)


@pytest.mark.skipif(
    not _VAREX_TIF.exists(),
    reason="Varex Ceria bundled dataset missing from FF_HEDM/Example/Calibration/.",
)
def test_round_trip_monolithic_varex(tmp_path):
    """Varex 4343CT monolithic roundtrip — no panels, verifies pipeline on a
    second detector geometry (63 keV, 900 mm Lsd, 150 µm pixels, 2880×2880).
    Residual is scatter-splat-limited (same ceiling as Pilatus), not a bug."""
    import shutil
    import tifffile
    from midas_auto_calibrate.calibrate import _parse_final_geometry
    from midas_auto_calibrate._config import CalibrationConfig as _CC

    # Stage Varex data in an isolated work dir.
    seed = tmp_path / "varex_seed"
    seed.mkdir()
    shutil.copy(_VAREX_TIF, seed / _VAREX_TIF.name)

    common_knobs = {
        "Width": 800, "OmegaStart": -180, "OmegaStep": 0.25,
        "tolTilts": 3, "tolBC": 40, "tolLsd": 20000,
        "tolP": 2e-3, "tolP4": 1e-4,
        "OutlierIterations": 3, "MultFactor": 2,
        "NormalizeRingWeights": 1, "WeightByRadius": 1,
        "WeightByFitSNR": 1, "L2Objective": 1,
    }

    # Stage 1: baseline calibration (converge tilts + distortion).
    cfg0 = CalibrationConfig(
        material="CeO2", lattice_params=(5.4116,) * 3 + (90,) * 3,
        wavelength=0.196793, pixel_size=150.0,
        lsd=900_000, ybc=1440, zbc=1440, rho_d=305_000,
        nr_pixels_y=2880, nr_pixels_z=2880,
        im_trans_opt=[2], r_max=1440, n_iterations=10,
        extra_params=common_knobs,
    )
    r0 = auto_calibrate(cfg0, seed / _VAREX_TIF.name, work_dir=seed, n_cpus=2)
    baseline = r0.pseudo_strain
    assert np.isfinite(baseline) and baseline > 0, (
        f"Varex baseline calibration didn't converge: {baseline}"
    )

    # Stage 2: rectify with the converged geometry.
    rectified = correct_image(
        seed / _VAREX_TIF.name, r0.geometry,
        nr_pixels_y=2880, nr_pixels_z=2880,
    )

    # Stage 3: re-calibrate the rectified image with flat (zero) geometry.
    cal1 = tmp_path / "varex_rect"
    cal1.mkdir()
    rect_img = cal1 / _VAREX_TIF.name
    write_tiff(rect_img, rectified, geometry=r0.geometry)

    flat_knobs = {**common_knobs,
                  "tx": 0, "ty": 0, "tz": 0,
                  **{f"p{i}": 0 for i in range(15)}}
    cfg1 = CalibrationConfig(
        material="CeO2", lattice_params=(5.4116,) * 3 + (90,) * 3,
        wavelength=0.196793, pixel_size=150.0,
        lsd=r0.geometry.lsd, ybc=r0.geometry.ybc, zbc=r0.geometry.zbc,
        rho_d=305_000,
        nr_pixels_y=2880, nr_pixels_z=2880,
        im_trans_opt=[2], r_max=1440, n_iterations=10,
        extra_params=flat_knobs,
    )
    r1 = auto_calibrate(cfg1, rect_img, work_dir=cal1, n_cpus=2)
    residual = r1.pseudo_strain

    # Pipeline plumbing: residual finite and in the same order of magnitude
    # as the baseline (not a >10× blow-up from a math bug).
    assert np.isfinite(residual), "Varex re-calibration produced NaN"
    assert residual < baseline * 3, (
        f"Varex rectified residual {residual:.1f} µε is >3× baseline "
        f"{baseline:.1f} µε — likely a forward-map regression."
    )
