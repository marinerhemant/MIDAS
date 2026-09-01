"""global_powder: η-coverage correction of GrainRadius.

The stage exists because ``calc_radius`` normalises each spot by the powder
sum it actually *observed* on that ring. When a ring is not fully sampled in
η — a pinwheel panel seeing ~75°, or a single panel whose outer rings run off
the edge — that denominator is short while the numerator is not, so every
volume on the ring is inflated by ``360°/coverage``.

The two properties worth protecting here are opposites:

* on a **complete** ring the stage must do *nothing at all*, byte for byte,
  because it is enabled for every existing single-panel user; and
* on a **truncated** ring it must recover the known factor.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from midas_pipeline.config import PipelineConfig, ScanGeometry
from midas_pipeline.stages import global_powder
from midas_pipeline.stages._base import StageContext


# 1-ID GE5, the geometry the reference measurement was made on.
LSD_UM = 767765.75
PX_UM = 200.0
N_PIXELS = 2048
Y_BC, Z_BC = 1022.76327, 974.64506
RING1_UM = 64279.18          # 321.4 px — comfortably inside the panel
RING20_UM = 291420.0         # 1457.1 px — only the corners remain


# -----------------------------------------------------------------------------
#  Fixture construction
# -----------------------------------------------------------------------------

def _paramstest(rings: list[tuple[int, float]], *, extra: str = "") -> str:
    lines = [
        f"LsdFit {LSD_UM:.6f}",
        f"YBCFit {Y_BC:.6f}",
        f"ZBCFit {Z_BC:.6f}",
        "txFit 0.000000",
        "tyFit 0.000000",
        "tzFit 0.000000",
        f"NrPixelsY {N_PIXELS};",
        f"NrPixelsZ {N_PIXELS};",
        f"px {PX_UM:.6f};",
    ]
    for rn, _ in rings:
        lines.append(f"RingNumbers {rn};")
    for _, rad in rings:
        lines.append(f"RingRadii {rad:.6f};")
    if extra:
        lines.append(extra)
    return "\n".join(lines) + "\n"


def _radius_csv(spots: list[tuple[int, float, float]]) -> str:
    """``spots`` = [(ring, eta_deg, integrated_intensity), ...]."""
    out = ["SpotID IntegratedIntensity Omega YCen ZCen IMax MinOme MaxOme "
           "Radius Theta Eta DeltaOmega NImgs RingNr"]
    for i, (ring, eta, inten) in enumerate(spots, start=1):
        out.append(f"{i} {inten:.6f} 0 0 0 0 0 0 0 0 {eta:.6f} 0.25 2 {ring}")
    return "\n".join(out) + "\n"


def _input_all(rows: list[tuple[int, float, float]]) -> str:
    """``rows`` = [(ring, eta_deg, grain_radius_um), ...]."""
    out = ["YLab ZLab Omega GrainRadius SpotID RingNumber Eta Ttheta DetID"]
    for i, (ring, eta, gr) in enumerate(rows, start=1):
        out.append(f"0 0 0 {gr:.6f} {i} {ring} {eta:.6f} 0 1")
    return "\n".join(out) + "\n"


def _ctx(tmp_path: Path) -> StageContext:
    params = tmp_path / "P.txt"
    params.write_text("SpaceGroup 225\n")
    cfg = PipelineConfig(
        result_dir=str(tmp_path / "run"),
        params_file=str(params),
        scan=ScanGeometry.ff(),
        device="cpu",
        dtype="float64",
    )
    layer_dir = tmp_path / "Layer1"
    layer_dir.mkdir(exist_ok=True)
    log_dir = layer_dir / "midas_log"
    log_dir.mkdir(exist_ok=True)
    return StageContext(config=cfg, layer_nr=1, layer_dir=layer_dir,
                        log_dir=log_dir)


def _build(tmp_path: Path, *, rings, spots, rows, extra="") -> StageContext:
    ctx = _ctx(tmp_path)
    d = ctx.layer_dir
    (d / "paramstest.txt").write_text(_paramstest(rings, extra=extra))
    (d / "Radius_StartNr_1_EndNr_10.csv").write_text(_radius_csv(spots))
    (d / "InputAll.csv").write_text(_input_all(rows))
    return ctx


def _uniform_eta(ring: int, n: int, inten: float = 100.0,
                 lo: float = -180.0, hi: float = 180.0):
    """``n`` spots spread evenly over [lo, hi) on ``ring``."""
    return [(ring, lo + (hi - lo) * (k + 0.5) / n, inten) for k in range(n)]


# -----------------------------------------------------------------------------
#  The no-op guarantee on complete rings
# -----------------------------------------------------------------------------

def test_full_coverage_leaves_input_all_byte_identical(tmp_path):
    """A ring that is fully on the panel must not be touched at all.

    This is the property that makes the stage safe to enable for every
    existing single-panel run: no silent rescaling of anyone's grain sizes.
    """
    rings = [(1, RING1_UM)]
    ctx = _build(
        tmp_path,
        rings=rings,
        spots=_uniform_eta(1, 720),
        rows=[(1, -180.0 + k * 0.5, 5.0 + 0.001 * k) for k in range(720)],
    )
    before = (ctx.layer_dir / "InputAll.csv").read_bytes()

    res = global_powder.run(ctx)

    assert (ctx.layer_dir / "InputAll.csv").read_bytes() == before
    assert res.metrics["n_spots_rescaled"] == 0
    assert res.metrics["min_coverage_frac"] == pytest.approx(1.0)


def test_full_coverage_scale_is_exactly_one(tmp_path):
    """Not merely ~1: the ratio must be 1.0 to the bit, or the no-op leaks."""
    ctx = _build(
        tmp_path,
        rings=[(1, RING1_UM)],
        # deliberately lumpy in η — the scale must still be exactly 1
        spots=(_uniform_eta(1, 300, inten=100.0, lo=-180.0, hi=0.0)
               + _uniform_eta(1, 60, inten=900.0, lo=0.0, hi=180.0)),
        rows=[(1, 0.0, 7.0)],
    )
    global_powder.run(ctx)
    cov = _read_coverage(ctx.layer_dir)
    assert cov[1]["ScaleVolume"] == 1.0
    assert cov[1]["ScaleRadius"] == 1.0


# -----------------------------------------------------------------------------
#  Recovering the known factor on a truncated ring
# -----------------------------------------------------------------------------

def test_half_covered_ring_recovers_the_known_factor(tmp_path):
    """Explicit EtaCoverage over half the ring ⇒ scale = covered fraction.

    With a flat η profile the interpolant across the gap is that same
    constant, so the correction collapses to exactly ``coverage/360`` — the
    scalar answer, arrived at without being hard-coded.
    """
    r0 = 6.0
    ctx = _build(
        tmp_path,
        rings=[(1, RING1_UM)],
        spots=_uniform_eta(1, 360, inten=50.0, lo=-90.0, hi=90.0),
        rows=[(1, 0.0, r0)],
        extra="EtaCoverage_Det1 1 -90.000000 89.999000",
    )
    global_powder.run(ctx)

    cov = _read_coverage(ctx.layer_dir)
    # the arc spans exactly the 180 populated bins, half the ring
    expected = 0.5
    assert cov[1]["ScaleVolume"] == pytest.approx(expected, rel=1e-9)
    assert cov[1]["CoverageFrac"] == pytest.approx(expected, rel=1e-9)

    gr = _read_grain_radii(ctx.layer_dir)
    assert gr[0] == pytest.approx(r0 * expected ** (1.0 / 3.0), rel=1e-6)


def test_correction_shrinks_never_grows(tmp_path):
    """Truncation inflates volume, so the fix can only reduce the radius."""
    r0 = 10.0
    ctx = _build(
        tmp_path,
        rings=[(1, RING1_UM)],
        spots=_uniform_eta(1, 200, inten=50.0, lo=-45.0, hi=45.0),
        rows=[(1, 0.0, r0)],
        extra="EtaCoverage_Det1 1 -45.000000 45.000000",
    )
    global_powder.run(ctx)
    assert _read_grain_radii(ctx.layer_dir)[0] < r0


def test_disc_model_takes_the_square_root(tmp_path):
    """``DiscModel 1`` is 2-D (R = √(V/π)), so the exponent is ½ not ⅓."""
    r0 = 8.0
    ctx = _build(
        tmp_path,
        rings=[(1, RING1_UM)],
        spots=_uniform_eta(1, 360, inten=50.0, lo=-90.0, hi=90.0),
        rows=[(1, 0.0, r0)],
        extra=("EtaCoverage_Det1 1 -90.000000 89.999000\n"
               "DiscModel 1;"),
    )
    global_powder.run(ctx)
    expected = 0.5
    assert _read_grain_radii(ctx.layer_dir)[0] == pytest.approx(
        r0 * math.sqrt(expected), rel=1e-6)


# -----------------------------------------------------------------------------
#  Coverage derived from panel geometry
# -----------------------------------------------------------------------------

def test_geometry_detects_single_panel_edge_truncation():
    """No EtaCoverage rows: coverage must come from the panel geometry.

    This is the case the stage previously skipped outright. Ring 1 at 321 px
    is complete; ring 20 at 1457 px keeps only the detector corners.
    """
    text = _paramstest([(1, RING1_UM), (20, RING20_UM)])
    arcs = global_powder._coverage_from_geometry(text)
    assert set(arcs) == {1}, "arcs are keyed by a single synthetic panel id"

    cov = global_powder._coverage_per_bin(arcs)
    frac1 = (cov[1] > 0).sum() / global_powder.N_ETA_BINS
    frac20 = (cov[20] > 0).sum() / global_powder.N_ETA_BINS
    assert frac1 == pytest.approx(1.0, abs=0.01)
    assert frac20 < 0.05
    # the whole point: the truncated ring's volume error is large
    assert 1.0 / frac20 > 20.0


def test_geometry_path_needs_a_ring_table():
    """Missing RingRadii ⇒ no coverage rather than a wrong one."""
    text = _paramstest([]) + "RingNumbers 1;\n"
    assert global_powder._coverage_from_geometry(text) == {}


def test_no_coverage_available_is_a_skip_not_a_silent_pass(tmp_path):
    ctx = _ctx(tmp_path)
    d = ctx.layer_dir
    (d / "paramstest.txt").write_text("SpaceGroup 225\n")
    (d / "Radius_StartNr_1_EndNr_10.csv").write_text(
        _radius_csv(_uniform_eta(1, 10)))
    before = _input_all([(1, 0.0, 5.0)])
    (d / "InputAll.csv").write_text(before)

    res = global_powder.run(ctx)

    assert res.metrics.get("skipped") is True
    assert (d / "InputAll.csv").read_text() == before


# -----------------------------------------------------------------------------
#  Gap filling
# -----------------------------------------------------------------------------

def test_fill_uncovered_interpolates_across_the_wrap():
    """η is periodic: a gap straddling ±180° is filled from both sides.

    Covered bins sit at 0–9 (value 10) and 180–189 (value 20). The gap
    190–359 must run *down* from 20 toward 10 as it approaches the wrap. A
    non-circular interpolant would clamp it at 20 instead.
    """
    n = global_powder.N_ETA_BINS
    hat = np.zeros(n)
    cov = np.zeros(n, dtype=np.int64)
    hat[0:10] = 10.0
    cov[0:10] = 1
    hat[180:190] = 20.0
    cov[180:190] = 1

    filled, is_full = global_powder._fill_uncovered(hat, cov)

    assert is_full is False
    assert filled[189] == pytest.approx(20.0)
    assert filled[0] == pytest.approx(10.0)
    # just before wrapping back onto bin 0, we should be near 10, not 20
    assert filled[359] < 12.0
    # and monotone through the descending gap
    assert filled[250] > filled[330] > filled[359]


def test_fill_uncovered_is_identity_when_complete():
    n = global_powder.N_ETA_BINS
    hat = np.linspace(1.0, 2.0, n)
    cov = np.ones(n, dtype=np.int64)
    filled, is_full = global_powder._fill_uncovered(hat, cov)
    assert is_full is True
    assert np.array_equal(filled, hat)
    assert filled.sum() == hat.sum()      # exact, so the ratio is exactly 1.0


# -----------------------------------------------------------------------------
#  The flat-Î(η) premise is checked, not assumed
# -----------------------------------------------------------------------------

def test_profile_cv_small_when_flat_large_when_textured():
    n = global_powder.N_ETA_BINS
    cov = np.ones(n, dtype=np.int64)
    flat = np.full(n, 5.0)
    assert global_powder._profile_cv(flat, cov) == pytest.approx(0.0, abs=1e-12)

    eta = np.deg2rad(np.arange(n) - 180.0)
    textured = 5.0 * (1.0 + 0.9 * np.cos(2 * eta)) ** 4
    assert global_powder._profile_cv(textured, cov) > global_powder.PROFILE_CV_WARN


def test_profile_cv_is_nan_when_too_little_of_the_ring_is_intact():
    n = global_powder.N_ETA_BINS
    hat = np.full(n, 3.0)
    cov = np.zeros(n, dtype=np.int64)
    cov[:30] = 1
    assert math.isnan(global_powder._profile_cv(hat, cov))


def test_reported_profile_cv_comes_from_full_coverage_rings(tmp_path):
    """A flat, complete ring should report a small CV in the metrics."""
    ctx = _build(
        tmp_path,
        rings=[(1, RING1_UM)],
        spots=_uniform_eta(1, 3600, inten=100.0),
        rows=[(1, 0.0, 5.0)],
    )
    res = global_powder.run(ctx)
    assert res.metrics["profile_cv"] < 0.05


# -----------------------------------------------------------------------------
#  Helpers
# -----------------------------------------------------------------------------

def _read_coverage(layer_dir: Path) -> dict[int, dict[str, float]]:
    path = layer_dir / "PowderCoverage.csv"
    lines = path.read_text().splitlines()
    head = lines[0].split()
    out: dict[int, dict[str, float]] = {}
    for line in lines[1:]:
        toks = line.split()
        row = {k: float(v) for k, v in zip(head, toks)}
        out[int(row["RingNr"])] = row
    return out


def _read_grain_radii(layer_dir: Path) -> list[float]:
    lines = (layer_dir / "InputAll.csv").read_text().splitlines()
    col = lines[0].split().index("GrainRadius")
    return [float(ln.split()[col]) for ln in lines[1:] if ln.strip()]
