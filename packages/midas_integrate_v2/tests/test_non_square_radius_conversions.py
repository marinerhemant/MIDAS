"""Non-square pixels: every conversion of a v2 radius to 2theta / Q must use the MEAN pitch.

``pixel_to_REta`` returns R_px in units of px_mean = (pxY_eff + pxZ_eff) / 2, so tan(2theta) = R_px * px_mean / Lsd.
Several modules used ``spec.pxY`` instead (wrong by pxY / mean, +33 % at 200 x 100 um). Pins:

1. ``IntegrationSpec.mean_pixel_um``: square, non-square, pxZ unset (= square), hex.
2. The forward map and the mean-pitch conversion agree with the true lab-frame angle (independent geometry).
3. pxZ unset gives the same forward map as an explicit pxZ = pxY (it used to collapse the Z axis).
4. io.milk and io.zarr_gsas (``reta_map``, Q-mode edges) convert with the mean pitch.
5. GISAXS pixel -> q uses pxY for columns and pxZ for rows.
6. Source scan: no radius conversion in midas_integrate_v2 / midas_pdf passes ``spec.pxY`` as the pitch.
7. v1 mapper and Map.bin writer warn about non-square pixels.
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import math
import re
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch

from midas_integrate.params import IntegrationParams
from midas_integrate_v2 import spec_from_v1_params
from midas_integrate_v2.forward import eval_pixel_REta


def _spec(pxy, pxz, NY=40, NZ=32, **kw):
    p = IntegrationParams(
        NrPixelsY=NY, NrPixelsZ=NZ, pxY=pxy, pxZ=pxz, Lsd=20_000.0,
        BC_y=NY / 2.0 + 0.37, BC_z=NZ / 2.0 - 0.41, RhoD=float(NY),
        RMin=1.0, RMax=26.0, RBinSize=1.0, EtaMin=-180.0, EtaMax=180.0, EtaBinSize=30.0, SubPixelLevel=1,
        Wavelength=0.18,
    )
    for k, v in kw.items():
        setattr(p, k, v)
    return spec_from_v1_params(p)


def _true_two_theta(spec):
    """Lab-frame 2theta of each pixel (zero tilt): atan(hypot(Yc, Zc) / Lsd), Yc, Zc in um with per-axis pitches."""
    NY, NZ = spec.NrPixelsY, spec.NrPixelsZ
    Y, Z = np.meshgrid(np.arange(NY, dtype=float), np.arange(NZ, dtype=float))
    pxy, pxz = spec.effective_pxYZ()
    rho = np.hypot((float(spec.BC_y) - Y) * pxy, (Z - float(spec.BC_z)) * pxz)
    return np.arctan(rho / float(spec.Lsd))


# ---- 1 -----------------------------------------------------------------------------------
def test_mean_pixel_um_cases():
    assert _spec(200.0, 200.0).mean_pixel_um() == 200.0
    assert _spec(172.0, 200.0).mean_pixel_um() == 186.0
    s = _spec(150.0, 150.0)
    s.pxZ = 0.0                                   # unset => square
    assert s.mean_pixel_um() == 150.0 and s.effective_pxYZ() == (150.0, 150.0)
    h = _spec(150.0, 150.0)
    h.lattice = "hex_offset_y"
    h.Apothem = 30.0
    assert h.mean_pixel_um() == pytest.approx(0.5 * (60.0 + 30.0 * math.sqrt(3.0)))


# ---- 2 -----------------------------------------------------------------------------------
@pytest.mark.parametrize("pxy,pxz", [(172.0, 200.0), (200.0, 100.0), (150.0, 150.0)])
def test_forward_R_with_mean_pitch_is_the_true_angle(pxy, pxz):
    s = _spec(pxy, pxz)
    with torch.no_grad():
        R, _ = eval_pixel_REta(s)
    tth = np.arctan(R.numpy() * s.mean_pixel_um() / float(s.Lsd))
    np.testing.assert_allclose(tth, _true_two_theta(s), rtol=1e-12, atol=1e-15)
    if abs(pxy - pxz) > 1:                         # the old conversion (pxY alone) is demonstrably wrong here
        old = np.arctan(R.numpy() * pxy / float(s.Lsd))
        assert np.abs(old / _true_two_theta(s) - 1).max() > 0.02


# ---- 3 -----------------------------------------------------------------------------------
def test_pxZ_unset_means_square():
    a = _spec(150.0, 150.0)
    b = _spec(150.0, 150.0)
    b.pxZ = 0.0
    with torch.no_grad():
        Ra, Ea = eval_pixel_REta(a)
        Rb, Eb = eval_pixel_REta(b)
    torch.testing.assert_close(Ra, Rb, rtol=0, atol=0)
    torch.testing.assert_close(Ea, Eb, rtol=0, atol=0)


# ---- 4 -----------------------------------------------------------------------------------
def test_milk_unit_conversion_uses_mean_pitch():
    from midas_integrate_v2.io.milk import _spec_to_pyfai_unit
    s = _spec(172.0, 200.0)
    with torch.no_grad():
        R, _ = eval_pixel_REta(s)
    got = _spec_to_pyfai_unit(s, R.numpy().ravel(), unit="2th_deg")
    np.testing.assert_allclose(got, np.degrees(_true_two_theta(s)).ravel(), rtol=1e-12, atol=1e-13)


def test_zarr_reta_map_and_q_edges_use_mean_pitch():
    from midas_integrate_v2.io.zarr_gsas import reta_map
    s = _spec(172.0, 200.0)
    with torch.no_grad():
        R, _ = eval_pixel_REta(s)
    r0 = float(R[10, 12])
    true_tth = float(np.degrees(_true_two_theta(s)[10, 12]))
    s.RMin, s.RMax, s.RBinSize = r0 - 0.5, r0 + 2.5, 1.0          # bin 0 is centred on r0
    out = reta_map(s, bin_area=np.ones((s.n_r_bins, s.n_eta_bins)))
    assert out[1][0, 0] == pytest.approx(true_tth, rel=1e-12)


# ---- 5 -----------------------------------------------------------------------------------
def test_gisaxs_uses_per_axis_pitch():
    from midas_integrate_v2.grazing.gisaxs import pixel_to_qy_qz
    s = _spec(172.0, 200.0)
    by, bz = float(s.BC_y), float(s.BC_z)
    Y = torch.tensor([by + 10.0, by], dtype=torch.float64)
    Z = torch.tensor([bz, bz + 10.0], dtype=torch.float64)
    qy, qz = pixel_to_qy_qz(Y, Z, spec=s, incidence_angle_deg=0.0)
    k = 2 * math.pi / float(s.Wavelength)
    L = float(s.Lsd)
    # alpha = 0: qy = k dy / sqrt(L^2 + dy^2 + dz^2), qz = k dz / ... - 0 ; dy along columns (pxY), dz along rows (pxZ)
    dy0, dz1 = 10.0 * 172.0, 10.0 * 200.0
    assert float(qy[0]) == pytest.approx(k * dy0 / math.sqrt(L * L + dy0 * dy0), rel=1e-12)
    assert float(qz[1]) == pytest.approx(k * dz1 / math.sqrt(L * L + dz1 * dz1), rel=1e-12)


# ---- 6 -----------------------------------------------------------------------------------
def test_no_radius_conversion_passes_pxY_as_the_pitch():
    root = Path(__file__).resolve().parents[2]
    bad = re.compile(r"px_um\s*=\s*\w+(\[\w+\])?\.pxY|\*\s*\w+(\[\w+\])?\.pxY\s*/|\.pxY\s*\*\s*R")
    hits = []
    for pkg in ("midas_integrate_v2/midas_integrate_v2",):   # midas_pdf is scanned by its own tests
        for f in (root / pkg).rglob("*.py"):
            for i, line in enumerate(f.read_text().splitlines(), 1):
                if bad.search(line) and not line.lstrip().startswith("#"):
                    hits.append(f"{f.relative_to(root)}:{i}: {line.strip()}")
    assert not hits, "radius converted with pxY instead of spec.mean_pixel_um():\n" + "\n".join(hits)


# ---- 7 -----------------------------------------------------------------------------------
def test_v1_mapper_and_map_writer_warn_on_non_square_pixels(tmp_path):
    from midas_integrate.detector_mapper import build_map
    from midas_integrate_v2 import HardBinGeometry, write_map_bin_from_geometry
    p = IntegrationParams(NrPixelsY=24, NrPixelsZ=24, pxY=172.0, pxZ=200.0, Lsd=1_000_000.0, BC_y=12.3, BC_z=11.7, RhoD=24.0,
                          RMin=1.0, RMax=10.0, RBinSize=1.0, EtaMin=-180.0, EtaMax=180.0, EtaBinSize=60.0, SubPixelLevel=1)
    with pytest.warns(UserWarning, match="single-pitch"):
        build_map(p, auto_load=False, verbose=False, use_numba=False)
    p.pxZ = 172.0
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=".*single-pitch.*")
        build_map(p, auto_load=False, verbose=False, use_numba=False)
    s = _spec(172.0, 200.0, NY=24, NZ=24)
    s.RMin, s.RMax = 1.0, 10.0
    with pytest.warns(UserWarning, match="MEAN-pixel"):
        write_map_bin_from_geometry(HardBinGeometry.from_spec(s), s, tmp_path)
    sq = _spec(172.0, 172.0, NY=24, NZ=24)
    sq.RMin, sq.RMax = 1.0, 10.0
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=".*MEAN-pixel.*")
        write_map_bin_from_geometry(HardBinGeometry.from_spec(sq), sq, tmp_path)


# ---- 8. default R range and provenance for non-square pixels ---------------------------------
def test_bootstrap_default_rmax_is_the_physical_corner_for_non_square_pixels():
    from midas_integrate_v2.bootstrap import estimate_initial_spec
    NY, NZ, by, bz = 120, 80, 61.0, 41.0
    img = np.ones((NZ, NY))
    kw = dict(NrPixelsY=NY, NrPixelsZ=NZ, Lsd_um=100_000.0, Wavelength_A=0.18, initial_BC=(by, bz), mask=np.zeros((NZ, NY), bool))
    s = estimate_initial_spec(img, pxY_um=172.0, pxZ_um=200.0, **kw)
    corner_um = math.hypot(max(s.BC_y, NY - s.BC_y) * 172.0, max(s.BC_z, NZ - s.BC_z) * 200.0)
    assert s.RMax + 5.0 == pytest.approx(corner_um / 186.0, rel=1e-6)
    sq = estimate_initial_spec(img, pxY_um=150.0, pxZ_um=150.0, **kw)
    assert sq.RMax + 5.0 == pytest.approx(math.hypot(max(sq.BC_y, NY - sq.BC_y), max(sq.BC_z, NZ - sq.BC_z)), rel=1e-12)


# ---------------------------------------------------------------------------------------------------------------
# dac.gasket_mask and compat.pyfai: per-axis pitch
# ---------------------------------------------------------------------------------------------------------------

def test_gasket_mask_wedge_is_physical_for_non_square_pixels():
    from midas_integrate_v2.dac.gasket_mask import build_gasket_mask

    # pxY = 200, pxZ = 100 um. Index offsets (dy, dz) = (10, 20) are 2000 um in both axes: a physical 45 deg direction.
    NY, NZ, BC = 61, 61, (30.0, 30.0)
    kw = dict(BC=BC, eta_open_deg=(40.0, 50.0), symmetry="single")
    m_phys = build_gasket_mask(NY, NZ, px_um=(200.0, 100.0), **kw)
    m_idx = build_gasket_mask(NY, NZ, **kw)                      # default: square pixels, index-space angle
    assert m_phys[30 + 20, 30 + 10], "45 deg physical direction must be inside the 40-50 deg wedge"
    assert not m_idx[30 + 20, 30 + 10], "negative control: the index-space angle there is 63 deg, outside the wedge"
    assert not m_phys[30 + 10, 30 + 10], "(10, 10) index is physically 26.6 deg, outside the wedge"
    assert m_idx[30 + 10, 30 + 10]
    # square pitch passed explicitly == default
    assert np.array_equal(build_gasket_mask(NY, NZ, px_um=(150.0, 150.0), **kw), m_idx)


def test_eta_coverage_per_ring_uses_physical_geometry_for_non_square_pixels():
    from midas_integrate_v2.dac.gasket_mask import build_gasket_mask, eta_coverage_per_ring

    spec = _spec(200.0, 100.0, NY=61, NZ=61, BC_y=30.0, BC_z=30.0)
    pitches = spec.effective_pxYZ()
    assert pitches == (200.0, 100.0)
    ring = np.array([8.0])            # mean-pixel units: 1200 um, fully on the detector
    m = build_gasket_mask(61, 61, BC=(30.0, 30.0), eta_open_deg=(-30.0, 30.0), symmetry="single", px_um=pitches)
    cov = float(eta_coverage_per_ring(spec, m, ring, capture_radius_px=1.0)[0])
    assert abs(cov - 60.0 / 360.0) < 0.03, cov
    # negative control: a mask built on the index-space angle covers a different physical fraction of the same ring
    m_idx = build_gasket_mask(61, 61, BC=(30.0, 30.0), eta_open_deg=(-30.0, 30.0), symmetry="single")
    cov_idx = float(eta_coverage_per_ring(spec, m_idx, ring, capture_radius_px=1.0)[0])
    assert abs(cov_idx - 60.0 / 360.0) > 0.03, cov_idx


def test_make_pyfai_integrator_pxZ_unset_is_square_and_non_square_keeps_both_pitches(monkeypatch):
    """pyFAI's dim1 is the ROW axis (MIDAS Z) and dim2 the column axis (MIDAS Y): pixel1 = pxZ, pixel2 = pxY, poni1 from BC_z, poni2 from BC_y."""
    import sys
    import types

    captured = {}

    class _AI:
        def __init__(self, **kw):
            captured.update(kw)

    fake = types.ModuleType("pyFAI")
    fake.AzimuthalIntegrator = _AI
    monkeypatch.setitem(sys.modules, "pyFAI", fake)
    from midas_integrate_v2.compat.pyfai import make_pyfai_integrator

    spec = _spec(172.0, 0.0)                      # pxZ unset = square
    make_pyfai_integrator(spec)
    assert captured["pixel1"] == pytest.approx(172e-6) and captured["pixel2"] == pytest.approx(172e-6)
    assert captured["poni1"] == pytest.approx((float(spec.BC_z) + 0.5) * 172e-6)       # pxZ = 0 used to collapse this to 0
    assert captured["poni2"] == pytest.approx((float(spec.BC_y) + 0.5) * 172e-6)

    captured.clear()
    spec = _spec(200.0, 100.0)
    make_pyfai_integrator(spec)
    assert captured["pixel1"] == pytest.approx(100e-6) and captured["pixel2"] == pytest.approx(200e-6)
    assert captured["poni1"] == pytest.approx((float(spec.BC_z) + 0.5) * 100e-6)
    assert captured["poni2"] == pytest.approx((float(spec.BC_y) + 0.5) * 200e-6)



def test_eta_coverage_per_ring_radius_axis_is_mean_pixel_units():
    from midas_integrate_v2.dac.gasket_mask import eta_coverage_per_ring

    spec = _spec(200.0, 100.0, NY=61, NZ=61, BC_y=30.0, BC_z=30.0)
    yy, zz = np.meshgrid(np.arange(61), np.arange(61), indexing="xy")
    r_um = np.hypot((yy - 30.0) * 200.0, (zz - 30.0) * 100.0)
    mask = (r_um > 1000.0) & (r_um < 1400.0)               # a physical annulus around 1200 um = 8 mean pixels (mean = 150 um)
    cov = float(eta_coverage_per_ring(spec, mask, np.array([8.0]), capture_radius_px=1.0)[0])
    assert cov > 0.9, cov                                  # R / pxY (= 200 um) would look at 1400-1800 um and find no open pixel


# ---------------------------------------------------------------------------------------------------------------
# Survivors of the statistics lens: pxZ-unset and hex specs at every site, Q-mode edges, SA/POL pitches, hex writer warning
# ---------------------------------------------------------------------------------------------------------------

def _pxz_unset(pxy=150.0):
    s = _spec(pxy, pxy)
    s.pxZ = 0.0
    return s


def _hex(apothem=30.0):
    s = _spec(150.0, 150.0)            # raw pxY, pxZ deliberately different from the apothem-derived pitches
    s.lattice = "hex_offset_y"
    s.Apothem = torch.tensor(apothem, dtype=torch.float64)
    s.LatticeOrientation = torch.tensor(0.0, dtype=torch.float64)
    return s


@pytest.mark.parametrize("mk", [lambda: _spec(172.0, 200.0), _pxz_unset, _hex], ids=["nonsquare", "pxZ_unset", "hex"])
def test_zarr_q_mode_edges_are_r_of_q_in_mean_pixels(mk):
    from midas_integrate_v2.io.zarr_gsas import _r_bin_edges
    s = mk()
    s.QMin, s.QMax, s.QBinSize = 1.0, 3.0, 0.5
    assert s.q_mode_active
    lo, hi = _r_bin_edges(s)
    lam, L, px = float(s.Wavelength), float(s.Lsd), s.mean_pixel_um()
    q0 = np.array([1.0, 1.5, 2.0, 2.5])
    expect_lo = (L / px) * np.tan(2.0 * np.arcsin(q0 * lam / (4.0 * math.pi)))      # independent of the module's helper
    np.testing.assert_allclose(lo, expect_lo, rtol=1e-12)
    assert hi[0] == pytest.approx(lo[1], rel=1e-12)


def test_pixel_divisor_pxZ_unset_equals_explicit_square_spec():
    from midas_integrate_v2.binning.pixel_correction import pixel_divisor_from_spec
    a, b = _pxz_unset(), _spec(150.0, 150.0)
    for s in (a, b):
        s.SolidAngleCorrection, s.PolarizationCorrection, s.PolarizationFraction = 1, 1, 0.9
        s.tx, s.ty = torch.tensor(0.3, dtype=torch.float64), torch.tensor(-0.2, dtype=torch.float64)
    da, db = pixel_divisor_from_spec(a), pixel_divisor_from_spec(b)
    torch.testing.assert_close(da, db, rtol=1e-12, atol=0.0)
    # negative control: a pxZ of 0 taken literally would collapse the solid-angle denominator
    a2 = _pxz_unset()
    a2.SolidAngleCorrection = 1
    a2.tx, a2.ty = torch.tensor(0.3, dtype=torch.float64), torch.tensor(-0.2, dtype=torch.float64)
    assert torch.isfinite(pixel_divisor_from_spec(a2)).all()


def test_pixel_divisor_hex_uses_apothem_derived_pitches():
    from midas_calibrate_v2.forward.geometry import build_tilt_matrix
    from midas_integrate_v2.binning.pixel_correction import pixel_divisor_from_spec
    from midas_integrate_v2.corrections.intensity import SolidAngleCorrection
    s = _hex()
    s.SolidAngleCorrection = 1
    s.tx, s.ty = torch.tensor(0.3, dtype=torch.float64), torch.tensor(-0.2, dtype=torch.float64)
    div = pixel_divisor_from_spec(s)
    py, pz = s.effective_pxYZ()
    assert (py, pz) == pytest.approx((60.0, 30.0 * math.sqrt(3.0)))
    zs, ys = torch.meshgrid(torch.arange(s.NrPixelsZ, dtype=torch.float64), torch.arange(s.NrPixelsY, dtype=torch.float64), indexing="ij")

    def sa(pxy, pxz):
        return SolidAngleCorrection()(ys.reshape(-1), zs.reshape(-1), Ycen=s.BC_y, Zcen=s.BC_z,
                                       TRs=build_tilt_matrix(s.tx, s.ty, s.tz), Lsd=s.Lsd,
                                       pxY=torch.tensor(pxy, dtype=torch.float64), pxZ=torch.tensor(pxz, dtype=torch.float64)).clamp(min=1e-12)
    torch.testing.assert_close(div, sa(py, pz).to(torch.float64), rtol=1e-12, atol=0.0)
    assert not torch.allclose(div, sa(150.0, 150.0).to(torch.float64), rtol=1e-6)         # raw pitches would differ


def test_map_writer_warns_for_hex_pixels_too(tmp_path):
    from midas_integrate_v2 import SubpixelBinGeometry, write_map_bin_from_geometry
    s = _hex()
    s.RMin, s.RMax, s.RBinSize = 1.0, 10.0, 1.0
    s.pixel_shape = "rect" if hasattr(s, "pixel_shape") else getattr(s, "pixel_shape", None)
    with pytest.warns(UserWarning, match="MEAN-pixel"):
        write_map_bin_from_geometry(SubpixelBinGeometry.from_spec(s), s, tmp_path)


def test_source_scan_second_family_of_slip_patterns():
    """Broader than test 6: assignments and keyword forms that carry spec.pxY into a radius conversion."""
    root = Path(__file__).resolve().parents[2]
    bad = re.compile(
        r"\bpx(_um)?\s*=\s*(float\()?(torch\.as_tensor\()?(self\.)?spec\.pxY\b"          # px = spec.pxY / float(spec.pxY)
        r"|px_um\s*=\s*(float\()?(self\.)?spec\.pxY\b"
        r"|px_um\s*=\s*\w*\.?effective_pxYZ\(\)\[0\]"
        r"|\*\s*(float\()?(self\.)?spec\.pxY\)?\s*/\s*(L|Lsd|float)"
        r"|/\s*(float\()?(self\.)?spec\.pxY\b"
    )
    allow = {  # legitimate per-axis or square-pixel uses, reviewed one by one (path suffix, substring)
        ("grazing/gisaxs.py", "spec.pxY"),                 # columns use pxY by design
    }
    hits = []
    for pkg in ("midas_integrate_v2/midas_integrate_v2",):   # midas_pdf is scanned by its own tests
        for f in (root / pkg).rglob("*.py"):
            rel = str(f.relative_to(root / pkg))
            for i, line in enumerate(f.read_text().splitlines(), 1):
                if line.lstrip().startswith("#") or not bad.search(line):
                    continue
                if any(rel.endswith(a) and b in line for a, b in allow):
                    continue
                hits.append(f"{f.relative_to(root)}:{i}: {line.strip()}")
    assert not hits, "possible mean-pixel radius converted with a single axis pitch:\n" + "\n".join(hits)
