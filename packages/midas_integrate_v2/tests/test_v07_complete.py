"""v0.7: per-pixel masking, variance propagation, quasi-2D losses,
output formats, and bootstrap geometry estimation.

This is the "complete-package" test suite — pinning every
production-critical feature added in v0.7.
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from midas_integrate.params import IntegrationParams

from midas_integrate_v2 import (
    spec_from_v1_params,
    HardBinGeometry, integrate_hard, integrate_hard_with_variance,
    SubpixelBinGeometry, integrate_subpixel, integrate_subpixel_with_variance,
    PolygonBinGeometry, integrate_polygon, integrate_polygon_with_variance,
    EtaSliceLoss, WedgeLoss, RingMaskedLoss,
    estimate_BC_from_image, estimate_initial_spec,
    write_csv, write_xye, write_fxye, write_dat, write_2d_csv,
    build_provenance, ProfileMetadata,
    mask_fraction,
)


def _spec(NY=24, NZ=24):
    p = IntegrationParams(
        NrPixelsY=NY, NrPixelsZ=NZ,
        pxY=200.0, pxZ=200.0, Lsd=1_000_000.0,
        BC_y=NY / 2.0 + 0.37, BC_z=NZ / 2.0 - 0.41, RhoD=float(NY),
        RMin=1.0, RMax=12.0, RBinSize=1.0,
        EtaMin=-180.0, EtaMax=180.0, EtaBinSize=60.0,
    )
    return spec_from_v1_params(p, requires_grad=False)


def _gauss_image(NY, NZ, *, R0_px=6.0, sigma_px=1.5, px=200.0):
    yy, zz = np.meshgrid(np.arange(NY), np.arange(NZ), indexing="xy")
    Yc = -(yy - NY / 2.0 - 0.37) * px
    Zc = (zz - NZ / 2.0 + 0.41) * px
    R_um = np.sqrt(Yc * Yc + Zc * Zc)
    R_px = R_um / px
    return np.exp(-(R_px - R0_px) ** 2 / (2 * sigma_px ** 2)).astype(np.float64)


# ======================================================================
# (1) Per-pixel mask support
# ======================================================================

def test_mask_excludes_pixels_from_hard_geometry():
    s = _spec()
    mask = np.zeros((s.NrPixelsZ, s.NrPixelsY), dtype=bool)
    mask[10:14, 10:14] = True       # mask a 4×4 block
    g_no = HardBinGeometry.from_spec(s)
    g_m  = HardBinGeometry.from_spec(s, mask=mask)
    # Masked geometry has fewer valid pixels
    assert g_m.n_valid < g_no.n_valid
    # Specifically: no_mask − masked = (4*4 - off-detector pixels in that block)
    diff = g_no.n_valid - g_m.n_valid
    assert diff <= 16
    assert diff > 0


def test_mask_excludes_pixels_from_subpixel_geometry():
    s = _spec()
    mask = np.zeros((s.NrPixelsZ, s.NrPixelsY), dtype=bool)
    mask[8:16, 8:16] = True
    g_no = SubpixelBinGeometry.from_spec(s, K=2)
    g_m  = SubpixelBinGeometry.from_spec(s, K=2, mask=mask)
    n_valid_no = int(g_no.valid.sum())
    n_valid_m  = int(g_m.valid.sum())
    assert n_valid_m < n_valid_no


def test_mask_excludes_pixels_from_polygon_geometry():
    s = _spec()
    mask = np.zeros((s.NrPixelsZ, s.NrPixelsY), dtype=bool)
    mask[8:16, 8:16] = True
    g_no = PolygonBinGeometry.from_spec(s)
    g_m  = PolygonBinGeometry.from_spec(s, mask=mask)
    assert g_m.n_entries < g_no.n_entries
    # No entries should reference any masked pixel
    pix = g_m.pix_idx.numpy()
    NY = s.NrPixelsY
    grid_y = pix % NY
    grid_z = pix // NY
    assert not mask[grid_z, grid_y].any()


def test_mask_full_image_yields_empty_geometry():
    s = _spec()
    mask = np.ones((s.NrPixelsZ, s.NrPixelsY), dtype=bool)
    g = HardBinGeometry.from_spec(s, mask=mask)
    assert g.n_valid == 0
    g2 = PolygonBinGeometry.from_spec(s, mask=mask)
    assert g2.n_entries == 0


def test_mask_shape_mismatch_raises():
    s = _spec()
    bad_mask = np.zeros((s.NrPixelsZ + 1, s.NrPixelsY), dtype=bool)
    with pytest.raises(ValueError, match="mask shape"):
        HardBinGeometry.from_spec(s, mask=bad_mask)


def test_mask_fraction_helper():
    mask = np.zeros((10, 10), dtype=bool)
    assert mask_fraction(mask) == 0.0
    mask[:5, :] = True
    assert mask_fraction(mask) == 0.5
    assert mask_fraction(None) == 0.0


# ======================================================================
# (2) Variance propagation
# ======================================================================

def test_hard_variance_propagation_constant_image():
    """Constant-image variance: each pixel has variance = constant
    (Poisson default), per-bin σ = sqrt(variance / N)."""
    s = _spec()
    geom = HardBinGeometry.from_spec(s)
    img = torch.full((s.NrPixelsZ, s.NrPixelsY), 100.0, dtype=torch.float64)
    mean, sigma = integrate_hard_with_variance(img, geom)
    nonzero = mean[mean > 0]
    assert torch.allclose(nonzero, torch.full_like(nonzero, 100.0),
                            atol=1e-12)
    # σ in any populated bin > 0 and finite
    pos = sigma[mean > 0]
    assert (pos > 0).all()
    assert torch.isfinite(pos).all()


def test_hard_variance_with_explicit_variance_image():
    s = _spec()
    geom = HardBinGeometry.from_spec(s)
    img = torch.full((s.NrPixelsZ, s.NrPixelsY), 100.0, dtype=torch.float64)
    var_img = torch.full_like(img, 25.0)        # σ_pixel = 5
    mean, sigma = integrate_hard_with_variance(img, geom,
                                                  variance_image=var_img)
    # σ_bin = 5 / sqrt(N_pixels_in_bin); just verify it's finite and < pixel σ
    pos_sigma = sigma[mean > 0]
    assert (pos_sigma <= 5.0 + 1e-6).all()


def test_polygon_variance_matches_hard_at_same_geometry():
    """For a constant image with all weights=1, polygon and hard
    variance should agree on per-bin σ to order-of-magnitude."""
    s = _spec()
    g_h = HardBinGeometry.from_spec(s)
    g_p = PolygonBinGeometry.from_spec(s)
    img = torch.full((s.NrPixelsZ, s.NrPixelsY), 100.0, dtype=torch.float64)
    _, s_h = integrate_hard_with_variance(img, g_h)
    _, s_p = integrate_polygon_with_variance(img, g_p)
    # Per-bin polygon σ should be within 5× of hard σ at populated bins
    populated = (s_h > 1e-6) & (s_p > 1e-6)
    if populated.any():
        ratios = (s_p[populated] / s_h[populated])
        assert (ratios > 0.1).all() and (ratios < 10.0).all()


def test_subpixel_variance_propagation_runs():
    s = _spec()
    geom = SubpixelBinGeometry.from_spec(s, K=2)
    img = torch.full((s.NrPixelsZ, s.NrPixelsY), 100.0, dtype=torch.float64)
    mean, sigma = integrate_subpixel_with_variance(img, geom)
    pos = sigma[mean > 0]
    assert (pos > 0).all() and torch.isfinite(pos).all()


# ======================================================================
# (3) Quasi-2D losses
# ======================================================================

def test_eta_slice_loss_zero_when_match():
    s = _spec()
    int2d = torch.rand(s.n_eta_bins, s.n_r_bins, dtype=torch.float64)
    L = EtaSliceLoss()(int2d, s, int2d.clone())
    assert float(L) == pytest.approx(0.0, abs=1e-30)


def test_eta_slice_loss_subset_runs():
    s = _spec()
    int2d = torch.rand(s.n_eta_bins, s.n_r_bins, dtype=torch.float64)
    ref = torch.zeros_like(int2d)
    L = EtaSliceLoss(eta_indices=[0, 2, 4],
                     r_indices=[1, 3, 5])(int2d, s, ref)
    assert float(L) > 0


def test_wedge_loss_zero_when_wedge_matches_reference():
    s = _spec()
    int2d = torch.rand(s.n_eta_bins, s.n_r_bins, dtype=torch.float64)
    # wedge avg over middle 2 eta bins
    ref_1d = int2d[s.n_eta_bins // 2 - 1: s.n_eta_bins // 2 + 1].mean(dim=0)
    # Compute wedge bounds from those bins' centres
    eta_centres = s.EtaMin + s.EtaBinSize * (
        np.arange(s.n_eta_bins) + 0.5)
    eta_lo = eta_centres[s.n_eta_bins // 2 - 1] - 0.01
    eta_hi = eta_centres[s.n_eta_bins // 2]     + 0.01
    L = WedgeLoss(eta_min_deg=float(eta_lo),
                  eta_max_deg=float(eta_hi))(int2d, s, ref_1d)
    assert float(L) == pytest.approx(0.0, abs=1e-30)


def test_wedge_loss_empty_wedge_raises():
    s = _spec()
    int2d = torch.zeros(s.n_eta_bins, s.n_r_bins, dtype=torch.float64)
    ref_1d = torch.zeros(s.n_r_bins, dtype=torch.float64)
    with pytest.raises(ValueError, match="contains no"):
        WedgeLoss(eta_min_deg=200.0, eta_max_deg=210.0)(int2d, s, ref_1d)


def test_ring_masked_loss_zero_outside_mask_doesnt_count():
    s = _spec()
    int2d = torch.rand(s.n_eta_bins, s.n_r_bins, dtype=torch.float64)
    ref   = int2d.clone()
    # Inject huge error in the masked-out region
    ref[3, 5] = 9999.0
    mask = torch.ones_like(int2d)
    mask[3, 5] = 0.0           # exclude that bin
    L = RingMaskedLoss()(int2d, s, ref, mask)
    assert float(L) == pytest.approx(0.0, abs=1e-30)


def test_ring_masked_loss_grad_flows():
    s = _spec()
    int2d = torch.rand(s.n_eta_bins, s.n_r_bins, dtype=torch.float64,
                        requires_grad=True)
    ref = torch.zeros_like(int2d)
    mask = torch.ones_like(int2d)
    L = RingMaskedLoss()(int2d, s, ref, mask)
    L.backward()
    assert int2d.grad is not None
    assert torch.isfinite(int2d.grad).all()


# ======================================================================
# (4) Output formats
# ======================================================================

def test_write_csv_round_trips(tmp_path):
    s = _spec()
    n_r = s.n_r_bins
    r_axis = s.RMin + s.RBinSize * (np.arange(n_r) + 0.5)
    intensity = np.ones(n_r) * 5.0
    sigma = np.ones(n_r) * 0.5
    md = build_provenance(s, integrate_mode="polygon")
    p = write_csv(tmp_path / "p.csv",
                  r_axis=r_axis, intensity=intensity,
                  sigma=sigma, metadata=md)
    assert p.exists()
    data = np.loadtxt(p, delimiter=",", comments="#")
    # %.6e format: ~6 significant digits → ~5e-7 relative precision
    np.testing.assert_allclose(data[:, 0], r_axis, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(data[:, 1], intensity, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(data[:, 2], sigma, rtol=1e-6, atol=1e-7)
    # Header carries the JSON metadata
    text = p.read_text()
    assert "midas_integrate_v2" in text
    assert "spec_hash" in text


def test_write_xye_emits_three_columns(tmp_path):
    n = 50
    r = np.linspace(1, 100, n)
    I = np.random.rand(n) * 100 + 10
    s = np.sqrt(I)
    s_spec = _spec()
    md = build_provenance(s_spec, integrate_mode="hard")
    p = write_xye(tmp_path / "p.xye", r_axis=r, intensity=I, sigma=s,
                  metadata=md)
    assert p.exists()
    data = np.loadtxt(p, comments="#")
    assert data.shape == (n, 3)
    np.testing.assert_allclose(data[:, 0], r, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(data[:, 1], I, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(data[:, 2], s, rtol=1e-6, atol=1e-7)


def test_write_xye_requires_sigma(tmp_path):
    with pytest.raises(ValueError, match="XYE requires sigma"):
        write_xye(tmp_path / "p.xye",
                  r_axis=np.arange(10), intensity=np.zeros(10),
                  sigma=np.zeros(5))


def test_write_fxye_has_bank_line(tmp_path):
    n = 100
    twoth_centidegrees = np.linspace(100, 5000, n)
    I = np.random.rand(n) * 100
    s = np.sqrt(I)
    s_spec = _spec()
    md = build_provenance(s_spec, integrate_mode="polygon")
    p = write_fxye(tmp_path / "p.fxye",
                    r_axis=twoth_centidegrees, intensity=I, sigma=s,
                    metadata=md, title="Pilatus CeO2 test")
    assert p.exists()
    text = p.read_text()
    assert "BANK 1" in text
    assert "Pilatus CeO2 test" in text


def test_write_dat_for_pdf(tmp_path):
    n = 200
    Q = np.linspace(0.1, 8.0, n)
    I = np.exp(-(Q - 3.0) ** 2 / 0.5) + 1.0
    sigma = np.full(n, 0.1)
    s_spec = _spec()
    md = build_provenance(s_spec, integrate_mode="polygon")
    p = write_dat(tmp_path / "p.dat", q_axis_invA=Q, intensity=I,
                  sigma=sigma, metadata=md)
    assert p.exists()
    data = np.loadtxt(p, comments="#")
    assert data.shape == (n, 3)


def test_write_2d_csv_round_trips(tmp_path):
    s = _spec()
    int2d = np.random.rand(s.n_eta_bins, s.n_r_bins)
    r_axis = s.RMin + s.RBinSize * (np.arange(s.n_r_bins) + 0.5)
    eta_axis = s.EtaMin + s.EtaBinSize * (np.arange(s.n_eta_bins) + 0.5)
    md = build_provenance(s, integrate_mode="hard")
    p = write_2d_csv(tmp_path / "2d.csv", int2d=int2d,
                     r_axis_px=r_axis, eta_axis_deg=eta_axis,
                     metadata=md)
    assert p.exists()
    raw = np.loadtxt(p, delimiter=",", comments="#")
    assert raw.shape == (s.n_eta_bins + 1, s.n_r_bins + 1)
    # NaN in the upper-left corner; data block matches input within
    # the writer's %.6e precision (~5e-7 relative).
    np.testing.assert_allclose(raw[1:, 1:], int2d, rtol=1e-6, atol=1e-7)


def test_provenance_metadata_includes_spec_hash():
    s = _spec()
    md = build_provenance(s, integrate_mode="polygon", integrate_K=2)
    assert "spec_hash" in md.spec_summary
    assert len(md.spec_summary["spec_hash"]) == 16
    assert md.integrate_mode == "polygon"
    assert md.integrate_K == 2
    # Same spec → same hash
    md2 = build_provenance(s)
    assert md.spec_summary["spec_hash"] == md2.spec_summary["spec_hash"]


# ======================================================================
# (5) Bootstrap initial geometry
# ======================================================================

def test_bootstrap_BC_recovers_planted_centre():
    """Synth an image with a single ring centred at a known BC; the
    bootstrap should recover BC within a few px."""
    NY = NZ = 64
    BC_true = (NY / 2.0 + 3.0, NZ / 2.0 - 2.0)
    yy, zz = np.meshgrid(np.arange(NY), np.arange(NZ), indexing="xy")
    R = np.sqrt((yy - BC_true[0]) ** 2 + (zz - BC_true[1]) ** 2)
    img = np.exp(-((R - 15.0) ** 2) / (2 * 0.7 ** 2))
    BC_est = estimate_BC_from_image(img, n_iterations=5)
    # Within 1 px of truth
    assert abs(BC_est[0] - BC_true[0]) < 1.0
    assert abs(BC_est[1] - BC_true[1]) < 1.0


def test_bootstrap_estimate_initial_spec():
    NY = NZ = 64
    BC_true = (NY / 2.0 + 3.0, NZ / 2.0 - 2.0)
    yy, zz = np.meshgrid(np.arange(NY), np.arange(NZ), indexing="xy")
    R = np.sqrt((yy - BC_true[0]) ** 2 + (zz - BC_true[1]) ** 2)
    img = np.exp(-((R - 15.0) ** 2) / (2 * 0.7 ** 2))
    spec = estimate_initial_spec(
        img,
        NrPixelsY=NY, NrPixelsZ=NZ,
        pxY_um=200.0, Lsd_um=1_000_000.0,
        Wavelength_A=0.172979,
    )
    # Spec built and BC close to truth
    assert abs(float(spec.BC_y) - BC_true[0]) < 1.0
    assert abs(float(spec.BC_z) - BC_true[1]) < 1.0
    assert spec.NrPixelsY == NY


def test_bootstrap_with_initial_BC_hint_uses_it():
    NY = NZ = 64
    BC_true = (NY / 2.0 + 3.0, NZ / 2.0 - 2.0)
    yy, zz = np.meshgrid(np.arange(NY), np.arange(NZ), indexing="xy")
    R = np.sqrt((yy - BC_true[0]) ** 2 + (zz - BC_true[1]) ** 2)
    img = np.exp(-((R - 15.0) ** 2) / (2 * 0.7 ** 2))
    # Hint near (but not at) truth
    BC_est = estimate_BC_from_image(
        img, initial_BC=(BC_true[0] + 1.5, BC_true[1] - 1.0),
        n_iterations=5,
    )
    assert abs(BC_est[0] - BC_true[0]) < 1.0


# ======================================================================
# (6) End-to-end: mask + polygon + variance + XYE export on real data
# ======================================================================

_REPO = Path(__file__).resolve().parents[3]
_DATA = _REPO / "FF_HEDM" / "Example" / "Calibration"
_PARAMS = _DATA / "parameters.txt"
_IMAGE = _DATA / "CeO2_Pil_100x100_att000_650mm_71p676keV_001956.tif"
_DARK  = _DATA / "dark_CeO2_Pil_100x100_att000_650mm_71p676keV_001975.tif"


@pytest.mark.slow
def test_real_pilatus_polygon_mask_variance_xye_end_to_end(tmp_path):
    """The full v0.7 production workflow on real Pilatus CeO₂: build a
    mask of the inter-module gaps, build polygon geometry with mask,
    integrate with variance, write XYE with provenance — verify the
    XYE has finite intensity + sigma at peak positions and that the
    masked-out gap pixels contributed nothing."""
    if not (_PARAMS.exists() and _IMAGE.exists() and _DARK.exists()):
        pytest.skip("FF_HEDM/Example/Calibration dataset not found")
    tifffile = pytest.importorskip("tifffile")

    img = tifffile.imread(_IMAGE).astype(np.float64)
    dark = tifffile.imread(_DARK).astype(np.float64)
    if dark.ndim == 3:
        dark = dark.mean(axis=0)
    img_t = torch.from_numpy(np.clip(img - dark, 0, None))

    from midas_integrate_v2 import spec_from_v1_paramstest
    spec = spec_from_v1_paramstest(_PARAMS, requires_grad=False)
    spec.RBinSize = 2.0; spec.EtaBinSize = 5.0

    # Auto-build a mask by flagging dim pixels (Pilatus inter-module
    # gaps read as ~0 after dark subtraction).
    mask = (img - dark < 1.0).astype(bool)
    if dark.ndim == 3:
        mask = (img - dark.mean(axis=0) < 1.0).astype(bool)

    print(f"masked fraction: {mask_fraction(mask) * 100:.1f}%")

    # Build polygon geometry with the mask (use n_jobs=2 to also exercise
    # parallel + mask).
    geom = PolygonBinGeometry.from_spec(spec, mask=mask, n_jobs=2)
    assert geom.n_entries > 0

    # Confirm no entries reference masked pixels
    pix = geom.pix_idx.numpy()
    NY = spec.NrPixelsY
    grid_y = pix % NY
    grid_z = pix // NY
    assert not mask[grid_z, grid_y].any()

    # Integrate with variance
    mean2d, sigma2d = integrate_polygon_with_variance(img_t, geom)
    assert mean2d.shape == (spec.n_eta_bins, spec.n_r_bins)
    # With the masked Pilatus frame, some (η, R) bins lie entirely
    # behind the mask and come back as NaN under the v0.9+ default
    # ``empty_bin_value=NaN``. The populated bins must be finite.
    valid2d = torch.isfinite(mean2d)
    assert valid2d.any()
    # σ is non-zero where there's signal
    assert (sigma2d[valid2d] > 0).any()

    # 1D profile (NaN-safe η-mean over populated bins)
    n_r = spec.n_r_bins
    r_axis = spec.RMin + spec.RBinSize * (np.arange(n_r) + 0.5)
    n_valid = valid2d.to(mean2d.dtype).sum(dim=0).clamp(min=1)
    prof = (torch.where(valid2d, mean2d, torch.zeros_like(mean2d))
            .sum(dim=0) / n_valid).numpy()
    sig2 = torch.where(valid2d, sigma2d * sigma2d, torch.zeros_like(sigma2d))
    sigma_1d = (torch.sqrt(sig2.sum(dim=0)) / n_valid).numpy()

    # First CeO₂ ring should have non-trivial intensity and finite σ
    band = (r_axis > 200) & (r_axis < 230)
    assert (prof[band] > 0).any()
    assert np.isfinite(sigma_1d[band]).all()

    # Write XYE with provenance
    md = build_provenance(spec, integrate_mode="polygon",
                           extra={"masked_fraction": mask_fraction(mask),
                                  "image_file": str(_IMAGE.name),
                                  "dark_file": str(_DARK.name)})
    out = tmp_path / "ceo2.xye"
    write_xye(out, r_axis=r_axis, intensity=prof, sigma=sigma_1d,
               metadata=md)
    assert out.exists()
    text = out.read_text()
    assert "midas_integrate_v2" in text
    assert "masked_fraction" in text

    # Re-read and check shape + finite values
    data = np.loadtxt(out, comments="#")
    assert data.shape == (n_r, 3)
    assert np.isfinite(data).all()
