"""Phase 1 tests: resolution function, objective optics, DFXM forward."""
import math

import pytest
import torch

from midas_dfxm import (
    GoniometerSetting,
    ObjectiveOptics,
    aligned_resolution,
    bragg_two_theta_deg,
    dfxm_image,
    dfxm_stack,
    fcc_reference_crystal,
    make_uniform_field,
    mosaicity_curve,
    reference_q_nom,
    rocking_scan,
    scan_angles,
    structure_factor_intensity,
    voxel_intensity,
    with_orientation_gradient,
    with_uniform_strain,
)

DT = torch.float64
WAVELENGTH = 0.172979  # Angstrom (demk)


def _fwhm(x: torch.Tensor, y: torch.Tensor) -> float:
    """FWHM of a single-peaked curve by linear interpolation at half-max."""
    y = y / y.max()
    half = 0.5
    idx = torch.where(y >= half)[0]
    lo, hi = idx[0].item(), idx[-1].item()
    # interpolate edges
    def interp(i_in, i_out):
        x0, x1 = x[i_out].item(), x[i_in].item()
        y0, y1 = y[i_out].item(), y[i_in].item()
        return x0 + (half - y0) * (x1 - x0) / (y1 - y0)
    left = interp(lo, max(lo - 1, 0)) if lo > 0 else x[lo].item()
    right = interp(hi, min(hi + 1, len(x) - 1)) if hi < len(x) - 1 else x[hi].item()
    return right - left


# --------------------------------------------------------------------------
# resolution function
# --------------------------------------------------------------------------
@pytest.mark.unit
def test_resolution_peaks_at_qnom():
    q_nom = torch.tensor([2.0, 0.0, 1.0], dtype=DT)
    res = aligned_resolution(q_nom, sigma_par=5e-3, sigma_perp=5e-3)
    assert res.weight(q_nom).item() == pytest.approx(1.0, abs=1e-12)
    off = res.weight(q_nom + torch.tensor([0.05, 0.0, 0.0], dtype=DT))
    assert off.item() < 1.0


@pytest.mark.unit
def test_rocking_fwhm_matches_analytic():
    # Perfect crystal: mosaicity curve FWHM == analytic 2.355*sigma_perp/|q_nom|.
    field = make_uniform_field(shape=(1, 1, 1), lattice_params=(3.6356,) * 3 + (90.0,) * 3, dtype=DT)
    hkl = (1, 1, 1)
    center = GoniometerSetting()
    q_nom = reference_q_nom(field, hkl, center)
    res = aligned_resolution(q_nom, sigma_par=5e-3, sigma_perp=8e-3)
    settings = rocking_scan(center, axis="chi", span=(-0.5, 0.5), n=201)
    curve = mosaicity_curve(field, hkl, settings, res)
    angles = scan_angles(settings, "chi")
    fwhm = _fwhm(angles, curve)
    assert fwhm == pytest.approx(res.rocking_fwhm_deg(), rel=0.03)


@pytest.mark.unit
def test_strain_scan_shifts_peak():
    # A uniform tensile strain shifts |Q| -> shifts the aligned setting's weight.
    field0 = make_uniform_field(shape=(1, 1, 1), dtype=DT)
    hkl = (1, 1, 1)
    center = GoniometerSetting()
    q_nom = reference_q_nom(field0, hkl, center)
    res = aligned_resolution(q_nom, sigma_par=3e-3, sigma_perp=3e-3)
    w0 = voxel_intensity(field0, hkl, center, res).sum()
    field_str = with_uniform_strain(field0, 2e-3 * torch.eye(3, dtype=DT))
    w1 = voxel_intensity(field_str, hkl, center, res).sum()
    # Strain moves Q off the aligned longitudinal centre -> lower weight.
    assert w1.item() < w0.item()


# --------------------------------------------------------------------------
# structure factor
# --------------------------------------------------------------------------
@pytest.mark.unit
def test_structure_factor_intensity_positive():
    crystal = fcc_reference_crystal()
    sf2 = structure_factor_intensity(crystal, (1, 1, 1), wavelength_A=WAVELENGTH, dtype=DT)
    assert sf2.item() > 0.0


@pytest.mark.unit
def test_fcc_forbidden_reflection_is_dark():
    # FCC: (1,0,0) is forbidden (mixed parity) -> |F|^2 ~ 0.
    crystal = fcc_reference_crystal()
    allowed = structure_factor_intensity(crystal, (1, 1, 1), wavelength_A=WAVELENGTH, dtype=DT)
    forbidden = structure_factor_intensity(crystal, (1, 0, 0), wavelength_A=WAVELENGTH, dtype=DT)
    assert forbidden.item() < 1e-6 * allowed.item()


@pytest.mark.unit
def test_bragg_two_theta_consistency():
    a = 3.6356
    d111 = a / math.sqrt(3)
    q = 2 * math.pi / d111
    tt = bragg_two_theta_deg(q, WAVELENGTH)
    # Back out d from 2-theta and compare.
    theta = math.radians(tt / 2)
    d_back = WAVELENGTH / (2 * math.sin(theta))
    assert d_back == pytest.approx(d111, rel=1e-9)


# --------------------------------------------------------------------------
# optics + forward image
# --------------------------------------------------------------------------
@pytest.mark.unit
def test_dfxm_image_shape_and_signal():
    field = make_uniform_field(shape=(16, 16, 1), spacing_um=1.0, dtype=DT)
    hkl = (1, 1, 1)
    center = GoniometerSetting()
    q_nom = reference_q_nom(field, hkl, center)
    res = aligned_resolution(q_nom, sigma_par=5e-3, sigma_perp=5e-2)
    tt = bragg_two_theta_deg(float(torch.linalg.vector_norm(q_nom)), WAVELENGTH)
    optics = ObjectiveOptics(two_theta_deg=tt, magnification=8.0, pixel_um=1.0, detector_shape=(128, 128))
    img = dfxm_image(field, hkl, center, res, optics)
    assert img.shape == (128, 128)
    assert torch.isfinite(img).all()
    assert img.sum().item() > 0.0


@pytest.mark.unit
def test_orientation_gradient_produces_contrast_variation():
    # A curved crystal lights up different regions at different rocking angles.
    field = make_uniform_field(shape=(32, 4, 1), spacing_um=1.0, dtype=DT)
    field = with_orientation_gradient(field, axis=(0, 0, 1), deg_per_um=0.02, along=0)
    hkl = (1, 1, 1)
    center = GoniometerSetting()
    q_nom = reference_q_nom(field, hkl, center)
    res = aligned_resolution(q_nom, sigma_par=1e-2, sigma_perp=5e-3)
    settings = rocking_scan(center, axis="chi", span=(-0.4, 0.4), n=9)
    inten = torch.stack([
        voxel_intensity(field, hkl, s, res).reshape(32, 4).mean(dim=1)
        for s in settings
    ])  # (S, 32)
    # The peak-response x-position should move across the crystal with rocking angle.
    argmax_x = inten.argmax(dim=1)
    assert argmax_x.max() > argmax_x.min()


# --------------------------------------------------------------------------
# differentiability + device
# --------------------------------------------------------------------------
@pytest.mark.autograd
def test_forward_differentiable_in_field():
    field = make_uniform_field(shape=(6, 6, 1), dtype=DT)
    field.F.requires_grad_(True)
    hkl = (1, 1, 1)
    center = GoniometerSetting()
    q_nom = reference_q_nom(field, hkl, center)
    res = aligned_resolution(q_nom, sigma_par=1e-2, sigma_perp=1e-2)
    loss = mosaicity_curve(field, hkl, rocking_scan(center, span=(-0.2, 0.2), n=5), res).sum()
    loss.backward()
    assert field.F.grad is not None and torch.isfinite(field.F.grad).all()
    assert field.F.grad.abs().sum() > 0


@pytest.mark.autograd
def test_forward_differentiable_in_sigma():
    field = make_uniform_field(shape=(4, 4, 1), dtype=DT)
    hkl = (1, 1, 1)
    center = GoniometerSetting()
    q_nom = reference_q_nom(field, hkl, center)
    sigma = torch.tensor(1e-2, dtype=DT, requires_grad=True)
    from midas_dfxm import ResolutionFunction
    res = ResolutionFunction(q_nom=q_nom, sigma_par=sigma, sigma_perp=sigma)
    val = voxel_intensity(field, hkl, GoniometerSetting(chi=0.1), res).sum()
    val.backward()
    assert sigma.grad is not None and torch.isfinite(sigma.grad)


@pytest.mark.device
@pytest.mark.parametrize("device", ["cpu", "mps", "cuda"])
def test_forward_device_portable(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("cuda unavailable")
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("mps unavailable")
    dt = torch.float32 if device == "mps" else DT
    field = make_uniform_field(shape=(8, 8, 1), device=device, dtype=dt)
    hkl = (1, 1, 1)
    center = GoniometerSetting()
    q_nom = reference_q_nom(field, hkl, center)
    res = aligned_resolution(q_nom, sigma_par=1e-2, sigma_perp=1e-2)
    tt = bragg_two_theta_deg(float(torch.linalg.vector_norm(q_nom)), WAVELENGTH)
    optics = ObjectiveOptics(two_theta_deg=tt, detector_shape=(32, 32), pixel_um=2.0)
    img = dfxm_image(field, hkl, center, res, optics)
    assert img.device.type == device
    assert torch.isfinite(img).all()
