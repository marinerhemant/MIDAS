"""6-ID-C polymer-optics beamline model: SU-8 index, XPL condenser, vibration PSF.

Validation targets are the measured numbers in Qiao et al., Rev. Sci. Instrum. 91, 113703
(2020) -- the instrument paper for the APS 6-ID-C DFXM microscope.
"""
import pytest
import torch

from midas_dfxm.polymer import su8_index, beryllium_index
from midas_dfxm.beamline import delta_beryllium
from midas_dfxm.beamline_6idc import xpl_condenser, xpl_square_wavefront
from midas_dfxm.vibration import (
    effective_blur_rms_um, resolution_vs_exposure, frames_to_recover,
)

FREQS = torch.logspace(0, 2, 24, dtype=torch.float64)   # 1..100 Hz


@pytest.mark.unit
def test_polymer_index_matches_anchored_beryllium():
    # first-principles Be delta (midas_hkls) must match the package's Poulsen-anchored value
    for E in (17.0, 20.0):
        ratio = beryllium_index(E)["delta"] / float(delta_beryllium(E))
        assert 0.99 < ratio < 1.01


@pytest.mark.unit
def test_su8_index_physical():
    s = su8_index(20.0)
    assert 6.0e-7 < s["delta"] < 7.5e-7          # SU-8 refractive decrement at 20 keV
    assert s["beta"] > 0 and s["mu_lin_per_cm"] > 0
    # SU-8 is far more transparent than it is refractive
    assert s["beta"] < 1e-3 * s["delta"]


@pytest.mark.unit
def test_xpl_condenser_na_matches_paper():
    h = xpl_condenser(20.0, usable_aperture_mm=1.42)["NA"]
    v = xpl_condenser(20.0, usable_aperture_mm=1.55)["NA"]
    assert abs(h - 0.00032) < 0.2e-4             # paper 0.00032 (H)
    assert abs(v - 0.00035) < 0.2e-4             # paper 0.00035 (V)


@pytest.mark.unit
def test_xpl_focal_spot_and_wavefront():
    c = xpl_condenser(20.0, source_size_um=1740.0)
    assert 55.0 < c["focal_spot_um"] < 72.0      # paper 63 x 73 um (demagnified source)
    assert c["transmission"] > 0.9               # SU-8 optics are transmissive
    w = xpl_square_wavefront(size=96)
    pv = float(w.max() - w.min())
    assert 200.0 < pv < 320.0                     # ~ +/-131 nm measured (Fig 2d)


@pytest.mark.unit
def test_vibration_blur_saturates_and_freezes():
    sigma = 0.428
    blur_long = float(effective_blur_rms_um(sigma, FREQS, 10.0))
    blur_short = float(effective_blur_rms_um(sigma, FREQS, 0.001))
    assert blur_long == pytest.approx(sigma, rel=0.05)   # long exposure -> full jitter
    assert blur_short < 0.25 * sigma                     # short exposure -> frozen


@pytest.mark.unit
def test_vibration_resolution_monotone_and_saturates():
    sigma, base = 0.428, 0.216
    taus = [1e-3, 1e-2, 1e-1, 1.0, 10.0]
    res = resolution_vs_exposure(sigma, base, FREQS, taus)
    r = res.tolist()
    assert all(r[i] <= r[i + 1] + 1e-9 for i in range(len(r) - 1))   # monotone in exposure
    assert res[-1] == pytest.approx(res[-2], rel=0.02)               # saturates past 1 s
    assert res[-1] > 1.7 * res[0]                                    # ~2x long/short degradation


@pytest.mark.unit
def test_vibration_recovery_design():
    # a short exposure must recover the vibration-free limit; a long one must not
    good = frames_to_recover(0.428, 0.216, FREQS, target_res_um=0.240,
                             short_exposure_s=0.001, total_dose_s=10.0)
    bad = frames_to_recover(0.428, 0.216, FREQS, target_res_um=0.240,
                            short_exposure_s=0.010, total_dose_s=10.0)
    assert good["meets_target"] and good["n_frames"] == 10000
    assert not bad["meets_target"]


@pytest.mark.unit
def test_vibration_blur_differentiable_in_sigma():
    sigma = torch.tensor(0.4, dtype=torch.float64, requires_grad=True)
    blur = effective_blur_rms_um(sigma, FREQS, 5.0)
    blur.backward()
    assert sigma.grad is not None and float(sigma.grad) > 0


# --- Talbot wavefront sensor (component 4): validated forward + differential sensing ---
from midas_dfxm.talbot import (            # noqa: E402
    wavelength_um, line_grating, interferogram, talbot_distance_um,
    differential_phase, integrate_gradient,
)

_N, _DX, _P = 256, 0.60, 4.8
_LAM = wavelength_um(20.0)
_GX = line_grating(_N, _DX, _P, angle_deg=0.0)
_GY = line_grating(_N, _DX, _P, angle_deg=90.0)
_xs = (torch.arange(_N, dtype=torch.float64) - _N // 2) * _DX
_X, _Y = torch.meshgrid(_xs, _xs, indexing="xy")
_APER = (_X ** 2 + _Y ** 2) <= (0.42 * _N * _DX / 2) ** 2
_Z = float(talbot_distance_um(_P, _LAM) * 0.20)


@pytest.mark.unit
def test_talbot_self_imaging_contrast():
    # a flat wavefront develops grating fringes at a fractional Talbot distance
    flat = torch.zeros(_N, _N, dtype=torch.float64)
    I = interferogram(flat, _GX, _Z, _DX, _LAM)
    assert float(I.std() / I.mean()) > 0.3          # visible self-image fringes


@pytest.mark.unit
def test_talbot_tilt_calibration_and_crosstalk():
    flat = torch.zeros(_N, _N, dtype=torch.float64)
    refx = interferogram(flat, _GX, _Z, _DX, _LAM)
    refy = interferogram(flat, _GY, _Z, _DX, _LAM)
    sl = 2.0e-6
    px = differential_phase(interferogram(sl * _X * 1e3, _GX, _Z, _DX, _LAM), refx, _P, _DX, "x")
    py = differential_phase(interferogram(sl * _Y * 1e3, _GY, _Z, _DX, _LAM), refy, _P, _DX, "y")
    cal_x = sl / float(px[_APER].median())
    cal_y = sl / float(py[_APER].median())
    assert cal_x == pytest.approx(cal_y, rel=0.1)   # isotropic sensor
    # x-tilt must not leak into the y-grating channel
    leak = differential_phase(interferogram(sl * _X * 1e3, _GY, _Z, _DX, _LAM), refy, _P, _DX, "y")
    assert abs(float(leak[_APER].median()) * cal_y / sl) < 0.05


@pytest.mark.unit
def test_talbot_gradient_integration():
    # Frankot-Chellappa integrates a known analytic gradient field back to the wavefront
    S = 0.5 * _N * _DX
    u, v = _X / S, _Y / S
    W = 120.0 * (u ** 2 - v ** 2) + 60.0 * (2 * u * v)
    dWdx = (240.0 * u + 120.0 * v) / S * 1e-3
    dWdy = (-240.0 * v + 120.0 * u) / S * 1e-3
    W_int = integrate_gradient(dWdx, dWdy, _DX) * 1e3
    a, b = W[_APER], W_int[_APER]
    b = b - b.mean() + a.mean()
    assert float(torch.corrcoef(torch.stack([a, b]))[0, 1]) > 0.85


@pytest.mark.unit
def test_talbot_forward_differentiable():
    W = torch.zeros(_N, _N, dtype=torch.float64, requires_grad=True)
    I = interferogram(W, _GX, _Z, _DX, _LAM)
    I.sum().backward()
    assert W.grad is not None


@pytest.mark.unit
def test_talbot_wavefront_roundtrip():
    # full forward+inverse of a low-order wavefront in the unwrapped (nm) sensing regime
    flat = torch.zeros(_N, _N, dtype=torch.float64)
    refx = interferogram(flat, _GX, _Z, _DX, _LAM)
    refy = interferogram(flat, _GY, _Z, _DX, _LAM)
    sl = 2.0e-8
    cal_x = sl / float(differential_phase(interferogram(sl * _X * 1e3, _GX, _Z, _DX, _LAM), refx, _P, _DX, "x")[_APER].median())
    cal_y = sl / float(differential_phase(interferogram(sl * _Y * 1e3, _GY, _Z, _DX, _LAM), refy, _P, _DX, "y")[_APER].median())
    S = 0.5 * _N * _DX
    u, v = _X / S, _Y / S
    W = 1.0 * (u ** 2 - v ** 2) + 0.5 * (2 * u * v)          # ~1 nm astig
    gx = differential_phase(interferogram(W, _GX, _Z, _DX, _LAM), refx, _P, _DX, "x")
    gy = differential_phase(interferogram(W, _GY, _Z, _DX, _LAM), refy, _P, _DX, "y")
    Wr = integrate_gradient(gx * cal_x, gy * cal_y, _DX) * 1e3
    a, b = W[_APER], Wr[_APER]
    b = b - b.mean() + a.mean()
    assert float(torch.corrcoef(torch.stack([a, b]))[0, 1]) > 0.85
