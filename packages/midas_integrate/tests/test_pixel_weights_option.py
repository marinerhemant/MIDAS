"""build_csr(pixel_weights=...): 'area' (default, MIDAS: a pixel counts by its overlap area in the projected plane) versus 'fraction' (every pixel counts once, the
pyFAI / azint convention: a pixel's weights over all bins sum to 1).  Checked against an explicit loop over the map entries, not against the implementation."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
from midas_integrate.detector_mapper import build_map
from midas_integrate.kernels import build_csr, integrate, profile_1d
from midas_integrate.params import IntegrationParams

NY = NZ = 48


def _map(tilt=(0.0, 12.0, -5.0)):
    p = IntegrationParams(NrPixelsY=NY, NrPixelsZ=NZ, pxY=172.0, pxZ=172.0, Lsd=200_000.0, BC_y=23.4, BC_z=24.2, RhoD=1.0e6, tx=tilt[0], ty=tilt[1], tz=tilt[2],
                          RMin=3.0, RMax=30.0, RBinSize=0.5, EtaMin=-180.0, EtaMax=180.0, EtaBinSize=10.0, SubPixelLevel=1)
    return p, build_map(p, verbose=False, auto_load=False)


def _geom(p, m, **kw):
    return build_csr(m.as_pixel_map(), n_r=p.n_r_bins, n_eta=p.n_eta_bins, n_pixels_y=NY, n_pixels_z=NZ, device="cpu", dtype=torch.float64,
                     bc_y=p.BC_y, bc_z=p.BC_z, build_modes=("floor",), **kw)


def test_default_is_area_and_unchanged():
    p, m = _map()
    g0 = _geom(p, m); g1 = _geom(p, m, pixel_weights="area")
    assert g0.pixel_weights == "area" and g1.pixel_weights == "area"
    assert torch.equal(g0.csr_floor.to_dense(), g1.csr_floor.to_dense()) and torch.equal(g0.area_per_bin, g1.area_per_bin)


def test_invalid_option_raises():
    p, m = _map()
    with pytest.raises(ValueError):
        _geom(p, m, pixel_weights="nonsense")


def test_fraction_columns_sum_to_one_and_area_columns_do_not_under_tilt():
    p, m = _map()
    inside = np.zeros((NZ, NY), bool)
    Y, Z = np.meshgrid(np.arange(NY), np.arange(NZ)); r = np.hypot(Y - p.BC_y, Z - p.BC_z); inside = (r > 7.0) & (r < 26.0)   # whole pixel inside the radial range, away from the beam centre
    col_f = _geom(p, m, pixel_weights="fraction").csr_floor.to_dense().numpy().sum(axis=0).reshape(NZ, NY)
    col_a = _geom(p, m, pixel_weights="area").csr_floor.to_dense().numpy().sum(axis=0).reshape(NZ, NY)
    assert inside.sum() > 500
    assert np.abs(col_f[inside] - 1.0).max() < 1e-6          # float32 storage of the overlap fields
    assert np.abs(col_a[inside] - 1.0).max() > 1e-3          # a tilted pixel's mapped area is not 1, so the area convention differs


@pytest.mark.parametrize("pw", ["area", "fraction"])
def test_uniform_image_integrates_to_one(pw):
    p, m = _map()
    g = _geom(p, m, pixel_weights=pw)
    out = integrate(torch.ones(NZ, NY, dtype=torch.float64), g, mode="floor", normalize=True).numpy()
    ok = g.area_per_bin.numpy().reshape(out.shape) > 1e-6
    assert ok.sum() > 100 and np.abs(out[ok] - 1.0).max() < 1e-6


def test_fraction_matches_explicit_loop_over_the_entries():
    p, m = _map()
    pm = m.as_pixel_map(); rec = pm.pxList; counts = pm.counts.astype(np.int64); offs = pm.offsets.astype(np.int64)
    rng = np.random.default_rng(3); img = rng.uniform(1.0, 100.0, size=(NZ, NY))
    iy = np.rint(rec["y"]).astype(np.int64); iz = np.rint(rec["z"]).astype(np.int64)
    a_pix = {}
    for k in range(len(rec)):
        key = (int(iz[k]), int(iy[k])); a_pix[key] = a_pix.get(key, 0.0) + float(rec["areaWeight"][k])
    n_bins = len(counts); exp = np.zeros(n_bins); den = np.zeros(n_bins)
    for b in range(n_bins):
        for k in range(int(offs[b]), int(offs[b]) + int(counts[b])):
            f = float(rec["frac"][k]) / a_pix[(int(iz[k]), int(iy[k]))]; w = float(rec["areaWeight"][k]) / a_pix[(int(iz[k]), int(iy[k]))]
            exp[b] += f * img[iz[k], iy[k]]; den[b] += w
    expected = np.where(den > 1e-9, exp / np.maximum(den, 1e-9), 0.0).reshape(p.n_r_bins, p.n_eta_bins)
    got = integrate(torch.from_numpy(img), _geom(p, m, pixel_weights="fraction"), mode="floor", normalize=True).numpy()
    assert np.abs(got - expected).max() < 1e-9


def test_one_dimensional_profiles_differ_for_an_azimuthally_varying_image_and_agree_for_a_uniform_one():
    p, m = _map()
    Y, Z = np.meshgrid(np.arange(NY), np.arange(NZ)); img = 100.0 + 20.0 * np.cos(np.arctan2(Z - p.BC_z, Y - p.BC_y) * 2.0)
    prof = {}
    for pw in ("area", "fraction"):
        g = _geom(p, m, pixel_weights=pw)
        prof[pw] = profile_1d(integrate(torch.from_numpy(img), g, mode="floor", normalize=True), g).numpy()
    sel = slice(8, 40)
    assert np.abs(prof["area"][sel] - prof["fraction"][sel]).max() > 1e-4
    g1, g2 = _geom(p, m, pixel_weights="area"), _geom(p, m, pixel_weights="fraction")
    one = torch.ones(NZ, NY, dtype=torch.float64)
    a = profile_1d(integrate(one, g1, mode="floor"), g1).numpy(); f = profile_1d(integrate(one, g2, mode="floor"), g2).numpy()
    assert np.abs(a[sel] - 1.0).max() < 1e-6 and np.abs(f[sel] - 1.0).max() < 1e-6
