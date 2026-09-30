"""The torch backend must reproduce the numpy support estimator exactly (float64, to rounding)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
from midas_dfxm.rocking import RockingScan
from midas_dfxm.support import support_curve, reduce_support
from midas_dfxm.support_torch import support_curve_torch


def _same(a, b, name):
    a = np.asarray(a); b = np.asarray(b)
    assert a.shape == b.shape, name
    if a.dtype == bool or b.dtype == bool:
        assert np.array_equal(a, b), name
        return
    a = a.astype(float); b = b.astype(float)
    assert np.array_equal(np.isnan(a), np.isnan(b)), f"{name}: NaN pattern differs"
    m = np.isfinite(a)
    assert np.allclose(a[m], b[m], rtol=1e-9, atol=1e-9), f"{name}: max |diff| {np.max(np.abs(a[m]-b[m]))}"


def _cmp(D, shape, coords, **kw):
    o1 = support_curve(D, shape, coords, **kw)
    t = lambda z: None if z is None else torch.as_tensor(z, dtype=torch.float64)
    kw_t = dict(kw)
    if "halves" in kw_t and kw_t["halves"] is not None:
        kw_t["halves"] = [t(h) for h in kw_t["halves"]]
    if "Dw" in kw_t:
        kw_t["Dw"] = t(kw_t["Dw"])
    o2 = support_curve_torch(t(D), shape, coords, **kw_t)
    for k in o1:
        _same(o1[k], o2[k], k)


def test_1d_cases():
    rng = np.random.default_rng(0)
    x = np.linspace(-0.02, 0.02, 41)
    N = 500
    c = rng.uniform(-0.005, 0.005, N)
    for fw in (0.002, 0.015):
        D = rng.poisson(120 + 800 * np.exp(-0.5 * ((x[:, None] - c) / (fw / 2.3548)) ** 2)).astype(float)
        Dw = D + rng.normal(0, 0.1, D.shape)
        H = [D + rng.normal(0, 3, D.shape), D + rng.normal(0, 3, D.shape)]
        _cmp(D, (41,), [x])
        _cmp(D, (41,), [x], Dw=Dw, halves=H)
    cut = rng.poisson(100 + 800 * np.exp(-0.5 * ((x[:, None] - 0.019) / 0.004) ** 2) * np.ones((1, 200))).astype(float)
    _cmp(cut, (41,), [x])
    _cmp(rng.poisson(400.0, (41, 300)).astype(float), (41,), [x])


def test_2d_missing_and_3d():
    rng = np.random.default_rng(1)
    a = np.linspace(-0.02, 0.02, 21); b = np.linspace(-0.05, 0.05, 11)
    A, B = np.meshgrid(a, b, indexing="ij")
    ca = rng.uniform(-0.003, 0.003, 150); cb = rng.uniform(-0.01, 0.01, 150)
    D = rng.poisson(100 + 2000 * np.exp(-0.5 * (((A[..., None] - ca) / 0.004) ** 2 + ((B[..., None] - cb) / 0.012) ** 2))).astype(float)
    D[3, 7] = np.nan; D[15, 2] = np.nan
    _cmp(D, (21, 11), [a, b])
    ax = [np.linspace(-0.01, 0.01, 11), np.linspace(-0.03, 0.03, 7), np.linspace(-0.02, 0.02, 9)]
    G = np.meshgrid(*ax, indexing="ij")
    q = sum((G[i][..., None] / w) ** 2 for i, w in enumerate((0.003, 0.01, 0.006)))
    D3 = rng.poisson(100 + 3000 * np.exp(-0.5 * q) * np.ones((1, 1, 1, 60))).astype(float)
    _cmp(D3, (11, 7, 9), ax)


@pytest.mark.parametrize("halves", [True, False])
def test_reduce_support_backends_agree(halves):
    rng = np.random.default_rng(2)
    th = 15.0 + np.arange(31) * 0.001
    yy, xx = np.mgrid[:48, :40]
    c = th[15] + 0.003 * np.sin(yy / 9.0)
    lam = 110 + (400 * np.exp(-0.5 * ((th[:, None, None] - c[None]) / 0.004) ** 2)) * (xx[None] > 8)
    A = rng.poisson(lam).astype(np.float32); B = rng.poisson(lam).astype(np.float32)
    scan = RockingScan.from_arrays(0.5 * (A + B), {"th": th}, halves=(A, B) if halves else None, n_repeats=2 if halves else 1)
    m1 = reduce_support(scan, chunk_rows=16)
    m2 = reduce_support(scan, chunk_rows=16, backend="torch", device="cpu")
    for k in ("value", "centre_deg", "cov_deg2", "fwhm_mdeg", "intensity", "snr", "sigma_frame", "baseline",
              "baseline_ok", "lit", "truncated", "cut", "end_fraction", "n_support", "main_share", "shape_resid",
              "sigma", "fwhm_halfmax_mdeg", "snr_neighbourhood"):
        _same(getattr(m1, k), getattr(m2, k), k)
    assert m1.sigma_global == pytest.approx(m2.sigma_global, rel=1e-9)
