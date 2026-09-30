"""midas_dfxm.population_share: exactness against scipy nnls, then known-truth phantoms run through the WHOLE
pipeline (pixel_populations -> two_population -> centre_maps -> template_share).

Timing note (not a test): on the real S168 700 x 700 crop (h2h/ctx_cache.npz, 489,895 lit px x 61 frames) the
per-pixel bank NNLS takes 9.7 s (10 cores); the scipy loop it replaces takes
~10 min.
"""
import numpy as np
import pytest
from scipy import ndimage
from scipy.optimize import curve_fit, nnls

from midas_dfxm import population_share as ps
from midas_dfxm import populations as pp

X = 15.4732 + 0.01 * np.arange(61)          # the S168 theta grid (61 points, 10 mdeg)
FW = 0.124                                   # single-population FWHM (deg), S168
LOW, HIGH = 15.70, 15.89
EDGE = 15.8132                               # S168 edge angle (between the two populations)
OFFS = (-0.02, -0.01, 0.0, 0.01, 0.02)
WM = (0.7, 1.0, 1.4)


def _unit(c, sg):
    g = np.exp(-0.5 * ((X[None] - c[:, None]) / sg[:, None]) ** 2)
    return g / g.sum(1, keepdims=True)


def _scipy_share(Y, cl, ch, fw):
    s0 = fw / 2.3548
    out = np.full(len(Y), np.nan)
    for i in range(len(Y)):
        cols, side = [], []
        for sd, c in ((0, cl[i]), (1, ch[i])):
            for d in OFFS:
                for m in WM:
                    v = np.exp(-0.5 * ((X - (c + d)) / (s0 * m)) ** 2)
                    cols.append(v / v.sum()); side.append(sd)
        a, _ = nnls(np.array(cols).T, Y[i])
        t = a.sum()
        if t > 0:
            out[i] = a[np.array(side) == 1].sum() / t
    return out


def test_equals_scipy_nnls():
    rng = np.random.default_rng(0)
    n = 5000
    cl = LOW + 0.01 * rng.standard_normal(n); ch = HIGH + 0.01 * rng.standard_normal(n)
    sg = FW / 2.3548 * rng.uniform(0.8, 1.25, n)
    s = rng.uniform(0, 1, n); I = 10 ** rng.uniform(2, 4, n)
    sig = I[:, None] * ((1 - s)[:, None] * _unit(cl, sg) + s[:, None] * _unit(ch, sg))
    Y = sig + rng.standard_normal(sig.shape) * np.sqrt(16 + 0.6 * sig)
    got = ps.nnls_share(Y, X, cl, ch, FW)
    ref = _scipy_share(Y, cl, ch, FW)
    assert np.array_equal(np.isnan(got), np.isnan(ref))
    d = np.abs(got - ref)[np.isfinite(ref)]
    print(f"numba vs scipy nnls, {n} px: max |dshare| {d.max():.3e}, median {np.median(d):.3e}")
    assert d.max() <= 1e-4


def test_pure_populations_and_zero_signal():
    # a pure-LOW curve gives share ~0, a pure-HIGH curve ~1, an all-zero curve NaN
    cl = np.array([LOW, LOW, LOW]); ch = np.array([HIGH, HIGH, HIGH])
    sg = np.full(3, FW / 2.3548)
    Y = np.vstack([100 * _unit(cl[:1], sg[:1]), 100 * _unit(ch[:1], sg[:1]), np.zeros((1, len(X)))])
    sh = ps.nnls_share(Y, X, cl, ch, FW)
    assert sh[0] < 0.05 and sh[1] > 0.95 and np.isnan(sh[2])


def test_template_share_shape_checks():
    with pytest.raises(ValueError):
        ps.template_share(np.zeros((61, 8, 8), np.float32), X, np.zeros((8, 8)), np.ones((9, 9), bool),
                          np.zeros(81), np.zeros(81), FW)


# ------------------------------------------------------------------------------------------- phantoms
H = W = 160
LIT = np.ones((H, W), bool)
BASE = np.full((H, W), 110.0, np.float32)
RR, CC = np.nonzero(LIT)


def _field(rng, sd, s):
    f = ndimage.gaussian_filter(rng.standard_normal((H, W)), s)
    return (f / f.std() * sd)[LIT]


def _phantom(kind, seed):
    """Known-truth scan. Per pixel: brightness Isum (log-linear gradient, x40 across the image, along a
    direction unrelated to the ramp), centres LOW/HIGH + a smooth random field of sd 10 mdeg, width x U(0.8,
    1.25) (smooth), share s(u); noise variance 16 + 0.6 signal."""
    rng = np.random.default_rng(100 + seed)
    ph = np.radians(30.0)
    u = (CC - W / 2) * np.cos(ph) + (RR - H / 2) * np.sin(ph)
    wr = 40.0                                                   # ramp width (80 px on the real 700 px crop)
    s = {"T1": np.clip(u / wr + 0.5, 0, 1), "T2": (u > 0).astype(float), "T3": np.full(u.shape, 0.30)}[kind]
    t = (CC - RR) / (2.0 * W)                                    # -0.5 .. 0.5, gradient along +col, -row
    Isum = 900.0 * 40.0 ** (t + 0.5)                             # 900 .. 36000 (x40)
    cl = LOW + _field(rng, 0.010, 5); ch = HIGH + _field(rng, 0.010, 5)
    fz = _field(rng, 1.0, 5)
    wf = np.exp(np.log(0.8) + (np.log(1.25) - np.log(0.8)) * (0.5 + 0.5 * np.tanh(fz)))
    sg = wf * FW / 2.3548
    sig = Isum[:, None] * ((1 - s)[:, None] * _unit(cl, sg) + s[:, None] * _unit(ch, sg))
    y = sig + rng.standard_normal(sig.shape) * np.sqrt(16.0 + 0.6 * sig)
    F = np.repeat(BASE[None], len(X), 0)
    F[:, LIT] = (BASE[LIT][None] + y.T).astype(np.float32)
    return F, Isum, u, s


def _run(F):
    """The whole pipeline on a phantom scan."""
    P = pp.pixel_populations(F, X, BASE, LIT, np.full(int(LIT.sum()), 4.0))
    tp = pp.two_population(P, EDGE)
    cl, ch = ps.centre_maps(P, tp, LIT)
    return ps.template_share(F, X, BASE, LIT, cl, ch, P["fwhm_single"]), P


def _ramp(u, u0, w, lo, hi):
    return lo + (hi - lo) * np.clip((u - u0) / w + 0.5, 0, 1)


def _fit_ramp(s, phis=np.arange(0, 180, 2.0), bin_px=4.0):
    """Monotone ramp along the best direction: median share in 4 px bins, curve_fit of a clipped linear ramp."""
    m = np.isfinite(s)
    y, r_, c_ = s[m], RR[m].astype(float), CC[m].astype(float)
    sst = ((y - y.mean()) ** 2).sum(); best = None
    for ph in phis:
        u = c_ * np.cos(np.radians(ph)) + r_ * np.sin(np.radians(ph))
        e = np.arange(u.min(), u.max() + bin_px, bin_px); k = np.digitize(u, e)
        cnt = np.bincount(k, minlength=len(e) + 1); idx = np.flatnonzero(cnt >= 30)
        if idx.size < 6:
            continue
        med = np.array([np.median(y[k == i]) for i in idx]); uc = e[np.clip(idx - 1, 0, len(e) - 1)] + bin_px / 2
        try:
            p, _ = curve_fit(_ramp, uc, med, p0=(np.median(uc), 20.0, float(np.clip(med.min(), .01, .99)),
                                                 float(np.clip(med.max(), .01, .99))),
                             sigma=1 / np.sqrt(cnt[idx]),
                             bounds=([uc.min(), 1.0, 0.0, 0.0], [uc.max(), 5 * (uc.max() - uc.min()), 1.0, 1.0]), maxfev=5000)
        except Exception:
            continue
        r2 = 1 - ((y - _ramp(u, *p)) ** 2).sum() / sst
        if best is None or r2 > best["R2"]:
            best = dict(phi=float(ph), w=float(p[1]), lo=float(p[2]), hi=float(p[3]), R2=float(r2))
    if best["lo"] > best["hi"]:
        best.update(phi=(best["phi"] + 180) % 360, lo=best["hi"], hi=best["lo"])
    return best


def _deciles(s, Isum):
    q = np.percentile(Isum, np.linspace(0, 100, 11)); out = []
    for k in range(10):
        m = np.isfinite(s) & (Isum >= q[k]) & (Isum <= q[k + 1])
        out.append(float(np.median(s[m])))
    return out


def test_T3_uniform_share_recovered_in_every_brightness_decile():
    F, Isum, _, _ = _phantom("T3", 3)
    assert Isum.max() / Isum.min() >= 30
    s, _ = _run(F)
    dec = _deciles(s, Isum)
    print("T3 decile medians:", np.round(dec, 3))
    assert all(abs(d - 0.30) <= 0.05 for d in dec)


def test_T1_ramp_direction_width_plateaus():
    F, *_ = _phantom("T1", 1)
    s, _ = _run(F)
    r = _fit_ramp(s)
    print("T1:", r)
    assert min(abs(r["phi"] - 30.0), 180 - abs(r["phi"] - 30.0)) <= 5.0
    assert abs(r["w"] - 40.0) <= 0.25 * 40.0
    assert r["lo"] <= 0.1 and r["hi"] >= 0.9


def test_T2_step_is_narrower_than_the_ramp():
    F, *_ = _phantom("T2", 2)
    s, _ = _run(F)
    r = _fit_ramp(s)
    print("T2:", r)
    assert r["w"] <= 0.15 * 40.0
