"""Known-answer tests for the omega-axis spot -> frame assignment in
``midas_diffract.simulate_panel_zarrs`` (``frame_integrate`` opt-in).

Pure numpy/torch, CPU, no forward model: exercises ``_per_frame_contribs``,
``_frame_overlap_weights`` and ``_render_frame`` directly.
"""
from __future__ import annotations

import math
from dataclasses import asdict

import numpy as np
import pytest

from midas_diffract.simulate_panel_zarrs import (
    SimConfig,
    _frame_overlap_weights,
    _per_frame_contribs,
    _render_frame,
)


def _phi(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _w_analytic(f: float, c: int, s: float) -> float:
    """Gaussian(f, s) integrated over frame cell c = [c-1/2, c+1/2)."""
    return _phi((c + 0.5 - f) / s) - _phi((c - 0.5 - f) / s)


def _one_spot(f: float, sigma: float, *, integrate: bool, n_frames: int = 400, peak: float = 1.0) -> dict:
    """Run the assignment for one spot; return {frame: amplitude}."""
    pf = _per_frame_contribs(
        np.array([10.0]), np.array([10.0]), np.array([f]),
        peak_intensity=peak, omega_sigma_frames=sigma,
        frame_integrate=integrate, n_frames=n_frames,
    )
    out = {}
    for fr, contribs in pf.items():
        out[fr] = sum(float(a.sum()) for _, _, a in contribs)
    return out


def _legacy_reference(ys, zs, fs, *, peak_intensity, omega_sigma_frames, n_frames):
    """VERBATIM copy of the pre-change inline block of simulate_panel_zarrs
    (sampled Gaussian / nearest frame). Do not 'tidy' this: it is the lock."""
    floor_f = np.floor(fs).astype(np.int64)
    near_f = np.floor(fs + 0.5).astype(np.int64)
    per_frame = {}
    if omega_sigma_frames <= 0:
        for tf in np.unique(near_f):
            idx = np.where(near_f == tf)[0]
            if 0 <= int(tf) < n_frames:
                per_frame.setdefault(int(tf), []).append((
                    ys[idx], zs[idx],
                    np.full(idx.size, peak_intensity, dtype=np.float64),
                ))
    else:
        sigma_f = omega_sigma_frames
        K_f = max(1, int(math.ceil(3.0 * sigma_f)))
        for k in range(-K_f, K_f + 1):
            target = floor_f + k
            d = (fs - target.astype(np.float64))
            w = np.exp(-(d ** 2) / (2.0 * sigma_f ** 2))
            sel = (target >= 0) & (target < n_frames) & (w > 1e-3)
            if not sel.any():
                continue
            t_sel = target[sel]
            for tf in np.unique(t_sel):
                idx = np.where((target == tf) & sel)[0]
                per_frame.setdefault(int(tf), []).append((
                    ys[idx], zs[idx],
                    (peak_intensity * w[idx]).astype(np.float64),
                ))
    return per_frame


def _assert_same(a: dict, b: dict) -> None:
    assert sorted(a) == sorted(b)
    for k in a:
        assert len(a[k]) == len(b[k])
        for (y1, z1, w1), (y2, z2, w2) in zip(a[k], b[k]):
            assert np.array_equal(y1, y2) and np.array_equal(z1, z2)
            assert np.array_equal(w1, w2)


# ---------------------------------------------------------------- T1 analytic
@pytest.mark.parametrize("sigma", [0.1, 0.3, 0.4, 0.7, 1.5])
@pytest.mark.parametrize("f", [100.0, 100.13, 100.5 - 1e-9, 100.77, 101.49])
def test_integrated_weights_match_analytic(f, sigma):
    got = _one_spot(f, sigma, integrate=True)
    for c, amp in got.items():
        assert amp == pytest.approx(_w_analytic(f, c, sigma), abs=1e-12)
    # cells the support cut skipped carry (almost) nothing: < Phi(-5) ~ 2.9e-7
    for c in range(int(f) - 12, int(f) + 14):
        if c not in got:
            assert _w_analytic(f, c, sigma) < 3.0e-7
    assert sum(got.values()) == pytest.approx(1.0, abs=1e-6)


def test_overlap_weights_function_matches_erf():
    fs = np.array([50.0, 50.3, 50.49, 51.2])
    for sigma in (0.2, 0.6, 1.1):
        for dc in (-2, -1, 0, 1, 2):
            tgt = np.floor(fs + 0.5).astype(np.int64) + dc
            got = _frame_overlap_weights(fs, tgt, sigma)
            ref = np.array([_w_analytic(f, int(c), sigma) for f, c in zip(fs, tgt)])
            assert np.allclose(got, ref, rtol=0, atol=1e-12)


# ---------------------------------------------------------------- T2 conservation
@pytest.mark.parametrize("sigma", [0.2, 0.3, 0.5])
def test_integrated_total_is_offset_independent(sigma):
    offs = np.linspace(0.0, 0.9, 10)
    totals = [sum(_one_spot(100.0 + o, sigma, integrate=True).values()) for o in offs]
    assert max(abs(t - 1.0) for t in totals) < 1e-6


def test_sampled_total_depends_on_offset_documents_old_behaviour():
    """The legacy sampled Gaussian is NOT offset-independent below ~1 frame
    (analytic: sigma 0.3 -> 1.008 at offset 0 vs 0.499 at 0.5). Asserted so an
    accidental change to the default path is caught."""
    t0 = sum(_one_spot(100.0, 0.3, integrate=False).values())
    t5 = sum(_one_spot(100.5 - 1e-9, 0.3, integrate=False).values())
    assert t0 == pytest.approx(1.0077, abs=2e-3)
    assert t5 == pytest.approx(0.4987, abs=2e-3)
    assert t0 / t5 > 1.9


def test_image_level_conservation_integrated():
    """Rendered counts summed over frames and pixels are offset-independent."""
    n_pix, peak, sig_px = 40, 1.0e7, 1.0
    totals = []
    for off in (0.0, 0.17, 0.33, 0.5 - 1e-9):
        pf = _per_frame_contribs(
            np.array([20.0]), np.array([20.0]), np.array([100.0 + off]),
            peak_intensity=peak, omega_sigma_frames=0.3,
            frame_integrate=True, n_frames=400,
        )
        tot = 0
        for contribs in pf.values():
            img = np.zeros((n_pix, n_pix), dtype=np.int32)
            for ys, zs, amp in contribs:
                _render_frame(img, ys, zs, amp, sig_px, n_pix)
            tot += int(img.sum())
        totals.append(tot)
    totals = np.array(totals, dtype=float)
    assert (totals.max() - totals.min()) / totals.mean() < 2e-5


# ---------------------------------------------------------------- T3 limit
def test_integrated_sigma_to_zero_equals_delta():
    rng = np.random.default_rng(3)
    fs = rng.uniform(10.0, 90.0, 500)
    ys = np.arange(500, dtype=float); zs = ys + 0.5
    kw = dict(peak_intensity=5000.0, n_frames=100)
    delta = _per_frame_contribs(ys, zs, fs, omega_sigma_frames=0.0, frame_integrate=False, **kw)
    lim = _per_frame_contribs(ys, zs, fs, omega_sigma_frames=1e-9, frame_integrate=True, **kw)
    _assert_same(delta, lim)


def test_flag_is_inert_at_sigma_zero():
    rng = np.random.default_rng(4)
    fs = rng.uniform(-1.0, 61.0, 300)
    ys = np.arange(300, dtype=float); zs = ys
    kw = dict(peak_intensity=5000.0, omega_sigma_frames=0.0, n_frames=60)
    _assert_same(_per_frame_contribs(ys, zs, fs, frame_integrate=False, **kw),
                 _per_frame_contribs(ys, zs, fs, frame_integrate=True, **kw))


# ---------------------------------------------------------------- T4 mirror
def test_mirror_symmetry_negative_step_equivalence():
    """Weights live in frame units: a mirrored scan (omega step of opposite
    sign) mirrors the frame stack exactly."""
    n = 200
    rng = np.random.default_rng(5)
    fs = rng.uniform(20.0, 180.0, 50)
    ys = np.arange(50, dtype=float)

    def per_spot(f_arr):
        pf = _per_frame_contribs(ys, ys, f_arr, peak_intensity=1.0, omega_sigma_frames=0.4,
                                 frame_integrate=True, n_frames=n)
        d = {i: {} for i in range(50)}
        for fr, contribs in pf.items():
            for y, _, a in contribs:
                for yi, ai in zip(y, a):
                    d[int(yi)][fr] = d[int(yi)].get(fr, 0.0) + float(ai)
        return d

    a = per_spot(fs)
    b = per_spot((n - 1) - fs)
    for i in range(50):
        for fr, w in a[i].items():
            assert b[i][(n - 1) - fr] == pytest.approx(w, abs=1e-12)


# ---------------------------------------------------------------- T5 regression lock
@pytest.mark.parametrize("sigma", [0.0, 0.2, 0.5, 1.0, 2.0])
def test_default_path_identical_to_legacy_block(sigma):
    rng = np.random.default_rng(int(1000 * sigma) + 7)
    n_frames = 60
    fs = np.concatenate([rng.uniform(-3.0, n_frames + 3.0, 800),
                         np.array([0.0, 0.5, 59.4999, 59.5, 30.5, 30.4999999])])
    ys = np.arange(fs.size, dtype=float); zs = ys * 0.5
    new = _per_frame_contribs(ys, zs, fs, peak_intensity=5000.0, omega_sigma_frames=sigma,
                              frame_integrate=False, n_frames=n_frames)
    ref = _legacy_reference(ys, zs, fs, peak_intensity=5000.0, omega_sigma_frames=sigma, n_frames=n_frames)
    _assert_same(new, ref)


def test_simconfig_default_is_legacy():
    cfg = SimConfig(space_group=225, lattice_a=4.08, wavelength_A=0.22291, n_pixels=64, px_um=200.0,
                    omega_start_deg=180.0, omega_end_deg=-180.0, omega_step_deg=-0.25, rings_max=5)
    assert cfg.frame_integrate is False and cfg.omega_sigma_frames == 0.0
    assert "frame_integrate" in asdict(cfg)


# ---------------------------------------------------------------- T6 edges
def test_scan_edge_drops_weight_outside_scan():
    """f = 0.2, sigma 0.4: cell -1 lies outside the scan and is dropped, so the
    in-scan total is 1 - Phi((-0.5-0.2)/0.4) = 0.95994, NOT 1 (edge loss is
    documented, not conserved)."""
    got = _one_spot(0.2, 0.4, integrate=True, n_frames=400)
    assert -1 not in got and min(got) == 0
    assert got[0] == pytest.approx(0.7333, abs=1e-4)
    assert got[1] == pytest.approx(0.2261, abs=1e-4)
    assert sum(got.values()) == pytest.approx(1.0 - _phi((-0.5 - 0.2) / 0.4), abs=1e-6)
    hi = _one_spot(399.0 - 0.2, 0.4, integrate=True, n_frames=400)  # mirror at the top edge
    assert 400 not in hi and max(hi) == 399
    assert sum(hi.values()) == pytest.approx(sum(got.values()), abs=5e-8)
