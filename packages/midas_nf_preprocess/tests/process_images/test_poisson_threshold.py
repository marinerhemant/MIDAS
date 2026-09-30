"""SpotDetect poisson: equal false-alarm rate across a detector whose background rate varies.

Synthetic stand-in for bt_20id_sep26 PUR #255: sparse Poisson counts over a zero median, with one
band ~2.5x busier than the rest (measured 1.47 vs 1.18 counts/px mean there; exaggerated here so the
test has power), plus real spots of modest brightness in both regions.
"""
import numpy as np
import pytest
import torch
from scipy import ndimage as ndi

from midas_nf_preprocess.process_images.poisson_threshold import (
    detect_labels_poisson, expected_false_alarms, local_rate_from_stack, threshold_map, window_sum)

Z, Y = 320, 320
BAND = slice(120, 200)


def _rate():
    r = np.full((Z, Y), 0.30)
    r[BAND] = 0.75
    return r


def _bg(rng, n):
    return rng.poisson(_rate(), size=(n, Z, Y)).astype(np.float32)


def _spots(rng, n_spots=60, amp=4.0, sigma=1.3):
    img = np.zeros((Z, Y), np.float64); cen = []
    for _ in range(n_spots):
        z, y = rng.integers(8, Z - 8), rng.integers(8, Y - 8)
        zz, yy = np.mgrid[z - 6:z + 7, y - 6:y + 7]
        img[z - 6:z + 7, y - 6:y + 7] += amp * np.exp(-((zz - z) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2))
        cen.append((z, y))
    return img, cen


def _components(mask, min_px=4):
    lab, n = ndi.label(mask, structure=np.ones((3, 3)))
    if n == 0:
        return lab, 0
    s = np.bincount(lab.ravel()); keep = np.flatnonzero(s >= min_px); keep = keep[keep > 0]
    return np.where(np.isin(lab, keep), lab, 0), len(keep)


def test_threshold_rises_monotonically_with_rate():
    rate = torch.linspace(0.05, 5.0, 400).reshape(20, 20)
    t = threshold_map(rate, window=3, fp_per_frame=5).flatten().numpy()
    assert np.all(np.diff(t) >= 0) and t[-1] > t[0]


def test_rate_map_recovers_both_bands_with_spots_present():
    rng = np.random.default_rng(0)
    stack = _bg(rng, 60)
    for j in range(0, 60, 6):                               # spots in some frames
        stack[j] += rng.poisson(_spots(rng, amp=20)[0]).astype(np.float32)
    rate = local_rate_from_stack(torch.from_numpy(stack), torch.zeros(Z, Y), n_frames=60,
                                 smooth_px=16).numpy()
    assert abs(np.median(rate[BAND]) / 0.75 - 1) < 0.12
    assert abs(np.median(rate[:100]) / 0.30 - 1) < 0.12


def test_false_alarms_equal_per_area_in_busy_and_quiet_rows():
    """Poisson map: background blobs per unit area comparable in both bands and near budget.
    A single global threshold chosen for the quiet rows floods the busy band."""
    rng = np.random.default_rng(1)
    rate = torch.from_numpy(_rate().astype(np.float32))
    thr = threshold_map(rate, window=3, fp_per_frame=5)
    frames = _bg(rng, 30)
    n_band = n_quiet = 0
    for f in frames:
        lab, n, _ = detect_labels_poisson(torch.from_numpy(f), thr, window=3, min_px=4)
        lab = lab.numpy()
        ids_band = np.unique(lab[BAND]); ids_band = ids_band[ids_band > 0]
        n_band += len(ids_band); n_quiet += n - len(ids_band)
    area_band = (BAND.stop - BAND.start) / Z
    per_frame = (n_band + n_quiet) / len(frames)
    assert per_frame <= 5.0, per_frame                           # within the pixel-level budget
    # busy band must not dominate: per-area density within 3x of the quiet rows
    dens_band = n_band / area_band; dens_quiet = max(n_quiet, 1) / (1 - area_band)
    assert dens_band <= 3 * dens_quiet + 3 / area_band, (n_band, n_quiet)

    # contrast: one global threshold = the quiet-row value
    t_global = float(thr[0, 0])
    nb = nq = 0
    for f in frames:
        S = window_sum(torch.from_numpy(f), 3).numpy()
        lab, _ = _components(S > t_global)
        ids = np.unique(lab[BAND]); nb += int((ids > 0).sum())
        nq += int((np.unique(lab) > 0).sum()) - int((ids > 0).sum())
    assert nb > 20 * max(n_band, 1), (nb, n_band)                # global threshold floods the busy band


def test_recall_of_real_spots_in_both_bands():
    rng = np.random.default_rng(2)
    rate = torch.from_numpy(_rate().astype(np.float32))
    thr = threshold_map(rate, window=3, fp_per_frame=5)
    spots, cen = _spots(rng, n_spots=60, amp=6.0)
    frame = (rng.poisson(_rate()) + rng.poisson(spots)).astype(np.float32)
    lab, n, _ = detect_labels_poisson(torch.from_numpy(frame), thr, window=3, min_px=4)
    lab = lab.numpy()
    hit = [lab[max(z - 1, 0):z + 2, max(y - 1, 0):y + 2].max() > 0 for z, y in cen]
    assert np.mean(hit) >= 0.8, np.mean(hit)


def test_expected_false_alarms_matches_budget():
    rate = torch.from_numpy(_rate().astype(np.float32))
    thr = threshold_map(rate, window=3, fp_per_frame=5)
    assert expected_false_alarms(rate, thr, 3) <= 5.0 + 1e-6


def test_pipeline_poisson_mode_end_to_end():
    from midas_nf_preprocess.process_images.params import ProcessParams
    from midas_nf_preprocess.process_images.pipeline import ProcessImagesPipeline

    rng = np.random.default_rng(3)
    stack = torch.from_numpy(_bg(rng, 24))
    spots, _ = _spots(rng, n_spots=30, amp=6.0)
    stack[5] += torch.from_numpy(rng.poisson(spots).astype(np.float32))
    p = ProcessParams(nr_pixels=Z, n_distances=1, nr_files_per_distance=24, spot_detect="poisson",
                      poisson_rate_frames=24, poisson_rate_smooth=16, mean_filt_radius=0)
    pipe = ProcessImagesPipeline(p, device="cpu")
    median = pipe.temporal_median(stack)
    pipe._prepare_poisson(median, stack=stack)
    res_spot = pipe.process_frame(5, stack[5], median, 1)
    res_bg = pipe.process_frame(6, stack[6], median, 1)
    assert torch.allclose(res_spot.filtered, stack[5] - median)   # intensities untouched
    assert res_spot.n_spots >= 20 and res_bg.n_spots <= 6, (res_spot.n_spots, res_bg.n_spots)


def test_poisson_keys_parse(tmp_path):
    from midas_nf_preprocess.process_images.params import ProcessParams
    f = tmp_path / "p.txt"
    f.write_text("SpotDetect poisson\nPoissonWindow 5\nPoissonFPPerFrame 2.5\nPoissonRateFrames 40\n"
                 "PoissonRateSmooth 24\nPoissonClip 7\nPoissonMinPx 6\n")
    p = ProcessParams.from_paramfile(f)
    assert (p.spot_detect, p.poisson_window, p.poisson_fp_per_frame, p.poisson_rate_frames,
            p.poisson_rate_smooth, p.poisson_clip, p.poisson_min_px) == ("poisson", 5, 2.5, 40, 24, 7.0, 6)


# --- measured dispersion (scintillator-like, clustered, non-Poisson ADU) -----------------------

def _clustered_bg(rng, n):
    """Each X-ray event lights a cluster at 4 ADU: 1x1 in quiet rows, 3x3 in the busy band (same mean
    per pixel), mimicking PUR #255 (3x3-sum var/mean 3.1 quiet vs 20 busy)."""
    out = np.zeros((n, Z, Y), np.float32)
    for i in range(n):
        ev_q = rng.poisson(0.38, size=(Z, Y)).astype(np.float32)            # 1x1 events
        ev_b = rng.poisson(0.38 / 9, size=(Z, Y)).astype(np.float32)        # 3x3 events, same mean
        img = 4 * ev_q
        img_b = 4 * ndi.uniform_filter(ev_b, 3, mode="constant") * 9
        img[BAND] = img_b[BAND]
        out[i] = img
    return out


def test_measured_dispersion_holds_budget_where_poisson_fails():
    from midas_nf_preprocess.process_images.poisson_threshold import local_moments_from_stack, threshold_map_nb
    rng = np.random.default_rng(7)
    stack = torch.from_numpy(_clustered_bg(rng, 60))
    med = torch.zeros(Z, Y)
    mean, var = local_moments_from_stack(stack, med, window=3, n_frames=60, smooth_px=16)
    disp_b = float((var / mean)[BAND].median()); disp_q = float((var / mean)[:100].median())
    assert disp_b > 3 * disp_q                                               # busy band far more dispersed
    t_nb = threshold_map_nb(mean, var, fp_per_frame=5)
    rate_px = mean / 9.0
    t_pois = threshold_map(rate_px, window=3, fp_per_frame=5)
    test = _clustered_bg(np.random.default_rng(8), 20)
    fp_nb = fp_pois = 0
    for f in test:
        fp_nb += detect_labels_poisson(torch.from_numpy(f), t_nb, window=3, min_px=4)[1]
        fp_pois += detect_labels_poisson(torch.from_numpy(f), t_pois, window=3, min_px=4)[1]
    assert fp_nb / len(test) <= 10, fp_nb / len(test)          # near the 5/frame pixel budget
    assert fp_pois > 20 * max(fp_nb, 1), (fp_pois, fp_nb)      # assuming Poisson floods


def test_streaming_moments_equal_in_memory():
    from midas_nf_preprocess.process_images.poisson_threshold import local_moments_from_stack, streaming_local_moments
    rng = np.random.default_rng(9)
    stack = _clustered_bg(rng, 30)

    class Src:
        n_frames, nz, ny = 30, Z, Y
        def read_rows(self, idx, r0, r1):
            return stack[list(idx), r0:r1]
    med = torch.zeros(Z, Y)
    kw = dict(window=3, n_frames=30, clip=40.0, smooth_px=8)
    m_a, v_a = local_moments_from_stack(torch.from_numpy(stack), med, **kw)
    m_b, v_b = streaming_local_moments(Src(), med, row_block=70, **kw)
    assert torch.allclose(m_a, m_b, atol=1e-4) and torch.allclose(v_a, v_b, atol=1e-3)


def test_robust_variance_ignores_rare_spot_windows():
    """Regression (PUR #255, 2026-09-29): plain variance over frames read var/mean 22.8 in the busy band
    because a few windows sit on real spots; MAD-based read 2.7. Contamination must not inflate the
    threshold map."""
    from midas_nf_preprocess.process_images.poisson_threshold import local_moments_from_stack, threshold_map_nb
    rng = np.random.default_rng(31)
    stack = torch.from_numpy(_clustered_bg(rng, 60))
    for j in range(0, 60, 2):                       # a bright spot in half the frames, different places
        z, y = rng.integers(20, Z - 20), rng.integers(20, Y - 20)
        stack[j, z - 2:z + 3, y - 2:y + 3] += 60.0
    med = torch.zeros(Z, Y)
    m_r, v_r = local_moments_from_stack(stack, med, window=3, smooth_px=9, robust=True, clip=200.0)
    m_p, v_p = local_moments_from_stack(stack, med, window=3, smooth_px=9, robust=False, clip=200.0)
    d_r = float((v_r / m_r).median()); d_p = float((v_p / m_p).max())
    assert d_p > 3 * d_r, (d_r, d_p)                # plain variance is hijacked near spots, robust is not
    assert float(threshold_map_nb(m_r, v_r).median()) < float(threshold_map_nb(m_p, v_p).max())


def test_fine_smoothing_follows_static_structure():
    """A static bright patch (mean 3x the surroundings) must raise its own threshold at smooth_px=9 but
    is averaged away at smooth_px=64 -- which is why a coarse map let static clumps through."""
    from midas_nf_preprocess.process_images.poisson_threshold import local_moments_from_stack, threshold_map_nb
    rng = np.random.default_rng(32)
    rate = np.full((Z, Y), 0.3); rate[140:180, 140:180] = 0.9
    stack = torch.from_numpy(rng.poisson(rate, size=(60, Z, Y)).astype(np.float32) * 2)
    med = torch.zeros(Z, Y)
    t_fine = threshold_map_nb(*local_moments_from_stack(stack, med, smooth_px=9), fp_per_frame=5)
    t_coarse = threshold_map_nb(*local_moments_from_stack(stack, med, smooth_px=64), fp_per_frame=5)
    inside_f, outside_f = float(t_fine[150:170, 150:170].mean()), float(t_fine[20:60, 20:60].mean())
    inside_c = float(t_coarse[150:170, 150:170].mean())
    assert inside_f > 1.8 * outside_f, (inside_f, outside_f)
    assert inside_f > 1.3 * inside_c, (inside_f, inside_c)
