"""Stages after the per-window pass: windows, matrix classification, features
(with a raw-photon before/after test), and the candidate-cell test."""
from __future__ import annotations

import json
import os
from typing import Dict, Optional, Sequence

import numpy as np

from midas_hkls.feature_phase import allowed_d_lines, feature_phase_test, pixel_weighted_sampler
from midas_integrate_v2.streaming.snapshot_profile import windows_from_trace
from midas_peakfit.snapshot_detect import local_background, ring_mean_background, valid_region
from midas_peakfit.tracks import merge_detections, window_test

from .config import SnapshotConfig
from .geometry import Maps, rel_to_lines
from .io import list_frames, load_mask, sum_frames


def load_run(out: str, W: int):
    # atleast_1d: a one-window series (single still) parses to a 0-d record otherwise
    win = np.atleast_1d(np.genfromtxt(os.path.join(out, f"windows_W{W}.csv"), delimiter=",", names=True))
    spots = np.load(os.path.join(out, f"spots_W{W}.npy"))
    with open(os.path.join(out, f"meta_W{W}.json")) as f:
        meta = json.load(f)
    return win, spots, meta


def choose_windows(cfg: SnapshotConfig, win, n_frames: int) -> dict:
    """Explicit windows from the config, else the stated rule on the halo trace."""
    if cfg.before and cfg.after:
        return dict(before=tuple(cfg.before), after=tuple(cfg.after), rule="explicit (config)")
    w = windows_from_trace(win["first_frame"], win["halo"], n_frames, rise=cfg.rise,
                           guard=cfg.guard, after_len=cfg.after_len,
                           baseline_until=cfg.baseline_until, min_before=cfg.min_before)
    if w is None:
        return dict(before=None, after=None, rule="no onset found / before window too short")
    return dict(before=w.before, after=w.after, onset=w.onset, baseline=w.baseline, rule=w.rule)


def in_window(first: np.ndarray, W: int, rng) -> np.ndarray:
    return (first >= rng[0]) & (first + W - 1 <= rng[1])


def classify(spots: np.ndarray, W: int, windows: dict, cfg: SnapshotConfig) -> dict:
    """sigma_matrix from the matrix core of the before+after windows; off-matrix flags."""
    first, rel = spots[:, 0], spots[:, 9]
    sel = np.zeros(len(spots), bool)
    for key in ("before", "after"):
        if windows.get(key):
            sel |= in_window(first, W, windows[key])
    core = sel & np.isfinite(rel) & (np.abs(rel) < cfg.core_rel)
    c = rel[core]
    sigma = float(1.4826 * np.median(np.abs(c - np.median(c)))) if c.size else float("nan")
    off = np.isfinite(rel) & (np.abs(rel) > cfg.nsig_off * sigma)
    return dict(sigma_matrix=sigma, off=off, analysed=sel, n_core=int(core.sum()))


def features(cfg: SnapshotConfig, spots: np.ndarray, W: int, cls: dict, windows: dict,
             maps: Maps, stride: int = 1) -> dict:
    """Off-matrix detections in the BEFORE window, merged into features, then tested
    on raw photons: present before, absent after (aperture Poisson test)."""
    b, a = windows["before"], windows["after"]
    first = spots[:, 0]
    sel = cls["off"] & in_window(first, W, b)
    s = spots[sel]
    F = merge_detections(s[:, 0], s[:, 1], s[:, 2], radius=cfg.merge_px, min_det=cfg.min_det,
                         values={"d": s[:, 8], "tth": s[:, 3], "eta": s[:, 4], "rel": s[:, 9]})
    n = len(F["row"])
    out = {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in F.items() if k != "label"}
    out.update(n_features=n, window_before=b, window_after=a, stride=stride)
    if n == 0 or a is None:
        # no "after" window (static series): no before/after photon test
        out.update(present_before=[True] * n if a is None else [], absent_after=[False] * n if a is None else [])
        return out
    files = list_frames(cfg.frames)
    mask = load_mask(cfg.mask, cfg.flip)
    pb = files[b[0]:b[1] + 1:stride]
    pa = files[a[0]:a[1] + 1:stride]
    A = sum_frames(pb, cfg.flip, mask, cfg.invalid_below)
    B = sum_frames(pa, cfg.flip, mask, cfg.invalid_below)
    ok = (A >= 0) & (B >= 0)
    bkgA = local_background(A, ok, ring_mean_background(A, ok, maps.ring_index)[1], cfg.local_box)
    bkgB = local_background(B, ok, ring_mean_background(B, ok, maps.ring_index)[1], cfg.local_box)
    t = window_test(A, bkgA, len(pb), B, bkgB, len(pa), ok, F["row"], F["col"])
    out.update(present_before=t.present_a.tolist(), absent_after=t.absent_b.tolist(),
               net_rate_before=t.net_rate_a.tolist(), net_rate_after=t.net_rate_b.tolist(),
               p_present_before=t.p_present_a.tolist(), p_drop=t.p_drop.tolist())
    return out


def matrix_band_exclude(lines: np.ndarray, scale: float, width: float, extra=()):
    """True where d lies within ``width`` of a known-phase ring: the primary (``lines``,
    ``scale``) or any ``extra`` (lines, scale) pair."""
    def excl(d):
        m = np.abs(rel_to_lines(d, lines, scale)) <= width
        for el, es in extra:
            if np.isfinite(es):
                m |= np.abs(rel_to_lines(d, el, es)) <= width
        return m
    return excl


def cross_series_recurrence(positions: Dict[str, tuple], radius: float = 4.0) -> Dict[str, np.ndarray]:
    """Features that recur at the same detector pixel in ANOTHER series.

    ``positions`` maps a series name to (rows, cols) of its features. Series taken at
    different sample positions see different crystallites, so a feature within ``radius``
    px of a feature of another series points to a source fixed to the instrument (or a
    coincidence); such features must be removed, or at least counted, before series are
    pooled into one statistic -- the pooled null assumes independent features.
    Returns name -> bool array (True = recurs elsewhere)."""
    names = list(positions)
    out = {}
    for n in names:
        r, c = (np.asarray(x, float) for x in positions[n])
        flag = np.zeros(r.size, bool)
        for m in names:
            if m == n or r.size == 0:
                continue
            r2, c2 = (np.asarray(x, float) for x in positions[m])
            if r2.size == 0:
                continue
            dist = np.hypot(r[:, None] - r2[None, :], c[:, None] - c2[None, :])
            flag |= dist.min(axis=1) <= radius
        out[n] = flag
    return out


def phase_test(cfg: SnapshotConfig, feats: dict, cls: dict, maps: Maps, matrix_lines: np.ndarray,
               matrix_scale: float, candidates: Dict[str, object], *, require_vanishing: bool = True,
               control_factors: Sequence[float] = (0.93, 1.07), scale_range=(1.0, 1.025),
               n_null: int = 2000, alpha: float = 0.05,
               valid: Optional[np.ndarray] = None, extra=()) -> dict:
    """Candidate-cell test on features. ``candidates`` maps a label to a Crystal.
    Negative controls are generated automatically: each candidate's line pattern
    rescaled by ``control_factors`` (same pattern, wrong cell); they must fail.
    ``valid`` is the detector validity mask (gaps, mask); the null's 2-theta coverage
    is built only from pixels where a feature could have been reported."""
    sel = np.ones(feats["n_features"], bool)
    if require_vanishing and feats["n_features"]:
        sel = np.asarray(feats["present_before"]) & np.asarray(feats["absent_after"])
    d = np.asarray(feats["d"], float)[sel] if feats["n_features"] else np.array([])
    tol = 2 * cls["sigma_matrix"]
    ok = np.isfinite(maps.tth) if valid is None else (valid & np.isfinite(maps.tth))
    safe = valid_region(ok, 3, cfg.margin_px)
    tth_valid = maps.tth[safe]
    if cfg.tth_max:
        tth_valid = tth_valid[tth_valid < cfg.tth_max]
    sampler = pixel_weighted_sampler(tth_valid, maps.wavelength,
                                     exclude=matrix_band_exclude(matrix_lines, matrix_scale,
                                                                 cfg.nsig_off * cls["sigma_matrix"], extra))
    tmax = float(np.nanmax(tth_valid))
    dmin = maps.wavelength / (2 * np.sin(np.radians(tmax / 2)))
    cand_lines = {k: allowed_d_lines(c, dmin, 50.0) for k, c in candidates.items()}
    controls = {f"{k}_x{f:g}": v * f for k, v in cand_lines.items() for f in control_factors}
    res = feature_phase_test(d, tol, cand_lines, sampler, controls=controls, scale_range=scale_range,
                             n_null=n_null, alpha=alpha, seed=cfg.seed)
    res["rows"] = [r.__dict__ for r in res["rows"]]
    res.update(n_features_tested=int(len(d)), tol=tol, require_vanishing=require_vanishing)
    return res


def map_features(cfg: SnapshotConfig, spots: np.ndarray, cls: dict, *, recur_frac: float = 0.05,
                 recur_min: int = 3) -> dict:
    """Features for a POSITION MAP (each frame is a different sample position).

    Detections are not merged across frames: that would glue crystals from different
    positions together. Instead a detection is dropped as detector-fixed when its pixel
    (within ``merge_px``) hosts off-matrix detections in more than ``recur_frac`` of the
    frames (and at least ``recur_min``). Within one frame, detections closer than
    ``merge_px`` are one feature.
    """
    from scipy.spatial import cKDTree
    off = cls["off"]
    s = spots[off]
    n_frames = len(np.unique(spots[:, 0])) if len(spots) else 0
    if len(s) == 0:
        return dict(n_features=0, n_detector_fixed=0, d=[], frame=[], row=[], col=[], tth=[])
    tree = cKDTree(s[:, 1:3])
    neigh = tree.query_ball_point(s[:, 1:3], cfg.merge_px)
    n_fr = np.array([len(np.unique(s[j, 0])) for j in neigh])
    fixed = n_fr > max(recur_min, recur_frac * max(n_frames, 1))
    keep = ~fixed
    k = s[keep]
    # within-frame dedupe: keep the highest-score detection of each close group
    order = np.argsort(-k[:, 7])
    taken, sel = [], []
    for i in order:
        if any(k[j, 0] == k[i, 0] and np.hypot(*(k[j, 1:3] - k[i, 1:3])) < cfg.merge_px for j in taken):
            continue
        taken.append(i); sel.append(i)
    k = k[np.sort(sel)]
    return dict(n_features=int(len(k)), n_detector_fixed=int(fixed.sum()), n_frames=n_frames,
                d=k[:, 8].tolist(), frame=k[:, 0].tolist(), row=k[:, 1].tolist(),
                col=k[:, 2].tolist(), tth=k[:, 3].tolist(), rule=(
                    f"off-matrix; detector-fixed if its pixel has off-matrix detections in "
                    f"> max({recur_min}, {recur_frac} x frames) frames; within-frame dedupe {cfg.merge_px} px"))
