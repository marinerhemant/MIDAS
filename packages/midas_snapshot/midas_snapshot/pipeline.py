"""Per-window pass over a still-frame series.

For every window of ``W`` consecutive frames (W = 1 is single frames): sum,
ring profile, matrix scale, halo index, calibrated spot detection, and for every
spot its scattering angle, d-spacing and signed relative distance to the nearest
matrix line at that window's scale.

Outputs (in ``cfg.out``): ``windows_W{W}.csv``, ``spots_W{W}.npy`` (+ columns in
``meta_W{W}.json``), and the threshold and settings used.
"""
from __future__ import annotations

import json
import os
import time
from multiprocessing import Pool
from typing import Optional

import numpy as np
from scipy import ndimage

from midas_integrate_v2.streaming.snapshot_profile import band_excess, fit_matrix_scale, ring_profile
from midas_peakfit.snapshot_detect import (SnapshotDetectorConfig, calibrate_threshold,
                                           detect_spots, measure_spot_sigma, ring_mean_background)

from .config import SnapshotConfig
from .geometry import Maps, build_maps, fit_window, matrix_lines, rel_to_lines
from .io import list_frames, load_mask, sum_frames

SPOT_COLUMNS = ["first_frame", "row", "col", "tth", "eta", "flux", "peak", "score", "d", "rel_matrix"]
_G: dict = {}


def detector_config(cfg: SnapshotConfig) -> SnapshotDetectorConfig:
    return SnapshotDetectorConfig(sigma_px=cfg.sigma_px, local_box=cfg.local_box,
                                  statistic=cfg.statistic, fa_per_image=cfg.fa_per_image,
                                  margin_px=cfg.margin_px)


def _init(cfg_dict, maps: Maps, lines, fit_lines, threshold, extra=()):
    _G.update(cfg=SnapshotConfig(**cfg_dict), maps=maps, lines=lines, fit_lines=fit_lines,
              T=threshold, mask=load_mask(cfg_dict["mask"], cfg_dict["flip"]), extra=extra)


def _bkg_level(coarse: np.ndarray, ok: np.ndarray) -> float:
    """A window's background level for threshold lookup: median ring-mean background."""
    v = coarse[ok]
    return float(np.median(v)) if v.size else 0.0


def threshold_at(T, coarse: np.ndarray, ok: np.ndarray) -> float:
    """``T`` is a number, or a calibration table {"b": [...], "T": [...]} (background level ->
    Poisson-null threshold): the window gets the threshold interpolated at its own level."""
    if isinstance(T, dict):
        b = np.asarray(T["b"], float)
        t = np.asarray(T["T"], float)
        o = np.argsort(b)
        return float(np.interp(_bkg_level(coarse, ok), b[o], t[o]))
    return float(T)


def process_image(img: np.ndarray, W: int, first: int, cfg: SnapshotConfig, maps: Maps,
                  lines: np.ndarray, fit_lines: np.ndarray, T, extra=()):
    """``extra``: further known phases as (lines, fit_lines) pairs. A spot's
    ``rel_matrix`` is its signed distance to the nearest ring of ANY known phase, each
    at its own fitted scale in this window (a phase absent from the window is skipped)."""
    ok = img >= 0
    tc, prof = ring_profile(img, ok, maps.tth, cfg.tth_step, tth0=maps.tth0)
    grid = np.arange(*cfg.scale_grid)
    wdeg = fit_window(cfg.fit_window_deg, maps)
    def _scale(fl):
        # A window's scale is used only if the fit is consistent (per-ring scales agree):
        # after a phase change a reference can land on another phase's rings at a wrong
        # scale; an inconsistent fit gives NO scale rather than a wrong one.
        if not len(fl):
            return None, np.nan
        f = fit_matrix_scale(tc, prof / W, fl, maps.wavelength, scale_grid=grid, window_deg=wdeg,
                             ring_tol_deg=maps.pixel_deg)
        return f, (f.scale if f.consistent(cfg.spread_tol) else np.nan)

    mf, scale = _scale(fit_lines)
    extra_scales = [_scale(fl)[1] for _, fl in extra]
    halo = band_excess(tc, prof / W, tuple(cfg.halo_band), tuple(cfg.base_band)) \
        if cfg.halo_band and cfg.base_band else np.nan
    coarse = ring_mean_background(img, ok, maps.ring_index)[1]
    if isinstance(T, str) and T == "per_window":           # this window's own Poisson-null threshold
        from dataclasses import replace as _replace
        T_used = calibrate_threshold(img, ok, coarse, _replace(detector_config(cfg), n_null_images=cfg.n_null_window),
                                     np.random.default_rng([cfg.seed, int(first)]), ring_index=maps.ring_index)
    else:
        T_used = threshold_at(T, coarse, ok)
    det = detect_spots(img, ok, coarse, T_used, detector_config(cfg))
    rows = []
    if len(det):
        r, c = det[:, 0], det[:, 1]
        tth = ndimage.map_coordinates(maps.tth, [r, c], order=1)
        eta = maps.eta[np.clip(np.rint(r).astype(int), 0, img.shape[0] - 1),
                       np.clip(np.rint(c).astype(int), 0, img.shape[1] - 1)]
        d = maps.wavelength / (2 * np.sin(np.radians(tth / 2)))
        rels = np.vstack([rel_to_lines(d, lines, scale)] +
                         [rel_to_lines(d, el, es) for (el, _), es in zip(extra, extra_scales)])
        with np.errstate(invalid="ignore"):
            pick = np.nanargmin(np.where(np.isfinite(rels), np.abs(rels), np.inf), axis=0)
        rel = rels[pick, np.arange(len(d))]
        keep = np.ones(len(det), bool) if cfg.tth_max is None else tth < cfg.tth_max
        rows = np.c_[np.full(len(det), first), r, c, tth, eta, det[:, 2], det[:, 3], det[:, 4], d, rel][keep]
    return (first, mf.scale_coarse if mf else np.nan, scale, mf.n_rings if mf else 0, halo,
            len(rows), *extra_scales), (np.asarray(rows, float).reshape(-1, len(SPOT_COLUMNS))), float(T_used)


def _job(args):
    first, paths = args
    cfg = _G["cfg"]
    img = sum_frames(paths, cfg.flip, _G["mask"], cfg.invalid_below)
    return process_image(img, len(paths), first, cfg, _G["maps"], _G["lines"], _G["fit_lines"], _G["T"],
                         _G.get("extra", ()))


def setup(cfg: SnapshotConfig):
    maps = build_maps(cfg.geometry, cfg.tth_step)
    tmax = float(np.nanmax(maps.tth))
    lines = matrix_lines(cfg.matrix_cif, maps.wavelength, tmax)
    fit_lines = np.sort(lines)[::-1][:cfg.matrix_fit_lines] if len(lines) else lines
    return maps, lines, fit_lines


def _fallback_rel(spots, res, lines, extra):
    """Spots from windows without a consistent fit of their own (sparse single frames)
    are classified against each phase's MEDIAN consistent scale over the series."""
    if not len(spots):
        return spots
    first = np.array([r[0][0] for r in res])
    prim = np.array([r[0][2] for r in res], float)
    ex = [np.array([r[0][6 + k] for r in res], float) for k in range(len(extra))]
    med = [float(np.nanmedian(prim)) if np.isfinite(prim).any() else np.nan] + \
          [float(np.nanmedian(e)) if np.isfinite(e).any() else np.nan for e in ex]
    nan_win = set(first[~np.isfinite(prim) & np.all([~np.isfinite(e) for e in ex] or [np.ones(len(first), bool)], axis=0)])
    idx = np.array([f in nan_win for f in spots[:, 0]]) if nan_win else np.zeros(len(spots), bool)
    if idx.any():
        d = spots[idx, 8]
        rels = np.vstack([rel_to_lines(d, lines, med[0])] +
                         [rel_to_lines(d, el, m) for (el, _), m in zip(extra, med[1:])])
        with np.errstate(invalid="ignore"):
            pick = np.argmin(np.where(np.isfinite(rels), np.abs(rels), np.inf), axis=0)
        spots[idx, 9] = rels[pick, np.arange(len(d))]
    return spots


def extra_phases(cfg: SnapshotConfig, maps: Maps):
    """(lines, fit_lines) for each further known phase in ``cfg.phase_cifs``."""
    tmax = float(np.nanmax(maps.tth))
    out = []
    for cif in cfg.phase_cifs or []:
        L = matrix_lines(cif, maps.wavelength, tmax)
        out.append((L, np.sort(L)[::-1][:cfg.matrix_fit_lines]))
    return out


def threshold_for(cfg: SnapshotConfig, maps: Maps, paths, with_level: bool = False):
    img = sum_frames(paths, cfg.flip, load_mask(cfg.mask, cfg.flip), cfg.invalid_below)
    ok = img >= 0
    coarse = ring_mean_background(img, ok, maps.ring_index)[1]
    T = calibrate_threshold(img, ok, coarse, detector_config(cfg),
                            np.random.default_rng(cfg.seed), ring_index=maps.ring_index)
    return (T, _bkg_level(coarse, ok)) if with_level else T


def spot_sigma_for(cfg: SnapshotConfig, maps: Maps, paths) -> tuple:
    """Measured matched-filter width on a window (see midas_peakfit measure_spot_sigma)."""
    img = sum_frames(paths, cfg.flip, load_mask(cfg.mask, cfg.flip), cfg.invalid_below)
    ok = img >= 0
    coarse = ring_mean_background(img, ok, maps.ring_index)[1]
    probe = SnapshotDetectorConfig(sigma_px=1.0, local_box=cfg.local_box, margin_px=cfg.margin_px)
    return measure_spot_sigma(img, ok, coarse, probe)


def calibrate_windows(cfg: SnapshotConfig, maps: Maps, wins, files) -> tuple:
    """Calibration table over ``cfg.n_calib_windows`` windows spread through the series:
    (background level, Poisson-null threshold) per window. A Poisson-null threshold depends on the
    background level: on real data a cold pre-melt window needed ~45 where hot and after-melt windows
    needed ~8, and one threshold for all windows either floods the pre-melt period with false
    peaks (0.05-0.30 per image against a 0.05 target) or blinds the rest. Each window then uses
    the threshold interpolated at its own level (:func:`threshold_at`).
    Returns (table, per-window thresholds, window starts)."""
    k = max(1, min(cfg.n_calib_windows, len(wins)))
    pick = sorted({int(round(x)) for x in np.linspace(0, len(wins) - 1, k)})
    tb = [threshold_for(cfg, maps, [files[i] for i in wins[j]], with_level=True) for j in pick]
    table = {"b": [float(b) for _, b in tb], "T": [float(t) for t, _ in tb]}
    return table, table["T"], [int(wins[j][0]) for j in pick]


def run(cfg: SnapshotConfig, W: int, first: int = 0, last: Optional[int] = None,
        calib_paths=None) -> dict:
    """Process all windows of size W; returns a summary dict (also written as meta)."""
    t0 = time.time()
    os.makedirs(cfg.out, exist_ok=True)
    files = list_frames(cfg.frames)
    last = len(files) if last is None else last
    idx = list(range(first, last))
    wins = [idx[i:i + W] for i in range(0, len(idx) - W + 1, W)]
    maps, lines, fit_lines = setup(cfg)
    extra = extra_phases(cfg, maps)
    sigma_note = None
    if isinstance(cfg.sigma_px, str):                      # "auto": measure on the last (solid) window
        sg, n_sg = spot_sigma_for(cfg, maps, [files[i] for i in wins[-1]])
        cfg.sigma_px = float(np.clip(sg, 0.6, 3.0)) if np.isfinite(sg) else 1.7
        sigma_note = dict(measured=sg, n_spots=n_sg, used=cfg.sigma_px)
    calib = calib_paths or [files[i] for i in wins[-1]]
    T_all = T_starts = None
    if calib_paths or cfg.threshold_mode == "single":
        T = threshold_for(cfg, maps, calib)
    elif cfg.threshold_mode == "table":
        T, T_all, T_starts = calibrate_windows(cfg, maps, wins, files)
    else:                                                   # "per_window" (default)
        T = "per_window"
    jobs = [(w[0], [files[i] for i in w]) for w in wins]
    from dataclasses import asdict
    with Pool(cfg.nproc, initializer=_init, initargs=(asdict(cfg), maps, lines, fit_lines, T, extra)) as pool:
        res = pool.map(_job, jobs, chunksize=max(1, len(jobs) // (4 * cfg.nproc)))
    res.sort(key=lambda x: x[0][0])
    win = np.array([r[0][:6] for r in res], float)
    if extra:
        ph = np.array([[r[0][0], *r[0][6:]] for r in res], float)
        names = [os.path.splitext(os.path.basename(c))[0] for c in cfg.phase_cifs]
        np.savetxt(os.path.join(cfg.out, f"phase_scales_W{W}.csv"), ph, delimiter=",", comments="",
                   header="first_frame," + ",".join(names), fmt="%.6f")
    spots = np.concatenate([r[1] for r in res]) if res else np.zeros((0, len(SPOT_COLUMNS)))
    spots = _fallback_rel(spots, res, lines, extra)
    np.savetxt(os.path.join(cfg.out, f"windows_W{W}.csv"), win, delimiter=",", comments="",
               header="first_frame,scale_coarse,scale,n_rings,halo,n_spots",
               fmt=["%d", "%.6f", "%.6f", "%d", "%.5f", "%d"])
    np.save(os.path.join(cfg.out, f"spots_W{W}.npy"), spots)
    meta = dict(window=W, n_windows=len(jobs), n_frames=len(files), threshold=T,
                spot_columns=SPOT_COLUMNS, wavelength=maps.wavelength,
                fit_window_deg=fit_window(cfg.fit_window_deg, maps), pixel_deg=maps.pixel_deg,
                matrix_lines=[float(x) for x in np.sort(lines)], matrix_fit_lines=[float(x) for x in fit_lines],
                calibration_frames=[os.path.basename(p) for p in calib],
                threshold_mode=("single" if calib_paths else cfg.threshold_mode),
                thresholds_per_window=T_all if T_all is not None else [r[2] for r in res],
                threshold_window_starts=T_starts if T_starts is not None else [int(r[0][0]) for r in res],
                sigma_px_used=cfg.sigma_px, sigma_px_measurement=sigma_note,
                wall_s=time.time() - t0, config=asdict(cfg))
    with open(os.path.join(cfg.out, f"meta_W{W}.json"), "w") as f:
        json.dump(meta, f, indent=1)
    return meta
