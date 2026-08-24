"""Recommend ``RingThresh`` per ring, from the data.

Replaces the manual recipe in ``manuals/manuals/ff-hedm/README.md`` section 6b.

Why the old recipe is not enough
--------------------------------
The handbook rule was "sweep the threshold, pick the lowest value before the
blob count explodes". On ``Au3_cubes_ff_000008`` that gives 10, and 10 is what
was used -- yet a later audit of that run found **1309 of 2076** recorded spots
sat below SNR 5, i.e. the peak finder was working deep in the noise.

The knee is real but it is not the onset of noise admission; it is the point
where noise *percolates* into detector-spanning blobs. Measured on that
dataset, blobs/frame surviving the size filter went 18 (thr 5) -> 4 (10) ->
2.5 (20) -> 1.0 (40..150). The explosion is between 5 and 10, but noise is
still being removed all the way out to ~100. Picking the knee therefore picks
a threshold roughly an order of magnitude too low.

What this module does instead
-----------------------------
Two independent criteria, always reported side by side (they should agree; a
disagreement is itself diagnostic of a bad band or a broken dark):

**A. Blob SNR.** For each candidate threshold, every surviving blob gets its
own local SNR -- peak height over a local median, divided by ``1.4826*MAD`` of
a surrounding annulus. This is the discriminator that separated real spots
(median SNR ~1989) from noise (~2.7) in the Au3 audit. The recommendation is
the lowest threshold at which at least ``snr_clean_frac`` of surviving blobs
clear ``snr_min``.

**B. Expected false positives.** From the per-cell noise sigma, the expected
number of noise blobs over the *whole scan* is estimated analytically, taking
the ``minNrPx`` size filter into account (a blob needs >= 2 adjacent pixels
above threshold, which is far rarer than one). The recommendation is the
lowest threshold whose expected false-positive count over the scan is below
``max_false_positives``. This is the criterion that matters in a sparse
regime -- a 2-grain dataset has ~1-2 real peaks per frame against ~3e5 in-band
pixels per frame, so even a 1e-6 per-pixel false rate swamps the signal.

**C. Peak resolvability (segmentation).** A and B are both *detection*
criteria: they ask whether a blob is real. Neither can see a blob that is real
but is actually SEVERAL spots fused into one connected component, because
merging changes neither the blob's SNR nor the noise statistics -- only what
the blob CONTAINS. Lower the threshold far enough and every peak finder
percolates: on bt_1id_jun25b s1 the A/B recommendation of 20-30 produced regions
holding >=400 peaks (the ``maxNPeaks`` cap), against a healthy 1-15 on the
other three samples of the same experiment, and a 400-peak coupled fit is a
blob decomposition rather than a peak fit. So each surviving region is passed
to the PRODUCTION seeder (``find_regional_maxima`` -- the very call the fitter
makes, so the count here IS the fitter's ``n_peaks``) and the recommendation is
the lowest threshold whose 99th-percentile peaks-per-region stays at or below
``p99_peaks_max``. The threshold maximising the number of cleanly-resolved
single-peak regions is reported too: lowering the threshold gains real spots
until percolation and then loses them again, so that maximum is a physical
operating point that needs no per-dataset tuning.

The final recommendation is the strictest of A, B and C -- all three are lower
bounds, for independent reasons.

Run it **per scan**. Crowding is a property of the scan, not the experiment: a
sparse scan should not be forced to the higher threshold that a dense one
needs.

Everything runs through the *production* peak-search path
(``compute_good_coords`` -> ``preprocess_frame`` -> ``find_regions`` ->
``filter_regions_by_size``). That is deliberate: an independent
reimplementation of the band mask disagreed with the real pipeline by ~67x,
because it missed ``transpose_square`` and the distortion-corrected ``Rt``.
Geometry is evaluated once and only the threshold values change per sweep.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

# ── MIDAS preflight: richer argument errors when midas-params is installed ───
_MIDAS_DIST = "midas-peakfit"


def _midas_make_parser(*a, **kw):
    """ArgumentParser factory. Uses midas_params' subclass when available so
    argument errors carry the running version and a did-you-mean; falls back to
    stock argparse otherwise, so this stays an optional dependency."""
    try:
        from midas_params.preflight import MidasArgumentParser
    except Exception:
        return argparse.ArgumentParser(*a, **kw)
    return MidasArgumentParser(*a, package=_MIDAS_DIST, **kw)


DEFAULT_SWEEP = (5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 500)
DEFAULT_SNR_MIN = 5.0
DEFAULT_SNR_CLEAN_FRAC = 0.90
DEFAULT_MAX_FALSE_POSITIVES = 10.0
SIG_HALF, BG_INNER, BG_OUTER = 2, 6, 40


@dataclass
class RingSweepPoint:
    threshold: float
    n_blobs: float               # per frame, before the size filter
    n_kept: float                # per frame, after minNrPx/maxNrPx
    largest: float
    median_snr: float
    frac_snr_ok: float
    expected_false_positives: float   # over the WHOLE scan
    # Criterion C — peak resolvability. Counted with the PRODUCTION seeder
    # (find_regional_maxima), so these are the same n_peaks the fitter will see.
    n_resolved: float = 0.0      # per frame, ONE maximum AND above snr_min
    frac_merged: float = 0.0     # fraction of kept regions holding >= 2 maxima
    p99_peaks: float = 0.0       # 99th pct of peaks-per-region
    max_peaks: float = 0.0       # worst region (CLIPPED at p.maxNPeaks)


@dataclass
class RingRecommendation:
    ring_nr: int
    radius_px: float
    sweep: List[RingSweepPoint] = field(default_factory=list)
    thresh_snr: Optional[float] = None
    thresh_fp: Optional[float] = None
    thresh_merge: Optional[float] = None    # criterion C: segmentation floor
    thresh_best_resolved: Optional[float] = None   # argmax n_resolved
    noise_sigma: float = 0.0
    bg_spread: float = 0.0
    warnings: List[str] = field(default_factory=list)

    @property
    def recommended(self) -> Optional[float]:
        """The strictest of the three criteria; whichever exist.

        All three are LOWER bounds on the threshold and for different reasons:
        A and B say "below this you admit noise"; C says "below this, distinct
        spots merge into one connected component and stop being separately
        fittable". A and B are blind to C — merging changes neither a blob's
        SNR nor the noise statistics, only what the blob CONTAINS — which is
        how bt_1id_jun25b s1 was recommended 20-30 and produced regions of >=400
        peaks (the cap), against a healthy 1-15 elsewhere in the same
        experiment.
        """
        vals = [v for v in (self.thresh_snr, self.thresh_fp, self.thresh_merge)
                if v is not None]
        return max(vals) if vals else None

    @property
    def criteria_agree(self) -> bool:
        if self.thresh_snr is None or self.thresh_fp is None:
            return False
        lo = min(self.thresh_snr, self.thresh_fp)
        hi = max(self.thresh_snr, self.thresh_fp)
        return hi <= 2.0 * max(lo, 1e-9)


#: Criterion C tolerance: the 99th percentile of peaks-per-region must stay
#: at or below this. 3 is deliberately loose — a genuinely uncrowded band
#: sits at 1-2 (bt_1id_jun25b s5 measured median 1 / p90 3), while a merged one
#: runs to hundreds, so anything in 2..10 separates the two regimes.
DEFAULT_P99_PEAKS_MAX = 3.0


def blob_snr(
    img: np.ndarray, rows: np.ndarray, cols: np.ndarray,
    valid: Optional[np.ndarray] = None,
) -> float:
    """Local SNR of one blob: peak height over an annulus median, in MAD units.

    The annulus (``BG_INNER``..``BG_OUTER`` px around the blob centroid)
    excludes the blob itself, so a bright spot cannot inflate its own
    background -- the failure mode that makes a mean/std estimator useless
    here.

    ``img`` must be a corrected frame built WITHOUT the ring-band mask, so it
    carries real background everywhere. Two earlier versions of this got it
    wrong in opposite directions and both produced confident nonsense:

    * on a *thresholded* frame every sub-threshold pixel is 0, so the MAD
      collapses and every SNR reads 0;
    * on a *band-masked* frame the annulus is mostly out-of-band zeros, so
      restricting it to in-band pixels was needed -- but that strip carries
      spot wings and elevated ring background, which understates the noise
      floor. Measured on the reference dataset that made this estimator report
      90.8% of blobs above SNR 5 where an unrestricted 81x81 box on the raw
      frame said 53.4%. The band-restricted form is over-optimistic; do not
      reintroduce it.

    ``valid`` is kept for callers that genuinely have only a masked frame, but
    the default (unrestricted, on an unmasked frame) is the honest one.
    """
    if rows.size == 0:
        return 0.0
    r0, c0 = int(round(rows.mean())), int(round(cols.mean()))
    n_r, n_c = img.shape
    lo_r, hi_r = max(0, r0 - BG_OUTER), min(n_r, r0 + BG_OUTER + 1)
    lo_c, hi_c = max(0, c0 - BG_OUTER), min(n_c, c0 + BG_OUTER + 1)
    patch = img[lo_r:hi_r, lo_c:hi_c]
    if patch.size < 16:
        return 0.0
    rr, cc = np.mgrid[lo_r:hi_r, lo_c:hi_c]
    d = np.maximum(np.abs(rr - r0), np.abs(cc - c0))
    sel = d >= BG_INNER
    if valid is not None:
        sel &= valid[lo_r:hi_r, lo_c:hi_c]
    ann = patch[sel]
    if ann.size < 16:
        return 0.0
    bg = float(np.median(ann))
    sd = float(np.median(np.abs(ann - bg))) * 1.4826
    if sd <= 0:
        return 0.0
    peak = float(img[rows, cols].max())
    return (peak - bg) / sd


def expected_false_blobs(
    threshold: float, sigma: float, n_band_px: int, n_frames: int,
    min_n_px: int = 1,
) -> float:
    """Expected count of pure-noise blobs over the whole scan.

    Gaussian tail per pixel ``p = 0.5*erfc(t / (sigma*sqrt2))``. A blob must
    exceed ``min_n_px`` pixels; with the default ``minNrPx=1`` that means at
    least 2 *adjacent* pixels both over threshold. Treating neighbours as
    independent, the per-site rate for a 2-pixel 8-connected cluster is
    ``~4*p^2`` (4 distinct neighbour directions per pixel, avoiding double
    counting). This is an order-of-magnitude estimate, not a precise model --
    it is used to locate the threshold where false positives stop dominating,
    and both the tail and the independence assumption are conservative
    (real detector noise is correlated, so the true count is higher).
    """
    if sigma <= 0 or threshold <= 0:
        return float("inf")
    p = 0.5 * math.erfc(threshold / (sigma * math.sqrt(2.0)))
    if p <= 0.0:
        return 0.0
    trials = float(n_band_px) * float(n_frames)
    if min_n_px <= 0:
        return trials * p
    k = int(min_n_px) + 1          # blob must EXCEED min_n_px pixels
    if k <= 1:
        return trials * p
    # k adjacent pixels all over threshold; 4 independent directions.
    return trials * 4.0 * (p ** k)


def _pick_snr(sweep: Sequence[RingSweepPoint], frac: float) -> Optional[float]:
    for pt in sweep:
        if pt.n_kept > 0 and pt.frac_snr_ok >= frac:
            return pt.threshold
    return None


def _pick_merge(sweep: Sequence[RingSweepPoint], p99_max: float) -> Optional[float]:
    """Segmentation floor: lowest threshold whose regions are still resolvable.

    Sweep is ascending in threshold. Merging is a LOW-threshold failure, so the
    acceptable set is an upper tail: return the first threshold from which
    p99 peaks-per-region stays at or below ``p99_max`` for every higher
    threshold too. Requiring the whole tail (not just the first crossing) stops
    a single noisy sweep point from being read as the floor.
    """
    n = len(sweep)
    for i in range(n):
        if all(sweep[j].p99_peaks <= p99_max for j in range(i, n)):
            # Nothing merged anywhere: the floor does not bind.
            return None if i == 0 else float(sweep[i].threshold)
    return None


def _pick_best_resolved(sweep: Sequence[RingSweepPoint]) -> Optional[float]:
    """Threshold maximising cleanly-resolved (single-maximum) regions.

    Lowering the threshold gains real spots until percolation, then loses them
    to merging, so ``n_resolved`` has an interior maximum. That maximum is a
    physical operating point and needs no per-dataset tuning. Reported for
    inspection; the recommendation itself stays conservative (the floor).
    """
    best = [pt for pt in sweep if pt.n_resolved > 0]
    if not best:
        return None
    return float(max(best, key=lambda pt: pt.n_resolved).threshold)


def _pick_fp(sweep: Sequence[RingSweepPoint], max_fp: float) -> Optional[float]:
    for pt in sweep:
        if pt.expected_false_positives <= max_fp:
            return pt.threshold
    return None


def format_recommendations(recs: Sequence[RingRecommendation]) -> str:
    """Human-readable report plus paste-ready ``RingThresh`` lines."""
    out: List[str] = []
    for rec in recs:
        out.append("")
        out.append(f"── Ring {rec.ring_nr}  (radius {rec.radius_px:.1f} px, "
                   f"noise sigma {rec.noise_sigma:.2f}, "
                   f"background spread {rec.bg_spread:.1f})")
        out.append(f"  {'thr':>7s} {'blobs/fr':>9s} {'kept/fr':>8s} "
                   f"{'largest':>8s} {'med SNR':>9s} {'frac SNR ok':>12s} "
                   f"{'exp. false/scan':>16s} {'p99 pk/reg':>11s} "
                   f"{'resolved/fr':>12s}")
        for pt in rec.sweep:
            out.append(f"  {pt.threshold:7.0f} {pt.n_blobs:9.1f} "
                       f"{pt.n_kept:8.1f} {pt.largest:8.0f} "
                       f"{pt.median_snr:9.1f} {pt.frac_snr_ok:11.0%} "
                       f"{pt.expected_false_positives:16.3g} "
                       f"{pt.p99_peaks:11.0f} {pt.n_resolved:12.1f}")
        a = "n/a" if rec.thresh_snr is None else f"{rec.thresh_snr:.0f}"
        b = "n/a" if rec.thresh_fp is None else f"{rec.thresh_fp:.0f}"
        out.append(f"  criterion A (blob SNR)          -> {a}")
        out.append(f"  criterion B (expected false +)  -> {b}")
        c = "does not bind" if rec.thresh_merge is None else f"{rec.thresh_merge:.0f}"
        out.append(f"  criterion C (peak resolvability) -> {c}")
        if rec.thresh_best_resolved is not None:
            out.append(f"      (most cleanly-resolved spots at threshold "
                       f"{rec.thresh_best_resolved:.0f})")
        if rec.thresh_merge is not None:
            out.append("      C BINDS: below this, distinct spots merge into one "
                       "connected component. A and B cannot see this — merging "
                       "changes neither blob SNR nor the noise floor.")
        if rec.thresh_snr is not None and rec.thresh_fp is not None:
            out.append("  criteria A and B AGREE (within 2x)" if rec.criteria_agree
                       else "  ** criteria A and B DISAGREE by >2x — inspect the "
                            "band and the dark before trusting either **")
        for w in rec.warnings:
            out.append(f"  WARNING: {w}")

    out.append("")
    out.append("Paste into the parameter file:")
    for rec in recs:
        v = rec.recommended
        out.append(f"RingThresh {rec.ring_nr} "
                   f"{'??' if v is None else f'{v:.0f}'}"
                   + ("" if v is not None else
                      "   # NO SAFE VALUE FOUND — see warnings"))
    return "\n".join(out)


# ─── Analysis driver ────────────────────────────────────────────────────────
def _ring_index_map(Rt: np.ndarray, ring_radii: np.ndarray, width_px: float) -> np.ndarray:
    """Per-pixel ring index, -1 outside all bands.

    Uses the SAME last-ring-wins rule as ``geometry.compute_good_coords`` so a
    blob is attributed to the ring whose threshold actually gated it.
    """
    out = np.full(Rt.shape, -1, dtype=np.int32)
    for r, rad in enumerate(ring_radii):
        out[(Rt > rad - width_px) & (Rt < rad + width_px)] = r
    return out


def analyze(
    data_file: str,
    *,
    result_folder: Optional[str] = None,
    n_frames: int = 12,
    sweep: Sequence[float] = DEFAULT_SWEEP,
    snr_min: float = DEFAULT_SNR_MIN,
    snr_clean_frac: float = DEFAULT_SNR_CLEAN_FRAC,
    max_false_positives: float = DEFAULT_MAX_FALSE_POSITIVES,
    n_sectors: int = 36,
    p99_peaks_max: float = DEFAULT_P99_PEAKS_MAX,
) -> List[RingRecommendation]:
    """Sweep thresholds through the production peak-search path.

    Frames are sampled evenly across the full omega range so the estimate is
    not biased by a locally spot-rich or spot-poor stretch of the scan.
    """
    from midas_peakfit.background import build_background_bins, estimate_cell_stats
    from midas_peakfit.connected import filter_regions_by_size, find_regions
    from midas_peakfit.seeds import find_regional_maxima
    from midas_peakfit.geometry import compute_good_coords, compute_rt_eta, load_ring_radii
    from midas_peakfit.orchestrator import _build_panels
    from midas_peakfit.preprocess import (
        apply_threshold, correct_frame, prepare_dark, prepare_flood, prepare_mask,
    )
    from midas_peakfit.zarr_io import load_corrections, parse_zarr_params, read_frame

    p = parse_zarr_params(data_file)
    if result_folder:
        p.ResultFolder = result_folder
    panels = _build_panels(p)
    load_corrections(data_file, p)
    ring_rads = load_ring_radii(p, p.ResultFolder)

    if ring_rads is None or p.nRingsThresh == 0:
        raise SystemExit(
            "No ring radii available. This tool needs hkls.csv in the result "
            "folder and RingThresh entries in the parameter file (it only "
            "reads their ring NUMBERS; the values are what it recommends)."
        )

    global_warnings: List[str] = []
    if getattr(p, "DoFullImage", 0) == 1:
        global_warnings.append(
            "DoFullImage=1: the peak finder uses Thresholds[0] for EVERY pixel "
            "and ignores per-ring values, so only the first recommendation "
            "below will take effect."
        )

    Rt, Eta = compute_rt_eta(p, panels)
    rads = np.asarray(ring_rads, dtype=float)[: p.nRingsThresh]
    ring_idx = _ring_index_map(Rt, rads, float(p.Width))

    # Overlapping bands silently let the higher-index ring win (no break in
    # geometry.compute_good_coords). Detect and say so rather than emit
    # per-ring numbers that cannot all apply.
    for a in range(len(rads)):
        for b in range(a + 1, len(rads)):
            if abs(rads[a] - rads[b]) < 2 * p.Width:
                global_warnings.append(
                    f"ring bands {p.RingNrs[a]} and {p.RingNrs[b]} OVERLAP "
                    f"(radii {rads[a]:.1f}/{rads[b]:.1f} px, Width {p.Width:.1f} px) "
                    f"-- in the overlap the LATER entry silently wins, so these "
                    f"two thresholds are not independent."
                )

    bg_bins = build_background_bins(Rt, Eta, rads, float(p.Width), n_sectors=n_sectors)

    dark = prepare_dark(p.dark, p.NrPixels, p.NrPixelsY, p.NrPixelsZ, p.TransOpt)
    flood = prepare_flood(p.flood, p.NrPixels, p.NrPixelsY, p.NrPixelsZ, p.TransOpt)
    # Captured, not discarded: criterion C needs it to call the production
    # seeder (find_regional_maxima) exactly as the fitter does.
    mask_prepared = prepare_mask(
        p.mask, p.NrPixels, p.NrPixelsY, p.NrPixelsZ, p.TransOpt)

    n_take = max(1, min(int(n_frames), int(p.nFrames)))
    idxs = np.unique(np.linspace(0, p.nFrames - 1, n_take).astype(int))
    raws = [read_frame(data_file, int(i) + p.skipFrame) for i in idxs]

    orig = np.array(p.Thresholds, dtype=float).copy()

    # Per-ring noise sigma and background spread, measured on the corrected
    # frames with NO threshold applied (threshold 0 keeps every in-band pixel).
    # A tiny positive value keeps every in-band pixel (good_coords > 0 is the
    # "in band" test) while gating nothing, so the stats see the raw band.
    gc0 = np.where(ring_idx >= 0, 1e-12, 0.0)
    sig_per_ring: Dict[int, float] = {}
    spread_per_ring: Dict[int, float] = {}
    cell_meds: Dict[int, List[float]] = {r: [] for r in range(len(rads))}
    cell_sigs: Dict[int, List[float]] = {r: [] for r in range(len(rads))}

    # Correct each frame ONCE, ungated. Every noise/SNR number below is taken
    # from these: on a thresholded frame all sub-threshold pixels are 0, so a
    # local background is identically 0 and its MAD collapses -- which makes
    # every SNR read as 0 and every sigma far too small.
    ungated = [
        correct_frame(
            raw, NrPixels=p.NrPixels, NrPixelsY=p.NrPixelsY, NrPixelsZ=p.NrPixelsZ,
            transform_options=p.TransOpt, dark=dark, flood=flood, good_coords=gc0,
            bc=p.bc, bad_px_intensity=p.BadPxIntensity, make_map=p.makeMap,
        )
        for raw in raws
    ]

    # A second, UNMASKED correction: blob SNR needs real background around the
    # blob, and the band-masked frame is 0 outside a 2*Width-wide strip.
    gc_full = np.ones_like(gc0)
    ungated_full = [
        correct_frame(
            raw, NrPixels=p.NrPixels, NrPixelsY=p.NrPixelsY, NrPixelsZ=p.NrPixelsZ,
            transform_options=p.TransOpt, dark=dark, flood=flood,
            good_coords=gc_full, bc=p.bc,
            bad_px_intensity=p.BadPxIntensity, make_map=p.makeMap,
        )
        for raw in raws
    ]
    for img in ungated:
        med, sig = estimate_cell_stats(img, bg_bins)
        for r in range(len(rads)):
            lo, hi = r * n_sectors, (r + 1) * n_sectors
            live = [b for b in range(lo, hi) if bg_bins.counts[b] > 0]
            if not live:
                continue
            cell_meds[r].extend(med[b] for b in live)
            cell_sigs[r].extend(sig[b] for b in live)
    for r in range(len(rads)):
        sig_per_ring[r] = float(np.median(cell_sigs[r])) if cell_sigs[r] else 0.0
        if cell_meds[r]:
            q5, q95 = np.percentile(cell_meds[r], [5, 95])
            spread_per_ring[r] = float(q95 - q5)
        else:
            spread_per_ring[r] = 0.0

    recs = [
        RingRecommendation(
            ring_nr=int(p.RingNrs[r]), radius_px=float(rads[r]),
            noise_sigma=sig_per_ring[r], bg_spread=spread_per_ring[r],
            warnings=list(global_warnings),
        )
        for r in range(len(rads))
    ]

    n_band_px = {r: int((ring_idx == r).sum()) for r in range(len(rads))}
    per_ring_counts: Dict[float, Dict[int, list]] = {}

    for thr in sweep:
        p.Thresholds = np.full_like(orig, float(thr))
        gc = compute_good_coords(p, panels, ring_rads)
        acc = {r: {"n": [], "k": [], "mx": [], "snr": [], "pk": [], "res": []}
               for r in range(len(rads))}
        for raw_img, snr_img in zip(ungated, ungated_full):
            # Gate the already-corrected frame: identical to what
            # preprocess_frame would return at this threshold, but the ungated
            # copy stays available for the SNR measurement below.
            img = apply_threshold(raw_img, gc)
            regs = find_regions(img, gc)
            kept = filter_regions_by_size(regs, p.minNrPx, p.maxNrPx)
            per = {r: {"n": 0, "k": 0, "mx": 0, "snr": [], "pk": [], "res": 0}
                   for r in range(len(rads))}
            for reg in regs:
                r = int(ring_idx[reg.pixel_rows[0], reg.pixel_cols[0]])
                if r < 0:
                    continue
                per[r]["n"] += 1
                per[r]["mx"] = max(per[r]["mx"], reg.n_pixels)
            for reg in kept:
                r = int(ring_idx[reg.pixel_rows[0], reg.pixel_cols[0]])
                if r < 0:
                    continue
                per[r]["k"] += 1
                _snr = blob_snr(snr_img, reg.pixel_rows, reg.pixel_cols)
                per[r]["snr"].append(_snr)
                # Criterion C: how many peaks will the fitter see in this
                # region? Same call, same image, same caps as the production
                # seeder, so the number here IS the fitter's n_peaks.
                fm = find_regional_maxima(
                    reg, img, mask_prepared, p.IntSat, p.maxNPeaks)
                if fm is not None:
                    n_pk = int(len(fm[0]))
                    per[r]["pk"].append(n_pk)
                    # "Resolved" must mean a real spot that is cleanly
                    # separated, so require BOTH one maximum and SNR. Without
                    # the SNR gate this count is dominated by noise: on
                    # bt_1id_jun25b s1 ring 5 at threshold 20 there were 3067
                    # blobs/frame at median SNR 2.2 with 3% clearing snr_min,
                    # and every one of those noise specks is a single-maximum
                    # region — which put the "best resolved" operating point
                    # at the noisiest threshold swept.
                    if n_pk == 1 and _snr > snr_min:
                        per[r]["res"] += 1
            for r in range(len(rads)):
                acc[r]["n"].append(per[r]["n"])
                acc[r]["k"].append(per[r]["k"])
                acc[r]["mx"].append(per[r]["mx"])
                acc[r]["snr"].extend(per[r]["snr"])
                acc[r]["pk"].extend(per[r]["pk"])
                acc[r]["res"].append(per[r]["res"])
        per_ring_counts[float(thr)] = acc

        for r, rec in enumerate(recs):
            s = np.asarray(acc[r]["snr"], dtype=float)
            rec.sweep.append(RingSweepPoint(
                threshold=float(thr),
                # MEAN, not median: spots cluster in omega, so on a sparse
                # sample the median reads 0 or 1 while the scan average is
                # several times higher. The median understates exactly the
                # bursty frames that dominate the total count.
                n_blobs=float(np.mean(acc[r]["n"])),
                n_kept=float(np.mean(acc[r]["k"])),
                largest=float(np.max(acc[r]["mx"])),
                median_snr=float(np.median(s)) if s.size else 0.0,
                frac_snr_ok=float((s > snr_min).mean()) if s.size else 0.0,
                expected_false_positives=expected_false_blobs(
                    float(thr), sig_per_ring[r], n_band_px[r], int(p.nFrames),
                    min_n_px=p.minNrPx,
                ),
                n_resolved=float(np.mean(acc[r]["res"])) if acc[r]["res"] else 0.0,
                frac_merged=(float(np.mean(np.asarray(acc[r]["pk"]) >= 2))
                             if acc[r]["pk"] else 0.0),
                p99_peaks=(float(np.percentile(acc[r]["pk"], 99))
                           if acc[r]["pk"] else 0.0),
                max_peaks=float(np.max(acc[r]["pk"])) if acc[r]["pk"] else 0.0,
            ))

    p.Thresholds = orig

    for rec in recs:
        rec.thresh_snr = _pick_snr(rec.sweep, snr_clean_frac)
        rec.thresh_fp = _pick_fp(rec.sweep, max_false_positives)
        rec.thresh_merge = _pick_merge(rec.sweep, p99_peaks_max)
        rec.thresh_best_resolved = _pick_best_resolved(rec.sweep)

        counts = [pt.n_kept for pt in rec.sweep]
        if len(set(counts)) == 1 and counts[0] > 0:
            rec.warnings.append(
                "blob count is INVARIANT to threshold -- this is the signature "
                "of a missing/all-zero dark (see handbook 3d). Fix the dark "
                "before trusting any threshold."
            )
        if all(c == 0 for c in counts):
            rec.warnings.append(
                "no blobs survive at ANY swept threshold. Either the band is "
                "empty (wrong ring radii / Width) or minNrPx is shaving every "
                "blob to a single pixel (minNrPx is STRICT: nPx must EXCEED it)."
            )
        if rec.bg_spread > 5.0 * max(rec.noise_sigma, 1e-9):
            rec.warnings.append(
                f"background varies by {rec.bg_spread:.0f} counts around this "
                f"ring versus a noise sigma of {rec.noise_sigma:.1f} "
                f"({rec.bg_spread / max(rec.noise_sigma, 1e-9):.0f}x). NO single "
                f"absolute threshold can be clean over the whole band -- this "
                f"recommendation is a compromise. Set BgSubtract 1 to remove "
                f"the local background first and make the threshold meaningful."
            )
    return recs


__all__ = [
    "DEFAULT_MAX_FALSE_POSITIVES",
    "DEFAULT_SNR_CLEAN_FRAC",
    "DEFAULT_SNR_MIN",
    "DEFAULT_SWEEP",
    "RingRecommendation",
    "RingSweepPoint",
    "blob_snr",
    "expected_false_blobs",
    "analyze",
    "format_recommendations",
]


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Console entry point ``midas-ring-thresh``."""
    import argparse

    ap = _midas_make_parser(
        prog="midas-ring-thresh",
        description="Recommend RingThresh per ring from the data, using the "
                    "production peak-search path. Reports two independent "
                    "criteria (blob SNR and expected false positives) side by "
                    "side; they should agree.",
    )
    ap.add_argument("data_file", help="MIDAS .zip (zarr) file")
    ap.add_argument("--result-folder", default=None,
                    help="folder holding hkls.csv (default: from the zarr)")
    ap.add_argument("--n-frames", type=int, default=12,
                    help="frames sampled evenly across omega (default 12)")
    ap.add_argument("--sweep", default=None,
                    help="comma-separated thresholds to try "
                         f"(default {','.join(str(s) for s in DEFAULT_SWEEP)})")
    ap.add_argument("--snr-min", type=float, default=DEFAULT_SNR_MIN,
                    help=f"SNR a blob must clear to count as real "
                         f"(default {DEFAULT_SNR_MIN})")
    ap.add_argument("--snr-clean-frac", type=float, default=DEFAULT_SNR_CLEAN_FRAC,
                    help="fraction of surviving blobs that must clear --snr-min "
                         f"(default {DEFAULT_SNR_CLEAN_FRAC})")
    ap.add_argument("--max-false-positives", type=float,
                    default=DEFAULT_MAX_FALSE_POSITIVES,
                    help="tolerated expected noise blobs over the WHOLE scan "
                         f"(default {DEFAULT_MAX_FALSE_POSITIVES})")
    ap.add_argument("--n-sectors", type=int, default=36,
                    help="azimuthal cells per ring for background/noise stats")
    args = ap.parse_args(argv)

    sweep = DEFAULT_SWEEP
    if args.sweep:
        sweep = tuple(float(s) for s in args.sweep.split(",") if s.strip())

    recs = analyze(
        args.data_file, result_folder=args.result_folder, n_frames=args.n_frames,
        sweep=sweep, snr_min=args.snr_min, snr_clean_frac=args.snr_clean_frac,
        max_false_positives=args.max_false_positives, n_sectors=args.n_sectors,
    )
    print(format_recommendations(recs))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
