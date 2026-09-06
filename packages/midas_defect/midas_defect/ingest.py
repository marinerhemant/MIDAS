"""Front end: raw rotation frames → mask, background, 3-D spot list.

Everything else in `midas_defect` starts from a q-space voxel cloud or a spot
list and assumes somebody else produced it. This module is that somebody.

Why it is 3-D and not per-frame
-------------------------------
`midas_peakfit.connected` labels 8-connected blobs **within one frame**, which
is right for FF-HEDM peak fitting. It is wrong here. A reflection is a single
object that sweeps through the Bragg condition over several ω frames, so
per-frame segmentation counts it once per frame and leaves no way to tell a
genuine second reflection from the same one a frame later. Labelling in
(ω, row, col) makes one reflection one object and returns ω as a fitted
coordinate rather than a label.

Why the background is polar
---------------------------
An isotropic rolling median is the wrong shape: the background is strongly
radial (beamstop falloff, powder rings), so a disk straddling a radial gradient
mis-estimates it. Taking the median within (2θ, azimuth-sector) cells removes
what is azimuthally uniform — smooth background *and* powder rings — and keeps
what is localised — Bragg spots, streaks, rods.

The number of azimuth sectors is not a free knob. One sector assumes the
background is azimuthally uniform, which it is not for a cell that absorbs
through anvils and a gasket by an azimuth-dependent amount. :func:`choose_sectors`
picks it by a control that can fail: the signal is positive-only, so every
coherent *negative* structure the subtraction leaves behind is an artifact of
the background model. Fewer negatives is a better model.

Counting what was discarded
---------------------------
Every function here that selects a subset also reports how many it rejected.
This is not decoration. In the analysis this module was extracted from, "keep
one, discard the rest" silently drove a result four separate times — the worst
case discarded 43 of 45 reflections and then reported a rate over the surviving
2 as if it were a rate over 45.

References
----------
Extracted and generalised from the La₃Ni₂O₇ DAC analysis (HPCAT Sector 16,
Aug 2026); see that project's ``PORT_TO_MIDAS.md``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import ndimage

__all__ = [
    "MaskResult", "BackgroundChoice", "RingSet",
    "build_mask", "polar_median_background", "subtract_background",
    "count_signed_blobs", "choose_sectors",
    "find_blobs_3d", "SPOT_COLUMNS",
    "detect_powder_rings", "flag_powder",
]

#: 26-connectivity in (ω, row, col).
_CONN3 = np.ones((3, 3, 3), bool)

#: Columns of the spot table returned by :func:`find_blobs_3d`.
SPOT_COLUMNS = (
    "blob_id", "sub_id", "frame", "row", "col",
    "integrated", "volume_vox", "n_frames", "peak_counts",
    "length_px", "width_px", "pos_angle_deg", "aspect",
)


# ---------------------------------------------------------------------------
# 1. mask
# ---------------------------------------------------------------------------

@dataclass
class MaskResult:
    """A detector mask plus an account of where each masked pixel came from."""
    mask: np.ndarray                 # (n_rows, n_cols) bool, True = masked
    negative: np.ndarray             # bool, vendor gap/defect convention
    low_count: np.ndarray            # bool, persistently low = dead or shadowed
    median_frame: np.ndarray         # per-pixel median over the frames
    grow: int
    low_count_threshold: float

    @property
    def counts(self) -> Dict[str, int]:
        return {
            "negative": int(self.negative.sum()),
            "low_count": int(self.low_count.sum()),
            "grown": int(self.mask.sum() - (self.negative | self.low_count).sum()),
            "total": int(self.mask.sum()),
            "pixels": int(self.mask.size),
        }

    def __str__(self) -> str:
        c = self.counts
        return (f"mask: {c['total']} px ({100*c['total']/c['pixels']:.2f} %) = "
                f"{c['negative']} gap/defect + {c['low_count']} low-count "
                f"(< {self.low_count_threshold:g}) + {c['grown']} from grow="
                f"{self.grow}")


def build_mask(frames: np.ndarray, *,
               low_count_threshold: float = 20.0,
               grow: int = 2) -> MaskResult:
    """Build a detector mask from a rotation series.

    Two components, both explicit and separately reported:

    1. **Vendor gap/defect** — the Pilatus/Eiger convention that a negative
       pixel value marks a module gap or a known bad pixel.
    2. **Persistently low count** — dead pixels and the beamstop-mount shadow.
       Applied to the per-pixel *median over frames*, never to one frame: a
       pixel low once is counting statistics, a pixel low always is bad.

    ``grow`` dilates the finished mask. Pixels bordering a module gap have
    anomalous response and, after background subtraction, produce coherent
    *negative* structures. Growing removes them at negligible cost to signal;
    verify with :func:`count_signed_blobs` rather than assuming.

    ``low_count_threshold`` should sit in a plateau — sweep it and check the
    added-pixel count is flat, which separates "shadowed or dead" (near zero)
    from "normal". A tuned value that is not on a plateau is a red flag.

    Parameters
    ----------
    frames : (n_frames, n_rows, n_cols) array
        The raw rotation series, unmodified.
    """
    frames = np.asarray(frames)
    if frames.ndim != 3:
        raise ValueError(f"frames must be (n_frames, n_rows, n_cols), got {frames.shape}")
    if frames.shape[0] < 1:
        raise ValueError("need at least one frame")

    negative = frames[0] < 0
    median_frame = np.median(frames, axis=0)
    low_count = (median_frame < low_count_threshold) & ~negative
    mask = negative | low_count
    if grow:
        mask = ndimage.binary_dilation(mask, iterations=int(grow))
    return MaskResult(mask=mask, negative=negative, low_count=low_count,
                      median_frame=median_frame, grow=int(grow),
                      low_count_threshold=float(low_count_threshold))


# ---------------------------------------------------------------------------
# 2. background
# ---------------------------------------------------------------------------

def polar_median_background(frame: np.ndarray,
                            tth_deg: np.ndarray,
                            azimuth_deg: np.ndarray,
                            mask: np.ndarray, *,
                            n_sectors: int = 8,
                            tth_bin: float = 0.02,
                            min_pixels_per_cell: int = 8,
                            smooth_bins: int = 5) -> np.ndarray:
    """Median background in (2θ, azimuth-sector) cells.

    ``n_sectors = 1`` reduces to a full-azimuth median, i.e. the assumption
    that the background is azimuthally uniform. Test that assumption before
    relying on it; see :func:`choose_sectors`.

    Masked pixels contribute nothing to the estimate but still receive a
    background value, so the returned array is the same shape as ``frame``.

    .. warning::

       The model is smoothed over ``smooth_bins`` bins in 2θ, so **it cannot
       follow a ring narrower than that window** and such a ring survives
       subtraction largely intact. Keep ``smooth_bins * tth_bin`` below the
       narrowest ring you need removed, or a sharp calibrant line will be
       carried through into the spot list as a train of false reflections.
       :func:`detect_powder_rings` measures the ring widths present, so the
       window can be checked against the data rather than guessed.
    """
    frame = np.asarray(frame, dtype=np.float64)
    for name, arr in (("tth_deg", tth_deg), ("azimuth_deg", azimuth_deg),
                      ("mask", mask)):
        if np.shape(arr) != frame.shape:
            raise ValueError(f"{name} shape {np.shape(arr)} != frame {frame.shape}")
    if n_sectors < 1:
        raise ValueError("n_sectors must be >= 1")

    lo, hi = float(np.nanmin(tth_deg)), float(np.nanmax(tth_deg))
    n_tth = int((hi - lo) / tth_bin) + 1
    ti = np.clip(((tth_deg - lo) / tth_bin).astype(np.int32), 0, n_tth - 1)
    ai = np.clip(((azimuth_deg + 180.0) / (360.0 / n_sectors)).astype(np.int32),
                 0, n_sectors - 1)

    good = ~np.asarray(mask, bool)
    cell = (ti * n_sectors + ai)[good]
    value = frame[good]
    order = np.argsort(cell, kind="stable")
    cell, value = cell[order], value[order]

    n_cells = n_tth * n_sectors
    edges = np.searchsorted(cell, np.arange(n_cells + 1))
    med = np.zeros(n_cells)
    have = np.zeros(n_cells, bool)
    for i in range(n_cells):
        chunk = value[edges[i]:edges[i + 1]]
        if chunk.size >= min_pixels_per_cell:
            med[i] = np.median(chunk)
            have[i] = True
    med = med.reshape(n_tth, n_sectors)
    have = have.reshape(n_tth, n_sectors)

    # Fill along 2θ within each sector, then smooth in 2θ only. Smoothing
    # across sectors would undo the azimuthal dependence we are modelling.
    for j in range(n_sectors):
        nz = np.flatnonzero(have[:, j])
        if nz.size:
            med[:, j] = np.interp(np.arange(n_tth), nz, med[nz, j])
    if smooth_bins > 1:
        med = ndimage.uniform_filter1d(med, int(smooth_bins), axis=0)
    return med[ti, ai]


def subtract_background(frames: np.ndarray,
                        tth_deg: np.ndarray,
                        azimuth_deg: np.ndarray,
                        mask: np.ndarray, *,
                        n_sectors: int = 8,
                        tth_bin: float = 0.02) -> np.ndarray:
    """Polar-median-subtract every frame; masked pixels come back as zero."""
    frames = np.asarray(frames)
    mask = np.asarray(mask, bool)
    out = np.empty(frames.shape, dtype=np.float32)
    for k in range(frames.shape[0]):
        a = frames[k].astype(np.float64, copy=True)
        # Masked pixels must not drag the median of their own cell.
        a[mask] = np.median(a[~mask]) if (~mask).any() else 0.0
        s = a - polar_median_background(a, tth_deg, azimuth_deg, mask,
                                        n_sectors=n_sectors, tth_bin=tth_bin)
        s[mask] = 0.0
        out[k] = s.astype(np.float32)
    return out


def count_signed_blobs(stack: np.ndarray, mask: np.ndarray, *,
                       threshold: float = 200.0,
                       min_vol: int = 10,
                       min_frames: int = 2) -> Tuple[int, int]:
    """Count coherent positive and negative structures in a subtracted stack.

    Diffraction is positive-only, so any coherent **negative** structure is an
    artifact of the background model. This is the control that lets a
    background be chosen rather than asserted — and it can fail.

    Returns ``(n_positive, n_negative)``.
    """
    stack = np.asarray(stack)
    real = ~np.broadcast_to(np.asarray(mask, bool), stack.shape)
    out: List[int] = []
    for sign in (1, -1):
        lab, n = ndimage.label(sign * stack > threshold, structure=_CONN3)
        if n == 0:
            out.append(0)
            continue
        vol = np.bincount(lab[real].ravel(), minlength=n + 1)
        ids = np.where(vol >= min_vol)[0]
        ids = ids[ids > 0]
        if not ids.size:
            out.append(0)
            continue
        present = np.zeros((n + 1, stack.shape[0]), bool)
        for k in range(stack.shape[0]):
            u = np.unique(lab[k])
            present[u[u > 0], k] = True
        out.append(int((present[ids].sum(1) >= min_frames).sum()))
    return out[0], out[1]


@dataclass
class BackgroundChoice:
    """Which azimuth-sector count won, and the control that chose it."""
    n_sectors: int
    stack: np.ndarray
    table: pd.DataFrame          # one row per candidate: n_sectors, pos, neg, ratio

    def __str__(self) -> str:
        return (f"background: {self.n_sectors} azimuth sectors "
                f"(negative/positive = "
                f"{self.table.set_index('n_sectors').loc[self.n_sectors, 'neg_over_pos']:.3f})")


def choose_sectors(frames: np.ndarray,
                   tth_deg: np.ndarray,
                   azimuth_deg: np.ndarray,
                   mask: np.ndarray, *,
                   candidates: Sequence[int] = (1, 8, 24, 48, 96),
                   threshold: float = 200.0,
                   min_vol: int = 10,
                   tth_bin: float = 0.02) -> BackgroundChoice:
    """Pick the azimuth-sector count by the negative-structure control.

    Returns the winning subtracted stack together with the full comparison
    table, so the choice is auditable rather than asserted.
    """
    rows = []
    best: Optional[Tuple[float, int, np.ndarray]] = None
    for n in candidates:
        stack = subtract_background(frames, tth_deg, azimuth_deg, mask,
                                    n_sectors=int(n), tth_bin=tth_bin)
        pos, neg = count_signed_blobs(stack, mask, threshold=threshold,
                                      min_vol=min_vol)
        ratio = neg / max(pos, 1)
        rows.append({"n_sectors": int(n), "positive": pos, "negative": neg,
                     "neg_over_pos": ratio})
        if best is None or ratio < best[0]:
            best = (ratio, int(n), stack)
    assert best is not None
    return BackgroundChoice(n_sectors=best[1], stack=best[2],
                            table=pd.DataFrame(rows))


# ---------------------------------------------------------------------------
# 3. 3-D spot finding
# ---------------------------------------------------------------------------

def _bridge_gaps(binary: np.ndarray, mask: np.ndarray, size: int) -> np.ndarray:
    """Reconnect a feature cut by a detector gap, frame by frame.

    Connectivity is added **only at masked pixels**, and only where a binary
    closing shows the feature continues on both sides. Nothing is invented in
    blank regions: a closing that spans a gap requires signal on either side.
    """
    if size <= 1:
        return binary
    st = np.ones((size, size), bool)
    out = binary.copy()
    for k in range(binary.shape[0]):
        out[k] = binary[k] | (mask & ndimage.binary_closing(binary[k], structure=st))
    return out


def _split_blob(sub_img: np.ndarray, sub_mask: np.ndarray,
                ratio: float, floor: float) -> np.ndarray:
    """Watershed one blob into sub-peaks, seeded by SCALE-FREE prominence.

    ``ratio`` is how far a maximum must rise above its saddle as a *multiple*
    of the saddle, not as a count. Implemented as h-maxima on log intensity,
    where a constant h is a constant ratio, so a 10³-count peak and a 10⁶-count
    peak are judged on equal terms.

    Two scale-dependent criteria were tried first and both failed, for the same
    reason — the threshold was tied to something other than the peak itself.
    Absolute prominence shatters bright streaks; prominence relative to the
    *blob* maximum annihilates weak-but-real lobes in a blob that spans orders
    of magnitude. Neither failure is visible without hand-marked ground truth.
    """
    from skimage.morphology import h_maxima
    from skimage.segmentation import watershed

    if ratio <= 1:
        seeds = (sub_img == ndimage.maximum_filter(sub_img, size=(3, 5, 5))) & sub_mask
    else:
        lg = np.log10(np.clip(sub_img, floor, None))
        seeds = (h_maxima(lg, np.log10(float(ratio))) > 0) & sub_mask
    if not seeds.any():
        return sub_mask.astype(np.int32)
    seed_lab, n_seed = ndimage.label(seeds, structure=_CONN3)
    if n_seed <= 1:
        return sub_mask.astype(np.int32)
    return watershed(-sub_img, markers=seed_lab, mask=sub_mask)


def find_blobs_3d(stack: np.ndarray, mask: np.ndarray, *,
                  threshold: float = 200.0,
                  min_vol: int = 10,
                  split_ratio: float = 3.0,
                  gap_bridge: int = 21,
                  core_frac: float = 0.5,
                  return_counts: bool = False):
    """Find reflections as 3-D objects in (ω, row, col).

    Steps: threshold → bridge detector gaps → label with 26-connectivity →
    drop blobs below ``min_vol`` *real* (unmasked) voxels → watershed-split
    each survivor by scale-free prominence → report per sub-peak.

    Position comes from the high-intensity **core** only (voxels above
    ``core_frac`` × the sub-region maximum). An intensity-weighted centroid
    over the whole sub-region is dragged down a streak's faint tail; the core
    restriction is flat over ``core_frac`` = 0.3–0.7, so it is not tuned.
    Intensity and shape still use the full region.

    In-plane shape is reported as second **moments**, not a Gaussian fit: these
    features are elongated streaks, so a Gaussian is a misspecified model that
    reports a meaningless width and drags the centroid.

    Returns
    -------
    pandas.DataFrame with :data:`SPOT_COLUMNS`. ``row``/``col`` index the array
    exactly as supplied — this function applies no flip and knows of none.
    With ``return_counts=True``, returns ``(df, counts)`` where ``counts``
    records how many blobs each step rejected.
    """
    stack = np.asarray(stack, dtype=np.float32)
    mask = np.asarray(mask, bool)
    if stack.ndim != 3:
        raise ValueError(f"stack must be 3-D (ω, row, col), got {stack.shape}")
    if mask.shape != stack.shape[1:]:
        raise ValueError(f"mask {mask.shape} does not match frames {stack.shape[1:]}")
    if not 0.0 < core_frac < 1.0:
        raise ValueError("core_frac must be in (0, 1)")

    stack = stack.copy()
    stack[:, mask] = 0.0

    binary = _bridge_gaps(stack > threshold, mask, int(gap_bridge))
    lab, n_lab = ndimage.label(binary, structure=_CONN3)
    real = ~np.broadcast_to(mask, stack.shape)
    vol = np.bincount(lab[real].ravel(), minlength=n_lab + 1)
    keep = np.zeros(n_lab + 1, bool)
    keep[1:] = vol[1:] >= min_vol

    counts = {"labelled": int(n_lab),
              "rejected_small": int(n_lab - keep.sum()),
              "kept": int(keep.sum()),
              "blobs_split": 0}

    objects = ndimage.find_objects(lab)
    rows: List[tuple] = []
    for L in np.flatnonzero(keep):
        sl = objects[L - 1]
        m = (lab[sl] == L)
        img = stack[sl] * m
        sub = _split_blob(img, m, split_ratio, threshold)
        ids = np.unique(sub)
        ids = ids[ids > 0]
        if ids.size > 1:
            counts["blobs_split"] += 1
        for j in ids:
            mm = (sub == j)
            w = img * mm
            total = float(w.sum())
            if total <= 0:
                continue
            idx = np.array(np.nonzero(mm))
            wt = w[mm]

            core = mm & (w > core_frac * w.max())
            if not core.any():
                core = mm
            cidx = np.array(np.nonzero(core))
            cwt = w[core]
            com = (cidx * cwt).sum(1) / cwt.sum()
            k0, r0, c0 = sl[0].start, sl[1].start, sl[2].start

            fcom = (idx * wt).sum(1) / wt.sum()
            dy = idx[1] - fcom[1]
            dx = idx[2] - fcom[2]
            W = wt.sum()
            cyy = float((wt * dy * dy).sum() / W)
            cxx = float((wt * dx * dx).sum() / W)
            cyx = float((wt * dy * dx).sum() / W)
            ev, evec = np.linalg.eigh(np.array([[cyy, cyx], [cyx, cxx]]))
            length = float(np.sqrt(max(ev[1], 0.0)))
            width = float(np.sqrt(max(ev[0], 0.0)))
            vy, vx = evec[0, 1], evec[1, 1]
            pos_angle = float(np.degrees(np.arctan2(-vy, vx)) % 180.0)
            # A single-voxel-wide blob has width 0. Report the aspect ratio as
            # infinite rather than dividing by an epsilon and emitting 3e5,
            # which reads like a measurement and silently survives a filter.
            aspect = float(length / width) if width > 0 else float("inf")

            rows.append((int(L), int(j),
                         float(com[0] + k0), float(com[1] + r0), float(com[2] + c0),
                         total, int(mm.sum()), int(np.unique(idx[0]).size),
                         float(w.max()), length, width, pos_angle, aspect))

    df = pd.DataFrame(rows, columns=list(SPOT_COLUMNS))
    counts["sub_peaks"] = len(df)
    return (df, counts) if return_counts else df


# ---------------------------------------------------------------------------
# 4. powder separation — no ring table
# ---------------------------------------------------------------------------

@dataclass
class RingSet:
    """Powder rings detected from the data itself."""
    centre_deg: np.ndarray
    width_deg: np.ndarray
    profile_tth_deg: np.ndarray = field(repr=False, default=None)
    profile: np.ndarray = field(repr=False, default=None)
    occupancy: np.ndarray = field(repr=False, default=None)
    """Fraction of azimuthal sectors each ring actually occupies.

    NaN when :func:`detect_powder_rings` was called without an azimuth map, in
    which case the ring-continuity test did not run.
    """

    def __len__(self) -> int:
        return len(self.centre_deg)


def detect_powder_rings(image: np.ndarray,
                        tth_deg: np.ndarray,
                        mask: np.ndarray, *,
                        azimuth_deg: Optional[np.ndarray] = None,
                        tth_bin: float = 0.01,
                        tth_range: Tuple[float, float] = (2.0, 25.0),
                        prominence_sigma: float = 3.0,
                        width_floor_deg: float = 0.03,
                        min_occupancy: float = 0.5,
                        n_sectors: int = 36,
                        min_pixels_per_sector: int = 12) -> RingSet:
    r"""Find powder rings in the azimuthal-median radial profile.

    The median over azimuth is the first half of the trick: a powder ring
    occupies every azimuth so it survives, a Bragg spot occupies a few so it
    does not. Rings come from the data, so this transfers to a gasket,
    substrate or capillary that has never been seen before — no ring table, no
    phase list.

    The median alone is not enough. A strongly diffracting single crystal puts
    several reflections of one multiplicity family at exactly one \|G\|; on a
    max-projection over a rotation those, plus their streaks, lift the profile
    enough to be found as a peak. Measured on La3Ni2O7 in a DAC, that admitted
    114 "rings" over 23° of 2θ whose median azimuthal occupancy was 0.19 —
    blanketing 44% of the range and destroying the ring half of
    :func:`flag_powder`'s two-sided test, which then discarded the crystal's
    own brightest reflections.

    So a candidate must also be *continuous*: pass ``azimuth_deg`` and each
    peak is kept only if it is present in at least ``min_occupancy`` of the
    azimuthal sectors that have enough unmasked pixels to judge. Sectors lost
    to module gaps are excluded from the denominator rather than counted
    absent.

    Omitting ``azimuth_deg`` skips the continuity test and restores the older,
    permissive behaviour; ``occupancy`` is then NaN. Prefer passing it.
    """
    from scipy import signal

    image = np.asarray(image, dtype=np.float64)
    good = ~np.asarray(mask, bool)
    lo, hi = tth_range
    sel = good & (tth_deg >= lo) & (tth_deg < hi)
    n_bin = int((hi - lo) / tth_bin) + 1
    idx = ((tth_deg[sel] - lo) / tth_bin).astype(np.int32)
    val = image[sel]
    order = np.argsort(idx, kind="stable")
    idx, val = idx[order], val[order]
    edges = np.searchsorted(idx, np.arange(n_bin + 1))
    prof = np.full(n_bin, np.nan)
    for i in range(n_bin):
        chunk = val[edges[i]:edges[i + 1]]
        if chunk.size >= 8:
            prof[i] = np.median(chunk)
    finite = np.isfinite(prof)
    if finite.sum() < 10:
        return RingSet(np.array([]), np.array([]))
    prof = np.interp(np.arange(n_bin), np.flatnonzero(finite), prof[finite])

    baseline = ndimage.median_filter(prof, size=max(3, int(0.5 / tth_bin)))
    excess = prof - baseline
    noise = 1.4826 * np.median(np.abs(excess - np.median(excess)))
    peaks, props = signal.find_peaks(
        excess, prominence=max(prominence_sigma * noise, 1e-9))
    axis = lo + tth_bin * np.arange(n_bin)
    if not peaks.size:
        return RingSet(np.array([]), np.array([]), axis, prof, np.array([]))
    widths = signal.peak_widths(excess, peaks, rel_height=0.5)[0] * tth_bin
    widths = np.maximum(widths, width_floor_deg)
    centres = lo + tth_bin * peaks

    if azimuth_deg is None:
        occ = np.full(centres.shape, np.nan)
        return RingSet(centres, widths, axis, prof, occ)

    # Ring continuity. A powder ring is present at nearly every azimuth; a
    # single-crystal multiplicity family is present at a handful.
    azimuth_deg = np.asarray(azimuth_deg, dtype=np.float64)
    if azimuth_deg.shape != tth_deg.shape:
        raise ValueError("azimuth_deg must have the same shape as tth_deg")
    sector_all = ((azimuth_deg + 180.0) / 360.0 * n_sectors).astype(np.int64)
    np.mod(sector_all, n_sectors, out=sector_all)
    occ = np.empty(centres.shape, dtype=np.float64)
    for i, (c, w) in enumerate(zip(centres, widths)):
        band = good & (np.abs(tth_deg - c) <= w)
        if not band.any():
            occ[i] = 0.0
            continue
        vals = image[band]
        secs = sector_all[band]
        base = float(np.interp(c, axis, baseline))
        thresh = base + 2.0 * noise
        counts = np.bincount(secs, minlength=n_sectors)
        judged = counts >= min_pixels_per_sector
        if not judged.any():
            occ[i] = 0.0
            continue
        order = np.argsort(secs, kind="stable")
        secs_s, vals_s = secs[order], vals[order]
        bnd = np.searchsorted(secs_s, np.arange(n_sectors + 1))
        present = np.zeros(n_sectors, dtype=bool)
        for j in np.flatnonzero(judged):
            present[j] = np.median(vals_s[bnd[j]:bnd[j + 1]]) > thresh
        occ[i] = float(present[judged].sum()) / float(judged.sum())

    keep = occ >= min_occupancy
    return RingSet(centres[keep], widths[keep], axis, prof, occ[keep])


def flag_powder(spot_tth_deg: np.ndarray,
                spot_azimuth_deg: np.ndarray,
                spot_radius_px: np.ndarray,
                rings: RingSet, *,
                min_companions: int = 8,
                azimuth_separation_deg: float = 8.0,
                radius_tolerance_px: float = 4.0) -> np.ndarray:
    """Flag spots that are powder, using a **two-sided** discriminant.

    Neither half works alone, which is why both are required:

    - *"many azimuthal companions"* alone flags a large fraction of a perfectly
      good single-crystal index — a crystal with complete (hkL) rows genuinely
      puts several reflections at one |G|.
    - *"sits at a detected ring 2θ"* alone flags every real reflection that
      happens to coincide with a gasket or anvil line, and there are many.

    A spot is powder only if it lies inside a detected ring **and** has at
    least ``min_companions`` neighbours at its own radius and a different
    azimuth.

    Returns a boolean array, True = powder.
    """
    tth = np.asarray(spot_tth_deg, float)
    eta = np.asarray(spot_azimuth_deg, float)
    rad = np.asarray(spot_radius_px, float)
    if not (tth.shape == eta.shape == rad.shape):
        raise ValueError("spot arrays must have the same shape")

    on_ring = np.zeros(tth.shape, bool)
    for c, w in zip(rings.centre_deg, rings.width_deg):
        on_ring |= np.abs(tth - c) <= w
    if not on_ring.any():
        return on_ring

    n_comp = np.zeros(tth.shape, int)
    for i in np.flatnonzero(on_ring):
        d_eta = np.abs(((eta - eta[i] + 180.0) % 360.0) - 180.0)
        near = (np.abs(rad - rad[i]) <= radius_tolerance_px) & \
               (d_eta >= azimuth_separation_deg)
        n_comp[i] = int(near.sum())
    return on_ring & (n_comp >= min_companions)


def live_frames(frames: "np.ndarray", *, frac_of_median: float = 0.002):
    """Which frames of a rotation stack contain signal. Returns (mask, maxima).

    Threshold the frame MAXIMUM, not the sum. A dead frame still sums to ~3e5
    because hundreds of thousands of pixels sit at low values, so a sum
    threshold never fires -- and did not: a full 621-position raster ran with
    two dead frames included before this was caught. The maxima separate by
    three orders of magnitude. Measured on La3Ni2O7 2604: live frames peak at
    5031-687086 counts while frames 0 and 39 peak at 44 and 58.

    The cut is a FRACTION OF THE MEDIAN maximum, not an absolute count, so it
    transfers between samples and exposure times.

    **The default is 0.002, not 0.05.** A first attempt at 0.05 discarded REAL
    frames: at 2604 p=329 it dropped four (0, 1, 2, 39) where only 0 and 39 are
    dead, because frames 1 and 2 peak at 5031 and 30660 counts -- 0.8 % and
    4.8 % of the 639779 median, i.e. below a 5 % cut but carrying obvious
    signal. Raster-wide that threshold threw away 1724 frames, 247 of them
    peaking above 1000 counts. The genuinely dead frames sit at 0.007-0.009 %
    of the median, two orders of magnitude below the faintest real one, so the
    cut belongs between those scales and nowhere near 5 %.
    """
    mx = frames.reshape(len(frames), -1).max(axis=1)
    return mx > frac_of_median * np.median(mx), mx
