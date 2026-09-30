"""Combine several :class:`~midas_dfxm.rocking.RockingScan` planes (one per fixed value of a
third angle, e.g. ESRF ID03's `obpitch`) into per-pixel orientation and strain maps.

A different reduction from :mod:`midas_dfxm.rocking`. `reduce_rocking` fits one rocking curve
per pixel from a peak window on a *single* plane; this module reproduces the campaign's own
joint-moment method (`reduce.py`/`maps.py` on the Mg-4Al ID03 dataset this was built against):
accumulate each plane's frames into per-pixel marginal intensity distributions over each moving
angle, remove each marginal's own angular baseline, take the first moment, and turn the third
angle's centre-of-mass into a relative Delta d/d strain. Reuse `reduce_rocking` for a single
scan; reach for this when you have several planes of the same mu/chi mesh and want the extra
axis.

Two things this got wrong on the first pass, both from applying a full-frame recipe to a
pre-cropped region -- read before changing either default:

* **The grain mask's background percentile.** `maps.py`'s mask threshold is a percentile of the
  *whole* 2048 x 2048 frame, where background is the majority. Crop first to the grain's own
  bounding box (as this module's caller typically does) and background becomes the *minority* --
  the same percentile then sits near the grain's own median intensity and carves a moth-eaten,
  disconnected mask out of real sub-grain contrast. `derive_grain_mask`'s default
  `background_percentile=5.0` assumes a mostly-foreground crop; raise it (`~50`) for an
  uncropped, mostly-background frame.
* **The per-frame pedestal.** Skipping it before accumulating a moment let the *angular*
  baseline-removal step in `angular_moments` absorb the whole per-frame pedestal as well --
  91 % "baseline" instead of the correct ~30 %. The downstream tilt/strain maps barely moved (a
  uniform per-frame offset does not shift a centroid much), so this would have shipped a
  misleading intermediate diagnostic without visibly breaking the headline numbers.
  `accumulate_marginals` always subtracts it (checked: stable, ~2-20 ADU spread across
  percentiles 1-10, even on an already-mostly-grain crop, because any *single* frame is mostly
  unlit -- DFXM only lights up pixels currently satisfying Bragg).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from .rocking import RockingScan

__all__ = [
    "accumulate_marginals",
    "angular_moments",
    "derive_grain_mask",
    "StrainTiltMaps",
    "strain_and_tilt",
    "predicted_fov_row_gradient",
    "HeldOutPlaneResult",
    "held_out_plane_check",
]


def accumulate_marginals(scans: Sequence[RockingScan], *, pedestal_percentile: float = 5.0,
                         mu_key: str = "mu", chi_key: str = "chi", round_decimals: int = 3):
    """Bin every plane's frames into mu, chi and per-plane marginal intensity distributions.

    Parameters
    ----------
    scans : one ``RockingScan`` per plane, all ``tilt2d`` over the same ``(mu_key, chi_key)``
        pair and the same ``(H, W)`` frame shape -- e.g. one call to
        :func:`midas_dfxm.io_id03.load_id03_scan` per `obpitch` plane.
    pedestal_percentile : per-frame percentile subtracted (and clipped at 0) before
        accumulating, matching the campaign's own `reduce.py`. Not sensitive to how much of the
        crop is grain (see module docstring) -- leave at the default unless you have measured a
        different sensor floor.
    mu_key, chi_key : the two ``RockingScan.motors`` columns that define the mesh.
    round_decimals : motor values are rounded to this many decimals before grouping into grid
        bins, to absorb encoder jitter within one nominal step.

    Returns
    -------
    mu_g, chi_g : (n_mu,), (n_chi,) sorted unique grid values (from the first scan; every scan
        must share the same grid).
    Mmu, Mchi : (n_mu, H, W), (n_chi, H, W) marginals, summed over all planes.
    Mob : (n_planes, H, W), one plane's pedestal-subtracted total intensity per entry, in the
        order ``scans`` was given.
    """
    if not scans:
        raise ValueError("scans must be non-empty")
    mu_g = chi_g = None
    Mmu = Mchi = Mob = None
    for n, scan in enumerate(scans):
        if scan.scan_type != "tilt2d":
            raise ValueError(f"scans[{n}] is {scan.scan_type!r}, not tilt2d; accumulate_marginals "
                             "needs a mu x chi mesh per plane")
        frames = scan.frames
        ped = np.percentile(frames.reshape(frames.shape[0], -1), pedestal_percentile, axis=1)
        frames = np.maximum(frames - ped[:, None, None], 0)
        mu_round = np.round(scan.motors[mu_key], round_decimals)
        chi_round = np.round(scan.motors[chi_key], round_decimals)
        if mu_g is None:
            mu_g = np.sort(np.unique(mu_round)); chi_g = np.sort(np.unique(chi_round))
            H, W = frames.shape[1:]
            Mmu = np.zeros((len(mu_g), H, W)); Mchi = np.zeros((len(chi_g), H, W))
            Mob = np.zeros((len(scans), H, W))
        elif frames.shape[1:] != Mmu.shape[1:]:
            raise ValueError(f"scans[{n}] frame shape {frames.shape[1:]} does not match "
                             f"scans[0]'s {Mmu.shape[1:]}")
        for i, g in enumerate(mu_g):
            Mmu[i] += frames[mu_round == g].sum(0)
        for i, g in enumerate(chi_g):
            Mchi[i] += frames[chi_round == g].sum(0)
        Mob[n] = frames.sum(0)
    return mu_g, chi_g, Mmu, Mchi, Mob


def angular_moments(M: np.ndarray, grid: np.ndarray, *, base_frac: float = 0.30) -> dict:
    """Per-pixel centre-of-mass and width of a marginal, with and without its angular baseline.

    The angular baseline is the mean of the lowest ``base_frac`` of bins, per pixel -- a
    per-pixel floor spanning the whole grid (this campaign's version of a rocking scan's frame
    pedestal, at one level up: the *angular profile* also sits on an offset). Removed the same
    way :func:`midas_dfxm.rocking.reduce_rocking` removes its baseline: measured, not assumed.

    Parameters
    ----------
    M : (n_grid, H, W) marginal intensity, e.g. one of :func:`accumulate_marginals`'s outputs.
    grid : (n_grid,) the angle (or, for the third/obpitch axis, its own coordinate) each slice
        of ``M`` was accumulated at.
    base_frac : fraction of grid points treated as baseline. 0.25-0.30 matches the campaign this
        was built against; the right value depends on how much of the grid the real peak
        occupies (smaller peak -> smaller base_frac).

    Returns
    -------
    dict with ``S0``/``com_raw`` (raw first moment and its intensity) and ``S0c``/``com``/``sig``
    (baseline-removed), plus ``frac_base``, the fraction of intensity the baseline carried
    (report and read before trusting ``com``: 90 %+ means the baseline swallowed real signal,
    e.g. because a pedestal was left in before this function ever saw the data), and ``base``,
    the per-pixel baseline level itself (same units as ``M``; used by
    :func:`held_out_plane_check` to rebuild a predicted total from ``com``/``sig`` alone).
    """
    n = len(grid); g = np.asarray(grid, dtype=np.float32)
    S0 = M.sum(0)
    com0 = (M * g[:, None, None]).sum(0) / np.maximum(S0, 1e-9)
    var0 = (M * (g[:, None, None] - com0) ** 2).sum(0) / np.maximum(S0, 1e-9)
    k = max(1, int(round(base_frac * n)))
    base = np.sort(M, 0)[:k].mean(0)
    Mc = np.maximum(M - base, 0)
    S0c = Mc.sum(0)
    comc = (Mc * g[:, None, None]).sum(0) / np.maximum(S0c, 1e-9)
    varc = (Mc * (g[:, None, None] - comc) ** 2).sum(0) / np.maximum(S0c, 1e-9)
    return dict(S0=S0, com_raw=com0, S0c=S0c, com=comc, sig=np.sqrt(np.maximum(varc, 0)),
               frac_base=1 - S0c / np.maximum(S0, 1e-9), base=base)


def derive_grain_mask(total_intensity: np.ndarray, *, background_percentile: float = 5.0,
                      threshold_frac: float = 0.10, smooth_sigma: float = 15.0) -> np.ndarray:
    """The largest connected bright region of ``total_intensity``, holes filled.

    **Read the module docstring first.** ``background_percentile`` must match the true
    background fraction of the image, not a generic "half is background" guess: the default
    (5) is for a crop already mostly filled by the grain (its own bounding box, say). Raise it
    towards 50 for a frame where background genuinely is the majority.

    Parameters
    ----------
    total_intensity : (H, W), e.g. one axis's ``angular_moments(...)["S0"]``.
    background_percentile : percentile of the smoothed image used as the background level.
    threshold_frac : threshold = background + this fraction of (p99.9 - background).
    smooth_sigma : box size (pixels) for the uniform smoothing before thresholding.

    Returns
    -------
    (H, W) bool mask.
    """
    from scipy import ndimage
    sm = ndimage.uniform_filter(total_intensity, int(round(smooth_sigma)))
    bg = np.percentile(sm, background_percentile)
    thr = bg + threshold_frac * (np.percentile(sm, 99.9) - bg)
    m = sm > thr
    lab, n = ndimage.label(m)
    if n == 0:
        raise ValueError("no pixel exceeds the threshold; check background_percentile/"
                         "threshold_frac against this image's own intensity range")
    sizes = ndimage.sum(m, lab, range(1, n + 1))
    return ndimage.binary_fill_holes(lab == (1 + np.argmax(sizes)))


@dataclass
class StrainTiltMaps:
    """Output of :func:`strain_and_tilt`. Angles in degrees, ``strain`` dimensionless (Delta d/d)."""

    theta_B_deg: float
    d_spacing_A: float
    tilt_mu: np.ndarray
    tilt_chi: np.ndarray
    strain: np.ndarray


def strain_and_tilt(mu_m: dict, chi_m: dict, ob_m: dict, grain: np.ndarray,
                    wavelength_A: float) -> StrainTiltMaps:
    """Tilts and a *relative* Delta d/d strain map from three :func:`angular_moments` results.

    ``theta_B`` and the strain reference are the grain's own median obpitch centre-of-mass --
    this gives no absolute d-spacing (the obpitch zero is not independently known) and no
    absolute orientation, only relative maps across the grain.

    One reflection gives Delta d/d, not elastic strain: a composition change moves d exactly
    like strain does, and a strain *tensor* needs several non-coplanar reflections. See
    :mod:`midas_dfxm.multiplane`'s campaign notes (or ``RETRACTION_strain.md``, if you have it)
    before calling a Delta d/d map "elastic strain."
    """
    ob_ref = np.median(ob_m["com"][grain])
    th = np.deg2rad(ob_ref) / 2.0
    d_spacing = wavelength_A / (2 * np.sin(th))
    strain = -(np.deg2rad(ob_m["com"] - ob_ref) / 2.0) / np.tan(th)
    tilt_mu = mu_m["com"] - np.median(mu_m["com"][grain])
    tilt_chi = chi_m["com"] - np.median(chi_m["com"][grain])
    return StrainTiltMaps(theta_B_deg=float(np.rad2deg(th)), d_spacing_A=float(d_spacing),
                          tilt_mu=tilt_mu, tilt_chi=tilt_chi, strain=strain)


def predicted_fov_row_gradient(*, ffz: np.ndarray, ob_g: np.ndarray, obx: float,
                               pixel_um: float, magnification: float,
                               theta_B_deg: float) -> float:
    """Parameter-free predicted |apparent strain gradient| along detector rows, in microstrain/px.

    Every input is a number already in the campaign's own file -- no fitted parameter. Derives
    the sample->detector and objective->detector distances from how far the detector stage
    (``ffz``) moves per `obpitch` step and how far the objective sits from the sample (``obx``),
    then the angular change accepted per detector row.

    **This predicts a magnitude, not a signed correction.** The sign needs a camera-row-vs-lab-z
    readout convention that is not recoverable from the motor record alone -- do not subtract
    this (signed) from a strain map and call it corrected; fit and remove the map's own measured
    trend instead (its sign is unambiguous by construction), and treat that as cosmetic
    de-planing, not a calibrated correction, until the sign question is settled independently.

    Parameters
    ----------
    ffz : (n_planes,) the detector-stage position at each plane, mm.
    ob_g : (n_planes,) the `obpitch` value at each plane, deg.
    obx : objective-to-sample distance, mm.
    pixel_um, magnification : sensor pixel size (um) and objective magnification.
    theta_B_deg : Bragg angle, deg (e.g. :attr:`StrainTiltMaps.theta_B_deg`).

    Returns
    -------
    Predicted |d(strain)/d(row)|, microstrain per pixel.
    """
    ob_g = np.asarray(ob_g, dtype=float); ffz = np.asarray(ffz, dtype=float)
    if ob_g.shape != ffz.shape:
        raise ValueError(f"ob_g {ob_g.shape} and ffz {ffz.shape} must be the same length, in "
                         "the same plane order (ffz[i] is the plane at ob_g[i])")
    order = np.argsort(ob_g)
    ob_step_deg = np.median(np.diff(ob_g[order]))
    ffz_step_mm = np.median(np.diff(ffz[order]))
    L_sample_to_det = ffz_step_mm / np.deg2rad(ob_step_deg)
    L_obj_to_det = L_sample_to_det - obx
    if L_obj_to_det <= 0:
        raise ValueError(f"objective->detector distance came out non-positive "
                         f"({L_obj_to_det:.1f} mm) -- check ffz/ob_g are sorted the same way "
                         "and obx is in mm")
    eff_pixel_um = pixel_um / magnification
    s_rad_per_px = (eff_pixel_um * 1e-6) / (L_obj_to_det * 1e-3)
    return float(1e6 * (s_rad_per_px / 2.0) / np.tan(np.deg2rad(theta_B_deg)))


@dataclass
class HeldOutPlaneResult:
    """Output of :func:`held_out_plane_check`."""

    held_out: int              # index into the original scans/ob_g/Mob ordering
    obpitch_deg: float
    actual: np.ndarray         # (H, W) the held-out plane's own measured total intensity
    predicted: np.ndarray      # (H, W) predicted from the OTHER planes' fit alone
    r: float                   # Pearson correlation, over the grain mask
    resid_rms_frac: float      # RMS(actual - predicted) / RMS(actual - baseline), over the grain


def held_out_plane_check(Mob: np.ndarray, ob_g: np.ndarray, grain: np.ndarray, *,
                         held_out: Optional[int] = None, base_frac: float = 0.25) -> HeldOutPlaneResult:
    """Leave-one-plane-out accuracy check: fit the per-pixel single-Gaussian-in-obpitch model
    on every plane EXCEPT one, predict that held-out plane's own total-intensity map from the
    fit alone, and compare to what was actually measured there.

    This is a different kind of check than :func:`~midas_dfxm.rocking.forward_predict_check`:
    that one fits and evaluates a pixel's model against the SAME data it was fit from (a
    within-plane spot check of curve shape); this one fits on 7 of the campaign's 8 planes and
    predicts the 8th, whose own counts never enter the *obpitch-Gaussian fit* at all. It is not
    the round-trip ``manuals/dfxm/README.md`` rule 5 warns against (a forward-then-inverse of
    one's own generator returns ~1e-16 because it inverts itself): an adversarial ``/verify``
    pass on the real Mg-4Al g9 grain (claim ``2f8370bd5108``, PROVISIONAL) confirmed a trivial
    shared-illumination/grain-silhouette baseline caps out at r~0.19-0.46 here, far below this
    function's r~0.887 -- the fit is recovering real per-pixel structure, not just "this pixel
    is generally bright."

    **Two honest caveats from that same verification pass, kept here so they are not lost:**
    (1) ``grain`` is normally derived (by the caller) from a marginal accumulated over ALL
    planes including the held-out one, so it is not a strictly clean holdout -- measured on the
    real data, re-deriving the mask from only the 7 retained planes changes r by 0.007 (0.75 %
    relative), i.e. real but not material here; a genuinely leakage-free mask still needs a
    caller-supplied ``grain`` built without the held-out plane if that matters for your use.
    (2) an exact permutation null (shuffling which retained plane's obpitch label attaches to
    which plane's pixel data, refitting each time) found ``resid_rms_frac`` alone is NOT
    significant on the real grain (p=0.157) -- report ``r`` as the primary evidence of fit
    quality, not ``resid_rms_frac``, until a better residual normalisation is found.

    Parameters
    ----------
    Mob : (n_planes, H, W) one plane's pedestal-subtracted total intensity per entry (see
        :func:`accumulate_marginals`).
    ob_g : (n_planes,) the obpitch value of each plane, same order as ``Mob``.
    grain : (H, W) boolean mask restricting the reported correlation/residual to real signal --
        predicting outside the grain is meaningless (no peak there to hold out). See caveat (1)
        above on how this mask is usually derived.
    held_out : which plane index to leave out. Default: the plane whose obpitch is closest to
        the grain's own median obpitch centre (fit on all 8 planes) -- the plane nearest the
        peak. **Not necessarily the hardest test**: a leave-one-out sweep over all 8 planes on
        the real data found this default plane was the BEST of the 8 by ``resid_rms_frac``, not
        the hardest, most likely because it also has the best per-pixel signal-to-noise -- an
        earlier version of this docstring claimed the opposite without checking.
    base_frac : passed through to :func:`angular_moments`; match whatever value the notebook's
        own strain/tilt reduction used, for a fair comparison.

    Returns
    -------
    HeldOutPlaneResult
    """
    Mob = np.asarray(Mob, dtype=float)
    ob_g = np.asarray(ob_g, dtype=float)
    n = Mob.shape[0]
    if n < 3:
        raise ValueError("held_out_plane_check needs at least 3 planes (2 to fit a peak, 1 to "
                         "hold out)")
    if grain.sum() == 0:
        raise ValueError("grain mask is empty -- nothing to check")
    if held_out is None:
        full = angular_moments(Mob, ob_g, base_frac=base_frac)
        ob_ref = float(np.median(full["com"][grain]))
        held_out = int(np.argmin(np.abs(ob_g - ob_ref)))
    keep = np.array([i for i in range(n) if i != held_out])
    fit = angular_moments(Mob[keep], ob_g[keep], base_frac=base_frac)
    amp = np.max(Mob[keep] - fit["base"], axis=0)      # peak height, from the RETAINED planes only
    z = (ob_g[held_out] - fit["com"]) / np.maximum(fit["sig"], 1e-9)
    predicted = fit["base"] + amp * np.exp(-0.5 * z ** 2)
    actual = Mob[held_out]

    a = actual[grain]; p = predicted[grain]
    ac = a - a.mean(); pc = p - p.mean()
    denom = np.sqrt((ac ** 2).sum() * (pc ** 2).sum())
    r = float((ac * pc).sum() / denom) if denom > 0 else float("nan")
    signal = np.clip(a - fit["base"][grain], 0.0, None)
    sig_rms = float(np.sqrt(np.mean(signal ** 2)))
    resid = a - p
    resid_rms_frac = float(np.sqrt(np.mean(resid ** 2)) / sig_rms) if sig_rms > 0 else float("nan")
    return HeldOutPlaneResult(held_out=held_out, obpitch_deg=float(ob_g[held_out]), actual=actual,
                              predicted=predicted, r=r, resid_rms_frac=resid_rms_frac)
