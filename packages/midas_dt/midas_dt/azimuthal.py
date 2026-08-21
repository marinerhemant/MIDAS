"""Per-azimuth ring extraction from an integrated (R, eta) cake.

Two numbers come out of a ring at each azimuth, and they behave completely
differently:

* **area** -- the background-subtracted integrated intensity, which carries
  *texture*. It is a **difference of large numbers**, so it inherits the
  background model's error in full.
* **centroid** -- the intensity-weighted radius, which carries *strain* through
  the d-spacing. It is a **ratio**, so a slowly varying background largely
  cancels.

That asymmetry decides what a dataset can support. On a diamond-anvil-cell Ti
scan measured here, peaks sat only **0.5-17 % above background**: strain came out
consistent across six independent reflections, while every azimuthal texture
signal was incoherent extraction error. Same frames, same windows, same code --
the difference is arithmetic, not analysis. Read
``manuals/xrd-ct/ENVELOPE.md`` before promising a texture map from a
low-contrast scan.

Four things here were each established by a measurement that invalidated an
earlier attempt, and none of them are refinements:

1. **Background from ring-FREE radii, interpolated across the peaks** --
   :func:`background_from_ring_free`. A rolling low percentile over blocks
   comparable to the peak width sits on the peak *flank* and biases the
   background up exactly where the peak is. With an 8-px FWHM peak in a 30-px
   block that is ~27 % of the block, and at 1 % contrast a 1 % background error
   is a ~20 % area error.
2. **Vet for singlets before fitting anything** -- :func:`count_maxima`. A
   sub-pixel ring assignment happily matches a doublet as one line. Multiplets
   are reported and *excluded*, not fitted.
3. **Gate on per-azimuth SNR, not on the ring's median** --
   :func:`snr_per_eta`. Gating on the median still lets dead azimuthal bins
   through, and a peak-to-peak spread over eta is a max-minus-min, so a handful
   of them dominate it. That is how one reflection reported 107,410 ue and
   another 638,282 ue (11 % and 64 % strain, i.e. nonsense).
4. **Reject azimuthal outliers by MAD** -- :func:`mad_filter`, following Muerer
   et al. 2021 (IUCrJ 8, 747), to suppress single large crystallites that make a
   powder ring spotty.

A trap this module cannot fix, only expose: **under differential stress the peak
MOVES with azimuth**, ``d(psi) = d0 [1 + (1 - 3 cos^2 psi) Q(hkl)]`` (Singh's
lattice-strain relation -- it is how DAC differential stress is measured). A
fixed radial window then converts that movement into azimuth-dependent
*intensity*, i.e. fake texture. :func:`radial_half_correlation` is the
discriminator: **strongly negative means the peak is moving**, because intensity
leaves one radial half as it enters the other. Measured -0.72 on a CeO2 standard
that should have had no texture at all.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

__all__ = [
    "RingExtraction",
    "area_and_centroid",
    "azimuthal_rebin",
    "background_from_ring_free",
    "radial_half_correlation",
    "count_maxima",
    "extract_ring",
    "mad_filter",
    "refine_ring_centres",
    "ring_free_mask",
    "ring_windows",
    "snr_per_eta",
    "strain_from_centroid",
]

log = logging.getLogger(__name__)

#: Fixed-point iterations used when re-centring a radial window on its peak.
#: The centre of mass seen through an off-centre window is biased toward the
#: window centre, so the correction must be iterated rather than applied once.
_RECENTRE_ITERS = 8


@dataclass
class RingExtraction:
    """One ring's per-azimuth extraction, with the gates it did or did not pass."""

    area: np.ndarray           # (n_eta,) background-subtracted integrated intensity
    centroid: np.ndarray       # (n_eta,) intensity-weighted radius, in r_axis units
    snr: np.ndarray            # (n_eta,) integrated signal / Poisson noise
    n_maxima: int              # 1 == singlet; >1 == multiplet, do not fit
    contrast: float            # peak height / background at the ring centre
    half_width_bins: int
    centre_bin: int
    #: Measured peak position minus the window centre, in bins. A window centred
    #: on a CATALOGUED ring position rather than on the peak is asymmetric about
    #: the peak, so any change in peak WIDTH between measurements is converted
    #: into an apparent change in CENTROID -- i.e. into apparent strain. This is
    #: reported always so the defect cannot go unnoticed; see :func:`extract_ring`.
    centre_offset_bins: float = 0.0

    @property
    def is_singlet(self) -> bool:
        return self.n_maxima == 1

    def usable(self, min_snr: float = 3.0) -> bool:
        """Singlet **and** median per-azimuth SNR above the gate."""
        return self.is_singlet and float(np.nanmedian(self.snr)) >= min_snr

    def live_mask(self, min_snr: float = 3.0) -> np.ndarray:
        """Per-azimuth mask. Use this, not :meth:`usable`, before taking spreads."""
        return np.nan_to_num(self.snr) >= min_snr

    def describe(self, min_snr: float = 3.0) -> str:
        why = ("SINGLET" if self.is_singlet else f"multiplet({self.n_maxima})")
        if self.is_singlet and not self.usable(min_snr):
            why = "low SNR"
        off = ""
        if abs(self.centre_offset_bins) > 0.1 * max(self.half_width_bins, 1):
            off = f"  OFFSET {self.centre_offset_bins:+.1f} bins"
        return (f"centre {self.centre_bin:5d}  half {self.half_width_bins:3d} bins  "
                f"peak/bg {self.contrast:6.3f}  "
                f"SNR/eta {np.nanmedian(self.snr):7.2f}  "
                f"live {int(self.live_mask(min_snr).sum())}/{self.snr.size}  "
                f"{why}{off}")


# ------------------------------------------------------------------ windows
def ring_windows(r_axis: np.ndarray, centres_px, *, max_half_px: float = 16.0,
                 gap_frac: float = 0.45, min_half_bins: int = 6
                 ) -> tuple[dict, dict]:
    """Choose a radial window per ring: wide for area, but never onto a neighbour.

    Widths are set in **pixels and converted to bins**, never assumed to be bins.
    An integrated cake is commonly binned finer than a pixel (0.25 px/bin is
    typical), so a window written in bins is silently ~4x too narrow -- a mistake
    that costs most of the peak's tails, which is precisely where area lives.

    Returns
    -------
    (index, half)
        ``index[centre] -> nearest bin``, ``half[centre] -> half-width in bins``.
        Keyed by the requested centre so the caller can label rings.
    """
    r_axis = np.asarray(r_axis, dtype=float)
    n_r = r_axis.size
    px_per_bin = float(np.median(np.diff(r_axis)))
    if px_per_bin <= 0:
        raise ValueError("r_axis must be increasing")
    cent = np.asarray(sorted(float(c) for c in centres_px), dtype=float)
    index = {float(c): int(np.argmin(np.abs(r_axis - c))) for c in cent}
    half = {}
    for c in cent:
        others = cent[cent != c]
        gap = float(np.min(np.abs(others - c))) if others.size else 2 * max_half_px
        want_px = min(max_half_px, gap_frac * gap)
        hb = max(min_half_bins, int(round(want_px / px_per_bin)))
        i = index[float(c)]
        half[float(c)] = int(min(hb, i, n_r - 1 - i))      # never run off the axis
    return index, half


def refine_ring_centres(net: np.ndarray, r_axis: np.ndarray, index: dict,
                        half: dict, *, max_shift_frac: float = 0.5) -> dict:
    """Re-centre each window on the **measured** peak rather than a catalogue value.

    :func:`ring_windows` places a window at the bin nearest a *catalogued* ring
    position. If that position is even a fraction of a pixel off the real peak,
    the window is asymmetric about the peak -- and then any change in peak
    **width** between measurements is converted into an apparent change in
    **centroid**, which reads as strain that is not there. The offset is a fixed
    fraction of the window, so it does **not** dilute as the window widens, which
    is why a window-width sweep cannot detect it.

    Measured on an APS 1-ID DAC Ti scan whose ring positions came from a
    catalogue: offsets of −0.563, +0.044, −1.524 and −0.848 px on four rings, the
    largest being 19 % of that ring's FWHM. Re-centring moved one ring's apparent
    strain by **55 %** and flipped another's sign, while the ring whose offset was
    +0.044 px was unchanged to the bit.

    Parameters
    ----------
    net
        Background-subtracted cake, ``(n_r, n_eta)``, from
        :func:`background_from_ring_free`.
    index, half
        As returned by :func:`ring_windows`.
    max_shift_frac
        Refuse to move a centre by more than this fraction of its half-width. A
        larger implied shift means the window is not on the peak it was meant
        for -- a misassigned ring or a multiplet -- and quietly sliding the
        window onto whatever is brightest nearby would hide that.

    Returns
    -------
    dict
        ``centre -> refined bin index``, same keys as ``index``. Centres that
        could not be refined are returned unchanged.
    """
    net = np.asarray(net, dtype=float)
    r_axis = np.asarray(r_axis, dtype=float)
    refined = {}
    for c, i0 in index.items():
        hb = int(half[c])
        lo, hi = max(0, int(i0) - hb), min(net.shape[0], int(i0) + hb + 1)
        prof = np.clip(net[lo:hi, :].mean(axis=1), 0.0, None)
        total = prof.sum()
        if total <= 0 or prof.size < 3:
            refined[c] = int(i0)
            continue
        # intensity-weighted centre of the window profile, in bins
        com = float((prof * np.arange(lo, hi)).sum() / total)
        shift = com - float(i0)
        if abs(shift) > max_shift_frac * hb:
            log.warning(
                "ring at %.3f: measured peak is %+.1f bins from its catalogued "
                "centre, more than %.0f%% of the half-width (%d bins). NOT "
                "re-centred -- check the ring assignment; this looks like a "
                "misassignment or a multiplet, not a small offset.",
                c, shift, 100 * max_shift_frac, hb)
            refined[c] = int(i0)
            continue
        refined[c] = int(round(com))
    return refined


def ring_free_mask(n_r: int, index: dict, half: dict, *, widen: float = 1.6
                   ) -> np.ndarray:
    """``True`` where a known ring sits, widened, so the background never sees one.

    ``widen`` above 1 matters: the tails carry area, and a background estimated
    from radii that still contain tail is biased up.
    """
    m = np.zeros(int(n_r), dtype=bool)
    for c, i in index.items():
        h = int(half[c] * widen)
        m[max(0, i - h):min(int(n_r), i + h + 1)] = True
    return m


# --------------------------------------------------------------- background
def background_from_ring_free(cake: np.ndarray, peak_mask: np.ndarray,
                              block_bins: int, *, percentile: float = 20.0
                              ) -> tuple[np.ndarray, np.ndarray]:
    """Background per azimuth from ring-free radii, interpolated across the peaks.

    Parameters
    ----------
    cake
        ``(n_r, n_eta)`` integrated intensities.
    peak_mask
        ``(n_r,)`` boolean, ``True`` where a ring sits -- from
        :func:`ring_free_mask`.
    block_bins
        Block width for the percentile, in bins. Should be **much wider than a
        peak but narrower than the background's own variation**; ~30 px is a
        reasonable default at typical DT geometries.
    percentile
        Low percentile of the ring-free samples in each block.

    Returns
    -------
    (net, background)
        Both ``(n_r, n_eta)``. ``net`` is *not* clipped at zero -- clipping is a
        decision for whoever integrates, and doing it here would bias the
        centroid.

    Notes
    -----
    Per-azimuth, not per-ring: the background varies with **both** R and eta
    (Compton from the anvils falls with angle, absorption varies with path
    length). A single radial background applied to every azimuth leaves an eta
    pattern that looks exactly like texture. That is the specific reading this
    module exists to prevent -- an "hkl-dependent eta pattern therefore texture"
    conclusion drawn from windowed sums that were 60-85 % background had to be
    withdrawn.
    """
    a = np.asarray(cake, dtype=float)
    if a.ndim != 2:
        raise ValueError(f"cake must be (n_r, n_eta), got {a.shape}")
    n_r, n_eta = a.shape
    free = ~np.asarray(peak_mask, dtype=bool)
    if free.size != n_r:
        raise ValueError(f"peak_mask has {free.size} entries, cake has {n_r} radii")
    if free.mean() < 0.35:
        log.warning("only %.0f%% of radii are ring-free; falling back to all radii, "
                    "which biases the background UP under every peak",
                    100 * free.mean())
        free = np.ones(n_r, dtype=bool)

    nb = max(8, n_r // max(1, int(block_bins)))
    edges = np.linspace(0, n_r, nb + 1).astype(int)
    centres, vals = [], []
    for k in range(nb):
        sl = slice(edges[k], edges[k + 1])
        sel = free[sl]
        if sel.sum() < 3:
            continue
        centres.append(0.5 * (edges[k] + edges[k + 1]))
        vals.append(np.percentile(a[sl][sel], percentile, axis=0))
    if len(centres) < 2:
        log.warning("fewer than two usable background blocks; using a per-azimuth "
                    "minimum, which is a floor rather than a background")
        bg = np.broadcast_to(a.min(axis=0, keepdims=True), a.shape).copy()
        return a - bg, bg

    idx_all = np.arange(n_r)
    cs = np.asarray(centres, dtype=float)
    vs = np.asarray(vals, dtype=float)                     # (n_block, n_eta)
    bg = np.empty_like(a)
    for j in range(n_eta):
        bg[:, j] = np.interp(idx_all, cs, vs[:, j])
    return a - bg, bg


# ----------------------------------------------------------------- vetting
def count_maxima(profile: np.ndarray, *, min_frac: float = 0.5,
                 min_prominence: float = 0.10) -> int:
    """Distinct local maxima in a window. ``1`` means singlet.

    Run this on the **background-subtracted** azimuthal mean of the window. A
    sub-pixel ring assignment will match a doublet as a single line -- one hcp Ti
    (101) window held maxima at 381.6 and 393.6 px with a dip between, assigned
    as one ring at 393.6.

    Parameters
    ----------
    min_frac
        A maximum must reach this fraction of the tallest one to count.
    min_prominence
        Two maxima are **distinct** only if the minimum between them falls at
        least this fraction of the peak height below the *shorter* of the two.

    Notes
    -----
    Height alone is not enough, and this is not a refinement. Photon noise on a
    peak's own plateau readily produces several samples that are each above half
    height and each larger than their two neighbours -- so a height-only count
    calls a clean singlet a triplet, and the ring gets rejected for being a
    multiplet it is not. A genuine doublet is defined by the **dip** between its
    lobes, not by having two high points: the Ti (101) doublet's dip sits ~60 %
    below its lobes, while noise ripple on a plateau is a few percent. That gap
    is wide, so the discriminator is robust rather than tuned.
    """
    p = np.asarray(profile, dtype=float)
    if p.size < 3 or not np.isfinite(p).any():
        return 0
    pk = float(np.nanmax(p))
    if not np.isfinite(pk) or pk <= 0:
        return 0
    cand = [k for k in range(1, p.size - 1)
            if p[k] >= min_frac * pk and p[k] >= p[k - 1] and p[k] > p[k + 1]]
    if len(cand) <= 1:
        return len(cand)
    # Merge candidates not separated by a deep enough dip, left to right.
    kept = [cand[0]]
    for k in cand[1:]:
        prev = kept[-1]
        dip = float(np.nanmin(p[prev:k + 1]))
        if min(p[prev], p[k]) - dip >= min_prominence * pk:
            kept.append(k)
        elif p[k] > p[prev]:
            kept[-1] = k                       # same lobe: keep the taller sample
    return len(kept)


def snr_per_eta(raw_window: np.ndarray, net_window: np.ndarray) -> np.ndarray:
    """Integrated peak signal over Poisson noise, per azimuth.

    The decisive number for a low-contrast scan, and cheap. At ~220 counts of
    background a bin carries ~15 counts of Poisson noise, while a peak at
    ``peak/bg = 0.02`` has an amplitude of ~4 counts -- **below the per-bin
    noise**. Signal only emerges after integrating the window, and then only for
    the strongest rings. Computing areas and centroids without checking this is
    how every unstable texture number on the DAC Ti scan was produced.
    """
    sig = np.clip(np.asarray(net_window, dtype=float), 0.0, None).sum(axis=0)
    noise = np.sqrt(np.maximum(np.asarray(raw_window, dtype=float), 0.0).sum(axis=0))
    return sig / np.maximum(noise, 1e-9)


def mad_filter(x: np.ndarray, *, cut: float = 3.0) -> np.ndarray:
    """Keep-mask for points within ``cut`` median absolute deviations.

    Muerer et al. 2021's azimuthal outlier rejection: a few large crystallites
    make a ring spotty, and a spot is not a pole figure. Returns all-``True`` when
    the MAD is zero or undefined, so a degenerate window is passed through rather
    than silently emptied.
    """
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x)
    mad = 1.4826 * np.nanmedian(np.abs(x - med))
    if not np.isfinite(mad) or mad <= 0:
        return np.ones_like(x, dtype=bool)
    return np.abs(x - med) <= cut * mad


def radial_half_correlation(net_window: np.ndarray) -> float:
    """Correlation over azimuth between the inner and outer halves of a ring.

    **The discriminator for a peak that is MOVING rather than changing amplitude**
    -- which is the difference between real strain and fake texture (see the module
    docstring on Singh's relation).

    Returns
    -------
    float
        ``corr(inner_half_intensity, outer_half_intensity)`` over azimuth.

        * **Strongly negative** -- the peak is **moving**. Intensity leaves one half
          as it enters the other, so the halves are anti-correlated. Any azimuthal
          "texture" from a fixed window is then largely window truncation.
        * **Positive** -- the peak's **amplitude** is changing while its position
          holds. That is what a genuine pole figure looks like.
        * **Near zero** -- neither dominates, or the ring is noise.

    Notes
    -----
    Measured **−0.72** on the 11-ID-C CeO₂ scan, a sample that should have no
    texture at all, which is what identified peak movement as the cause of a
    spurious structured ODF (``manuals/xrd-ct/LAB_NOTEBOOK.md`` §5b).

    This replaced an edge-occupancy ("containment") test that could not work. That
    test compared the window's edge level against a fraction of the peak, and at
    realistic contrast the comparison is noise-limited: at 26 % contrast on a
    ~200-count background, the 3-sigma floor on an edge mean is ~21 counts against
    a peak amplitude of ~45, so truncation is simply not detectable that way and a
    perfectly stationary ring scored 0.05. Summing each **half** of the window
    instead averages over many bins, so this statistic is a ratio of large numbers
    and survives the same contrast where the edge test collapses.
    """
    w = np.asarray(net_window, dtype=float)
    if w.shape[0] < 4:
        raise ValueError(
            f"window has {w.shape[0]} bins; need at least 4 to split into halves")
    mid = w.shape[0] // 2
    inner = w[:mid].sum(axis=0)
    outer = w[mid:].sum(axis=0)
    if inner.size < 3 or np.std(inner) == 0 or np.std(outer) == 0:
        return float("nan")
    return float(np.corrcoef(inner, outer)[0, 1])


# -------------------------------------------------------------- extraction
def area_and_centroid(net: np.ndarray, r_axis: np.ndarray, centre_bin: int,
                      half_bins: int) -> tuple[np.ndarray, np.ndarray]:
    """Integrated area and intensity-weighted radius per azimuth.

    Negative net values are clipped **for the moments only**: a centroid weighted
    by negative intensities is not a position. The clip is why the centroid is
    robust here and the area is not.
    """
    net = np.asarray(net, dtype=float)
    lo = max(0, int(centre_bin) - int(half_bins))
    hi = min(net.shape[0], int(centre_bin) + int(half_bins) + 1)
    w = net[lo:hi, :]
    rw = np.asarray(r_axis, dtype=float)[lo:hi]
    pos = np.clip(w, 0.0, None)
    area = pos.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        cen = (pos * rw[:, None]).sum(axis=0) / area
    return area, cen


def extract_ring(cake: np.ndarray, net: np.ndarray, background: np.ndarray,
                 r_axis: np.ndarray, centre_bin: int, half_bins: int,
                 *, min_frac: float = 0.5, recentre: bool = False,
                 warn_offset_frac: float = 0.1) -> RingExtraction:
    """Everything about one ring at every azimuth, gates included.

    ``net`` and ``background`` come from :func:`background_from_ring_free` on the
    same ``cake``; they are passed in rather than recomputed because the
    background is a whole-cake quantity and computing it per ring would let each
    ring see a different one.

    ``centre_offset_bins`` is **always** measured and reported: it is the peak's
    own centre of mass minus the window centre. A non-zero offset means the
    window is asymmetric about the peak, which turns width changes into apparent
    centroid changes -- apparent strain. Set ``recentre=True`` to place the
    window on the measured peak instead (see :func:`refine_ring_centres`); the
    default is ``False`` so this never silently changes an existing analysis's
    numbers, but an offset above ``warn_offset_frac`` of the half-width logs a
    warning, because a silent 1.5-px offset has cost this project a result.

    **Magnitude, so this is not over-claimed.** For a Gaussian peak with a +45 %
    width change, the artefact depends steeply on how wide the window is relative
    to the peak: ``half/sigma`` 5.0 gives only 0.02-0.22 bins, ``half/sigma`` 2.5
    gives 0.72-2.38 bins. So an off-centre window is a **second-order** error for
    a comfortably wide window and a first-order one near or below the ~2.3x FWHM
    safety line. Where real data moves far more than this model predicts, the
    cause is non-Gaussian content entering the window (background residual,
    neighbour tails), not Gaussian tail truncation.

    **Limitation.** ``centre_bin`` is an integer, so re-centring cannot converge
    below ~0.5 bin and leaves that much residual offset. Measured: at
    ``half/sigma`` 2.5 the correction is essentially exact, while at
    ``half/sigma`` 1.2 -- a window *narrower than the peak*, already outside the
    safety line -- it removes only ~2.6x. Sub-bin centring would need the window
    resampled, which would change the extraction; widen the window instead.
    """
    cake = np.asarray(cake, dtype=float)
    net = np.asarray(net, dtype=float)
    centre_bin = int(centre_bin)
    half_bins = int(half_bins)

    def _window(c):
        return (max(0, c - half_bins), min(net.shape[0], c + half_bins + 1))

    lo, hi = _window(centre_bin)
    prof = np.clip(net[lo:hi, :].mean(axis=1), 0.0, None)
    total = prof.sum()
    offset = (float((prof * np.arange(lo, hi)).sum() / total) - centre_bin
              if total > 0 else 0.0)

    if recentre and total > 0 and abs(offset) <= 0.5 * half_bins:
        # ITERATE. The centre of mass measured through an off-centre window is
        # itself pulled toward the window centre, so one correction step
        # systematically under-shoots -- measured, a single pass removed only
        # 2.4x of the artefact where iterating removes >10x.
        for _ in range(_RECENTRE_ITERS):
            if abs(offset) < 0.05:
                break
            centre_bin = int(round(centre_bin + offset))
            lo, hi = _window(centre_bin)
            prof = np.clip(net[lo:hi, :].mean(axis=1), 0.0, None)
            tot2 = prof.sum()
            if tot2 <= 0:
                break
            offset = float((prof * np.arange(lo, hi)).sum() / tot2) - centre_bin
    elif abs(offset) > warn_offset_frac * max(half_bins, 1):
        log.warning(
            "ring window at bin %d is %+.2f bins off its own peak (%.0f%% of "
            "the half-width). An off-centre window converts peak-WIDTH changes "
            "into apparent CENTROID changes, i.e. into strain that is not "
            "there, and a window-width sweep cannot detect it. Pass "
            "recentre=True or use refine_ring_centres().",
            centre_bin, offset, 100 * abs(offset) / max(half_bins, 1))

    area, cen = area_and_centroid(net, r_axis, centre_bin, half_bins)
    prof_raw = net[lo:hi, :].mean(axis=1)
    base = float(np.asarray(background, dtype=float)[centre_bin, :].mean())
    return RingExtraction(
        area=area,
        centroid=cen,
        snr=snr_per_eta(cake[lo:hi, :], net[lo:hi, :]),
        n_maxima=count_maxima(prof_raw, min_frac=min_frac),
        contrast=float(np.nanmax(prof_raw)) / max(base, 1e-9),
        half_width_bins=half_bins,
        centre_bin=centre_bin,
        centre_offset_bins=float(offset),
    )


def azimuthal_rebin(x: np.ndarray, factor: int, *, keep: np.ndarray | None = None,
                    how: str = "mean") -> np.ndarray:
    """Bin an ``(n_eta,)`` or ``(..., n_eta)`` array by an integer factor.

    Trailing azimuths that do not fill a bin are dropped, not partially filled.
    ``keep`` is an optional same-shape boolean mask (e.g. from :func:`mad_filter`)
    applied within each bin; bins with nothing kept come back as 0 for ``'mean'``.
    ``how='quadrature'`` combines uncertainties as ``sqrt(sum sigma^2) / n``.
    """
    x = np.asarray(x, dtype=float)
    f = int(factor)
    if f < 1:
        raise ValueError("factor must be >= 1")
    n = x.shape[-1] // f
    if n == 0:
        raise ValueError(f"factor {f} exceeds the {x.shape[-1]} azimuths available")
    xr = x[..., :n * f].reshape(*x.shape[:-1], n, f)
    if keep is None:
        if how == "mean":
            return xr.mean(axis=-1)
        if how == "quadrature":
            return np.sqrt((xr ** 2).sum(axis=-1)) / f
        raise ValueError(f"unknown how={how!r}")
    k = np.asarray(keep, dtype=bool)[..., :n * f].reshape(*x.shape[:-1], n, f)
    cnt = k.sum(axis=-1)
    denom = np.maximum(cnt, 1)
    if how == "mean":
        return np.where(cnt > 0, (xr * k).sum(axis=-1) / denom, 0.0)
    if how == "quadrature":
        return np.sqrt((xr ** 2 * k).sum(axis=-1)) / denom
    raise ValueError(f"unknown how={how!r}")


def strain_from_centroid(centroid: np.ndarray, r_axis: np.ndarray,
                         two_theta_axis: np.ndarray, wavelength_a: float,
                         *, reference_d: float | None = None) -> np.ndarray:
    """Azimuthal strain from ring centroids, via the (R -> 2theta) calibration map.

    ``two_theta_axis`` is the per-radius 2theta in **degrees**, as written by the
    integrator's ``.REtaAreaMap`` -- use it rather than an idealised
    ``arctan(r/L)``, because it carries the detector tilts and the distortion
    coefficients.

    With ``reference_d`` unset the reference is the **median over the azimuths
    supplied**, which makes this a *relative* (deviatoric-like) strain and
    absorbs any error in the sample-to-detector distance. That is deliberate: an
    absolute strain from a DT scan needs an independently calibrated distance,
    and on one 11-ID-C dataset the metadata distance and the beamline calibration
    were both wrong by ~2-3 % for the data as collected. Supply
    ``reference_d`` only when you have a d0 you can defend.
    """
    cen = np.asarray(centroid, dtype=float)
    if reference_d is None and not np.isfinite(cen).any():
        raise ValueError(
            "reference d-spacing is not positive; every centroid is NaN, so "
            "there is nothing to reference against (all azimuths gated out?)")
    tt = np.interp(cen, np.asarray(r_axis, dtype=float),
                   np.asarray(two_theta_axis, dtype=float))
    d = float(wavelength_a) / (2.0 * np.sin(np.radians(tt) / 2.0))
    ref = float(np.nanmedian(d)) if reference_d is None else float(reference_d)
    if not np.isfinite(ref) or ref <= 0:
        raise ValueError("reference d-spacing is not positive; all centroids NaN?")
    return d / ref - 1.0
