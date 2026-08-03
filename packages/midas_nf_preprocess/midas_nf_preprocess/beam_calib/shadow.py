"""``ybc``, particle count, and particle positions from the sample shadow.

The sample sits in the direct beam and absorbs, so it casts a shadow in the
beam stripe.  A particle offset from the rotation axis by ``u`` projects to
``+u`` at omega and ``-u`` at omega+180, so tracking the shadow over a full
turn gives

    centre(omega) = axis + u*cos(omega) + v*sin(omega)

The CONSTANT term is the projection of the rotation axis, which is what MIDAS
wants for ``ybc``: the axis lies on the beam, and a point on the beam axis
shadows at BC (handbook §6e).  ``hypot(u, v)`` is how far the particle really
sits off the axis.

A particle exactly ON the axis casts a STATIONARY shadow -- so the number of
distinct shadow features, and which of them move, tells you how many
particles there are and which is centred.  On `nfdev_jul26` this is how two
gold cubes were found where one was expected.

Three traps, all of which produced wrong numbers before being fixed
-------------------------------------------------------------------
1. **The reference must be the omega-MEDIAN of the stripe profile.**  A median
   filter along the COLUMN axis cannot separate a stationary absorber from a
   moving one; the dip finder locks onto the stationary feature and the
   sinusoid fit returns nonsense (rms 230-310 px on a ~900 px amplitude).  The
   omega-median cancels everything stationary by construction.
2. **Reject clipped points before fitting.**  When the swing is comparable to
   the beam width the shadow leaves the illuminated band and the tracker
   pins it to the band edge; those flat tops bias the amplitude DOWN and
   wreck the fit.
3. **Do not centroid ``1 - T`` over the whole band.**  The band is far wider
   than the dip and the noise integral swamps it -- 66 px of scatter at 1-ID
   versus 0.2 px for the edge-based estimator (handbook §6e).
4. **``band_frac=0.30`` is NOT a safe default -- tune it against a known
   answer.**  At 20-ID the 0.30 default admits the beam's dim wings, the dip
   finder wanders there, and the axis comes back **+96 to +130 px wrong** with
   a clipped amplitude.  Swept against the known ``nfdev_jul26`` axis
   (col 2625.47)::

       band_frac   axis_col   amplitude   rms      is_reliable
       0.30        2721.15    634 px      216 px   False
       0.50        2787.83    500 px      202 px   False
       0.70        2625.88    918 px      1.5 px   True     <-- +0.41 px

   At 0.70 the amplitude also independently reproduces the 496.8 um cube-2
   offset.  Note that :attr:`ShadowFit.is_reliable` was False for every bad
   row -- **branch on it**; it is the whole point of the flag.

   If the beam profile carries a narrow bright spike (there is one near
   col 3600 in ``xzhang_jul26``), ``band_frac * ref.max()`` selects only the
   spike and everything reports as clipped.  Crop to the flat core first.

Scope limit -- this method needs a COMPACT absorber
---------------------------------------------------
The fit assumes the shadow centre traces a rigid sinusoid.  That holds for a
particle and fails for an extended irregular specimen: on ``s6061_NF`` the
shadow width swung 56 -> 886 px with omega and :func:`fit_axis` refused at
every setting.  When it refuses, ybc is not measurable from that scan --
inherit it, mark it inherited, and let the refinement move it.  Do not lower
the bar until something passes.

Handbook: §6e, §6e-0, §6i-bis.  Lab notebook: §7c, §7d, §7f (F3), §8e.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

__all__ = ["ShadowTrack", "ShadowFit", "StationaryFeature",
           "omega_median_reference", "track_shadow", "fit_axis",
           "find_stationary"]


@dataclass
class ShadowTrack:
    """Per-frame shadow centre, before any fitting."""

    omega_deg: np.ndarray
    centre_col: np.ndarray
    width_px: np.ndarray
    depth: np.ndarray
    n_frames_total: int

    @property
    def found_fraction(self) -> float:
        return len(self.centre_col) / max(self.n_frames_total, 1)


@dataclass
class ShadowFit:
    """Rotation-axis projection and the particle's true offset."""

    axis_col: float
    amplitude_px: float
    phase_deg: float
    resid_rms_px: float
    n_used: int
    n_clipped: int
    amplitude_um: Optional[float] = None
    width_med_um: Optional[float] = None

    @property
    def is_reliable(self) -> bool:
        """Residual must be small against the amplitude it is fitting."""
        return (self.n_used >= 30
                and self.resid_rms_px < 0.05 * max(self.amplitude_px, 1.0))

    def ybc(self, n_pixels_y: int) -> float:
        """Beam centre in the MIDAS convention (``NrPixelsY-1 - col``)."""
        return (n_pixels_y - 1) - self.axis_col

    def position_candidates_um(self, px_um: float) -> List[Tuple[float, float]]:
        """The particle's sample-frame ``(x, y)``.

        MEASURED CONVENTION (20-ID, aero stage, omega = -theta)::

            angle = -phase - 90 deg      i.e.   (x, y) = (-a sin phi, -a cos phi)

        Pinned on TWO independent campaigns whose reconstructions located the
        same off-axis Au cube:

        ==============  ===========  ==========  ===========
        campaign          phase phi   recon ang   theta+phi
        ==============  ===========  ==========  ===========
        nfdev_jul26        +34.18      -123.8      -89.62
        NF_Au_cube_0802    +35.05      -125.1      -90.05
        ==============  ===========  ==========  ===========

        ``theta + phi = -90 deg`` to 0.43 deg across both; ``theta - phi`` is
        NOT constant (-158.0 vs -160.2), so the relation is unambiguously the
        sum.  Predicted positions match the reconstructions to 0.05 deg and
        0.38 deg respectively.

        THIS REPLACES the old ``(a cos phi, a sin phi)`` form, which was wrong
        in FORM, not merely in sign -- masking it and its antipode found nothing
        on BOTH campaigns (every off-axis voxel exactly 0.0000), because the
        true position is 90 deg away from either.  The antipode is still
        returned second as a fallback.

        **Do not build a point mask from this on a new beamline.**  The relation
        above encodes an omega sign and a detector handedness; on unfamiliar
        geometry use a full annulus at :attr:`amplitude_px` (the RADIUS is
        convention-free) and let the reconstruction pick the angle.  Two
        campaigns agreeing is evidence, not proof.
        """
        a = self.amplitude_px * px_um
        phi = np.radians(self.phase_deg)
        x = -a * np.sin(phi)
        y = -a * np.cos(phi)
        return [(float(x), float(y)), (float(-x), float(-y))]


@dataclass
class StationaryFeature:
    """An absorber that does not move with omega, i.e. one on the axis."""

    col_centroid: float
    width_px: int
    transmission: float
    width_um: Optional[float] = None


def omega_median_reference(profiles: np.ndarray) -> np.ndarray:
    """Reference beam profile: the median over omega.

    ``profiles`` is ``(n_frames, n_cols)``.  Everything stationary -- the beam
    envelope, fixed absorbers, dead columns, scintillator defects -- survives
    the median and therefore cancels when each frame is divided by it.  Only
    a MOVING absorber leaves a residual.
    """
    return np.median(profiles, axis=0)


def _boxcar(x: np.ndarray, n: int) -> np.ndarray:
    return np.convolve(x, np.ones(n) / n, mode="same")


def track_shadow(
    profiles: np.ndarray,
    omega_deg: Sequence[float],
    *,
    band_frac: float = 0.30,
    smooth_px: int = 15,
    min_depth: float = 0.05,
) -> ShadowTrack:
    """Follow the MOVING shadow across omega."""
    ref = omega_median_reference(profiles)
    band = ref > band_frac * ref.max()
    cols = np.arange(profiles.shape[1])

    om, ce, wd, dp = [], [], [], []
    for k in range(profiles.shape[0]):
        with np.errstate(invalid="ignore", divide="ignore"):
            T = np.where(band, profiles[k] / np.maximum(ref, 1e-9), np.nan)
        Ts = np.where(band, _boxcar(np.nan_to_num(T, nan=1.0), smooth_px), np.nan)
        if np.all(np.isnan(Ts)):
            continue
        depth = 1.0 - float(np.nanmin(Ts))
        if depth < min_depth:
            continue
        imin = int(np.nanargmin(Ts))
        deficit = np.nan_to_num(np.where(band, 1.0 - Ts, 0.0), nan=0.0)
        half = 0.5 * deficit[imin]
        lo = imin
        while lo > 0 and deficit[lo] > half:
            lo -= 1
        hi = imin
        while hi < len(deficit) - 1 and deficit[hi] > half:
            hi += 1
        seg = deficit[lo:hi + 1]
        if seg.sum() <= 0:
            continue
        om.append(float(omega_deg[k]))
        ce.append(float((cols[lo:hi + 1] * seg).sum() / seg.sum()))
        wd.append(float(hi - lo + 1))
        dp.append(depth)
    return ShadowTrack(np.array(om), np.array(ce), np.array(wd), np.array(dp),
                       n_frames_total=profiles.shape[0])


def fit_axis(
    track: ShadowTrack,
    *,
    px_um: Optional[float] = None,
    edge_reject_px: float = 90.0,
    n_sigma: float = 3.0,
    max_iter: int = 6,
) -> ShadowFit:
    """Fit ``centre = axis + u cos(omega) + v sin(omega)``, rejecting clipping.

    Points within ``edge_reject_px`` of the observed extremes are dropped
    first: when the swing approaches the beam width the shadow leaves the
    illuminated band and the tracker pins it to the edge, and those flat tops
    bias the amplitude DOWN.  On `nfdev_jul26` this took the fit residual from
    230 px to 6 px and the amplitude from a clipped 497 px to a true 906 px.
    """
    c, th = track.centre_col, np.radians(track.omega_deg)
    if c.size < 8:
        raise ValueError("too few tracked frames to fit an axis")
    lo, hi = c.min(), c.max()
    keep = (c > lo + edge_reject_px) & (c < hi - edge_reject_px)
    n_clipped = int((~keep).sum())
    if keep.sum() < 8:
        raise ValueError("everything was clipped -- the shadow never stays "
                         "inside the illuminated band; widen the band or use "
                         "a wider beam")
    for _ in range(max_iter):
        A = np.column_stack([np.ones(keep.sum()), np.cos(th[keep]), np.sin(th[keep])])
        coef, *_ = np.linalg.lstsq(A, c[keep], rcond=None)
        Aall = np.column_stack([np.ones_like(th), np.cos(th), np.sin(th)])
        r = c - Aall @ coef
        s = float(np.std(r[keep]))
        new = keep & (np.abs(r) < n_sigma * s)
        if new.sum() == keep.sum() or new.sum() < 8:
            break
        keep = new
    A = np.column_stack([np.ones(keep.sum()), np.cos(th[keep]), np.sin(th[keep])])
    coef, *_ = np.linalg.lstsq(A, c[keep], rcond=None)
    r = c[keep] - A @ coef
    amp = float(np.hypot(coef[1], coef[2]))
    return ShadowFit(
        axis_col=float(coef[0]), amplitude_px=amp,
        phase_deg=float(np.degrees(np.arctan2(coef[2], coef[1]))),
        resid_rms_px=float(np.sqrt((r ** 2).mean())),
        n_used=int(keep.sum()), n_clipped=n_clipped,
        amplitude_um=(amp * px_um) if px_um else None,
        width_med_um=(float(np.median(track.width_px)) * px_um) if px_um else None,
    )


def find_stationary(
    profiles: np.ndarray,
    *,
    px_um: Optional[float] = None,
    envelope_px: int = 301,
    band_frac: float = 0.30,
) -> Optional[StationaryFeature]:
    """Find an absorber that does NOT move -- i.e. one on the rotation axis.

    Detected by dividing the omega-median profile by its own smooth envelope:
    the envelope rides over a narrow dip, so a stationary absorber survives
    while the beam shape divides out.  A stationary shadow is only possible
    for something on the axis, so this both counts particles and identifies
    the centred one.
    """
    from scipy.ndimage import median_filter

    ref = omega_median_reference(profiles)
    env = median_filter(ref, size=envelope_px, mode="nearest")
    band = env > band_frac * env.max()
    with np.errstate(invalid="ignore", divide="ignore"):
        T = np.where(band, ref / np.maximum(env, 1e-9), np.nan)
    if np.all(np.isnan(T)):
        return None
    imin = int(np.nanargmin(T))
    tmin = float(T[imin])
    if tmin > 0.97:                     # no meaningful stationary absorber
        return None
    deficit = np.nan_to_num(np.where(band, 1.0 - T, 0.0), nan=0.0)
    half = 0.5 * deficit[imin]
    lo = imin
    while lo > 0 and deficit[lo] > half:
        lo -= 1
    hi = imin
    while hi < len(deficit) - 1 and deficit[hi] > half:
        hi += 1
    seg = deficit[lo:hi + 1]
    cols = np.arange(len(ref))
    cen = float((cols[lo:hi + 1] * seg).sum() / seg.sum()) if seg.sum() > 0 else float(imin)
    w = int(hi - lo + 1)
    return StationaryFeature(cen, w, tmin, (w * px_um) if px_um else None)
