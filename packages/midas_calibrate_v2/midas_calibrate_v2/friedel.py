"""Calibration from Friedel pairs — for data that has no calibrant.

Everything in :mod:`midas_calibrate_v2.seed` finds a beam centre from powder
rings. That is the right tool when a calibrant exposure exists. It is useless on
a spotty single-crystal frame — a diamond-anvil cell, a protein-style rotation
series, a sample measured without a CeO₂ shot — where the only azimuthal
reference on the detector is the sample's own diffraction.

A Friedel pair carries that reference for free. +G and -G land at equal
distance on opposite sides of the beam centre, so pair midpoints locate the
centre with no geometry file at all. This module does that, and two things that
follow from it.

Antipodal is not Friedel
------------------------
With a short wavelength the Ewald sphere is nearly a plane through the origin,
and a plane through the origin is centrosymmetric — so *every* grain's
accessible reflections come in near-antipodal positions on the detector. In a
multi-grain sample the antipode is frequently a **different grain**, and
position alone cannot tell the two apart.

There is a test that can, and it needs no orientation, no cell and no free
parameter. A genuine pair's ω separation is fixed by |G| alone::

    abs(sin(dω/2)) = c/ρ >= c/|G| = λ|G|/2 ,     c = λ|G|²/2

A pair whose observed ω separation falls below that floor cannot be a Friedel
pair. :func:`is_friedel` applies it. The consequence is not cosmetic: it decides
which pairs are allowed to set the beam centre, and on a short ω scan it can
reject the large majority of them.

tx, which a powder calibration cannot see at all
------------------------------------------------
The detector's in-plane rotation about the beam preserves every pixel's radius,
so powder rings do not move under it and a ring calibration is structurally
blind to it. It also rotates *both* members of a Friedel pair equally, so their
antipodal symmetry is untouched — pair *symmetry* cannot see it either.

The ω **splitting** can. From the diffraction condition with a vertical
rotation axis, a pair's radius and its Δω together predict its lab azimuth with
no crystal information in it at all; the offset between that prediction and the
observed detector azimuth is tx. :func:`tx_from_pairs` returns it.

Validate before pointing it at data: :func:`simulate_tx_recovery` plants a known
tx and recovers it, which is the positive control this estimator earns its
credibility from.

Branch ambiguity
----------------
G_z enters through a square root, so each pair admits two azimuths and only one
is physical. Picking the nearer one biases the estimate toward zero. Both are
kept instead and the estimator is the **median of signed offsets**:
wrong-branch pairs scatter symmetrically and cancel, the correct branch
accumulates. The median *absolute* offset is therefore large and meaningless;
the median *signed* offset is the measurement.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "BeamCentreResult", "TxResult",
    "friedel_domega_floor", "is_friedel",
    "antipodal_pairs", "beam_centre_from_pairs",
    "predicted_eta", "tx_from_pairs",
]


# ---------------------------------------------------------------------------
# geometry helpers
# ---------------------------------------------------------------------------

def _q_from_radius(radius_px, *, wavelength_A: float,
                   lsd_um: float, pixel_um: float) -> np.ndarray:
    """|G| in 1/Å from a detector radius, flat untilted model."""
    tth = np.arctan2(np.asarray(radius_px, float) * pixel_um, lsd_um)
    return 2.0 * np.sin(tth / 2.0) / wavelength_A


def friedel_domega_floor(q_mag_inv_A, wavelength_A: float) -> np.ndarray:
    """Smallest ω separation, in degrees, a Friedel pair of this |G| can have.

    From ``abs(sin(dω/2)) >= λ|G|/2``. Returns NaN where ``λ|G|/2 > 1``, i.e.
    beyond the limiting sphere where no pair exists.
    """
    s = np.asarray(q_mag_inv_A, float) * wavelength_A / 2.0
    out = np.full(np.shape(s), np.nan, float)
    ok = s <= 1.0
    out[ok] = 2.0 * np.degrees(np.arcsin(s[ok]))
    return out


def is_friedel(radius_px, domega_deg, *, wavelength_A: float,
               lsd_um: float, pixel_um: float,
               tolerance_deg: float = 0.0) -> np.ndarray:
    """True where an antipodal pair's ω separation is physically allowed.

    ``radius_px`` is the pair's mean radius from the beam centre;
    ``domega_deg`` its observed ω separation. ``tolerance_deg`` loosens the
    floor to absorb the ω step — set it from the scan, not by eye.
    """
    q = _q_from_radius(radius_px, wavelength_A=wavelength_A,
                       lsd_um=lsd_um, pixel_um=pixel_um)
    floor = friedel_domega_floor(q, wavelength_A)
    obs = np.abs(np.asarray(domega_deg, float))
    return np.isfinite(floor) & (obs >= floor - tolerance_deg)


# ---------------------------------------------------------------------------
# beam centre
# ---------------------------------------------------------------------------

@dataclass
class BeamCentreResult:
    """A beam centre measured from pair midpoints, with the null that tested it."""
    row: float
    col: float
    row_sigma: float
    col_sigma: float
    n_pairs: int
    pairs: np.ndarray                  # (n_pairs, 2) int index pairs
    half_separation_px: np.ndarray
    n_matched: int                     # spots with an antipodal partner
    null_counts: np.ndarray            # matched count per null draw
    p_value: float

    def __str__(self) -> str:
        return (f"beam centre row {self.row:.3f} +/- {self.row_sigma:.3f}, "
                f"col {self.col:.3f} +/- {self.col_sigma:.3f} from "
                f"{self.n_pairs} pairs; matched {self.n_matched} vs null median "
                f"{np.median(self.null_counts):.0f} "
                f"(max {self.null_counts.max()}), p = {self.p_value:.3f}")


def _match_antipodal(centre, points: np.ndarray, tol: float):
    partner = 2.0 * np.asarray(centre, float) - points
    d = np.linalg.norm(points[:, None, :] - partner[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    return d


def antipodal_pairs(points: np.ndarray, centre, tol_px: float = 3.0
                    ) -> np.ndarray:
    """Distinct antipodal index pairs about ``centre``, within ``tol_px``."""
    points = np.asarray(points, float)
    d = _match_antipodal(centre, points, tol_px)
    j = d.argmin(1)
    ok = d.min(1) < tol_px
    seen = {tuple(sorted((int(i), int(j[i])))) for i in np.flatnonzero(ok)}
    return np.array(sorted(seen), dtype=int).reshape(-1, 2)


def beam_centre_from_pairs(points: np.ndarray, *,
                           seed: Optional[Sequence[float]] = None,
                           search_px: float = 30.0,
                           step_px: float = 1.0,
                           tol_px: float = 3.0,
                           n_null: int = 200,
                           rng_seed: int = 0) -> BeamCentreResult:
    """Measure the beam centre from Friedel-pair midpoints. No geometry file.

    ``points`` is an (N, 2) array of spot (row, col) in whatever order the
    detector array is stored — this function applies no flip and knows of none,
    which is exactly why it can be used to *test* a delivered PONI rather than
    inherit it.

    The search must be wide. A beamstop shadow split by a module gap gives a
    seed that can sit tens of pixels off, and a narrow search then locks onto a
    spurious optimum with a handful of pairs while the true centre, with many
    times more, is never visited. Widen ``search_px`` before believing a low
    pair count.

    The null keeps each spot's **radius** and randomises only its azimuth —
    same radial support, so it could match and generally does not. A null that
    cannot produce matches would prove nothing.
    """
    points = np.asarray(points, float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"points must be (N, 2), got {points.shape}")
    if len(points) < 4:
        raise ValueError(f"need at least 4 spots, got {len(points)}")
    if seed is None:
        seed = points.mean(0)
    seed = np.asarray(seed, float)

    offsets = np.arange(-search_px, search_px + 1e-9, step_px)
    best_k, best_c = -1, seed
    for dr in offsets:
        for dc in offsets:
            c = seed + (dr, dc)
            k = int((_match_antipodal(c, points, tol_px).min(1) < tol_px).sum())
            if k > best_k:
                best_k, best_c = k, c

    pairs = antipodal_pairs(points, best_c, tol_px)
    if len(pairs) == 0:
        raise RuntimeError(
            f"no antipodal pairs within {tol_px} px anywhere in a "
            f"+/-{search_px} px search. Widen the search or the tolerance, or "
            "the spot list is not centrosymmetric.")
    mids = 0.5 * (points[pairs[:, 0]] + points[pairs[:, 1]])
    half = 0.5 * np.linalg.norm(points[pairs[:, 0]] - points[pairs[:, 1]], axis=1)
    centre = mids.mean(0)
    sigma = mids.std(0, ddof=1) if len(mids) > 1 else np.zeros(2)

    rng = np.random.default_rng(rng_seed)
    radii = np.linalg.norm(points - centre, axis=1)
    null = np.empty(n_null, int)
    for k in range(n_null):
        th = rng.uniform(0.0, 2.0 * np.pi, len(radii))
        q = np.stack([centre[0] + radii * np.sin(th),
                      centre[1] + radii * np.cos(th)], axis=1)
        null[k] = int((_match_antipodal(centre, q, tol_px).min(1) < tol_px).sum())
    observed = int((_match_antipodal(centre, points, tol_px).min(1) < tol_px).sum())

    return BeamCentreResult(
        row=float(centre[0]), col=float(centre[1]),
        row_sigma=float(sigma[0]), col_sigma=float(sigma[1]),
        n_pairs=len(pairs), pairs=pairs, half_separation_px=half,
        n_matched=observed, null_counts=null,
        p_value=float((null >= observed).mean()))


# ---------------------------------------------------------------------------
# tx
# ---------------------------------------------------------------------------

def predicted_eta(radius_px: float, domega_deg: float, *,
                  wavelength_A: float, lsd_um: float, pixel_um: float
                  ) -> Optional[Tuple[float, float]]:
    """Both branch solutions for a pair's LAB azimuth, in degrees.

    Contains no crystal orientation, no cell and no free parameter — only the
    pair's radius and its ω separation. Returns ``None`` when the pair is
    unusable (Δω ≈ 0, or ρ > |G| which no real pair satisfies).
    """
    q = float(_q_from_radius(radius_px, wavelength_A=wavelength_A,
                             lsd_um=lsd_um, pixel_um=pixel_um))
    c = wavelength_A * q * q / 2.0
    s = np.sin(np.radians(domega_deg) / 2.0)
    if abs(s) < 1e-9:
        return None
    rho = abs(c / s)
    if rho > q:
        return None
    gz = np.sqrt(max(q * q - rho * rho, 0.0))
    ky = rho * np.cos(np.radians(domega_deg) / 2.0)
    return (float(np.degrees(np.arctan2(gz, ky))),
            float(np.degrees(np.arctan2(-gz, ky))))


@dataclass
class TxResult:
    """The detector's in-plane rotation, from Friedel-pair ω splitting."""
    tx_deg: float
    ci_low: float
    ci_high: float
    n_pairs: int
    offsets_deg: np.ndarray
    trimmed_mean_deg: float
    n_trimmed: int

    def __str__(self) -> str:
        return (f"tx = {self.tx_deg:+.3f} deg  95% CI "
                f"[{self.ci_low:+.3f}, {self.ci_high:+.3f}]  "
                f"from {self.n_pairs} pairs "
                f"(trimmed mean {self.trimmed_mean_deg:+.3f} over {self.n_trimmed})")


def tx_from_pairs(pairs: Sequence[Tuple[float, float, float, float, float, float]],
                  centre: Sequence[float], *,
                  wavelength_A: float, lsd_um: float, pixel_um: float,
                  trim_window_deg: float = 20.0,
                  n_bootstrap: int = 2000,
                  rng_seed: int = 5) -> TxResult:
    """Measure tx from the ω splitting of Friedel pairs.

    Parameters
    ----------
    pairs
        One tuple per pair: ``(row1, col1, omega1, row2, col2, omega2)``.
    centre
        Beam centre ``(row, col)`` in the same coordinates as the pairs.

    Returns a :class:`TxResult`. The estimator is the median of signed offsets
    over **both** branches; see the module docstring for why the nearer branch
    must not be chosen. A bootstrap CI and a trimmed mean over the pairs whose
    branch is unambiguous (within ``trim_window_deg``) come along as a
    cross-check — if the two disagree badly, the pair set is not usable.
    """
    cr, cc = float(centre[0]), float(centre[1])
    offsets: List[float] = []
    for (r1, c1, o1, r2, c2, o2) in pairs:
        radius = float(np.hypot(r1 - cr, c1 - cc))
        branches = predicted_eta(radius, o2 - o1, wavelength_A=wavelength_A,
                                 lsd_um=lsd_um, pixel_um=pixel_um)
        if branches is None:
            continue
        eta_obs = float(np.degrees(np.arctan2(r1 - cr, c1 - cc)))
        cand = [((eta_obs - e + 180.0) % 360.0 - 180.0) for e in branches]
        offsets.append(min(cand, key=abs))

    offs = np.asarray(offsets, float)
    if offs.size == 0:
        raise RuntimeError("no usable Friedel pairs — every one had "
                           "d(omega) ~ 0 or rho > |G|")
    median = float(np.median(offs))
    rng = np.random.default_rng(rng_seed)
    boot = np.median(rng.choice(offs, (n_bootstrap, offs.size), replace=True),
                     axis=1)
    lo, hi = (float(v) for v in np.percentile(boot, [2.5, 97.5]))
    inside = offs[np.abs(offs) < trim_window_deg]
    return TxResult(tx_deg=median, ci_low=lo, ci_high=hi, n_pairs=offs.size,
                    offsets_deg=offs,
                    trimmed_mean_deg=float(inside.mean()) if inside.size else float("nan"),
                    n_trimmed=int(inside.size))
