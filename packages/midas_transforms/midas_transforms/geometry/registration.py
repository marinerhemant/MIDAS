"""Is the sample shape actually registered to the diffraction data?

A :class:`~midas_transforms.geometry.SampleShape` read from a tomogram carries
a pixel size, a rotation-axis position, a vertical offset and a handedness —
four numbers supplied by a human, none of them checked by anything. Get any of
them wrong and the reconstruction is still sharp, the mask is still smooth, and
every grain volume downstream is quietly wrong.

Two checks live here, both usable on data that already exists:

**V1 — the illuminated-volume sinogram.** Per-omega ring intensity is
proportional to the lit volume, so the *shape* of that curve is a prediction.
The three failure modes have three different signatures:

===================  ===========================================================
handedness flipped   odd Fourier components flip sign  -> correlation drops
axis offset          injects a one-cycle ``cos w``; its amplitude MEASURES it
pixel size wrong     scales the curve without moving its phase
===================  ===========================================================

**V2 — grain-centroid containment.** Indexed grain centroids must lie inside
the mask. Fit the translation on half the grains, score on the other half;
fitting and scoring on the same set will report ~100 % for any shape big enough
to swallow the cloud.

The thing that makes these worth running
----------------------------------------
Both can be run in a form that *cannot fail*, and then reported as a pass.

V1 on a cylinder is the clearest case: a cylinder's lit volume does not vary
with omega at all, so the predicted curve is flat, any measured curve
"agrees" with it to within noise, and a chi-square test sails through having
tested nothing. :func:`sinogram_check` measures the predicted modulation first
and returns ``NO_POWER`` rather than a verdict.

V2 has the same problem in a subtler form: on a near-symmetric sample a
*mirrored* mask contains the centroids just as well as the correct one. That is
what :meth:`SampleShape.mirrored` and :func:`meta_null` are for — every check
must degrade on the mirrored mask by a stated margin, or it has no power over
the failure it exists to catch.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence

import numpy as np

from .sample_shape import SampleShape

__all__ = [
    "CheckResult",
    "centroid_containment_check",
    "meta_null",
    "sinogram_check",
]


@dataclass
class CheckResult:
    """One registration check. ``verdict`` is the only thing to act on."""

    name: str
    verdict: str                       # PASS | FAIL | NO_POWER
    statistic: float = float("nan")
    detail: Dict[str, Any] = field(default_factory=dict)
    message: str = ""

    def __bool__(self) -> bool:
        # A NO_POWER check is not a pass. Make `if check:` say so.
        return self.verdict == "PASS"


# ------------------------------------------------------------------ V1

def sinogram_check(
    shape: SampleShape,
    omegas_deg,
    measured,
    *,
    beam_width_um: float,
    beam_height_um: Optional[float] = None,
    beam_centre_z_um: float = 0.0,
    min_modulation: float = 0.02,
    min_correlation: float = 0.8,
) -> CheckResult:
    """V1: does per-omega ring intensity track the predicted lit volume?

    ``measured`` is any quantity proportional to the illuminated volume — the
    summed intensity of a ring per omega frame is the usual one. Only its
    *shape* is used; scale and offset are fitted out, because the constant of
    proportionality contains the flux, the detector gain and the structure
    factor and is not knowable here.

    Returns ``NO_POWER`` when the predicted modulation is below
    ``min_modulation``: a shape that does not vary with omega cannot test a
    registration, and reporting a pass there is the failure this whole module
    is written around.
    """
    omegas = np.asarray(omegas_deg, dtype=np.float64).ravel()
    meas = np.asarray(measured, dtype=np.float64).ravel()
    if omegas.shape != meas.shape:
        raise ValueError(
            f"omegas {omegas.shape} and measured {meas.shape} must match"
        )
    if omegas.size < 8:
        raise ValueError(
            f"{omegas.size} omega samples is too few to fit a one-cycle "
            "component and its phase; use at least 8."
        )

    pred = shape.illuminated_volume_sinogram(
        omegas, beam_width_um=beam_width_um, beam_height_um=beam_height_um,
        beam_centre_z_um=beam_centre_z_um,
    )
    if pred.mean() <= 0:
        return CheckResult(
            "V1-sinogram", "FAIL", 0.0,
            {"predicted_mean_um3": float(pred.mean())},
            "the beam lights no sample at any omega - check slice0_z_um and "
            "beam_centre_z_um against the stage position",
        )

    modulation = float(pred.std() / pred.mean())
    detail: Dict[str, Any] = {
        "predicted_modulation": modulation,
        "min_modulation": float(min_modulation),
        "n_omegas": int(omegas.size),
    }

    if modulation < min_modulation:
        return CheckResult(
            "V1-sinogram", "NO_POWER", modulation, detail,
            f"predicted modulation {modulation:.4f} is below {min_modulation}: "
            "this shape's lit volume barely varies with omega (a cylinder on "
            "the rotation axis does not vary at all), so the check cannot "
            "distinguish a correct registration from a wrong one. This is NOT "
            "a pass - use a check with power, or say the registration is "
            "unverified.",
        )

    # Shape-only comparison: centre both, then correlate.
    p = pred - pred.mean()
    m = meas - meas.mean()
    denom = float(np.linalg.norm(p) * np.linalg.norm(m))
    corr = float(p @ m / denom) if denom > 0 else 0.0
    detail["correlation"] = corr

    # The one-cycle component is the axis-offset signature; report its phase so
    # a failure says WHICH way the mask is off rather than only that it is.
    w = np.radians(omegas)
    c1 = float(2.0 * (m @ np.cos(w)) / max(1, m.size))
    s1 = float(2.0 * (m @ np.sin(w)) / max(1, m.size))
    detail["measured_1cycle_amplitude"] = math.hypot(c1, s1)
    detail["measured_1cycle_phase_deg"] = math.degrees(math.atan2(s1, c1))

    if corr >= min_correlation:
        return CheckResult("V1-sinogram", "PASS", corr, detail,
                           f"correlation {corr:.3f} >= {min_correlation}")
    return CheckResult(
        "V1-sinogram", "FAIL", corr, detail,
        f"correlation {corr:.3f} < {min_correlation}. A negative or near-zero "
        "value with a strong one-cycle term is the axis-offset signature; a "
        "sign flip on the odd components is the handedness signature.",
    )


# ------------------------------------------------------------------ V2

def centroid_containment_check(
    shape: SampleShape,
    centroids_um,
    *,
    min_contained: float = 0.98,
    fit_fraction: float = 0.5,
    search_px: int = 3,
    seed: int = 0,
    threshold: float = 0.5,
) -> CheckResult:
    """V2: do indexed grain centroids fall inside the mask?

    A translation is fitted on ``fit_fraction`` of the grains and scored on the
    rest. **The held-out fraction is the result**; the in-sample number is
    reported only so the gap between them is visible, because fitting and
    scoring on the same grains reports ~100 % for any mask large enough to
    contain the cloud.

    ``search_px`` bounds an integer-voxel local search around the
    centroid-difference starting point. It is small on purpose: this check
    verifies a registration, and a translation of many voxels is a
    registration failure to report, not a parameter to fit away.

    **What this check cannot see.** The fit starts from the difference of
    centroids, so a *pure rigid translation* of the whole grain cloud is
    absorbed exactly and always passes — V2 tests the shape of the
    registration, never its origin. It does catch a scale error (a wrong pixel
    size makes the mask the wrong size at every offset). For handedness, run it
    through :func:`meta_null`: on a near-symmetric sample containment scores
    the same on a mirrored mask, and the meta-null is what says so.
    """
    pts = np.asarray(centroids_um, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"centroids_um must be (N, 3); got {pts.shape}")
    n = pts.shape[0]
    if n < 10:
        raise ValueError(
            f"{n} centroids is too few to split into fit and held-out halves"
        )

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_fit = max(1, int(round(fit_fraction * n)))
    fit_idx, held_idx = perm[:n_fit], perm[n_fit:]
    if held_idx.size == 0:
        raise ValueError("fit_fraction leaves no held-out grains")

    # Start from the difference of centroids, then search integer voxel steps.
    occ_pos = shape.voxel_positions_um()[shape.occupancy >= threshold]
    if occ_pos.size == 0:
        return CheckResult("V2-containment", "FAIL", 0.0, {},
                           "the mask is empty at this threshold")
    t0 = occ_pos.mean(axis=0) - pts[fit_idx].mean(axis=0)

    px, pitch = shape.pixel_size_um, shape.slice_pitch_um
    steps = range(-int(search_px), int(search_px) + 1)
    best_t, best_score = t0, -1.0
    for dz in steps:
        for dy in steps:
            for dx in steps:
                t = t0 + np.array([dx * px, dy * px, dz * pitch])
                s = float(shape.contains(pts[fit_idx], threshold=threshold,
                                         translation_um=t).mean())
                if s > best_score:
                    best_score, best_t = s, t

    held = float(shape.contains(pts[held_idx], threshold=threshold,
                                translation_um=best_t).mean())
    detail = {
        "in_sample_contained": best_score,
        "held_out_contained": held,
        "n_fit": int(n_fit), "n_held": int(held_idx.size),
        "translation_um": tuple(float(v) for v in best_t),
        "min_contained": float(min_contained),
    }
    verdict = "PASS" if held >= min_contained else "FAIL"
    return CheckResult(
        "V2-containment", verdict, held, detail,
        f"{100 * held:.1f} % of {held_idx.size} held-out centroids inside "
        f"(in-sample {100 * best_score:.1f} %)",
    )


# ------------------------------------------------------- the meta-null

def meta_null(
    check_fn,
    shape: SampleShape,
    *args,
    min_degradation: float = 0.1,
    **kwargs,
) -> CheckResult:
    """N3: rerun a check on the MIRRORED mask; it must get materially worse.

    A mirrored mask is the failure that matters and the one that is hardest to
    see — it reconstructs perfectly and gives smooth, plausible, wrong path
    lengths. If a check scores the same on the mirror as on the original, the
    check did not test handedness, whatever verdict it returned.

    Returns a ``CheckResult`` whose verdict is:

    * ``PASS`` — the mirror degraded the statistic by at least
      ``min_degradation``, so the original check had power;
    * ``NO_POWER`` — it did not, so the original verdict says nothing about
      handedness;
    * ``FAIL`` — the mirror scored *better*, which means the mask is mirrored.
    """
    real = check_fn(shape, *args, **kwargs)
    mirror = check_fn(shape.mirrored(), *args, **kwargs)
    delta = float(real.statistic - mirror.statistic)
    detail = {
        "real": real.verdict, "real_statistic": real.statistic,
        "mirror": mirror.verdict, "mirror_statistic": mirror.statistic,
        "degradation": delta, "min_degradation": float(min_degradation),
    }

    if delta < -abs(min_degradation):
        return CheckResult(
            f"N3-meta-null[{real.name}]", "FAIL", delta, detail,
            f"the MIRRORED mask scores better ({mirror.statistic:.3f} vs "
            f"{real.statistic:.3f}). The shape's handedness is inverted.",
        )
    if delta < min_degradation:
        return CheckResult(
            f"N3-meta-null[{real.name}]", "NO_POWER", delta, detail,
            f"mirroring the mask changed the statistic by only {delta:.4f} "
            f"(< {min_degradation}), so {real.name} has no power over "
            "handedness on this sample. Its verdict is not evidence that the "
            "in_plane choice is right.",
        )
    return CheckResult(
        f"N3-meta-null[{real.name}]", "PASS", delta, detail,
        f"mirroring degrades {real.name} by {delta:.3f}; the check has power",
    )
