"""The next composition tier above :func:`midas_defect.domains.find_domains`.

``find_domains`` is already "every domain at ONE raster position, in one call" (see its
docstring). Nothing before this module composed a *raster* of positions end to end -- from raw
frames, through the ingest chain, through indexing, through the completeness/honesty gates, to a
reported (and gated) a/b splitting -- or looped that over many positions with the sharding a
large scan needs. The composition here is deliberately thin: every heavy-lifting call is a
package function documented in ``manuals/defect/`` and ``manuals/solve-cell/``; this module wires
them in the validated order and refuses to guess a material.

**Every material parameter is required, with no default anywhere in this file.** Six
``midas_defect.rows`` functions default to one DAC sample's cell (``manuals/solve-cell/
PACKAGE_NOTES.md`` §2); this module passes yours to every one of them, on purpose.

Two things this module does NOT do, on purpose:

* **No per-position self-calibration.** ``midas_defect.selfcal.selfcalibrate_from_crystals`` has
  no convergence guarantee (measured: it cycled between three tilt solutions on one held-out
  position) and would silently move the answer if it ran inside every call here. Run it
  separately, as an explicit, inspected step, if you don't trust the seed geometry.
* **No raster-wide pooling.** Each position is refined against its OWN domain(s) only. Pooling
  domains from many positions into one globally-refined cell is a further, separate analysis
  (``phase-3-refine.md``: "refine several domains jointly when they share a cell") -- deliberately
  left to the caller, so a batch run cannot quietly assume every position shares one crystal.

**Validated against real data (2026-09-12), not just the synthetic suite below.** Run on the
delivered La3Ni2O7 2604 raster position (p=329, 40 frames) and four real neighbours (327, 328,
330, 331) pulled from the same raster:

* Reproduced the ESTABLISHED packaged-code answer at p=329 (``repro/gate_packaged.py``,
  ``CHECKPOINT_ENGINEERING.md``'s own ledger: 51 INDEXED of 72 at d_min 1.10 A, a=3.6124,
  c=19.2512) almost exactly through this DIFFERENT search path (row/pair seeding, not
  ``find_seed_orientation``): 51 claimed reflections, a=3.6121, c=19.2416.
* **p=329 is a genuine 3-domain position, and this module found all three.** The recorded
  ``raster_v6`` output (the project's own ``reduce_v4`` driver) has 3 domains there
  (57/31/19 reflections); this module's own free search found 3 domains that each match one of
  those by orientation to 0.36-1.43 deg (the three cross-pairs sit 17-60 deg apart) -- not a
  coincidence. c agrees to 0.04-0.22% per matched domain across the two independent drivers;
  the matched domains' own claimed-reflection counts differ somewhat (51/24/13 here vs
  57/31/19 there), most likely from ingest threshold/mask differences between this module's
  defaults and the older ad-hoc chain, not from either finding a domain the other invented.
  Two real bugs surfaced getting here and were fixed (see the two commented sections below and
  ``tests/test_raster.py``'s two regression tests for them); neither is visible on the
  homogeneous synthetic scenes the rest of this test file uses.
* This validates the two "does NOT do" choices above, on real data, rather than leaving them as
  untested assumptions:
  - the single shared ``Geometry`` (no self-cal) reproduced c to within 0.05% at FOUR further
    real positions with zero per-position calibration;
  - the lighter-weight ω-sign check (below) correctly and decisively (91:0) picked the
    established sign, once fed real intensity (see the fix) -- it did not need the heavier
    ``resolve_conventions``/``ConventionScan`` machinery this module deliberately avoids.
* What did NOT come for free: the domains themselves are real, but the JOINT a/b split computed
  across all three (``quotable=True``, "1.18% +/- 0.12%, z=9.9") is not trustworthy on that
  basis alone -- ALL THREE have an extreme ``index_asymmetry`` ratio (0.0, inf, inf), the exact
  artifact ``ENVELOPE.md`` #14 already documents on this same sample. `res.notes` flags every
  one of them, but `quotable` alone does not distinguish "three real domains, an untrustworthy
  joint split" from "one real domain, a trustworthy one." Read the notes, always.
* **The raster-wide honesty check itself was run on this real block and its conclusion was
  /verify'd REFUTED (2026-09-12, claim a1c6f1f505cb, 4/4 lenses).** Five real positions
  (327-331) gave "observed spread 0.9511% vs an identical-crystal null of 0.0325%, ~29x --
  a real trend is supported." That conclusion does not survive: the null (built one clean
  synthetic domain per point) returned ZERO domains at 3 of its 5 points, so "0.0325%" is a
  2-point std between near-duplicate synthetic orientations with MILD asymmetry (ratio
  0.18-0.20) -- not a null of the mechanism in question. Separately, every one of the THREE
  real "quotable" points (327, 329, 330), not only 329, carries the same extreme-asymmetry
  domains. And `split_with_error` returns an UNSIGNED magnitude, while the documented artifact
  signature (``ab_splitting.py``'s own docstring) is a CONSISTENT SIGN across domains -- a
  magnitude-spread test cannot see it either way. `06_raster_lattice_batch.ipynb`'s Step 3 null
  now warns when too few null points survive to compare (see its own cell); treat any of its
  "trend supported" conclusions as provisional until the null's own surviving-point count is
  checked, not just its reported spread.
  See `phase-3-refine.md` before trusting a single position's split regardless of `quotable`.
* **A sign-aware successor to the magnitude check exists, but does NOT settle the question
  either -- /verify REFUTED it too (2026-09-13, claim `d93ca3c0a6e2`, 3 SURVIVES, 1 REFUTED).**
  :func:`raster_wide_asymmetry_sign_test` tests the thing the documented artifact actually
  predicts: whether ``sign(a-b)`` correlates with which of ``|h|``, ``|k|`` dominates that SAME
  domain's own reflections (gauge-safe -- no cross-domain a/b convention, no null raster
  needed). On the full historical 2604 raster (621 positions, both the recorded driver and its
  duplicate-check-aligned re-run) it found 65-69% agreement, n=792-1102 domains, p ~ 1e-25 to
  1e-27 -- physics, statistics and reproduction lenses all SURVIVED that math. But the artifact
  lens built a synthetic control with ``a == b`` PLANTED EXACTLY (zero real crystallographic
  splitting), the SAME already-documented index-population skew this raster carries, plus one
  small SHARED (not domain-specific) quadratic radial detector-distortion term, and reproduced
  the observed agreement and significance almost exactly (0.61-0.65, p down to 2e-19) with no
  real physics at all. **The sign test is real diagnostic progress -- it is not blind to the
  documented mechanism the way the magnitude check was -- but on THIS raster it cannot yet
  separate "the ENVELOPE.md #14 index-asymmetry artifact" from "an even more mundane, ordinary
  shared geometric-calibration residual."** Net effect: this raster's a/b story is not
  established as real crystal physics by either check, and the honest reading (the sign
  correlation appearing SO easily under a null of zero splitting) is if anything an argument
  for MORE caution about quoting one, not less. `index_asymmetry`'s own docstring in
  ``ab_splitting.py`` was also corrected: "any radial systematic" is imprecise -- an ISOTROPIC
  (uniform, wavelength-type) error is exactly degenerate with an overall scale change and
  produces NO effect; the mechanism needs a \|q\|- or 2theta-DEPENDENT systematic (detector
  distance, beam centre, or distortion residual), which the artifact lens's control used.
"""
from __future__ import annotations

import dataclasses
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .completeness import CompletenessAudit, audit_completeness, targeted_recovery, window_from_residuals
from .domains import DomainSearch, find_domains
from .geometry import (Geometry, detector_angle_maps, ewald_crossing_omegas, pixel_to_qlab,
                       qlab_to_pixel, qlab_to_qsample, qsample_to_qlab)
from .honesty import decoy_test, feature_in_raw, inflated_cell
from .ingest import build_mask, detect_powder_rings, find_blobs_3d, flag_powder, subtract_background
from .rows import hkl_box_from_geometry
from .seed_index import bootstrap_orientation_uncertainty

__all__ = ["PositionResult", "reduce_one_position", "reduce_raster_block",
          "assemble_raster_results", "omega_sign_check", "OmegaSignCheck",
          "predict_reflections", "raster_wide_asymmetry_sign_test",
          "plot_domain_overlay", "plot_orientation_envelope", "plot_lattice_envelope"]


# ---------------------------------------------------------------------------
# ω-sign check
# ---------------------------------------------------------------------------

@dataclass
class OmegaSignCheck:
    """Which ω sign the domain search prefers at this position, and by how much.

    A lighter-weight substitute for :func:`midas_defect.indexing.resolve_conventions` --
    it reuses :func:`midas_defect.domains.find_domains` itself as the discriminator (this
    module's own validated composition) instead of a separate forward-model pipeline. Reports
    the evidence, not a verdict: use :attr:`decisive`, don't assume it.
    """
    chosen_sign: int
    n_explained: Dict[int, int]           # sign -> spots explained by find_domains there
    min_assigned: int = 8

    @property
    def decisive(self) -> bool:
        n = sorted(self.n_explained.values(), reverse=True)
        return len(n) > 1 and n[0] >= self.min_assigned and n[0] >= 2 * max(n[1], 1)

    def __str__(self) -> str:
        body = ", ".join(f"omega {s:+d}: {n} spots explained" for s, n in self.n_explained.items())
        return (f"omega sign {self.chosen_sign:+d} ({body})"
               f"{'' if self.decisive else '  -- NOT DECISIVE, do not adopt blindly'}")


def omega_sign_check(qlab: np.ndarray, omega_deg: np.ndarray, intensity: np.ndarray,
                     row: np.ndarray, col: np.ndarray, frame: np.ndarray, *,
                     a: float, c: float, space_group_number: int,
                     seedable: Optional[np.ndarray] = None,
                     live: Optional[np.ndarray] = None,
                     omega_signs: Sequence[int] = (1, -1),
                     **find_domains_kwargs) -> OmegaSignCheck:
    """Try both ω signs and see which one :func:`find_domains` actually explains more of.

    ``qlab``/``omega_deg`` are the pixel-derived lab-frame q and the (unsigned) omega reading
    for every spot; the sign multiplies ``omega_deg`` before rotating into the sample frame
    (``qlab_to_qsample`` convention: ``q_sample = R_z(-sign*omega) @ q_lab``). Feed it
    **seedable** spots only (non-powder, non-stationary) -- gasket and anvil reflections can
    out-vote the crystal's own (measured on a real La3Ni2O7 position: -1 at 4:1 from stationary
    spots, +1 at 38:0 from the crystal's own).
    """
    n_explained: Dict[int, int] = {}
    for sign in omega_signs:
        q = qlab_to_qsample(torch.as_tensor(qlab, dtype=torch.float64),
                            torch.deg2rad(torch.as_tensor(sign * np.asarray(omega_deg), dtype=torch.float64))
                            ).detach().cpu().numpy()
        res = find_domains(q, intensity, row, col, frame, a=a, c=c,
                           space_group_number=space_group_number,
                           live=live, seedable=seedable, **find_domains_kwargs)
        n_explained[int(sign)] = int(res.explained.sum()) if res.domains else 0
    chosen = max(n_explained, key=n_explained.get)
    return OmegaSignCheck(chosen_sign=chosen, n_explained=n_explained)


# ---------------------------------------------------------------------------
# one position
# ---------------------------------------------------------------------------

@dataclass
class PositionResult:
    """Every diagnostic at one raster position -- the evidence, not a verdict.

    ``quotable`` distills every gate below into one flag, but every gate's own outcome is kept
    in ``notes`` so a reader can see WHY, not just the answer. Nothing here overrides a failed
    gate; ``quotable`` is False the moment any required gate is.
    """
    point: int
    omega_sign: OmegaSignCheck
    n_spots: int
    ingest_counts: dict
    domains: DomainSearch
    completeness: List[CompletenessAudit]         # one per domain
    recovery: List[Optional[dict]]                # one per domain: targeted_recovery at MISSED sites, or None
    decoy: List[dict]                             # one per domain, from honesty.decoy_test
    feature_in_raw: List[dict]                    # one per domain's brightest reflection
    ab_gates: List[dict]                          # one per domain: ab_separable, partner_multiplicity, ...
    refined_cell: Optional[tuple]                 # (a, b, c, alpha, beta, gamma) or None
    split_pct: Optional[Tuple[float, float, float]]   # (delta, sigma, z) from split_with_error, or None
    notes: List[str] = field(default_factory=list)
    quotable: bool = False
    #: SKETCH, 2026-09-13. **`/verify` REFUTED the claim that this envelope's width is "a real,
    #: differentiating signal of orientation determinability"** (claim b82502d6b5ad: physics,
    #: statistics, and artifact lenses all REFUTED; independent reproduction SURVIVED -- the
    #: numbers themselves are computed correctly, the interpretation was not). Do not present
    #: a sparse-domain envelope as evidence that domain's orientation is "real but uncertain"
    #: without also checking `decoy` for that domain -- a domain the pipeline's own honesty
    #: gate does not trust can show a wide envelope for reasons having nothing to do with
    #: orientation physics (see `bootstrap_orientation_uncertainty`'s docstring for the full
    #: mechanism: generic 1/sqrt(n) bootstrap scaling, non-independent resamples from a tiny
    #: combinatorial pool below ``keep_n``, or -- the bug actually fixed by this claim -- a
    #: silently-tetragonal orientation fit for a domain that is not tetragonal).
    #:
    #: One entry per domain (aligned with ``domains.domains``), from
    #: ``seed_index.bootstrap_orientation_uncertainty``, or ``None`` where not computed
    #: (default: not computed at all, see ``reduce_one_position``'s ``orientation_uncertainty``
    #: flag -- this is NOT free, each entry re-runs the orientation fit
    #: ``n_orientation_boot`` times). Keys: ``pair_angle_mean_deg`` / ``pair_angle_p95_deg`` /
    #: ``pair_angle_max_deg`` (pairwise misorientation across the bootstrap U's), plus
    #: ``U_mean``/``U_bootstraps``/``n_distinct_subsets``/``independent_resamples`` -- READ
    #: ``independent_resamples`` before trusting a spread from a sparse domain.
    #:
    #: ``max`` several times ``mean`` is NOT by itself evidence of a bimodal/non-unimodal
    #: spread -- checked directly on the cleanest synthetic domain available (321 reflections,
    #: textbook single orientation): max/mean still came out ~3.7x, purely from comparing the
    #: MAX of many bootstrap pairs against their MEAN (an order-statistics effect, not a
    #: property of the underlying distribution's shape). To check for a genuinely multi-modal
    #: spread, inspect ``U_bootstraps`` directly rather than reading a ratio off the summary
    #: numbers.
    orientation_envelope: List[Optional[dict]] = field(default_factory=list)
    #: SKETCH, 2026-09-13, not yet independently /verify'd. (n_kept, 6) array of every
    #: bootstrap resample's full expanded cell -- (a, b, c, alpha, beta, gamma) -- from
    #: `midas_hkls.refine_cell_joint`'s own resampling loop, or None when no domain reached
    #: `refine_cell_joint` (``refined_cell is None``) or too few resamples succeeded.
    #: `split_pct`'s sigma is this array's std on columns 0/1; this is the same data before
    #: being collapsed to that one number -- read individual a/b/c ranges from columns
    #: 0/1/2, not just their difference.
    cell_bootstrap_samples: Optional[np.ndarray] = None
    #: (a, b, c, alpha, beta, gamma) bootstrap sigma, i.e. ``cell_bootstrap_samples.std(axis=0,
    #: ddof=1)`` -- the compact, per-parameter summary of the same data, kept separately
    #: because it (not the full array) is what belongs in a raster-wide map: cheap to carry
    #: through JSON for hundreds of positions, where the full bootstrap array is not (see
    #: ``to_dict()``, which drops ``cell_bootstrap_samples`` but keeps this).
    cell_sigma_bootstrap: Optional[Tuple[float, ...]] = None

    def summary(self) -> str:
        lines = [f"position {self.point}: {self.n_spots} spots, {self.omega_sign}",
                f"{len(self.domains.domains)} domain(s) found"]
        for i, (aud, dec, fr, gate) in enumerate(zip(self.completeness, self.decoy,
                                                     self.feature_in_raw, self.ab_gates)):
            lines.append(f"  domain {i}: completeness {aud.counts}; "
                        f"honesty {dec['verdict']}; feature-in-raw {fr['verdict']}; "
                        f"ab gate: {gate.get('reason', gate.get('stage'))}")
        for i, rec in enumerate(self.recovery):
            if rec is not None:
                lines.append(f"  domain {i}: targeted recovery {rec['n_recovered']}/"
                            f"{rec['n_scored']} of {rec['n_missed']} missed sites "
                            f"(excess {rec['excess_sigma']:.1f} sigma over the same-ring null)")
        if self.split_pct is not None:
            d, s, z = self.split_pct
            lines.append(f"a/b split: {d:.3g} % +/- {s:.3g} % (z={z:.2g})")
        lines.append(f"quotable: {self.quotable}")
        for n in self.notes:
            lines.append(f"note: {n}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["domains"] = [
            dict(branch=dom.branch, seeded=dom.seeded, seed_source=dom.seed_source,
                n_claim=int(dom.claim.sum()), hkl=dom.hkl.tolist(), U=dom.U.tolist(),
                a=float(dom.lat.a), b=float(dom.lat.b), c=float(dom.lat.c))
            for dom in self.domains.domains
        ]
        d["completeness"] = [dict(counts=a.counts, window_px=a.window_px,
                                 window_omega_deg=a.window_omega_deg)
                             for a in self.completeness]
        d["omega_sign"] = dict(chosen_sign=self.omega_sign.chosen_sign,
                              n_explained=self.omega_sign.n_explained,
                              decisive=self.omega_sign.decisive)
        # orientation_envelope/cell_bootstrap_samples carry raw bootstrap arrays (U_bootstraps:
        # n_domains * n_orientation_boot * 3x3 floats; cell_bootstrap_samples: n_bootstrap * 6
        # floats) meant for one position's own live plotting (05), not for a whole raster's
        # worth of JSON files (06/reduce_raster_block) -- keep only the compact summaries a
        # raster map actually needs; U_mean/cell_bootstrap_samples stay in-memory-only.
        d["orientation_envelope"] = [
            None if oe is None else
            dict(U_mean=oe["U_mean"].tolist(), pair_angle_mean_deg=oe["pair_angle_mean_deg"],
                pair_angle_p95_deg=oe["pair_angle_p95_deg"], pair_angle_max_deg=oe["pair_angle_max_deg"],
                n_boot=oe["n_boot"], keep_n=oe["keep_n"], n_distinct_subsets=oe["n_distinct_subsets"],
                independent_resamples=oe["independent_resamples"])
            for oe in self.orientation_envelope
        ]
        d["cell_bootstrap_samples"] = None
        return d

    def save(self, path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=1, default=str))


def _ingest_position(frames: np.ndarray, geom: Geometry, *,
                     mask_low_count_threshold: float = 20.0,
                     blob_threshold_sigma: float = 8.0,
                     blob_min_vol: int = 10, gap_bridge: int = 21, split_ratio: float = 3.0):
    """The ``phase-1-ingest.md`` chain, copied, not reconstructed. Returns (spots, q, omega_deg, mask, counts)."""
    tth, az = detector_angle_maps(geom)
    m = build_mask(frames, low_count_threshold=mask_low_count_threshold)
    mask = m.mask if hasattr(m, "mask") else m
    sub = subtract_background(frames, tth, az, mask)
    noise = 1.4826 * float(np.median(np.abs(sub - np.median(sub))))
    threshold = max(blob_threshold_sigma * noise, mask_low_count_threshold)
    spots, counts = find_blobs_3d(sub, mask, threshold=threshold, min_vol=blob_min_vol,
                                  split_ratio=split_ratio, gap_bridge=gap_bridge,
                                  return_counts=True)
    spots = spots[spots.n_frames >= 2].reset_index(drop=True)
    if len(spots) == 0:
        return spots, torch.zeros((0, 3)), np.zeros(0), mask, sub, counts
    omega_deg = geom.omega_first_deg + geom.omega_step_deg * spots.frame.values
    qlab = pixel_to_qlab(spots.row.values.astype(float), spots.col.values.astype(float),
                         geom, device="cpu")
    return spots, qlab, omega_deg, mask, sub, counts


def _hkl_box(hmax: int, kmax: int, lmax: int, space_group_number: int) -> np.ndarray:
    from midas_hkls import SpaceGroup, centring_allowed
    h = np.arange(-hmax, hmax + 1)
    k = np.arange(-kmax, kmax + 1)
    l = np.arange(-lmax, lmax + 1)
    H, K, L = np.meshgrid(h, k, l, indexing="ij")
    hkl = np.stack([H.ravel(), K.ravel(), L.ravel()], axis=-1)
    hkl = hkl[np.any(hkl != 0, axis=1)]
    sg = SpaceGroup.from_number(int(space_group_number))
    return hkl[centring_allowed(hkl, sg)]


def predict_reflections(U: np.ndarray, B: np.ndarray, hkl: np.ndarray, geom: Geometry,
                         omega_sign: int):
    """Predict (row, col, omega_deg) for every ``hkl`` under (U, B), restricted to the measured
    ω range and the detector.

    Returns ``(row, col, omega_deg, hkl_kept, kept_idx)``: one entry per reflection that is
    actually observable, in the same relative order as ``hkl``, plus ``kept_idx`` -- the index
    into the ORIGINAL ``hkl`` array each entry came from, so a caller can select the matching
    observed data (``some_array[kept_idx]``) rather than assuming every input was kept.
    """
    hkl = np.asarray(hkl, float)
    q_sample = (U @ B @ hkl.T).T
    omega_lo = math.radians(geom.omega_first_deg - 0.5 * geom.omega_step_deg)
    omega_hi = math.radians(geom.omega_first_deg + (geom.n_frames - 0.5) * geom.omega_step_deg)
    if omega_lo > omega_hi:
        omega_lo, omega_hi = omega_hi, omega_lo

    rows, cols, omegas, kept = [], [], [], []
    for i in range(len(hkl)):
        qs = q_sample[i]
        if np.linalg.norm(qs) < 1e-9:
            continue
        for w in ewald_crossing_omegas(qs, geom.wavelength_A):
            for n in (-1, 0, 1):
                ww = w + 2 * math.pi * n
                w_reported = omega_sign * ww
                if not (omega_lo <= w_reported <= omega_hi):
                    continue
                qlab = qsample_to_qlab(torch.as_tensor(qs, dtype=torch.float64), ww)
                try:
                    row, col = qlab_to_pixel(qlab.reshape(1, 3), geom, device="cpu")
                except RuntimeError:
                    continue
                r, c = float(row[0]), float(col[0])
                if not (np.isfinite(r) and np.isfinite(c) and 0 <= r < geom.n_pix_z and 0 <= c < geom.n_pix_y):
                    continue
                rows.append(r); cols.append(c); omegas.append(math.degrees(w_reported))
                kept.append(i)
                break                                       # one crossing per hkl is enough here
            else:
                continue
            break
    if not rows:
        return (np.zeros(0), np.zeros(0), np.zeros(0), np.zeros((0, 3), int), np.zeros(0, int))
    kept_idx = np.asarray(kept, int)
    return (np.asarray(rows), np.asarray(cols), np.asarray(omegas),
           hkl[kept_idx].astype(int), kept_idx)


# ---------------------------------------------------------------------------
# Reporting plots -- SKETCH, 2026-09-13. matplotlib is the package's optional `viz`
# extra, imported lazily inside each function so it stays optional at module import time.
# ---------------------------------------------------------------------------

def plot_domain_overlay(frames: np.ndarray, res: "PositionResult", geom: Geometry, *, ax=None):
    """The max-intensity projection over ``frames``, with every domain's own claimed
    reflections predicted back onto it (same discipline DFXM calls "look at the curves
    before the summary"). One color per domain, labelled with its `decoy_test` verdict --
    a domain whose own honesty gate failed should look suspect here too, not just in a table.
    """
    import matplotlib.pyplot as plt
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(7.5, 6.6))
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(res.domains.domains), 1)))
    im = ax.imshow(np.log10(np.clip(frames.max(axis=0), 1, None)), cmap="magma")
    if fig is not None:
        fig.colorbar(im, ax=ax, fraction=0.046, label="log10(max over frames)")
    for i, dom in enumerate(res.domains.domains):
        pr, pc, _pw, _phkl, _kept = predict_reflections(dom.U, dom.lat.B, dom.hkl, geom,
                                                        res.omega_sign.chosen_sign)
        ax.scatter(pc, pr, s=32, facecolors="none", edgecolors=[colors[i]], linewidths=1.3,
                  label=f"domain {i} ({len(pr)} claimed, decoy={res.decoy[i]['verdict']})")
    ax.set_title(f"position {res.point}: {len(res.domains.domains)} domain(s) found")
    ax.set_xlabel("column"); ax.set_ylabel("row")
    if res.domains.domains:
        ax.legend(fontsize=8, frameon=False, loc="lower right")
    return fig


def plot_orientation_envelope(res: "PositionResult"):
    """Per domain, a histogram of each bootstrap ``U``'s misorientation from the bootstrap
    mean -- the visual form of ``PositionResult.orientation_envelope``.

    **Read this plot, not just the summary numbers, before trusting a sparse domain's
    envelope.** `/verify` REFUTED an early version of this envelope as a general
    "orientation determinability signal" (claim ``b82502d6b5ad``) partly because a domain
    resampled from too few reflections draws from a tiny combinatorial pool
    (``n_distinct_subsets``) and can show a visibly LUMPY, non-smooth, even bimodal
    histogram rather than a genuine continuous spread -- exactly the kind of thing a summary
    mean/p95/max cannot show but a histogram does immediately. Domains with
    ``orientation_envelope[i] is None`` (fewer than 4 claimed reflections) are skipped.
    """
    import matplotlib.pyplot as plt
    n_dom = len(res.domains.domains)
    fig, axes = plt.subplots(1, max(n_dom, 1), figsize=(5.5 * max(n_dom, 1), 4.2), squeeze=False)
    axes = axes[0]
    colors = plt.cm.tab10(np.linspace(0, 1, max(n_dom, 1)))
    for i in range(n_dom):
        ax = axes[i]
        oe = res.orientation_envelope[i] if i < len(res.orientation_envelope) else None
        dom = res.domains.domains[i]
        if oe is None:
            ax.text(0.5, 0.5, "not computed\n(<4 claimed reflections, or\n"
                              "orientation_uncertainty=False)", ha="center", va="center",
                   transform=ax.transAxes, fontsize=9)
            ax.set_title(f"domain {i}"); ax.set_xticks([]); ax.set_yticks([])
            continue
        U_mean = oe["U_mean"]
        angles = []
        for Ub in oe["U_bootstraps"]:
            delta = Ub @ np.asarray(U_mean).T
            c = np.clip((np.trace(delta) - 1) / 2, -1, 1)
            angles.append(np.degrees(np.arccos(c)))
        ax.hist(angles, bins=min(12, max(len(angles) // 2, 3)), color=colors[i], edgecolor="k", alpha=0.85)
        ax.axvline(oe["pair_angle_mean_deg"], color="k", ls="--", lw=1,
                  label=f"mean pairwise = {oe['pair_angle_mean_deg']:.3f} deg")
        ax.set_xlabel("misorientation vs bootstrap mean U (deg)")
        ax.set_ylabel(f"count (of {oe['n_boot']} resamples)")
        indep = "independent" if oe["independent_resamples"] else "NOT independent (too few reflections)"
        ax.set_title(f"domain {i}: {len(dom.hkl)} claimed refl., keep_n={oe['keep_n']}\n"
                    f"decoy={res.decoy[i]['verdict']}, resamples {indep}", fontsize=9)
        ax.legend(fontsize=8, frameon=False)
    fig.suptitle(f"Orientation envelope -- position {res.point} "
                f"(/verify claim b82502d6b5ad: REFUTED as a general determinability signal; "
                f"mechanism correctness-fixed, read decoy_test alongside this)", fontsize=9)
    fig.tight_layout()
    return fig


def plot_lattice_envelope(res: "PositionResult"):
    """``a``, ``b``, ``c`` bootstrap distributions, individually -- NOT just their difference.

    Needs ``res.cell_bootstrap_samples``, which is only populated in-memory by
    :func:`reduce_one_position` (never persisted through :meth:`PositionResult.to_dict`,
    since it is one position's full raw bootstrap array, not something a whole raster's
    JSON files should carry). Call this on a live ``reduce_one_position`` result, not on
    something read back via :func:`assemble_raster_results`.
    """
    import matplotlib.pyplot as plt
    if res.cell_bootstrap_samples is None:
        raise ValueError(
            "res.cell_bootstrap_samples is None -- either no domain reached refine_cell_joint "
            "(res.refined_cell is None), too few bootstrap resamples succeeded, or this "
            "PositionResult came from assemble_raster_results (which does not carry it; "
            "see this function's docstring)."
        )
    samples = res.cell_bootstrap_samples
    a_fit, b_fit, c_fit = res.refined_cell[0], res.refined_cell[1], res.refined_cell[2]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for j, (lab, fit_val) in enumerate(zip(["a", "b", "c"], [a_fit, b_fit, c_fit])):
        ax = axes[j]
        vals = samples[:, j]
        ax.hist(vals, bins=15, color="steelblue", edgecolor="k", alpha=0.85)
        ax.axvline(fit_val, color="crimson", lw=1.5, label=f"joint fit: {fit_val:.5f} A")
        lo, hi = np.percentile(vals, [2.5, 97.5])
        ax.axvspan(lo, hi, color="crimson", alpha=0.08, label=f"95% range: [{lo:.5f}, {hi:.5f}]")
        ax.set_xlabel(f"{lab} (A)")
        ax.set_ylabel(f"count (of {len(vals)} bootstrap resamples)")
        ax.set_title(f"{lab}: {fit_val:.5f} +/- {vals.std(ddof=1):.5f} A", fontsize=10)
        ax.legend(fontsize=7, frameon=False)
    fig.suptitle(f"Lattice envelope -- position {res.point}, joint fit over "
                f"{len(res.domains.domains)} domain(s) (quotable={res.quotable})", fontsize=10)
    fig.tight_layout()
    return fig


def reduce_one_position(
    frames: np.ndarray, geom: Geometry, *,
    a: float, c: float, space_group_number: int,
    sigma_rtn: Tuple[float, float, float],
    point: int = 0,
    stationary_row: Optional[np.ndarray] = None,
    stationary_col: Optional[np.ndarray] = None,
    stationary_tol_px: float = 3.0,
    tth_max_deg: Optional[float] = None,
    decoy_fractions: Sequence[float] = (0.03, -0.03),
    n_bootstrap: int = 500,
    system: str = "orthorhombic",
    seed_from_nominal: bool = False,
    ingest_kwargs: Optional[dict] = None,
    find_domains_kwargs: Optional[dict] = None,
    orientation_uncertainty: bool = False,
    n_orientation_boot: int = 25,
) -> PositionResult:
    """One raster position, start to finish: frames -> a gated a/b splitting, if one is possible.

    ``frames`` is one position's raw ``(n_frames, n_rows, n_cols)`` stack (dark already
    subtracted if you have a dark; see ``phase-1-ingest.md``). It must already exclude any
    dead/shutter-ramp frames (``midas_defect.ingest.live_frames``) -- this function maps frame
    index to ω as ``geom.omega_first_deg + geom.omega_step_deg * frame`` directly, which is only
    correct for a stack with no frames dropped; a caller whose raw stack has dead frames must
    drop them (and remap ``geom`` accordingly) before calling this. ``a``, ``c`` and
    ``space_group_number`` are the material -- there is no default. ``sigma_rtn`` is YOUR
    sample's measured (radial, transverse, normal) residual budget (``manuals/solve-cell/
    phase-3-refine.md``): the package default is one DAC sample's, not universal.

    ``stationary_row``/``stationary_col`` mark detector cells that see a gasket/anvil
    reflection at every position (mark them not-seedable, keep them live -- deleting them can
    remove a faint crystallite's own reflections that happen to sit nearby, see
    ``phase-2-index.md``).

    Nothing here decides FOR you whether a splitting is real. ``PositionResult.quotable`` is
    only set once every gate in ``phase-3-refine.md``'s order has been checked; read
    ``PositionResult.notes`` for which one, if any, said no.

    ``orientation_uncertainty`` (SKETCH, 2026-09-13, off by default): also bootstrap each
    domain's OWN orientation fit (``seed_index.bootstrap_orientation_uncertainty``,
    ``n_orientation_boot`` resamples each) and report the result in
    ``PositionResult.orientation_envelope`` -- the same "don't report a bare point estimate"
    discipline ``split_with_error`` already applies to the cell, extended to ``U``. This is
    NOT free: it re-runs an Adam refinement ``n_orientation_boot`` times per domain, on top of
    everything else this function already does, so it defaults to off for raster-wide batch
    runs where that cost multiplies by every position. Independent of the a/b gate chain --
    computed for every domain FOUND, not only ones that reach ``refine_cell_joint``, since
    orientation determinability is a property of a domain's own indexing fit, not of whether
    its a/b split is separable.
    """
    ik = dict(ingest_kwargs or {})
    fk = dict(find_domains_kwargs or {})
    notes: List[str] = []

    spots, qlab, omega_deg, mask, sub, ingest_counts = _ingest_position(frames, geom, **ik)
    n_spots = len(spots)
    if n_spots == 0:
        notes.append("ingest found no spots with >= 2 frames; nothing to index")
        empty = DomainSearch(domains=[], explained=np.zeros(0, bool), null_threshold=None, n_rows_tried=0)
        sign = OmegaSignCheck(chosen_sign=1, n_explained={1: 0})
        return PositionResult(point=point, omega_sign=sign, n_spots=0, ingest_counts=ingest_counts,
                             domains=empty, completeness=[], recovery=[], decoy=[], feature_in_raw=[],
                             ab_gates=[], refined_cell=None, split_pct=None, notes=notes, quotable=False)

    # ring separation: azimuthally-uniform intensity is not a crystal reflection.
    # Detect rings on the UNSUBTRACTED median, never on the background-subtracted stack: the
    # polar background removes powder rings BY DESIGN (that is its job), so ring detection on
    # `sub` finds only what the background step missed. Measured on a real, heavily gasket/anvil
    # -contaminated 2604 position (2026-09-12): 6 rings from `sub.max(axis=0)` flagged 2.2 % of
    # spots as powder; 59 rings from the unsubtracted median flagged 21.8 %, and only the latter
    # left `find_domains` able to seed a domain at all (the crystal's own reflections were
    # otherwise swamped by un-flagged gasket/anvil spots in the row-seed pool). Same lesson as
    # `manuals/solve-cell/phase-4-phase-id.md`'s pressure-gauge rings, generalized: it isn't a
    # phase-4-only trap, it is a background-subtraction property.
    tth, az = detector_angle_maps(geom)
    qmag = np.linalg.norm(qlab.detach().cpu().numpy(), axis=1)
    spot_tth = np.degrees(2 * np.arcsin(np.clip(qmag * geom.wavelength_A / (4 * math.pi), -1, 1)))
    spot_eta = np.degrees(np.arctan2(spots.row.values - geom.bcz_px, spots.col.values - geom.bcy_px))
    spot_rad = np.hypot(spots.row.values - geom.bcz_px, spots.col.values - geom.bcy_px)
    rings = detect_powder_rings(np.median(frames, axis=0), tth, mask, azimuth_deg=az)
    powder = flag_powder(spot_tth, spot_eta, spot_rad, rings)

    stationary = np.zeros(n_spots, bool)
    if stationary_row is not None and len(stationary_row):
        d = np.hypot(spots.row.values[:, None] - np.asarray(stationary_row)[None, :],
                    spots.col.values[:, None] - np.asarray(stationary_col)[None, :])
        stationary = (d.min(axis=1) <= stationary_tol_px)

    live = ~powder
    seedable = live & ~stationary

    # `find_lattice_rows` ranks its seed pool by THIS array, descending (`rows.py`:
    # `order = argsort(-I)[:n_seed]`) -- it must be actual brightness, not a blob's frame span.
    # Using `n_frames` here (measured on this real 2604 position, 2026-09-12) let broad, dim,
    # many-frame gasket/anvil debris outrank the crystal's own genuinely bright reflections in
    # the top-`n_seed` pool, and `find_domains` found nothing at either omega sign; switching to
    # `spots.integrated` (matching every other real caller: `repro/gate_intree.py`,
    # `manuals/defect/phase-2-index.md`'s own contract) is what let it seed correctly.
    intensity = spots.integrated.values.astype(float)

    sign = omega_sign_check(qlab.detach().cpu().numpy(), omega_deg, intensity,
                            spots.row.values, spots.col.values, spots.frame.values,
                            a=a, c=c, space_group_number=space_group_number,
                            seedable=seedable, live=live,
                            sigma_rtn=sigma_rtn, seed_from_nominal=seed_from_nominal, **fk)
    if not sign.decisive:
        notes.append(f"omega-sign check NOT decisive ({sign}); proceeding with "
                    f"sign {sign.chosen_sign:+d}, the more-explained option, but do not trust "
                    "this position's answer on that basis alone")

    q = qlab_to_qsample(qlab, torch.deg2rad(torch.as_tensor(
        sign.chosen_sign * omega_deg, dtype=qlab.dtype))).detach().cpu().numpy()

    domains = find_domains(q, intensity, spots.row.values,
                           spots.col.values, spots.frame.values, a=a, c=c,
                           space_group_number=space_group_number, live=live, seedable=seedable,
                           sigma_rtn=sigma_rtn, seed_from_nominal=seed_from_nominal, **fk)
    if not domains.domains:
        notes.append("find_domains returned no domains at this position")

    if tth_max_deg is None:
        tth_max_deg = float(np.nanpercentile(spot_tth[np.isfinite(spot_tth)], 95)) if n_spots else 20.0
    hmax, lmax = hkl_box_from_geometry(a, c, wavelength_A=geom.wavelength_A, tth_max_deg=tth_max_deg)

    completeness_list: List[CompletenessAudit] = []
    recovery_list: List[Optional[dict]] = []
    decoy_list: List[dict] = []
    feature_list: List[dict] = []
    gate_list: List[dict] = []
    orientation_list: List[Optional[dict]] = []
    fit_domains = []

    hkl_box = _hkl_box(hmax, hmax, lmax, space_group_number)
    live_row, live_col, live_frame = spots.row.values[live], spots.col.values[live], spots.frame.values[live]
    live_omega = geom.omega_first_deg + geom.omega_step_deg * live_frame

    for dom in domains.domains:
        claim_idx = np.flatnonzero(dom.claim)

        # SKETCH, 2026-09-13. /verify REFUTED an early version of this on real La3Ni2O7
        # 2601_LT p364 data (claim b82502d6b5ad, physics+statistics+artifact lenses; numbers
        # reproduced exactly, REPRODUCTION survived -- the interpretation did not):
        #  - physics: the FIRST version called bootstrap_orientation_uncertainty with only
        #    (a, c), silently assuming h and k share one axis (tetragonal) -- wrong for any
        #    domain this project's own a/b work has already found is NOT tetragonal. Fixed:
        #    dom.lat.b is now passed explicitly.
        #  - statistics/artifact: a sparse domain's "wider envelope" is not automatically a
        #    meaningful determinability signal -- it can be (a) the generic 1/sqrt(n) scaling
        #    any bootstrap M-estimator shows regardless of data quality, compounded by (b) too
        #    few centroids to draw independent resamples at all (see
        #    `independent_resamples`/`n_distinct_subsets` in the returned dict), and (c) the
        #    domain itself may be one the pipeline's OWN honesty gates (decoy_test) do not
        #    trust -- check `res.decoy[i]['verdict']` alongside this envelope, not instead of
        #    it, before reading a wide spread as "this domain's orientation is real but
        #    uncertain" rather than "this domain's fit may not be trustworthy at all."
        # Off by default (orientation_uncertainty=False) -- see reduce_one_position's
        # docstring for the cost tradeoff. Computed for every domain FOUND, independent of the
        # a/b gate chain below (a domain's orientation determinability is not the same
        # question as whether its a/b split is separable) -- but NOT independent of whether
        # the domain's own fit passes `decoy_test`; read both together.
        if orientation_uncertainty:
            centroids = list(zip((tuple(int(x) for x in row) for row in dom.hkl), q[claim_idx]))
            if len(centroids) >= 4:
                orientation_list.append(bootstrap_orientation_uncertainty(
                    centroids, dom.U, a=float(dom.lat.a), b=float(dom.lat.b), c=float(dom.lat.c),
                    n_boot=n_orientation_boot))
            else:
                orientation_list.append(None)   # bootstrap_orientation_uncertainty floors keep_n at 4
        else:
            orientation_list.append(None)

        # completeness: predict this domain's own cell/orientation over the WHOLE box, and
        # separately over its own claimed hkl (to derive the window from its own residuals --
        # never guessed, per `window_from_residuals`).
        cr, cc, cw, _ckept_hkl, ckept_idx = predict_reflections(
            dom.U, dom.lat.B, dom.hkl, geom, sign.chosen_sign)
        if len(ckept_idx):
            obs_row_c = spots.row.values[claim_idx][ckept_idx]
            obs_col_c = spots.col.values[claim_idx][ckept_idx]
            obs_omega_c = omega_deg[claim_idx][ckept_idx]
            res_px = np.hypot(cr - obs_row_c, cc - obs_col_c)
            res_om = np.abs(cw - obs_omega_c)
            window_px, window_omega_deg = window_from_residuals(res_px, res_om)
        else:
            window_px, window_omega_deg = 3.0, 2.0 * geom.omega_step_deg

        pr, pc, pw, phkl, _pkept_idx = predict_reflections(
            dom.U, dom.lat.B, hkl_box, geom, sign.chosen_sign)
        aud = audit_completeness(predicted_hkl=phkl, predicted_row=pr, predicted_col=pc,
                                 predicted_omega_deg=pw, observed_row=live_row,
                                 observed_col=live_col, observed_omega_deg=live_omega,
                                 assigned_hkl=[tuple(h) for h in dom.hkl],
                                 mask=mask, window_px=max(window_px, 1.0),
                                 window_omega_deg=max(window_omega_deg, geom.omega_step_deg))
        completeness_list.append(aud)

        # Targeted extraction at the sites blind blob-finding MISSED, against a same-ring null
        # (`manuals/defect/phase-2-index.md`) -- the weaker of the two nulls the method needs,
        # so read `excess_sigma`, never the raw recovered count.
        if aud.missed:
            miss_row = np.array([m["row"] for m in aud.missed])
            miss_col = np.array([m["col"] for m in aud.missed])
            miss_omega = np.array([m["omega"] for m in aud.missed])
            miss_frame = np.rint((miss_omega - geom.omega_first_deg) / geom.omega_step_deg).astype(int)
            in_range = (miss_frame >= 0) & (miss_frame < geom.n_frames)
            if in_range.any():
                rec = targeted_recovery(sub, frames, mask, miss_frame[in_range], miss_row[in_range],
                                        miss_col[in_range], beam_centre=(geom.bcz_px, geom.bcy_px))
                recovery_list.append(dict(n_missed=len(aud.missed), n_scored=rec.n_scored,
                                         n_recovered=rec.n_recovered, excess_sigma=rec.excess_sigma))
            else:
                recovery_list.append(None)
        else:
            recovery_list.append(None)

        # The "real" model for this test is the domain's OWN fitted (a, b, c), not the caller's
        # nominal declared cell -- decoys are deliberately-wrong PERTURBATIONS of the answer
        # under test (honesty.decoy_test's own example: "a decoy cell with a inflated 3 %").
        # `match_mask` cannot express an independent b (its `a=`/`c=` build a tetragonal b=a
        # cell), so score directly against this domain's own claimed reflections instead: does
        # a candidate (a, b, c) put q within tolerance of what was actually claimed.
        fit_cell = (float(dom.lat.a), float(dom.lat.b), float(dom.lat.c))
        claim_hkl = dom.hkl
        claim_q = q[claim_idx]
        tol_q_decoy = 3.0 * float(np.linalg.norm(sigma_rtn))

        def score_of(cell, hkl_=claim_hkl, target_q=claim_q, tol=tol_q_decoy, U=dom.U):
            B = np.diag([2 * math.pi / cell[0], 2 * math.pi / cell[1], 2 * math.pi / cell[2]])
            pred = (U @ B @ hkl_.T).T
            resid = np.linalg.norm(pred - target_q, axis=1)
            return int(np.count_nonzero(resid < tol))

        decoys = {f"{'+' if f >= 0 else ''}{f:.0%} on a": inflated_cell(fit_cell, f, axes=(0,))
                 for f in decoy_fractions}
        decoys.update({f"{'+' if f >= 0 else ''}{f:.0%} on b": inflated_cell(fit_cell, f, axes=(1,))
                      for f in decoy_fractions})
        decoys.update({f"{'+' if f >= 0 else ''}{f:.0%} on c": inflated_cell(fit_cell, f, axes=(2,))
                      for f in decoy_fractions})
        dt = decoy_test(score_of, fit_cell, decoys, threshold=int(0.95 * len(claim_hkl)))
        decoy_list.append(dt)

        # The brightest claimed reflection, checked against the RAW frames -- is it a real
        # feature, or something background subtraction created/erased? This is a simplified
        # use of `feature_in_raw`: a short axis-aligned path through the spot's own centroid,
        # not a fitted streak direction (the project's own attempt at this, step55_raw_check.py,
        # never landed a canonical call site either -- see `honesty.py`'s module note).
        if len(claim_idx):
            j = claim_idx[int(np.argmax(spots.integrated.values[claim_idx]))]
            r0, c0 = float(spots.row.values[j]), float(spots.col.values[j])
            path_r = np.array([r0 - 2, r0, r0 + 2])
            path_c = np.array([c0, c0, c0])
            perp_r = np.zeros(3); perp_c = np.ones(3)
            fr = feature_in_raw(frames.max(axis=0), sub.max(axis=0), path_r, path_c, perp_r, perp_c)
        else:
            fr = dict(verdict="absent", in_raw=False, in_processed=False)
        feature_list.append(fr)

        hkl = dom.hkl
        from midas_hkls import ab_separable, partner_multiplicity, index_asymmetry, shear_separable
        gate: dict = {"stage": None, "reason": None}
        if not ab_separable(hkl):
            gate.update(stage="ab_separable", reason="rank < 2: this domain cannot separate a from b")
        else:
            pm = partner_multiplicity(hkl)
            if pm and all(min(nf, nr) <= 1 for nf, nr in pm.values()):
                gate.update(stage="partner_multiplicity",
                          reason=f"every a/b pair rests on <= 1 spot on one side: {pm}")
            else:
                asym = index_asymmetry(hkl)
                gate.update(stage="index_asymmetry", reason=str(asym))
                gate["passed_to_refine"] = True
                ratio = asym.get("ratio")
                if ratio is not None and (ratio < 0.2 or ratio > 5.0):
                    notes.append(
                        f"domain {len(gate_list)}: index_asymmetry ratio {ratio:.3g} is far from "
                        "1 (measured 5x on the 2604 raster was the signature of a radial-systematic "
                        "artifact, not a real splitting). A SINGLE position cannot rule this out -- "
                        "'a common sign across domains is physically impossible' only becomes "
                        "checkable with several positions/domains; see the raster-wide check.")
        gate["shear_separable"] = bool(shear_separable(hkl))
        gate_list.append(gate)
        if gate.get("passed_to_refine"):
            fit_domains.append(dom)

    refined_cell = None
    split_pct = None
    cell_bootstrap_samples = None
    cell_sigma_bootstrap = None
    if fit_domains:
        from midas_hkls import DomainData, refine_cell_joint, split_with_error
        dd = [DomainData(hkl=d.hkl, g=q[np.flatnonzero(d.claim)], label=f"domain{i}")
             for i, d in enumerate(fit_domains)]
        cell0 = (a, a, c, 90.0, 90.0, 90.0)
        fit = refine_cell_joint(dd, system=system, cell0=cell0, two_pi=True,
                               n_bootstrap=n_bootstrap)
        refined_cell = fit.cell
        split_pct = split_with_error(fit)
        cell_bootstrap_samples = fit.cell_bootstrap_samples   # (n_kept, 6): a,b,c,alpha,beta,gamma
        cell_sigma_bootstrap = fit.cell_sigma_bootstrap
    else:
        notes.append("no domain passed the a/b gate chain; no cell was jointly refined")

    fit_passed_decoy = all(decoy_list[i]["verdict"] == "informative"
                          for i, g in enumerate(gate_list) if g.get("passed_to_refine"))
    quotable = (
        bool(fit_domains)
        and fit_passed_decoy
        and split_pct is not None
        and np.isfinite(split_pct[1])
    )
    if fit_domains and not fit_passed_decoy:
        failing_verdicts = sorted({decoy_list[i]["verdict"]
                                  for i, g in enumerate(gate_list) if g.get("passed_to_refine")}
                                 - {"informative"})
        if "real_fails" in failing_verdicts:
            notes.append("a domain that passed the a/b gate chain did not pass the decoy test "
                        "(the real fit itself did not clear the acceptance threshold) -- "
                        "not quotable")
        if "uninformative" in failing_verdicts:
            notes.append("a domain that passed the a/b gate chain did not pass the decoy test "
                        "(a deliberately wrong cell scored as well) -- not quotable")
    if split_pct is not None and not np.isfinite(split_pct[1]):
        notes.append("a and b are equal by the refined symmetry: no splitting is being measured")

    return PositionResult(point=point, omega_sign=sign, n_spots=n_spots,
                         ingest_counts=ingest_counts, domains=domains,
                         completeness=completeness_list, recovery=recovery_list,
                         decoy=decoy_list, feature_in_raw=feature_list, ab_gates=gate_list,
                         refined_cell=refined_cell, split_pct=split_pct, notes=notes,
                         quotable=bool(quotable), orientation_envelope=orientation_list,
                         cell_bootstrap_samples=cell_bootstrap_samples,
                         cell_sigma_bootstrap=cell_sigma_bootstrap)


# ---------------------------------------------------------------------------
# a whole raster: one point, or a sharded block of them
# ---------------------------------------------------------------------------

def reduce_raster_block(
    loader: Callable[[int], np.ndarray], geom: Geometry, point_indices: Sequence[int],
    out_dir, *, block_nr: int = 0, n_blocks: int = 1, **position_kwargs,
) -> List[int]:
    """Run :func:`reduce_one_position` over one shard of a raster, one file per point.

    ``loader(p)`` returns point ``p``'s raw frame stack -- the one place a beamline's file
    format lives; this module knows nothing about it. ``point_indices[block_nr::n_blocks]``
    is this shard's share (mirrors ``midas_fit_grain.scan_driver.refine_scanning_block``'s
    ``voxel_block_nr``/``voxel_n_blocks``): every point is claimed by exactly one shard, and
    shards can run as separate processes -- on one machine, across a few terminals, or one per
    task of any scheduler's array-job syntax (``--block-nr`` maps directly to a task index; see
    ``examples/raster_batch_cli.py``).

    Writes ``out_dir/position_{p}.json`` per point -- idempotent (a rerun overwrites, never
    leaves a partial file: written to a temp path and renamed) so an interrupted block can
    simply be resubmitted. Returns the list of point indices this call actually processed.
    """
    if n_blocks < 1 or not (0 <= block_nr < n_blocks):
        raise ValueError(f"invalid sharding: block_nr={block_nr}, n_blocks={n_blocks}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    shard = list(point_indices)[block_nr::n_blocks]
    for p in shard:
        frames = loader(p)
        result = reduce_one_position(frames, geom, point=p, **position_kwargs)
        tmp = out_dir / f".position_{p}.json.tmp"
        tmp.write_text(json.dumps(result.to_dict(), indent=1, default=str))
        tmp.rename(out_dir / f"position_{p}.json")
    return shard


def raster_wide_asymmetry_sign_test(rows: Sequence[Optional[dict]]) -> dict:
    """The sign-aware, gauge-safe alternative to comparing a raster's MAGNITUDE spread of
    a/b splits against an identical-crystal null.

    /verify REFUTED that magnitude-spread comparison's conclusion on a real 5-position 2604
    block (2026-09-12, claim ``a1c6f1f505cb``): the artifact this whole doc set documents
    (``ENVELOPE.md`` #14, ``midas_hkls.index_asymmetry``'s own docstring) manufactures a fake
    splitting of a CONSISTENT SIGN, and a magnitude-only test cannot see a sign, so it cannot
    tell a real crystal-to-crystal trend from the SAME artifact expressing itself with
    different severity at each position.

    This calls :func:`midas_hkls.asymmetry_sign_test` on every domain pooled across the WHOLE
    raster (not just "quotable" positions, and using each domain's own INDEPENDENT free fit,
    not a position-level joint one) -- gauge-safe, because it only ever compares a domain's own
    ``sign(a-b)`` against that SAME domain's own ``sign(n_h_gt_k - n_k_gt_h)``, never one
    domain's "a" against another's. Verified (2026-09-13, claim ``d93ca3c0a6e2``) on the full
    historical 2604 raster (621 positions, both the recorded driver and its duplicate-check
    -aligned re-run): 65-69% agreement, n=792-1102 domains, p ~ 1e-25 to 1e-27, direction
    ``|h|>|k| -> a<b`` -- see this function's own docstring history / the project's CHECKPOINT
    for the verified verdict once logged.

    ``rows``: results as returned by :func:`assemble_raster_results` (or a list of
    :meth:`PositionResult.to_dict` outputs) -- ``None`` entries (an unfinished shard, or a
    position where ``find_domains`` found nothing) are skipped, matching
    ``assemble_raster_results``'s own "a partial raster stays visibly partial" convention.

    Returns :func:`midas_hkls.asymmetry_sign_test`'s own dict, plus ``n_domains_pooled``. A
    small ``n_used`` (fewer domains passed the internal tie/rank exclusion than were pooled)
    means this batch is too small to trust either way -- check it before reading ``p_value``.
    """
    from midas_hkls import asymmetry_sign_test
    domains = [(dom["hkl"], dom["a"], dom["b"]) for r in rows if r for dom in r.get("domains", [])]
    result = asymmetry_sign_test(domains)
    result["n_domains_pooled"] = len(domains)
    return result


def assemble_raster_results(out_dir, point_indices: Sequence[int]) -> List[Optional[dict]]:
    """Read back every ``position_{p}.json`` in ``point_indices`` order, ``None`` where missing.

    Mirrors the assemble-loop in ``midas_pipeline/notebooks/05_pf_real_data_recon.ipynb``: a
    point a shard has not reached yet (or that raised) is simply absent, not a zero -- keep the
    ``None`` in the table rather than filling it in, so a partially-run raster is visibly
    partial.
    """
    out_dir = Path(out_dir)
    rows: List[Optional[dict]] = []
    for p in point_indices:
        f = out_dir / f"position_{p}.json"
        rows.append(json.loads(f.read_text()) if f.exists() else None)
    return rows
