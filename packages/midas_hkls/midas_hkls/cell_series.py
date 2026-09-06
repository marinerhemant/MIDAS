"""Tracking a cell through an in-situ series, and failing loudly at a transition.

A pressure, temperature or load series asks one question repeatedly: *is this
still the same phase, with a cell that has moved, or is it a different phase?*
The two demand opposite responses and they are easy to confuse, because a freely
refined triclinic cell is flexible enough to absorb a great deal.

What a free triclinic refinement can never detect
------------------------------------------------
**A pure homogeneous deformation of the old lattice is invisible to this test,
at any magnitude.** Any such deformation *is* a triclinic cell, so free
refinement recovers it exactly and the fit stays perfect. Compressing a cell by
20 % anisotropically produces no signal at all. This was found by the positive
control below failing to separate, which is what controls are for.

So the detectable signature of a transition is not a worse fit to the same
reflections. It is that **no single lattice takes all the observed spots** —
first-order transitions coexist, and during coexistence part of the pattern
belongs to each phase. That is what the gate measures, over the **whole**
observed spot list rather than the carried-forward assignment.

Two things follow. The gate needs the unindexed spots, not just the indexed
ones. And a strictly displacive, fully-transformed change of shape will pass —
:func:`cell_deformation` is what flags that, by the size and anisotropy of the
strain, not the gate.

That flexibility is the danger this module is built around, so the gate refuses
to return a verdict until it has established that it **could** have detected
one:

* a **null** — the same reflection count and noise, with the cell merely drifted;
* a **positive control** — the same again, with a transition actually planted.

If the control does not separate from the null, the answer is
``INDETERMINATE``, whatever the data did. A test that cannot see a planted
transition cannot report its absence — the same discipline a planted rod imposes
on a directional test.

What counts as "explained"
--------------------------
The statistic is the fraction of **observed** g-vectors that a freely refined
cell can put within ``tol`` of an integer hkl (ImageD11's ``drlv2`` criterion).
At a drift that fraction stays high because refinement absorbs the change; at a
transition the observed spots belong to a different lattice and no triclinic
cell reaches them.

Reporting the strain
--------------------
:func:`cell_deformation` gives the deformation gradient taking one cell to the
next, its Green strain and its volume ratio. A drift along an equation of state
is a smooth, small, usually near-hydrostatic strain; a jump or a strongly
non-hydrostatic step is worth looking at even when the gate passes.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from .lattice import Lattice
from .ub_refine import UBFit, drlv2, refine_ub_from_gvectors, ub_to_cell

__all__ = ["CellDeformation", "TransitionVerdict", "cell_deformation",
           "explained_fraction", "track_cell", "detect_transition",
           "poisson_upper_p", "min_detectable_excess"]


def _B_of(cell) -> np.ndarray:
    lat = Lattice(a=cell[0], b=cell[1], c=cell[2],
                  alpha=cell[3], beta=cell[4], gamma=cell[5])
    return np.asarray(lat.reciprocal_cartesian_vectors(), float).T


def _A_of(cell) -> np.ndarray:
    lat = Lattice(a=cell[0], b=cell[1], c=cell[2],
                  alpha=cell[3], beta=cell[4], gamma=cell[5])
    return np.asarray(lat.cartesian_vectors(), float).T


@dataclass
class CellDeformation:
    """How one cell became the next."""
    F: np.ndarray                 # deformation gradient, A_new = F @ A_old
    green_strain: np.ndarray      # (F^T F - I)/2
    volume_ratio: float
    principal_strains: Tuple[float, float, float]
    max_shear: float

    @property
    def is_near_hydrostatic(self) -> bool:
        e = np.array(self.principal_strains)
        spread = e.max() - e.min()
        return bool(spread < 0.25 * max(abs(e).max(), 1e-12))

    def __str__(self) -> str:
        e = self.principal_strains
        return (f"V/V0 = {self.volume_ratio:.5f}, principal strain "
                f"({e[0]:+.4%}, {e[1]:+.4%}, {e[2]:+.4%}), max shear "
                f"{self.max_shear:.4%}"
                f"{'' if self.is_near_hydrostatic else '  [NON-HYDROSTATIC]'}")


def cell_deformation(cell_old, cell_new) -> CellDeformation:
    """Deformation gradient, Green strain and volume ratio between two cells.

    Both cells must be in the **same setting** — the same hkl labelling — or the
    strain is meaningless. Carrying an assignment forward, as a series does,
    guarantees that; re-indexing from scratch does not.
    """
    A_old, A_new = _A_of(cell_old), _A_of(cell_new)
    F = A_new @ np.linalg.inv(A_old)
    E = 0.5 * (F.T @ F - np.eye(3))
    ev = np.linalg.eigvalsh(E)
    return CellDeformation(F=F, green_strain=E, volume_ratio=float(np.linalg.det(F)),
                           principal_strains=tuple(float(v) for v in ev),
                           max_shear=float((ev.max() - ev.min()) / 2.0))


def explained_fraction(UBI: np.ndarray, g_obs: np.ndarray,
                       tol: float = 0.15) -> float:
    """Fraction of observed g-vectors within ``tol`` of an integer hkl."""
    return float((np.sqrt(drlv2(UBI, g_obs)) < tol).mean())


def track_cell(cell_prev, hkl, g_obs, *, weights=None, sigma_g=None,
               tol: float = 0.15) -> Tuple[UBFit, CellDeformation, float]:
    """Refine freely from the previous cell. Returns ``(fit, deformation, frac)``.

    The refinement is **free triclinic** — the previous cell supplies only the
    hkl labelling, not a constraint. That is deliberate: a symmetry-lowering
    transition is invisible to a refinement locked to the old symmetry.
    """
    fit = refine_ub_from_gvectors(hkl, g_obs, weights=weights, sigma_g=sigma_g)
    return (fit, cell_deformation(cell_prev, fit.cell),
            explained_fraction(fit.UBI, np.asarray(g_obs, float), tol))


# ---------------------------------------------------------------------------
# the transition gate — a RATE test
# ---------------------------------------------------------------------------
#
# Three earlier versions compared a *fraction* of a fixed spot count and were
# each refuted by their own control. The framing error, recorded so it is not
# repeated: **a second phase does not take a share of the unassigned pool, it
# ADDS spots to it.** A statistic that holds the total fixed cannot express the
# alternative hypothesis at all, and a simulated null with a deterministic
# background count has no scatter, so its 5th percentile equals its median and
# one stray spot reads as a transition.
#
# So the statistic is a RATE. The previous point of the series measures how many
# unexplained spots a pattern carries when nothing is happening; a transition
# adds more. Counts of independent rare events are Poisson, which supplies the
# scatter for free — no Monte Carlo, no degenerate null — and makes the power
# question answerable in closed form.


@dataclass
class TransitionVerdict:
    """The gate's answer, with the excess it could and could not have seen."""
    verdict: str                      # CONTINUOUS | TRANSITION | INDETERMINATE
    n_unexplained: int
    n_expected: float                 # baseline_rate * n_assigned
    p_value: float                    # Poisson upper tail
    min_detectable_excess: int        # smallest excess reachable at this alpha/power
    has_power: bool
    fit: Optional[UBFit] = None
    deformation: Optional[CellDeformation] = None
    n_assigned: int = 0
    notes: list = field(default_factory=list)

    def __str__(self) -> str:
        head = (f"{self.verdict}: {self.n_unexplained} unexplained vs "
                f"{self.n_expected:.1f} expected from the baseline rate "
                f"(p = {self.p_value:.4f}); this test can detect an excess of "
                f">= {self.min_detectable_excess} spots")
        if not self.has_power:
            head += "  -- NO POWER at the excess you said to expect"
        return head + (("\n  " + "\n  ".join(self.notes)) if self.notes else "")


def poisson_upper_p(k: int, lam: float) -> float:
    """``P(X >= k)`` for ``X ~ Poisson(lam)``."""
    from scipy import stats
    if lam <= 0:
        return 0.0 if k > 0 else 1.0
    return float(stats.poisson.sf(k - 1, lam))


def min_detectable_excess(lam: float, *, alpha: float = 0.01,
                          power: float = 0.8, cap: int = 10_000) -> int:
    """Smallest true excess this rate test can detect, in spots.

    Closed form, no simulation: find the critical count at significance
    ``alpha`` under the baseline, then the smallest excess whose distribution
    puts ``power`` of its mass at or above it. Reporting this is the whole point
    — a gate that cannot say what it would have missed is not a gate.
    """
    from scipy import stats
    if lam < 0:
        raise ValueError("lam must be non-negative")
    k_crit = int(stats.poisson.isf(alpha, max(lam, 1e-12))) + 1
    for d in range(0, int(cap) + 1):
        if stats.poisson.sf(k_crit - 1, lam + d) >= power:
            return d
    return int(cap)


def detect_transition(cell_prev, hkl, g_assigned, g_all, *,
                      baseline_rate: float,
                      sigma_g: Optional[float] = None,
                      tol: float = 0.15,
                      alpha: float = 0.01,
                      power: float = 0.8,
                      expected_excess: Optional[int] = None,
                      ) -> TransitionVerdict:
    """Has a second phase appeared, or has the cell merely drifted?

    Parameters
    ----------
    cell_prev
        The cell at the previous point — the seed for the free refinement.
    hkl, g_assigned
        The assignment carried forward; used to refine, and clean by definition.
    g_all
        **Every** observed g-vector, indexed or not. A transition's evidence is
        entirely in the unassigned ones, so omitting them removes the signal.
    baseline_rate
        Unexplained spots **per assigned spot**, measured at a point of the
        series where you know nothing was happening. This is the null, and it
        must come from the experiment — there is no defensible default.
    expected_excess
        How many extra spots a transition would plausibly add here. Used only to
        decide whether to report ``INDETERMINATE``; if omitted, the gate flags
        no-power when the minimum detectable excess exceeds the assigned count.

    Notes
    -----
    A **pure homogeneous strain** of the old lattice is invisible to this test
    at any magnitude, because such a deformation *is* a triclinic cell and free
    refinement absorbs it exactly. Judge that on
    :attr:`TransitionVerdict.deformation`, never on this verdict.
    """
    hkl = np.asarray(hkl, float)
    g_assigned = np.asarray(g_assigned, float)
    g_all = np.asarray(g_all, float)
    if g_all.ndim != 2 or g_all.shape[1] != 3:
        raise ValueError("g_all must be (N, 3) — every observed spot")
    if len(g_all) < len(g_assigned):
        raise ValueError("g_all must contain at least the assigned spots")
    if baseline_rate < 0:
        raise ValueError("baseline_rate must be non-negative")

    notes: list = []
    fit, defo, _ = track_cell(cell_prev, hkl, g_assigned, sigma_g=sigma_g, tol=tol)

    n_assigned = len(hkl)
    explained = np.sqrt(drlv2(fit.UBI, g_all)) < tol
    n_unexplained = int((~explained).sum())
    lam = baseline_rate * n_assigned
    p = poisson_upper_p(n_unexplained, lam)
    mde = min_detectable_excess(lam, alpha=alpha, power=power)

    threshold = expected_excess if expected_excess is not None else n_assigned
    has_power = mde <= threshold

    if not has_power:
        verdict = "INDETERMINATE"
        notes.append(
            f"the smallest excess this test can detect is {mde} spots, more "
            f"than the {threshold} you said to expect. At a baseline of "
            f"{baseline_rate:.3g} per assigned spot the Poisson noise alone is "
            f"+/-{math.sqrt(max(lam, 1e-12)):.1f}. Collect more reflections, or "
            "reduce the background rate, before asking.")
    elif p < alpha:
        verdict = "TRANSITION"
        notes.append(
            f"{n_unexplained} unexplained where the baseline predicts "
            f"{lam:.1f}: p = {p:.4g} < {alpha}. A single lattice does not take "
            "these spots.")
    else:
        verdict = "CONTINUOUS"
        notes.append(f"unexplained count is consistent with the baseline rate "
                     f"(p = {p:.3g}). {defo}")
    if verdict == "CONTINUOUS" and defo is not None and not defo.is_near_hydrostatic:
        notes.append("NOTE: strongly non-hydrostatic strain. A displacive change "
                     "of shape passes this gate by construction — judge it on "
                     "the strain, not on the count.")

    return TransitionVerdict(verdict=verdict, n_unexplained=n_unexplained,
                             n_expected=float(lam), p_value=float(p),
                             min_detectable_excess=int(mde), has_power=has_power,
                             fit=fit, deformation=defo, n_assigned=n_assigned,
                             notes=notes)
