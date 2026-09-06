"""Phase identification from a handful of d-spacings, done so it can fail.

Matching a few observed d values against candidate phases is the oldest trick
in powder diffraction and one of the easiest to get wrong. Four faults, all of
which have been committed on real data and all of which this module makes hard:

**1. A residual without a line count is not evidence.**
A candidate with 400 allowed lines in the observed range will match a handful of
d values better than one with 40, by chance alone and regardless of whether it
is the right phase. Every result here carries ``n_lines``, and
:func:`chance_worst_residual` converts that count into the residual a *random*
phase with the same number of lines would reach — which is the number the
candidate has to beat.

**2. Hand-written reflection conditions.**
Writing "F-centred, so h,k,l all even" by hand is how a rival gets 50 lines when
its real space group gives 97, and then loses a comparison it should have won.
Lines come from :func:`~midas_hkls.generate_hkls` with the candidate's actual
space group, so the conditions are never retyped.

**3. A local minimum in the free scale.**
The worst-residual landscape over a free scale factor is full of local minima.
An optimiser started at 1.0 can stop in the third-deepest one and hand back a
"physically impossible" compression that is simply the wrong minimum.
:func:`global_minimax_scale` does a dense sweep before refining.

**4. Asymmetric cell provenance.**
Giving the favoured phase a cell refined from *this* dataset at *this* pressure,
while its rivals get ambient literature cells, decides the comparison before it
starts. This module cannot detect that for you — but
:func:`identify_phase` records each candidate's cell in the output so the
asymmetry is visible in the result rather than buried in the setup.

What the free scale is not
--------------------------
A free scale multiplying every predicted d is **degenerate with the
sample-detector distance and the wavelength**. The fitted value therefore
absorbs whatever is unverified in Lsd and λ, and is **not a compression
measurement**. It exists only to give every candidate its best possible shot.
Quoting it as a strain or a pressure is a category error.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = ["PhaseCandidate", "PhaseMatch", "candidate_d_lines",
           "worst_relative_residual", "global_minimax_scale",
           "chance_worst_residual", "identify_phase"]


@dataclass
class PhaseCandidate:
    """A phase to test: a name and a `midas_hkls.Crystal`."""
    name: str
    crystal: object                      # midas_hkls.Crystal
    cell_source: str = ""                # provenance — record it, see fault 4

    def describe_cell(self) -> str:
        lat = self.crystal.lattice
        return (f"a={lat.a:.4f} b={lat.b:.4f} c={lat.c:.4f} "
                f"al={lat.alpha:.2f} be={lat.beta:.2f} ga={lat.gamma:.2f}")


@dataclass
class PhaseMatch:
    """One candidate's result. `n_lines` is not decoration — see the module doc."""
    name: str
    n_lines: int
    worst_fixed_pct: float
    best_scale: float
    worst_free_pct: float
    p_value: Optional[float] = None       # vs a random phase of the same size
    chance_median_pct: Optional[float] = None
    cell: str = ""
    cell_source: str = ""

    def __str__(self) -> str:
        p = "" if self.p_value is None else f", p = {self.p_value:.3f}"
        return (f"{self.name}: {self.n_lines} lines, worst {self.worst_free_pct:.3f} % "
                f"at scale {self.best_scale:.5f}{p}")


def candidate_d_lines(crystal, *, d_min: float, d_max: float) -> np.ndarray:
    """Unique allowed d-spacings of a crystal in ``[d_min, d_max]``.

    Absences come from the crystal's own space group. Symmetry-equivalent
    reflections collapse to one line, which is what a d-match sees.
    """
    from .hkl_gen import generate_hkls
    if not (0 < d_min < d_max):
        raise ValueError("need 0 < d_min < d_max")
    refs = generate_hkls(crystal.space_group, crystal.lattice, d_min=d_min)
    ds = np.array(sorted({round(float(r.d_spacing), 9) for r in refs
                          if d_min <= r.d_spacing <= d_max}))
    return ds


def worst_relative_residual(lines: np.ndarray, d_obs: Sequence[float],
                            scale: float = 1.0) -> float:
    """Worst |Δd|/d over the observations, in percent, at a given scale.

    The **worst** case, not the mean: a phase that explains five of six spots
    beautifully and misses the sixth entirely has not explained the data.
    """
    lines = np.asarray(lines, float)
    if lines.size == 0:
        return float("inf")
    scaled = lines * scale
    d_obs = np.asarray(d_obs, float)
    idx = np.searchsorted(scaled, d_obs)
    idx = np.clip(idx, 1, len(scaled) - 1) if len(scaled) > 1 else np.zeros_like(idx)
    lo = np.clip(idx - 1, 0, len(scaled) - 1)
    hi = np.clip(idx, 0, len(scaled) - 1)
    best = np.minimum(np.abs(scaled[lo] - d_obs), np.abs(scaled[hi] - d_obs))
    return float(np.max(best / d_obs) * 100.0)


def global_minimax_scale(lines: np.ndarray, d_obs: Sequence[float], *,
                         scale_range: Tuple[float, float] = (0.88, 1.14),
                         n_coarse: int = 26_001,
                         n_fine: int = 4_001) -> Tuple[float, float]:
    """The GLOBAL best scale, by dense sweep then local refine.

    A gradient step from 1.0 lands in whichever local minimum is nearest, which
    is routinely not the deepest. Returns ``(scale, worst_residual_pct)``.
    """
    lo, hi = scale_range
    if not (0 < lo < hi):
        raise ValueError("scale_range must be positive and increasing")
    grid = np.linspace(lo, hi, int(n_coarse))
    w = np.array([worst_relative_residual(lines, d_obs, s) for s in grid])
    k = int(np.argmin(w))
    a = grid[max(k - 3, 0)]
    b = grid[min(k + 3, len(grid) - 1)]
    fine = np.linspace(a, b, int(n_fine))
    wf = np.array([worst_relative_residual(lines, d_obs, s) for s in fine])
    j = int(np.argmin(wf))
    return float(fine[j]), float(wf[j])


def chance_worst_residual(n_lines: int, d_obs: Sequence[float], *,
                          d_min: float, d_max: float,
                          n_draws: int = 400,
                          free_scale: bool = True,
                          rng_seed: int = 0,
                          scale_range: Tuple[float, float] = (0.88, 1.14)
                          ) -> np.ndarray:
    """What worst-residual would a RANDOM phase with this many lines reach?

    This is what turns the line count from a caveat into a number. Random lines
    are drawn **uniform in reciprocal-space volume** (density in |G| ∝ |G|²),
    not uniform in d — a uniform-in-d null under-populates the small-d end where
    real reflections crowd, and so makes every candidate look significant.

    Returns the distribution of worst residuals over ``n_draws`` random phases.
    """
    rng = np.random.default_rng(rng_seed)
    g_lo, g_hi = 1.0 / d_max, 1.0 / d_min
    out = np.empty(n_draws)
    for i in range(n_draws):
        u = rng.random(int(n_lines))
        g = (u * (g_hi ** 3 - g_lo ** 3) + g_lo ** 3) ** (1.0 / 3.0)
        lines = np.sort(1.0 / g)
        out[i] = (global_minimax_scale(lines, d_obs, scale_range=scale_range,
                                       n_coarse=2001, n_fine=401)[1]
                  if free_scale else worst_relative_residual(lines, d_obs))
    return out


def identify_phase(d_obs: Sequence[float],
                   candidates: Sequence[PhaseCandidate], *,
                   d_min: Optional[float] = None,
                   d_max: Optional[float] = None,
                   free_scale: bool = True,
                   n_null_draws: int = 200,
                   scale_range: Tuple[float, float] = (0.88, 1.14),
                   rng_seed: int = 0) -> List[PhaseMatch]:
    """Rank candidates against a few observed d values, honestly.

    Returns one :class:`PhaseMatch` per candidate, sorted best first. Each
    carries its line count, its cell and its cell provenance, and — when
    ``n_null_draws > 0`` — a p-value against a random phase with the *same
    number of lines*, which is the comparison the line count exists to enable.

    A candidate that wins on residual but has ten times the lines of its rival,
    and a p-value near 0.5, has not won.
    """
    d_obs = np.asarray(d_obs, float)
    if d_obs.size == 0:
        raise ValueError("no observed d values")
    lo = d_min if d_min is not None else float(d_obs.min()) * 0.7
    hi = d_max if d_max is not None else float(d_obs.max()) * 1.4

    out: List[PhaseMatch] = []
    for cand in candidates:
        lines = candidate_d_lines(cand.crystal, d_min=lo, d_max=hi)
        fixed = worst_relative_residual(lines, d_obs, 1.0)
        if free_scale and lines.size:
            scale, worst = global_minimax_scale(lines, d_obs,
                                                scale_range=scale_range)
        else:
            scale, worst = 1.0, fixed
        p = med = None
        if n_null_draws and lines.size:
            null = chance_worst_residual(len(lines), d_obs, d_min=lo, d_max=hi,
                                         n_draws=n_null_draws,
                                         free_scale=free_scale,
                                         rng_seed=rng_seed,
                                         scale_range=scale_range)
            p = float((null <= worst).mean())
            med = float(np.median(null))
        out.append(PhaseMatch(name=cand.name, n_lines=int(lines.size),
                              worst_fixed_pct=fixed, best_scale=scale,
                              worst_free_pct=worst, p_value=p,
                              chance_median_pct=med,
                              cell=cand.describe_cell(),
                              cell_source=cand.cell_source))
    out.sort(key=lambda m: m.worst_free_pct)
    return out
