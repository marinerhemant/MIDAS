"""Per-spot soft attribution: which grain does a contested reflection belong to?

A reflection claimed by two grains is either **legitimately shared** — twin
variants sit on a common plane and genuinely diffract into the same spot — or
**accidentally overlapping**, in which case at least one grain is being fitted
against a spot that is not its own. Measured on 1-ID LSHR (2026-09-02) the two
classes behave in opposite directions, against the grain's *own* uncontested
spots as the reference:

===========================  ===================  ==============================
class                        median DiffLenPost   vs the grain's own uncontested
===========================  ===================  ==============================
accidental overlap           152.02 µm            **3.98× worse**, in 96.1 % of grains
Σ3 twin-shared                23.16 µm            **0.62× BETTER**, in only 31.3 %
===========================  ===================  ==============================

and removing them has opposite effects on a refit scored against held-out spots:
dropping the accidental ones improved the fit by 32.2 %, while dropping the same
*number* of random spots made it 32.8 % worse and dropping the twin-shared ones
48.7 % worse. So a filter that deletes every co-claimed spot destroys the best
spots in the dataset. The classes must be told apart.

How this module tells them apart
--------------------------------
Without needing twin labels at all. Each claim gets a **consistency**

.. math::  w_{gs} = \\exp(-d_{gs}^2 / 2\\sigma^2)

from its own residual, and is judged on ``rel_likelihood`` — its consistency
relative to the *best* claimant of that spot. A twin pair fits a shared spot
about equally well, so both sit near 1.0 and both survive; an accidental
overlap where one grain fits at 30 µm and another at 300 µm collapses to 1.0
and ~0.

The rule has to be **relative**, not a responsibility threshold. Responsibility
(:math:`w / \\sum w`) is ~0.5 for each half of a twin pair, so any cut at 0.5 —
"give the spot to whoever fits it best" — would arbitrarily delete one twin of
every pair, which is precisely the failure the measurement above warns about.

Two weights, for two different jobs
-----------------------------------
``consistency`` is **unnormalised** and is the right per-spot weight for a
weighted fit: a spot that genuinely belongs to both twins should carry full
weight for *both*, not half each. ``responsibility`` sums to 1 across claimants
and answers the ownership question. Do not use responsibility as a fit weight.

Relation to :mod:`~midas_process_grains.compute.spot_budget`
------------------------------------------------------------
That module solves a different problem and stays the right tool for it. It
attributes each spot to the highest **per-grain ``quality_score``** claimant
(crediting twin families) in order to cull grains via a keep-mask. Because its
score is per grain, a good grain wins *all* its contested spots and a bad one
loses all of them — it cannot say "this one spot does not belong to this
otherwise-good grain", and it must be handed ``twin_family_id``. This module is
per **claim**, driven by that claim's own residual, needs no twin labels, and
emits weights for re-refinement rather than a grain keep-mask.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

#: Keep a claim whose consistency is at least this fraction of the best
#: claimant's. 0.05 ⇒ drop a claim the best claimant beats by >20× in
#: likelihood. Well below the ~1.0 a twin pair scores, so twins are safe.
DEFAULT_REL_THRESHOLD = 0.05

#: Never strip a grain below this many spots — an under-determined refit is
#: worse than a contaminated one.
DEFAULT_MIN_SPOTS = 25


@dataclass
class SpotAttribution:
    """Per-claim attribution over a set of ``(grain, spot)`` claims."""

    grain_id: np.ndarray          # (M,)
    spot_id: np.ndarray           # (M,)
    residual: np.ndarray          # (M,) µm
    n_claimants: np.ndarray       # (M,) how many grains claim this spot
    consistency: np.ndarray       # (M,) exp(-d²/2σ²) — UNNORMALISED fit weight
    responsibility: np.ndarray    # (M,) consistency / Σ over claimants
    rel_likelihood: np.ndarray    # (M,) consistency / max over claimants
    keep: np.ndarray              # (M,) bool — survives the filter
    sigma: float
    rel_threshold: float
    min_spots: int
    n_grains: int
    n_spots: int
    n_protected: int = 0          # claims kept only by the min_spots floor
    provenance: Dict[str, object] = field(default_factory=dict)

    @property
    def n_contested(self) -> int:
        return int((self.n_claimants > 1).sum())

    @property
    def n_dropped(self) -> int:
        return int((~self.keep).sum())

    def kept_spot_sets(self) -> Dict[int, List[int]]:
        """``{grain_id: [spot ids to fit]}`` — the input for a second pass."""
        out: Dict[int, List[int]] = {}
        for g, s, k in zip(self.grain_id, self.spot_id, self.keep):
            if k:
                out.setdefault(int(g), []).append(int(s))
        return {g: sorted(v) for g, v in out.items()}

    def weight_map(self) -> Dict[Tuple[int, int], float]:
        """``{(grain, spot): consistency}`` for a weighted refit.

        Consistency, not responsibility — see the module docstring.
        """
        return {(int(g), int(s)): float(w)
                for g, s, w in zip(self.grain_id, self.spot_id, self.consistency)}

    def per_grain_dropped(self) -> Dict[int, int]:
        out: Dict[int, int] = {}
        for g, k in zip(self.grain_id, self.keep):
            if not k:
                out[int(g)] = out.get(int(g), 0) + 1
        return out

    def summary(self) -> str:
        cont = self.n_claimants > 1
        lines = [
            f"{len(self.grain_id)} claims by {self.n_grains} grains over "
            f"{self.n_spots} spots; {self.n_contested} contested "
            f"({100*self.n_contested/max(len(self.grain_id),1):.1f}%)",
            f"  sigma {self.sigma:.2f} um (robust scale of uncontested residuals), "
            f"rel_threshold {self.rel_threshold:g}",
            f"  dropped {self.n_dropped} claims "
            f"({100*self.n_dropped/max(len(self.grain_id),1):.1f}% of all, "
            f"{100*self.n_dropped/max(self.n_contested,1):.1f}% of contested)",
        ]
        if self.n_protected:
            lines.append(f"  {self.n_protected} claims kept only by the "
                         f"{self.min_spots}-spot floor")
        # nanmedian, not median: a handful of rows carry Matched=1 with a NaN
        # residual (32 of 563k on the LSHR layer), and one NaN turns the whole
        # summary line into 'nan'.
        if cont.any():
            lines.append(
                f"  contested claims: median residual "
                f"{np.nanmedian(self.residual[cont]):.1f} um, "
                f"kept {100*self.keep[cont].mean():.1f}%")
        unc = ~cont
        if unc.any():
            lines.append(f"  uncontested:      median residual "
                         f"{np.nanmedian(self.residual[unc]):.1f} um")
        n_nan = int((~np.isfinite(self.residual)).sum())
        if n_nan:
            lines.append(f"  {n_nan} claims had a non-finite residual and were "
                         f"dropped as unusable")
        return "\n".join(lines)


def _robust_sigma(residual: np.ndarray, n_claimants: np.ndarray) -> float:
    """Scale from the UNCONTESTED residuals — the spots nobody disputes.

    Deriving it from all residuals would let the contamination set the scale it
    is supposed to be judged against, and the filter would dissolve as
    contamination rose.
    """
    unc = residual[(n_claimants == 1) & np.isfinite(residual)]
    if unc.size < 20:
        unc = residual[np.isfinite(residual)]
    if unc.size == 0:
        raise ValueError("no finite residuals to derive a scale from")
    s = float(np.median(unc))
    return s if s > 0 else float(np.mean(unc)) or 1.0


def attribute_spots(
    grain_id: Sequence[int],
    spot_id: Sequence[int],
    residual: Sequence[float],
    *,
    sigma: Optional[float] = None,
    rel_threshold: float = DEFAULT_REL_THRESHOLD,
    min_spots: int = DEFAULT_MIN_SPOTS,
) -> SpotAttribution:
    """Attribute each ``(grain, spot)`` claim from its own residual.

    ``residual`` is the per-claim observed-to-predicted distance in µm
    (``DiffLenPost`` in a ``SpotMatrix.csv``). Non-finite residuals — the
    ``Matched=0`` rows, which are reflections a grain *predicted* but never
    claimed — are treated as infinitely inconsistent and always dropped.

    ``sigma`` defaults to the median uncontested residual, so the scale comes
    from spots nobody disputes.
    """
    g = np.asarray(grain_id, dtype=np.int64)
    s = np.asarray(spot_id, dtype=np.int64)
    d = np.asarray(residual, dtype=np.float64)
    if not (g.shape == s.shape == d.shape):
        raise ValueError(f"shape mismatch: {g.shape}, {s.shape}, {d.shape}")
    if g.size == 0:
        z = np.zeros(0)
        return SpotAttribution(z.astype(np.int64), z.astype(np.int64), z,
                               z.astype(np.int64), z, z, z, z.astype(bool),
                               1.0, rel_threshold, min_spots, 0, 0)

    uniq_s, inv = np.unique(s, return_inverse=True)
    n_claim_per_spot = np.bincount(inv)
    n_claimants = n_claim_per_spot[inv]

    sigma_from = "caller" if sigma is not None else "median uncontested residual"
    if sigma is None:
        sigma = _robust_sigma(d, n_claimants)
    sigma = float(sigma)
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError(f"sigma must be finite and positive, got {sigma}")

    with np.errstate(over="ignore"):
        cons = np.exp(-0.5 * (np.where(np.isfinite(d), d, np.inf) / sigma) ** 2)
    cons = np.where(np.isfinite(cons), cons, 0.0)

    # per-spot sum and max, over claimants
    tot = np.bincount(inv, weights=cons, minlength=uniq_s.size)
    mx = np.zeros(uniq_s.size, dtype=np.float64)
    np.maximum.at(mx, inv, cons)
    denom_tot, denom_mx = tot[inv], mx[inv]
    resp = np.divide(cons, denom_tot, out=np.zeros_like(cons), where=denom_tot > 0)
    rel = np.divide(cons, denom_mx, out=np.zeros_like(cons), where=denom_mx > 0)
    # A spot every claimant fits terribly has max == 0; nobody can own it.
    rel = np.where(denom_mx > 0, rel, 0.0)

    keep = (n_claimants == 1) | (rel >= rel_threshold)
    keep &= np.isfinite(d)

    # Floor: never strip a grain below min_spots. Restore its best-ranked
    # dropped claims until it clears the floor.
    n_protected = 0
    if min_spots > 0:
        order = np.argsort(-rel, kind="stable")
        by_grain: Dict[int, List[int]] = {}
        for i in order:
            by_grain.setdefault(int(g[i]), []).append(int(i))
        for gg, idxs in by_grain.items():
            kept = sum(1 for i in idxs if keep[i])
            if kept >= min_spots:
                continue
            for i in idxs:
                if kept >= min_spots:
                    break
                if not keep[i] and np.isfinite(d[i]):
                    keep[i] = True
                    kept += 1
                    n_protected += 1

    return SpotAttribution(
        grain_id=g, spot_id=s, residual=d, n_claimants=n_claimants,
        consistency=cons, responsibility=resp, rel_likelihood=rel, keep=keep,
        sigma=sigma, rel_threshold=float(rel_threshold), min_spots=int(min_spots),
        n_grains=int(np.unique(g).size), n_spots=int(uniq_s.size),
        n_protected=n_protected,
        provenance={"sigma_from": sigma_from},
    )


def attribute_from_spot_matrix(
    path,
    *,
    residual_column: str = "DiffLenPost",
    **kw,
) -> SpotAttribution:
    """:func:`attribute_spots` straight from a ``SpotMatrix.csv``.

    Reads through the canonical :func:`midas_process_grains.io.read.
    read_spot_matrix` with ``matched_only=True``: an unmatched row is a
    prediction, not a claim, and counting those inflated the observed
    ``max claims/spot`` from **9** to 1933 on the LSHR layer.
    """
    from .io.read import read_spot_matrix

    sm = read_spot_matrix(path, matched_only=True)
    try:
        resid = sm.column(residual_column)
    except KeyError:
        resid = sm.column("DiffLen")     # pre-fit fallback for older files
        residual_column = "DiffLen"
    att = attribute_spots(sm.grain_id, sm.spot_id, resid, **kw)
    att.provenance.update({"path": str(path), "residual_column": residual_column,
                           "n_rows_unmatched": sm.n_rows_unmatched})
    return att


def twin_agreement(att: SpotAttribution, twin_of: Dict[int, set]) -> Dict[str, float]:
    """Check the residual-only rule against twin labels it never saw.

    Validation, not a dependency: if soft attribution is doing what the physics
    says, it should keep twin-shared claims and drop accidental ones without
    being told which is which. Returns the keep rate for each class.
    """
    cont = att.n_claimants > 1
    if not cont.any():
        return {"n_twin_claims": 0.0, "n_accidental_claims": 0.0,
                "keep_rate_twin": float("nan"), "keep_rate_accidental": float("nan")}
    spot_to_grains: Dict[int, List[int]] = {}
    for gg, ss in zip(att.grain_id, att.spot_id):
        spot_to_grains.setdefault(int(ss), []).append(int(gg))
    is_twin = np.zeros(len(att.grain_id), dtype=bool)
    for i in np.nonzero(cont)[0]:
        gg, ss = int(att.grain_id[i]), int(att.spot_id[i])
        others = set(spot_to_grains[ss]) - {gg}
        is_twin[i] = bool(others & twin_of.get(gg, set()))
    tw, ac = cont & is_twin, cont & ~is_twin
    return {
        "n_twin_claims": float(tw.sum()),
        "n_accidental_claims": float(ac.sum()),
        "keep_rate_twin": float(att.keep[tw].mean()) if tw.any() else float("nan"),
        "keep_rate_accidental": float(att.keep[ac].mean()) if ac.any() else float("nan"),
    }


__all__ = ["SpotAttribution", "attribute_spots", "attribute_from_spot_matrix",
           "twin_agreement", "DEFAULT_REL_THRESHOLD", "DEFAULT_MIN_SPOTS"]
