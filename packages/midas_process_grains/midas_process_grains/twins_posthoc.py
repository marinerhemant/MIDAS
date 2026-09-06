"""Post-hoc twin analysis of a finished grain list.

Runs on the ``Grains.csv`` a completed reconstruction already produced — it does
not change how grains are found. Two things it adds over
:func:`midas_process_grains.compute.twins.find_twin_pairs`:

* **Variant resolution** — which of the four {111} planes, reusing that module's
  own :func:`default_fcc_twin_relations` so the labels match
  (``FCC_Sigma3_<111>`` and friends).
* **A spatial adjacency control**, which is what makes a twin fraction mean
  anything. Any large grain list contains pairs that sit near a CSL
  misorientation by chance; a real annealing twin shares a boundary with its
  parent. So the near-neighbour rate is always reported against the rate among
  far-apart pairs, and :attr:`TwinResult.trustworthy` is False when the two are
  comparable, whatever the raw count.

Why post-hoc rather than ``--mode physics``
-------------------------------------------
``mode=physics`` carries twin labelling, but ``v4_pipeline`` does not read
FitBest, so it has no spot-level input: on 1-ID LSHR layer 6 it produced
``n_spots_matched = -1`` for all 6446 grains, no measured σ_Z, radii ~13× small
(median 0.65 µm against 8.61) and a packing fraction of 0.125 %. Its variant
labels are sound; its sizes are not. Post-hoc keeps the c_parity grain list and
its volumes, and takes only the labelling.

Reference measurement (``park_dmi_sam5`` layer 6, 2466 grains, 2026-09-01):
Σ3 rate 10.60 % among neighbours within 45 µm against 0.23 % beyond 200 µm —
**47× enrichment**, 65.5 % of grains with an adjacent Σ3 partner, and the four
{111} variants populated evenly, which is itself a check that the population is
real rather than a tolerance artefact.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class TwinResult:
    """Adjacent twin pairs with variant labels, plus the adjacency control."""

    pairs: np.ndarray                       # (m, 2) indices into the grain list
    variants: List[str]                     # (m,) e.g. "FCC_Sigma3_<111>"
    residual_deg: np.ndarray                # (m,) misorientation from the ideal OR
    distance_um: np.ndarray                 # (m,)
    n_partners: np.ndarray                  # (n,) adjacent twin partners per grain
    variant_counts: Dict[str, int]
    max_distance_um: float
    tol_deg: float
    rate_near: float
    rate_far: float                         # the null
    enrichment: float
    twinned_fraction: float
    n_grains: int
    n_near_pairs: int
    n_far_sampled: int
    provenance: Dict[str, object] = field(default_factory=dict)

    @property
    def trustworthy(self) -> bool:
        """Enough neighbour pairs, a properly sampled null, and real enrichment.

        ``enrichment`` is ``inf`` when the null rate is exactly zero, which is
        the STRONGEST evidence, not a failure — so infinity passes, provided the
        null was sampled well enough that a nonzero rate would have shown up.
        (An earlier version tested ``np.isfinite(enrichment)`` and so rejected
        precisely the cleanest case.)
        """
        if self.n_near_pairs < 50 or self.n_far_sampled < 200:
            return False
        if np.isnan(self.enrichment):
            return False
        return bool(self.enrichment >= 3.0)

    @property
    def variant_balance(self) -> float:
        """min/max over variant counts. A real population fills all evenly.

        Near 1 is healthy. A low value means one variant dominates, which
        usually indicates a tolerance or symmetry-handling problem rather than
        a physical preference.
        """
        v = [c for c in self.variant_counts.values()]
        if not v or max(v) == 0:
            return float("nan")
        return float(min(v) / max(v))

    def summary(self) -> str:
        verdict = ("adjacent — real twins" if self.trustworthy
                   else "NOT spatially preferred — coincidental orientations")
        lines = [
            f"{len(self.pairs)} adjacent twin pairs among {self.n_near_pairs} "
            f"neighbour pairs (<{self.max_distance_um:.0f} µm, tol {self.tol_deg:.2f}°)",
            f"  rate near {100*self.rate_near:.2f}%  far {100*self.rate_far:.2f}% "
            f"(null)  enrichment {self.enrichment:.1f}×  ->  {verdict}",
            f"  {100*self.twinned_fraction:.1f}% of {self.n_grains} grains have "
            f"an adjacent twin partner",
            f"  variant balance {self.variant_balance:.2f} "
            f"(1.0 = all {len(self.variant_counts)} variants equally populated)",
        ]
        for k, c in sorted(self.variant_counts.items(), key=lambda kv: -kv[1]):
            lines.append(f"    {k:24s} {c:6d}")
        return "\n".join(lines)


def _relation_matrices(relations) -> List[Tuple[str, np.ndarray]]:
    from midas_stress.orientation import quat_to_orient_mat
    out = []
    for tw in relations:
        m = np.asarray(quat_to_orient_mat(list(np.asarray(tw.quaternion,
                                                          dtype=float))),
                       dtype=float).reshape(3, 3)
        out.append((tw.name, m))
    return out


def find_twins(
    orient_mat: np.ndarray,
    positions: np.ndarray,
    space_group: int,
    *,
    relations=None,
    max_distance_um: float = 45.0,
    tol_deg: float = 1.0,
    far_distance_um: float = 200.0,
    n_far_samples: int = 4000,
    max_near_pairs: int = 200_000,
    seed: int = 0,
) -> TwinResult:
    """Find adjacent twin pairs and label the variant.

    ``relations`` defaults to the four FCC Σ3 operators from
    :func:`~midas_process_grains.compute.twins.default_fcc_twin_relations`.
    Pass ``default_cubic_twin_relations()`` for Σ9/Σ27 as well.

    ``max_distance_um`` should be a few grain diameters: too small misses real
    twins, too large and the adjacency control loses its power.
    """
    from scipy.spatial import cKDTree
    from midas_stress.orientation import misorientation_om
    from .compute.twins import default_fcc_twin_relations

    if relations is None:
        relations = default_fcc_twin_relations()
    rel = _relation_matrices(relations)

    orient_mat = np.asarray(orient_mat, dtype=float).reshape(-1, 3, 3)
    positions = np.asarray(positions, dtype=float).reshape(-1, 3)
    n = orient_mat.shape[0]
    if positions.shape[0] != n:
        raise ValueError(f"{n} orientations but {positions.shape[0]} positions")
    if not rel:
        raise ValueError("no twin relations supplied")

    def best_variant(i: int, j: int) -> Tuple[Optional[str], float]:
        """Smallest residual over the relations; the twin acts in the CRYSTAL frame."""
        best_name, best_res = None, np.inf
        for name, T in rel:
            ang, _ = misorientation_om(orient_mat[i] @ T, orient_mat[j], space_group)
            a = float(np.degrees(ang))
            if a < best_res:
                best_res, best_name = a, name
        return (best_name, best_res) if best_res <= tol_deg else (None, best_res)

    tree = cKDTree(positions)
    near = tree.query_pairs(max_distance_um, output_type="ndarray")
    rng = np.random.default_rng(seed)
    if len(near) > max_near_pairs:
        near = near[rng.choice(len(near), max_near_pairs, replace=False)]

    pairs, variants, resid, dists = [], [], [], []
    for i, j in near:
        name, res = best_variant(int(i), int(j))
        if name is not None:
            pairs.append((int(i), int(j)))
            variants.append(name)
            resid.append(res)
            dists.append(float(np.linalg.norm(positions[i] - positions[j])))
    rate_near = len(pairs) / len(near) if len(near) else float("nan")

    far_hits, tried, sampled = 0, 0, 0
    while sampled < n_far_samples and tried < 50 * n_far_samples and n > 2:
        tried += 1
        i, j = int(rng.integers(0, n)), int(rng.integers(0, n))
        if i == j or np.linalg.norm(positions[i] - positions[j]) < far_distance_um:
            continue
        sampled += 1
        if best_variant(i, j)[0] is not None:
            far_hits += 1
    rate_far = (far_hits / sampled) if sampled else float("nan")

    n_partners = np.zeros(n, dtype=np.int64)
    for i, j in pairs:
        n_partners[i] += 1
        n_partners[j] += 1

    counts = {name: 0 for name, _ in rel}
    for v in variants:
        counts[v] = counts.get(v, 0) + 1

    enrich = (rate_near / rate_far) if (rate_far and np.isfinite(rate_far)
                                        and rate_far > 0) else float("inf")
    return TwinResult(
        pairs=np.array(pairs, dtype=np.int64).reshape(-1, 2),
        variants=variants,
        residual_deg=np.array(resid, dtype=float),
        distance_um=np.array(dists, dtype=float),
        n_partners=n_partners,
        variant_counts=counts,
        max_distance_um=float(max_distance_um),
        tol_deg=float(tol_deg),
        rate_near=float(rate_near),
        rate_far=float(rate_far),
        enrichment=float(enrich),
        twinned_fraction=float(np.mean(n_partners > 0)) if n else float("nan"),
        n_grains=int(n),
        n_near_pairs=int(len(near)),
        n_far_sampled=int(sampled),
        provenance={"space_group": space_group,
                    "relations": [name for name, _ in rel],
                    "far_distance_um": far_distance_um},
    )


def find_twins_in_grains(grains, space_group: int, **kw) -> TwinResult:
    """:func:`find_twins` straight from a :class:`~.io.read.GrainsTable`."""
    return find_twins(grains.orient_mat, grains.positions, int(space_group), **kw)


def tolerance_sweep(orient_mat, positions, space_group,
                    tols: Sequence[float] = (0.25, 0.5, 1.0, 2.0, 3.0),
                    **kw) -> List[Tuple[float, int, float, float]]:
    """``(tol_deg, n_pairs, twinned_fraction, enrichment)`` across tolerances.

    A real population is not sharply sensitive to the tolerance; a count that
    keeps climbing while the enrichment falls is picking up chance pairs.
    Report this rather than quoting one tolerance.
    """
    out = []
    for t in tols:
        r = find_twins(orient_mat, positions, space_group, tol_deg=t, **kw)
        out.append((float(t), len(r.pairs), r.twinned_fraction, r.enrichment))
    return out


__all__ = ["TwinResult", "find_twins", "find_twins_in_grains", "tolerance_sweep"]
