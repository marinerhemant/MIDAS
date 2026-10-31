"""ODF-level readout of node-basis texture results (:mod:`midas_dt.texture_nodes`).

**Why this exists.** The medoid orientation of a voxel's node weights hides how much of the ODF is spread: a recovered ODF with 80 % of its
mass at the truth and 20 % smeared over a 5 degree cloud has the same medoid as one with 100 % at the truth. These functions read the
ODF itself: the top atoms, the medoid, a weighted mean over the atoms near the medoid, the mass within a tolerance of a reference set,
and an optimal-transport (Wasserstein-1) distance to a reference set.

**The weight floor is an explicit argument everywhere, with no default.** The recovered weights of a node basis are a cloud of many small
atoms; a floor (the fraction of the voxel's total weight below which an atom is dropped before the remaining weights are renormalised)
changes the W1 by an order of magnitude. Measured in the regime-map study on one plant, interior-voxel median W1 was 5.4 / 2.7 / 0.26 /
0.25 deg at floors none / 1 % / 5 % / 10 %. Quote the floor next to every W1, and report several.

Dense node lists inflate the apparent cloud (about two dozen oracle nodes lay within 1.5 deg of one truth in that study): merge
near-duplicates (:meth:`VoxelODF.merged`, or :func:`midas_dt.texture_nodes.merge_nodes` on the node list) before counting atoms.

Units: degrees at the API. Orientation matrices map crystal -> sample. All misorientations are symmetry-reduced by ``midas_stress``
(``misorientation_om_batch``) for the Laue class of ``space_group``; the symmetry operators come from ``midas_stress`` too.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .texture_nodes import _laue_matrices, merge_nodes

__all__ = ["VoxelODF", "voxel_odf", "misorientation_matrix_deg", "readout_table"]


def misorientation_matrix_deg(a, b, space_group: int) -> np.ndarray:
    """``(na, nb)`` symmetry-reduced misorientation angles in degrees between two orientation lists (``midas_stress``)."""
    from midas_stress.orientation import misorientation_om_batch

    a = np.asarray(a, dtype=float).reshape(-1, 3, 3)
    b = np.asarray(b, dtype=float).reshape(-1, 3, 3)
    ia, ib = np.meshgrid(np.arange(len(a)), np.arange(len(b)), indexing="ij")
    d = misorientation_om_batch(a[ia.ravel()], b[ib.ravel()], space_group)
    return np.degrees(d).reshape(len(a), len(b))


def _w1(cost: np.ndarray, wa: np.ndarray, wb: np.ndarray) -> float:
    na, nb = cost.shape
    if na == 1:
        return float(cost[0] @ wb)
    if nb == 1:
        return float(cost[:, 0] @ wa)
    from scipy.optimize import linprog

    a_eq = np.zeros((na + nb, na * nb))
    for i in range(na):
        a_eq[i, i * nb:(i + 1) * nb] = 1.0
    for j in range(nb):
        a_eq[na + j, j::nb] = 1.0
    r = linprog(cost.ravel(), A_eq=a_eq[:-1], b_eq=np.concatenate([wa, wb])[:-1], bounds=(0, None), method="highs")
    if r.status != 0:                                           # pragma: no cover - transport problems are always feasible
        raise RuntimeError(f"transport problem failed: {r.message}")
    return float(r.fun)


@dataclass
class VoxelODF:
    """The atoms of one voxel after the weight floor: ``index`` (node numbers, descending weight), ``weight`` (sums to 1),
    ``orientations`` ``(n, 3, 3)``. ``total`` is the raw weight sum before flooring and ``dropped`` the fraction of it the floor / top-k removed."""

    index: np.ndarray
    weight: np.ndarray
    orientations: np.ndarray
    space_group: int
    floor: float
    total: float
    dropped: float

    def __len__(self) -> int:
        return len(self.index)

    @property
    def top_weight(self) -> float:
        return float(self.weight[0]) if len(self.weight) else float("nan")

    def top(self, n: int):
        """The ``n`` heaviest atoms: ``(node indices, weights)`` (weights are the floored, renormalised ones)."""
        return self.index[:n], self.weight[:n]

    def medoid(self) -> int:
        """Node index of the weighted medoid: the atom with the smallest weight-averaged misorientation to all atoms. Always a real node."""
        cost = misorientation_matrix_deg(self.orientations, self.orientations, self.space_group)
        return int(self.index[int(np.argmin(cost @ self.weight))])

    def mean_orientation(self, radius_deg: float):
        """Weighted mean over the atoms within ``radius_deg`` of the medoid. Returns ``(U, mass)``: the mean orientation matrix and the share
        of the (floored) weight that was inside the radius.

        Each atom is first replaced by its crystal-symmetry variant nearest the medoid (a mean of unreduced variants is meaningless), then
        averaged as a chordal quaternion mean with the weights.
        """
        from scipy.spatial.transform import Rotation

        pos = int(np.flatnonzero(self.index == self.medoid())[0])
        centre = self.orientations[pos]
        d = misorientation_matrix_deg(centre[None], self.orientations, self.space_group)[0]
        inside = d <= radius_deg
        sym = _laue_matrices(self.space_group)
        var = np.einsum("aij,sjk->asik", self.orientations[inside], sym)                  # U S for every operator
        score = np.einsum("ij,asij->as", centre, var)                                      # trace(centre^T U S)
        reduced = var[np.arange(var.shape[0]), np.argmax(score, axis=1)]
        w = self.weight[inside]
        mean = Rotation.from_matrix(reduced).mean(weights=w).as_matrix()
        return mean, float(w.sum())

    def mass_within(self, reference, tol_deg: float) -> float:
        """Share of the (floored) weight whose atom lies within ``tol_deg`` of ANY orientation in ``reference`` ``(R, 3, 3)``."""
        d = misorientation_matrix_deg(self.orientations, reference, self.space_group)
        return float(self.weight[d.min(axis=1) <= tol_deg].sum())

    def w1(self, reference, reference_weights=None) -> float:
        """Wasserstein-1 distance (degrees, ground cost = symmetry-reduced misorientation) between this voxel's floored ODF and a reference
        set (uniform weights unless given). Depends on :attr:`floor`; always report it."""
        ref = np.asarray(reference, dtype=float).reshape(-1, 3, 3)
        wr = np.full(len(ref), 1.0 / len(ref)) if reference_weights is None else np.asarray(reference_weights, dtype=float)
        wr = wr / wr.sum()
        cost = misorientation_matrix_deg(self.orientations, ref, self.space_group)
        return _w1(cost, self.weight, wr)

    def merged(self, tol_deg: float) -> "VoxelODF":
        """Merge atoms within ``tol_deg`` into the heaviest one of each cluster (weights summed)."""
        m = merge_nodes(self.orientations, tol_deg, self.space_group, priority=self.weight)
        w = np.array([self.weight[m.labels == j].sum() for j in range(len(m.reps))])
        order = np.argsort(-w, kind="stable")
        reps = m.reps[order]
        return VoxelODF(index=self.index[reps], weight=w[order], orientations=self.orientations[reps], space_group=self.space_group,
                        floor=self.floor, total=self.total, dropped=self.dropped)


def voxel_odf(coef_row, nodes, *, space_group: int, floor: float, top: int | None = None) -> VoxelODF:
    """Atoms of one voxel from its node weights ``coef_row`` ``(K,)`` and the node orientations ``nodes`` ``(K, 3, 3)``.

    ``floor`` (keyword, required): atoms with weight below ``floor`` times the voxel's total weight are dropped, then ``top`` (if given)
    keeps the heaviest, then the weights are renormalised to 1. Use ``floor=0`` for no floor -- but say so.
    """
    c = np.asarray(coef_row, dtype=float).ravel()
    if not 0.0 <= floor < 1.0:
        raise ValueError("floor is a fraction of the voxel's total weight, in [0, 1)")
    tot = float(c.sum())
    if not tot > 0:
        return VoxelODF(np.zeros(0, int), np.zeros(0), np.zeros((0, 3, 3)), space_group, floor, 0.0, 1.0)
    idx = np.flatnonzero(c > 0)
    idx = idx[np.argsort(-c[idx], kind="stable")]
    idx = idx[c[idx] >= floor * tot]
    if top is not None:
        idx = idx[:top]
    w = c[idx]
    return VoxelODF(index=idx, weight=w / w.sum(), orientations=np.asarray(nodes, dtype=float)[idx], space_group=space_group,
                    floor=float(floor), total=tot, dropped=1.0 - float(w.sum()) / tot)


def readout_table(coef, nodes, truth, *, space_group: int, floors, tol_deg: float = 1.5, merge_tol_deg: float | None = None) -> dict:
    """Per-voxel readout against per-voxel reference sets, at several floors.

    ``coef`` ``(n_vox, K)``, ``truth`` a sequence of ``(R_v, 3, 3)`` arrays (uniform weights), ``floors`` an explicit sequence of floors.
    Returns ``{floor: dict(mass=, w1=, n_atoms=, top_weight=, medoid_err_deg=)}`` with ``(n_vox,)`` arrays; voxels with no weight give NaN.
    ``merge_tol_deg`` merges near-duplicate atoms first when given. ``mass`` is :meth:`VoxelODF.mass_within` at ``tol_deg``,
    ``medoid_err_deg`` the misorientation of the medoid to the nearest reference orientation.
    """
    coef = np.asarray(coef)
    out = {}
    for f in floors:
        keys = ("mass", "w1", "n_atoms", "top_weight", "medoid_err_deg")
        res = {k: np.full(len(coef), np.nan) for k in keys}
        for v in range(len(coef)):
            odf = voxel_odf(coef[v], nodes, space_group=space_group, floor=f)
            if len(odf) == 0:
                continue
            if merge_tol_deg is not None:
                odf = odf.merged(merge_tol_deg)
            ref = np.asarray(truth[v], dtype=float).reshape(-1, 3, 3)
            res["mass"][v] = odf.mass_within(ref, tol_deg)
            res["w1"][v] = odf.w1(ref)
            res["n_atoms"][v] = len(odf)
            res["top_weight"][v] = odf.top_weight
            med = np.flatnonzero(odf.index == odf.medoid())[0]
            res["medoid_err_deg"][v] = misorientation_matrix_deg(odf.orientations[med][None], ref, space_group)[0].min()
        out[float(f)] = res
    return out
