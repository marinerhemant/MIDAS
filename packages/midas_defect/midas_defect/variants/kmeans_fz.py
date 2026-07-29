"""K-medoids variant assignment in disorientation-distance metric.

A naive K-means on FZ-reduced quaternions breaks on clusters that straddle
the cubic FZ boundary (e.g. Σ3 twins, whose canonical 60°/<111> rotation
lies on the FZ surface and spreads across multiple symmetry-equivalent
wedges after reduction). K-medoids with the symmetry-aware disorientation
angle as its distance is robust to that pathology -- it never tries to
average across the wedge boundary because medoids are existing data points,
not centroid arithmetic.

Returns the same ``dict`` shape as the original plan's k-means API to keep
downstream code agnostic to the change.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..types import CrystalPhase

_PHASE_TO_SPACE_GROUP = {
    CrystalPhase.FCC: 225,
    CrystalPhase.BCC: 229,
    CrystalPhase.HCP: 194,
}


def _om_to_quat(OM: NDArray[np.floating]) -> NDArray[np.floating]:
    """Convert (n, 3, 3) rotation matrices to (n, 4) unit quaternions (w, x, y, z)."""
    OM = np.asarray(OM, dtype=float)
    n = OM.shape[0]
    q = np.empty((n, 4), dtype=float)
    for i in range(n):
        m = OM[i]
        tr = m[0, 0] + m[1, 1] + m[2, 2]
        if tr > 0:
            s = np.sqrt(tr + 1.0) * 2.0
            q[i, 0] = 0.25 * s
            q[i, 1] = (m[2, 1] - m[1, 2]) / s
            q[i, 2] = (m[0, 2] - m[2, 0]) / s
            q[i, 3] = (m[1, 0] - m[0, 1]) / s
        elif m[0, 0] >= m[1, 1] and m[0, 0] >= m[2, 2]:
            s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
            q[i, 0] = (m[2, 1] - m[1, 2]) / s
            q[i, 1] = 0.25 * s
            q[i, 2] = (m[0, 1] + m[1, 0]) / s
            q[i, 3] = (m[0, 2] + m[2, 0]) / s
        elif m[1, 1] >= m[2, 2]:
            s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
            q[i, 0] = (m[0, 2] - m[2, 0]) / s
            q[i, 1] = (m[0, 1] + m[1, 0]) / s
            q[i, 2] = 0.25 * s
            q[i, 3] = (m[1, 2] + m[2, 1]) / s
        else:
            s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
            q[i, 0] = (m[1, 0] - m[0, 1]) / s
            q[i, 1] = (m[0, 2] + m[2, 0]) / s
            q[i, 2] = (m[1, 2] + m[2, 1]) / s
            q[i, 3] = 0.25 * s
    flip = q[:, 0] < 0
    q[flip] = -q[flip]
    return q


def _quat_to_om(q: NDArray[np.floating]) -> NDArray[np.floating]:
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    out = np.empty(q.shape[:-1] + (3, 3), dtype=float)
    out[..., 0, 0] = 1 - 2 * (y * y + z * z)
    out[..., 0, 1] = 2 * (x * y - z * w)
    out[..., 0, 2] = 2 * (x * z + y * w)
    out[..., 1, 0] = 2 * (x * y + z * w)
    out[..., 1, 1] = 1 - 2 * (x * x + z * z)
    out[..., 1, 2] = 2 * (y * z - x * w)
    out[..., 2, 0] = 2 * (x * z - y * w)
    out[..., 2, 1] = 2 * (y * z + x * w)
    out[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return out


def _disorientation_deg(q_a: NDArray[np.floating], q_b: NDArray[np.floating], space_group: int) -> float:
    """Disorientation angle in degrees between two quaternions."""
    import midas_stress.orientation as o

    om_a = _quat_to_om(q_a).ravel()
    om_b = _quat_to_om(q_b).ravel()
    ang_rad, _ = o.misorientation_om(om_a, om_b, space_group=space_group)
    return float(np.degrees(ang_rad))


def _pairwise_disorientation_matrix(
    OM: NDArray[np.floating], space_group: int
) -> NDArray[np.floating]:
    """Full (n, n) pairwise disorientation matrix in degrees."""
    import midas_stress.orientation as o

    n = OM.shape[0]
    oms = OM.reshape(n, 9)
    out = np.zeros((n, n), dtype=float)
    # Vectorize one row at a time via misorientation_om_batch.
    for i in range(n):
        a = np.broadcast_to(oms[i], (n - i - 1, 9)) if i < n - 1 else None
        if a is not None:
            b = oms[i + 1 :]
            ang = np.asarray(
                o.misorientation_om_batch(a, b, space_group=space_group), dtype=float
            )
            out[i, i + 1 :] = np.degrees(ang)
            out[i + 1 :, i] = out[i, i + 1 :]
    return out


def assign_variants_kmeans(
    OM: NDArray[np.floating],
    n_variants: int = 2,
    n_init: int = 30,
    random_state: int = 0,
    phase: CrystalPhase = CrystalPhase.FCC,
    max_iter: int = 200,
) -> dict:
    """K-medoids variant assignment using the cubic / hex disorientation metric.

    Returns
    -------
    dict with
        labels         (n_grains,)
        means_quat     (n_variants, 4)  unit quaternion of the medoid grain
        means_OM       (n_variants, 3, 3)
        disorientations (n_variants, n_variants) deg between medoids
        counts         (n_variants,)
        inertia        sum of within-cluster disorientation angles (degrees)
        random_state   the seed
    """
    if phase not in _PHASE_TO_SPACE_GROUP:
        raise ValueError(f"unknown phase {phase!r}")
    OM = np.asarray(OM, dtype=float)
    n_grains = OM.shape[0]
    if n_grains < n_variants:
        raise ValueError(
            f"need at least n_variants={n_variants} grains; got {n_grains}"
        )

    space_group = _PHASE_TO_SPACE_GROUP[phase]
    D = _pairwise_disorientation_matrix(OM, space_group)
    quats = _om_to_quat(OM)

    rng = np.random.default_rng(random_state)
    best_inertia = np.inf
    best_labels: NDArray[np.intp] | None = None
    best_medoids: NDArray[np.intp] | None = None

    for _trial in range(n_init):
        medoids = rng.choice(n_grains, size=n_variants, replace=False)
        for _ in range(max_iter):
            labels = np.argmin(D[:, medoids], axis=1)
            new_medoids = medoids.copy()
            for k in range(n_variants):
                members = np.where(labels == k)[0]
                if members.size == 0:
                    new_medoids[k] = rng.integers(n_grains)
                    continue
                # New medoid minimises sum of intra-cluster distances.
                sub = D[np.ix_(members, members)]
                best = members[np.argmin(sub.sum(axis=1))]
                new_medoids[k] = best
            if (new_medoids == medoids).all():
                medoids = new_medoids
                break
            medoids = new_medoids

        inertia = float(D[np.arange(n_grains), medoids[labels]].sum())
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels
            best_medoids = medoids

    assert best_labels is not None and best_medoids is not None
    means_quat = quats[best_medoids]
    means_OM = OM[best_medoids]
    counts = np.array([(best_labels == k).sum() for k in range(n_variants)], dtype=int)
    diso = D[np.ix_(best_medoids, best_medoids)]

    return {
        "labels": best_labels,
        "means_quat": means_quat,
        "means_OM": means_OM,
        "disorientations": diso,
        "counts": counts,
        "inertia": best_inertia,
        "random_state": random_state,
    }


__all__ = ["assign_variants_kmeans"]
