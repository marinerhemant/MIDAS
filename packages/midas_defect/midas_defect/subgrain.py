"""P3 — Sub-grain decomposition within asterism patches.

Within each per-hkl asterism patch, look for discrete sub-clusters of bright
voxels. Each sub-cluster is interpreted as a coherent sub-grain whose
orientation deviates slightly from the average crystal U. Cross-hkl matching
of sub-clusters → list of sub-grains with relative misorientations.

This implementation is intentionally simple (DBSCAN-per-hkl-patch with no
cross-hkl matching yet) so it lands in v0.1 as a diagnostic; cross-hkl
matching is a v0.2 follow-up.

Differentiability: the DBSCAN step is discrete and lives in
`_discrete_dbscan_patch`; downstream sub-spot refinement reuses the
differentiable `asterism_fit.fit_single_patch`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple, Union

import math
import numpy as np

from .asterism_fit import AsterismFit, fit_single_patch


__all__ = ["SubGrain", "decompose_asterism_patches"]


@dataclass
class SubGrain:
    """One discrete sub-cluster within an asterism patch."""
    parent_hkl: Tuple[int, int, int]
    sub_index: int                  # 0, 1, 2... within the parent patch
    q_center: np.ndarray            # (3,) intensity-weighted centroid
    integrated_intensity: float
    n_voxels: int
    sigma_eig: Optional[np.ndarray] = None   # (3,) per-axis half-widths (if fitted)
    sigma_axes: Optional[np.ndarray] = None
    fit_loss: Optional[float] = None
    fit_converged: Optional[bool] = None


def _discrete_dbscan_patch(
    q_patch: np.ndarray,            # (M, 3)
    I_patch: np.ndarray,            # (M,)
    *,
    eps: float, min_intensity_frac: float, min_cluster_size: int,
) -> list[np.ndarray]:
    """A minimal DBSCAN that operates on intensity-weighted points.

    Bright voxels (I > min_intensity_frac * max(I)) form the "core" set;
    standard DBSCAN with `eps` is run on the cores. Returns a list of arrays
    of indices into the original `q_patch`/`I_patch` arrays — one entry per
    cluster, with the unclustered points dropped.
    """
    if len(q_patch) == 0:
        return []
    threshold = min_intensity_frac * float(I_patch.max())
    core_mask = I_patch > threshold
    core_idx = np.where(core_mask)[0]
    if len(core_idx) < min_cluster_size:
        return []

    pts = q_patch[core_idx]
    eps2 = eps * eps
    n = len(pts)
    labels = -np.ones(n, dtype=np.int64)
    cur_label = 0

    for i in range(n):
        if labels[i] != -1:
            continue
        # find neighbors of i
        diff = pts - pts[i:i + 1]
        d2 = (diff * diff).sum(axis=1)
        neigh = np.where(d2 < eps2)[0]
        if len(neigh) < min_cluster_size:
            continue
        # start a new cluster
        labels[i] = cur_label
        stack = list(neigh)
        while stack:
            j = stack.pop()
            if labels[j] == -1:
                labels[j] = cur_label
                diff2 = pts - pts[j:j + 1]
                d22 = (diff2 * diff2).sum(axis=1)
                more = np.where(d22 < eps2)[0]
                if len(more) >= min_cluster_size:
                    stack.extend(int(m) for m in more if labels[m] == -1)
        cur_label += 1

    out = []
    for k in range(cur_label):
        members = core_idx[labels == k]
        if len(members) >= min_cluster_size:
            out.append(members)
    return out


def decompose_asterism_patches(
    qx: np.ndarray, qy: np.ndarray, qz: np.ndarray, intensity: np.ndarray,
    fits: Sequence[AsterismFit],
    *,
    eps: float = 0.02,                  # 1/Å; sub-cluster reach
    min_intensity_frac: float = 0.5,    # only bright voxels seed clusters
    min_cluster_size: int = 5,
    crop_q_pad: float = 1.5,            # crop half-width = pad * patch σ_max
    refine_sub_fit: bool = False,        # 3-D Gaussian per sub-cluster (slower)
    device: Optional[Union[str, "torch.device"]] = None,
) -> List[SubGrain]:
    """For each asterism patch, run DBSCAN on its bright voxels.

    Returns a flat list of `SubGrain` across all patches. Patches with only
    one sub-cluster (the "main" asterism core) contribute one entry; patches
    with multiple coherent sub-cores contribute one entry per cluster.
    """
    import torch  # only needed if refine_sub_fit=True
    q_all = np.stack([qx, qy, qz], axis=1)
    out: List[SubGrain] = []
    for f in fits:
        crop_half = crop_q_pad * float(f.sigma_eig.max())
        in_box = np.all(np.abs(q_all - f.q_fit[None, :]) < crop_half, axis=1)
        if in_box.sum() < min_cluster_size:
            continue
        q_patch = q_all[in_box]
        I_patch = intensity[in_box]
        clusters = _discrete_dbscan_patch(
            q_patch, I_patch,
            eps=eps, min_intensity_frac=min_intensity_frac,
            min_cluster_size=min_cluster_size,
        )
        for k, member_idx in enumerate(clusters):
            q_sub = q_patch[member_idx]
            I_sub = I_patch[member_idx]
            weights = I_sub / I_sub.sum()
            centroid = (weights[:, None] * q_sub).sum(axis=0)
            sub = SubGrain(
                parent_hkl=f.hkl, sub_index=k,
                q_center=centroid,
                integrated_intensity=float(I_sub.sum()),
                n_voxels=int(len(member_idx)),
            )
            if refine_sub_fit and len(q_sub) >= 6:
                try:
                    q_t = torch.as_tensor(q_sub, dtype=torch.float64,
                                          device=device or "cpu")
                    I_t = torch.as_tensor(I_sub, dtype=torch.float64,
                                          device=device or "cpu")
                    q0_t = torch.as_tensor(centroid, dtype=torch.float64,
                                            device=device or "cpu")
                    fit = fit_single_patch(
                        q_t, I_t, q0_t,
                        sigma_init=max(eps * 0.5, 0.005),
                        n_steps=200, lr=1e-2,
                    )
                    sub.sigma_eig = fit["sigma_eig"]
                    sub.sigma_axes = fit["sigma_axes"]
                    sub.fit_loss = fit["final_loss"]
                    sub.fit_converged = fit["converged"]
                except Exception:
                    pass
            out.append(sub)
    return out
