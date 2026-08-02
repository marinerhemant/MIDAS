"""Standard reconstruction maps: orientation, confidence, grains."""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .ipf import ipf_rgb
from .mic import MicMap, read_mic

__all__ = [
    "orientation_map", "confidence_map", "grain_labels", "grain_map",
    "compare_maps",
]

# A low-confidence region on an orientation map looks like microstructure --
# the fit returns *an* orientation for every voxel it is asked about, whether
# or not there is material there. Anything below this is annotated on the
# figure rather than left for the reader to infer.
TRUST_FLOOR = 0.3


def _marker_size(pitch_um: float, ax_span_um: float = 1200.0) -> float:
    if pitch_um <= 0:
        return 4.0
    return max(1.0, (pitch_um / 2.5) ** 2 * 1.6)


def orientation_map(
    mic: MicMap | str,
    ax=None,
    *,
    space_group: int = 225,
    cmin: float = 0.1,
    axis: Sequence[float] = (0.0, 0.0, 1.0),
    show_unindexed: bool = True,
    annotate_trust: bool = True,
    title: Optional[str] = None,
):
    """IPF map of a ``.mic``.

    ``cmin`` below :data:`TRUST_FLOOR` is allowed but annotated: the fit assigns
    an orientation to every voxel it evaluates, so a permissive cut fills the
    whole grid with plausible-looking colour whether or not material is there.
    """
    import matplotlib.pyplot as plt

    if not isinstance(mic, MicMap):
        mic = read_mic(mic)
    if ax is None:
        _, ax = plt.subplots(figsize=(6.4, 6.4))

    k = mic.mask(cmin)
    s = _marker_size(mic.pitch)
    if show_unindexed:
        ax.scatter(mic.x, mic.y, c="0.94", s=s * 0.5, linewidths=0)
    if k.any():
        ax.scatter(mic.x[k], mic.y[k],
                   c=ipf_rgb(mic.euler[k], space_group, axis), s=s,
                   linewidths=0)
    ax.set_aspect("equal")
    ax.set_xlabel("x (um)")
    ax.set_ylabel("y (um)")
    ax.set_title(title or f"{mic.path.name}\n{int(k.sum()):,} of {len(mic):,} "
                          f"voxels at C >= {cmin}", fontsize=10)
    if annotate_trust and cmin < TRUST_FLOOR:
        ax.text(0.02, 0.02,
                f"C >= {cmin} is BELOW the trust floor ({TRUST_FLOOR}).\n"
                "Low-confidence colour is an assigned orientation,\n"
                "not evidence of material.",
                transform=ax.transAxes, fontsize=7.5, va="bottom",
                bbox=dict(boxstyle="round", fc="#fff3cd", ec="#d39e00",
                          alpha=0.9))
    return ax


def confidence_map(
    mic: MicMap | str, ax=None, *, vmin: float = 0.0,
    vmax: Optional[float] = None, cmap: str = "viridis",
    title: Optional[str] = None,
):
    """Confidence (FracOverlap) map with a colourbar."""
    import matplotlib.pyplot as plt

    if not isinstance(mic, MicMap):
        mic = read_mic(mic)
    if ax is None:
        _, ax = plt.subplots(figsize=(6.4, 6.4))
    vmax = float(mic.confidence.max()) if vmax is None else vmax
    sc = ax.scatter(mic.x, mic.y, c=mic.confidence, s=_marker_size(mic.pitch),
                    cmap=cmap, vmin=vmin, vmax=vmax, linewidths=0)
    ax.figure.colorbar(sc, ax=ax, fraction=0.046)
    ax.set_aspect("equal")
    ax.set_xlabel("x (um)")
    ax.set_ylabel("y (um)")
    ax.set_title(title or f"{mic.path.name}\nconfidence, max {vmax:.4f}",
                 fontsize=10)
    return ax


def grain_labels(
    mic: MicMap | str, *, space_group: int = 225, cmin: float = 0.3,
    miso_tol_deg: float = 5.0, min_voxels: int = 3,
):
    """Label voxels into grains: spatially adjacent AND within ``miso_tol_deg``.

    Orientation connectivity is the point -- adjacency alone merges neighbouring
    grains that happen to touch.

    Returns ``(indices, labels, n_grains_with_min_voxels)``.
    """
    import torch
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree
    from midas_stress.orientation import (
        euler_to_orient_mat_batch, misorientation_om_batch,
    )

    if not isinstance(mic, MicMap):
        mic = read_mic(mic)
    k = np.where(mic.mask(cmin))[0]
    if k.size == 0:
        return k, np.zeros(0, int), 0

    pairs = cKDTree(np.column_stack([mic.x[k], mic.y[k]])).query_pairs(
        r=mic.pitch * 1.6, output_type="ndarray")
    if pairs.size:
        e = mic.euler[k]
        om = lambda a: torch.as_tensor(
            np.asarray(euler_to_orient_mat_batch(a)), dtype=torch.float64)
        miso = np.degrees(misorientation_om_batch(
            om(e[pairs[:, 0]]), om(e[pairs[:, 1]]), int(space_group)).numpy())
        linked = pairs[miso < miso_tol_deg]
    else:
        linked = np.zeros((0, 2), int)

    g = coo_matrix((np.ones(len(linked)), (linked[:, 0], linked[:, 1])),
                   shape=(k.size, k.size))
    n, lab = connected_components(g, directed=False)
    sizes = np.bincount(lab)
    return k, lab, int((sizes >= min_voxels).sum())


def grain_map(
    mic: MicMap | str, ax=None, *, space_group: int = 225, cmin: float = 0.3,
    min_voxels: int = 3, seed: int = 0, title: Optional[str] = None,
):
    """One random colour per resolved grain.

    Colours are NOT stable between figures -- only patch count and size are
    meaningful. Use :func:`orientation_map` when colour has to mean something.
    """
    import matplotlib.pyplot as plt

    if not isinstance(mic, MicMap):
        mic = read_mic(mic)
    if ax is None:
        _, ax = plt.subplots(figsize=(6.4, 6.4))
    k, lab, n_big = grain_labels(mic, space_group=space_group, cmin=cmin,
                                 min_voxels=min_voxels)
    s = _marker_size(mic.pitch)
    ax.scatter(mic.x, mic.y, c="0.94", s=s * 0.5, linewidths=0)
    if k.size:
        sizes = np.bincount(lab)
        keep = (sizes >= min_voxels)[lab]
        pal = np.random.default_rng(seed).random((max(lab.max() + 1, 1), 3))
        pal = pal * 0.75 + 0.2
        ax.scatter(mic.x[k][keep], mic.y[k][keep], c=pal[lab[keep]], s=s,
                   linewidths=0)
    ax.set_aspect("equal")
    ax.set_xlabel("x (um)")
    ax.set_ylabel("y (um)")
    ax.set_title(title or f"{mic.path.name}\n{n_big} grains "
                          f"(>={min_voxels} voxels, C >= {cmin})", fontsize=10)
    return ax


def compare_maps(
    mics: Sequence[MicMap | str], kind: str = "orientation", *,
    titles: Optional[Sequence[str]] = None, suptitle: Optional[str] = None,
    **kw,
):
    """Row of maps sharing one figure. ``kind``: orientation|confidence|grain."""
    import matplotlib.pyplot as plt

    fn = {"orientation": orientation_map, "confidence": confidence_map,
          "grain": grain_map}[kind]
    n = len(mics)
    fig, axes = plt.subplots(1, n, figsize=(6.2 * n, 6.4), squeeze=False)
    for ax, m, t in zip(axes[0], mics,
                        titles or [None] * n):
        fn(m, ax=ax, title=t, **kw)
    if suptitle:
        fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout()
    return fig
