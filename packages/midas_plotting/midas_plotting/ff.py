"""Plots for far-field reconstructions (``Grains.csv``).

Far-field output is a **grain list**, not a voxel grid, so these are scatter and
distribution plots rather than the images :mod:`midas_plotting.maps` draws for
near-field ``.mic`` data. Colour, symmetry and the IPF triangle come from
:mod:`midas_plotting.ipf`, shared with the NF side so one grain gets the same
colour whichever modality found it.

Every function accepts a :class:`~midas_plotting.grains.GrainList` or a path,
takes an optional ``ax``, and returns the axes -- matching
:mod:`midas_plotting.maps`.

Two things worth knowing before reading any of these plots:

* **Grain positions are good to ~100 µm**, not to the six decimals
  ``Grains.csv`` prints (``manuals/ff-hedm/LAB_NOTEBOOK.md`` §2d). Do not over-read
  small spatial structure.
* **``GrainRadius`` is only correct with ``midas-process-grains >= 0.6.1``.**
  Older versions report approximately the sample-wide mean radius for *every*
  grain, which looks like a suspiciously monodisperse microstructure.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .grains import GrainList, read_grains
from .ipf import direction_rgb, ipf_rgb_from_matrix, laue_class, sym_matrices, CUBIC

__all__ = [
    "ipf_legend", "grain_map", "grain_map_3d", "grain_size_distribution",
    "completeness_hist", "strain_scalar", "strain_map", "strain_distribution",
    "pole_figure", "summary",
]

_PLANES = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2), "yx": (1, 0),
           "zx": (2, 0), "zy": (2, 1)}


def _as_grains(g) -> GrainList:
    return g if isinstance(g, GrainList) else read_grains(g)


def _sg(g: GrainList, space_group: Optional[int]) -> int:
    """Resolve the space group: explicit argument, else the file's own header.

    Defaulting to a hard-coded 225 would silently colour a hexagonal or
    tetragonal sample with the cubic IPF triangle -- a figure that looks
    entirely plausible and is wrong. ``Grains.csv`` states its space group in
    the ``%\tSpaceGroup:`` preamble, so use that.
    """
    if space_group is not None:
        return int(space_group)
    sg = g.space_group
    if sg is None:
        raise ValueError(
            f"{g.path.name} has no SpaceGroup in its header; pass "
            "space_group=... explicitly rather than assuming cubic.")
    return sg


def _marker_sizes(radius: Optional[np.ndarray], n: int,
                  smin: float = 12.0, smax: float = 320.0) -> np.ndarray:
    """Marker AREA from grain radius.

    Area is scaled linearly in radius, not in radius**2. A true area-accurate
    encoding makes the largest grain dominate the figure so completely that the
    rest of the microstructure is unreadable; this keeps the ordering honest
    while staying legible. Do not measure grain size off this plot -- use
    :func:`grain_size_distribution`.
    """
    if radius is None or not np.any(np.isfinite(radius)):
        return np.full(n, 40.0)
    r = np.nan_to_num(np.asarray(radius, dtype=float), nan=0.0)
    lo, hi = float(np.min(r)), float(np.max(r))
    if hi <= lo:
        return np.full(n, 60.0)
    return smin + (smax - smin) * (r - lo) / (hi - lo)


# ─── IPF legend ─────────────────────────────────────────────────────────────
def ipf_legend(space_group: int = 225, ax=None, *, n: int = 400,
               axis_label: str = "Z", title: Optional[str] = None):
    """Draw the IPF colour key (the standard stereographic triangle).

    An IPF map without its key is not interpretable, and this package had no
    way to draw one. Colours come from :func:`midas_plotting.ipf.direction_rgb`
    -- the same function the maps use -- so the key cannot drift out of step
    with the figure it explains.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(3.6, 3.2))

    fam = laue_class(space_group)
    # Sample the stereographic plane, back-project to directions, keep the ones
    # already inside the standard triangle.
    if fam == CUBIC:
        corners = np.array([[0, 0, 1.0], [1, 0, 1.0], [1, 1, 1.0]])
    else:
        corners = np.array([[0, 0, 1.0], [1, 0, 0.0], [np.sqrt(3) / 2, 0.5, 0.0]])
    # Normalise BEFORE projecting: the stereographic map is defined on unit
    # vectors. Projecting the raw index triple puts [111] at (0.5, 0.5)
    # instead of (0.366, 0.366), so the corner marker and its label sit
    # outside the coloured triangle.
    corners = corners / np.linalg.norm(corners, axis=1, keepdims=True)
    cx = corners[:, 0] / (1.0 + corners[:, 2])
    cy = corners[:, 1] / (1.0 + corners[:, 2])

    pad = 0.02 * max(np.ptp(cx), np.ptp(cy))
    x0, x1 = cx.min() - pad, cx.max() + pad
    y0, y1 = cy.min() - pad, cy.max() + pad
    gx, gy = np.meshgrid(np.linspace(x0, x1, n), np.linspace(y0, y1, n))
    X, Y = gx.ravel(), gy.ravel()
    den = 1.0 + X ** 2 + Y ** 2
    d = np.stack([2 * X / den, 2 * Y / den, (1 - X ** 2 - Y ** 2) / den], axis=1)

    if fam == CUBIC:
        inside = (d[:, 0] >= -1e-9) & (d[:, 1] >= -1e-9) & \
                 (d[:, 1] <= d[:, 0] + 1e-9) & (d[:, 0] <= d[:, 2] + 1e-9)
    else:
        az = np.degrees(np.arctan2(d[:, 1], d[:, 0]))
        inside = (d[:, 2] >= -1e-9) & (az >= -1e-9) & (az <= 30.0 + 1e-9)

    rgb = np.ones((d.shape[0], 3))
    if inside.any():
        rgb[inside] = direction_rgb(d[inside], space_group)
    img = rgb.reshape(n, n, 3)
    alpha = inside.reshape(n, n).astype(float)

    ax.imshow(img, origin="lower", alpha=alpha,
              extent=(x0, x1, y0, y1))
    labels = (["[001]", "[101]", "[111]"] if fam == CUBIC
              else ["[0001]", r"[10$\bar{1}$0]", r"[2$\bar{1}\bar{1}$0]"])
    for (px, py), lab in zip(zip(cx, cy), labels):
        ax.plot(px, py, "k.", ms=4)
        ax.annotate(lab, (px, py), textcoords="offset points",
                    xytext=(4, 4), fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title or f"IPF-{axis_label} key (SG {space_group})", fontsize=9)
    ax.set_aspect("equal")
    return ax


# ─── grain maps ─────────────────────────────────────────────────────────────
def _colours(g: GrainList, color: str, space_group: int,
             axis: Sequence[float], cmap: str, vmin, vmax):
    """Returns (facecolors, scalar_for_colourbar or None, label)."""
    if color == "ipf":
        return ipf_rgb_from_matrix(g.orient_mat, space_group, axis), None, None
    if color == "completeness":
        v = g.completeness
        if v is None:
            raise ValueError("no Confidence column in this Grains.csv")
        return None, v, "completeness"
    if color == "radius":
        if g.radius is None:
            raise ValueError("no GrainRadius column in this Grains.csv")
        return None, g.radius, "grain radius (µm)"
    if color == "diffpos":
        if g.diff_pos is None:
            raise ValueError("no DiffPos column in this Grains.csv")
        return None, g.diff_pos, "DiffPos (µm)"
    raise ValueError(
        f"unknown color={color!r}; use 'ipf', 'completeness', 'radius' or 'diffpos'")


def grain_map(
    grains, ax=None, *, plane: str = "xy", space_group: Optional[int] = None,
    axis: Sequence[float] = (0.0, 0.0, 1.0), color: str = "ipf",
    size_by_radius: bool = True, cmin: float = 0.0, cmap: str = "viridis",
    vmin=None, vmax=None, annotate_ids: bool = False,
    title: Optional[str] = None,
):
    """Grain centres projected onto ``plane``, coloured by ``color``.

    ``plane`` is one of xy, xz, yz (or their reverses). Marker size encodes
    ``GrainRadius`` when available. ``cmin`` drops grains below a completeness.

    NOTE this is a **projection**: grains at different depths overlap, and FF
    positions carry ~100 µm uncertainty. Use :func:`grain_map_3d` to see the
    layer volume.
    """
    import matplotlib.pyplot as plt

    g = _as_grains(grains)
    space_group = _sg(g, space_group)
    if plane not in _PLANES:
        raise ValueError(f"plane must be one of {sorted(_PLANES)}, got {plane!r}")
    i, j = _PLANES[plane]

    keep = np.ones(len(g), bool)
    if cmin > 0 and g.completeness is not None:
        keep = g.completeness >= cmin
    if not keep.any():
        raise ValueError(f"no grains with completeness >= {cmin}")

    if ax is None:
        _, ax = plt.subplots(figsize=(6.0, 5.6))

    fc, scalar, clabel = _colours(g, color, space_group, axis, cmap, vmin, vmax)
    sizes = _marker_sizes(g.radius if size_by_radius else None, len(g))[keep]

    if fc is not None:
        ax.scatter(g.pos[keep, i], g.pos[keep, j], s=sizes, c=fc[keep],
                   edgecolors="k", linewidths=0.3)
    else:
        sc = ax.scatter(g.pos[keep, i], g.pos[keep, j], s=sizes,
                        c=scalar[keep], cmap=cmap, vmin=vmin, vmax=vmax,
                        edgecolors="k", linewidths=0.3)
        cb = ax.figure.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label(clabel)

    if annotate_ids:
        for k in np.where(keep)[0]:
            ax.annotate(str(int(g.ids[k])), (g.pos[k, i], g.pos[k, j]),
                        fontsize=6, textcoords="offset points", xytext=(3, 3))

    names = "XYZ"
    ax.set_xlabel(f"{names[i]} (µm)")
    ax.set_ylabel(f"{names[j]} (µm)")
    ax.set_aspect("equal", adjustable="datalim")
    n_shown = int(keep.sum())
    extra = "" if n_shown == len(g) else f" of {len(g)}"
    ax.set_title(title or
                 f"{g.path.name}: {n_shown}{extra} grains, "
                 f"{plane.upper()}, colour = {color}", fontsize=10)
    return ax


def grain_map_3d(
    grains, ax=None, *, space_group: Optional[int] = None,
    axis: Sequence[float] = (0.0, 0.0, 1.0), cmin: float = 0.0,
    size_by_radius: bool = True, title: Optional[str] = None,
):
    """IPF-coloured 3-D scatter of grain centres.

    Useful for a few hundred grains to get a sense of the illuminated volume;
    beyond that it is cluttered and :func:`grain_map` projections read better.
    """
    import matplotlib.pyplot as plt

    g = _as_grains(grains)
    space_group = _sg(g, space_group)
    keep = np.ones(len(g), bool)
    if cmin > 0 and g.completeness is not None:
        keep = g.completeness >= cmin
    if not keep.any():
        raise ValueError(f"no grains with completeness >= {cmin}")

    if ax is None:
        fig = plt.figure(figsize=(6.4, 5.8))
        ax = fig.add_subplot(111, projection="3d")

    rgb = ipf_rgb_from_matrix(g.orient_mat, space_group, axis)
    sizes = _marker_sizes(g.radius if size_by_radius else None, len(g))[keep]
    ax.scatter(g.pos[keep, 0], g.pos[keep, 1], g.pos[keep, 2],
               s=sizes, c=rgb[keep], edgecolors="k", linewidths=0.3, depthshade=False)
    ax.set_xlabel("X (µm)"); ax.set_ylabel("Y (µm)"); ax.set_zlabel("Z (µm)")
    ax.set_title(title or f"{g.path.name}: {int(keep.sum())} grains (IPF)",
                 fontsize=10)
    return ax


# ─── distributions ──────────────────────────────────────────────────────────
def grain_size_distribution(grains, ax=None, *, bins: int = 30,
                            cmin: float = 0.0, title: Optional[str] = None):
    """Histogram of ``GrainRadius`` with median/mean marked."""
    import matplotlib.pyplot as plt

    g = _as_grains(grains)
    if g.radius is None:
        raise ValueError("no GrainRadius column in this Grains.csv")
    keep = np.isfinite(g.radius)
    if cmin > 0 and g.completeness is not None:
        keep &= g.completeness >= cmin
    r = g.radius[keep]
    if r.size == 0:
        raise ValueError("no grains left after filtering")

    if ax is None:
        _, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.hist(r, bins=bins, color="#4a7fb5", edgecolor="k", linewidth=0.4)
    med, mean = float(np.median(r)), float(np.mean(r))
    ax.axvline(med, color="#e8453c", lw=1.6, label=f"median {med:.1f} µm")
    ax.axvline(mean, color="k", ls="--", lw=1.2, label=f"mean {mean:.1f} µm")
    ax.set_xlabel("grain radius (µm)")
    ax.set_ylabel("grains")
    ax.legend(fontsize=8)
    ax.set_title(title or f"{g.path.name}: {r.size} grains", fontsize=10)
    return ax


def completeness_hist(grains, ax=None, *, bins: int = 30,
                      title: Optional[str] = None):
    """Histogram of per-grain completeness (``Confidence``)."""
    import matplotlib.pyplot as plt

    g = _as_grains(grains)
    if g.completeness is None:
        raise ValueError("no Confidence column in this Grains.csv")
    if ax is None:
        _, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.hist(g.completeness, bins=bins, range=(0, 1),
            color="#2f855a", edgecolor="k", linewidth=0.4)
    med = float(np.median(g.completeness))
    ax.axvline(med, color="#e8453c", lw=1.6, label=f"median {med:.3f}")
    ax.set_xlabel("completeness")
    ax.set_ylabel("grains")
    ax.legend(fontsize=8)
    ax.set_title(title or f"{g.path.name}: {len(g)} grains", fontsize=10)
    return ax


# ─── strain ─────────────────────────────────────────────────────────────────
def strain_scalar(grains, kind: str = "hydrostatic", *,
                  convention: str = "ken") -> np.ndarray:
    """Reduce the per-grain strain tensor to one number per grain.

    ``convention`` defaults to **eKen**, matching
    ``midas_stress.io.read_grains_csv``'s ``strain`` key and its documented
    recommendation, so a strain map from this module and a stress computed by
    ``midas_stress`` from the same Grains.csv are the same tensor. It used to
    default to ``"fab"``; both sites labelled their choice, so nothing was
    mis-read, but the two answers silently differed. Pass ``convention="fab"``
    for the lattice-parameter form.

    ``kind``:
      ``hydrostatic``  trace/3
      ``vonmises``     von Mises equivalent of the deviatoric part
      ``11``/``22``/``33``/``12``/``13``/``23``  a single component

    Returned in **microstrain**, which is the unit ``Grains.csv`` already
    stores -- the ``eFab``/``eKen`` columns are NOT dimensionless strain and
    must not be scaled by 1e6. Verified on `Au3_cubes_ff_000008`: ``eFab``
    trace/3 gives 245.7 / 265.7 while the independent lattice dilation
    ``(a - a0)/a0`` gives 390.6 / 405.3 µε -- same unit, same order. Reading
    them as dimensionless would report a physically impossible 2.3e8 µε.
    """
    g = _as_grains(grains)
    e = g.strain(convention)
    k = kind.lower()
    if k in ("hyd", "hydro", "hydrostatic", "mean"):
        v = np.trace(e, axis1=1, axis2=2) / 3.0
    elif k in ("vm", "vonmises", "von_mises", "equivalent"):
        dev = e - (np.trace(e, axis1=1, axis2=2) / 3.0)[:, None, None] * np.eye(3)
        v = np.sqrt(2.0 / 3.0 * np.einsum("nij,nij->n", dev, dev))
    elif len(k) == 2 and set(k) <= set("123"):
        i, j = int(k[0]) - 1, int(k[1]) - 1
        v = e[:, i, j]
    else:
        raise ValueError(
            f"unknown strain kind {kind!r}; use 'hydrostatic', 'vonmises' "
            "or a component like '11' or '13'")
    return v


def strain_map(grains, ax=None, *, kind: str = "hydrostatic",
               convention: str = "ken", plane: str = "xy", cmin: float = 0.0,
               cmap: str = "coolwarm", vmin=None, vmax=None,
               symmetric: bool = True, title: Optional[str] = None):
    """Grain map coloured by a strain scalar (µε).

    For signed quantities the colour scale is symmetric about zero by default,
    so the sign is readable and a diverging colormap means what it looks like.
    """
    import matplotlib.pyplot as plt

    g = _as_grains(grains)
    if plane not in _PLANES:
        raise ValueError(f"plane must be one of {sorted(_PLANES)}")
    i, j = _PLANES[plane]
    v = strain_scalar(g, kind, convention=convention)

    keep = np.isfinite(v)
    if cmin > 0 and g.completeness is not None:
        keep &= g.completeness >= cmin
    if not keep.any():
        raise ValueError("no grains left after filtering")

    if ax is None:
        _, ax = plt.subplots(figsize=(6.4, 5.6))
    if symmetric and vmin is None and vmax is None:
        lim = float(np.nanmax(np.abs(v[keep]))) or 1.0
        vmin, vmax = -lim, lim
    sizes = _marker_sizes(g.radius, len(g))[keep]
    sc = ax.scatter(g.pos[keep, i], g.pos[keep, j], s=sizes, c=v[keep],
                    cmap=cmap, vmin=vmin, vmax=vmax,
                    edgecolors="k", linewidths=0.3)
    cb = ax.figure.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(f"{kind} strain (µε), {convention}")
    names = "XYZ"
    ax.set_xlabel(f"{names[i]} (µm)"); ax.set_ylabel(f"{names[j]} (µm)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title(title or f"{g.path.name}: {kind} strain", fontsize=10)
    return ax


def strain_distribution(grains, ax=None, *, convention: str = "ken",
                        bins: int = 30, title: Optional[str] = None):
    """Histograms of the three normal strain components (µε)."""
    import matplotlib.pyplot as plt

    g = _as_grains(grains)
    if ax is None:
        _, ax = plt.subplots(figsize=(6.4, 4.0))
    for comp, colr in (("11", "#e8453c"), ("22", "#2f855a"), ("33", "#2b6cb0")):
        v = strain_scalar(g, comp, convention=convention)
        ax.hist(v, bins=bins, histtype="step", lw=1.6, color=colr,
                label=f"ε{comp}  median {np.median(v):+.0f} µε")
    ax.axvline(0.0, color="k", lw=0.8, ls=":")
    ax.set_xlabel(f"strain (µε), {convention}")
    ax.set_ylabel("grains")
    ax.legend(fontsize=8)
    ax.set_title(title or f"{g.path.name}: {len(g)} grains", fontsize=10)
    return ax


# ─── pole figure ────────────────────────────────────────────────────────────
def pole_figure(grains, ax=None, *, hkl: Sequence[float] = (0, 0, 1),
                space_group: Optional[int] = None, cmin: float = 0.0,
                projection: str = "stereographic", color: str = "ipf",
                axis: Sequence[float] = (0.0, 0.0, 1.0),
                title: Optional[str] = None):
    """Discrete pole figure of a crystal direction, over all grains.

    Every symmetry equivalent of ``hkl`` is plotted for every grain, projected
    onto the upper hemisphere. With few grains this is a scatter of poles, not
    a texture density -- do not read it as an ODF.

    ``projection``: ``stereographic`` (equal-angle, the usual choice for
    reading orientations) or ``equal_area`` (Schmidt, the usual choice when
    comparing *densities*, since it does not distort area).
    """
    import matplotlib.pyplot as plt

    g = _as_grains(grains)
    space_group = _sg(g, space_group)
    keep = np.ones(len(g), bool)
    if cmin > 0 and g.completeness is not None:
        keep = g.completeness >= cmin
    if not keep.any():
        raise ValueError(f"no grains with completeness >= {cmin}")

    om = g.orient_mat[keep]
    h = np.asarray(hkl, dtype=float)
    h = h / np.linalg.norm(h)
    sym = sym_matrices(space_group)
    hs = np.einsum("sij,j->si", sym, h)                    # equivalents

    # g maps sample -> crystal, so the sample-frame pole is g.T @ h_crystal.
    d = np.einsum("nji,sj->nsi", om, hs).reshape(-1, 3)
    d = d / np.linalg.norm(d, axis=1, keepdims=True)
    d[d[:, 2] < 0] *= -1.0                                 # upper hemisphere

    if projection.startswith("stereo"):
        X, Y = d[:, 0] / (1.0 + d[:, 2]), d[:, 1] / (1.0 + d[:, 2])
    elif projection.startswith("equal"):
        f = np.sqrt(2.0 / (1.0 + d[:, 2]))
        X, Y = d[:, 0] * f / np.sqrt(2), d[:, 1] * f / np.sqrt(2)
    else:
        raise ValueError("projection must be 'stereographic' or 'equal_area'")

    if ax is None:
        _, ax = plt.subplots(figsize=(5.0, 5.0))
    if color == "ipf":
        rgb = ipf_rgb_from_matrix(om, space_group, axis)
        c = np.repeat(rgb, len(sym), axis=0)
    else:
        c = color
    ax.scatter(X, Y, s=14, c=c, edgecolors="k", linewidths=0.2)
    th = np.linspace(0, 2 * np.pi, 361)
    ax.plot(np.cos(th), np.sin(th), "k-", lw=1.0)
    ax.plot([-1, 1], [0, 0], "k:", lw=0.6)
    ax.plot([0, 0], [-1, 1], "k:", lw=0.6)
    ax.set_xlim(-1.08, 1.08); ax.set_ylim(-1.08, 1.08)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    lab = "".join(str(int(v)) for v in hkl)
    ax.set_title(title or
                 f"{{{lab}}} pole figure — {int(keep.sum())} grains, "
                 f"{projection}", fontsize=10)
    return ax


# ─── overview ───────────────────────────────────────────────────────────────
def summary(grains, *, space_group: Optional[int] = None, cmin: float = 0.0,
            axis: Sequence[float] = (0.0, 0.0, 1.0), figsize=(13.0, 8.0)):
    """One-page overview: IPF map + key, size, completeness, strain, poles.

    The 'is this reconstruction sane' figure. Returns the Figure.
    """
    import matplotlib.pyplot as plt

    g = _as_grains(grains)
    space_group = _sg(g, space_group)
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 3, hspace=0.32, wspace=0.30)

    grain_map(g, fig.add_subplot(gs[0, 0]), space_group=space_group,
              axis=axis, cmin=cmin, title="grain map (IPF-Z)")
    ipf_legend(space_group, fig.add_subplot(gs[0, 1]))
    try:
        pole_figure(g, fig.add_subplot(gs[0, 2]), hkl=(0, 0, 1),
                    space_group=space_group, cmin=cmin, axis=axis)
    except Exception as e:                                  # noqa: BLE001
        fig.add_subplot(gs[0, 2]).set_title(f"pole figure unavailable: {e}",
                                            fontsize=8)
    try:
        grain_size_distribution(g, fig.add_subplot(gs[1, 0]), cmin=cmin,
                                title="grain size")
    except Exception as e:                                  # noqa: BLE001
        fig.add_subplot(gs[1, 0]).set_title(f"size unavailable: {e}", fontsize=8)
    try:
        completeness_hist(g, fig.add_subplot(gs[1, 1]), title="completeness")
    except Exception as e:                                  # noqa: BLE001
        fig.add_subplot(gs[1, 1]).set_title(f"completeness unavailable: {e}",
                                            fontsize=8)
    try:
        strain_distribution(g, fig.add_subplot(gs[1, 2]), title="strain")
    except Exception as e:                                  # noqa: BLE001
        fig.add_subplot(gs[1, 2]).set_title(f"strain unavailable: {e}", fontsize=8)

    fig.suptitle(f"{g.path.name} — {len(g)} grains", fontsize=12)
    return fig
