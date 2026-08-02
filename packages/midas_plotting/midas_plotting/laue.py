"""Plots for Laue microdiffraction reconstructions.

Namespaced like :mod:`midas_plotting.ff` rather than exported flat, because
``pole_figure`` and ``grain_map`` already mean different things per modality and
flattening them would collide.

Laue differs from FF and NF in ways the plots have to respect:

* **A "grain" is a cluster of per-frame orientations**, not a row in a file.
  Nothing is a grain until it has been clustered at a stated tolerance, so every
  grain number here carries the tolerance that produced it.
* **An orientation found at every raster position is not a grain.** The beam
  moves a micron or two between frames, so a real grain leaves the probe volume.
  Anything spanning more than half the map is the substrate, a detector
  artefact, or a mis-index -- :func:`cluster` flags it and the plots exclude it.
* **Reading a pole figure needs its chance level.** Half of a randomly oriented
  population has its c-axis more than 60 deg from any fixed direction, purely
  from solid angle. Quoting "70% of grains lie near the surface plane" without
  that reference turns a near-random distribution into a texture.
  :func:`tilt_histogram` draws the reference by default.

The 45 deg stage geometry and the surface normal are arguments with 34-ID-E
defaults, never hard-coded: they differ by beamline and a wrong normal rotates
every pole figure without any other symptom.
"""
from __future__ import annotations

import warnings
from typing import Optional, Sequence

import numpy as np

from .ipf import direction_rgb, laue_class, sym_matrices  # re-exported for tests/callers
from .solutions import COS45, LaueSolutions, LaueSpots

__all__ = [
    "SURFACE_NORMAL_34IDE", "cluster", "GrainClusters", "effective_n",
    "misorientation_matrix", "sym_matrices",
    "orientation_map", "pole_figure", "grain_size_distribution",
    "tilt_histogram", "random_tilt_fractions", "texture_strength",
    "tolerance_sweep", "spot_overlay", "occupancy_map", "summary",
]

#: Sample surface normal in the 34-ID-E sample frame, for a specimen mounted at
#: 45 deg. Every tilt and azimuth here is measured against this.
SURFACE_NORMAL_34IDE = np.array([0.0, -COS45, COS45])

#: Complete-linkage clustering needs every pairwise misorientation, so both time
#: and memory go as N^2. Above this the pairwise matrix stops fitting
#: comfortably in RAM (6000 instances is ~144 MB as float32) and the honest
#: response is to refuse with an instruction rather than to appear to hang.
MAX_CLUSTER = 6000


def _unit(v):
    v = np.asarray(v, float).ravel()
    n = np.linalg.norm(v)
    if n == 0:
        raise ValueError("direction must be non-zero")
    return v / n


def _axis_dirs(om: np.ndarray, hkl=(0, 0, 1), *, normal=None, B=None):
    """Sample-frame unit vectors of a crystal direction, folded to one hemisphere.

    ``+d`` and ``-d`` are the same axis, so the sign is fixed by the surface
    normal. Without that fold a single population straddles both hemispheres and
    every density is halved somewhere.
    """
    om = np.asarray(om, float).reshape(-1, 3, 3)
    if om.size == 0:
        return np.zeros((0, 3))
    h = _unit(hkl)
    v = np.einsum("nij,j->ni", om if B is None else om @ np.asarray(B), h)
    v = v / np.linalg.norm(v, axis=1, keepdims=True)
    n = SURFACE_NORMAL_34IDE if normal is None else _unit(normal)
    return v * np.sign(v @ n)[:, None]


def effective_n(sizes) -> float:
    """Kish effective sample size of a set of cluster sizes.

    ``(sum w)^2 / sum w^2``. The number to quote next to a grain count: where a
    few large clusters dominate it falls far below the nominal count, and every
    per-grain percentage is correspondingly less certain than it looks. One G31
    map had 99 clusters and an effective n of 2.5.
    """
    s = np.asarray(sizes, float).ravel()
    if s.size == 0 or s.sum() == 0:
        return 0.0
    w = s / s.sum()
    return float(1.0 / np.sum(w ** 2))


class GrainClusters:
    """Result of :func:`cluster`: labels, sizes, extents and the full-field flag."""

    def __init__(self, labels, sizes, extent, full_field, tolerance, pos=None):
        self.labels = labels
        self.sizes = sizes
        self.extent = extent
        self.full_field = full_field
        self.tolerance = float(tolerance)
        self.pos = pos

    @property
    def n_clusters(self) -> int:
        return int(self.sizes.size)

    @property
    def n_grains(self) -> int:
        """Clusters that survive the full-field filter."""
        return int((~self.full_field).sum())

    @property
    def n_eff(self) -> float:
        return effective_n(self.sizes)

    @property
    def n_eff_grains(self) -> float:
        return effective_n(self.sizes[~self.full_field])

    def representatives(self, om: np.ndarray) -> np.ndarray:
        """One orientation matrix per surviving grain."""
        om = np.asarray(om).reshape(-1, 3, 3)
        keep = np.where(~self.full_field)[0]
        return np.stack([om[self.labels == g][0] for g in keep]) if keep.size \
            else np.zeros((0, 3, 3))

    def __repr__(self) -> str:
        return (f"<GrainClusters {self.n_grains} grains at "
                f"{self.tolerance}deg (of {self.n_clusters} clusters, "
                f"{int(self.full_field.sum())} spanning >half the map), "
                f"n_eff {self.n_eff_grains:.1f}>")


def cluster(sol: LaueSolutions, tolerance: float = 1.0, *,
            space_group: int = 194, full_field_frac: float = 0.5):
    """Complete-linkage clustering of per-frame orientations into grains.

    Complete linkage ("diameter") rather than single linkage on purpose: it
    guarantees every member of a grain lies within ``tolerance`` of **every**
    other member. Single linkage only constrains nearest neighbours, which lets
    a grain drift arbitrarily far across orientation space one small step at a
    time.

    Clusters whose spatial extent exceeds ``full_field_frac`` of the map are
    flagged, not deleted -- the count of them is a diagnostic worth seeing.
    """
    om = np.asarray(sol.orient_mat).reshape(-1, 3, 3)
    n = len(om)
    if n == 0:
        z = np.zeros(0)
        return GrainClusters(np.zeros(0, int), z, z, np.zeros(0, bool),
                             tolerance, sol.pos)
    if n > MAX_CLUSTER:
        raise ValueError(
            f"{n} instances exceeds the {MAX_CLUSTER} that complete-linkage "
            f"clustering handles here (it needs the full N x N misorientation "
            f"matrix). Subsample UNIFORMLY ACROSS POSITIONS first -- "
            f"subsampling by match quality biases toward large bright grains "
            f"and changes the answer.")

    d = misorientation_matrix(om, sym_matrices(space_group))
    tol = float(tolerance)
    labels = np.full(n, -1, int)
    nxt = 0
    for i in range(n):
        if labels[i] >= 0:
            continue
        labels[i] = nxt
        members = [i]
        for j in np.where((labels < 0) & (d[i] <= tol))[0]:
            if labels[j] >= 0:
                continue
            if d[j, members].max() <= tol:          # complete linkage
                labels[j] = nxt
                members.append(int(j))
        nxt += 1

    sizes = np.bincount(labels, minlength=nxt).astype(float)
    if sol.pos is not None:
        ext = np.array([
            max(np.ptp(sol.pos[labels == g, 0]) if (labels == g).sum() > 1 else 0.0,
                np.ptp(sol.pos[labels == g, 1]) if (labels == g).sum() > 1 else 0.0)
            for g in range(nxt)])
        lim = full_field_frac * max(np.ptp(sol.pos[:, 0]), np.ptp(sol.pos[:, 1]))
        full = ext > lim
    else:
        ext = np.zeros(nxt)
        full = np.zeros(nxt, bool)
    return GrainClusters(labels, sizes, ext, full, tolerance, sol.pos)


def misorientation_matrix(om: np.ndarray, sym: np.ndarray,
                          chunk: int = 256) -> np.ndarray:
    """Symmetry-reduced pairwise misorientation angles in **degrees**, ``(N, N)``.

    ``min_S angle(A S B^T)`` over the proper rotations of the Laue group,
    evaluated in row chunks because the intermediate is ``(chunk, N, n_sym)``.
    """
    om = np.asarray(om, float).reshape(-1, 3, 3)
    n = len(om)
    out = np.zeros((n, n), np.float32)
    AS = np.einsum("nij,sjk->nsik", om, sym)        # (N, n_sym, 3, 3)
    for i0 in range(0, n, chunk):
        i1 = min(i0 + chunk, n)
        # trace(A S B^T) contracted directly, without forming the product
        tr = np.einsum("nsij,mij->nms", AS[i0:i1], om)
        c = np.clip((tr - 1.0) / 2.0, -1.0, 1.0)
        out[i0:i1] = np.degrees(np.arccos(c)).min(axis=2).astype(np.float32)
    np.fill_diagonal(out, 0.0)
    return out


# --------------------------------------------------------------------------
# the reference every Laue pole figure needs
# --------------------------------------------------------------------------

def random_tilt_fractions(edges_deg=(0, 15, 30, 45, 60, 90)) -> np.ndarray:
    """Share of a **randomly oriented** population in each tilt band.

    For directions uniform on the sphere and folded to one hemisphere, the
    fraction with tilt in ``[a, b]`` is ``cos a - cos b``. For the default bands
    that is 3.4 / 10.0 / 15.9 / 20.7 / **50.0** per cent.

    That last number is the one that matters. Half of a random population lies
    beyond 60 deg simply because that band is half the hemisphere, so a deposit
    showing "70% of grains near the surface plane" is barely above random, and
    one showing 30% is *depleted* there. Reading a tilt histogram without this
    reference is how a non-texture gets reported as a prismatic one.
    """
    e = np.radians(np.asarray(edges_deg, float))
    return np.cos(e[:-1]) - np.cos(e[1:])


def tilt_histogram(sol, ax=None, *, hkl=(0, 0, 1), normal=None,
                   edges_deg=(0, 15, 30, 45, 60, 90), weights=None,
                   reference: bool = True, label: Optional[str] = None,
                   title: Optional[str] = None):
    """Tilt of a crystal direction from the surface normal, against random.

    Parameters
    ----------
    sol : LaueSolutions or (N, 3, 3) array
        One entry per **grain** if you want a per-grain statistic; passing raw
        per-frame solutions weights the answer by how many frames each grain
        was seen on, which is a different (area-weighted) question.
    weights : (N,) array, optional
        Cluster sizes, to show the share of mapped *area* rather than of grains.
    reference : bool
        Draw the random-orientation expectation. Leave this on.
    """
    import matplotlib.pyplot as plt

    om = sol.orient_mat if isinstance(sol, LaueSolutions) else sol
    v = _axis_dirs(om, hkl, normal=normal)
    n = SURFACE_NORMAL_34IDE if normal is None else _unit(normal)
    tilt = np.degrees(np.arccos(np.clip(v @ n, -1.0, 1.0)))

    edges = np.asarray(edges_deg, float)
    w = np.ones(len(tilt)) if weights is None else np.asarray(weights, float)
    counts, _ = np.histogram(tilt, bins=edges, weights=w)
    share = 100.0 * counts / max(counts.sum(), 1e-12)
    rand = 100.0 * random_tilt_fractions(edges)

    if ax is None:
        _, ax = plt.subplots(figsize=(6.4, 4.0))
    centres = np.arange(len(share))
    bw = 0.38
    if reference:
        ax.bar(centres - bw / 2, rand, bw, color="#9AA3AB", zorder=3,
               label="random orientations")
        ax.bar(centres + bw / 2, share, bw, color="#A8452F", zorder=3,
               label=label or "measured")
        ax.legend(fontsize=9, frameon=False)
    else:
        ax.bar(centres, share, 0.62, color="#A8452F", zorder=3,
               label=label or "measured")
    ax.set_xticks(centres)
    ax.set_xticklabels([f"{a:.0f}–{b:.0f}°"
                        for a, b in zip(edges[:-1], edges[1:])])
    ax.set_xlabel(f"{tuple(int(x) for x in hkl)} tilt from the surface normal")
    ax.set_ylabel("share of grains (%)" if weights is None
                  else "share of mapped area (%)")
    ax.grid(axis="y", lw=0.4, alpha=0.5, zorder=0)
    ax.set_title(title or "Where the crystal axes point\n"
                          "grey = what random orientations give", fontsize=10)
    return ax


# --------------------------------------------------------------------------
# pole figures and texture strength
# --------------------------------------------------------------------------

def _kernel_grid(normal, step_deg=4.0):
    n = _unit(normal)
    a = np.cross(n, [1.0, 0.0, 0.0])
    if np.linalg.norm(a) < 1e-8:
        a = np.cross(n, [0.0, 1.0, 0.0])
    a = a / np.linalg.norm(a)
    b = np.cross(n, a)
    g = []
    for dec in np.arange(step_deg / 2.0, 90.0, step_deg):
        naz = max(int(round(90 * np.sin(np.radians(dec)))), 1)
        for az in np.linspace(0, 360, naz, endpoint=False):
            t, p = np.radians(dec), np.radians(az)
            g.append(np.cos(t) * n + np.sin(t) * (np.cos(p) * a + np.sin(p) * b))
    return np.asarray(g), a, b


def _mrd(dirs, grid, bandwidth_deg):
    """Kernel density on the sphere, normalised so uniform -> 1."""
    if len(dirs) == 0:
        return 0.0
    k = 1.0 / np.radians(bandwidth_deg) ** 2
    w = np.exp(k * (np.abs(grid @ dirs.T) - 1.0))
    d = w.sum(axis=1)
    return float((d / d.mean()).max())


def texture_strength(sol, *, hkl=(0, 0, 1), normal=None, bandwidth_deg=10.0,
                     n_null: int = 200, seed: int = 0):
    """Peak pole density and the chance level for **this** number of grains.

    Returns ``(peak_mrd, chance_95, ratio)``.

    The ratio is the number to compare across datasets. A raw peak density is
    not comparable between populations of different size -- a small population
    peaks higher by chance alone, and its chance level rises to match. Dividing
    each by its own null is what makes 85 grains and 631 grains commensurable.

    The null here is uniformly random orientations. If the indexing pipeline
    accepts some orientations more readily than others, an *indexability-matched*
    null is stricter and should be preferred; this one cannot see that bias.
    """
    om = sol.orient_mat if isinstance(sol, LaueSolutions) else sol
    v = _axis_dirs(om, hkl, normal=normal)
    n = SURFACE_NORMAL_34IDE if normal is None else _unit(normal)
    grid, _, _ = _kernel_grid(n)
    obs = _mrd(v, grid, bandwidth_deg)

    rng = np.random.default_rng(seed)
    null = np.empty(int(n_null))
    for i in range(int(n_null)):
        r = rng.normal(size=(len(v), 3))
        r /= np.linalg.norm(r, axis=1, keepdims=True)
        r *= np.sign(r @ n)[:, None]
        null[i] = _mrd(r, grid, bandwidth_deg)
    c95 = float(np.percentile(null, 95))
    return obs, c95, (obs / c95 if c95 > 0 else np.nan)


def pole_figure(sol, ax=None, *, hkl=(0, 0, 1), normal=None,
                sizes=None, title: Optional[str] = None, s: float = 6.0):
    """Equal-angle pole figure, centre = the surface normal.

    Rings at 30, 60 and 90 degrees. Pass grain representatives, not raw
    per-frame solutions, unless you mean to weight by residence time.
    """
    import matplotlib.pyplot as plt

    om = sol.orient_mat if isinstance(sol, LaueSolutions) else sol
    v = _axis_dirs(om, hkl, normal=normal)
    n = SURFACE_NORMAL_34IDE if normal is None else _unit(normal)
    _, a, b = _kernel_grid(n)
    dec = np.degrees(np.arccos(np.clip(v @ n, 0.0, 1.0)))
    az = np.arctan2(v @ b, v @ a)
    r = np.tan(np.radians(dec) / 2.0)

    if ax is None:
        _, ax = plt.subplots(figsize=(4.6, 4.6))
    ax.scatter(r * np.cos(az), r * np.sin(az),
               s=s if sizes is None else np.clip(np.asarray(sizes, float), 1, None),
               alpha=0.4, color="#A8542F", edgecolors="none", zorder=3)
    th = np.linspace(0, 2 * np.pi, 240)
    for d in (30, 60, 90):
        rr = np.tan(np.radians(d) / 2.0)
        ax.plot(rr * np.cos(th), rr * np.sin(th), lw=0.6, color="#999", zorder=2)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title or f"{tuple(int(x) for x in hkl)} pole figure "
                          f"({len(v)} points)\ncentre = surface normal, "
                          f"rings 30/60/90°", fontsize=10)
    return ax


# --------------------------------------------------------------------------
# maps
# --------------------------------------------------------------------------

def orientation_map(sol: LaueSolutions, ax=None, *, normal=None,
                    space_group: Optional[int] = None, hkl=(0, 0, 1),
                    color: str = "azimuth", title: Optional[str] = None):
    """Map of the scan, one colour per orientation.

    ``color='azimuth'`` (default) sets hue from the rotation about the surface
    normal and paleness from alignment with it, so a single-coloured region is
    one grain. ``color='ipf'`` uses the standard IPF triangle instead and needs
    ``space_group``.

    Where several orientations share a position the strongest (most matched
    reflections) wins the pixel, which is stated rather than silent: a Laue
    frame routinely carries more than one crystal.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import hsv_to_rgb

    if sol.pos is None:
        raise ValueError(
            "this LaueSolutions has no positions; pass positions= to "
            "read_solutions, or load a validated .npz which carries X and Z")
    v = _axis_dirs(sol.orient_mat, hkl, normal=normal)
    n = SURFACE_NORMAL_34IDE if normal is None else _unit(normal)

    if color == "ipf":
        if space_group is None:
            raise ValueError("color='ipf' needs space_group")
        laue_class(space_group)                       # refuse unknown families
        rgb = direction_rgb(np.einsum("nij,j->ni", sol.orient_mat, _unit(hkl)),
                            space_group)
    elif color == "azimuth":
        _, a, b = _kernel_grid(n)
        dec = np.degrees(np.arccos(np.clip(v @ n, 0.0, 1.0)))
        az = (np.degrees(np.arctan2(v @ b, v @ a)) % 360.0) / 360.0
        rgb = hsv_to_rgb(np.stack(
            [az, np.clip(dec / 60.0, 0.15, 1.0), np.ones_like(az)], axis=1))
    else:
        raise ValueError("color must be 'azimuth' or 'ipf'")

    x, y = sol.pos[:, 0], sol.pos[:, 1]
    ux, uy = np.unique(x), np.unique(y)
    img = np.ones((len(uy), len(ux), 3))
    xi = np.searchsorted(ux, x)
    yi = np.searchsorted(uy, y)
    for k in np.argsort(sol.n_matches):               # strongest wins the pixel
        img[yi[k], xi[k]] = rgb[k]

    if ax is None:
        _, ax = plt.subplots(figsize=(6.0, 5.0))
    ax.imshow(img, origin="lower", aspect="equal", interpolation="nearest",
              extent=[ux.min(), ux.max(), uy.min(), uy.max()])
    ax.set_xlabel("x (µm)")
    ax.set_ylabel("y (µm, sample frame)")
    ax.set_title(title or ("orientation map\nhue = rotation about the surface "
                           "normal, paleness = alignment with it"), fontsize=10)
    return ax


def grain_size_distribution(clusters: GrainClusters, ax=None, *,
                            bins: int = 30, title: Optional[str] = None):
    """Distribution of grain size in raster positions, full-field excluded."""
    import matplotlib.pyplot as plt

    s = clusters.sizes[~clusters.full_field]
    if ax is None:
        _, ax = plt.subplots(figsize=(5.2, 4.0))
    if s.size == 0:
        ax.text(0.5, 0.5, "no grains\n(every cluster spans >half the map)",
                ha="center", va="center", transform=ax.transAxes, fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
        return ax
    ax.hist(s, bins=np.logspace(0, np.log10(max(s.max(), 2)), bins),
            color="#3E6E7E")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("positions per grain")
    ax.set_ylabel("grains")
    ax.set_title(title or f"grain size distribution\n{clusters.n_grains} "
                          f"grains at {clusters.tolerance}°", fontsize=10)
    return ax


def tolerance_sweep(sol: LaueSolutions, ax=None, *,
                    tolerances: Sequence[float] = (0.5, 1.0, 2.0, 3.0, 5.0),
                    space_group: int = 194, title: Optional[str] = None):
    """Grain count and effective n against clustering tolerance.

    The tolerance is a choice, so its consequences belong on the figure: a grain
    count that moves wildly with it is telling you about the clustering, not the
    sample. The right-hand axis carries the effective n, and the annotation
    counts clusters spanning more than half the map -- on one dataset a single
    such object held 59% of all measurements and dragged the effective n from 29
    to 2.5.

    Returns ``(ax, rows)`` where rows is a list of dicts.
    """
    import matplotlib.pyplot as plt

    rows = []
    for t in tolerances:
        c = cluster(sol, t, space_group=space_group)
        rows.append(dict(tolerance=float(t), clusters=c.n_clusters,
                         grains=c.n_grains, n_eff=c.n_eff,
                         n_eff_grains=c.n_eff_grains,
                         full_field=int(c.full_field.sum()),
                         largest=int(c.sizes.max()) if c.sizes.size else 0))
    if ax is None:
        _, ax = plt.subplots(figsize=(6.2, 4.0))
    t = [r["tolerance"] for r in rows]
    ax.plot(t, [r["grains"] for r in rows], "o-", color="#A8452F",
            label="grains (full-field excluded)")
    ax.set_xlabel("clustering tolerance (°)")
    ax.set_ylabel("grains", color="#A8452F")
    ax.tick_params(axis="y", labelcolor="#A8452F")
    ax2 = ax.twinx()
    ax2.plot(t, [r["n_eff_grains"] for r in rows], "s--", color="#2F6B8F",
             label="effective n")
    ax2.set_ylabel("effective (Kish) n", color="#2F6B8F")
    ax2.tick_params(axis="y", labelcolor="#2F6B8F")
    nf = sum(r["full_field"] for r in rows)
    if nf:
        ax.annotate(f"{nf} cluster(s) across the sweep span >½ the map "
                    f"and are excluded", xy=(0.02, 0.02),
                    xycoords="axes fraction", fontsize=8.5, color="#9E4A48")
    ax.set_title(title or "How the grain count depends on the tolerance",
                 fontsize=10)
    return ax, rows


# --------------------------------------------------------------------------
# detector-frame diagnostics
# --------------------------------------------------------------------------

def spot_overlay(image: np.ndarray, spots: LaueSpots, ax=None, *,
                 grain: Optional[int] = None, percentile: float = 99.5,
                 title: Optional[str] = None, marker_size: float = 70.0):
    """A frame with an accepted orientation's assigned reflections drawn on it.

    The check no orientation result should be published without. A solution can
    match a large number of predicted reflections while sitting on the wrong
    crystal, and the only cheap way to see that is to look at where its
    reflections land against the actual peaks.

    ``spots`` should already be restricted to one frame
    (``LaueSpots.for_frame``); ``grain`` further restricts to one solution.
    """
    import matplotlib.pyplot as plt

    img = np.asarray(image)
    if img.ndim != 2:
        raise ValueError(
            f"expected a single 2-D frame, got shape {img.shape}. These files "
            f"store one image per file as a 2-D dataset, so h5[...][0] is the "
            f"first ROW, not the first frame -- read [:] instead.")
    s = spots if grain is None else _spots_for_grain(spots, grain)
    if ax is None:
        _, ax = plt.subplots(figsize=(6.4, 6.4))
    vmax = np.percentile(img, percentile)
    ax.imshow(img, cmap="gray_r", vmin=float(np.median(img)), vmax=float(vmax),
              origin="lower", interpolation="nearest")
    if len(s):
        ax.scatter(s.xy[:, 0], s.xy[:, 1], s=marker_size, facecolors="none",
                   edgecolors="#D24B3E", linewidths=1.1,
                   label=f"{len(s)} assigned reflections")
        ax.legend(fontsize=9, loc="upper right", framealpha=0.85)
    ax.set_xlabel("detector x (px)")
    ax.set_ylabel("detector y (px)")
    ax.set_title(title or "frame with assigned reflections overlaid",
                 fontsize=10)
    return ax


def _spots_for_grain(spots: LaueSpots, grain: int) -> LaueSpots:
    m = spots.grain == int(grain)
    return LaueSpots(image=spots.image[m], grain=spots.grain[m],
                     hkl=spots.hkl[m], xy=spots.xy[m],
                     qhat=None if spots.qhat is None else spots.qhat[m],
                     intensity=None if spots.intensity is None
                     else spots.intensity[m], columns=spots.columns)


def occupancy_map(peaks_per_frame, ax=None, *, shape=(2048, 2048),
                  bin_px: int = 6, hi: float = 0.8,
                  title: Optional[str] = None):
    """How often each detector position carries a peak, across the raster.

    A position that fires in nearly every frame cannot be a deposit grain: the
    beam moves microns between frames and a small grain leaves the probe volume.
    So this separates substrate from deposit **with no orientation, no lattice
    and no assumption that the substrate is a single crystal** -- which makes it
    an independent check on any substrate identified by indexing.

    Parameters
    ----------
    peaks_per_frame : sequence of (M_i, 2) arrays
        Detector x, y of the peaks detected on each sampled frame.
    hi : float
        Occupancy at or above which a bin is called stage-invariant.

    Returns ``(ax, stats)``.
    """
    import matplotlib.pyplot as plt

    nb = shape[0] // bin_px + 1
    count = np.zeros((nb, nb), int)
    total = seen = 0
    frames = list(peaks_per_frame)
    for pk in frames:
        p = np.asarray(pk, float).reshape(-1, 2)
        if not len(p):
            continue
        bx = (p[:, 0] / bin_px).astype(int)
        by = (p[:, 1] / bin_px).astype(int)
        ok = (bx >= 0) & (bx < nb) & (by >= 0) & (by < nb)
        total += int(ok.sum())
        for b, a in set(zip(by[ok].tolist(), bx[ok].tolist())):
            count[b, a] += 1
    frac = count / max(len(frames), 1)
    inv = frac >= hi
    for pk in frames:
        p = np.asarray(pk, float).reshape(-1, 2)
        if not len(p):
            continue
        bx = (p[:, 0] / bin_px).astype(int)
        by = (p[:, 1] / bin_px).astype(int)
        ok = (bx >= 0) & (bx < nb) & (by >= 0) & (by < nb)
        seen += int(inv[by[ok], bx[ok]].sum())
    share = seen / max(total, 1)

    if ax is None:
        _, ax = plt.subplots(figsize=(6.0, 5.2))
    im = ax.imshow(frac, origin="lower", cmap="magma", vmin=0, vmax=1,
                   extent=[0, shape[1], 0, shape[0]], interpolation="nearest")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, label="fraction of frames")
    ax.set_xlabel("detector x (px)")
    ax.set_ylabel("detector y (px)")
    ax.set_title(title or (f"detector occupancy over {len(frames)} frames\n"
                           f"{int(inv.sum())} bins fire in ≥{100*hi:.0f}% "
                           f"of frames, carrying {100*share:.1f}% of all peaks"),
                 fontsize=10)
    return ax, dict(n_invariant_bins=int(inv.sum()), frac_peaks_invariant=share,
                    n_frames=len(frames), bin_px=bin_px, hi=hi)


def summary(sol: LaueSolutions, *, tolerance: float = 1.0,
            space_group: int = 194, hkl=(0, 0, 1), normal=None,
            suptitle: Optional[str] = None):
    """One-page overview: map, pole figure, tilt histogram, size distribution."""
    import matplotlib.pyplot as plt

    c = cluster(sol, tolerance, space_group=space_group)
    reps = c.representatives(sol.orient_mat)
    sizes = c.sizes[~c.full_field]

    fig, ax = plt.subplots(2, 2, figsize=(12.0, 9.0))
    if sol.pos is not None:
        orientation_map(sol, ax[0, 0], normal=normal, hkl=hkl)
    else:
        ax[0, 0].text(0.5, 0.5, "no positions supplied", ha="center",
                      va="center", transform=ax[0, 0].transAxes)
        ax[0, 0].set_xticks([]); ax[0, 0].set_yticks([])
    pole_figure(reps, ax[0, 1], hkl=hkl, normal=normal,
                title=f"{tuple(int(x) for x in hkl)} pole figure\n"
                      f"{len(reps)} grains, one point each")
    tilt_histogram(reps, ax[1, 0], hkl=hkl, normal=normal)
    grain_size_distribution(c, ax[1, 1])

    head = (f"{len(sol)} solutions → {c.n_grains} grains at {tolerance}° "
            f"(effective n {c.n_eff_grains:.0f})")
    if len(reps) >= 2:
        obs, c95, ratio = texture_strength(reps, hkl=hkl, normal=normal)
        head += f"; texture {obs:.2f} vs chance {c95:.2f} = {ratio:.2f}×"
    elif c.n_clusters:
        # Every cluster spanned more than half the map, so there is no grain
        # population to have a texture. Saying so beats printing "nan x".
        head += (f"; no texture — all {c.n_clusters} clusters span >half the "
                 f"map")
    fig.suptitle(suptitle or head, fontsize=12)
    fig.tight_layout()
    return fig
