"""Model-free peak tracking across a raster: no hkl, no indexing, no cell.

Why detector-space tracking, not hkl matching: each raster position's crystal domain has its
OWN orientation (U), so "hkl=(2,0,2)" at position A and the same label at position B are not
the same physical reflection unless the two domains happen to share orientation. But NEIGHBORING
raster positions very likely share the same physical grain with only a slowly drifting
orientation (mosaic), so a peak's DETECTOR position (row, col, frame) moves smoothly from one
raster point to its neighbor. Tracking that trajectory directly -- flood-fill from a seed pixel,
re-centering a small search window at each step from the previous position's own centroid -- is
a raw-data-first measurement that sidesteps the whole indexing/index-asymmetry discussion (an
a/b-splitting artifact already documented for one project) entirely: no cell, no lattice, no
domain. Where the peak's signal drops out (grain boundary, moved off a family's Bragg condition,
etc.) the walk simply stops there and is not propagated further -- that is real information (a
discontinuity), not a bug to paper over.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional

import numpy as np

__all__ = ["windowed_centroid", "track_peak_over_raster", "plot_peak_maps",
           "snap_to_local_max", "STATUS_NO_SIGNAL", "STATUS_REJECTED", "STATUS_TRACKED"]

STATUS_NO_SIGNAL = 0   # windowed_centroid found nothing above the SNR floor
STATUS_REJECTED = 1    # a candidate was found, but the continuity guard rejected it
STATUS_TRACKED = 2     # accepted and propagated to neighbors

_FIELDS = ("centroid_row", "centroid_col", "centroid_frame", "intensity",
           "sigma_row", "sigma_col", "sigma_frame", "peak_max", "background")


def windowed_centroid(frames: np.ndarray, row0: int, col0: int, frame0: int, *,
                      win_rc: int = 15, win_f: int = 4, min_snr: float = 5.0
                      ) -> Optional[Dict[str, float]]:
    """Intensity-weighted 3-D centroid and width in a box around (row0, col0, frame0).

    Background is the box's own median (robust to a few bright pixels); a peak is accepted
    only if its max exceeds `min_snr` times the local background's MAD-based noise estimate --
    otherwise this position has genuinely lost the peak, and the caller must not propagate a
    centroid onward from it.

    Returns a dict (centroid_row, centroid_col, centroid_frame, intensity, sigma_row, sigma_col,
    sigma_frame, peak_max, background) or None if the peak is not found in this window.
    """
    n_f, n_r, n_c = frames.shape
    r0, r1 = max(0, row0 - win_rc), min(n_r, row0 + win_rc + 1)
    c0, c1 = max(0, col0 - win_rc), min(n_c, col0 + win_rc + 1)
    f0, f1 = max(0, frame0 - win_f), min(n_f, frame0 + win_f + 1)
    box = frames[f0:f1, r0:r1, c0:c1].astype(np.float64)
    if box.size == 0:
        return None

    bg = float(np.median(box))
    mad = float(np.median(np.abs(box - bg))) + 1e-9
    noise = 1.4826 * mad
    peak_max = float(box.max())
    if noise <= 0 or (peak_max - bg) < min_snr * noise:
        return None

    w = np.clip(box - bg, 0, None)
    total = w.sum()
    if total <= 0:
        return None

    ff, rr, cc = np.meshgrid(np.arange(f0, f1), np.arange(r0, r1), np.arange(c0, c1),
                             indexing="ij")
    cen_f = float((w * ff).sum() / total)
    cen_r = float((w * rr).sum() / total)
    cen_c = float((w * cc).sum() / total)
    sig_f = float(np.sqrt((w * (ff - cen_f) ** 2).sum() / total))
    sig_r = float(np.sqrt((w * (rr - cen_r) ** 2).sum() / total))
    sig_c = float(np.sqrt((w * (cc - cen_c) ** 2).sum() / total))

    return dict(centroid_row=cen_r, centroid_col=cen_c, centroid_frame=cen_f,
                intensity=float(total), sigma_row=sig_r, sigma_col=sig_c, sigma_frame=sig_f,
                peak_max=peak_max, background=bg)


def snap_to_local_max(proj: np.ndarray, row: int, col: int, radius: int = 12):
    """Snap an approximate (row, col) to the true local maximum of `proj` within `radius`.

    Used to turn an imprecise click (or any coarse guess) on a max-over-frames projection into
    a usable seed pixel -- clicking exactly on a 1-pixel-wide spot is unrealistic.
    """
    n_r, n_c = proj.shape
    r0, r1 = max(0, row - radius), min(n_r, row + radius + 1)
    c0, c1 = max(0, col - radius), min(n_c, col + radius + 1)
    box = proj[r0:r1, c0:c1]
    rr, cc = np.unravel_index(np.argmax(box), box.shape)
    return r0 + rr, c0 + cc


def track_peak_over_raster(loader: Callable[[int], np.ndarray], n_rows: int, n_cols: int,
                           seed_pos: int, seed_row: float, seed_col: float, seed_frame: float,
                           *, win_rc: int = 15, win_f: int = 4, min_snr: float = 5.0,
                           max_step_px: float = 8.0, max_step_frames: float = 2.0,
                           max_intensity_ratio: float = 4.0, log: Callable = print
                           ) -> Dict[str, np.ndarray]:
    """Flood-fill a peak's detector trajectory over a row-major (n_rows x n_cols) raster.

    `loader(p)` returns position p's (already dead-frame-cleaned) raw frame stack.
    `seed_pos` is the 0-indexed position the user picked the peak from; `seed_row/col/frame`
    is that peak's approximate location there (need not be exact -- the first window re-centers).

    **Continuity guard, added after checking a real run for jumps between raster-grid
    neighbors:** a first version with only an SNR cut let the tracker HIJACK a different,
    unrelated bright peak whenever one drifted within the `win_rc` search box -- 391/1171
    grid-adjacent pairs jumped >5 px, several 20-32 px with a simultaneous 3-20x intensity
    change, concentrated where the search entered a busier (powder-rich) region. A 4 um raster
    step should not move a real peak far on the detector, so a candidate is accepted only if
    BOTH its centroid shift from the seed (`max_step_px`, `max_step_frames`) AND its intensity
    ratio to the seed's own last intensity (`max_intensity_ratio`) stay bounded; otherwise this
    branch stops here rather than silently propagating a wrong lock forward looking locally
    smooth.

    A rejected candidate is not simply discarded: it is recorded with ``status`` ==
    ``STATUS_REJECTED`` and its field values are still stored, so a caller can show "a peak was
    found here but it does not connect continuously" rather than an indistinguishable blank.
    Neighbors are not enqueued from a rejected step, same as from a no-signal one -- only an
    accepted step propagates the walk, since a rejected candidate's own position is exactly
    what should not be trusted as a re-centering seed.

    Returns a dict of (n_rows, n_cols) float arrays (NaN where `status` ==
    :data:`STATUS_NO_SIGNAL`): the fields in :data:`_FIELDS`, plus an integer ``status`` array
    (:data:`STATUS_NO_SIGNAL` / :data:`STATUS_REJECTED` / :data:`STATUS_TRACKED`) and a boolean
    ``tracked`` array (``status == STATUS_TRACKED``, kept for callers that only care about the
    accepted walk).
    """
    grids = {f: np.full((n_rows, n_cols), np.nan) for f in _FIELDS}
    status = np.full((n_rows, n_cols), STATUS_NO_SIGNAL, dtype=np.int8)
    visited = np.zeros((n_rows, n_cols), dtype=bool)

    queue = [(seed_pos, seed_row, seed_col, seed_frame, None)]
    n_done = 0
    n_rejected = 0
    while queue:
        p, sr, sc, sf, seed_intensity = queue.pop(0)
        r, c = p // n_cols, p % n_cols
        if visited[r, c]:
            continue
        visited[r, c] = True

        frames = loader(p)
        result = windowed_centroid(frames, int(round(sr)), int(round(sc)), int(round(sf)),
                                   win_rc=win_rc, win_f=win_f, min_snr=min_snr)
        n_done += 1
        if n_done % 25 == 0:
            log(f"  tracked {n_done} positions so far ({int((status == STATUS_TRACKED).sum())} "
               f"with signal, {n_rejected} rejected as likely hijacks)...")

        if result is None:
            continue  # lost here -- do not enqueue neighbors from a failed centroid

        accepted = True
        if seed_intensity is not None:
            step_px = np.hypot(result["centroid_row"] - sr, result["centroid_col"] - sc)
            step_f = abs(result["centroid_frame"] - sf)
            ratio = max(result["intensity"], seed_intensity) / min(result["intensity"], seed_intensity)
            if step_px > max_step_px or step_f > max_step_frames or ratio > max_intensity_ratio:
                accepted = False

        for f in _FIELDS:
            grids[f][r, c] = result[f]

        if not accepted:
            n_rejected += 1
            status[r, c] = STATUS_REJECTED
            continue  # likely hijacked a different, unrelated peak -- stop this branch here

        status[r, c] = STATUS_TRACKED
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < n_rows and 0 <= nc < n_cols and not visited[nr, nc]:
                queue.append((nr * n_cols + nc, result["centroid_row"], result["centroid_col"],
                            result["centroid_frame"], result["intensity"]))

    grids["status"] = status
    grids["tracked"] = status == STATUS_TRACKED
    n_tracked = int(grids["tracked"].sum())
    log(f"done: {n_tracked} / {n_rows * n_cols} tracked, {n_rejected} rejected as likely "
       f"hijacks, {n_rows * n_cols - n_tracked - n_rejected} no signal")
    return grids


def plot_peak_maps(data, *, fields=("centroid_row", "centroid_col", "centroid_frame",
                                    "intensity", "sigma_row", "sigma_col"),
                   cmap: str = "viridis", title_prefix: str = "", show: bool = True):
    """Render a grid of per-field raster maps from a :func:`track_peak_over_raster` result.

    `data` is the dict returned by :func:`track_peak_over_raster` (or an ``np.load`` of an
    ``.npz`` saved from it -- anything indexable by the field names plus ``status``).

    Three-state per pixel, using ``status``: STATUS_NO_SIGNAL pixels are left blank (no color,
    via ``cmap.set_bad``); STATUS_TRACKED pixels are colored normally by the field value;
    STATUS_REJECTED pixels are colored by their (still-real) field value but additionally
    outlined with a red square marker, so "a discontinuous candidate was found here" stays
    visible rather than reading identically to "nothing here" or to a trusted walk step.

    `fields` selects which panels to draw and in what order -- pass any subset/order of
    ``centroid_row``, ``centroid_col``, ``centroid_frame``, ``intensity``, ``sigma_row``,
    ``sigma_col``, ``sigma_frame``, ``peak_max``, ``background``. `cmap` is any matplotlib
    colormap name, applied to every panel.

    Returns the ``(fig, axes)`` matplotlib objects (matplotlib is imported lazily so the
    package can be installed without it).

    Accepts a result from a version of :func:`track_peak_over_raster` predating ``status``
    (only ``tracked``, a plain bool grid) -- degrades to the old two-state rendering (no
    STATUS_REJECTED outline) rather than raising, since existing saved ``.npz`` results should
    not need re-tracking just to be plotted with the new function.
    """
    import matplotlib.pyplot as plt

    tracked = np.asarray(data["tracked"])
    if "status" in data:
        status = np.asarray(data["status"])
    else:
        status = np.where(tracked, STATUS_TRACKED, STATUS_NO_SIGNAL)
    n_total = status.size
    n_tracked = int(tracked.sum())

    labels = {"centroid_row": "centroid row (px)", "centroid_col": "centroid col (px)",
             "centroid_frame": "centroid frame (omega step)",
             "intensity": "integrated intensity", "sigma_row": "width, row (px)",
             "sigma_col": "width, col (px)", "sigma_frame": "width, omega (frames)",
             "peak_max": "peak max (counts)", "background": "local background (counts)"}

    n = len(fields)
    ncols = 3 if n > 2 else n
    nrows = -(-n // ncols)  # ceil
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.3 * ncols, 3.6 * nrows), squeeze=False)
    axes_flat = axes.flat

    rej_r, rej_c = np.where(status == STATUS_REJECTED)

    for ax, key in zip(axes_flat, fields):
        cmap_obj = plt.get_cmap(cmap).copy()
        cmap_obj.set_bad("#dddddd")
        arr = np.where(status >= STATUS_REJECTED, np.asarray(data[key]), np.nan)
        im = ax.imshow(arr, origin="lower", cmap=cmap_obj)
        if rej_r.size:
            ax.scatter(rej_c, rej_r, s=16, facecolors="none", edgecolors="red",
                       linewidths=0.9, marker="s")
        ax.set_title(labels.get(key, key), fontsize=10)
        ax.set_xlabel("col (raster)")
        ax.set_ylabel("row (raster)")
        fig.colorbar(im, ax=ax, fraction=0.046)
    for ax in list(axes_flat)[n:]:
        ax.axis("off")

    seed_p = data["seed_p"] if "seed_p" in data else None
    seed_str = f"seed p={int(seed_p)} -- " if seed_p is not None else ""
    fig.suptitle(f"{title_prefix}{seed_str}{n_tracked}/{n_total} tracked, "
                f"{int((status == STATUS_REJECTED).sum())} rejected "
                f"(red outline = rejected, found but discontinuous)")
    fig.tight_layout()
    if show:
        plt.show()
    return fig, axes
