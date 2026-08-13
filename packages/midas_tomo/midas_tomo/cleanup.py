"""Vo et al. (2018) stripe-removal tuning.

Stripes in a sinogram become rings in the reconstruction, and the three Vo
parameters (``snr``, large-filter size, small-filter size) have no universal
setting — they depend on detector width and on how bad the stripes are. The
engine can sweep a grid of them in a single call via ``stripeConfigFile``;
this module drives that, scores each result, and recommends one.

The score is deliberately crude (:func:`ring_metric`) and is a *ranking* aid,
not a physical quantity. Look at the montage before trusting it.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import numpy as np

from . import backend_c
from .api import read_recon_cube, write_thetas
from .config import TomoConfig, parse_shift_arg

__all__ = [
    "default_cleanup_grid",
    "load_cleanup_grid",
    "ring_metric",
    "run_tomo_cleanup_sweep",
]

log = logging.getLogger(__name__)


def _odd(n) -> int:
    """Round *n* to the nearest odd integer >= 3."""
    n = max(3, int(round(n)))
    return n if n % 2 else n + 1


def default_cleanup_grid(detector_xdim: int) -> list[dict]:
    """A four-point starting grid, scaled to the detector width.

    ``0`` baseline (no cleanup, anchors the comparison), ``1`` moderate,
    ``2`` broader filters, ``3`` more aggressive SNR. For a 128-px detector
    this gives ``la in {31, 41}``, ``sm in {11, 15}``; for 2048 px,
    ``la in {511, 681}``, ``sm in {171, 227}``.
    """
    w = int(detector_xdim)
    la_mid, la_big = _odd(w / 4), _odd(w / 3)
    sm_mid, sm_big = _odd(w / 12), _odd(w / 9)
    return [
        {"snr": 0.0, "la": 0, "sm": 0},
        {"snr": 3.0, "la": la_mid, "sm": sm_mid},
        {"snr": 3.0, "la": la_big, "sm": sm_big},
        {"snr": 1.5, "la": la_mid, "sm": sm_mid},
    ]


def load_cleanup_grid(path: str | os.PathLike) -> list[dict]:
    """Read a grid file: one ``snr la sm`` per line, ``#`` for comments."""
    configs = []
    for line in Path(path).read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        parts = s.split()
        if len(parts) < 3:
            continue
        configs.append({"snr": float(parts[0]), "la": int(parts[1]), "sm": int(parts[2])})
    if not configs:
        raise ValueError(f"no valid configs in cleanup grid file {path}")
    return configs


def ring_metric(img: np.ndarray) -> float:
    """Standard deviation of the radial first-difference. Lower is better.

    Sample features vary smoothly with radius; concentric ring artefacts put
    sharp spikes at fixed radii, which shows up as a larger spread in the
    first difference of the azimuthally-averaged radial profile.

    This is a heuristic. It cannot tell a ring from a genuinely sharp circular
    feature, so a sample that really does have concentric structure will score
    badly no matter what the cleanup does.
    """
    img = np.asarray(img)
    ny, nx = img.shape
    cy, cx = ny // 2, nx // 2
    y, x = np.indices((ny, nx))
    r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    rmax = min(cy, cx) - 2
    if rmax < 2:
        raise ValueError(f"image too small for a radial profile: {img.shape}")
    # One pass with bincount rather than rmax boolean masks over the full image.
    rb = r.astype(np.int32).ravel()
    keep = rb <= rmax
    counts = np.bincount(rb[keep], minlength=rmax + 1).astype(np.float64)
    sums = np.bincount(rb[keep], weights=img.ravel()[keep], minlength=rmax + 1)
    profile = sums[: rmax + 1] / np.maximum(counts[: rmax + 1], 1)
    return float(np.std(np.diff(profile)))


def run_tomo_cleanup_sweep(
    data,
    dark,
    whites,
    workingdir,
    thetas,
    *,
    cleanup_configs=None,
    shift: float = 0.0,
    tuning_slices=None,
    filter_nr: int = 2,
    do_log: bool = True,
    extra_pad: bool = False,
    auto_centering: bool = True,
    n_cpus: int = 40,
    do_cleanup: bool = True,
    ring_removal: float = 0.0,
    make_montage: bool = True,
) -> dict:
    """Reconstruct a thin slab under every cleanup config and rank the results.

    One engine call handles the whole grid. Side effects in *workingdir*:
    ``cleanup_tuning_scores.txt``, ``cleanup_tuning_recommended.txt``, and
    (unless disabled) ``cleanup_tuning_montage.png``.

    Parameters
    ----------
    cleanup_configs : list[dict] | path | None
        ``{'snr', 'la', 'sm'}`` dicts, a grid file, or None for
        :func:`default_cleanup_grid`. ``snr <= 0`` marks the baseline.
    shift : float
        A single rotation-axis shift for tuning. Cleanup tuning is not very
        sensitive to small shift errors, so an approximate value is fine.
    tuning_slices : list[int] | None
        Slice indices for the montage. None picks four near the middle.

    Returns
    -------
    dict
        ``configs``, ``recons`` ``(n_cfg, n_tuning_slices, X, X)``,
        ``ring_metric`` ``(n_cfg,)``, ``best_idx``, ``best_config``,
        ``tuning_slices``.
    """
    workingdir = Path(workingdir)
    workingdir.mkdir(parents=True, exist_ok=True)

    data = np.asarray(data)
    if data.ndim != 3:
        raise ValueError(f"data must be 3-D (theta, slice, x); got {data.shape}")
    _, n_slices_total, xdim = data.shape

    if cleanup_configs is None:
        cleanup_configs = default_cleanup_grid(xdim)
    elif isinstance(cleanup_configs, (str, os.PathLike)):
        cleanup_configs = load_cleanup_grid(cleanup_configs)
    if len(cleanup_configs) < 2:
        raise ValueError(
            "a sweep needs at least 2 configs; for one fixed config call "
            "run_tomo(do_stripe_removal=True, ...) directly"
        )

    if tuning_slices is None:
        mid = n_slices_total // 2
        tuning_slices = sorted(
            {max(0, min(n_slices_total - 1, mid + d)) for d in (-3, -1, 1, 3)}
        )
    tuning_slices = sorted(int(s) for s in tuning_slices)
    if len(tuning_slices) % 2:
        tuning_slices.append(tuning_slices[-1])  # engine reconstructs in pairs

    infn = workingdir / "cleanup_tuning_input.bin"
    with infn.open("wb") as f:
        np.asarray(dark, dtype=np.float32).tofile(f)
        np.asarray(whites, dtype=np.float32).tofile(f)
        data.astype(np.uint16).tofile(f)

    thetas_fn = write_thetas(thetas, workingdir / "cleanup_tuning_thetas.txt")

    grid_fn = workingdir / "cleanup_tuning_grid.txt"
    with grid_fn.open("w") as f:
        f.write("# snr  la_size  sm_size  (snr<=0 means baseline)\n")
        for c in cleanup_configs:
            f.write(f'{c["snr"]:.4f}  {int(c["la"])}  {int(c["sm"])}\n')

    slices_fn = workingdir / "cleanup_tuning_slices.txt"
    slices_fn.write_text("".join(f"{s}\n" for s in tuning_slices))

    # The engine's inner loop reconstructs shifts in pairs, so asking for a
    # single shift would still cost a pair. We request two 0.1 apart and keep
    # the first, rather than adding a single-shift path to the C.
    cfg = TomoConfig(
        data_file=infn,
        recon_file=workingdir / "cleanup_tuning_output",
        are_sinos=False,
        det_xdim=xdim,
        det_ydim=n_slices_total,
        theta_file=thetas_fn,
        filter_nr=filter_nr,
        shift_values=(float(shift), float(shift) + 0.1, 0.1),
        do_log=do_log,
        extra_pad=extra_pad,
        auto_centering=auto_centering,
        slices_to_process=slices_fn,
        ring_removal_coeff=ring_removal or None,
        stripe_config_file=grid_fn,
    )
    par_fn = cfg.to_param_file(workingdir / "cleanup_tuning.par")

    t0 = time.time()
    backend_c.run_binary(par_fn, n_cpus, cwd=workingdir)
    log.info("cleanup sweep took %.2fs", time.time() - t0)

    n_cfg = len(cleanup_configs)
    n_tune = len(tuning_slices)
    cube, outfn = read_recon_cube(cfg, n_tune, n_cleanup=n_cfg)
    cube = cube[:, 0]  # keep the first of the shift pair

    rm = np.array(
        [np.mean([ring_metric(cube[ci, si]) for si in range(n_tune)]) for ci in range(n_cfg)]
    )

    # Lowest score wins, with one caveat: if the *baseline* wins, that usually
    # means no config changed the data materially rather than that cleanup
    # hurts. Prefer a cleaned config that lands within 1% of it.
    order = np.argsort(rm)
    best_idx = int(order[0])
    if cleanup_configs[best_idx]["snr"] <= 0 and n_cfg > 1:
        baseline = rm[best_idx]
        for cand in order[1:]:
            if cleanup_configs[cand]["snr"] > 0 and rm[cand] <= baseline * 1.01:
                best_idx = int(cand)
                break

    scores_fn = workingdir / "cleanup_tuning_scores.txt"
    with scores_fn.open("w") as f:
        f.write("cleanup_idx\tsnr\tla\tsm\tring_metric\tbest\n")
        for ci, c in enumerate(cleanup_configs):
            mark = "BEST" if ci == best_idx else ""
            f.write(f'{ci}\t{c["snr"]:.4f}\t{c["la"]}\t{c["sm"]}\t{rm[ci]:.6e}\t{mark}\n')

    bc = cleanup_configs[best_idx]
    rec_fn = workingdir / "cleanup_tuning_recommended.txt"
    rec_fn.write_text(f'{bc["snr"]:.4f} {bc["la"]} {bc["sm"]}\n')
    log.info("recommended cleanup: snr=%s la=%s sm=%s", bc["snr"], bc["la"], bc["sm"])

    if make_montage:
        _write_montage(workingdir, cube, cleanup_configs, rm, best_idx)

    if do_cleanup:
        for p in [outfn, par_fn, thetas_fn, infn, slices_fn,
                  Path(str(cfg.recon_file) + "_cleanup_configs.txt"),
                  *cfg.wisdom_paths(workingdir)]:
            try:
                Path(p).unlink()
            except (FileNotFoundError, IsADirectoryError):
                pass

    return {
        "configs": cleanup_configs,
        "recons": cube,
        "ring_metric": rm,
        "best_idx": best_idx,
        "best_config": bc,
        "tuning_slices": tuning_slices,
    }


def _write_montage(workingdir, cube, configs, rm, best_idx) -> None:
    """Render the per-config comparison PNG.

    Uses the Agg canvas directly rather than ``matplotlib.use()``, which would
    mutate the caller's global backend and break ``%matplotlib inline``.
    """
    try:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
    except ImportError:
        log.info("matplotlib not installed — skipping cleanup montage")
        return

    n_cfg = len(configs)
    mid = cube.shape[1] // 2
    cols = min(4, n_cfg)
    rows = (n_cfg + cols - 1) // cols
    fig = Figure(figsize=(cols * 4, rows * 4))
    FigureCanvasAgg(fig)
    axes = np.atleast_1d(fig.subplots(rows, cols)).ravel()
    for ci in range(n_cfg):
        sl = cube[ci, mid]
        ax = axes[ci]
        ax.imshow(sl, cmap="gray",
                  vmin=float(np.percentile(sl, 2)), vmax=float(np.percentile(sl, 98)))
        c = configs[ci]
        label = (f'snr={c["snr"]:.1f} la={c["la"]} sm={c["sm"]}'
                 if c["snr"] > 0 else "BASELINE")
        title = f"#{ci}  {label}\nring={rm[ci]:.2e}"
        if ci == best_idx:
            title = "BEST: " + title
            for sp in ax.spines.values():
                sp.set_edgecolor("red")
                sp.set_linewidth(3)
        ax.set_title(title, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes[n_cfg:]:
        ax.axis("off")
    fig.tight_layout()
    png = Path(workingdir) / "cleanup_tuning_montage.png"
    fig.savefig(png, dpi=140, bbox_inches="tight")
    log.info("wrote %s", png)
