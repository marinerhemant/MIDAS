"""V-map refinement diagnostics (figures + tables).

Companion to :mod:`midas_pipeline.stages.refine_vmap`.  Renders the
artifacts described in §A9 of the V-map plan:

* ``v_map.tif`` — per-voxel V image (2-D slice for PF, 3-D stack for FF).
* ``v_map_overlay.png`` — grain map + V-map side-by-side + colourbars.
* ``spot_residuals.png`` — observed vs predicted intensity scatter (log-log).
* ``refine_loss_history.png`` — convergence trace (semilog-y).
* ``per_grain_v_histograms.png`` — one V-histogram per grain.
* ``k_per_ring.csv`` — K[ring] + per-ring residual statistics.

Matplotlib is imported lazily inside each helper so the base
``midas_pipeline`` install stays light.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


__all__ = [
    "plot_loss_history",
    "plot_per_grain_v_histograms",
    "plot_spot_residuals",
    "plot_v_map_overlay",
    "write_k_per_ring_table",
    "write_v_map_tif",
]


# ---------------------------------------------------------------- helpers


def _ensure_dir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def _voxel_grid_to_image(
    voxel_pos_um: np.ndarray, values: np.ndarray,
    *, axes: tuple[int, int] = (0, 1),
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """Project a flat per-voxel array onto a regular 2-D grid by binning.

    Inputs are flat ``(N,)`` per-voxel values + their lab-frame positions.
    Output is a 2-D image with the same y-axis convention as imshow
    (origin='lower'), plus an ``extent`` for plotting in µm.

    Voxels at the same (axis-0, axis-1) bin are averaged.
    """
    a, b = axes
    pa = voxel_pos_um[:, a]
    pb = voxel_pos_um[:, b]
    uniq_a = np.unique(np.round(pa, 6))
    uniq_b = np.unique(np.round(pb, 6))
    H = len(uniq_b)
    W = len(uniq_a)
    img = np.full((H, W), np.nan, dtype=np.float64)
    cnt = np.zeros_like(img)
    a_to_i = {v: i for i, v in enumerate(uniq_a)}
    b_to_j = {v: j for j, v in enumerate(uniq_b)}
    for n in range(values.shape[0]):
        i = a_to_i[round(pa[n], 6)]
        j = b_to_j[round(pb[n], 6)]
        if np.isnan(img[j, i]):
            img[j, i] = float(values[n])
            cnt[j, i] = 1.0
        else:
            img[j, i] = (img[j, i] * cnt[j, i] + float(values[n])) / (cnt[j, i] + 1)
            cnt[j, i] += 1.0
    extent = (float(uniq_a.min()), float(uniq_a.max()),
              float(uniq_b.min()), float(uniq_b.max()))
    return img, extent


# ---------------------------------------------------------------- writers


def write_v_map_tif(
    voxel_pos_um: np.ndarray,    # (N, 3)
    V_voxel: np.ndarray,          # (N,)
    output_path: str | Path,
    *,
    axes: tuple[int, int] = (0, 1),
) -> Path:
    """Write a 2-D float32 TIFF of V projected onto ``axes`` (PF: y vs x).

    Uses ``tifffile`` if available, else falls back to a numpy ``.npy``
    side-car (so the test suite doesn't hard-depend on tifffile).
    """
    out = Path(output_path)
    _ensure_dir(out)
    img, _ = _voxel_grid_to_image(voxel_pos_um, V_voxel, axes=axes)
    img32 = np.nan_to_num(img, nan=0.0).astype(np.float32)
    try:
        import tifffile
        tifffile.imwrite(out, img32)
        return out
    except ImportError:
        npy = out.with_suffix(".npy")
        np.save(npy, img32)
        return npy


def write_k_per_ring_table(
    K_ring: np.ndarray,                  # (R,)
    I_theory: np.ndarray,                # (R,)
    output_path: str | Path,
    *,
    ring_numbers: Optional[np.ndarray] = None,
    residual_stats: Optional[dict] = None,  # {ring_idx: {'mean': .., 'std': .., 'n': ..}}
) -> Path:
    """CSV: ring_idx, ring_number, K_ring, I_theory, mean_log_resid, std_log_resid, n_spots."""
    out = Path(output_path)
    _ensure_dir(out)
    R = int(K_ring.shape[0])
    rn = (
        np.asarray(ring_numbers, dtype=np.int64)
        if ring_numbers is not None
        else np.arange(R, dtype=np.int64)
    )
    means = np.zeros(R); stds = np.zeros(R); counts = np.zeros(R, dtype=np.int64)
    if residual_stats:
        for r, s in residual_stats.items():
            if 0 <= int(r) < R:
                means[int(r)] = float(s.get("mean", 0.0))
                stds[int(r)] = float(s.get("std", 0.0))
                counts[int(r)] = int(s.get("n", 0))
    cols = np.column_stack([
        np.arange(R, dtype=np.int64), rn, K_ring, I_theory,
        means, stds, counts,
    ])
    np.savetxt(
        out, cols,
        header="ring_idx ring_number K_ring I_theory mean_log_resid std_log_resid n_spots",
        fmt=["%d", "%d", "%.6e", "%.6e", "%.6e", "%.6e", "%d"],
        comments="",
    )
    return out


# ---------------------------------------------------------------- plotters


def plot_loss_history(
    loss_history: np.ndarray,
    output_path: str | Path,
    *,
    title: str = "Joint V+K refinement — loss history",
) -> Path:
    out = Path(output_path)
    _ensure_dir(out)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    losses = np.asarray(loss_history, dtype=np.float64)
    pos = losses > 0
    fig, ax = plt.subplots(figsize=(6, 4))
    if pos.any():
        ax.semilogy(np.where(pos)[0], losses[pos], marker="o", linestyle="-")
    else:
        ax.plot(losses, marker="o", linestyle="-")
    ax.set_xlabel("LBFGS / Adam iteration")
    ax.set_ylabel("log-residual loss (mean over valid spots)")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_spot_residuals(
    I_obs: np.ndarray, I_pred: np.ndarray,
    output_path: str | Path,
    *,
    title: str = "Observed vs predicted spot intensities",
) -> Path:
    out = Path(output_path)
    _ensure_dir(out)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    obs = np.asarray(I_obs, dtype=np.float64)
    pred = np.asarray(I_pred, dtype=np.float64)
    mask = (obs > 0) & (pred > 0)
    obs = obs[mask]; pred = pred[mask]

    fig, ax = plt.subplots(figsize=(6, 6))
    if obs.size > 0:
        ax.loglog(obs, pred, "o", ms=3, alpha=0.5)
        lo = float(min(obs.min(), pred.min()))
        hi = float(max(obs.max(), pred.max()))
        ax.loglog([lo, hi], [lo, hi], "k--", lw=1, label="y = x")
        log_resid = np.log(pred) - np.log(obs)
        ax.set_title(
            f"{title}\nmean log-resid = {log_resid.mean():+.3f}, "
            f"σ = {log_resid.std():.3f} ({obs.size} spots)"
        )
    else:
        ax.set_title(title + "\n(no valid spots)")
    ax.set_xlabel("I_observed")
    ax.set_ylabel("I_predicted")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_v_map_overlay(
    voxel_pos_um: np.ndarray,
    V_voxel: np.ndarray,
    grain_map: np.ndarray,
    output_path: str | Path,
    *,
    axes: tuple[int, int] = (0, 1),
    title: str = "Grain map + V_voxel map",
) -> Path:
    out = Path(output_path)
    _ensure_dir(out)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    V_img, extent = _voxel_grid_to_image(voxel_pos_um, V_voxel, axes=axes)
    G_img, _ = _voxel_grid_to_image(
        voxel_pos_um, grain_map.astype(np.float64), axes=axes,
    )

    fig, axarr = plt.subplots(1, 2, figsize=(11, 5))
    im0 = axarr[0].imshow(
        G_img, origin="lower", extent=extent, cmap="tab20",
        interpolation="nearest",
    )
    axarr[0].set_title("grain_map")
    axarr[0].set_xlabel(f"axis {axes[0]} (µm)")
    axarr[0].set_ylabel(f"axis {axes[1]} (µm)")
    fig.colorbar(im0, ax=axarr[0], shrink=0.8, label="grain id")

    im1 = axarr[1].imshow(
        V_img, origin="lower", extent=extent, cmap="viridis",
        interpolation="nearest",
    )
    axarr[1].set_title("V_voxel")
    axarr[1].set_xlabel(f"axis {axes[0]} (µm)")
    axarr[1].set_ylabel(f"axis {axes[1]} (µm)")
    fig.colorbar(im1, ax=axarr[1], shrink=0.8, label="V (arb)")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_per_grain_v_histograms(
    V_voxel: np.ndarray, grain_map: np.ndarray,
    output_path: str | Path,
    *,
    bins: int = 30,
    max_panels: int = 12,
    title: str = "Per-grain V distribution",
) -> Path:
    out = Path(output_path)
    _ensure_dir(out)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grain_ids = np.unique(grain_map[grain_map >= 0])
    grain_ids = grain_ids[:max_panels]
    n_panels = int(grain_ids.size)
    if n_panels == 0:
        # Still emit a placeholder file so the caller can rely on the path.
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "no grains", ha="center", va="center")
        fig.savefig(out, dpi=150)
        plt.close(fig)
        return out

    ncols = min(4, n_panels)
    nrows = (n_panels + ncols - 1) // ncols
    fig, axarr = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 2.5 * nrows),
                               squeeze=False)
    for k, g in enumerate(grain_ids):
        r, c = divmod(k, ncols)
        ax = axarr[r][c]
        Vg = V_voxel[grain_map == int(g)]
        if Vg.size > 0:
            ax.hist(Vg, bins=bins, color="steelblue", edgecolor="k")
            ax.set_title(f"grain {int(g)}  (n={Vg.size}, μ={Vg.mean():.3g})",
                         fontsize=9)
        else:
            ax.set_title(f"grain {int(g)}  (empty)", fontsize=9)
        ax.tick_params(axis="both", labelsize=8)
    # Hide unused panels
    for k in range(n_panels, nrows * ncols):
        r, c = divmod(k, ncols)
        axarr[r][c].axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out
