"""Side-by-side comparison of binary vs soft attribution V-map runs (P9 TODO(d)).

Runs ``calc_radius_v`` + ``refine_vmap`` twice on the same layer — once
with ``soft_attribution.enable=False`` (legacy binary scan-pos filter +
max-pool sino) and once with ``True`` (continuous beam-weight in the
indexer's compare_spots + sum-pool ``sinos_softsum``).  Writes a
unified diff report at ``<layer_dir>/diag/compare_modes/`` containing:

* ``v_map_binary.h5`` + ``v_map_soft.h5`` — per-voxel V from each run
* ``v_map_diff.png``                       — overlay |V_soft − V_binary|
* ``loss_compare.png``                     — both convergence traces
* ``compare_summary.csv``                  — per-grain summary table
* ``k_per_ring_compare.csv``               — K_binary vs K_soft + δ%

This driver is the canonical "did soft attribution actually help on
this dataset?" tool.  Run it after the regular pipeline completes — it
does NOT re-run indexing / refinement; it consumes the existing
upstream artifacts (``Output/Radius_V.csv`` etc. must already exist,
which means ``calc_radius_v`` has been run at least once).

For the *full* soft-mode benefit (including the soft attribution
flowing into the indexer-side scoring), the indexing stage must have
been run with ``--soft-attribution`` as well — this driver does NOT
re-run indexing; pair it with two full ``midas-pipeline run`` passes
(one without, one with the flag) and feed its outputs in.
"""
from __future__ import annotations

import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


__all__ = ["CompareModesResult", "run_compare_modes"]


@dataclass
class CompareModesResult:
    out_dir: str
    n_voxels: int
    n_rings: int
    mean_abs_dV: float                       # mean |V_soft − V_binary|
    mean_rel_dV: float                       # mean |V_soft − V_binary| / |V_binary|
    final_loss_binary: float
    final_loss_soft: float
    artifacts: list[str]


def _load_v_map(path: Path):
    if path.suffix == ".h5":
        import h5py
        with h5py.File(path, "r") as f:
            return (
                f["voxels/V"][:].astype(np.float64),
                f["rings/K"][:].astype(np.float64),
                f["voxels/positions_um"][:],
                f["voxels/grain_map"][:].astype(np.int64),
            )
    z = np.load(path)
    return (
        z["V"].astype(np.float64),
        z["K"].astype(np.float64),
        z["positions_um"],
        z["grain_map"].astype(np.int64),
    )


def run_compare_modes(
    layer_dir: str | Path,
    *,
    binary_v_map: str | Path,           # path to v_map.h5 (or .npz) from the binary run
    soft_v_map:   str | Path,           # same, but from the soft-mode run
    binary_loss_csv: Optional[str | Path] = None,
    soft_loss_csv:   Optional[str | Path] = None,
    output_subdir: str = "diag/compare_modes",
    axes: tuple[int, int] = (0, 1),
) -> CompareModesResult:
    """Build the side-by-side comparison report.

    Both V-map inputs must come from refine_vmap runs on the same
    underlying layer (same voxel grid + ring table); this driver does
    NOT validate that — it just diffs.
    """
    layer_dir = Path(layer_dir)
    out_dir = layer_dir / output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    artifacts: list[str] = []

    Vb, Kb, pos_b, gm_b = _load_v_map(Path(binary_v_map))
    Vs, Ks, pos_s, gm_s = _load_v_map(Path(soft_v_map))
    if Vb.shape != Vs.shape:
        raise ValueError(
            f"V-map shapes differ: binary {Vb.shape} vs soft {Vs.shape}; "
            "compare_modes requires runs on the same voxel grid."
        )

    # Copy the source H5/NPZ into the report dir for self-contained provenance.
    for src, label in [(binary_v_map, "binary"), (soft_v_map, "soft")]:
        src = Path(src)
        dst = out_dir / f"v_map_{label}{src.suffix}"
        shutil.copyfile(src, dst)
        artifacts.append(str(dst))

    # ---- per-voxel diff stats
    dV = Vs - Vb
    mean_abs_dV = float(np.abs(dV).mean())
    safe_Vb = np.where(np.abs(Vb) > 1e-30, np.abs(Vb), 1.0)
    rel = np.abs(dV) / safe_Vb
    mean_rel_dV = float(rel[Vb != 0].mean()) if np.any(Vb != 0) else 0.0

    # ---- per-grain summary table
    grain_ids = np.unique(gm_b[gm_b >= 0])
    rows = []
    for g in grain_ids:
        sel = gm_b == int(g)
        if not sel.any():
            continue
        rows.append([
            int(g),
            int(sel.sum()),
            float(Vb[sel].mean()),
            float(Vs[sel].mean()),
            float((Vs[sel] - Vb[sel]).mean()),
        ])
    summary_csv = out_dir / "compare_summary.csv"
    if rows:
        np.savetxt(
            summary_csv, np.array(rows),
            header="grain_id n_voxels V_binary_mean V_soft_mean dV_mean",
            fmt=["%d", "%d", "%.6e", "%.6e", "%.6e"], comments="",
        )
    else:
        summary_csv.write_text("# no grains\n")
    artifacts.append(str(summary_csv))

    # ---- K per ring comparison
    n_rings = int(min(Kb.size, Ks.size))
    k_compare_csv = out_dir / "k_per_ring_compare.csv"
    k_rows = np.column_stack([
        np.arange(n_rings),
        Kb[:n_rings], Ks[:n_rings],
        100.0 * (Ks[:n_rings] - Kb[:n_rings])
        / np.where(np.abs(Kb[:n_rings]) > 1e-30, np.abs(Kb[:n_rings]), 1.0),
    ])
    np.savetxt(
        k_compare_csv, k_rows,
        header="ring_idx K_binary K_soft delta_pct",
        fmt=["%d", "%.6e", "%.6e", "%+.3f"], comments="",
    )
    artifacts.append(str(k_compare_csv))

    # ---- side-by-side V-map image
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .vmap import _voxel_grid_to_image

    Vb_img, ext = _voxel_grid_to_image(pos_b, Vb, axes=axes)
    Vs_img, _ = _voxel_grid_to_image(pos_s, Vs, axes=axes)
    dV_img = Vs_img - Vb_img

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    for k, (img, t, cmap) in enumerate([
        (Vb_img, "V (binary)", "viridis"),
        (Vs_img, "V (soft)", "viridis"),
        (dV_img, "V_soft − V_binary", "RdBu_r"),
    ]):
        im = ax[k].imshow(img, origin="lower", extent=ext, cmap=cmap,
                          interpolation="nearest")
        ax[k].set_title(t)
        ax[k].set_xlabel(f"axis {axes[0]} (µm)")
        ax[k].set_ylabel(f"axis {axes[1]} (µm)")
        fig.colorbar(im, ax=ax[k], shrink=0.8)
    fig.tight_layout()
    diff_png = out_dir / "v_map_diff.png"
    fig.savefig(diff_png, dpi=150)
    plt.close(fig)
    artifacts.append(str(diff_png))

    # ---- loss overlay
    fb_loss = 0.0; fs_loss = 0.0
    if binary_loss_csv and Path(binary_loss_csv).exists():
        arr_b = np.loadtxt(binary_loss_csv, comments="#", skiprows=1)
        if arr_b.ndim == 1:
            arr_b = arr_b.reshape(1, -1)
        fb_loss = float(arr_b[-1, 1])
        if soft_loss_csv and Path(soft_loss_csv).exists():
            arr_s = np.loadtxt(soft_loss_csv, comments="#", skiprows=1)
            if arr_s.ndim == 1:
                arr_s = arr_s.reshape(1, -1)
            fs_loss = float(arr_s[-1, 1])
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.semilogy(arr_b[:, 0], arr_b[:, 1], "o-", label="binary")
            ax.semilogy(arr_s[:, 0], arr_s[:, 1], "s-", label="soft")
            ax.set_xlabel("iteration")
            ax.set_ylabel("loss")
            ax.set_title("Joint V+K refinement convergence: binary vs soft")
            ax.grid(True, which="both", alpha=0.3)
            ax.legend(loc="best")
            fig.tight_layout()
            loss_png = out_dir / "loss_compare.png"
            fig.savefig(loss_png, dpi=150)
            plt.close(fig)
            artifacts.append(str(loss_png))

    return CompareModesResult(
        out_dir=str(out_dir),
        n_voxels=int(Vb.shape[0]),
        n_rings=n_rings,
        mean_abs_dV=mean_abs_dV,
        mean_rel_dV=mean_rel_dV,
        final_loss_binary=fb_loss,
        final_loss_soft=fs_loss,
        artifacts=artifacts,
    )
