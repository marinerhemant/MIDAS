"""Static matplotlib visualizations for calibration inputs/outputs.

All plots use a lazy ``matplotlib.pyplot`` import so core installs that
don't need viz don't pay the import cost. Functions return the
``matplotlib.figure.Figure`` — callers can further customise before
saving. Pass ``save=<path>`` to save directly.

Public API
----------
- ``convergence(result, save=None, ax=None)``
- ``rings_overlay(result, image, save=None, ax=None)``
- ``residual_heatmap(result, save=None, ax=None)``
- ``fourier_harmonics(result, save=None, n_harmonics=8)``
- ``distortion_field(result, save=None, grid_step=50)``
- ``inspect(result, image, out_dir)`` — writes all five PNGs
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from ..calibrate import CalibrationResult

__all__ = [
    "convergence",
    "distortion_field",
    "fourier_harmonics",
    "inspect",
    "residual_heatmap",
    "rings_overlay",
]


# ---------------------------------------------------------------------------
# corr.csv parser — skips 2-row geometry summary, reads per-point table.
# ---------------------------------------------------------------------------

_CORR_COLUMNS = [
    "Eta", "Strain", "RadFit", "EtaCalc", "DiffCalc", "RadCalc",
    "Ideal2Theta", "Outlier", "YRawCorr", "ZRawCorr", "RingNr",
    "RadGlobal", "IdealR", "Fit2Theta", "IdealA", "FitA", "DeltaR", "DeltaA",
]


def _load_corr_csv(path: Path) -> dict[str, np.ndarray]:
    """Parse a MIDAS corr.csv into column arrays.

    Layout: 2 lines of geometry summary (CSV header + single row), 1 line
    of comment-prefixed space-separated header, then per-point data
    rows with 18 whitespace-separated floats/ints.
    """
    rows: list[list[float]] = []
    with path.open() as f:
        f.readline()  # geometry-summary header
        f.readline()  # geometry-summary row
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("%") or line.startswith("#"):
                continue
            tokens = line.split()
            if len(tokens) < len(_CORR_COLUMNS):
                continue
            try:
                rows.append([float(t) for t in tokens[: len(_CORR_COLUMNS)]])
            except ValueError:
                continue

    if not rows:
        return {k: np.array([]) for k in _CORR_COLUMNS}

    arr = np.asarray(rows)
    return {name: arr[:, i] for i, name in enumerate(_CORR_COLUMNS)}


# ---------------------------------------------------------------------------
# Convergence — MeanStrain vs iteration.
# ---------------------------------------------------------------------------

def convergence(
    result: CalibrationResult,
    *,
    save: str | Path | None = None,
    ax=None,
):
    """Plot pseudo-strain convergence vs iteration.

    Reads ``CalibrationResult.convergence_history`` (list of row dicts
    parsed from ``<rawFN>.convergence_history.csv``).
    """
    import matplotlib.pyplot as plt

    if not result.convergence_history:
        raise ValueError(
            "No convergence history available. The calibration may not have "
            "produced a .convergence_history.csv (check n_iterations > 0)."
        )

    rows = result.convergence_history
    iters = np.array([r["Iter"] for r in rows])
    mean_ppm = np.array([r["MeanStrain_ppm"] for r in rows])
    std_ppm = np.array([r.get("StdStrain_ppm", 0.0) for r in rows])

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.figure

    ax.errorbar(iters, mean_ppm, yerr=std_ppm, fmt="o-", capsize=3,
                color="tab:blue", label="MeanStrain ± StdStrain")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Pseudo-strain (µε)")
    ax.set_title("Calibration convergence")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()

    if save is not None:
        fig.savefig(save, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Rings overlay — image + calibrated rings as circles.
# ---------------------------------------------------------------------------

def rings_overlay(
    result: CalibrationResult,
    image: str | Path | np.ndarray,
    *,
    save: str | Path | None = None,
    ax=None,
    show_fitted: bool = True,
    log_scale: bool = True,
    cmap: str = "viridis",
):
    """Overlay calibrated rings on the detector image.

    Ideal rings (from the refined geometry's ring radii) are drawn as
    dashed circles; per-point fitted positions from ``corr.csv`` are
    drawn as faint dots to visualise residual-vs-ring scatter.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    img = _load_image(image) if not isinstance(image, np.ndarray) else image
    img = np.asarray(img, dtype=float)

    geom = result.geometry

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7))
    else:
        fig = ax.figure

    # Log-scale the display so bright rings don't wash out faint ones.
    disp = np.log1p(np.clip(img, 0, None)) if log_scale else img
    ax.imshow(disp, cmap=cmap, origin="upper", aspect="equal")

    # Calibrated beam center.
    ax.plot(geom.ybc, geom.zbc, "+", color="red", markersize=14, mew=2,
            label=f"BC ({geom.ybc:.1f}, {geom.zbc:.1f})")

    # Ideal rings from corr.csv if available (uses IdealR column).
    if show_fitted and result.corr_csv_path and result.corr_csv_path.exists():
        corr = _load_corr_csv(result.corr_csv_path)
        if corr["IdealR"].size:
            ideal_r_um = np.unique(np.round(corr["IdealR"], 1))
            # Convert μm → pixels via pixel size.
            ideal_r_px = ideal_r_um / geom.px
            for r in ideal_r_px:
                ax.add_patch(Circle((geom.ybc, geom.zbc), r,
                                    fill=False, linestyle="--",
                                    edgecolor="white", linewidth=0.8, alpha=0.6))
        # Fitted per-point radii as faint dots in (y, z).
        if corr["YRawCorr"].size:
            mask = corr["Outlier"] == 0
            ax.scatter(
                corr["YRawCorr"][mask], corr["ZRawCorr"][mask],
                s=0.5, c="cyan", alpha=0.4, label="Fitted (inliers)",
            )

    ax.set_xlabel("Y (pixels)")
    ax.set_ylabel("Z (pixels)")
    ax.set_title(
        f"Calibrated rings — Lsd={geom.lsd/1000:.1f} mm, "
        f"MeanStrain={geom.mean_strain:.2f} µε"
    )
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()

    if save is not None:
        fig.savefig(save, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Residual heatmap — DeltaR in (RingNr, Eta) space.
# ---------------------------------------------------------------------------

def residual_heatmap(
    result: CalibrationResult,
    *,
    save: str | Path | None = None,
    ax=None,
    cmap: str = "RdBu_r",
):
    """2D heatmap of the radial residual ΔR over (ring, η)."""
    import matplotlib.pyplot as plt

    _require_corr(result)
    corr = _load_corr_csv(result.corr_csv_path)

    rings = np.unique(corr["RingNr"]).astype(int)
    # 1° eta grid matches the C binary's default binning.
    eta_bins = np.arange(-180, 181, 1.0)
    grid = np.full((len(rings), len(eta_bins) - 1), np.nan)

    for i, ring in enumerate(rings):
        mask = (corr["RingNr"] == ring) & (corr["Outlier"] == 0)
        if not mask.any():
            continue
        eta = corr["Eta"][mask]
        dr = corr["DeltaR"][mask]
        # Histogram into 1° bins, averaging DeltaR within each bin.
        sums, _ = np.histogram(eta, bins=eta_bins, weights=dr)
        counts, _ = np.histogram(eta, bins=eta_bins)
        with np.errstate(invalid="ignore", divide="ignore"):
            grid[i] = np.where(counts > 0, sums / counts, np.nan)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.figure

    vmax = np.nanpercentile(np.abs(grid), 95) if np.isfinite(grid).any() else 1.0
    im = ax.imshow(
        grid, aspect="auto", origin="lower", cmap=cmap,
        vmin=-vmax, vmax=vmax,
        extent=[eta_bins[0], eta_bins[-1], rings[0] - 0.5, rings[-1] + 0.5],
        interpolation="nearest",
    )
    fig.colorbar(im, ax=ax, label="ΔR (µm)")
    ax.set_xlabel("η (deg)")
    ax.set_ylabel("Ring #")
    ax.set_title("Residual ΔR (fitted − ideal)")
    fig.tight_layout()

    if save is not None:
        fig.savefig(save, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Fourier harmonics — per-ring DFT of ΔR(η).
# ---------------------------------------------------------------------------

def fourier_harmonics(
    result: CalibrationResult,
    *,
    save: str | Path | None = None,
    ax=None,
    n_harmonics: int = 8,
    cmap: str = "magma",
):
    """Per-ring Fourier-harmonic amplitudes of ΔR(η), k = 1 … n_harmonics.

    Exposes angular structure in residuals that a well-calibrated
    distortion model should have suppressed. A bright column at k = 2
    typically points to a residual tilt; k = 4 at an astigmatism, etc.
    """
    import matplotlib.pyplot as plt

    _require_corr(result)
    corr = _load_corr_csv(result.corr_csv_path)

    rings = np.unique(corr["RingNr"]).astype(int)
    eta_bins = np.arange(-180, 181, 1.0)
    eta_centers = (eta_bins[:-1] + eta_bins[1:]) / 2
    amps = np.zeros((len(rings), n_harmonics))

    for i, ring in enumerate(rings):
        mask = (corr["RingNr"] == ring) & (corr["Outlier"] == 0)
        if not mask.any():
            continue
        eta = corr["Eta"][mask]
        dr = corr["DeltaR"][mask]
        # Resample onto a uniform 1° grid via bin-averaged histogram so
        # the DFT makes sense (non-uniform eta is common post-masking).
        sums, _ = np.histogram(eta, bins=eta_bins, weights=dr)
        counts, _ = np.histogram(eta, bins=eta_bins)
        with np.errstate(invalid="ignore", divide="ignore"):
            dr_grid = np.where(counts > 0, sums / counts, np.nan)
        # Fill NaNs with ring-mean so the DFT is well-defined.
        nan = ~np.isfinite(dr_grid)
        if nan.all():
            continue
        dr_grid[nan] = np.nanmean(dr_grid)
        spectrum = np.abs(np.fft.rfft(dr_grid))
        # Normalise so k=0 (mean) stays off the plot; amplitudes
        # peak-to-peak per harmonic.
        amps[i] = 2 * spectrum[1 : n_harmonics + 1] / len(dr_grid)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.figure

    im = ax.imshow(
        amps, aspect="auto", origin="lower", cmap=cmap,
        extent=[0.5, n_harmonics + 0.5, rings[0] - 0.5, rings[-1] + 0.5],
    )
    fig.colorbar(im, ax=ax, label="Amplitude (µm)")
    ax.set_xlabel("Harmonic k")
    ax.set_ylabel("Ring #")
    ax.set_title(f"ΔR(η) Fourier amplitudes (k=1…{n_harmonics})")
    ax.set_xticks(range(1, n_harmonics + 1))
    fig.tight_layout()

    if save is not None:
        fig.savefig(save, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Distortion field — per-pixel |ΔR| from corr.csv, scatter on detector.
# ---------------------------------------------------------------------------

def distortion_field(
    result: CalibrationResult,
    *,
    save: str | Path | None = None,
    ax=None,
    cmap: str = "plasma",
    dot_size: float = 1.0,
):
    """Scatter plot of |ΔR| at each fitted per-point location.

    Lighter patches = higher residual. Shows spatial structure of
    unmodelled distortion on the detector.
    """
    import matplotlib.pyplot as plt

    _require_corr(result)
    corr = _load_corr_csv(result.corr_csv_path)
    mask = corr["Outlier"] == 0
    if not mask.any():
        raise ValueError("corr.csv has no inlier points to plot.")

    y = corr["YRawCorr"][mask]
    z = corr["ZRawCorr"][mask]
    abs_dr = np.abs(corr["DeltaR"][mask])

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7))
    else:
        fig = ax.figure

    vmax = np.percentile(abs_dr, 95) if abs_dr.size else 1.0
    sc = ax.scatter(y, z, c=abs_dr, s=dot_size, cmap=cmap, vmin=0, vmax=vmax)
    fig.colorbar(sc, ax=ax, label="|ΔR| (µm)")
    ax.plot(result.geometry.ybc, result.geometry.zbc, "+",
            color="white", markersize=14, mew=2)
    ax.set_xlabel("Y (pixels)")
    ax.set_ylabel("Z (pixels)")
    ax.set_aspect("equal")
    ax.set_title("Residual distortion field")
    fig.tight_layout()

    if save is not None:
        fig.savefig(save, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# All-in-one bundle.
# ---------------------------------------------------------------------------

def inspect(
    result: CalibrationResult,
    image: str | Path | np.ndarray | None = None,
    out_dir: str | Path = ".",
    *,
    prefix: str = "calib",
) -> dict[str, Path]:
    """Render the full five-plot inspection bundle as PNGs.

    Returns a dict ``{name: path}`` of the written files. Skips plots
    whose inputs aren't available (e.g. ``rings_overlay`` when no image
    is passed, ``residual_heatmap`` when no ``corr.csv``).
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    import matplotlib.pyplot as plt

    try:
        convergence(result, save=out / f"{prefix}_convergence.png")
        written["convergence"] = out / f"{prefix}_convergence.png"
        plt.close("all")
    except ValueError:
        pass

    if image is not None:
        rings_overlay(result, image, save=out / f"{prefix}_rings.png")
        written["rings"] = out / f"{prefix}_rings.png"
        plt.close("all")

    if result.corr_csv_path and result.corr_csv_path.exists():
        residual_heatmap(result, save=out / f"{prefix}_residual.png")
        written["residual"] = out / f"{prefix}_residual.png"
        plt.close("all")
        fourier_harmonics(result, save=out / f"{prefix}_fourier.png")
        written["fourier"] = out / f"{prefix}_fourier.png"
        plt.close("all")
        distortion_field(result, save=out / f"{prefix}_distortion.png")
        written["distortion"] = out / f"{prefix}_distortion.png"
        plt.close("all")

    return written


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_image(path: str | Path) -> np.ndarray:
    import tifffile
    p = Path(path)
    if p.suffix.lower() in (".tif", ".tiff"):
        return tifffile.imread(p).astype(float)
    # h5/zarr paths are not shipped in v0.1.0 viz — users can pre-load.
    raise ValueError(f"Unsupported image format for viz: {p.suffix}. "
                     "Load it yourself (e.g. via h5py) and pass a numpy array.")


def _require_corr(result: CalibrationResult) -> None:
    if not result.corr_csv_path or not result.corr_csv_path.exists():
        raise ValueError(
            "result.corr_csv_path is missing. The calibration may have run "
            "with n_iterations=0 or aborted before writing corr.csv."
        )
