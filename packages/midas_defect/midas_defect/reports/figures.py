"""Publication-figure scaffolding for the matrix-twin asymmetry write-up.

Each generator takes a list of :class:`AnalysisResult` and an output path and
produces a single figure. The styling here is intentionally minimal -- final
paper polish (font sizes, panel labels, custom palette) is expected to be
applied in the paper's notebook, not in the library.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..types import AnalysisResult


def _by_name(results: list[AnalysisResult], name: str) -> AnalysisResult | None:
    for r in results:
        if r.name == name:
            return r
    return None


def _setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def matrix_twin_summary_figure(
    results: list[AnalysisResult],
    output_path: str | Path,
    metrics: list[str] | None = None,
) -> None:
    """8-panel matrix-vs-twin asymmetry summary.

    For each requested metric in ``metrics``, plot the matrix bootstrap CI
    against the twin bootstrap CI as a 2-bar errorbar. Metrics whose names
    end in ``"_matrix"`` and ``"_twin"`` (e.g. ``rho_matrix`` / ``rho_twin``)
    are paired automatically.
    """
    plt = _setup_matplotlib()
    if metrics is None:
        metrics = sorted({r.name[:-len("_matrix")] for r in results if r.name.endswith("_matrix")})
    n_metrics = len(metrics)
    n_cols = min(4, max(1, n_metrics))
    n_rows = (n_metrics + n_cols - 1) // n_cols if n_metrics else 1

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.0 * n_cols, 3.0 * n_rows))
    axes = np.atleast_1d(axes).ravel()

    for ax_idx, ax in enumerate(axes):
        if ax_idx >= len(metrics):
            ax.axis("off")
            continue
        m = metrics[ax_idx]
        rm = _by_name(results, f"{m}_matrix")
        rt = _by_name(results, f"{m}_twin")
        if rm is None or rt is None:
            ax.set_title(f"{m} (missing)")
            ax.axis("off")
            continue
        for i, r in enumerate([rm, rt]):
            ax.errorbar(
                i,
                r.population_median,
                yerr=[[r.population_median - r.population_ci[0]],
                      [r.population_ci[1] - r.population_median]],
                fmt="o",
                capsize=5,
            )
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["matrix", "twin"])
        ax.set_ylabel(rm.units)
        ax.set_title(m)
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def schmid_mechanism_figure(
    schmid_per_pair: np.ndarray,
    dEps_per_pair: np.ndarray,
    tier_edges: np.ndarray,
    output_path: str | Path,
) -> None:
    """Schmid-stratified twin-shear projection: bar chart per tercile."""
    plt = _setup_matplotlib()
    fig, ax = plt.subplots(figsize=(6, 4))
    n_tiers = len(tier_edges) - 1
    tier_idx = np.digitize(schmid_per_pair, tier_edges[1:-1], right=False)
    medians = np.array([np.nanmedian(dEps_per_pair[tier_idx == t]) for t in range(n_tiers)])
    p16 = np.array([np.nanpercentile(dEps_per_pair[tier_idx == t], 16) for t in range(n_tiers)])
    p84 = np.array([np.nanpercentile(dEps_per_pair[tier_idx == t], 84) for t in range(n_tiers)])
    err = np.vstack([medians - p16, p84 - medians])
    labels = [
        f"T{t+1}: {tier_edges[t]:.2f}-{tier_edges[t+1]:.2f}" for t in range(n_tiers)
    ]
    ax.errorbar(np.arange(n_tiers), medians, yerr=err, fmt="o", capsize=5)
    ax.set_xticks(np.arange(n_tiers))
    ax.set_xticklabels(labels, rotation=0)
    ax.axhline(0, ls="--", color="gray", lw=0.5)
    ax.set_xlabel("Schmid-factor tier")
    ax.set_ylabel("$\\Delta\\varepsilon$ on active twin shear")
    ax.set_title("Schmid-stratified mechanism signal")
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def length_scale_hierarchy(
    length_scales: dict[str, tuple[float, float, float]],
    output_path: str | Path,
) -> None:
    """Horizontal log-scale errorbar plot for a small dict of named length scales.

    ``length_scales`` maps label -> (median, ci_low, ci_high) all in the same
    unit (default Angstrom). Plotted on a single log axis to show the multi-
    decade hierarchy from sub-nm (lamella) to mm (sample).
    """
    plt = _setup_matplotlib()
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = list(length_scales.keys())
    meds = np.array([length_scales[l][0] for l in labels])
    los = np.array([length_scales[l][1] for l in labels])
    his = np.array([length_scales[l][2] for l in labels])
    err = np.vstack([meds - los, his - meds])
    ax.errorbar(meds, np.arange(len(labels)), xerr=err, fmt="o", capsize=4)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xscale("log")
    ax.set_xlabel("length scale ($\\AA$)")
    ax.set_title("Defect length-scale hierarchy")
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def cover_figure(
    results: list[AnalysisResult],
    output_path: str | Path,
    headline_metrics: list[str] | None = None,
) -> None:
    """Compact 2-panel cover figure.

    Left: KDE of per-grain dislocation density (matrix + twin overlaid).
    Right: error-bar comparison of any headline metrics passed in.
    """
    plt = _setup_matplotlib()
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(10, 4))

    rho_matrix = _by_name(results, "rho_matrix")
    rho_twin = _by_name(results, "rho_twin")
    plotted_any = False
    if rho_matrix is not None and rho_matrix.per_grain is not None:
        data_m = rho_matrix.per_grain[np.isfinite(rho_matrix.per_grain)]
        if data_m.size:
            axL.hist(np.log10(np.clip(data_m, 1e-30, None)), bins=30, alpha=0.5, label="matrix")
            plotted_any = True
    if rho_twin is not None and rho_twin.per_grain is not None:
        data_t = rho_twin.per_grain[np.isfinite(rho_twin.per_grain)]
        if data_t.size:
            axL.hist(np.log10(np.clip(data_t, 1e-30, None)), bins=30, alpha=0.5, label="twin")
            plotted_any = True
    axL.set_xlabel("$\\log_{10}(\\rho)$  (m$^{-2}$)")
    axL.set_ylabel("count")
    if plotted_any:
        axL.legend(loc="best")
    axL.set_title("Per-grain dislocation density")

    if headline_metrics is None:
        headline_metrics = [r.name for r in results[:4]]
    meds = []
    errs = []
    labels = []
    for m in headline_metrics:
        r = _by_name(results, m)
        if r is None:
            continue
        labels.append(m)
        meds.append(r.population_median)
        errs.append((r.population_ci[1] - r.population_ci[0]) / 2.0)
    if meds:
        axR.barh(np.arange(len(labels)), meds, xerr=errs, capsize=4)
        axR.set_yticks(np.arange(len(labels)))
        axR.set_yticklabels(labels)
    axR.set_title("Headline metrics")
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


__all__ = [
    "cover_figure",
    "length_scale_hierarchy",
    "matrix_twin_summary_figure",
    "schmid_mechanism_figure",
]
