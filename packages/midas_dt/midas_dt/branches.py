"""The two reconstruction routes, and the diagnostic that compares them.

Branch A -- fit, then reconstruct
    Fit each *projection's* lineout, then reconstruct the resulting parameter
    sinograms. Cheap: 12 reconstructions per channel, independent of binning.

Branch B -- reconstruct, then fit
    Reconstruct every (R, eta) bin, then fit the pattern at each voxel.
    Correct for all parameters, and costs ``n_r * n_eta`` reconstructions.

Both are supported because they answer different questions at different
prices. What must not be silent is *when Branch A is valid*.

Why Branch A needs care
-----------------------
Radon inversion is linear; peak fitting is not. They do not commute. A
quantity that adds along a ray may be back-projected directly; one that does
not, cannot:

* ``TotalIntensity``, ``TotalIntensityBackgroundCorr``,
  ``FitIntegratedIntensity`` add. Reconstruct them directly.
* ``RMEAN``, ``SigmaG``, ``SigmaL``, ``MixFactor`` do not. A projection's
  fitted ``RMEAN`` is the *intensity-weighted mean* of the voxel values along
  the ray, not their sum, so back-projecting it produces a number with no
  physical meaning -- and it looks entirely reasonable.

The fix is not to forbid Branch A but to reconstruct the moments:

    RMEAN_voxel = recon(RMEAN_proj * I_proj) / recon(I_proj)

Both numerator and denominator DO add along the ray, so this is correct
wherever the first-order weighted-mean approximation holds: a single peak,
shifts small compared with the peak width. It breaks when ``RMEAN`` is
multi-modal along a ray -- two phases with different d-spacing in one column --
because then the projection fit has no single peak to report.

That is what ``weighting="intensity"`` does, and it is the default for
non-additive quantities. ``weighting="none"`` reproduces the legacy
behaviour and marks its output ``linearity="approximate"``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from .channels import Channel
from .conventions import (
    FIT_OUTPUT_NAMES,
    RECON_SIGN,
    ScanKnownLimits,
    fit_output_index,
    is_additive,
)
from .peakfit import fit_lineout
from .recon import Reconstruction, reconstruct
from .sinogram import SinogramStack

__all__ = ["BranchResult", "run_fit_then_recon", "run_recon_then_fit", "compare"]

log = logging.getLogger(__name__)


@dataclass
class BranchResult:
    """Per-voxel maps of the fit outputs, however they were obtained."""

    maps: dict[str, np.ndarray]        # name -> (X, X)
    branch: str
    channel: Channel
    limits: ScanKnownLimits
    linearity: dict[str, str] = field(default_factory=dict)
    sigma: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def size(self) -> int:
        return int(next(iter(self.maps.values())).shape[-1])

    def approximate_outputs(self) -> list[str]:
        """Outputs whose map rests on an approximation, worth stating."""
        return sorted(k for k, v in self.linearity.items() if v != "exact")

    def describe(self) -> str:
        approx = self.approximate_outputs()
        s = (f"{self.branch}: {len(self.maps)} maps of {self.size}x{self.size} "
             f"for {self.channel.label}")
        if approx:
            s += f"\n  APPROXIMATE: {', '.join(approx)}"
        return s


# ------------------------------------------------------------------ Branch A
def run_fit_then_recon(
    stack: SinogramStack,
    *,
    weighting: str = "intensity",
    outputs: tuple[str, ...] = FIT_OUTPUT_NAMES,
    shift: float = 0.0,
    n_cpus: int = 8,
    **recon_kw,
) -> BranchResult:
    """Fit every projection, then reconstruct the parameter sinograms.

    Parameters
    ----------
    weighting : {'intensity', 'none'}
        ``'intensity'`` (default) reconstructs non-additive quantities as
        ``recon(x*I)/recon(I)``. ``'none'`` back-projects the fitted value
        directly, which is what the legacy scripts did; the affected outputs
        are marked ``approximate`` in the result.
    """
    if weighting not in {"intensity", "none"}:
        raise ValueError(
            f"weighting must be 'intensity' or 'none'; got {weighting!r}"
        )

    n_eta, n_r = stack.bin_shape
    ch = stack.channel
    radii = np.linspace(ch.r_min, ch.r_max, n_r)

    # Fit every (omega, translation) lineout, eta-collapsed.
    inten = stack.intensity.reshape(n_eta, n_r, stack.n_omega, stack.n_translations)
    lineouts = inten.sum(axis=0)                     # (n_r, n_omega, n_trans)

    n_out = len(FIT_OUTPUT_NAMES)
    par = np.zeros((n_out, stack.n_omega, stack.n_translations))
    for i in range(stack.n_omega):
        for j in range(stack.n_translations):
            f = fit_lineout(radii, lineouts[:, i, j], n_peaks=ch.n_peaks,
                            peak_centres=ch.peak_centres or None)
            par[:, i, j] = f.values[0]
    log.info("fitted %d projections for %s",
             stack.n_omega * stack.n_translations, ch.label)

    i_idx = fit_output_index("TotalIntensity")
    weight = par[i_idx]

    to_build, linearity = [], {}
    for name in outputs:
        k = fit_output_index(name)
        if is_additive(name):
            to_build.append(par[k])
            linearity[name] = "exact"
        elif weighting == "intensity":
            to_build.append(par[k] * weight)         # first moment
            linearity[name] = "weighted-moment"
        else:
            to_build.append(par[k])
            linearity[name] = "approximate"
    if weighting == "intensity":
        to_build.append(weight)                      # the denominator

    sinos = np.ascontiguousarray(np.stack(to_build), dtype=np.float32)
    rec = reconstruct(
        SinogramStack(intensity=sinos, variance=np.zeros_like(sinos),
                      omega_deg=stack.omega_deg, channel=ch,
                      bin_shape=(1, sinos.shape[0]), limits=stack.limits),
        shift=shift, n_cpus=n_cpus, apply_sign=True, **recon_kw,
    )

    maps: dict[str, np.ndarray] = {}
    denom = rec.intensity[-1] if weighting == "intensity" else None
    for n, name in enumerate(outputs):
        img = rec.intensity[n]
        if linearity[name] == "weighted-moment":
            with np.errstate(divide="ignore", invalid="ignore"):
                img = np.where(np.abs(denom) > 1e-12, img / denom, np.nan)
        maps[name] = img

    if weighting == "intensity":
        log.info("non-additive outputs reconstructed as recon(x*I)/recon(I)")
    else:
        bad = [n for n in outputs if linearity[n] == "approximate"]
        if bad:
            log.warning(
                "weighting='none': %s were back-projected directly. Radon "
                "inversion is linear and these do not add along a ray, so the "
                "maps have no physical interpretation. Use "
                "weighting='intensity' or Branch B.", ", ".join(bad),
            )
    return BranchResult(maps=maps, branch=f"fit-then-recon[{weighting}]",
                        channel=ch, limits=stack.limits, linearity=linearity)


# ------------------------------------------------------------------ Branch B
def run_recon_then_fit(
    stack: SinogramStack,
    *,
    outputs: tuple[str, ...] = FIT_OUTPUT_NAMES,
    shift: float = 0.0,
    n_cpus: int = 8,
    mask_threshold: float | None = None,
    **recon_kw,
) -> BranchResult:
    """Reconstruct every bin, then fit the pattern at each voxel.

    Exact for every output, at ``n_r * n_eta`` reconstructions. ``recon_kw``
    is forwarded to :func:`~midas_dt.recon.reconstruct`, so
    ``variance_samples=K`` gives per-voxel sigma via Monte Carlo.

    ``mask_threshold`` skips voxels whose pattern total falls below it -- air
    around the sample, where a fit converges to noise and costs as much as a
    real one.
    """
    rec = reconstruct(stack, shift=shift, n_cpus=n_cpus, **recon_kw)
    n_eta, n_r = rec.bin_shape
    ch = stack.channel
    radii = np.linspace(ch.r_min, ch.r_max, n_r)
    size = rec.size

    maps = {n: np.full((size, size), np.nan) for n in outputs}
    idx = {n: fit_output_index(n) for n in outputs}

    totals = rec.intensity.sum(axis=0)
    thr = (mask_threshold if mask_threshold is not None
           else float(np.nanpercentile(totals, 60)))
    fitted = 0
    for iy in range(size):
        for ix in range(size):
            if totals[iy, ix] <= thr:
                continue
            pattern = rec.voxel_pattern(iy, ix).sum(axis=0)   # collapse eta
            f = fit_lineout(radii, pattern, n_peaks=ch.n_peaks,
                            peak_centres=ch.peak_centres or None)
            for n in outputs:
                maps[n][iy, ix] = f.values[0, idx[n]]
            fitted += 1
    log.info("fitted %d of %d voxels (threshold %.3g)", fitted, size * size, thr)

    return BranchResult(
        maps=maps, branch="recon-then-fit", channel=ch, limits=stack.limits,
        linearity={n: "exact" for n in outputs},
    )


# ------------------------------------------------------------------- compare
def compare(a: BranchResult, b: BranchResult) -> dict[str, dict[str, float]]:
    """Per-output discrepancy between two branches.

    Two methods that disagree should be reported as two methods that disagree,
    with the size measured on the user's own data. That is the honest
    presentation, and it is what tells you whether Branch A's approximation is
    acceptable for a given sample.

    Returns, per output: ``rel_rms``, ``rel_max``, ``corr`` and ``n``, over the
    voxels where both branches produced a finite value.
    """
    out: dict[str, dict[str, float]] = {}
    for name in sorted(set(a.maps) & set(b.maps)):
        x, y = a.maps[name], b.maps[name]
        good = np.isfinite(x) & np.isfinite(y)
        if good.sum() < 2:
            out[name] = {"n": int(good.sum()), "rel_rms": float("nan"),
                         "rel_max": float("nan"), "corr": float("nan")}
            continue
        xv, yv = x[good], y[good]
        scale = max(float(np.nanmax(np.abs(yv))), 1e-12)
        d = xv - yv
        corr = (float(np.corrcoef(xv, yv)[0, 1])
                if xv.std() > 0 and yv.std() > 0 else float("nan"))
        out[name] = {
            "n": int(good.sum()),
            "rel_rms": float(np.sqrt(np.mean(d ** 2)) / scale),
            "rel_max": float(np.max(np.abs(d)) / scale),
            "corr": corr,
        }
    return out


def format_comparison(stats: dict[str, dict[str, float]],
                      a: BranchResult, b: BranchResult) -> str:
    """A table of :func:`compare`'s output, with the caveats attached."""
    lines = [f"{a.branch}  vs  {b.branch}", "",
             f"{'output':<30} {'rel_rms':>9} {'rel_max':>9} {'corr':>7} {'n':>7}"]
    for name, s in stats.items():
        lines.append(f"{name:<30} {s['rel_rms']:>9.3e} {s['rel_max']:>9.3e} "
                     f"{s['corr']:>7.3f} {s['n']:>7d}")
    approx = a.approximate_outputs() + b.approximate_outputs()
    if approx:
        lines += ["", "Approximate in at least one branch: " + ", ".join(sorted(set(approx)))]
    lines += [""] + [f"NOTE: {w}" for w in a.limits.warnings()]
    return "\n".join(lines)
