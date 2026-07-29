"""Williamson-Hall-style mosaicity and dislocation-density analysis.

For a heavily-deformed single crystal, the per-hkl asterism width Σ encodes
two contributions:
  * Size broadening (≈ constant in q-space, independent of |q|): finite
    coherence-domain size.
  * Strain broadening (∝ |q|): lattice-distortion field.

The Williamson-Hall plot is:  σ(|q|) = σ_size  +  ε · |q|
where σ is the per-hkl angular asterism width converted to q-units. The slope
ε is the average strain magnitude; the intercept σ_size relates to the
coherent domain size D via D = 2π / σ_size.

Dislocation density estimate (per Ungár-Borbély, simplified):
  ρ = (2 / b²) · ε²
where b is the Burgers vector magnitude.

The plain `williamson_hall` / `dislocation_density_per_grain` estimators here are
phenomenological / first-order — they give one number per layer (or grain), not a
per-hkl strain field, and treat every reflection as equally strain-sensitive
(elastically isotropic). The "modified-Williamson-Hall" treatment (Ungár &
Borbély 1996) that corrects for strain anisotropy via the dislocation contrast
factor is provided by `modified_williamson_hall` below, driven by
`contrast_factor.average_contrast_factor`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import math
import numpy as np


__all__ = ["WHResult", "williamson_hall",
           "DislocationResult", "dislocation_density_per_grain",
           "ModifiedWHResult", "contrast_factors_for_fits",
           "modified_williamson_hall"]


@dataclass
class WHResult:
    sigma_size: float           # intercept (1/Å)
    strain_eps: float           # slope (dimensionless)
    domain_size_A: float        # 2π / sigma_size
    dislocation_density_per_m2: float  # using ε and supplied Burgers vector
    correlation: float          # Pearson ρ of σ vs |q|
    n_hkls: int
    residual_norm: float
    q_values: list              # for plotting
    sigma_values: list


def williamson_hall(
    fits: Sequence,                # list[AsterismFit]
    *,
    burgers_A: float = 4.29,        # Å; default = a/sqrt(2) for (110)-type Burgers in θ-Al₂Cu
    sigma_kind: str = "max",        # "max" | "mean" of per-hkl Σ eigenvalues
    weight_by_intensity: bool = True,
) -> WHResult:
    """Fit σ(|q|) = σ_size + ε · |q| via weighted linear regression.

    Returns size-broadening intercept σ_size, strain slope ε, and a derived
    dislocation density assuming the supplied Burgers-vector magnitude.
    """
    if len(fits) < 4:
        raise ValueError(
            f"need at least 4 asterism fits to do a meaningful WH plot; got {len(fits)}"
        )
    q_vals = np.array([np.linalg.norm(f.q_fit) for f in fits])
    if sigma_kind == "max":
        sigma_vals = np.array([float(f.sigma_eig.max()) for f in fits])
    elif sigma_kind == "mean":
        sigma_vals = np.array([float(f.sigma_eig.mean()) for f in fits])
    else:
        raise ValueError(f"unknown sigma_kind: {sigma_kind!r}")
    if weight_by_intensity:
        w = np.array([f.integrated_intensity for f in fits])
        w = w / w.sum()
    else:
        w = np.ones(len(fits)) / len(fits)

    # Weighted linear regression σ = a + b · q
    q_bar = (w * q_vals).sum()
    s_bar = (w * sigma_vals).sum()
    cov = (w * (q_vals - q_bar) * (sigma_vals - s_bar)).sum()
    var = (w * (q_vals - q_bar) ** 2).sum()
    if var <= 0:
        slope = 0.0
        intercept = float(s_bar)
    else:
        slope = float(cov / var)
        intercept = float(s_bar - slope * q_bar)
    pred = intercept + slope * q_vals
    residual = float(np.linalg.norm(sigma_vals - pred))
    # Pearson correlation (unweighted, for reporting)
    corr = float(np.corrcoef(q_vals, sigma_vals)[0, 1]) if len(q_vals) > 1 else 0.0

    domain_size = 2.0 * math.pi / max(intercept, 1e-12)
    # Ungár-Borbély-style ρ = 2 ε² / b²; ε is the slope above; convert b from Å to m.
    b_m = burgers_A * 1e-10
    rho = 2.0 * (slope ** 2) / (b_m ** 2)

    return WHResult(
        sigma_size=float(intercept),
        strain_eps=float(slope),
        domain_size_A=float(domain_size),
        dislocation_density_per_m2=float(rho),
        correlation=corr,
        n_hkls=len(fits),
        residual_norm=residual,
        q_values=q_vals.tolist(),
        sigma_values=sigma_vals.tolist(),
    )


# ---------------------------------------------------------------------------
# Modified Williamson-Hall: contrast-factor-corrected strain anisotropy
# ---------------------------------------------------------------------------
#
# The plain `williamson_hall` above plots σ vs |q|, which assumes every hkl is
# equally strain-sensitive (elastically isotropic). In a real crystal the strain
# broadening is hkl-dependent. The *modified* Williamson-Hall method (Ungár &
# Borbély, Appl. Phys. Lett. 69 (1996) 3173) replaces |q| with |q|·C̄^{1/2},
# where C̄ is the dislocation contrast factor (see `contrast_factor`); the
# anisotropic scatter then collapses onto a single line. A higher post-correction
# correlation is the diagnostic that dislocations dominate the broadening.

@dataclass
class ModifiedWHResult:
    """Contrast-factor-corrected Williamson-Hall fit: σ = σ_size + ε·(|q|·C̄^{1/2})."""
    sigma_size: float                  # intercept (1/Å)
    strain_eps: float                  # slope vs |q|·C̄^{1/2}
    domain_size_A: float               # 2π / sigma_size
    dislocation_density_per_m2: float  # 2 ε² / b² (Ungár-Borbély simplified)
    correlation: float                 # Pearson ρ of σ vs |q|·C̄^{1/2}
    correlation_uncorrected: float     # Pearson ρ of σ vs |q| (for comparison)
    n_hkls: int
    residual_norm: float
    contrast_factors: list             # C̄ used per fit
    x_values: list                     # |q|·C̄^{1/2}
    sigma_values: list


def contrast_factors_for_fits(
    fits: Sequence,
    stiffness_voigt,
    *,
    crystal=None,
    family: str = "fcc",
    character: str = "screw",
    n_phi: int = 720,
) -> list:
    """Average contrast factor C̄ for each fit, keyed on its ``hkl``.

    Uses :func:`contrast_factor.average_contrast_factor` (slip-system average for
    the given crystal family and dislocation character). Identical ``hkl`` are
    computed once and cached. Pass ``crystal`` (a `midas_hkls.Crystal`) for any
    non-cubic cell so the Miller indices are mapped to Cartesian correctly; for a
    cubic cell it may be omitted.
    """
    from .contrast_factor import average_contrast_factor

    cache: dict = {}
    out = []
    for f in fits:
        hkl = tuple(int(x) for x in f.hkl)
        if hkl not in cache:
            cache[hkl] = float(average_contrast_factor(
                stiffness_voigt, hkl, crystal=crystal, family=family,
                character=character, n_phi=n_phi))
        out.append(cache[hkl])
    return out


def modified_williamson_hall(
    fits: Sequence,
    contrast_factors: Sequence[float],
    *,
    burgers_A: float = 2.556,        # Å; default a/√2 for Cu (a=3.615)
    sigma_kind: str = "max",
    weight_by_intensity: bool = True,
) -> ModifiedWHResult:
    """Modified Williamson-Hall: fit σ vs the contrast-corrected ``|q|·C̄^{1/2}``.

    ``contrast_factors`` must align 1:1 with ``fits`` (e.g. from
    :func:`contrast_factors_for_fits`). Returns the corrected slope/intercept plus
    both the corrected and *uncorrected* Pearson correlation, so the caller can
    see whether the contrast-factor correction tightened the plot.
    """
    if len(fits) < 4:
        raise ValueError(
            f"need at least 4 asterism fits for a modified-WH plot; got {len(fits)}")
    if len(contrast_factors) != len(fits):
        raise ValueError(
            f"contrast_factors ({len(contrast_factors)}) must align with fits ({len(fits)})")

    q_vals = np.array([np.linalg.norm(f.q_fit) for f in fits])
    cbar = np.asarray(contrast_factors, dtype=float)
    if np.any(cbar < 0):
        raise ValueError("contrast factors must be non-negative")
    x_vals = q_vals * np.sqrt(cbar)

    if sigma_kind == "max":
        sigma_vals = np.array([float(f.sigma_eig.max()) for f in fits])
    elif sigma_kind == "mean":
        sigma_vals = np.array([float(f.sigma_eig.mean()) for f in fits])
    else:
        raise ValueError(f"unknown sigma_kind: {sigma_kind!r}")

    if weight_by_intensity:
        w = np.array([f.integrated_intensity for f in fits])
        w = w / w.sum()
    else:
        w = np.ones(len(fits)) / len(fits)

    x_bar = (w * x_vals).sum()
    s_bar = (w * sigma_vals).sum()
    cov = (w * (x_vals - x_bar) * (sigma_vals - s_bar)).sum()
    var = (w * (x_vals - x_bar) ** 2).sum()
    if var <= 0:
        slope, intercept = 0.0, float(s_bar)
    else:
        slope = float(cov / var)
        intercept = float(s_bar - slope * x_bar)
    pred = intercept + slope * x_vals
    residual = float(np.linalg.norm(sigma_vals - pred))

    corr = float(np.corrcoef(x_vals, sigma_vals)[0, 1]) if len(x_vals) > 1 else 0.0
    corr_unc = float(np.corrcoef(q_vals, sigma_vals)[0, 1]) if len(q_vals) > 1 else 0.0

    domain_size = 2.0 * math.pi / max(intercept, 1e-12)
    b_m = burgers_A * 1e-10
    rho = 2.0 * (slope ** 2) / (b_m ** 2)

    return ModifiedWHResult(
        sigma_size=float(intercept),
        strain_eps=float(slope),
        domain_size_A=float(domain_size),
        dislocation_density_per_m2=float(rho),
        correlation=corr,
        correlation_uncorrected=corr_unc,
        n_hkls=len(fits),
        residual_norm=residual,
        contrast_factors=cbar.tolist(),
        x_values=x_vals.tolist(),
        sigma_values=sigma_vals.tolist(),
    )


@dataclass
class DislocationResult:
    """Per-grain Williamson-Hall dislocation density from Bragg radial breadth."""
    rho_median_per_m2: float
    rho_per_grain: np.ndarray
    domain_size_A_median: float
    microstrain_median: float
    burgers_A: float
    n_grains_fit: int
    n_grains: int


def dislocation_density_per_grain(
    q_sample: np.ndarray,
    intensity: np.ndarray,
    orientations,
    crystal,
    *,
    burgers_A: float | None = None,
    prefactor: float = 16.1,
    bragg_tol_inv_A: float = 0.05,
    query_radius_inv_A: float = 0.08,
    q_max_inv_A: float = 8.0,
    min_reflections: int = 4,
    min_voxels_per_peak: int = 6,
) -> DislocationResult:
    """Per-grain dislocation density via Williamson-Hall on Bragg radial breadth.

    The validated demk method (`scripts/c1_williamson_hall.py`): for each grain,
    measure the intensity-weighted radial second moment of the diffuse cloud
    about each of its Bragg reflections → an FWHM-equivalent breadth β(|q|); fit
    β = 2π/D + ε·|q| per grain; report ρ = ``prefactor``·ε²/b² (Williamson-Hall,
    prefactor ≈ 16.1) and the coherent domain size D = 2π/intercept.

    Burgers vector defaults to the FCC full dislocation b = a/√2 from the
    crystal's cell unless ``burgers_A`` is given.

    Caveats (carry these): no instrumental-resolution deconvolution → ρ is an
    upper bound; the prefactor is convention-dependent. Defensible to ~an order
    of magnitude.
    """
    from scipy.spatial import cKDTree
    from .bragg_diffuse import predicted_reflection_points, classify_voxels, enumerate_hkls

    q = np.asarray(q_sample, dtype=np.float64)
    I = np.asarray(intensity, dtype=np.float64)
    OM = np.asarray(orientations, dtype=np.float64)
    if OM.ndim == 2 and OM.shape[-1] == 9:
        OM = OM.reshape(-1, 3, 3)
    elif OM.shape == (3, 3):
        OM = OM.reshape(1, 3, 3)

    if burgers_A is None:
        burgers_A = float(crystal.lattice.a) / math.sqrt(2.0)
    b_m = burgers_A * 1e-10

    # Bragg (on-lattice) voxels only
    P_all = predicted_reflection_points(OM, crystal, q_max_inv_A=q_max_inv_A).numpy()
    split = classify_voxels(q, I, P_all, tol_inv_A=bragg_tol_inv_A)
    on = split.on_lattice
    qb = q[on]
    wb = I[on]
    if len(qb) < min_voxels_per_peak:
        return DislocationResult(float("nan"), np.array([]), float("nan"),
                                 float("nan"), burgers_A, 0, len(OM))
    treeB = cKDTree(qb)

    hkls = enumerate_hkls(crystal, q_max_inv_A=q_max_inv_A)
    G = np.empty((len(hkls), 3))
    a, c = float(crystal.lattice.a), float(crystal.lattice.c)
    twopi = 2.0 * math.pi
    G[:, 0] = twopi * hkls[:, 0] / a
    G[:, 1] = twopi * hkls[:, 1] / a
    G[:, 2] = twopi * hkls[:, 2] / c

    rho_list, D_list, eps_list = [], [], []
    for gi in range(len(OM)):
        B = (OM[gi] @ G.T).T
        Bm = np.linalg.norm(B, axis=1)
        keep = Bm < q_max_inv_A
        B, Bm = B[keep], Bm[keep]
        qv, bw = [], []
        for j in range(len(B)):
            ii = treeB.query_ball_point(B[j], query_radius_inv_A)
            if len(ii) < min_voxels_per_peak:
                continue
            Pj = qb[ii]; wj = wb[ii]
            rad = (Pj - B[j]) @ (B[j] / Bm[j])
            mu = np.average(rad, weights=wj)
            var = np.average((rad - mu) ** 2, weights=wj)
            qv.append(Bm[j]); bw.append(2.355 * math.sqrt(max(var, 1e-12)))
        if len(qv) < min_reflections:
            continue
        slope, intercept = np.polyfit(np.array(qv), np.array(bw), 1)
        if intercept <= 0 or slope <= 0:
            continue
        eps = slope / 2.0
        rho_list.append(prefactor * eps * eps / (b_m ** 2))
        D_list.append(twopi / intercept)
        eps_list.append(eps)

    rho = np.asarray(rho_list)
    return DislocationResult(
        rho_median_per_m2=float(np.median(rho)) if len(rho) else float("nan"),
        rho_per_grain=rho,
        domain_size_A_median=float(np.median(D_list)) if D_list else float("nan"),
        microstrain_median=float(np.median(eps_list)) if eps_list else float("nan"),
        burgers_A=burgers_A,
        n_grains_fit=len(rho),
        n_grains=len(OM),
    )
