"""Joint-NNLS grain volume corrector.

Replaces the standard ``GrainRadius = mean(R_per_spot)`` formula with a
sparse non-negative least squares solution that accounts for *shared spots*
between twin partners, sub-grains and crowded grain pairs.

The standard MIDAS formula attributes the **total** observed intensity at a
detector spot to every grain that matched it. Twin partners (Σ3, Σ9, Σ27) and
neighbouring grains often share reflections exactly — the total intensity is
the SUM of both grains' contributions, but each grain reports it as its own,
inflating both R values by 20–40 %.

The joint-NNLS approach:

.. math::

   I_{\\text{obs}}(s) \\;=\\; \\sum_{g\\,\\text{claiming}\\,s}\\, V_g \\cdot K(\\text{ring of }s)

with :math:`K(\\text{ring})` the per-ring intensity normaliser (proportional to
:math:`|F|^2 \\cdot \\mathrm{LP} \\cdot \\mathrm{DWF}` for that ring, with the
absolute scale absorbed into the calibrant). Setting
:math:`\\alpha(s) = I_{\\text{obs}}(s) / K(\\text{ring of }s)`, the system
becomes a sparse linear problem with non-negativity bounds on :math:`V_g`,
solved via :func:`scipy.optimize.lsq_linear`.

Grain radii are recovered as :math:`R_g = (3 V_g / 4\\pi)^{1/3}`. The absolute
volume scale is arbitrary (the calibrant is absorbed); the output is
rescaled so :math:`\\overline{R_{\\text{nnls}}} = \\overline{R_{\\text{naive}}}`
to give a direct, comparable correction.

For an isolated grain (no shared spots) :math:`R_{\\text{nnls}} = R_{\\text{naive}}`
up to the global rescale. For twin/overlap victims, :math:`R_{\\text{nnls}} < R_{\\text{naive}}`
in proportion to how much of the shared intensity actually belonged to the
partner.

References
----------
- Sharma et al. MIDAS Part I (in press, *Acta Cryst A* 2026): the standard
  formula this module corrects (§3.x grain-volume derivation).
- Lawson & Hanson, *Solving Least Squares Problems* (1974) for NNLS.
- :func:`scipy.optimize.lsq_linear` with ``method='trf'``, ``lsq_solver='lsmr'``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.optimize import lsq_linear
from scipy.sparse import csr_matrix


__all__ = [
    "compute_nnls_volumes",
    "NnlsVolumeResult",
    "physical_ring_K",
    "structure_factor_squared_fcc",
]


# ---------------------------------------------------------------------------
# Item 9: physical K(ring) = |F|² · LP · DWF · multiplicity
# ---------------------------------------------------------------------------

# Cromer-Mann (1968) coefficients for selected elements; for atomic
# scattering factor f(s) = Σ a_i exp(-b_i s²) + c, with s = sin θ / λ (Å⁻¹).
# Source: International Tables for Crystallography Vol. C, Table 6.1.1.4.
# Add elements here as needed; default falls back to a Ni-like profile when
# the species isn't tabulated.
_CROMER_MANN: Dict[str, Tuple[float, float, float, float, float, float, float, float, float]] = {
    "Ni": (12.8376, 3.8785, 7.292, 0.2565, 4.4438, 12.1763, 2.38, 66.3421, 1.0341),
    "Cu": (13.338, 3.5828, 7.1676, 0.247, 5.6158, 11.3966, 1.6735, 64.8126, 1.191),
    "Fe": (11.7695, 4.7611, 7.3573, 0.3072, 3.5222, 15.3535, 2.3045, 76.8805, 1.0369),
    "Ti": (9.7595, 7.8508, 7.3558, 0.5, 1.6991, 35.6338, 1.9021, 116.105, 1.2807),
    "Al": (6.4202, 3.0387, 1.9002, 0.7426, 1.5936, 31.5472, 1.9646, 85.0886, 1.1151),
    "Au": (16.8819, 0.4611, 18.5913, 8.6216, 25.5582, 1.4826, 5.86, 36.3956, 12.0658),
}


def _atomic_f(species: str, s: float) -> float:
    """Atomic scattering factor f(s) where s = sin θ / λ in 1/Å."""
    coef = _CROMER_MANN.get(species, _CROMER_MANN["Ni"])
    a1, b1, a2, b2, a3, b3, a4, b4, c = coef
    s2 = s * s
    return (a1 * math.exp(-b1 * s2) + a2 * math.exp(-b2 * s2)
            + a3 * math.exp(-b3 * s2) + a4 * math.exp(-b4 * s2) + c)


def structure_factor_squared_fcc(h: int, k: int, l: int, f: float) -> float:
    """|F(hkl)|² for face-centred cubic. FCC reflections are allowed when
    h, k, l are all even or all odd; |F|² = 16 f² in that case, else 0."""
    all_even = (h % 2 == 0) and (k % 2 == 0) and (l % 2 == 0)
    all_odd = (h % 2 == 1) and (k % 2 == 1) and (l % 2 == 1)
    return 16.0 * f * f if (all_even or all_odd) else 0.0


def physical_ring_K(
    hkls_df,
    *,
    wavelength: float,
    species: str = "Ni",
    B_factor: float = 0.4,
) -> Dict[int, float]:
    """Compute the physical per-ring intensity factor K(ring).

    K(ring) ∝ multiplicity(ring) · |F(ring)|² · LP(2θ) · DWF(2θ)

    where ``multiplicity(ring)`` is the number of (h,k,l) variants on that
    ring (counted from ``hkls_df`` rows per ring), ``|F|²`` is the FCC
    structure factor squared at one representative (h,k,l), ``LP(2θ)`` is
    the rotating-crystal Lorentz-polarisation factor
    ``(1 + cos²(2θ)) / (2 sin(2θ))`` (single-spot integrated over η),
    and ``DWF(2θ) = exp(-2 B sin²θ / λ²)`` is the isotropic Debye-Waller
    factor.

    The returned dict is normalised to median = 1, since only ring-to-ring
    *ratios* matter for the NNLS attribution (absolute scale is absorbed
    into the calibrant in MIDAS).

    Parameters
    ----------
    hkls_df : pandas DataFrame
        Must have columns ``h``, ``k``, ``l``, ``RingNr``, ``2Theta`` (in
        degrees). Standard MIDAS ``hkls.csv`` layout.
    wavelength : float
        X-ray wavelength in Å.
    species : str
        Atomic species; uses Cromer-Mann tabulated coefficients. Defaults
        to "Ni"; falls back to Ni-like profile if unknown.
    B_factor : float
        Debye-Waller temperature factor in Å². 0.4 is typical for Ni
        at room temperature; pure Cu is 0.55; Al is 0.85.

    Returns
    -------
    dict {ring_nr (int) → K_relative (float)}
    """
    import pandas as pd
    K_per_ring: Dict[int, float] = {}
    for r, sub in hkls_df.groupby("RingNr"):
        # Multiplicity = number of variants stored for this ring
        mult = int(len(sub))
        # Take the first hkl as representative (all variants have same |F|²
        # in cubic high-symmetry groups; FCC selection rule is the same)
        first = sub.iloc[0]
        h, k, l = int(first["h"]), int(first["k"]), int(first["l"])
        # 2θ in degrees; convert to radians
        two_theta_deg = float(first["2Theta"])
        two_theta = math.radians(two_theta_deg)
        theta = two_theta / 2.0
        # Bragg s = sin θ / λ (Å⁻¹)
        s = math.sin(theta) / max(wavelength, 1e-9)
        f = _atomic_f(species, s)
        F2 = structure_factor_squared_fcc(h, k, l, f)
        # Rotating-crystal Lorentz-polarisation (η-integrated, single-spot)
        if math.sin(two_theta) > 1e-9:
            LP = (1.0 + math.cos(two_theta) ** 2) / (2.0 * math.sin(two_theta))
        else:
            LP = 1.0
        # Debye-Waller
        DWF = math.exp(-2.0 * B_factor * s * s)
        K = mult * F2 * LP * DWF
        K_per_ring[int(r)] = float(K)
    # Normalise to median = 1 (absolute scale is degenerate)
    vals = np.array([v for v in K_per_ring.values() if v > 0], dtype=np.float64)
    if vals.size > 0:
        med = float(np.median(vals))
        if med > 0:
            for r in K_per_ring:
                K_per_ring[r] = K_per_ring[r] / med
    return K_per_ring


@dataclass
class NnlsVolumeResult:
    """Per-grain NNLS-corrected volumes/radii.

    Attributes
    ----------
    R_nnls : (n_grains,) float64
        Corrected grain radii in the same units as the input ``R_naive``.
        Rescaled so ``mean(R_nnls) == mean(R_naive)``.
    V_nnls_raw : (n_grains,) float64
        Raw NNLS-solved volumes (arbitrary units, pre-rescale).
    frac_spots_shared : (n_grains,) float64
        Per-grain fraction of assigned spots that are claimed by ≥1 other
        grain. A diagnostic of how overlap-prone each grain is.
    n_spots_shared : int
        Total unique spots claimed by ≥2 grains.
    n_unique_spots : int
        Total unique detected spots referenced by SpotMatrix.
    rescale_factor : float
        Multiplicative rescale applied to enforce mean-R parity with R_naive.
    nnls_status : int
        :func:`scipy.optimize.lsq_linear` status code.
    nnls_cost : float
        Final residual cost.
    nnls_n_iter : int
        Iteration count.
    """

    R_nnls: np.ndarray
    V_nnls_raw: np.ndarray
    frac_spots_shared: np.ndarray
    n_spots_shared: int
    n_unique_spots: int
    rescale_factor: float
    nnls_status: int
    nnls_cost: float
    nnls_n_iter: int
    # Per-grain volume σ from linearised covariance (M^T M)^-1 σ²_residual
    # for the active-set V > 0 grains. NaN for boundary grains (V ≈ 0).
    sigma_V_nnls_raw: np.ndarray | None = None
    # σ_R derived from σ_V via dR/dV = R/(3V): σ_R = σ_V · R / (3V).
    sigma_R_um: np.ndarray | None = None


def compute_nnls_volumes(
    *,
    grain_ids: np.ndarray,
    R_naive: np.ndarray,
    sm_grain_id: np.ndarray,
    sm_spot_id: np.ndarray,
    sm_ring_nr: np.ndarray,
    spot_intensity: Dict[int, float],
    spot_ring: Dict[int, int],
    ring_K: Optional[Dict[int, float]] = None,
    tol: float = 1e-6,
    max_iter: int = 200,
) -> NnlsVolumeResult:
    """Compute joint-NNLS volume-corrected grain radii.

    Parameters
    ----------
    grain_ids : (n_grains,) int
        Grain identifiers (must match ``sm_grain_id`` values).
    R_naive : (n_grains,) float
        The per-grain naive radius (mean of per-spot R), used only to set
        the absolute scale of the rescaled output.
    sm_grain_id, sm_spot_id, sm_ring_nr : (n_assigns,) int
        Three parallel arrays from SpotMatrix: which grain claimed which
        spot on which ring. One row per (grain, spot) assignment.
    spot_intensity, spot_ring : dict (SpotID → float / int)
        Per-spot integrated intensity and ring number (from InputAllExtraInfo).
    ring_K : dict (ring_nr → float), optional
        Per-ring intensity normaliser. If ``None``, derived from the data as
        the median of ``spot_intensity`` across all assigned spots on each
        ring.
    tol : float
        Stopping tolerance for :func:`lsq_linear` (relative cost change).
    max_iter : int
        Maximum :func:`lsq_linear` iterations.

    Returns
    -------
    :class:`NnlsVolumeResult`
    """
    n_grains = int(len(grain_ids))
    if n_grains == 0:
        return NnlsVolumeResult(
            R_nnls=np.empty(0), V_nnls_raw=np.empty(0),
            frac_spots_shared=np.empty(0),
            n_spots_shared=0, n_unique_spots=0,
            rescale_factor=1.0, nnls_status=-1, nnls_cost=0.0, nnls_n_iter=0,
        )
    gid_to_row = {int(g): i for i, g in enumerate(grain_ids)}

    # ----- assemble spot universe -----
    unique_spots = np.unique(sm_spot_id.astype(np.int64))
    spot_to_row = {int(s): i for i, s in enumerate(unique_spots)}
    n_spots = int(len(unique_spots))

    # per-spot claim multiplicity
    spot_claim_counts = np.zeros(n_spots, dtype=np.int64)
    for s in sm_spot_id:
        spot_claim_counts[spot_to_row[int(s)]] += 1
    n_shared = int((spot_claim_counts > 1).sum())

    # ----- derive ring_K if not given -----
    if ring_K is None:
        # median I per ring across all assigned spots
        ring_to_intensities: Dict[int, list] = {}
        for s in unique_spots:
            r = spot_ring.get(int(s), 0)
            ii = spot_intensity.get(int(s))
            if ii is None: continue
            ring_to_intensities.setdefault(r, []).append(float(ii))
        ring_K = {r: float(np.median(v)) if v else 1.0
                  for r, v in ring_to_intensities.items()}

    # ----- build alpha vector α(s) = I_obs(s) / K(ring of s) -----
    alpha = np.zeros(n_spots, dtype=np.float64)
    for i, s in enumerate(unique_spots):
        I = spot_intensity.get(int(s), 0.0)
        r = spot_ring.get(int(s), 0)
        K = ring_K.get(r, 1.0)
        alpha[i] = float(I) / max(K, 1e-12)

    # ----- assemble sparse M matrix; M[s, g] = 1 if grain g claims spot s -----
    rows = np.empty(len(sm_grain_id), dtype=np.int64)
    cols = np.empty(len(sm_grain_id), dtype=np.int64)
    keep = np.ones(len(sm_grain_id), dtype=bool)
    for k in range(len(sm_grain_id)):
        sid, gid = int(sm_spot_id[k]), int(sm_grain_id[k])
        gi = gid_to_row.get(gid)
        si = spot_to_row.get(sid)
        if gi is None or si is None:
            keep[k] = False; continue
        rows[k] = si
        cols[k] = gi
    rows = rows[keep]; cols = cols[keep]
    vals = np.ones(rows.shape[0], dtype=np.float64)
    M = csr_matrix((vals, (rows, cols)), shape=(n_spots, n_grains))

    # ----- per-grain shared-spot fraction (diagnostic) -----
    frac_shared = np.zeros(n_grains, dtype=np.float64)
    g_total = np.zeros(n_grains, dtype=np.int64)
    g_shared = np.zeros(n_grains, dtype=np.int64)
    for k in range(rows.shape[0]):
        g = int(cols[k]); s = int(rows[k])
        g_total[g] += 1
        if spot_claim_counts[s] > 1:
            g_shared[g] += 1
    valid_g = g_total > 0
    frac_shared[valid_g] = g_shared[valid_g] / g_total[valid_g]

    # ----- solve NNLS: α = M·V, V ≥ 0 -----
    # Rescale alpha to a well-conditioned magnitude
    alpha_scale = float(np.mean(alpha)) if np.any(alpha > 0) else 1.0
    alpha_n = alpha / max(alpha_scale, 1e-12)
    res = lsq_linear(
        M, alpha_n,
        bounds=(0.0, np.inf),
        method="trf",
        lsq_solver="lsmr",
        max_iter=max_iter,
        tol=tol,
        verbose=0,
    )
    V_raw = res.x * alpha_scale
    V_clipped = np.clip(V_raw, a_min=1e-12, a_max=None)
    R_raw = np.cbrt(3.0 * V_clipped / (4.0 * np.pi))

    # ----- per-grain uncertainty bands -----
    # For grains in the active set (V > tiny), the linearised covariance
    # of the unconstrained LS solution applies: Cov(V) ≈ σ_r² (M^T M)^-1.
    # σ_r is the RMS of the residual at MAP. Boundary grains (V = 0)
    # have ill-defined uncertainty under the active-set constraint;
    # report NaN for them.
    sigma_V = np.full(n_grains, np.nan, dtype=np.float64)
    sigma_R = np.full(n_grains, np.nan, dtype=np.float64)
    try:
        residual = M @ res.x - alpha_n
        if residual.size > n_grains:
            dof = residual.size - int((res.x > 1e-12).sum())
            sigma_r2 = float((residual ** 2).sum() / max(dof, 1))
        else:
            sigma_r2 = float((residual ** 2).mean()) if residual.size else 0.0
        active = res.x > 1e-12
        if active.any():
            M_active = M[:, active].toarray()  # (n_spots, n_active)
            MTM = M_active.T @ M_active
            # Regularise to avoid singular inverse on degenerate active sets
            MTM = MTM + 1e-9 * np.eye(MTM.shape[0])
            try:
                C = np.linalg.inv(MTM) * sigma_r2
                # diag → variance, mul by alpha_scale² to put back in V units
                sV_active = np.sqrt(np.diag(C)) * alpha_scale
                sigma_V[active] = sV_active
                # Convert σ_V → σ_R via dR/dV = (R/(3V)):
                # σ_R = σ_V · R / (3 V)
                sigma_R[active] = sV_active * R_raw[active] / (3.0 * V_clipped[active])
            except np.linalg.LinAlgError:
                pass
    except Exception:
        pass

    # rescale so mean(R_nnls) == mean(R_naive)
    mean_naive = float(np.mean(R_naive)) if len(R_naive) else 1.0
    mean_raw = float(np.mean(R_raw)) if len(R_raw) else 1.0
    rescale = mean_naive / max(mean_raw, 1e-12)
    R_nnls = R_raw * rescale
    # σ_R rescales with R
    sigma_R_um = sigma_R * rescale

    return NnlsVolumeResult(
        R_nnls=R_nnls,
        V_nnls_raw=V_raw,
        frac_spots_shared=frac_shared,
        n_spots_shared=n_shared,
        n_unique_spots=n_spots,
        rescale_factor=float(rescale),
        nnls_status=int(res.status),
        nnls_cost=float(res.cost),
        nnls_n_iter=int(res.nit),
        sigma_V_nnls_raw=sigma_V,
        sigma_R_um=sigma_R_um,
    )
