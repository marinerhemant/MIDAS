"""Uniaxial squared-modulus ODF: a 4-parameter per-voxel texture model.

Following Liebi et al. 2018 and Muerer et al. 2021 (IUCrJ 8, 747), the
orientation distribution about a single fibre axis is written as the **square of**
a truncated Legendre expansion,

.. math::

    I(\\hat q) = \\Big| \\sum_{l = 0,2,4,6} a_l\\, c_l\\, P_l(\\hat u \\cdot \\hat q) \\Big|^2,
    \\qquad c_l = \\sqrt{\\tfrac{2l+1}{4\\pi}}

Squaring buys two things that a general GSH solve does not have:

* **non-negativity for free** -- a squared real expansion cannot go negative, so
  there is no cone to project onto, no L1 penalty, and the odd-``l`` ghost
  subspace never arises, because the square generates only even orders;
* **four parameters per voxel** instead of the 23-61 a general
  :class:`~midas_dt.gsh.SymGSH` model needs. On a low-contrast scan that
  difference is decisive: a 61-parameter per-voxel fit will absorb background
  error as texture and report a map.

**This model has not yet produced a positive texture result on real data here.**
It was refuted on a DAC Ti scan: 0.17 % residual improvement over a uniform null
against a registered refute line of 5 %.

**What the positive control does and does not establish** (measured
2026-08-18 with ``scripts/odf_positive_control.py``; supersedes an earlier,
stronger claim):

* **Detection works robustly.** With a clean plant the global rung recovers
  planted texture at **23-34 %** residual improvement across peak/background
  0.02-0.5. Real Ti gave 0.17 %, a gap of more than 100x, so the *detection*
  half of the Ti refutation stands on solid ground.
* **Per-voxel resolution is weak and NON-MONOTONIC in SNR.** ``corr(S_recovered,
  S_planted_pole)`` runs 0.23-0.67, peaking near peak/bg = 0.1 and falling at
  *both* higher and lower contrast. At high SNR the fit chases an azimuthal shape
  four parameters cannot represent -- the plant is a discrete-crystallite fibre
  distribution, not a squared-modulus expansion -- and that model mismatch
  corrupts the per-voxel coefficients. Noise was regularising it.

So: a null on the **global** rung is interpretable as a statement about the
sample; a null on the **per-voxel** rung is not, and a sample-average bound
should be reported instead of a map. An earlier note recorded per-voxel recovery
at corr 0.60-0.75 holding to peak/bg 0.02; that came from a forward model which
subtracted an **exactly known** background and from an under-converged fit, and
it does not reproduce. See ``manuals/xrd-ct/LAB_NOTEBOOK.md`` §5g.

Run the control at your data's measured contrast, in **both** background modes,
before believing any texture map this produces.

**The ladder is the point.** :func:`fit_uniaxial_ladder` fits three nested models
-- uniform null, one *globally shared* texture, then per-voxel -- because a
per-voxel fit that fails is ambiguous on its own: it can mean "no texture" or
"texture present but not spatially resolvable". Adding three shared parameters
separates them. On the Ti scan the global model bought 0.11 % and per-voxel
0.17 %, which says the absence is not a resolution limit.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix

__all__ = [
    "LadderResult",
    "UniaxialODFModel",
    "explained_by_polynomial",
    "fibre_cos_theta",
    "fit_uniaxial_ladder",
    "hermans_parameter",
    "legendre_even",
    "normalisation_c_l",
    "uniaxial_design",
]

log = logging.getLogger(__name__)


def legendre_even(x: np.ndarray, n_l: int) -> np.ndarray:
    """``[P_0, P_2, P_4, P_6, ...]`` evaluated at ``x``, stacked on the last axis.

    Written out rather than recursed: ``n_l`` is at most a handful for this model,
    and explicit polynomials keep the design matrix readable in a debugger.
    """
    x = np.asarray(x, dtype=float)
    out = [np.ones_like(x)]
    if n_l > 1:
        out.append(0.5 * (3 * x ** 2 - 1))
    if n_l > 2:
        out.append((35 * x ** 4 - 30 * x ** 2 + 3) / 8.0)
    if n_l > 3:
        out.append((231 * x ** 6 - 315 * x ** 4 + 105 * x ** 2 - 5) / 16.0)
    if n_l > 4:
        raise NotImplementedError(
            f"n_l={n_l} needs P_8 and above; the squared-modulus model is not "
            "usually worth pushing past l=6 -- a low-contrast scan cannot "
            "support it, and a high-contrast one deserves the general "
            "midas_dt.gsh operator instead")
    return np.stack(out, axis=-1)


def normalisation_c_l(n_l: int) -> np.ndarray:
    """``c_l = sqrt((2l+1) / 4pi)`` for ``l = 0, 2, 4, ...``."""
    return np.sqrt((2 * np.arange(0, 2 * n_l, 2) + 1) / (4 * np.pi))


def fibre_cos_theta(two_theta_deg, eta_rad) -> np.ndarray:
    """``cos(Theta)`` between the scattering vector and the **rotation axis**.

    For the standard DT geometry -- beam along lab x, rotation about lab z,
    translation along lab y --

    .. math::

        \\hat q_{lab} = (-\\sin\\theta,\\; \\cos\\theta\\cos\\eta,\\;
                          \\cos\\theta\\sin\\eta), \\qquad
        \\hat n_s = R_z(\\omega)^{T} \\hat q_{lab}
        \\;\\Rightarrow\\;
        \\hat n_s \\cdot \\hat z = \\cos\\theta \\sin\\eta

    Returns ``(n_eta, n_ring)`` for 1-D inputs.

    Notes
    -----
    **The omega-independence is a real fact and a notorious trap.** Because
    ``n_s . z`` carries no omega, a fibre about the rotation axis produces an
    azimuthal pattern that is *static in omega*. It is therefore **wrong** to
    reason "this pattern does not vary with omega, so it must be instrumental" --
    that inference was made and had to be withdrawn. Static in omega is exactly
    what an axial fibre looks like.

    The converse is the model's real limit: if the sample's unique axis is **not**
    the rotation axis (a DAC loaded in radial geometry, say), the texture varies
    with omega and this model cannot fit it *by construction*. Establish the
    loading geometry before reading a null from this as "no texture".
    """
    th = np.radians(np.asarray(two_theta_deg, dtype=float)) / 2.0
    eta = np.asarray(eta_rad, dtype=float)
    return np.cos(th)[None, :] * np.sin(eta)[:, None]


def uniaxial_design(cos_theta: np.ndarray, n_l: int) -> np.ndarray:
    """Design matrix ``c_l P_l(cos Theta)``, shape ``(..., n_l)``.

    Multiply by coefficients and **square** to get intensity; this is the
    amplitude side, not the intensity side.
    """
    return legendre_even(cos_theta, n_l) * normalisation_c_l(n_l)


def hermans_parameter(coefs: np.ndarray) -> np.ndarray:
    """Hermans orientation parameter ``S = (3<cos^2 Theta> - 1) / 2`` per voxel.

    ``S`` is 0 for a random distribution, positive when weight concentrates along
    the fibre axis and negative when it concentrates perpendicular to it.

    Parameters
    ----------
    coefs
        ``(n_vox, n_l)`` amplitude coefficients.

    Notes
    -----
    **Exact, not approximate.** The integrand is a polynomial in ``mu`` of degree
    ``4 n_l - 2``, so Gauss-Legendre with enough nodes evaluates both integrals
    *exactly* -- a uniform ODF returns ``S = 0`` to 1e-15. A uniform grid with
    ``np.trapezoid`` leaves ~8e-6 of pure discretisation error instead, which is
    small but is the sort of floor that later gets mistaken for a weak signal.

    **S saturates near 0.61 with only a_2 free**, and then *decreases*: measured
    +0.59 at ``a_2 = 1``, a maximum of +0.611 near 1.35, then +0.586 at 2 and
    +0.447 at 5. ``a_2`` is therefore not a monotone texture-strength knob, and a
    sharper axial texture genuinely needs ``a_4`` and ``a_6``. A fit that stalls
    around ``S ~ 0.6`` may be at this ceiling rather than at the data's limit.

    Warning
    -------
    ``S`` of a **pole figure** is not ``S`` of the underlying crystal axis, and
    the two can differ in **sign**. Prism-plane normals in hcp are perpendicular
    to *c*, so a c-axis fibre shows up as a *negative* pole-figure ``S``. A
    positive control that scored recovered pole-figure ``S`` against a planted
    c-axis ``S`` read ``-0.75`` and looked like a total failure; it was correct.
    Compare like with like.
    """
    A = np.atleast_2d(np.asarray(coefs, dtype=float))
    n_l = A.shape[1]
    # exact for degree <= 2n-1; the integrand reaches 4*n_l - 2
    mu, w = np.polynomial.legendre.leggauss(2 * n_l + 4)
    Pm = legendre_even(mu, n_l) * normalisation_c_l(n_l)
    f = (Pm @ A.T) ** 2                                  # (n_nodes, n_vox)
    norm = w @ f
    cos2 = (w * mu ** 2) @ f / np.maximum(norm, 1e-30)
    return 0.5 * (3.0 * cos2 - 1.0)


def explained_by_polynomial(values: np.ndarray, xy: np.ndarray, *,
                            order: int = 3) -> float:
    """``r^2`` of a 2-D polynomial fit to a per-voxel field.

    The separability check. A recovered "per-voxel" field that a smooth
    low-order polynomial explains is an instrument, absorption or geometry
    signature wearing a map's clothes, not per-voxel physics. This has retracted
    a result in this project **three** times, which is why it is a function rather
    than a habit.

    Report it next to every per-voxel map; a value above ~0.5 should be read as
    "smooth field", not "texture".
    """
    v = np.asarray(values, dtype=float).ravel()
    p = np.asarray(xy, dtype=float)
    if p.shape != (v.size, 2):
        raise ValueError(f"xy must be (n_vox, 2) matching values, got {p.shape}")
    x, y = p[:, 0], p[:, 1]
    cols = [np.ones_like(x)]
    for total in range(1, int(order) + 1):
        for i in range(total + 1):
            cols.append(x ** (total - i) * y ** i)
    M = np.column_stack(cols)
    fit = M @ np.linalg.lstsq(M, v, rcond=None)[0]
    denom = float(((v - v.mean()) ** 2).sum())
    if denom <= 0:
        return 1.0
    return float(1.0 - ((v - fit) ** 2).sum() / denom)


class UniaxialODFModel:
    """Forward model and analytic sparse Jacobian for the squared-modulus ODF.

    Parameters
    ----------
    design
        ``(n_ang, n_l)`` amplitude design, from :func:`uniaxial_design` flattened
        over whatever angular channels the data has -- typically (eta x ring).
    rays
        ``(n_ray, n_vox)`` projection operator: entry ``(k, v)`` is voxel ``v``'s
        contribution to ray ``k``. Booleans are fine (equal weights); a proper
        path-length or Radon operator is better.
    good
        ``(n_ray, n_ang)`` boolean mask of which measurements exist and passed the
        gates. Residuals are formed over ``good`` only.
    data, weights
        ``(n_ray, n_ang)``; only entries under ``good`` are read.

    Notes
    -----
    The Jacobian is supplied **analytically and sparsely**. Each residual sees
    only the voxels on its own ray -- typically ~6 % of the voxel grid -- so a
    numerical Jacobian would cost one full forward evaluation per parameter
    (1396 of them in the Ti fit) per iteration, and would also be less accurate
    where the amplitude passes through zero.
    """

    def __init__(self, design: np.ndarray, rays: np.ndarray, good: np.ndarray,
                 data: np.ndarray, weights: np.ndarray):
        self.design = np.asarray(design, dtype=float)
        if self.design.ndim != 2:
            raise ValueError(f"design must be (n_ang, n_l), got {self.design.shape}")
        self.n_ang, self.n_l = self.design.shape
        self.rays = np.asarray(rays, dtype=float)
        if self.rays.ndim != 2:
            raise ValueError(f"rays must be (n_ray, n_vox), got {self.rays.shape}")
        self.n_ray, self.n_vox = self.rays.shape
        self.good = np.asarray(good, dtype=bool)
        if self.good.shape != (self.n_ray, self.n_ang):
            raise ValueError(
                f"good must be (n_ray, n_ang) = "
                f"({self.n_ray}, {self.n_ang}), got {self.good.shape}")
        self.data = np.asarray(data, dtype=float)[self.good]
        self.weights = np.asarray(weights, dtype=float)[self.good]
        gi = np.argwhere(self.good)
        self.row_ray = gi[:, 0]
        self.row_ang = gi[:, 1]
        self.n_resid = len(gi)
        # voxel support of each ray, precomputed once
        self._vox_of_ray = [np.flatnonzero(self.rays[k]) for k in range(self.n_ray)]

    @property
    def n_param(self) -> int:
        return self.n_vox * self.n_l

    def amplitude(self, coefs: np.ndarray) -> np.ndarray:
        """``(n_vox, n_ang)`` amplitude; square it for intensity."""
        return np.asarray(coefs, dtype=float).reshape(self.n_vox, self.n_l) @ self.design.T

    def predict(self, coefs: np.ndarray) -> np.ndarray:
        """Projected intensity at the ``good`` measurements, flattened."""
        amp = self.amplitude(coefs)
        return (self.rays @ (amp ** 2))[self.good]

    def residual(self, coefs: np.ndarray) -> np.ndarray:
        return (self.predict(coefs) - self.data) * self.weights

    def jacobian(self, coefs: np.ndarray):
        """Sparse ``d residual / d coefs``, as CSR."""
        amp = self.amplitude(coefs)
        rows, cols, vals = [], [], []
        for n in range(self.n_resid):
            vs = self._vox_of_ray[self.row_ray[n]]
            if not vs.size:
                continue
            ang = self.row_ang[n]
            w = self.rays[self.row_ray[n], vs] * self.weights[n]
            block = (2.0 * amp[vs, ang] * w)[:, None] * self.design[ang][None, :]
            rows.append(np.full(vs.size * self.n_l, n))
            cols.append((vs[:, None] * self.n_l
                         + np.arange(self.n_l)[None, :]).ravel())
            vals.append(block.ravel())
        if not vals:                                       # pragma: no cover
            return coo_matrix((self.n_resid, self.n_param)).tocsr()
        return coo_matrix((np.concatenate(vals),
                           (np.concatenate(rows), np.concatenate(cols))),
                          shape=(self.n_resid, self.n_param)).tocsr()


@dataclass
class LadderResult:
    """Three nested fits, and what their ordering means."""

    chi2_null: float
    chi2_global: float
    chi2_pervoxel: float
    coefs: np.ndarray               # (n_vox, n_l) from the per-voxel fit
    global_coefs: np.ndarray        # the shared a_2..a_L
    hermans_S: np.ndarray           # (n_vox,)
    n_param: dict = field(default_factory=dict)
    converged: bool = True
    n_fev: int = 0

    @staticmethod
    def _improvement(chi2: float, reference: float) -> float:
        """Percent reduction, with an exactly-fitting reference handled.

        A noiseless synthetic can drive the reference chi2 to 0, and then there is
        nothing left to improve on -- so the honest answer is 0 %, not an
        exception and not 100 %. Real data never hits this; synthetic controls do,
        and a control that crashes gets deleted rather than fixed.
        """
        if not np.isfinite(reference) or reference <= 0.0:
            return 0.0
        return 100.0 * (1.0 - chi2 / reference)

    @property
    def improvement_pct(self) -> float:
        """Per-voxel model against the uniform null, in percent."""
        return self._improvement(self.chi2_pervoxel, self.chi2_null)

    @property
    def global_improvement_pct(self) -> float:
        """A single sample-average texture against the uniform null."""
        return self._improvement(self.chi2_global, self.chi2_null)

    @property
    def pervoxel_over_global_pct(self) -> float:
        """What per-voxel freedom buys over one shared texture."""
        return self._improvement(self.chi2_pervoxel, self.chi2_global)

    def verdict(self, xy: np.ndarray | None = None, *, refute_pct: float = 5.0,
                confirm_pct: float = 20.0, smooth_r2: float = 0.5) -> str:
        """Read the ladder against fixed thresholds.

        Defaults are the ones registered for the DAC Ti analysis. Set your own
        **before** looking at the numbers, not after.
        """
        impr = self.improvement_pct
        if impr < refute_pct:
            return (f"REFUTED: {impr:.2f}% improvement over the uniform null "
                    f"(< {refute_pct}%) -- the data do not support a texture model")
        if xy is not None:
            r2 = explained_by_polynomial(self.hermans_S, xy)
            if r2 > smooth_r2:
                return (f"REFUTED: S is {r2*100:.0f}% explained by a smooth "
                        f"polynomial (> {smooth_r2*100:.0f}%) -- an instrumental "
                        "or absorption field, not per-voxel texture")
        if impr >= confirm_pct:
            return (f"CONFIRM candidate: {impr:.2f}% improvement -- pending "
                    "per-ring agreement and a positive control at this contrast")
        return (f"INCONCLUSIVE: {impr:.2f}% is between {refute_pct}% and "
                f"{confirm_pct}%; report as the achievable limit for this data")


def fit_uniaxial_ladder(model: UniaxialODFModel, *, max_nfev: int = 200,
                        verbose: int = 0) -> LadderResult:
    """Fit uniform null, then one shared texture, then per-voxel.

    Each rung starts from the one below it, so the ordering
    ``chi2_null >= chi2_global >= chi2_pervoxel`` holds by nesting and a violation
    means the optimiser stalled rather than that the model got worse.

    Returns
    -------
    LadderResult
        Read it with :meth:`LadderResult.verdict` against thresholds fixed in
        advance.
    """
    n_vox, n_l = model.n_vox, model.n_l
    c_l = normalisation_c_l(n_l)

    # ---- rung 1: uniform ODF, a_0 free per voxel ---------------------------
    def resid_null(p0):
        par = np.zeros((n_vox, n_l))
        par[:, 0] = p0
        return model.residual(par.ravel())

    def jac_null(p0):
        par = np.zeros((n_vox, n_l))
        par[:, 0] = p0
        return np.asarray(model.jacobian(par.ravel())[:, ::n_l].todense())

    scale = np.sqrt(max(float(np.nanmedian(model.data)), 1e-12))
    a0 = np.full(n_vox, scale / max(np.sqrt(model.rays.sum(axis=1).mean()), 1.0)
                 / c_l[0])
    r_null = least_squares(resid_null, a0, jac=jac_null, method="trf",
                           max_nfev=40, verbose=verbose)
    chi_null = float(np.sum(r_null.fun ** 2))
    log.info("null (uniform, %d par): chi2 %.4e", n_vox, chi_null)

    # ---- rung 2: ONE shared texture, a_0 still free per voxel --------------
    # Three extra parameters instead of n_vox*(n_l-1). This rung is what makes a
    # negative result interpretable: it separates "no texture" from "texture
    # present but not spatially resolvable".
    def unpack_global(p):
        par = np.zeros((n_vox, n_l))
        par[:, 0] = p[:n_vox]
        par[:, 1:] = p[n_vox:][None, :]
        return par

    def resid_glob(p):
        return model.residual(unpack_global(p).ravel())

    def jac_glob(p):
        Jd = np.asarray(model.jacobian(unpack_global(p).ravel()).todense())
        out = np.zeros((Jd.shape[0], n_vox + n_l - 1))
        out[:, :n_vox] = Jd[:, ::n_l]
        for l in range(1, n_l):
            out[:, n_vox + l - 1] = Jd[:, l::n_l].sum(axis=1)
        return out

    r_glob = least_squares(resid_glob, np.concatenate([r_null.x, np.zeros(n_l - 1)]),
                           jac=jac_glob, method="trf", max_nfev=60, verbose=verbose)
    chi_glob = float(np.sum(r_glob.fun ** 2))
    log.info("global texture (%d par): chi2 %.4e  (%.2f%% over null)",
             n_vox + n_l - 1, chi_glob,
             LadderResult._improvement(chi_glob, chi_null))

    # ---- rung 3: per-voxel ------------------------------------------------
    r_fit = least_squares(model.residual, unpack_global(r_glob.x).ravel(),
                          jac=model.jacobian, method="trf", tr_solver="lsmr",
                          max_nfev=max_nfev, verbose=verbose)
    chi_fit = float(np.sum(r_fit.fun ** 2))
    coefs = r_fit.x.reshape(n_vox, n_l)
    log.info("per-voxel (%d par): chi2 %.4e  (%.2f%% over null, %.2f%% over global)",
             model.n_param, chi_fit,
             LadderResult._improvement(chi_fit, chi_null),
             LadderResult._improvement(chi_fit, chi_glob))

    if not (chi_null >= chi_glob >= chi_fit):
        log.warning("ladder is not monotone (null %.4e, global %.4e, per-voxel "
                    "%.4e): the models are nested, so this means an optimiser "
                    "stall, not a worse model", chi_null, chi_glob, chi_fit)

    return LadderResult(
        chi2_null=chi_null,
        chi2_global=chi_glob,
        chi2_pervoxel=chi_fit,
        coefs=coefs,
        global_coefs=np.asarray(r_glob.x[n_vox:]),
        hermans_S=hermans_parameter(coefs),
        n_param={"null": n_vox, "global": n_vox + n_l - 1,
                 "pervoxel": model.n_param},
        converged=bool(r_fit.status > 0),
        n_fev=int(r_fit.nfev),
    )
