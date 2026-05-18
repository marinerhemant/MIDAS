"""One-shot Fisher pruning of the distortion basis.

The B2 BIC ladder picks a basis by re-running the calibration once per
candidate subset.  That's rigorous but expensive (89 min on Pilatus
for 7 rungs).

For most cases a much faster alternative works: run **one** full
calibration, look at each distortion coefficient's z-score
``z_k = |MAP_k| / σ_k`` from the Laplace covariance, and drop the
coefficients with z below a threshold.  Coefficients with z < 1 are
statistically indistinguishable from zero — the data does not need
them.

Usage::

    from midas_calibrate_v2.pipelines.fisher_prune import (
        auto_select_basis,
    )
    res, prune = auto_select_basis(v1_params, image,
                                     panel_layout=layout,
                                     z_threshold=1.0)
    print(prune)         # human-readable per-coef report
    print(prune.kept_names)
    print(prune.dropped_names)
    # res is the *re-fit* result with only the kept coefficients refined.

The Laplace approximation is only safe on amplitude / isotropic
parameters; the harmonic phase parameters (``phi3``, ``phi5``,
``phi6``) frequently have non-Gaussian posteriors per the NUTS
comparison.  We therefore prune by amplitude z-score only, dropping
the matching phase whenever its amplitude is dropped.  ``iso_R*``
parameters are also pruned by individual z-score.

Wall time on Mac M1 CPU:
  - Varex Ceria : ~2.5 min  (initial fit 80s + Fisher 30s + re-fit 60s)
  - Pilatus     : ~5 min   (initial 17m? — see below; Fisher ~1 min;
                              re-fit cost dominates)

If the user already has a fitted v2 result handy, pass it via
``initial_result`` and skip the initial fit.  Otherwise the routine
runs ``autocalibrate_pv`` once.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import copy
import math
import numpy as np
import torch

from midas_calibrate.params import CalibrationParams as V1Params

from ..compat.from_v1 import spec_from_v1_params
from ..forward.distortion import P_COEF_NAMES, AMP_NAMES, PHASE_NAMES, ISO_NAMES
from ..forward.panels import PanelLayout
from ..inference.laplace import fisher_at_map, LaplaceResult
from ..loss.pseudo_strain import pseudo_strain_residual
from ..parameters.spec import CalibrationSpec
from .single_pv import autocalibrate_pv, PVCalibrationResult


@dataclass
class PruneReport:
    """Per-coefficient prune outcome."""
    z_scores: Dict[str, float] = field(default_factory=dict)
    map_values: Dict[str, float] = field(default_factory=dict)
    sigmas: Dict[str, float] = field(default_factory=dict)
    z_threshold: float = 1.0
    kept_names: List[str] = field(default_factory=list)
    dropped_names: List[str] = field(default_factory=list)
    initial_strain_uE: float = float("nan")
    pruned_strain_uE: float = float("nan")
    initial_bic: float = float("nan")
    pruned_bic: float = float("nan")

    def __str__(self) -> str:
        lines = [f"Fisher prune report (z_threshold = {self.z_threshold:.2f})",
                 f"  initial: strain={self.initial_strain_uE:.2f} μϵ  "
                 f"BIC={self.initial_bic:.4e}",
                 f"  pruned:  strain={self.pruned_strain_uE:.2f} μϵ  "
                 f"BIC={self.pruned_bic:.4e}",
                 f"  ΔBIC = {self.pruned_bic - self.initial_bic:+.1f}  "
                 f"({'BETTER' if self.pruned_bic < self.initial_bic else 'WORSE'})",
                 f"  Kept ({len(self.kept_names)}): "
                 f"{', '.join(self.kept_names)}",
                 f"  Dropped ({len(self.dropped_names)}): "
                 f"{', '.join(self.dropped_names) if self.dropped_names else '(none)'}",
                 "  per-coef z-scores:"]
        # Sort by z-score ascending
        rows = sorted(self.z_scores.items(), key=lambda kv: kv[1])
        for nm, z in rows:
            tag = "DROP" if nm in self.dropped_names else "keep"
            v = self.map_values.get(nm, float("nan"))
            s = self.sigmas.get(nm, float("nan"))
            lines.append(f"    {nm:8s}  MAP={v:+.4e}  σ={s:.4e}  "
                          f"z={z:6.2f}  [{tag}]")
        return "\n".join(lines)


def _per_coef_sigma(lap: LaplaceResult, coef_names: List[str]
                    ) -> Dict[str, float]:
    """Return ``{name: σ}`` from a LaplaceResult for the listed coef
    names.  σ is the per-dim std (square root of the marginal variance)."""
    out: Dict[str, float] = {}
    sigma = lap.sigma_per_dim.detach().cpu().numpy()
    cur = 0
    for nm, sz in zip(lap.refined_names, lap.refined_sizes):
        if nm in coef_names and sz == 1:
            out[nm] = float(sigma[cur])
        cur += sz
    return out


def _bic(sse: float, k: int, n: int) -> float:
    if n <= 0 or not (sse == sse) or sse <= 0:
        return float("nan")
    return n * math.log(sse / n) + k * math.log(n)


def _empirical_sigma_r(fits, unpacked, panel_layout: Optional[PanelLayout]
                        ) -> float:
    with torch.no_grad():
        r = pseudo_strain_residual(
            fits.Y_pix, fits.Z_pix, fits.ring_two_theta_deg, unpacked,
            rho_d=fits.rho_d,
            panel_layout=panel_layout,
            panel_idx=getattr(fits, "panel_idx", None),
        )
    sse = float((r * r).sum())
    n = int(r.numel())
    return math.sqrt(sse / max(n, 1)), sse, n


def auto_select_basis(
    v1_params: V1Params,
    image: np.ndarray,
    *,
    dark: Optional[np.ndarray] = None,
    panel_layout: Optional[PanelLayout] = None,
    initial_result: Optional[PVCalibrationResult] = None,
    z_threshold: float = 1.0,
    drop_unrefined_phases: bool = True,    # drop phi_k whenever a_k drops
    require_bic_improvement: bool = False,  # if True, revert to initial when
                                              # ΔBIC > 0
    verbose: bool = True,
    **autocalibrate_kwargs,
) -> Tuple[PVCalibrationResult, PruneReport]:
    """Pick which distortion coefficients to refine, automatically.

    Stage 1: full calibration (or use ``initial_result`` if provided).
    Stage 2: Fisher J'J at MAP → per-coefficient σ.
    Stage 3: drop every coefficient with ``|MAP|/σ < z_threshold``.
             For each dropped amplitude ``a_k``, also drop ``phi_k``
             if ``drop_unrefined_phases``.
    Stage 4: re-fit with the pruned spec.
    Stage 5: compare BIC; if pruned BIC is worse and
             ``require_bic_improvement`` is True, revert to initial.

    Returns
    -------
    (final_result, report) : tuple
        ``final_result`` is the re-fit calibration; ``report`` describes
        which coefficients were kept / dropped and the BIC delta.
    """
    # ---------- Stage 1: initial fit.
    if initial_result is None:
        if verbose:
            print("[fisher_prune] stage 1: initial calibration with full basis…",
                  flush=True)
        initial_result = autocalibrate_pv(
            v1_params, image, dark=dark, panel_layout=panel_layout,
            verbose=False, **autocalibrate_kwargs,
        )
    res0 = initial_result
    fits = res0.fits_final
    map_unp0 = res0.unpacked
    if fits is None:
        raise RuntimeError("auto_select_basis: initial fit produced no fits")

    sigma_r_emp, sse0, n0 = _empirical_sigma_r(fits, map_unp0, panel_layout)
    strain0 = float(res0.history[-1].mean_strain_uE)
    if verbose:
        print(f"           strain={strain0:.2f} μϵ  σ_r_emp={sigma_r_emp:.3e}  "
              f"N={n0}", flush=True)

    # ---------- Stage 2: Fisher at MAP for the distortion coefficients only.
    if verbose:
        print("[fisher_prune] stage 2: Fisher J'J at MAP…", flush=True)

    # Use the spec from the initial fit; that's what's already refined.
    spec0 = res0.spec

    def residual_fn(unp):
        return pseudo_strain_residual(
            fits.Y_pix, fits.Z_pix, fits.ring_two_theta_deg, unp,
            rho_d=fits.rho_d,
            panel_layout=panel_layout,
            panel_idx=getattr(fits, "panel_idx", None),
        )

    lap = fisher_at_map(spec0, residual_fn, map_unp0,
                         sigma_r=sigma_r_emp, ridge=1e-9,
                         dtype=torch.float64, device="cpu")
    sigma_for = _per_coef_sigma(lap, P_COEF_NAMES)

    # ---------- Stage 3: prune by z-score.
    z_scores: Dict[str, float] = {}
    map_vals: Dict[str, float] = {}
    sigmas_used: Dict[str, float] = {}
    dropped: List[str] = []
    kept: List[str] = []

    # Iso radial: prune individually.
    for nm in ISO_NAMES:
        if nm not in spec0.parameters:
            continue
        if not spec0.parameters[nm].refined:
            continue
        v = float(map_unp0[nm].detach())
        s = sigma_for.get(nm, float("inf"))
        z = abs(v) / max(s, 1e-30)
        z_scores[nm] = z
        map_vals[nm] = v
        sigmas_used[nm] = s
        if z < z_threshold:
            dropped.append(nm)
        else:
            kept.append(nm)

    # Harmonic amplitude / phase pairs: prune by amplitude z-score.
    for k, amp in enumerate(AMP_NAMES, start=1):
        phi = f"phi{k}"
        if amp not in spec0.parameters:
            continue
        if not spec0.parameters[amp].refined:
            continue
        v = float(map_unp0[amp].detach())
        s = sigma_for.get(amp, float("inf"))
        z = abs(v) / max(s, 1e-30)
        z_scores[amp] = z
        map_vals[amp] = v
        sigmas_used[amp] = s
        # Phase z-score for the report (not used to decide).
        if phi in map_unp0:
            v_phi = float(map_unp0[phi].detach())
            s_phi = sigma_for.get(phi, float("inf"))
            z_phi = abs(v_phi) / max(s_phi, 1e-30)
            z_scores[phi] = z_phi
            map_vals[phi] = v_phi
            sigmas_used[phi] = s_phi
        if z < z_threshold:
            dropped.append(amp)
            if drop_unrefined_phases and phi in spec0.parameters:
                dropped.append(phi)
        else:
            kept.append(amp)
            if phi in spec0.parameters:
                kept.append(phi)

    if verbose:
        print(f"[fisher_prune] z_threshold={z_threshold}: keeping "
              f"{len(kept)} coefs, dropping {len(dropped)}", flush=True)

    # ---------- Stage 4: re-fit with the pruned spec.
    spec_pruned = copy.deepcopy(spec0)
    for nm in dropped:
        if nm in spec_pruned.parameters:
            p = spec_pruned.parameters[nm]
            p.refined = False
            p.init = 0.0

    if not dropped:
        if verbose:
            print("[fisher_prune] no coefficients to drop — returning initial",
                  flush=True)
        bic0 = _bic(sse0, len(spec0.refined_names()), n0)
        report = PruneReport(
            z_scores=z_scores, map_values=map_vals, sigmas=sigmas_used,
            z_threshold=z_threshold,
            kept_names=kept, dropped_names=dropped,
            initial_strain_uE=strain0, pruned_strain_uE=strain0,
            initial_bic=bic0, pruned_bic=bic0,
        )
        return res0, report

    if verbose:
        print(f"[fisher_prune] stage 4: re-fitting with pruned basis "
              f"({len(spec_pruned.refined_names())} refined)", flush=True)

    res1 = autocalibrate_pv(
        v1_params, image, dark=dark, spec=spec_pruned,
        panel_layout=panel_layout, verbose=False, **autocalibrate_kwargs,
    )
    fits1 = res1.fits_final
    map_unp1 = res1.unpacked
    sigma_r_emp1, sse1, n1 = _empirical_sigma_r(fits1, map_unp1, panel_layout)
    strain1 = float(res1.history[-1].mean_strain_uE)
    bic0 = _bic(sse0, len(spec0.refined_names()), n0)
    bic1 = _bic(sse1, len(spec_pruned.refined_names()), n1)

    if verbose:
        print(f"           strain={strain1:.2f} μϵ  BIC={bic1:.4e}  "
              f"(initial BIC={bic0:.4e})", flush=True)

    if require_bic_improvement and bic1 > bic0:
        if verbose:
            print(f"[fisher_prune] pruned BIC worse → reverting to initial",
                  flush=True)
        report = PruneReport(
            z_scores=z_scores, map_values=map_vals, sigmas=sigmas_used,
            z_threshold=z_threshold,
            kept_names=kept + dropped,    # nothing dropped in the end
            dropped_names=[],
            initial_strain_uE=strain0, pruned_strain_uE=strain0,
            initial_bic=bic0, pruned_bic=bic0,
        )
        return res0, report

    report = PruneReport(
        z_scores=z_scores, map_values=map_vals, sigmas=sigmas_used,
        z_threshold=z_threshold,
        kept_names=kept, dropped_names=dropped,
        initial_strain_uE=strain0, pruned_strain_uE=strain1,
        initial_bic=bic0, pruned_bic=bic1,
    )
    return res1, report


__all__ = ["auto_select_basis", "PruneReport"]
