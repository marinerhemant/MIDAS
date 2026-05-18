"""BIC-driven distortion-basis selection — automated B2 ladder.

For a single calibrant image, ask which subset of v2's 15-coefficient
distortion basis the data actually supports.  v2 names harmonics by
η-fold (1 .. 6) plus an isotropic radial polynomial (iso_R2/4/6).  The
ladder progressively drops the highest-fold harmonics first, recalibrates,
and computes BIC = N · log(SSE/N) + k · log(N).  Lower BIC = better.

Usage::

    from midas_calibrate_v2.pipelines.bic_search import select_basis_bic
    best, ladder = select_basis_bic(v1_params, image, panel_layout=layout)
    print(f"BIC-optimal drop set: {best.drop_folds}")
    print(f"BIC-optimal strain: {best.strain_uE:.2f} μϵ")

The B2 results showed:
  - Varex Ceria → full 15-coef basis is BIC-optimal.
  - GE offset   → drop fold-6, BIC-optimal at 14 coeffs.

Both are recovered automatically by this routine.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import math
import numpy as np
import torch

from midas_calibrate.params import CalibrationParams as V1Params

from ..forward.distortion import P_COEF_NAMES
from ..forward.panels import PanelLayout
from ..parameters.spec import CalibrationSpec
from ..compat.from_v1 import spec_from_v1_params
from .single_pv import autocalibrate_pv, PVCalibrationResult


@dataclass
class BasisFit:
    drop_folds: Tuple[int, ...]           # which η-folds were frozen at zero
    n_refined: int                          # n free parameters
    strain_uE: float                        # final mean strain
    sse: float                              # sum of squared residuals (kept set)
    n_data: int                             # n_kept fits
    bic: float                              # BIC = n*log(sse/n) + k*log(n)
    spec: Optional[CalibrationSpec] = None
    result: Optional[PVCalibrationResult] = None


def _pcoef_names_for_fold(fold: int) -> List[str]:
    """Return the v2 parameter names corresponding to η-fold k (or
    isotropic radial when fold=0).  Used to freeze them in the spec."""
    if fold == 0:
        return ["iso_R2", "iso_R4", "iso_R6"]
    if 1 <= fold <= 6:
        return [f"a{fold}", f"phi{fold}"]
    return []


def _spec_with_dropped_folds(
    base_spec: CalibrationSpec,
    drop_folds: Sequence[int],
) -> CalibrationSpec:
    """Return a deep copy of ``base_spec`` with all parameters belonging
    to ``drop_folds`` pinned at zero (refine=False, init=0.0)."""
    import copy
    spec = copy.deepcopy(base_spec)
    for fold in drop_folds:
        for nm in _pcoef_names_for_fold(fold):
            if nm in spec.parameters:
                p = spec.parameters[nm]
                p.refined = False
                p.init = 0.0
    return spec


def _compute_sse_n(history) -> Tuple[float, float, int]:
    """Best-effort SSE / N from the strain history."""
    if not history:
        return float("nan"), float("nan"), 0
    final = history[-1]
    n_kept = int(getattr(final, "n_fitted", 0))
    cost = float(getattr(final, "cost", float("nan")))
    # cost = 0.5 * Σ r² for LM in v2.
    sse = 2.0 * cost if cost == cost else float("nan")
    strain_uE = float(getattr(final, "mean_strain_uE", float("nan")))
    return sse, strain_uE, n_kept


def _bic(sse: float, k: int, n: int) -> float:
    if n <= 0 or not (sse == sse) or sse <= 0:
        return float("nan")
    return n * math.log(sse / n) + k * math.log(n)


def select_basis_bic(
    v1_params: V1Params,
    image: np.ndarray,
    *,
    dark: Optional[np.ndarray] = None,
    base_spec: Optional[CalibrationSpec] = None,
    panel_layout: Optional[PanelLayout] = None,
    drop_ladder: Optional[List[Tuple[int, ...]]] = None,
    verbose: bool = True,
    **autocalibrate_kwargs,
) -> Tuple[BasisFit, List[BasisFit]]:
    """Run the BIC ladder and return ``(best, all_results)``.

    Parameters
    ----------
    drop_ladder : optional list of fold tuples
        Each tuple lists the η-folds to freeze at zero for that ladder
        step.  Default ladder progressively drops the highest folds:
        ``[(), (6,), (5, 6), (4, 5, 6), (3, 4, 5, 6), (1, 3, 4, 5, 6),
        (0,)]`` — matches the B2 paper-section ladder.

    The default ladder is intentionally short — each rung re-runs a full
    calibration, which on Pilatus takes ~17 min per rung.  Callers may
    pass a custom ladder for cheaper sweeps.

    Returns
    -------
    best : :class:`BasisFit`
        The lowest-BIC basis.
    ladder_results : list of :class:`BasisFit`
        Every ladder step in evaluation order.
    """
    if drop_ladder is None:
        drop_ladder = [
            (),
            (6,),
            (5, 6),
            (4, 5, 6),
            (3, 4, 5, 6),
            (1, 3, 4, 5, 6),
            (0,),
        ]
    if base_spec is None:
        base_spec = spec_from_v1_params(v1_params)

    results: List[BasisFit] = []
    for drop in drop_ladder:
        if verbose:
            tag = "()" if not drop else f"({', '.join(str(d) for d in drop)},)"
            print(f"[bic] drop folds {tag}…", flush=True)
        spec_drop = _spec_with_dropped_folds(base_spec, drop)
        res = autocalibrate_pv(
            v1_params, image, dark=dark, spec=spec_drop,
            panel_layout=panel_layout, verbose=False,
            **autocalibrate_kwargs,
        )
        sse, strain, n = _compute_sse_n(res.history)
        k = len(spec_drop.refined_names())
        bic = _bic(sse, k, n)
        bf = BasisFit(
            drop_folds=tuple(drop),
            n_refined=k,
            strain_uE=strain,
            sse=sse,
            n_data=n,
            bic=bic,
            spec=spec_drop,
            result=res,
        )
        results.append(bf)
        if verbose:
            print(f"      n_refined={k:3d}  strain={strain:7.2f} μϵ  "
                  f"N={n:5d}  BIC={bic:.4e}", flush=True)

    finite = [r for r in results if r.bic == r.bic]    # filter NaN
    if not finite:
        raise RuntimeError("BIC ladder produced no finite results")
    best = min(finite, key=lambda r: r.bic)
    if verbose:
        print(f"\n[bic] BEST: drop={best.drop_folds!r}  "
              f"n={best.n_refined}  strain={best.strain_uE:.2f} μϵ  "
              f"BIC={best.bic:.4e}", flush=True)
    return best, results


__all__ = [
    "BasisFit",
    "select_basis_bic",
]
