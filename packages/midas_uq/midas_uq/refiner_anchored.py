"""Refiner-anchored UQ — consume the refiner's OWN predicted positions.

The fixed-assignment module (:mod:`midas_uq.fixed_assignment`) does the right
math but still calls :class:`~midas_diffract.HEDMForwardModel` to predict
spots at each candidate state. That's a problem when the model and the legacy
refiner are NOT byte-identical in their forward operator (different distortion
application order, different ω-binning, different tilt convention, etc.) —
predictions and observations end up systematically offset, the
nearest-neighbour cap rejects half the spots, and the resulting UQ summary
is dominated by the model-vs-refiner mismatch, not by the grain's actual
reproducibility.

The refiner already writes its own predicted ``(Y, Z, ω)`` per matched spot
to ``FitBest.bin`` (cols 7-9 = pred Y_um/Z_um/ω_deg, col 20 = ‖dY,dZ‖, col 21
= |dω|). This module reads those predictions directly and does UQ over the
PRE-COMPUTED residual distribution — no re-prediction, no convention drift,
no max-match cap.

Two diagnostics for each grain:

* :func:`per_grain_residuals`   — observed-vs-refiner-predicted residuals,
                                   typed and ready to histogram.
* :func:`bootstrap_uq`          — resample the per-spot residuals to get a
                                   spot-resampling uncertainty estimate.
* :func:`trust_score_anchored`  — convenience: one TrustScoreAnchored per
                                   grain.

This is the recommended path for population-scale grain trust scoring on data
that's already been refined by a stable refiner (legacy FitPosOrStrainsOMP,
c-omp FitUnified). For UQ of an in-flight refinement (where the optimizer
hasn't converged yet), use the fixed_assignment path with the forward model.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class PerGrainResiduals:
    """Refiner-emitted per-spot residuals for one grain.

    All arrays are ``(n_spots,)`` over the grain's matched spots.

    The signed residuals (``dy_um``, ``dz_um``, ``dome_deg``) come from
    ``obs - pred`` using FitBest cols (1,2,3) and (7,8,9). The L2 magnitudes
    ``diff_len_um`` and ``diff_ome_deg`` are the refiner's own col-20 / col-21
    values (already absolute).
    """
    n_spots: int
    spot_id: np.ndarray
    dy_um: np.ndarray             # signed: obs_y - pred_y (µm)
    dz_um: np.ndarray             # signed: obs_z - pred_z (µm)
    dome_deg: np.ndarray          # signed: obs_ω - pred_ω (degrees)
    diff_len_um: np.ndarray       # refiner-emitted ‖(dy,dz)‖ (µm)
    diff_ome_deg: np.ndarray      # refiner-emitted |dω|     (deg)
    min_ia_deg: np.ndarray        # refiner-emitted internal-angle metric (deg)

    # ----- aggregate summaries (the standard knobs a trust filter uses) -----
    @property
    def pos_med_um(self) -> float:
        return float(np.median(self.diff_len_um)) if self.n_spots else float("nan")

    @property
    def pos_p95_um(self) -> float:
        return float(np.percentile(self.diff_len_um, 95)) if self.n_spots else float("nan")

    @property
    def ome_med_deg(self) -> float:
        return float(np.median(self.diff_ome_deg)) if self.n_spots else float("nan")

    @property
    def ome_p95_deg(self) -> float:
        return float(np.percentile(self.diff_ome_deg, 95)) if self.n_spots else float("nan")

    @property
    def angle_med_deg(self) -> float:
        return float(np.median(self.min_ia_deg)) if self.n_spots else float("nan")

    @property
    def angle_p95_deg(self) -> float:
        return float(np.percentile(self.min_ia_deg, 95)) if self.n_spots else float("nan")


@dataclass
class BootstrapUQ:
    """Bootstrap-resampled UQ over the per-spot residual distribution.

    Per-spot residuals are treated as an empirical distribution; we resample
    with replacement ``n_boot`` times to get the variance of the per-grain
    aggregates (median/p95). This characterises how much the grain's trust
    metrics would move under a different (but statistically similar) set of
    matched spots — without re-running an optimiser. It's the cheapest
    label-free UQ axis you can give a user.
    """
    n_spots: int
    n_boot: int
    pos_med_um_p5_p95: tuple              # 5/95th percentile of bootstrap medians
    pos_p95_um_p5_p95: tuple
    ome_med_deg_p5_p95: tuple
    ome_p95_deg_p5_p95: tuple
    pos_med_std_um: float
    ome_med_std_deg: float


def per_grain_residuals(fitbest_view) -> PerGrainResiduals:
    """Wrap a :class:`midas_fit_grain.FitBestGrainView` into a typed
    :class:`PerGrainResiduals` record.

    No refit, no re-prediction — pure read from the refiner's emitted data.
    """
    return PerGrainResiduals(
        n_spots=fitbest_view.n_spots,
        spot_id=fitbest_view.spot_id,
        dy_um=fitbest_view.dy_um,
        dz_um=fitbest_view.dz_um,
        dome_deg=fitbest_view.dome_deg,
        diff_len_um=fitbest_view.diff_len_um,
        diff_ome_deg=fitbest_view.diff_ome_deg,
        min_ia_deg=fitbest_view.min_ia_deg,
    )


def bootstrap_uq(
    residuals: PerGrainResiduals,
    *,
    n_boot: int = 500,
    seed: int = 0,
) -> BootstrapUQ:
    """Bootstrap the per-spot residual distribution to characterize how stable
    the per-grain aggregates (median, p95) are under spot resampling.

    Each bootstrap iteration samples ``n_spots`` spots with replacement from
    the grain's matched-spot pool, computes the aggregate (median, p95) on
    that resample, and we report the 5/95th percentile of those aggregates.
    Tight bounds → trustworthy; wide bounds → the aggregate is driven by a
    handful of dominant spots.
    """
    n = residuals.n_spots
    if n < 5:
        nan = (float("nan"), float("nan"))
        return BootstrapUQ(
            n_spots=n, n_boot=n_boot,
            pos_med_um_p5_p95=nan, pos_p95_um_p5_p95=nan,
            ome_med_deg_p5_p95=nan, ome_p95_deg_p5_p95=nan,
            pos_med_std_um=float("nan"), ome_med_std_deg=float("nan"))
    rng = np.random.default_rng(seed)
    pos = residuals.diff_len_um
    ome = residuals.diff_ome_deg
    pm = np.zeros(n_boot); pp = np.zeros(n_boot)
    om = np.zeros(n_boot); op = np.zeros(n_boot)
    for k in range(n_boot):
        idx = rng.integers(0, n, size=n)
        pm[k] = np.median(pos[idx])
        pp[k] = np.percentile(pos[idx], 95)
        om[k] = np.median(ome[idx])
        op[k] = np.percentile(ome[idx], 95)
    return BootstrapUQ(
        n_spots=n, n_boot=n_boot,
        pos_med_um_p5_p95=(float(np.percentile(pm, 5)), float(np.percentile(pm, 95))),
        pos_p95_um_p5_p95=(float(np.percentile(pp, 5)), float(np.percentile(pp, 95))),
        ome_med_deg_p5_p95=(float(np.percentile(om, 5)), float(np.percentile(om, 95))),
        ome_p95_deg_p5_p95=(float(np.percentile(op, 5)), float(np.percentile(op, 95))),
        pos_med_std_um=float(np.std(pm)),
        ome_med_std_deg=float(np.std(om)),
    )


@dataclass
class TrustScoreAnchored:
    """Per-grain trust summary from refiner-emitted predictions.

    A lightweight, refiner-anchored alternative to
    :class:`midas_uq.TrustScore` (which calls the forward model).  Combines:

    * Per-grain median + p95 of position residuals (µm)
    * Per-grain median + p95 of omega residuals (deg)
    * Bootstrap stability of the medians

    Lower is better across the board.  Filter suggestion (Ni FF data):
    ``pos_med_um < 200, ome_med_deg < 0.1, pos_med_std_um < 30``.
    """
    n_spots: int
    pos_med_um: float
    pos_p95_um: float
    ome_med_deg: float
    ome_p95_deg: float
    angle_med_deg: float
    pos_med_bootstrap_std_um: float
    ome_med_bootstrap_std_deg: float


def trust_score_anchored(
    fitbest_view,
    *,
    n_boot: int = 200,
    seed: int = 0,
) -> TrustScoreAnchored:
    """One-shot grain trust score from the refiner's emitted predictions.

    Cheap: ~ms per grain.  Use this for population-scale auditing of a
    refined recon (10000s of grains).
    """
    res = per_grain_residuals(fitbest_view)
    if n_boot > 0 and res.n_spots >= 5:
        boot = bootstrap_uq(res, n_boot=n_boot, seed=seed)
        bp = boot.pos_med_std_um; bo = boot.ome_med_std_deg
    else:
        bp = bo = float("nan")
    return TrustScoreAnchored(
        n_spots=res.n_spots,
        pos_med_um=res.pos_med_um,
        pos_p95_um=res.pos_p95_um,
        ome_med_deg=res.ome_med_deg,
        ome_p95_deg=res.ome_p95_deg,
        angle_med_deg=res.angle_med_deg,
        pos_med_bootstrap_std_um=bp,
        ome_med_bootstrap_std_deg=bo,
    )
