"""Determinability + coverage metrics for XAF-HEDM.

The headline is **strain determinability**: for each grain we freeze the set of
accessible spots (across the chosen mountings), then differentiate their
continuous observables ``[2theta, eta, omega]`` with respect to the six
crystal-frame strain components via autograd.  The singular values of that
Jacobian say how strongly each strain direction imprints on the data:

* ``s_min`` (rad per unit strain) is the *worst-constrained* strain direction --
  the sensitivity that limits strain accuracy.  With angular measurement
  precision ``sigma``, the Cramer-Rao floor on that strain component is
  ``~ sigma / s_min``.
* ``cond = s_max / s_min`` is the anisotropy of strain resolvability.

Because it is autograd on the differentiable forward model, this needs no finite
differences and is exact.  Comparing ``s_min`` for one mounting vs both directly
quantifies what the orthogonal-axis merge buys, and sweeping the opening angle
gives the 15 vs 20 deg cell trade.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from .forward import XAFForwardModel, XAFSimulation
from .sample import GrainPopulation


# --------------------------------------------------------------------------- #
#  Detector dynamic range (integrating panels, e.g. Varex flat panels)        #
# --------------------------------------------------------------------------- #
def dynamic_range_survival(fwd, *, saturation_counts: float = 64000.0,
                           background_counts: float = 2000.0,
                           exposure_fill: float = 0.90) -> Dict[str, object]:
    """Per-reflection detectability under a charge-integrating detector.

    Unlike a photon counter, an integrating panel (Varex, Perkin-Elmer, ...)
    has a finite well: the strongest spot is exposed to ``exposure_fill`` of
    ``saturation_counts`` and a reflection is detectable only if its
    |F|^2 * Lorentz-polarisation (from midas-hkls structure factors) keeps it
    above the ``background_counts`` floor.  Because |F|^2 falls with Q, the
    reflections lost are the weak high-Q ones -- exactly those that carry the
    most elastic-strain information -- so the dynamic-range gate degrades strain
    precision more than the raw spot-count loss implies.  Returns the per-
    reflection boolean mask (indexed like ``fwd.hkls_int``), the surviving
    fraction, and the intensities, for use with an intensity-gated CRLB.
    """
    from . import crystal, structure
    cfg = fwd.cfg
    _, thetas, hkls_int = crystal.build_reflections(
        cfg.material, cfg.wavelength_A, cfg.tth_max_deg)
    inten = structure.reflection_intensities(
        cfg.material, hkls_int.numpy(), thetas.numpy(), cfg.wavelength_A)
    thresh = background_counts / (exposure_fill * saturation_counts)
    mask = inten > thresh
    return {"mask": mask, "fraction": float(mask.mean()),
            "n_survive": int(mask.sum()), "n_total": int(mask.size),
            "threshold": float(thresh), "intensities": inten,
            "dynamic_range": saturation_counts / background_counts}


# --------------------------------------------------------------------------- #
#  Coverage metrics (from the merged spot table)                              #
# --------------------------------------------------------------------------- #
def spots_per_grain(sim: XAFSimulation) -> np.ndarray:
    """Accessible spot count per grain (summed over mountings)."""
    n = sim.table.grain_id
    counts = np.zeros(sim.cfg.n_grains, dtype=int)
    if len(sim.table):
        gid, c = np.unique(n.cpu().numpy(), return_counts=True)
        counts[gid] = c
    return counts


def beam_mergeability(fwd: XAFForwardModel, grains: GrainPopulation, *,
                      half_y_um: float, half_z_um: float) -> Dict[str, object]:
    """Fraction of grains measured in EACH mounting and in ALL mountings.

    A grain is measured in a mounting if its lab position lies inside the beam
    (|Y| < half_y, |Z| < half_z) at some omega within the wedges. Because the
    sample is reoriented between mountings, a beam smaller than the sample
    illuminates a *different* sub-volume each time, so only the intersection is
    mergeable. A box beam that covers the sample gives 100% mergeability; a fixed
    slit collapses it (this is why slitting a large sample is incompatible with
    the multi-mounting merge -- use a box beam with coarser grains, or scan the
    full cross-section, instead).
    """
    from .geometry import mounting_matrix
    P = grains.position.cpu().numpy()
    centers = np.radians(np.asarray(fwd.cfg.wedge_centers_deg))
    half = np.radians(fwd.cfg.wedge_half_deg)
    oms = np.concatenate([np.linspace(c - half, c + half, 7) for c in centers])
    lit = np.zeros((P.shape[0], fwd.cfg.n_mountings), dtype=bool)
    for m in range(fwd.cfg.n_mountings):
        R = np.asarray(mounting_matrix(fwd.cfg, m), float)
        lp = (R @ P.T).T                             # lab position at omega=0
        z_ok = np.abs(lp[:, 2]) < half_z_um          # Z invariant under omega
        Yom = lp[:, 0][:, None] * np.sin(oms) + lp[:, 1][:, None] * np.cos(oms)
        y_ok = (np.abs(Yom) < half_y_um).any(axis=1)
        lit[:, m] = z_ok & y_ok
    return {
        "per_mounting_fraction": lit.mean(axis=0).tolist(),
        "mergeable_fraction": float(lit.all(axis=1).mean()),
        "n_grains": P.shape[0],
    }


def spot_overlap(sim: XAFSimulation, *, dy_px: float = 2.0, dz_px: float = 2.0,
                 domega_deg: float = 0.5) -> Dict[str, float]:
    """Fraction of spots that collide with a spot from a *different* grain.

    Two spots overlap if they land within (dy_px, dz_px) on the detector and
    within domega_deg in rotation, in the *same* mounting.  This is the classic
    FF-HEDM crowding limit: it sets the maximum grain count (minimum grain size)
    that can still be indexed.
    """
    t = sim.table
    if len(t) == 0:
        return {"overlap_fraction": 0.0, "n_spots": 0, "n_overlapped": 0}
    y = t.y_pixel.cpu().numpy()
    z = t.z_pixel.cpu().numpy()
    om = np.degrees(t.omega.cpu().numpy())
    mid = t.mounting_id.cpu().numpy()
    gid = t.grain_id.cpu().numpy()
    overlapped = np.zeros(len(t), dtype=bool)
    try:
        from scipy.spatial import cKDTree
        for m in np.unique(mid):
            idx = np.where(mid == m)[0]
            pts = np.stack([y[idx] / dy_px, z[idx] / dz_px, om[idx] / domega_deg], 1)
            tree = cKDTree(pts)
            for a, b in tree.query_pairs(r=1.0):
                if gid[idx[a]] != gid[idx[b]]:
                    overlapped[idx[a]] = overlapped[idx[b]] = True
    except ImportError:                       # grid-hash fallback (no scipy)
        seen: Dict[tuple, int] = {}
        for i in range(len(t)):
            key = (int(mid[i]), round(y[i] / dy_px), round(z[i] / dz_px),
                   round(om[i] / domega_deg))
            if key in seen and gid[seen[key]] != gid[i]:
                overlapped[i] = overlapped[seen[key]] = True
            seen[key] = i
    return {"overlap_fraction": float(overlapped.mean()),
            "n_spots": len(t), "n_overlapped": int(overlapped.sum())}


def friedel_completeness(sim: XAFSimulation) -> float:
    """Fraction of accessible spots whose Friedel mate (-hkl, same grain &
    mounting) is also accessible -- i.e. usable for grain-COM localisation."""
    t = sim.table
    if len(t) == 0:
        return 0.0
    key = torch.cat([t.mounting_id[:, None], t.grain_id[:, None], t.hkl], dim=1)
    key = key.cpu().numpy()
    have = set(map(tuple, key))
    paired = 0
    for row in key:
        mate = (row[0], row[1], -row[2], -row[3], -row[4])
        if mate in have:
            paired += 1
    return paired / len(key)


# --------------------------------------------------------------------------- #
#  Strain determinability (autograd Jacobian)                                 #
# --------------------------------------------------------------------------- #
@dataclass
class GrainDeterminability:
    grain_id: int
    n_spots: int
    singular_values: np.ndarray   # (<=6,) raw signal Jacobian (rad/strain)
    s_min: float
    s_max: float
    cond: float
    full_rank: bool               # all DOF observable
    #: worst-/best-direction 1-sigma CRLB precision in the target's units
    #: (microstrain for wrt="strain", micrometres for wrt="position"),
    #: folding in the measurement noise (detector px / Lsd, omega step).
    precision_worst: float = float("inf")
    precision_best: float = float("inf")

    @property
    def strain_precision_worst_ue(self) -> float:  # readable alias for strain
        return self.precision_worst


def _noise_weights(fwd, n_spots: int, device, dtype) -> torch.Tensor:
    """Per-observable inverse-noise weights for the CRLB.

    Observables are the *measured* quantities stacked as
    ``[y_pixel, z_pixel, frame]`` per spot: detector-plane centroid (pixels)
    and omega frame.  Noise ~ ``sigma_det_px`` (pixels) and
    ``sigma_omega_steps`` (frames).  Using pixels (not 2theta/eta) is what
    makes grain position observable -- position shifts where a spot lands, not
    its scattering angle.
    """
    cfg = fwd.cfg
    w = torch.tensor([1.0 / cfg.sigma_det_px, 1.0 / cfg.sigma_det_px,
                      1.0 / cfg.sigma_omega_steps], device=device, dtype=dtype)
    return w.repeat(n_spots)


def _frozen_indices(fwd: XAFForwardModel, euler_g, pos_g, strain_g, mountings):
    """Per-mounting ``(euler_m, pos_m, k, hkl)`` of spots accessible at nominal.

    Both the orientation and the position are carried through the remount
    transform (the grain is rigid: it rotates with the cell)."""
    idx_per_mounting = []
    from . import geometry as geo
    with torch.no_grad():
        latc = fwd._latc0.unsqueeze(0)
        for m in mountings:
            euler_m = fwd.mounting_euler(euler_g, m)
            pos_m = fwd.mounting_position(pos_g, m)
            sd = fwd.model(euler_m, pos_m, lattice_params=latc,
                           strain=strain_g.view(1, 6))
            mask = geo.accessible_mask(sd, fwd.cfg)[0]      # (K, M)
            kk, hh = torch.nonzero(mask, as_tuple=True)
            idx_per_mounting.append((euler_m, pos_m, kk, hh))
    return idx_per_mounting


# Number of free parameters and CRLB unit per determinability target.
_PARAM_NDOF = {"strain": 6, "position": 3}


def grain_determinability(
    fwd: XAFForwardModel,
    euler_g: torch.Tensor,     # (1, 3)
    pos_g: torch.Tensor,       # (1, 3)
    strain_g: torch.Tensor,    # (6,)
    wrt: str = "strain",
    grain_id: int = 0,
    mountings: Optional[Sequence[int]] = None,
) -> GrainDeterminability:
    """Jacobian-SVD determinability of one grain's ``wrt`` parameters.

    ``wrt="strain"`` -> 6 crystal-frame strain components (CRLB in microstrain);
    ``wrt="position"`` -> 3 position components (CRLB in micrometres, i.e. the
    Friedel/geometry-based localisation available in box mode).
    """
    if wrt not in _PARAM_NDOF:
        raise ValueError(f"wrt must be one of {list(_PARAM_NDOF)}")
    ndof = _PARAM_NDOF[wrt]
    if mountings is None:
        mountings = list(range(fwd.cfg.n_mountings))
    frozen = _frozen_indices(fwd, euler_g, pos_g, strain_g, mountings)
    latc = fwd._latc0.unsqueeze(0)
    dtype = latc.dtype

    def observables(theta: torch.Tensor) -> torch.Tensor:
        parts = []
        for (mi, (euler_m, pos_m, kk, hh)) in zip(mountings, frozen):
            if wrt == "strain":
                pos, strain = pos_m, theta.view(1, 6)
            else:  # position: vary the base position, re-apply the remount
                pos = fwd.mounting_position(theta.view(1, 3), mi)
                strain = strain_g.view(1, 6)
            sd = fwd.model(euler_m, pos, lattice_params=latc, strain=strain)
            # Measured quantities: detector centroid (px) + omega frame.  These
            # (unlike 2theta/eta) carry grain-position information.
            o = torch.stack([sd.y_pixel[0], sd.z_pixel[0], sd.frame_nr[0]], dim=-1)
            parts.append(o[kk, hh].reshape(-1))
        if not parts:
            return torch.zeros(0, dtype=theta.dtype, device=theta.device)
        return torch.cat(parts, dim=0)

    n_spots = sum(kk.numel() for (_, _, kk, _) in frozen)
    if n_spots == 0:
        return GrainDeterminability(grain_id, 0, np.zeros(ndof), 0.0, 0.0,
                                    float("inf"), False)

    theta0 = (strain_g if wrt == "strain" else pos_g.view(-1)).detach().clone().to(dtype)
    # Forward-mode AD: ndof (<=6) inputs << ~10^3 outputs.
    J = torch.func.jacfwd(observables)(theta0).reshape(-1, ndof)

    S = torch.linalg.svdvals(J).detach().cpu().numpy()
    if S.size < ndof:
        S = np.pad(S, (0, ndof - S.size))
    s_min, s_max = float(S.min()), float(S.max())
    cond = (s_max / s_min) if s_min > 0 else float("inf")

    w = _noise_weights(fwd, n_spots, J.device, J.dtype)
    Sw = torch.linalg.svdvals(J * w[:, None]).detach().cpu().numpy()
    ok = Sw.size == ndof and Sw.min() > 0
    prec_worst = (1.0 / Sw.min()) if ok else float("inf")
    prec_best = (1.0 / Sw.max()) if (Sw.size and Sw.max() > 0) else float("inf")
    # strain CRLB reported in microstrain; position CRLB already in um.
    scale = 1e6 if wrt == "strain" else 1.0

    return GrainDeterminability(
        grain_id, n_spots, S, s_min, s_max, cond, s_min > 0,
        precision_worst=prec_worst * scale,
        precision_best=prec_best * scale)


def grain_strain_determinability(fwd, euler_g, pos_g, strain_g, grain_id=0,
                                 mountings=None) -> GrainDeterminability:
    """Backwards-compatible strain determinability (see grain_determinability)."""
    return grain_determinability(fwd, euler_g, pos_g, strain_g, wrt="strain",
                                 grain_id=grain_id, mountings=mountings)


def population_strain_determinability(
    fwd: XAFForwardModel,
    grains: GrainPopulation,
    mountings: Optional[Sequence[int]] = None,
) -> Dict[str, object]:
    """Aggregate strain determinability over the whole grain population."""
    device = fwd.device
    per_grain: List[GrainDeterminability] = []
    for g in range(grains.n_grains):
        d = grain_strain_determinability(
            fwd,
            grains.euler[g:g + 1].to(device),
            grains.position[g:g + 1].to(device),
            grains.strain[g].to(device),
            grain_id=g, mountings=mountings,
        )
        per_grain.append(d)

    s_min = np.array([d.s_min for d in per_grain])
    n_spots = np.array([d.n_spots for d in per_grain])
    frac_full = float(np.mean([d.full_rank for d in per_grain]))
    prec = np.array([d.strain_precision_worst_ue for d in per_grain])
    good = s_min[s_min > 0]
    finite_prec = prec[np.isfinite(prec)]
    return {
        "per_grain": per_grain,
        "median_s_min": float(np.median(good)) if good.size else 0.0,
        "median_strain_precision_ue": (float(np.median(finite_prec))
                                       if finite_prec.size else float("inf")),
        "frac_full_rank": frac_full,
        "median_spots": float(np.median(n_spots)),
        "n_grains": grains.n_grains,
    }


def cross_axis_gain(fwd: XAFForwardModel, grains: GrainPopulation) -> Dict[str, float]:
    """Quantify the orthogonal-mounting payoff: single mounting vs merged."""
    single = population_strain_determinability(fwd, grains, mountings=[0])
    merged = population_strain_determinability(
        fwd, grains, mountings=list(range(fwd.cfg.n_mountings)))
    smin_s = single["median_s_min"]
    prec_s = single["median_strain_precision_ue"]
    prec_m = merged["median_strain_precision_ue"]
    return {
        "frac_full_rank_single": single["frac_full_rank"],
        "frac_full_rank_merged": merged["frac_full_rank"],
        "median_s_min_single": smin_s,
        "median_s_min_merged": merged["median_s_min"],
        "s_min_gain": (merged["median_s_min"] / smin_s) if smin_s > 0 else float("inf"),
        # worst-direction strain 1-sigma precision (microstrain), the decision #
        "strain_precision_ue_single": prec_s,
        "strain_precision_ue_merged": prec_m,
        "precision_gain": (prec_s / prec_m) if (math.isfinite(prec_s)
                          and math.isfinite(prec_m) and prec_m > 0) else float("inf"),
    }


def position_localization(
    fwd: XAFForwardModel,
    grains: GrainPopulation,
    mountings: Optional[Sequence[int]] = None,
) -> Dict[str, float]:
    """Compare grain-position localisation: box (Friedel/geometry) vs scanning.

    * box beam: position comes only from the spot geometry / Friedel pairs --
      the CRLB from the position Jacobian (micrometres).
    * line/point beam: the beam directly localises the grain to
      ``beam_size / sqrt(12)`` (uniform pencil), independent of the diffraction
      geometry -- this is the "point-focus to isolate grain positions" gain.
    """
    device = fwd.device
    precs = []
    for g in range(grains.n_grains):
        d = grain_determinability(
            fwd, grains.euler[g:g + 1].to(device),
            grains.position[g:g + 1].to(device), grains.strain[g].to(device),
            wrt="position", grain_id=g, mountings=mountings)
        precs.append(d.precision_worst)
    precs = np.array(precs)
    finite = precs[np.isfinite(precs)]
    box_um = float(np.median(finite)) if finite.size else float("inf")
    scan_um = fwd.cfg.beam_size_um / math.sqrt(12.0)
    scanning = fwd.cfg.beam_mode in ("line", "point")
    return {
        "beam_mode": fwd.cfg.beam_mode,
        "box_friedel_position_um": box_um,
        "scanning_position_um": scan_um,
        "effective_position_um": scan_um if scanning else box_um,
    }
