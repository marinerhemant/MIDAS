"""Rotation-axis ``Wedge`` (and the OmegaStep scale) from Friedel pairs.

This is the method of the original ``FitWedge.c`` (2014), applied to every
Friedel pair of a reconstruction at once and fitted jointly with an omega-scale
term.

Why Friedel pairs, and why not ``grain-tx``
-------------------------------------------
A reflection ``G`` and its Friedel mate ``-G`` diffract about 180 deg apart in
omega. With the rotation axis exactly perpendicular to the beam the separation
is EXACTLY 180 deg, whatever the grain's orientation, position or strain and
whatever the detector geometry. An axis tilt ``W`` (the MIDAS ``Wedge``,
``G_lab = R_y(-W) R_z(omega) O g``) breaks that symmetry; to first order

    omega_2 - omega_1 - 180 = -2 W cot(eta_1)         (radians; eta = atan2(-Y, Z), MIDAS)

(Sign checked against midas_diffract.forward at W = +-0.05 and +0.0246 by the 2026-10-07
/verify physics lens; earlier text said +2 W cot. The fitter solves the Bragg condition exactly
and never used this formula, so fitted values were unaffected.)

so the wedge lives in the omega of the Bragg events. ``grain-tx`` minimises the
on-detector (Y, Z) distance with the grain poses frozen at the values the
reconstruction refined at the old wedge, and never compares observed omegas, so
one pass recovers only part of a wedge (pipeline re-runs converge geometrically;
``manuals/FF_Grain_Geometry_tx.md``). This module uses the first-order channel
directly and is absolute in one pass.

An error in ``OmegaStep`` scales every omega about the first frame, which adds
a CONSTANT to every pair's separation (180 * relative step error). Fitting the
wedge pair by pair without that term (``FitWedgeParallel.c``) charges the
constant to ``W`` with opposite signs for spots above and below the beam
(cot eta > 0 vs < 0). Here both are fitted jointly, so the result also measures
the true OmegaStep.

What is fitted
--------------
For each same-grain, same-ring pair (spot ``a`` earlier, ``b`` later) the
observed diffraction vector of ``a`` (from its raw-frame detector position and
the grain's position) is rotated back to the stage frame at the trial wedge,
negated, and the Bragg condition for ``-G`` is solved EXACTLY at the trial
wedge. The residual is the observed omega of ``b`` minus the nearer of the two
solutions. Observed omegas are rescaled about ``OmegaStart`` by ``(1 + eps)``;
``(W, eps)`` minimise the loss selected by ``loss`` (default ``"quant"``, a
quantisation-aware weighted least squares; ``"ls"`` and ``"huber"`` are also
available). Standard errors come from a bootstrap over PAIRS in peaks mode and
over GRAINS in grains mode (pairs of one grain share its orientation and
position errors).

Inputs are the RAW frame: ``OmeRaw`` and (Y, Z) re-derived from the raw pixels
with the pipeline's own detector correction, so the result is the ABSOLUTE
wedge whatever Wedge the reconstruction itself ran with.
"""
from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

LOG = logging.getLogger(__name__)

__all__ = [
    "FriedelPairs", "FriedelWedgeResult", "find_friedel_pairs", "find_friedel_pairs_peaks",
    "chance_pair_fraction", "load_raw_peaks",
    "predict_mate_omega", "fit_wedge_omega_scale", "friedel_wedge",
    "corrected_params_text",
]


def _wrap(a):
    return (np.asarray(a) + 180.0) % 360.0 - 180.0


# ──────────────────────────────────────────────────────────────── pairs
@dataclass
class FriedelPairs:
    """Friedel pairs as parallel arrays (spot ``a`` = the EARLIER omega)."""
    grain: np.ndarray            # grain id of the pair
    ring: np.ndarray             # ring number
    ome_a: np.ndarray            # deg, raw omega of the earlier spot
    ome_b: np.ndarray            # deg, raw omega of the later spot
    y_a: np.ndarray              # um, raw-frame lab Y of spot a
    z_a: np.ndarray              # um, raw-frame lab Z of spot a
    eta_a: np.ndarray            # deg, atan2(-Y, Z) of spot a
    pos: np.ndarray              # (n, 3) um, grain position (stage frame), zeros if unknown
    idx_a: Optional[np.ndarray] = None   # indices of a / b in the input arrays
    idx_b: Optional[np.ndarray] = None

    def __len__(self) -> int:
        return int(self.grain.shape[0])

    def subset(self, m: np.ndarray) -> "FriedelPairs":
        f = ("grain", "ring", "ome_a", "ome_b", "y_a", "z_a", "eta_a", "pos")
        return FriedelPairs(*(getattr(self, k)[m] for k in f),
                            idx_a=None if self.idx_a is None else self.idx_a[m],
                            idx_b=None if self.idx_b is None else self.idx_b[m])


def find_friedel_pairs(grain, ring, ome, y, z, *, pos_by_grain: Optional[Dict[int, np.ndarray]] = None,
                       ome_tol: float = 1.0, eta_tol: float = 1.0) -> FriedelPairs:
    """Pair spots of the same grain and ring whose omegas differ by 180 deg
    (within ``ome_tol``) and whose etas mirror (``eta_b ~ sign(eta_a)*180 - eta_a``
    within ``eta_tol``). Only one-to-one matches are kept, and each pair once,
    ordered so that ``a`` is the EARLIER omega. (A pairing rule that admits the
    reversed order conditionally on the sign of the separation selects on the
    outcome and biases the fit.)"""
    grain = np.asarray(grain).astype(np.int64)
    ring = np.asarray(ring).astype(np.int64)
    ome = np.asarray(ome, float); y = np.asarray(y, float); z = np.asarray(z, float)
    eta = np.degrees(np.arctan2(-y, z))
    order = np.lexsort((ring, grain))
    g_s, r_s = grain[order], ring[order]
    brk = np.flatnonzero((np.diff(g_s) != 0) | (np.diff(r_s) != 0)) + 1
    ia_all, ib_all = [], []
    for blk in np.split(order, brk):
        if blk.size < 2:
            continue
        o, e = ome[blk], eta[blk]
        sep = _wrap(o[None, :] - o[:, None] - 180.0)              # [i, j]
        mir = np.where(e >= 0, 180.0, -180.0)[:, None] - e[:, None]
        ok = (np.abs(sep) < ome_tol) & (np.abs(_wrap(e[None, :] - mir)) < eta_tol)
        ok &= o[None, :] > o[:, None]
        i, j = np.nonzero(ok)
        if i.size == 0:
            continue
        ci = np.bincount(i, minlength=blk.size); cj = np.bincount(j, minlength=blk.size)
        # a spot may appear as ``a`` of one pair or ``b`` of one pair, not both roles twice
        keep = (ci[i] == 1) & (cj[j] == 1)
        ia_all.append(blk[i[keep]]); ib_all.append(blk[j[keep]])
    if ia_all:
        ia = np.concatenate(ia_all); ib = np.concatenate(ib_all)
    else:
        ia = ib = np.zeros(0, dtype=np.int64)
    pos = np.zeros((ia.size, 3))
    if pos_by_grain:
        for k, g in enumerate(grain[ia]):
            p = pos_by_grain.get(int(g))
            if p is not None:
                pos[k] = p
    return FriedelPairs(grain=grain[ia], ring=ring[ia], ome_a=ome[ia], ome_b=ome[ib],
                        y_a=y[ia], z_a=z[ia], eta_a=eta[ia], pos=pos, idx_a=ia, idx_b=ib)


def find_friedel_pairs_peaks(ring, ome, y, z, *, ome_tol: float = 0.5,
                             eta_tol: float = 0.6, rad_tol_um: float = 1500.0,
                             max_cot: float = 3.0, separation: float = 180.0) -> FriedelPairs:
    """Friedel pairs straight from the RAW peak list, with no grain assignment.

    Why: a reconstruction's SpotMatrix keeps only spots that matched a
    prediction within ``MarginOme`` of the omega predicted at the run's Wedge.
    A wrong wedge shifts the predicted Friedel separation by up to ~0.15 deg, so
    the matched set is selected on the very quantity being measured; on
    a 20-ID Varex run ~6 % of the run's Wedge leaked into the result that way.
    Pairing raw peaks with a window much wider than the signal removes it: the
    result does not depend on the reconstruction at all.

    A peak ``a`` (omega < max - 180) is paired with the UNIQUE peak of the same
    ring within ``ome_tol`` of ``omega_a + 180``, ``eta_tol`` of the mirrored eta
    and ``rad_tol_um`` of its radius (grain offsets move a spot by up to ~0.5 mm,
    oppositely for the mate, hence the generous eta/radius windows), and the
    match must be unique in reverse too. Pairs are restricted to
    ``|cot eta| <= max_cot`` up front (no wrap at the poles). ``grain`` is set to
    the pair index, so a bootstrap is over pairs (it ignores the correlation of
    pairs from one grain; quote the grain-mode SE when that matters).

    ``separation`` other than 180 is the chance-pair NULL: no Friedel mate sits
    at, say, 183 deg, so whatever passes the same windows and uniqueness tests
    there is a random coincidence. See :func:`chance_pair_fraction`."""
    from scipy.spatial import cKDTree
    ring = np.asarray(ring).astype(np.int64)
    ome = np.asarray(ome, float); y = np.asarray(y, float); z = np.asarray(z, float)
    eta = np.degrees(np.arctan2(-y, z)); R = np.hypot(y, z)
    t = np.tan(np.radians(eta))
    with np.errstate(divide="ignore"):
        cot = np.where(t == 0, np.inf, 1.0 / t)
    ok = (ring > 0) & (np.abs(cot) <= max_cot) & np.isfinite(ome)
    mirror = lambda e: np.where(e >= 0, 180.0, -180.0) - e
    scale = np.array([1.0 / ome_tol, 1.0 / eta_tol, 1.0 / rad_tol_um])
    ia_all, ib_all = [], []
    ome_max = ome[ok].max() if ok.any() else 0.0
    for r in np.unique(ring[ok]):
        idx = np.flatnonzero(ok & (ring == r))
        pts = np.column_stack([ome[idx], eta[idx], R[idx]]) * scale
        tree = cKDTree(pts)
        a = idx[ome[idx] < ome_max - separation + ome_tol]
        tgt = np.column_stack([ome[a] + separation, mirror(eta[a]), R[a]]) * scale
        nb = tree.query_ball_point(tgt, r=1.0, p=np.inf)
        cand = [(k, idx[n[0]]) for k, n in enumerate(nb) if len(n) == 1]
        if not cand:
            continue
        ka, b = map(np.array, zip(*cand))
        a = a[ka]
        # reverse uniqueness: from b, exactly one peak near omega_b - 180
        rev = np.column_stack([ome[b] - separation, mirror(eta[b]), R[b]]) * scale
        nb2 = tree.query_ball_point(rev, r=1.0, p=np.inf)
        keep = np.array([len(n) == 1 and idx[n[0]] == aa for n, aa in zip(nb2, a)], bool)
        ia_all.append(a[keep]); ib_all.append(b[keep])
    ia = np.concatenate(ia_all) if ia_all else np.zeros(0, np.int64)
    ib = np.concatenate(ib_all) if ib_all else np.zeros(0, np.int64)
    return FriedelPairs(grain=np.arange(ia.size), ring=ring[ia], ome_a=ome[ia], ome_b=ome[ib],
                        y_a=y[ia], z_a=z[ia], eta_a=eta[ia], pos=np.zeros((ia.size, 3)),
                        idx_a=ia, idx_b=ib)


def _candidate_counts(ring, ome, y, z, *, separation: float, ome_tol: float = 0.5,
                      eta_tol: float = 0.6, rad_tol_um: float = 1500.0,
                      max_cot: float = 3.0) -> np.ndarray:
    """For every eligible peak ``a``: how many peaks sit in the pairing window
    at ``omega_a + separation`` (same windows as :func:`find_friedel_pairs_peaks`)."""
    from scipy.spatial import cKDTree
    ring = np.asarray(ring).astype(np.int64)
    ome = np.asarray(ome, float); y = np.asarray(y, float); z = np.asarray(z, float)
    eta = np.degrees(np.arctan2(-y, z)); R = np.hypot(y, z)
    t = np.tan(np.radians(eta))
    with np.errstate(divide="ignore"):
        cot = np.where(t == 0, np.inf, 1.0 / t)
    ok = (ring > 0) & (np.abs(cot) <= max_cot) & np.isfinite(ome)
    mirror = lambda e: np.where(e >= 0, 180.0, -180.0) - e
    scale = np.array([1.0 / ome_tol, 1.0 / eta_tol, 1.0 / rad_tol_um])
    ome_max = ome[ok].max() if ok.any() else 0.0
    out = []
    for r in np.unique(ring[ok]):
        idx = np.flatnonzero(ok & (ring == r))
        tree = cKDTree(np.column_stack([ome[idx], eta[idx], R[idx]]) * scale)
        a = idx[ome[idx] < ome_max - max(separation, 180.0) - ome_tol]
        tgt = np.column_stack([ome[a] + separation, mirror(eta[a]), R[a]]) * scale
        out.append(np.array([len(n) for n in tree.query_ball_point(tgt, r=1.0, p=np.inf)]))
    return np.concatenate(out) if out else np.zeros(0, int)


def chance_pair_fraction(ring, ome, y, z, n_pairs: int, *, offsets=(-3.0, 3.0),
                         **kw) -> float:
    """Estimated fraction of the ``n_pairs`` peak-list pairs that are WRONG.

    A wrong pair (a, b) survives the two-way uniqueness test only if BOTH true
    mates are missing (too weak, overlapped, off the detector): a present mate
    of ``a`` makes the forward match ambiguous, a present mate of ``b`` the
    reverse one. So

        wrong ~= decoy_pairs * P(mate missing)^2

    ``decoy_pairs``: the full pairing run at ``180 + offset``, where no mate
    exists (both "missing" by construction) -- the same windows, the same peak
    density. ``P(mate missing) = P(empty window at 180) / P(empty at 180 +
    offset)`` from the forward candidate counts. The decoy count alone is NOT
    the rate: on synthetics with every mate present it read 3-35 % against a
    true 0.00-0.05 %."""
    if n_pairs <= 0:
        return float("nan")
    k0 = _candidate_counts(ring, ome, y, z, separation=180.0, **kw)
    if k0.size == 0:
        return float("nan")
    decoys, empty_off = [], []
    for o in offsets:
        decoys.append(len(find_friedel_pairs_peaks(ring, ome, y, z, separation=180.0 + o, **kw)))
        empty_off.append(float(np.mean(_candidate_counts(ring, ome, y, z,
                                                         separation=180.0 + o, **kw) == 0)))
    pe = float(np.mean(empty_off))
    p_missing = min(1.0, float(np.mean(k0 == 0)) / pe) if pe > 0 else 1.0
    return float(np.mean(decoys)) * p_missing ** 2 / float(n_pairs)


def load_raw_peaks(layer_dir) -> Dict[str, np.ndarray]:
    """Raw peak list of a reconstruction: ``InputAllExtraInfoFittingAll.csv``
    columns ``OmegaIni`` (pre-wedge omega), ``YOrigDetCor``/``ZOrigDetCor``
    (pre-wedge, detector-corrected lab Y/Z, um) and ``RingNumber``. These do not
    depend on the Wedge the run used."""
    f = Path(layer_dir) / "InputAllExtraInfoFittingAll.csv"
    with open(f) as fh:
        names = fh.readline().split()
    need = ("OmegaIni", "YOrigDetCor", "ZOrigDetCor", "RingNumber")
    missing = [n for n in need if n not in names]
    if missing:
        raise ValueError(f"{f} lacks columns {missing} (has {names}).")
    cols = [names.index(n) for n in need]
    arr = np.loadtxt(f, skiprows=1, usecols=cols, ndmin=2)
    return {"ome": arr[:, 0], "y": arr[:, 1], "z": arr[:, 2], "ring": arr[:, 3].astype(np.int64)}


# ─────────────────────────────────────────────────────────── forward
def predict_mate_omega(y_a, z_a, ome_a_deg, lsd_um, wedge_deg, ome_b_hint_deg,
                       pos_um=None) -> np.ndarray:
    """Omega (deg) at which ``-G`` diffracts, given the observed spot of ``G``.

    ``(y_a, z_a)``: raw-frame lab position of spot ``a`` on the detector plane at
    ``x = lsd_um`` (um). ``pos_um``: grain position in the stage frame (um), the
    ray origin at ``ome_a``; zeros if None. Convention (``midas_diffract``):
    ``G_lab = R_y(-W) R_z(omega) g`` with the beam along +x, so the Bragg
    condition is ``G_lab . x = -|G|^2 / 2`` in units of the wavenumber. Of the
    two solutions, the one nearer ``ome_b_hint_deg`` is returned (NaN when the
    mate cannot diffract).
    """
    y_a = np.asarray(y_a, float); z_a = np.asarray(z_a, float)
    om = np.radians(np.asarray(ome_a_deg, float))
    W = math.radians(float(wedge_deg)); cW, sW = math.cos(W), math.sin(W)
    if pos_um is None:
        px = py = pz = np.zeros_like(y_a)
    else:
        pos_um = np.asarray(pos_um, float)
        px, py, pz = pos_um[:, 0], pos_um[:, 1], pos_um[:, 2]
    # grain position in the lab at ome_a: R_y(-W) R_z(om) p
    mx = px * np.cos(om) - py * np.sin(om)
    my = px * np.sin(om) + py * np.cos(om)
    lx = cW * mx - sW * pz
    lz = sW * mx + cW * pz
    d = np.stack([lsd_um - lx, y_a - my, z_a - lz], axis=-1)
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    G = d - np.array([1.0, 0.0, 0.0])                          # k_out - k_in, |k| = 1
    # back to the stage frame: g = R_z(-om) R_y(W) G
    gx_ = cW * G[:, 0] + sW * G[:, 2]
    gz = -sW * G[:, 0] + cW * G[:, 2]
    gy_ = G[:, 1]
    gx = gx_ * np.cos(om) + gy_ * np.sin(om)
    gy = -gx_ * np.sin(om) + gy_ * np.cos(om)
    hx, hy, hz = -gx, -gy, -gz                                  # Friedel mate
    hh = np.hypot(hx, hy)
    h2 = hx * hx + hy * hy + hz * hz
    alpha = np.arctan2(hy, hx)
    # cW*(hh cos(w + alpha)) - sW*hz = -h2/2
    c = (-0.5 * h2 + sW * hz) / (cW * hh)
    with np.errstate(invalid="ignore"):
        base = np.arccos(c)
    w1 = np.degrees(base - alpha); w2 = np.degrees(-base - alpha)
    hint = np.asarray(ome_b_hint_deg, float)
    d1 = np.abs(_wrap(w1 - hint)); d2 = np.abs(_wrap(w2 - hint))
    out = np.where(d1 <= d2, hint + _wrap(w1 - hint), hint + _wrap(w2 - hint))
    out[np.abs(c) > 1] = np.nan
    return out


def _residual(pairs: FriedelPairs, lsd_um: float, ome_ref: float, wedge_deg: float,
              eps: float) -> np.ndarray:
    """Observed minus predicted omega of the later spot (deg)."""
    oa = ome_ref + (pairs.ome_a - ome_ref) * (1.0 + eps)
    ob = ome_ref + (pairs.ome_b - ome_ref) * (1.0 + eps)
    pred = predict_mate_omega(pairs.y_a, pairs.z_a, oa, lsd_um, wedge_deg, ob, pairs.pos)
    return _wrap(ob - pred)


# ─────────────────────────────────────────────────────────── fit
def _huber_gn(pairs, lsd_um, ome_ref, x0, fit_scale, *, c=1.345, it=40):
    """Gauss-Newton with Huber IRLS weights on (W deg, eps). ``eps`` is held at
    ``x0[1]`` when ``fit_scale`` is False. Returns x, robust sigma, n used."""
    x = np.array(x0, float)
    steps = np.array([1e-4, 1e-6])
    free = [0, 1] if fit_scale else [0]
    s = None
    for _ in range(it):
        r = _residual(pairs, lsd_um, ome_ref, *x)
        ok = np.isfinite(r)
        if ok.sum() < 10:
            raise RuntimeError("fewer than 10 Friedel pairs have a valid prediction")
        rr = r[ok]
        s = 1.4826 * np.median(np.abs(rr - np.median(rr))) + 1e-12
        w = np.clip(c * s / np.maximum(np.abs(rr), 1e-15), None, 1.0)
        J = np.empty((rr.size, len(free)))
        for k, f in enumerate(free):
            dx = np.zeros(2); dx[f] = steps[f]
            rp = _residual(pairs, lsd_um, ome_ref, *(x + dx))[ok]
            rm = _residual(pairs, lsd_um, ome_ref, *(x - dx))[ok]
            J[:, k] = (rp - rm) / (2 * steps[f])
        sw = np.sqrt(w)
        delta = np.linalg.lstsq(J * sw[:, None], -rr * sw, rcond=None)[0]
        x[free] += delta
        if np.all(np.abs(delta) < np.array([1e-7, 1e-9])[free]):
            break
    return x, float(s), int(np.isfinite(_residual(pairs, lsd_um, ome_ref, *x)).sum())


# ─────────────────────────────────────────── omega-grid quantisation
#: A spot that lies in ONE frame is reported at the frame LABEL (the merge takes a
#: label-weighted mean), so its omega carries no sub-frame information and its error
#: is a rounding error bounded by +-step/2, not a continuous Gaussian. A MAD-scaled
#: Huber loss reads these lattice-valued residuals (0, +-step, ...) as outliers and
#: biases the wedge toward 0 (measured on the AlON FF layers, 2026-10-12).
SNAP_TOL_FRAMES = 0.02


def omega_grid_phase(ome, ome_ref: float, step: float, nbin: int = 100) -> Tuple[float, float]:
    """Phase (frames, 0..1) of the omega frame grid and the fraction of omegas on it.

    ``frac((omega - ome_ref) / step)`` of snapped spots piles up in one bin; the
    chance level for a flat distribution is ``2 * SNAP_TOL_FRAMES``."""
    f = np.mod((np.asarray(ome, float) - float(ome_ref)) / float(step), 1.0)
    h, edges = np.histogram(f, bins=nbin, range=(0.0, 1.0))
    hs = h + np.roll(h, 1) + np.roll(h, -1)
    k = int(np.argmax(hs))
    c = 0.5 * (edges[k] + edges[k + 1])
    d = ((f - c + 0.5) % 1.0) - 0.5
    near = np.abs(d) < 0.03
    phase = float((c + d[near].mean()) % 1.0) if near.any() else float(c)
    return phase, float(on_omega_grid(ome, ome_ref, step, phase).mean())


def on_omega_grid(ome, ome_ref: float, step: float, phase: float,
                  tol: float = SNAP_TOL_FRAMES) -> np.ndarray:
    f = (np.asarray(ome, float) - float(ome_ref)) / float(step) - float(phase)
    return np.abs(f - np.round(f)) < tol


def snapped_members(pairs: "FriedelPairs", ome_ref: float, step: float,
                    min_fraction: float = 0.08) -> Tuple[np.ndarray, float, float]:
    """Per pair, how many of its two omegas sit on the frame grid (0, 1, 2).

    Returns ``(n_snapped, phase, fraction_of_members_on_grid)``. When the grid
    pile-up is not clearly above chance (``min_fraction``) nothing is treated as
    snapped (all zeros)."""
    allome = np.concatenate([pairs.ome_a, pairs.ome_b])
    phase, frac = omega_grid_phase(allome, ome_ref, step)
    if frac < min_fraction:
        return np.zeros(len(pairs), dtype=np.int64), phase, frac
    sa = on_omega_grid(pairs.ome_a, ome_ref, step, phase)
    sb = on_omega_grid(pairs.ome_b, ome_ref, step, phase)
    return (sa.astype(np.int64) + sb.astype(np.int64)), phase, frac


def _wls_gn(pairs, lsd_um, ome_ref, x0, fit_scale, cls, step, *, it=40, k_trim=4.0,
            s0_floor=0.02):
    """Gauss-Newton, WEIGHTED LEAST SQUARES on (W deg, eps) with class-specific variances.

    ``cls``: snapped members per pair (0, 1, 2). Each class gets its own residual
    variance from the pairs inside a PHYSICAL trim bound ``step/2`` per snapped
    member plus ``k_trim`` times the continuous scatter (from the class-0 pairs;
    a MAD only over those, never over the lattice-valued classes). Weights are
    ``1/variance``; pairs beyond the bound are dropped (wrong pairs). Squared loss
    is deliberate: rounding errors are bounded and zero-mean, so least squares is
    unbiased for them where a robust loss is not. Returns
    ``x, s0 (continuous scatter), n_used, info``."""
    x = np.array(x0, float)
    steps = np.array([1e-4, 1e-6])
    free = [0, 1] if fit_scale else [0]
    info = {}
    for _ in range(it):
        r = _residual(pairs, lsd_um, ome_ref, *x)
        ok = np.isfinite(r)
        if ok.sum() < 10:
            raise RuntimeError("fewer than 10 Friedel pairs have a valid prediction")
        rr, cc = r[ok], cls[ok]
        r0 = rr[cc == 0]
        ref = r0 if r0.size >= 30 else rr
        s0 = max(1.4826 * np.median(np.abs(ref - np.median(ref))), s0_floor)
        bound = 0.5 * step * cc + (0.0 if k_trim is None else k_trim) * s0 * math.sqrt(2.0)
        keep = (np.ones_like(rr, dtype=bool) if k_trim is None else (np.abs(rr) <= bound))
        var = {}
        for c in (0, 1, 2):
            m = keep & (cc == c)
            var[c] = float(np.mean(rr[m] ** 2)) if m.sum() >= 30 else None
        base = var[0] if var[0] is not None else 2.0 * s0 ** 2
        for c in (0, 1, 2):
            if var[c] is None:                      # too few pairs: model value
                var[c] = base + c * step ** 2 / 12.0
        v = np.array([var[int(c)] for c in cc])
        w = keep / v
        J = np.empty((rr.size, len(free)))
        for k, f in enumerate(free):
            dx = np.zeros(2); dx[f] = steps[f]
            rp = _residual(pairs, lsd_um, ome_ref, *(x + dx))[ok]
            rm = _residual(pairs, lsd_um, ome_ref, *(x - dx))[ok]
            J[:, k] = (rp - rm) / (2 * steps[f])
        sw = np.sqrt(w)
        delta = np.linalg.lstsq(J * sw[:, None], -rr * sw, rcond=None)[0]
        x[free] += delta
        info = {"s0": float(s0), "class_sigma_deg": {c: math.sqrt(var[c]) for c in (0, 1, 2)},
                "class_counts": {c: int((cls[ok] == c).sum()) for c in (0, 1, 2)},
                "class_used": {c: int((keep & (cc == c)).sum()) for c in (0, 1, 2)}}
        if np.all(np.abs(delta) < np.array([1e-7, 1e-9])[free]):
            break
    n = int(info["class_used"][0] + info["class_used"][1] + info["class_used"][2])
    return x, float(info["s0"]), n, info


def _cot_mask(pairs: FriedelPairs, max_cot: float) -> np.ndarray:
    t = np.tan(np.radians(pairs.eta_a))
    with np.errstate(divide="ignore"):
        cot = np.where(t == 0, np.inf, 1.0 / t)
    return np.abs(cot) <= max_cot


def fit_wedge_omega_scale(pairs: FriedelPairs, *, lsd_um: float, ome_ref: float,
                          fit_omega_scale: bool = True, wedge0: float = 0.0,
                          eps0: float = 0.0, max_cot: float = 3.0, n_boot: int = 100,
                          seed: int = 0, loss: str = "quant",
                          omega_step: Optional[float] = None,
                          cls: Optional[np.ndarray] = None, return_info: bool = False):
    """Fit ``(W, eps)``; return ``(x, sigma, n_used, boot_samples)`` (+ ``info`` if asked).

    ``loss``: ``"quant"`` (default) weighted least squares with class-specific
    variances for pairs whose omegas sit on the frame grid (needs ``omega_step``; with
    no ``cls`` it detects the grid itself), ``"ls"`` plain least squares,
    ``"huber"`` the legacy MAD-scaled Huber (biased toward 0 on quantised omegas).
    ``max_cot`` drops pairs within ~arccot(max_cot) of the poles (eta 0/180),
    where omega is ill-conditioned and spots streak. With
    ``fit_omega_scale=False`` eps is held at ``eps0``. The bootstrap resamples
    ``pairs.grain`` (grains in grains mode; each pair on its own in peaks mode)."""
    if loss not in ("quant", "ls", "huber"):
        raise ValueError(f"loss must be 'quant', 'ls' or 'huber', got {loss!r}")
    m = _cot_mask(pairs, max_cot)
    p = pairs.subset(m)
    if len(p) < 50:
        raise RuntimeError(f"only {len(p)} usable Friedel pairs (need >= 50)")
    c = None
    if loss in ("quant", "ls"):
        if cls is not None:
            c = np.asarray(cls)[m]
        elif loss == "quant" and omega_step is not None:
            c = snapped_members(p, ome_ref, omega_step)[0]
        else:
            c = np.zeros(len(p), dtype=np.int64)
        stp = float(omega_step) if omega_step is not None else 0.0
        if loss == "ls":
            c = np.zeros(len(p), dtype=np.int64)

    def _fit(pp, cc, x0, it):
        if loss == "huber":
            x_, s_, n_ = _huber_gn(pp, lsd_um, ome_ref, x0, fit_omega_scale, it=it)
            return x_, s_, n_, {}
        return _wls_gn(pp, lsd_um, ome_ref, x0, fit_omega_scale, cc, stp, it=it,
                       k_trim=None if loss == "ls" else 4.0)

    x, s, n, info = _fit(p, c, (wedge0, eps0), 40)
    boots = []
    if n_boot:
        rng = np.random.default_rng(seed)
        gids = np.unique(p.grain)
        idx = {g: np.flatnonzero(p.grain == g) for g in gids}
        for _ in range(n_boot):
            rows = np.concatenate([idx[g] for g in rng.choice(gids, gids.size)])
            boots.append(_fit(p.subset(rows), None if c is None else c[rows], x, 8)[0])
    out = (x, s, n, np.array(boots))
    return out + (info,) if return_info else out


# ─────────────────────────────────────────────────────────── driver
@dataclass
class FriedelWedgeResult:
    wedge_deg: float
    wedge_se_deg: float
    omega_scale_eps: float          # omega_true = OmegaStart + (omega_raw - OmegaStart) * (1 + eps)
    omega_scale_se: float
    omega_step_in: float
    omega_step_fit: float           # OmegaStep * (1 + eps)
    n_pairs: int
    n_grains: int
    sigma_deg: float                # robust scatter of the omega residual
    splits: Dict[str, Tuple[float, float, int]] = field(default_factory=dict)  # name -> (W, SE, n)
    warnings: list = field(default_factory=list)
    paramstest_out: Optional[Path] = None
    pair_source: str = "peaks"      # "peaks" (raw peak list) or "grains" (SpotMatrix)
    chance_pair_fraction: float = float("nan")  # peaks mode: null-separation pairs / pairs
    estimator: str = "huber"        # "quant" | "ls" | "huber"
    snapped_member_fraction: float = float("nan")   # share of pair members with omega ON the frame grid
    grid_phase_frames: float = float("nan")         # phase of that grid, frames
    class_counts: Dict[int, int] = field(default_factory=dict)       # pairs by # snapped members (0,1,2)
    class_sigma_deg: Dict[int, float] = field(default_factory=dict)  # residual scatter of each class


def _check_geometry(paramstest: Path, v1, hedm: dict) -> None:
    """The geometry must be the one the reconstruction ran with. Either the
    input parameter file or the layer ``paramstest.txt`` will do: the reader
    maps the layer file's ``LsdFit``/``YBCFit``/``txFit``... onto the canonical
    keys (``midas_calibrate.params.FIT_KEY_ALIASES``)."""
    if float(v1.Lsd) <= 0:
        raise ValueError(f"{paramstest} gives no positive Lsd (nor LsdFit/Distance).")
    if float(hedm.get("OmegaStep", 0.0)) == 0.0:
        raise ValueError(f"{paramstest} has no non-zero OmegaStep.")


def corrected_params_text(txt: str, *, wedge_deg: float,
                          omega_step: Optional[float] = None) -> str:
    """Write ``Wedge`` (absolute) and optionally ``OmegaStep`` into ``txt``."""
    def _emit(t, name, value):
        line = f"{name} {value:.10g}"
        pat = rf"(?m)^{name}\b.*$"
        if re.search(pat, t):
            return re.sub(pat, line, t)
        return t + ("" if t.endswith("\n") else "\n") + line + "\n"
    txt = _emit(txt, "Wedge", wedge_deg)
    if omega_step is not None:
        txt = _emit(txt, "OmegaStep", omega_step)
    return txt


def friedel_wedge(paramstest, layer_dir, *, pair_source: str = "peaks",
                  fit_omega_scale: bool = True,
                  max_cot: float = 3.0, ome_tol: Optional[float] = None,
                  eta_tol: Optional[float] = None,
                  n_boot: int = 100, min_pairs: int = 200, use_positions: bool = True,
                  splits: bool = True, out_paramstest=None, write_omega_step: bool = False,
                  seed: int = 0, loss: str = "quant") -> FriedelWedgeResult:
    """Measure the absolute ``Wedge`` (and the OmegaStep scale) of a reconstruction.

    ``loss``: ``"quant"`` (default) is quantisation-aware: a spot in ONE frame is
    reported at its frame label, so pairs whose omegas sit on the frame grid carry
    only the frame step; they are weighted by their own residual variance and
    trimmed at a physical bound (half a frame per snapped member), and the wedge is
    also reported per quantisation class. ``"ls"`` is plain least squares;
    ``"huber"`` the legacy MAD-scaled Huber, which is biased toward 0 when many
    omegas are snapped. The standard error is a bootstrap over PAIRS in
    ``pair_source="peaks"`` (no grain assignment) and over grains in ``"grains"``.

    ``pair_source``: ``"peaks"`` (default) pairs the raw peak list
    (``InputAllExtraInfoFittingAll.csv``) with no grain assignment, so the result
    cannot depend on the reconstruction; ``"grains"`` pairs within each grain's
    matched spots (``SpotMatrix.csv``), which is selected through ``MarginOme``
    around the omegas predicted at the run's Wedge and leaks ~6 % of that Wedge
    into the answer (seen on a 20-ID Varex run).

    ``paramstest`` is the geometry the reconstruction ran with: the input
    parameter file or the layer ``paramstest.txt``; ``layer_dir`` holds the
    reconstruction's ``SpotMatrix.csv`` (with ``OmeRaw``) and ``Grains.csv``.
    """
    import torch
    from midas_calibrate.params import CalibrationParams as V1Params
    from .grain_refine import _raw_frame_converter, _read_hedm_keys
    from .grain_observations import load_grains_csv, load_spot_matrix

    paramstest = Path(paramstest); layer_dir = Path(layer_dir)
    v1 = V1Params.from_file(str(paramstest))
    hedm = _read_hedm_keys(paramstest)
    _check_geometry(paramstest, v1, hedm)
    ome_ref = float(hedm["OmegaFirstFile"]); step = float(hedm["OmegaStep"])

    if pair_source == "peaks":
        pk = load_raw_peaks(layer_dir)
        pkw = dict(ome_tol=0.5 if ome_tol is None else ome_tol,
                   eta_tol=0.6 if eta_tol is None else eta_tol, max_cot=max_cot)
        pairs = find_friedel_pairs_peaks(pk["ring"], pk["ome"], pk["y"], pk["z"], **pkw)
        chance = chance_pair_fraction(pk["ring"], pk["ome"], pk["y"], pk["z"],
                                      len(pairs), **pkw)
    elif pair_source == "grains":
        chance = float("nan")
        s = load_spot_matrix(layer_dir / "SpotMatrix.csv")
        if s.get("ome_raw") is None:
            raise ValueError("SpotMatrix.csv has no OmeRaw column; regenerate it with "
                             "midas-process-grains (the 28-column format).")
        conv = _raw_frame_converter(v1, paramstest, hedm, dtype=torch.float64, device="cpu")
        bag = conv({"spot_id": s["spot_id"], "det_hor": s["det_hor"],
                    "det_vert": s["det_vert"], "ome_raw": s["ome_raw"]})
        pos_by_grain = None
        if use_positions and (layer_dir / "Grains.csv").exists():
            g = load_grains_csv(layer_dir / "Grains.csv")
            pos_by_grain = {int(i): np.asarray(p, float) for i, p in zip(g["ids"], g["positions"])}
        pairs = find_friedel_pairs(s["grain_id"], s["ring_nr"], bag["omega"], bag["y_lab"],
                                   bag["z_lab"], pos_by_grain=pos_by_grain,
                                   ome_tol=1.0 if ome_tol is None else ome_tol,
                                   eta_tol=1.0 if eta_tol is None else eta_tol)
    else:
        raise ValueError(f"pair_source must be 'peaks' or 'grains', got {pair_source!r}")
    if len(pairs) < min_pairs:
        raise RuntimeError(
            f"{len(pairs)} Friedel pairs found (need >= {min_pairs}). Too few grains or "
            "spots, a partial omega range (pairs need 180 deg), or the wrong parameter file.")
    lsd = float(v1.Lsd)
    gstep = abs(step)
    cls_all, grid_phase, snapped_frac = snapped_members(pairs, ome_ref, gstep)
    x, sig, n, boots, info = fit_wedge_omega_scale(
        pairs, lsd_um=lsd, ome_ref=ome_ref, fit_omega_scale=fit_omega_scale,
        wedge0=0.0, max_cot=max_cot, n_boot=n_boot, seed=seed, loss=loss,
        omega_step=gstep, cls=cls_all, return_info=True)
    se = boots.std(axis=0) if boots.size else np.array([np.nan, np.nan])
    cmask = _cot_mask(pairs, max_cot)
    used = pairs.subset(cmask)
    cls_used = cls_all[cmask]
    res = FriedelWedgeResult(
        wedge_deg=float(x[0]), wedge_se_deg=float(se[0]),
        omega_scale_eps=float(x[1]), omega_scale_se=float(se[1]) if fit_omega_scale else 0.0,
        omega_step_in=step, omega_step_fit=step * (1.0 + float(x[1])),
        n_pairs=int(n), n_grains=(int(np.unique(used.grain).size) if pair_source == "grains" else 0),
        sigma_deg=sig, pair_source=pair_source, chance_pair_fraction=chance,
        estimator=loss, snapped_member_fraction=float(snapped_frac),
        grid_phase_frames=float(grid_phase),
        class_counts=dict(info.get("class_counts", {})) if loss != "huber" else
        {c: int((cls_used == c).sum()) for c in (0, 1, 2)},
        class_sigma_deg=dict(info.get("class_sigma_deg", {})))
    p_both = float((cls_used == 2).mean()) if cls_used.size else 0.0
    if loss != "huber" and snapped_frac >= 0.08 and p_both > 0.30:
        cs = res.class_sigma_deg
        v0 = cs.get(0, float("nan")) ** 2
        n_eff = sum(res.class_counts.get(c, 0) * (v0 / cs[c] ** 2) for c in (0, 1, 2)
                    if c in cs and cs[c] > 0) if cs else float("nan")
        res.warnings.append(
            f"omega QUANTISATION: {100 * snapped_frac:.0f} % of pair members sit on the "
            f"{gstep:g} deg frame grid and {100 * p_both:.0f} % of pairs have BOTH on it "
            f"(single-frame spots report the frame label, rms rounding step/sqrt(12) = "
            f"{gstep / math.sqrt(12):.3f} deg). The wedge is constrained mostly by the "
            f"other pairs (effective ~{n_eff:.0f} sub-frame-equivalent pairs of {len(used)}); "
            "read the per-class wedges in the splits. Finer omega (smaller step) or sub-frame "
            "omegas would tighten it.")
    if loss == "huber" and snapped_frac >= 0.08:
        res.warnings.append(
            f"legacy Huber loss with {100 * snapped_frac:.0f} % of omegas on the frame grid: "
            "biased toward 0 (MAD scale ~ the sub-frame scatter, lattice-valued residuals "
            "read as outliers). Use loss='quant'.")
    if np.isfinite(chance) and chance > 0.10:
        res.warnings.append(
            f"{100 * chance:.1f} % of the peak-list pairs are expected to be chance "
            "coincidences (crowded pattern): tighten --ome-tol/--eta-tol or use "
            "--pairs grains.")

    if splits:
        # Wedge per subset with eps held at the global value: a single fixed axis
        # tilt gives ONE wedge in every subset. (Refitting eps inside a one-sided
        # subset -- all cot > 0, say -- makes it collinear with W.)
        cot = 1.0 / np.tan(np.radians(used.eta_a))
        defs = {"cot>0 (one side of beam)": cot > 0, "cot<0 (other side)": cot < 0,
                "|cot|<=1": np.abs(cot) <= 1, "|cot|>1": np.abs(cot) > 1}
        defs["omega_a first quarter"] = used.ome_a < ome_ref + 90
        defs["omega_a second quarter"] = (used.ome_a >= ome_ref + 90) & (used.ome_a < ome_ref + 180)
        for r in np.unique(used.ring):
            defs[f"ring {int(r)}"] = used.ring == r
        if loss != "huber" and snapped_frac >= 0.08:
            for c in (0, 1, 2):
                defs[f"snapped members = {c}"] = cls_used == c
        for name, m in defs.items():
            if m.sum() < 100:
                continue
            sub = used.subset(m)
            xs, _, ns, bs = fit_wedge_omega_scale(
                sub, lsd_um=lsd, ome_ref=ome_ref, fit_omega_scale=False,
                wedge0=x[0], eps0=x[1], max_cot=np.inf, n_boot=min(n_boot, 30),
                seed=seed, loss=loss, omega_step=gstep, cls=cls_used[m])
            bse = float(bs[:, 0].std()) if bs.size else float("nan")
            res.splits[name] = (float(xs[0]), bse, int(ns))
        vals = np.array([v[0] for v in res.splits.values()])
        if vals.size and np.ptp(vals) > max(5 * res.wedge_se_deg, 0.25 * abs(res.wedge_deg), 0.005):
            res.warnings.append(
                f"split wedges span {vals.min():+.4f}..{vals.max():+.4f} deg: not one fixed "
                "axis tilt, or another systematic in omega. Read the splits before using it.")
    if fit_omega_scale and abs(res.omega_scale_eps) > 5 * max(res.omega_scale_se, 1e-9):
        res.warnings.append(
            f"OmegaStep looks off by {res.omega_scale_eps:+.2e} (relative): {step:.6f} -> "
            f"{res.omega_step_fit:.6f} deg/frame. Over the sweep that is "
            f"{abs(res.omega_scale_eps) * 180:.3f} deg about the middle.")
    if out_paramstest is not None:
        out = Path(out_paramstest)
        out.write_text(corrected_params_text(
            paramstest.read_text(), wedge_deg=res.wedge_deg,
            omega_step=res.omega_step_fit if (write_omega_step and fit_omega_scale) else None))
        res.paramstest_out = out
    return res
