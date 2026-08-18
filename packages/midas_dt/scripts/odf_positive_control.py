#!/usr/bin/env python
"""POSITIVE CONTROL: can this pipeline recover a KNOWN texture at YOUR data's SNR?

**Run this before believing any texture map.** A per-voxel ODF fit that returns a
null has two readings and only a simulation separates them:

  (a) the sample really has no coherent texture, or
  (b) the pipeline cannot recover texture at this peak-to-background ratio, in
      which case the null says nothing about the sample.

So plant a texture, push it through a realistic forward model, run the **same**
extraction (:mod:`midas_dt.azimuthal`) and the **same** fit
(:mod:`midas_dt.odf_uniaxial`), and sweep the contrast.

**No inverse crime.** The forward path draws **discrete crystallites** from a fibre
distribution and bins their {hkl} plane normals into azimuth -- no Legendre
polynomials, no squared modulus, nothing the fit uses. Only the geometry is
shared. Peaks are then laid down as Gaussians of a measured FWHM on a background at
controlled contrast, and Poisson noise is applied to the total, so the extraction
faces the same small-difference-of-large-numbers problem the real data poses.

The plant carries genuine **voxel-scale** content (two regions plus per-voxel
random modulation), not a smooth polynomial. A smooth plant would be recovered by
a smooth instrumental artefact just as well, and the polynomial-vs-per-voxel
confusion has already retracted a result in this project three times.

  CONFIRM : recovered S tracks planted S at high contrast, and degrades as
            peak/bg falls. If it fails at YOUR data's measured contrast, your null
            is an SNR limit and says nothing about the sample.
  REFUTE  : recovery fails even at high contrast => the pipeline is broken, any
            null is uninterpretable, and the fault is ours.

**Known result on the DAC Ti geometry** (hcp, five vetted rings; measured
2026-08-18, and it supersedes an earlier, stronger claim):

* **Detection is robust.** The global rung recovers planted texture at **23-34 %**
  residual improvement across peak/bg 0.02-0.5. Real Ti gave **0.17 %** -- a gap of
  more than 100x -- so a global null on that data is a statement about the sample.
* **Per-voxel resolution is weak and NON-MONOTONIC in SNR**: ``|corr|`` runs
  0.23-0.67, peaking near peak/bg = 0.1 and falling at both higher and lower
  contrast. At high SNR the fit chases an azimuthal shape four parameters cannot
  represent -- the plant is a discrete-crystallite fibre distribution, not a
  squared-modulus expansion -- so model mismatch, not noise, is the limit there.
  Noise was regularising it.

An earlier note recorded per-voxel recovery at corr +0.60 to +0.75 holding to
peak/bg 0.02. That came from a forward model which subtracted an **exactly known**
background, plus a fit capped at 40 iterations with a numerical Jacobian, and it
does not reproduce under either the realistic background or a converged fit. The
``--background`` switch below exists so the two are never conflated again.

**Read the softener with any result.** Even the ``estimated`` arm lays peaks on a
**flat** synthetic background. Real backgrounds drift frame to frame, and that
error is systematic rather than Poisson, so it does not average down with more
frames. Any bound from this sweep is optimistic.

**A sign trap, recorded because it cost a false alarm.** The recovered quantity is
the **pole-figure** order parameter, not the crystal-axis one. Prism-plane normals
in hcp are perpendicular to c, so a c-axis fibre appears as a *negative*
pole-figure S. Scoring recovered pole-figure S against planted c-axis S reads
corr = -0.75 and looks like total failure. This script scores against ``S_pole``,
computed from the same binned normals the fit sees, and reports both.

Usage::

    KMP_DUPLICATE_LIB_OK=TRUE python odf_positive_control.py \\
        --geometry axial --contrasts 0.5 0.2 0.1 0.05 0.02 --out control.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from midas_dt.azimuthal import (
    area_and_centroid,
    background_from_ring_free,
    ring_free_mask,
    ring_windows,
    snr_per_eta,
)
from midas_dt.odf_uniaxial import (
    UniaxialODFModel,
    fibre_cos_theta,
    fit_uniaxial_ladder,
    hermans_parameter,
)

log = logging.getLogger("odf_positive_control")

# alpha-Ti rings used for the DAC Ti control, in degrees 2theta
RING_2TH = np.array([3.938, 4.492, 5.847, 6.823, 7.880])
N_L = 4
N_ETA_OUT = 50
FWHM_PX = 8.0
N_CRYST = 4000
VOX_MM, NGRID, N_TRANS, N_OMEGA = 0.05, 11, 11, 12
N_R, R0, R_STEP = 900, 100.0, 1.0

# Cartesian {hkl} normals per ring for hcp, c/a ~ 1.587. Written out rather than
# derived so this control shares as little machinery as possible with the fit.
HKL_C = {
    0: np.array([[1., 0, 0], [0, 1, 0], [-1, 1, 0], [-1, 0, 0], [0, -1, 0],
                 [1, -1, 0]]),
    1: np.array([[1., 0, .6], [0, 1, .6], [-1, 1, .6], [-1, 0, .6], [0, -1, .6],
                 [1, -1, .6]]),
    2: np.array([[1., 0, 1.2], [0, 1, 1.2], [-1, 0, -1.2], [0, -1, -1.2]]),
    3: np.array([[1., 1, 0], [-1, 2, 0], [-2, 1, 0]]),
    4: np.array([[2., 0, 0], [0, 2, 0], [-2, 2, 0]]),
}


def _align_to(vectors: np.ndarray, axis) -> np.ndarray:
    """Rotate ``vectors`` (given about +z) onto ``axis``."""
    ax = np.asarray(axis, dtype=float)
    ax = ax / np.linalg.norm(ax)
    a0 = np.array([0.0, 0.0, 1.0])
    v, c = np.cross(a0, ax), float(a0 @ ax)
    if np.linalg.norm(v) < 1e-12:
        return vectors if c > 0 else -vectors
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return vectors @ (np.eye(3) + vx + vx @ vx / (1.0 + c)).T


def fibre_orientations(n: int, axis, spread_deg: float, rng) -> np.ndarray:
    """``n`` crystallite orientation matrices with c clustered about ``axis``.

    Discrete crystallites, drawn directly. This is the whole point of the control,
    so nothing in here may touch the basis the fit uses.

    Built by explicit frame construction rather than ``Rotation.align_vectors`` in
    a loop: the loop version costs ~275k scipy calls for a realistic crystallite
    count, which made the control too slow to run at the counts it needs (and a
    control nobody runs is worse than no control). Column 3 of each matrix is the
    crystallite's c axis, and the frame is then spun about it uniformly -- the same
    construction the loop performed, checked in
    :func:`_check_fibre_orientations`.
    """
    cs = np.cos(np.radians(spread_deg))
    z = rng.uniform(cs, 1.0, n)
    ph = rng.uniform(0, 2 * np.pi, n)
    s = np.sqrt(np.maximum(1 - z * z, 0))
    cvec = _align_to(np.stack([s * np.cos(ph), s * np.sin(ph), z], 1), axis)
    cvec /= np.linalg.norm(cvec, axis=1, keepdims=True)

    # an in-plane reference perpendicular to c, avoiding the degenerate choice
    ref = np.tile(np.array([1.0, 0.0, 0.0]), (n, 1))
    near_x = np.abs(cvec[:, 0]) > 0.9
    ref[near_x] = np.array([0.0, 0.0, 1.0])
    a1 = np.cross(ref, cvec)
    a1 /= np.linalg.norm(a1, axis=1, keepdims=True)
    a2 = np.cross(cvec, a1)

    spin = rng.uniform(0, 2 * np.pi, n)[:, None]
    e1 = a1 * np.cos(spin) + a2 * np.sin(spin)
    e2 = -a1 * np.sin(spin) + a2 * np.cos(spin)
    return np.stack([e1, e2, cvec], axis=2)        # columns: e1, e2, c


def _check_fibre_orientations() -> None:
    """Assert the vectorised construction really produces rotations with c = col 3."""
    rng = np.random.default_rng(0)
    m = fibre_orientations(500, [0, 0, 1], 30.0, rng)
    assert np.allclose(np.einsum("kij,kjl->kil", m, m.transpose(0, 2, 1)),
                       np.eye(3), atol=1e-9), "not orthonormal"
    assert np.allclose(np.linalg.det(m), 1.0, atol=1e-9), "not proper rotations"
    cvec = m[:, :, 2]
    assert (cvec[:, 2] >= np.cos(np.radians(30.0)) - 1e-9).all(), "c not clustered"
    # the spin about c must be uniform: the in-plane component of column 1 has no
    # preferred azimuth
    az = np.arctan2(m[:, 1, 0], m[:, 0, 0])
    assert abs(float(np.mean(np.cos(2 * az)))) < 0.15, "spin is not uniform"


def pole_figure(mats: np.ndarray, hkl_c: np.ndarray, eta_edges: np.ndarray,
                th_rad: float) -> np.ndarray:
    """Bin {hkl} normals of discrete crystallites into azimuth for one ring.

    A normal diffracts when ``n . xhat = -sin(theta)``; at small theta the
    accessible set is close to a great circle, and eta is its azimuth about the
    beam. The tolerance widens if too few normals qualify, so a sharp texture does
    not silently produce an empty pole figure.
    """
    ns = np.einsum("kab,mb->kma", mats, hkl_c).reshape(-1, 3)
    ns /= np.linalg.norm(ns, axis=1, keepdims=True)
    ok = np.abs(ns[:, 0] + np.sin(th_rad)) < 0.06
    if ok.sum() < 20:
        ok = np.abs(ns[:, 0] + np.sin(th_rad)) < 0.15
    eta = np.arctan2(ns[ok, 2], ns[ok, 1])
    return np.histogram(eta, bins=eta_edges)[0].astype(float)


def plant_quality(PF: np.ndarray) -> tuple[float, float]:
    """(median counts per azimuthal bin, implied relative Poisson noise).

    The control's own noise floor, and it must be reported. Only normals within a
    narrow band of the diffraction condition land in a pole figure -- roughly 6 %
    of them -- so a modest crystallite count leaves single-digit counts per bin
    and a *planted* pole figure that is 40 % noise. Recovery then fails because
    the plant is noise, not because the pipeline is broken, and a control that
    cannot tell those apart is worthless.
    """
    per_bin = float(np.median(PF[PF > 0])) if (PF > 0).any() else 0.0
    return per_bin, (1.0 / np.sqrt(per_bin) if per_bin > 0 else float("inf"))


def plant(geometry: str, rng, n_cryst: int = N_CRYST):
    """Plant a per-voxel fibre texture and return its pole figures and true S."""
    c = (np.arange(NGRID) - (NGRID - 1) / 2.0) * VOX_MM
    X, Y = np.meshgrid(c, c, indexing="ij")
    inside = (X ** 2 + Y ** 2) <= (0.45 * NGRID * VOX_MM) ** 2
    vox = np.argwhere(inside)
    n_vox = len(vox)

    # Two regions PLUS per-voxel modulation: genuine voxel-scale content, not a
    # smooth field a polynomial (or an absorption gradient) could imitate.
    spread = np.where(X[inside] > 0, 20.0, 45.0)
    spread = np.clip(spread * (1.0 + 0.25 * rng.standard_normal(n_vox)), 8.0, 80.0)

    axis = [0, 0, 1] if geometry == "axial" else [0, 1, 0]
    eta_edges = np.linspace(-np.pi, np.pi, N_ETA_OUT + 1)
    th = np.radians(RING_2TH) / 2.0
    fam = {k: v / np.linalg.norm(v, axis=1, keepdims=True)
           for k, v in HKL_C.items()}

    PF = np.zeros((n_vox, N_ETA_OUT, len(RING_2TH)))
    cos2 = np.zeros(n_vox)
    for v in range(n_vox):
        mats = fibre_orientations(n_cryst, axis, spread[v], rng)
        cvec = mats[:, :, 2]
        cos2[v] = float(np.mean((cvec @ np.asarray(axis, float)) ** 2))
        for ri in range(len(RING_2TH)):
            PF[v, :, ri] = pole_figure(mats, fam[ri], eta_edges, th[ri])

    S_axis = 0.5 * (3 * cos2 - 1)                  # planted CRYSTAL-AXIS S
    # planted POLE-FIGURE S, from the same binned normals the fit will see. This
    # is the quantity the fit estimates, and it can differ in SIGN from S_axis.
    eta_c = 0.5 * (eta_edges[:-1] + eta_edges[1:])
    cosT = fibre_cos_theta(RING_2TH, eta_c)        # (n_eta, n_ring)
    w = PF / np.maximum(PF.sum(axis=1, keepdims=True), 1e-9)
    S_pole = 0.5 * (3 * np.einsum("ver,er->v", w, cosT ** 2) / len(RING_2TH) - 1)

    raw_counts = PF.copy()             # kept for plant_quality, before normalising
    PF /= np.maximum(PF.mean(axis=1, keepdims=True), 1e-9)
    xy = (vox - (NGRID - 1) / 2.0) * VOX_MM
    return PF, S_axis, S_pole, xy, eta_c, spread, raw_counts


def build_rays(n_vox: int, xy: np.ndarray):
    """Boolean (n_ray, n_vox) projection operator over translations x omega."""
    om = np.linspace(0, np.pi, N_OMEGA, endpoint=False)
    hxz = (np.arange(N_TRANS) - (N_TRANS - 1) / 2.0) * VOX_MM
    edges = np.concatenate([hxz - VOX_MM / 2, [hxz[-1] + VOX_MM / 2]])
    rays = np.zeros((N_TRANS * N_OMEGA, n_vox))
    for j, w in enumerate(om):
        proj = xy[:, 0] * np.sin(w) + xy[:, 1] * np.cos(w)
        ti = np.digitize(proj, edges) - 1
        for v, t in enumerate(ti):
            if 0 <= t < N_TRANS:
                rays[t * N_OMEGA + j, v] = 1.0
    keep = rays.sum(axis=1) > 0
    return rays[keep]


def synthesise_known_background(projected: np.ndarray, contrast: float, rng):
    """Poisson noise on an EXACTLY KNOWN background pedestal, subtracted exactly.

    The optimistic arm. There is no background *model* here: a flat pedestal is
    added and the same constant is subtracted, so the only error is photon
    counting. Run it to separate the two error sources -- the gap between this and
    :func:`synthesise_and_extract` **is** the background-model error, which is the
    term that dominates real low-contrast data and does not average down.

    Use it also as a regression check: this is the forward model the original DAC
    Ti control used, so it is what its published recovery figures refer to.
    """
    n_ray, n_eta, n_ring = projected.shape
    bg, npx = 220.0, FWHM_PX
    peak = contrast * bg
    counts = rng.poisson(bg * npx + peak * npx * projected
                         / max(float(projected.mean()), 1e-9))
    area = counts - bg * npx
    sig = np.sqrt(np.maximum(counts, 1.0))
    return area, sig


def synthesise_and_extract(projected: np.ndarray, contrast: float, rng):
    """Lay peaks on a background at ``contrast``, add Poisson noise, extract areas.

    The realistic arm, and the default. ``projected`` is
    ``(n_ray, n_eta, n_ring)`` relative intensity. Returns extracted areas and
    their uncertainties, having gone through the SAME background model, window
    choice and SNR gate as real data -- so it carries **background-model error**
    on top of photon noise, which is the point.
    """
    n_ray, n_eta, n_ring = projected.shape
    r_axis = R0 + R_STEP * np.arange(N_R)
    sigma = FWHM_PX / 2.3548
    # place the rings across the radial axis, well separated
    centres = np.linspace(R0 + 120.0, R0 + N_R * R_STEP - 120.0, n_ring)
    idx, half = ring_windows(r_axis, centres, max_half_px=16.0)
    mask = ring_free_mask(N_R, idx, half)
    bg_level, bg_slope = 220.0, -0.05

    area = np.zeros((n_ray, n_eta, n_ring))
    sig = np.zeros_like(area)
    for k in range(n_ray):
        bg = (bg_level + bg_slope * (r_axis - R0))[:, None] * np.ones((1, n_eta))
        ideal = np.zeros((N_R, n_eta))
        for ri, cen in enumerate(centres):
            local = bg_level + bg_slope * (cen - R0)
            amp = contrast * local * projected[k, :, ri]      # (n_eta,)
            ideal += amp[None, :] * np.exp(
                -0.5 * ((r_axis[:, None] - cen) / sigma) ** 2)
        cake = rng.poisson(np.clip(bg + ideal, 0, None)).astype(float)
        net, _ = background_from_ring_free(cake, mask, block_bins=30)
        for ri, cen in enumerate(centres):
            a, _ = area_and_centroid(net, r_axis, idx[cen], half[cen])
            lo = max(0, idx[cen] - half[cen])
            hi = min(N_R, idx[cen] + half[cen] + 1)
            s = snr_per_eta(cake[lo:hi], net[lo:hi])
            area[k, :, ri] = a
            # uncertainty from the measured SNR, floored so a dead bin is
            # down-weighted rather than infinitely trusted
            sig[k, :, ri] = np.abs(a) / np.maximum(s, 0.1)
    return area, sig


def run_contrast(projected, S_pole, xy, contrast, rng, background="estimated"):
    """One rung of the sweep: synthesise, extract, fit, score."""
    if background == "known":
        area, sig = synthesise_known_background(projected, contrast, rng)
    else:
        area, sig = synthesise_and_extract(projected, contrast, rng)
    n_ray, n_eta, n_ring = area.shape
    good = (area > 0) & (sig > 0) & np.isfinite(area) & np.isfinite(sig)
    scale = float(np.nanmedian(area[good]))
    data = np.where(good, area / scale, 0.0).reshape(n_ray, -1)
    weights = np.where(good, scale / np.maximum(sig, 1e-12), 0.0).reshape(n_ray, -1)
    weights /= max(float(np.nanmedian(weights[weights > 0])), 1e-12)

    eta_c = np.linspace(-np.pi, np.pi, n_eta, endpoint=False) + np.pi / n_eta
    from midas_dt.odf_uniaxial import uniaxial_design
    design = uniaxial_design(fibre_cos_theta(RING_2TH, eta_c), N_L).reshape(-1, N_L)

    rays = build_rays(len(xy), xy)
    model = UniaxialODFModel(design, rays, good.reshape(n_ray, -1)[:len(rays)],
                             data[:len(rays)], weights[:len(rays)])
    res = fit_uniaxial_ladder(model)
    S_rec = res.hermans_S
    corr = float(np.corrcoef(S_rec, S_pole)[0, 1])
    return {
        "contrast": contrast,
        "background": background,
        "improvement_pct": res.improvement_pct,
        "global_improvement_pct": res.global_improvement_pct,
        "corr_S_pole": corr,
        "abs_corr": abs(corr),
        "median_S_recovered": float(np.median(S_rec)),
        "median_S_pole": float(np.median(S_pole)),
        "verdict": res.verdict(xy),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--geometry", choices=("axial", "radial"), default="axial",
                    help="axial: fibre along the rotation axis, which is what the "
                         "fit assumes. radial: perpendicular, i.e. the fit is the "
                         "WRONG model -- run both, see the note below.")
    ap.add_argument("--contrasts", type=float, nargs="+",
                    default=[0.50, 0.20, 0.10, 0.05, 0.02],
                    help="peak/background values to sweep. Include YOUR data's "
                         "measured contrast.")
    ap.add_argument("--background", choices=("estimated", "known", "both"),
                    default="both",
                    help="'estimated' runs the real extraction chain, so the "
                         "background must be MODELLED (realistic, and the default "
                         "half of 'both'). 'known' subtracts an exact pedestal, "
                         "leaving Poisson noise only (optimistic). The GAP between "
                         "them is the background-model error -- the term that "
                         "dominates real low-contrast data.")
    ap.add_argument("--n-crystallites", type=int, default=N_CRYST,
                    help="per voxel. Watch the reported plant noise: only ~6%% of "
                         "normals land near the diffraction condition, so a low "
                         "count makes the PLANT the limiting factor.")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=Path, default=None, help="write JSON results here")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    t0 = time.time()
    rng = np.random.default_rng(args.seed)
    _check_fibre_orientations()

    print(f"planting {args.geometry} fibre texture "
          f"({args.n_crystallites} crystallites/voxel)...", flush=True)
    PF, S_axis, S_pole, xy, eta_c, spread, raw = plant(
        args.geometry, rng, args.n_crystallites)
    n_vox = len(xy)
    print(f"  {n_vox} voxels, {len(RING_2TH)} rings, {N_ETA_OUT} azimuths "
          f"[{time.time() - t0:.0f}s]")

    per_bin, plant_noise = plant_quality(raw)
    print(f"  plant quality: {per_bin:.0f} normals per azimuthal bin "
          f"=> {plant_noise * 100:.0f}% Poisson noise on the PLANTED pole figure")
    if plant_noise > 0.15:
        print(f"  *** THE PLANT ITSELF IS NOISY. Raise --n-crystallites "
              f"(need ~{int(args.n_crystallites * (plant_noise / 0.15) ** 2)} for "
              f"15% noise) or lower --n-eta. A failure at this plant quality says "
              f"nothing about the pipeline. ***")
    print(f"  planted crystal-axis S : median {np.median(S_axis):+.3f}  "
          f"range {S_axis.min():+.3f}..{S_axis.max():+.3f}")
    print(f"  planted pole-figure S  : median {np.median(S_pole):+.3f}  "
          f"range {S_pole.min():+.3f}..{S_pole.max():+.3f}   <- scored against this")
    if np.sign(np.median(S_axis)) != np.sign(np.median(S_pole)):
        print("  NOTE: the two have OPPOSITE signs, as expected for prism normals "
              "perpendicular to c. Scoring against the axis S would read as failure.")

    rays = build_rays(n_vox, xy)
    projected = np.einsum("kv,ver->ker", rays, PF)
    # Normalise by ONE GLOBAL scalar, never per ray. Per-ray normalisation was the
    # first version and it silently broke the control: the fit's forward model
    # predicts a ray sum proportional to how many voxels that ray crosses, so
    # rescaling each ray to mean 1 destroys exactly the additivity the model
    # assumes. The per-voxel a_0 then absorbs the mismatch, coupling voxels along
    # every ray and corrupting the a_2..a_6 recovery -- global improvement still
    # looked healthy at 32 %, while per-voxel correlation collapsed to 0.15.
    projected /= max(float(projected.mean()), 1e-9)

    modes = (["known", "estimated"] if args.background == "both"
             else [args.background])
    rows = []
    print(f"\n{'background':>11} {'peak/bg':>9} {'impr %':>9} {'global %':>9} "
          f"{'corr(S_rec,S_pole)':>20} {'|corr|':>8}")
    for mode in modes:
        for contrast in args.contrasts:
            row = run_contrast(projected, S_pole, xy, contrast, rng,
                               background=mode)
            rows.append(row)
            print(f"{mode:>11} {contrast:>9.3f} {row['improvement_pct']:>9.2f} "
                  f"{row['global_improvement_pct']:>9.2f} "
                  f"{row['corr_S_pole']:>20.3f} {row['abs_corr']:>8.3f}", flush=True)

    est = [r for r in rows if r["background"] == "estimated"]
    kno = [r for r in rows if r["background"] == "known"]
    if est and kno:
        b_est = max(r["abs_corr"] for r in est)
        b_kno = max(r["abs_corr"] for r in kno)
        print(f"\nBACKGROUND-MODEL COST: best |corr| falls {b_kno:.2f} -> {b_est:.2f} "
              f"when the background must be estimated rather than known.")
        print("  That gap is the term real low-contrast data is dominated by, and it "
              "does NOT average down with more frames the way Poisson noise does. "
              "Any bound quoted from the 'known' arm alone is optimistic.")

    # TWO DISTINCT CLAIMS, scored separately. Collapsing them into one number is
    # how an over-strong conclusion gets drawn: a pipeline can reliably DETECT a
    # planted texture while being unable to RESOLVE it per voxel, and those two
    # support completely different statements about a real null.
    scored = est if est else rows
    best_detect = max(r["global_improvement_pct"] for r in scored)
    best_resolve = max(r["abs_corr"] for r in scored)
    resolve_floor = min((r["contrast"] for r in scored if r["abs_corr"] > 0.5),
                        default=None)
    print("\n" + "=" * 72)
    if plant_noise > 0.15:
        print(f"INCONCLUSIVE: the PLANT carries {plant_noise * 100:.0f}% noise, so "
              "it is the limiting factor, not the pipeline.")
        print("  Raise --n-crystallites and re-run before drawing any conclusion.")
        verdict = "INCONCLUSIVE"
    else:
        print(f"Plant is clean ({plant_noise * 100:.0f}% noise). Two separate "
              "claims:\n")
        detect_ok = best_detect > 10.0
        print(f"  1. DETECT a planted texture at all "
              f"(global rung, {best_detect:.1f}% best improvement over null): "
              f"{'YES' if detect_ok else 'NO'}")
        print(f"  2. RESOLVE it PER VOXEL "
              f"(best |corr(S_rec, S_pole)| = {best_resolve:.2f}"
              + (f", above 0.5 down to peak/bg {resolve_floor:.3f}"
                 if resolve_floor is not None else ", never above 0.5")
              + f"): {'YES' if best_resolve > 0.5 else 'NO'}")
        if detect_ok and best_resolve > 0.5:
            verdict = "CONFIRM"
            print("\n  => Both hold. A real null below the per-voxel floor is an SNR "
                  "limit; above it, the null is about the sample.")
        elif detect_ok:
            verdict = "DETECT_ONLY"
            print("\n  => The pipeline DETECTS planted texture but does NOT resolve "
                  "it per voxel at this geometry and sampling.")
            print("     So: a null on the GLOBAL rung is interpretable as a "
                  "statement about the sample. A null on the PER-VOXEL rung is NOT "
                  "-- it is consistent with texture the reconstruction cannot "
                  "localise. Report a sample-average bound, not a map.")
        else:
            verdict = "REFUTE"
            print("\n  => The pipeline cannot even detect a planted texture. ANY null "
                  "from it is uninterpretable. Fix this before touching real data.")
    print("\n  Softener: even the 'estimated' arm carries only a FLAT synthetic "
          "background. Real backgrounds drift frame to frame, so this remains "
          "optimistic.")

    if args.out:
        args.out.write_text(json.dumps({
            "geometry": args.geometry,
            "n_crystallites": args.n_crystallites,
            "seed": args.seed,
            "n_voxels": n_vox,
            "rings_2theta_deg": RING_2TH.tolist(),
            "planted_S_axis_median": float(np.median(S_axis)),
            "planted_S_pole_median": float(np.median(S_pole)),
            "verdict": verdict,
            "plant_noise_frac": plant_noise,
            "best_detect_global_pct": best_detect if plant_noise <= 0.15 else None,
            "best_resolve_abs_corr": best_resolve if plant_noise <= 0.15 else None,
            "per_voxel_floor_contrast": resolve_floor,
            "rows": rows,
        }, indent=1))
        print(f"\nwrote {args.out}")
    print(f"[{time.time() - t0:.0f}s]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
