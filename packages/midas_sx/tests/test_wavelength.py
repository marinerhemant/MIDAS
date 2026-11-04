"""Joint wavelength / cell-scale refinement (``refine_wavelength``) in ``orient``.

* default off is inert (the wavelength is untouched, the extra result fields say "not refined");
* positive control with an INDEPENDENT forward model: spots from ``validation/simulate.py``'s own Ewald-crossing root finder (``_crossings``) and its own flat-detector ray projection, at a wavelength
  that differs from the header by +-0.2 %. Only the frame convention (``frame = (omega - omega_start)/step - 0.5``) is shared with the package; a +-0.2 % wavelength error is mostly but not fully absorbed by
  the distance, centre and rotation (the refinement without eps leaves a 25 % larger robust frame residual);
* the same with the package's own forward model and the wedge, the tilts and a wrong centre refined at the same time;
* a case where eps is not determined (a 10 deg omega range) is not reported as converged;
* the refined wavelength is what the downstream hkl assignment sees.
The parameter is a joint wavelength / cell scale (with a fixed cell eps is equivalent to a (1 - eps) scale of the reference cell): ``test_a_wrong_isotropic_cell_is_absorbed_as_the_opposite_scale``."""
from __future__ import annotations

import os
import sys
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from scipy.spatial.transform import Rotation as R

import simulate as sim                                              # noqa: E402

from midas_hkls import Lattice
from midas_sx.assign import assign_hkl, ewald_crossings, reciprocal_basis
from midas_sx.geometry import Scan, make_geometry, spot_directions
from midas_sx.orient import _unpack, index_and_refine, predict_positions

LAT = Lattice(3.83, 3.83, 20.5, 90, 90, 90)
LAM0 = 0.41328
U_TRUE = R.random(random_state=np.random.RandomState(4)).as_matrix()


def _independent_spots(eps, om_lo, om_hi, step=0.25, noise=(0.25, 0.2), seed=0):
    """Spots at wavelength LAM0 (1 + eps) from the simulator's own root finder and ray projection (centre (255.5, 255.5), 100 mm, 0.3 mm pixels, no tilt, no wedge)."""
    exp = sim.Experiment(wavelength_A=LAM0 * (1 + eps), omega_start=om_lo, omega_step=step, n_frames=int(round((om_hi - om_lo) / step)))
    B = reciprocal_basis(LAT)
    rows = []
    for h in range(-3, 4):
        for k in range(-3, 4):
            for l in range(-18, 19):
                if (h, k, l) == (0, 0, 0):
                    continue
                q0 = U_TRUE @ (B @ np.array([h, k, l], float))
                for om, slope in sim._crossings(q0, exp):
                    if slope < 1e-3:
                        continue
                    v = exp.k * sim.S0 + R.from_rotvec(np.radians(om) * sim.AXIS).apply(q0)
                    s = v / np.linalg.norm(v)
                    if s[0] <= 0.05:
                        continue
                    col, row = exp.centre_rc[1] - s[1] * exp.dist_mm / s[0] / exp.pixel_mm, exp.centre_rc[0] + s[2] * exp.dist_mm / s[0] / exp.pixel_mm
                    if 6 < row < 505 and 6 < col < 505 and abs(row - 255.5) + abs(col - 255.5) > 14:
                        rows.append((h, k, l, (om - exp.omega_start) / exp.omega_step - 0.5, row, col))
    d = pd.DataFrame(rows, columns=["h", "k", "l", "frame", "row", "col"])
    rng = np.random.default_rng(seed)
    d["row"] += rng.normal(0, noise[0], len(d)); d["col"] += rng.normal(0, noise[0], len(d)); d["frame"] += rng.normal(0, noise[1], len(d))
    scan = Scan(om_lo, step, exp.n_frames)
    g0 = make_geometry(wavelength_A=LAM0, dist_mm=100.0, pixel_mm=0.3, centre_rc=(255.5, 255.5), shape=(512, 512), scan=scan)
    return d, scan, g0


def _mis_deg(U1, U2):
    return float(np.degrees(np.linalg.norm(R.from_matrix(U1 @ U2.T).as_rotvec())))


U_START = R.from_rotvec([0.002, -0.003, 0.002]).as_matrix() @ U_TRUE


def test_default_off_leaves_the_wavelength_alone_and_says_so():
    d, scan, g0 = _independent_spots(0.0, -40.0, 40.0)
    res = index_and_refine(d, LAT, U_START, g0, scan)
    assert res.converged, res.reason
    assert res.geom.wavelength_A == g0.wavelength_A                         # bit-identical: not merely close
    assert res.lambda_cell_scale == 0.0 and np.isnan(res.sigma_lambda_cell_scale)
    p10 = np.array([0.001, -0.002, 0.003, 0.5, -0.4, 0.002, 0.1, -0.1, 0.2, 0.3])
    U10, g10 = _unpack(p10, U_START, g0)
    U11, g11 = _unpack(np.append(p10, 0.0), U_START, g0)
    assert np.array_equal(U10, U11) and g10 == g11                          # the 10-parameter vector of the earlier code is the 11-parameter one with eps = 0


@pytest.mark.parametrize("eps_true", [2.0e-3, -2.0e-3, 0.0])
def test_independent_forward_model_recovers_an_injected_wavelength_error(eps_true):
    d, scan, g0 = _independent_spots(eps_true, -40.0, 40.0)
    res = index_and_refine(d, LAT, U_START, g0, scan, refine_wavelength=True)
    assert res.converged, res.reason
    sig = res.sigma_lambda_cell_scale
    assert 3e-5 < sig < 4e-4                                                # formal sigma of the right size for 0.25 px / 0.2 frame noise on ~750 spots (measured 1.3e-4)
    assert abs(res.lambda_cell_scale - eps_true) < 3.5 * sig and abs(res.lambda_cell_scale - eps_true) < 4e-4
    assert res.geom.wavelength_A == pytest.approx(LAM0 * (1 + res.lambda_cell_scale), rel=1e-14)
    assert _mis_deg(res.U, U_TRUE) < 0.01                                   # the orientation is not dragged
    assert abs(res.geom.bcz_px - 255.5) < 0.1 and abs(res.geom.bcy_px - 255.5) < 0.1 and abs(res.geom.lsd_um / 1e5 - 1) < 1e-3
    if abs(eps_true) > 1e-3:                                                # eps is significant, not noise (|eps|/sigma 14-16), although distance and centre absorb most of it in the residual
        assert abs(res.lambda_cell_scale) / sig > 8.0
        plain = index_and_refine(d, LAT, U_START, g0, scan)
        assert plain.rms_frame > res.rms_frame                              # measured 0.265 vs 0.215 (+2e-3) and 0.238 vs 0.217 (-2e-3) frames, noise floor 0.20


def test_the_refined_wavelength_is_what_the_hkl_assignment_downstream_sees():
    d, scan, g0 = _independent_spots(2.0e-3, -40.0, 40.0, noise=(0.0, 0.0))
    res = index_and_refine(d, LAT, U_START, g0, scan, refine_wavelength=True)
    om = scan.omega_of(d.frame.to_numpy())
    s, _, _ = spot_directions(d.row.to_numpy(), d.col.to_numpy(), res.geom)
    r_ref = assign_hkl(s, om, LAT, res.U, res.geom.wavelength_A)[2]
    s0, _, _ = spot_directions(d.row.to_numpy(), d.col.to_numpy(), replace(res.geom, wavelength_A=LAM0))
    r_hdr = assign_hkl(s0, om, LAT, res.U, LAM0)[2]
    assert np.median(r_ref) < 0.3 * np.median(r_hdr), (np.median(r_ref), np.median(r_hdr))


def test_a_ten_degree_omega_range_does_not_determine_eps_and_is_not_reported_as_converged():
    d, scan, g0 = _independent_spots(0.0, -40.0, -30.0)
    assert index_and_refine(d, LAT, U_START, g0, scan).converged           # everything else is fine on these spots
    res = index_and_refine(d, LAT, U_START, g0, scan, refine_wavelength=True)
    assert res.sigma_lambda_cell_scale > 5e-4 and not res.converged and "not determined" in res.reason
    loose = index_and_refine(d, LAT, U_START, g0, scan, refine_wavelength=True, max_sigma_lambda=1e-2)
    assert loose.converged                                                  # the gate is the sigma threshold, nothing else


def test_a_wrong_isotropic_cell_is_absorbed_as_the_opposite_scale():
    """The documented degeneracy: spots from the true cell, a reference cell 0.2 % too large, wavelength header correct. eps comes out +0.2 % (the spots show a smaller cell than the reference), not 0."""
    d, scan, g0 = _independent_spots(0.0, -40.0, 40.0)
    big = Lattice(3.83 * 1.002, 3.83 * 1.002, 20.5 * 1.002, 90, 90, 90)
    res = index_and_refine(d, big, U_START, g0, scan, refine_wavelength=True)
    assert res.converged, res.reason
    assert res.lambda_cell_scale == pytest.approx(2.0e-3, abs=4e-4)


# ---- package forward model, everything refined together
LATW = Lattice(5.8033, 5.9595, 12.2978, 90, 90, 90)
LAMW = 0.6889


def _package_spots(g_true, U, scan, rng, noise_px=0.2, n_max=800):
    rows = []
    for h in range(-4, 5):
        for k in range(-4, 5):
            for l in range(-8, 9):
                if (h, k, l) == (0, 0, 0):
                    continue
                q_s = 2 * np.pi * (U @ reciprocal_basis(LATW) @ np.array([h, k, l], float))
                for w in np.degrees(ewald_crossings(q_s, g_true.wavelength_A, g_true.wedge_deg)):
                    for shift in (0.0, 360.0, -360.0):
                        if min(scan.omega_start, scan.omega_end) <= w + shift <= max(scan.omega_start, scan.omega_end):
                            rows.append((h, k, l, w + shift))
    d = pd.DataFrame(rows, columns=["h", "k", "l", "omega"])
    fr, ro, co, ok = predict_positions(U, LATW, d[["h", "k", "l"]].to_numpy(), d.omega.to_numpy(), g_true, scan)
    d = d.assign(frame=fr, row=ro, col=co)[ok & (ro > 5) & (ro < 1673) & (co > 5) & (co < 1469)]
    d = d.sample(min(len(d), n_max), random_state=0)
    return d.assign(row=d.row + rng.normal(0, noise_px, len(d)), col=d.col + rng.normal(0, noise_px, len(d)), frame=d.frame + rng.normal(0, 0.1, len(d)))


def test_eps_wedge_tilts_and_centre_are_recovered_together():
    rng = np.random.default_rng(5)
    scan = Scan(-150.0, 0.2, 850)
    U = R.from_rotvec([0.3, 0.2, -0.5]).as_matrix()
    eps_true = 1.8e-3
    g_true = replace(make_geometry(wavelength_A=LAMW * (1 + eps_true), dist_mm=171.0, pixel_mm=0.172, centre_rc=(861.9, 401.3), shape=(1679, 1475), scan=scan),
                     wedge_deg=0.30, tz_deg=-20.0 + 0.10, tx_deg=0.15, ty_deg=-0.08)
    d = _package_spots(g_true, U, scan, rng)
    assert len(d) > 300
    g_start = replace(g_true, wavelength_A=LAMW, wedge_deg=0.0, tz_deg=-20.0, tx_deg=0.0, ty_deg=0.0, bcz_px=862.4, bcy_px=401.0, lsd_um=170.3e3)
    U_start = R.from_rotvec([0.0005, -0.0004, 0.0006]).as_matrix() @ U
    res = index_and_refine(d, LATW, U_start, g_start, scan, refine_tilts=True, refine_wedge=True, refine_wavelength=True)
    assert res.converged, res.reason
    sig = res.sigma_lambda_cell_scale
    assert abs(res.lambda_cell_scale - eps_true) < 3.5 * sig and abs(res.lambda_cell_scale - eps_true) < 3e-4
    assert res.geom.wedge_deg == pytest.approx(0.30, abs=0.03)
    assert res.geom.tx_deg == pytest.approx(0.15, abs=0.05) and res.geom.ty_deg == pytest.approx(-0.08, abs=0.05) and res.geom.tz_deg == pytest.approx(-19.9, abs=0.05)
    assert res.geom.lsd_um == pytest.approx(171.0e3, rel=5e-4)
    assert _mis_deg(res.U, U) < 0.02
