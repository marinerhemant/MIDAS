"""Global 1-sigma on a tiled detector, with the gauge stated.

`calibrate()` freezes the globals before the per-(panel, ring) radius stage, so
its Laplace used to be conditional on them and returned no sigma for
Lsd/BC/tilts. `tiled_global_sigma` frees them again together with
`panel_ring_delta_r`. The joint problem has one null mode per ring (a uniform
shift of that ring on every panel is what a change of Lsd does), so a gauge is
required. The tests pin the three things that can go wrong:

1. the number is the right number: it matches an independent hard-constraint
   covariance (null-space projection of J'J/sigma^2, built with autograd on the
   plain parameter vector, no sigmoid reparameterisation, no penalty rows);
2. the gauge is what makes it finite: with the gauge weight at zero the Lsd
   sigma is orders of magnitude larger (otherwise the "gauge" would be decoration);
3. the two gauges are reported, and a quantity the gauge cannot touch (beam
   centre) agrees between them.

Synthetic geometry: 6 azimuthal-sector "panels", 5 rings, a known per-cell
offset field with zero mean per ring, 0.02 px noise on the radii.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from midas_calibrate_v2.forward.bragg import R_ideal_px
from midas_calibrate_v2.inference.tiled_sigma import tiled_global_sigma
from midas_calibrate_v2.loss.pseudo_strain import pseudo_strain_residual
from midas_calibrate_v2.parameters.parameter import Parameter
from midas_calibrate_v2.parameters.spec import CalibrationSpec

F64 = torch.float64
LSD, PX = 1_000_000.0, 200.0
BC_Y, BC_Z = 512.0, 512.0
P, K, N_AZ = 6, 5, 60
GLOBALS = ["Lsd", "BC_y", "BC_z", "ty", "tz"]


class _Fits:
    def __init__(self, seed=0):
        rng = np.random.default_rng(seed)
        tth = torch.linspace(5.0, 11.0, K, dtype=F64)
        r_ideal = R_ideal_px(tth, torch.tensor(LSD, dtype=F64), PX)
        d = rng.normal(0.0, 0.15, (P, K))
        d -= d.mean(axis=0, keepdims=True)                  # the gauge is TRUE here
        self.cell_offsets = torch.tensor(d, dtype=F64)
        Y, Z, T, Kk, Pp = [], [], [], [], []
        for k in range(K):
            phi = torch.linspace(0.0, 2 * math.pi, N_AZ + 1, dtype=F64)[:-1]
            pan = torch.clamp((phi / (2 * math.pi) * P).long(), max=P - 1)
            R = r_ideal[k] + self.cell_offsets[pan, k] \
                + torch.tensor(rng.normal(0.0, 0.02, N_AZ), dtype=F64)
            Y.append(BC_Y + R * torch.cos(phi))
            Z.append(BC_Z + R * torch.sin(phi))
            T.append(tth[k].expand(N_AZ))
            Kk.append(torch.full((N_AZ,), k, dtype=torch.long))
            Pp.append(pan)
        self.Y_pix, self.Z_pix = torch.cat(Y), torch.cat(Z)
        self.ring_two_theta_deg, self.ring_idx = torch.cat(T), torch.cat(Kk)
        self.panel_idx = torch.cat(Pp)
        self.rho_d = torch.tensor(1.0e6, dtype=F64)
        self.weights = None


def _spec(frozen=True):
    s = CalibrationSpec()
    for n, v, b in (("Lsd", LSD, (LSD - 5e4, LSD + 5e4)), ("BC_y", BC_Y, (BC_Y - 20, BC_Y + 20)),
                    ("BC_z", BC_Z, (BC_Z - 20, BC_Z + 20)), ("ty", 0.0, (-5.0, 5.0)),
                    ("tz", 0.0, (-5.0, 5.0))):
        s.add(Parameter(name=n, init=v, refined=not frozen, bounds=b))
    s.add(Parameter(name="pxY", init=PX, refined=False))
    s.add(Parameter(name="panel_ring_delta_r", init=torch.zeros(P, K, dtype=F64),
                    refined=True, bounds=(-2.0, 2.0)))
    return s


def _map(fits):
    return {"Lsd": torch.tensor(LSD, dtype=F64), "BC_y": torch.tensor(BC_Y, dtype=F64),
            "BC_z": torch.tensor(BC_Z, dtype=F64), "ty": torch.tensor(0.0, dtype=F64),
            "tz": torch.tensor(0.0, dtype=F64), "pxY": torch.tensor(PX, dtype=F64),
            "panel_ring_delta_r": fits.cell_offsets.clone()}


def _independent_sigma(fits, unp):
    """Hard-constraint covariance on the plain parameter vector."""
    def res(x):
        p = dict(unp)
        for i, n in enumerate(GLOBALS):
            p[n] = x[i]
        p["panel_ring_delta_r"] = x[len(GLOBALS):].reshape(P, K)
        return pseudo_strain_residual(
            fits.Y_pix, fits.Z_pix, fits.ring_two_theta_deg, p, rho_d=fits.rho_d,
            panel_layout=None, panel_idx=fits.panel_idx, ring_idx=fits.ring_idx)

    x0 = torch.cat([torch.stack([unp[n] for n in GLOBALS]),
                    unp["panel_ring_delta_r"].reshape(-1)])
    r0 = res(x0).detach()
    J = torch.autograd.functional.jacobian(res, x0)
    n = x0.numel()
    s2 = float((r0 * r0).sum()) / (r0.numel() - n)
    F = J.T @ J / s2
    C = torch.zeros(K, n, dtype=F64)                         # per-ring sum over panels = 0
    for k in range(K):
        for p in range(P):
            C[k, len(GLOBALS) + p * K + k] = 1.0
    # null space of C
    _, _, Vh = torch.linalg.svd(C)
    Z = Vh[K:].T
    cov = Z @ torch.linalg.inv(Z.T @ F @ Z) @ Z.T
    return {nm: float(cov[i, i].clamp(min=0).sqrt()) for i, nm in enumerate(GLOBALS)}


def test_matches_an_independent_hard_constraint_covariance():
    f = _Fits()
    ts = tiled_global_sigma(_spec(), _map(f), f, panel_layout=None, free_globals=GLOBALS)
    ref = _independent_sigma(f, _map(f))
    for n in GLOBALS:
        assert math.isfinite(ts.sigma[n]) and ts.sigma[n] > 0.0, (n, ts.sigma)
        assert abs(ts.sigma[n] / ref[n] - 1.0) < 0.05, (n, ts.sigma[n], ref[n])


def test_the_gauge_is_what_pins_lsd():
    f = _Fits()
    pinned = tiled_global_sigma(_spec(), _map(f), f, panel_layout=None,
                                free_globals=GLOBALS).sigma["Lsd"]
    loose = tiled_global_sigma(_spec(), _map(f), f, panel_layout=None,
                               free_globals=GLOBALS, lambda_gauge=0.0).sigma["Lsd"]
    assert loose > 100.0 * pinned, (loose, pinned)


def test_both_gauges_are_reported_and_named_and_the_beam_centre_does_not_care():
    f = _Fits()
    ts = tiled_global_sigma(_spec(), _map(f), f, panel_layout=None, free_globals=GLOBALS)
    assert "mean" in ts.gauge and "best" in ts.gauge_alt
    assert set(GLOBALS) <= set(ts.sigma) and set(GLOBALS) <= set(ts.sigma_alt)
    for n in ("BC_y", "BC_z"):
        assert 0.2 < ts.sigma_alt[n] / ts.sigma[n] < 5.0, (n, ts.sigma, ts.sigma_alt)
    assert "panel_ring_delta_r" not in ts.gauge_dependent


def test_no_ring_offset_parameter_means_no_result():
    f = _Fits()
    s = _spec()
    del s.parameters["panel_ring_delta_r"]
    unp = _map(f)
    del unp["panel_ring_delta_r"]
    assert tiled_global_sigma(s, unp, f, panel_layout=None, free_globals=GLOBALS) is None
