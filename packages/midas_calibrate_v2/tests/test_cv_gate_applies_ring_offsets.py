"""The cross-validation gate must score the model that was fitted.

`cross_validation_gate` called `pseudo_strain_residual` without `ring_idx`, and
that function applies the per-ring (`delta_r_k`) and per-(panel, ring)
(`panel_ring_delta_r`) radial offsets only when it has one. On a tiled detector
calibrated with `panel_mode="radius"`, which is `calibrate()`'s default, the gate
therefore evaluated the geometry WITHOUT the offsets the fit had used. Found by
a physics refuter reading the code on 2026-10-11; the reported Pilatus
cross-validation ratio (0.57x) described no fitted model.

The construction: the observed radius of the outermost ring is displaced by 3 px
from its ideal value, and the fitted `panel_ring_delta_r` carries exactly that
displacement. A gate that applies the offsets reads ~0 on every ring. One that
drops them sees a 3 px error on the outer ring and fails.
"""
from __future__ import annotations

import math

import torch

from midas_calibrate_v2.forward.bragg import R_ideal_px
from midas_calibrate_v2.loss.pseudo_strain import pseudo_strain_residual
from midas_calibrate_v2.pipelines.diagnostics import cross_validation_gate

F64 = torch.float64
LSD, PX = 1_000_000.0, 200.0
BC_Y, BC_Z = 512.0, 512.0
N_RING, N_AZ, SHIFT_PX = 6, 24, 3.0


class _Fits:
    """The few FittedDataset attributes the gate reads."""
    def __init__(self):
        tth = torch.linspace(5.0, 11.0, N_RING, dtype=F64)
        r_ideal = R_ideal_px(tth, torch.tensor(LSD, dtype=F64), PX)
        off = torch.zeros(N_RING, dtype=F64)
        off[-1] = SHIFT_PX                              # the outer ring sits 3 px out
        Y, Z, T, K = [], [], [], []
        for k in range(N_RING):
            phi = torch.linspace(0.0, 2 * math.pi, N_AZ + 1, dtype=F64)[:-1]
            R = r_ideal[k] + off[k]
            Y.append(BC_Y + R * torch.cos(phi))
            Z.append(BC_Z + R * torch.sin(phi))
            T.append(tth[k].expand(N_AZ))
            K.append(torch.full((N_AZ,), k, dtype=torch.long))
        self.Y_pix, self.Z_pix = torch.cat(Y), torch.cat(Z)
        self.ring_two_theta_deg, self.ring_idx = torch.cat(T), torch.cat(K)
        self.panel_idx = torch.zeros_like(self.ring_idx)
        self.rho_d = torch.tensor(1.0e6, dtype=F64)
        self.ring_d_spacing_A = None


def _params(with_offsets: bool):
    p = {"Lsd": torch.tensor(LSD, dtype=F64), "BC_y": torch.tensor(BC_Y, dtype=F64),
         "BC_z": torch.tensor(BC_Z, dtype=F64), "ty": torch.tensor(0.0, dtype=F64),
         "tz": torch.tensor(0.0, dtype=F64), "pxY": torch.tensor(PX, dtype=F64)}
    prd = torch.zeros(1, N_RING, dtype=F64)
    if with_offsets:
        prd[0, -1] = SHIFT_PX
    p["panel_ring_delta_r"] = prd
    return p


def test_the_premise_without_the_offset_the_outer_ring_is_far_off():
    """Control: a residual that ignores `ring_idx` sees the 3 px displacement."""
    f = _Fits()
    r = pseudo_strain_residual(f.Y_pix, f.Z_pix, f.ring_two_theta_deg, _params(True),
                               rho_d=f.rho_d, panel_idx=f.panel_idx)        # no ring_idx
    outer = (f.ring_idx == N_RING - 1)
    assert float(r[outer].abs().median()) * 1e6 > 1000.0, "premise gone: nothing to drop"
    r2 = pseudo_strain_residual(f.Y_pix, f.Z_pix, f.ring_two_theta_deg, _params(True),
                                rho_d=f.rho_d, panel_idx=f.panel_idx, ring_idx=f.ring_idx)
    assert float(r2.abs().max()) * 1e6 < 1.0


def test_gate_scores_the_fitted_model_including_the_ring_offsets():
    g = cross_validation_gate(_Fits(), _params(True))
    assert g.metrics["med_test_uE"] < 1.0, g.metrics
    assert g.severity == "ok", (g.severity, g.message)


def test_gate_still_catches_a_real_misfit():
    """Sensitivity: when the fitted offsets do NOT carry the displacement, the gate
    must still see the 3 px error. Otherwise 'ok' above would mean nothing."""
    g = cross_validation_gate(_Fits(), _params(False))
    assert g.metrics["med_test_uE"] > 1000.0 or g.severity != "ok", (g.severity, g.metrics)
