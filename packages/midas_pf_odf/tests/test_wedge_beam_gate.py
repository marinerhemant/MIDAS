"""The pf beam gate under a wedge (issues #11/#17).

MIDAS Wedge convention (``midas_diffract.forward`` module doc): a stage-frame
position moves as ``pos_lab = R_y(-W) R_z(omega) pos`` = ``R_n(omega) S pos``
with ``n = (-sin W, 0, cos W)`` and ``S = R_y(-W)``. A tilt about lab y does
not change lab y, so the gate's ``y_rot`` is ``px sin(omega) + py cos(omega)``
for every W -- the expression the C scanning refiner uses.

These tests hold the gate to an independent rigid-body construction
(Rodrigues about ``n`` and about lab y), check that it equals the about-z
expression bitwise, and that it is the model's own lab y.
"""
from __future__ import annotations

import dataclasses
import math

import pytest
import torch

import midas_pf_odf.forward as pf_forward
from midas_pf_odf import plant_single_grain, simulate_grain_patches
from midas_pf_odf.centroid_baseline import predicted_centroids
from midas_pf_odf.forward import soft_beam_gate

from tests.conftest import make_fcc_hkls, small_scan_config, standard_pf_geometry

from midas_diffract.forward import HEDMForwardModel

DEG2RAD = math.pi / 180.0


def _model(wedge_deg: float, scan=None) -> HEDMForwardModel:
    G, th, hi = make_fcc_hkls()
    geom = dataclasses.replace(standard_pf_geometry(), wedge=wedge_deg)
    sc = scan if scan is not None else small_scan_config()
    return HEDMForwardModel(hkls=G, thetas=th, geometry=geom, hkls_int=hi,
                            scan_config=sc).to(torch.float64)


def _rodrigues(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """R = I + sin(a) K + (1 - cos(a)) K^2 for unit ``axis``; batched on angle."""
    kx, ky, kz = (float(v) for v in axis)
    K = torch.tensor([[0.0, -kz, ky], [kz, 0.0, -kx], [-ky, kx, 0.0]],
                     dtype=torch.float64)
    I = torch.eye(3, dtype=torch.float64)
    s = torch.sin(angle)[..., None, None]
    c = torch.cos(angle)[..., None, None]
    return I + s * K + (1.0 - c) * (K @ K)


# Off-axis voxels WITH a z component (a pf layer has finite thickness and the
# fit may move voxels out of plane), plus the origin.
POS = torch.tensor([
    [0.0, 0.0, 0.0],
    [650.0, -300.0, 140.0],
    [-500.0, 450.0, -225.0],
    [120.0, 700.0, 60.0],
], dtype=torch.float64)
OMEGA = torch.linspace(-3.0, 3.0, 13, dtype=torch.float64).expand(4, 13).clone()
OMEGA = OMEGA + torch.tensor([[0.0], [0.3], [-0.7], [1.1]], dtype=torch.float64)
BEAM = torch.linspace(-800.0, 800.0, 17, dtype=torch.float64)
BEAM_SIZE = 20.0
TAU = 10.0


def _expected_gate(wedge_deg: float) -> torch.Tensor:
    """Independent: stage frame -> lab by Rodrigues (S, then about n), lab y."""
    W = wedge_deg * DEG2RAD
    n_hat = torch.tensor([-math.sin(W), 0.0, math.cos(W)], dtype=torch.float64)
    S = _rodrigues(torch.tensor([0.0, 1.0, 0.0]), torch.tensor(-W, dtype=torch.float64))
    R = _rodrigues(n_hat, OMEGA) @ S                         # (G, S, 3, 3)
    p_lab = torch.einsum("gsij,gj->gsi", R, POS)             # (G, S, 3)
    y = p_lab[..., 1]
    diff = y.unsqueeze(-1) - BEAM
    return torch.sigmoid((BEAM_SIZE / 2.0 - diff.abs()) / TAU)


def test_gate_w0_bit_identical_to_about_z():
    """W == 0: with or without the model the gate is the old expression, bitwise."""
    m0 = _model(0.0)
    g_none = soft_beam_gate(POS, OMEGA, BEAM, BEAM_SIZE, TAU)
    g_m0 = soft_beam_gate(POS, OMEGA, BEAM, BEAM_SIZE, TAU, model=m0)
    px, py = POS[:, 0:1], POS[:, 1:2]
    y_old = px * torch.sin(OMEGA) + py * torch.cos(OMEGA)
    g_old = torch.sigmoid((BEAM_SIZE / 2.0 - (y_old.unsqueeze(-1) - BEAM).abs())
                          / max(TAU, 1e-6))
    assert torch.equal(g_none, g_old)
    assert torch.equal(g_m0, g_old)


@pytest.mark.parametrize("wedge_deg", [3.0, -2.0, 0.25])
def test_gate_is_the_rigid_body_lab_y(wedge_deg):
    m = _model(wedge_deg)
    got = soft_beam_gate(POS, OMEGA, BEAM, BEAM_SIZE, TAU, model=m)
    want = _expected_gate(wedge_deg)
    assert (got - want).abs().max().item() < 1e-9
    # Null that must fail: the pre-2026-09 midas_diffract lab y,
    # R_y(W) R_z R_y(-W) p, differs measurably for these off-axis voxels.
    W = wedge_deg * DEG2RAD
    px, py, pz = POS[:, 0:1], POS[:, 1:2], POS[:, 2:3]
    y_old = (math.cos(W) * px - math.sin(W) * pz) * torch.sin(OMEGA) + py * torch.cos(OMEGA)
    g_old = torch.sigmoid((BEAM_SIZE / 2.0 - (y_old.unsqueeze(-1) - BEAM).abs()) / TAU)
    assert (g_old - want).abs().max().item() > 1e-3


def test_gate_matches_filter_by_scan_y():
    """The gate's y_rot is midas_diffract's own lab-y (same helper, same W)."""
    m = _model(2.5)
    cw, sw = torch.cos(OMEGA), torch.sin(OMEGA)
    _, y_md, _ = m._rotate_positions(POS[:, 0:1], POS[:, 1:2], POS[:, 2:3], cw, sw)
    want = torch.sigmoid((BEAM_SIZE / 2.0 - (y_md.unsqueeze(-1) - BEAM).abs()) / TAU)
    got = soft_beam_gate(POS, OMEGA, BEAM, BEAM_SIZE, TAU, model=m)
    assert (got - want).abs().max().item() < 1e-12


def _capture_gate_calls(monkeypatch):
    calls = []
    real = pf_forward.soft_beam_gate

    def spy(*args, **kwargs):
        calls.append(kwargs.get("model"))
        return real(*args, **kwargs)

    monkeypatch.setattr(pf_forward, "soft_beam_gate", spy)
    return calls


def test_simulate_plant_gate_uses_model_wedge(monkeypatch):
    """The planted-truth simulator gates through soft_beam_gate WITH the model,
    so plant and inverter share one (wedge-aware) gate."""
    m = _model(1.5)
    calls = _capture_gate_calls(monkeypatch)
    plant = plant_single_grain(grid_shape=(2, 2), voxel_size_um=2.0)
    simulate_grain_patches(plant, m, patch_F=3, patch_P=5)
    assert calls and all(c is m for c in calls)


def test_centroid_baseline_gate_uses_model_wedge(monkeypatch):
    import midas_pf_odf.centroid_baseline as cb
    m = _model(1.5)
    calls = []
    real = cb.soft_beam_gate

    def spy(*args, **kwargs):
        calls.append(kwargs.get("model"))
        return real(*args, **kwargs)

    monkeypatch.setattr(cb, "soft_beam_gate", spy)
    plant = plant_single_grain(grid_shape=(2, 2), voxel_size_um=2.0)
    predicted_centroids(m, plant.R_voxel, plant.eps_voxel, plant.lattice,
                        plant.voxel_pos, n_scans=int(m.scan_config.beam_positions.numel()))
    assert calls and all(c is m for c in calls)
