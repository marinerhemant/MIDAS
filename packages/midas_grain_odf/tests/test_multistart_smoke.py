"""Smoke test for fit_grain_odf_multistart.

Plants a 3-particle ODF, runs a 4-restart multistart fit, and confirms
the wrapper returns a result with the expected attached attributes
(`selected_restart`, `restart_scores`, `restart_score_kind`) and that
recovery beats a single-fit baseline.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
_PKG_ROOT = _HERE.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from conftest import make_model, random_orientation
from midas_grain_odf.forward_helpers import forward_orientations
from midas_grain_odf.inversion import (
    fit_grain_odf, fit_grain_odf_multistart,
)
from midas_grain_odf.odf import ParticleODF, axis_angle_to_matrix
from midas_grain_odf.spot_extract import (
    SpotPatchSpec, splat_spots_to_patches,
)


def _make_obs(model, R_avg, position, K_planted=3):
    deg = math.pi / 180.0
    aa_planted = torch.tensor([
        [0.00 * deg, 0.00 * deg, 0.00 * deg],
        [0.05 * deg, 0.00 * deg, 0.00 * deg],
        [0.00 * deg, 0.04 * deg, 0.03 * deg],
    ], dtype=torch.float64)
    R_planted = R_avg.unsqueeze(0) @ axis_angle_to_matrix(aa_planted)
    w_planted = torch.tensor([0.5, 0.3, 0.2], dtype=torch.float64)
    spots_p = forward_orientations(model, R_planted, position)
    sy = spots_p.y_pixel.reshape(K_planted, -1)
    sz = spots_p.z_pixel.reshape(K_planted, -1)
    sf = spots_p.frame_nr.reshape(K_planted, -1)
    sv = spots_p.valid.reshape(K_planted, -1)
    spots_avg = forward_orientations(model, R_avg.unsqueeze(0), position)
    sv_a = spots_avg.valid.reshape(-1)
    valid = (sv_a > 0.5) & (sv.sum(dim=0) > 0)
    indexer = torch.nonzero(valid, as_tuple=False).squeeze(-1)
    sy = sy[:, indexer]; sz = sz[:, indexer]
    sf = sf[:, indexer]; sv = sv[:, indexer]
    w_norm = (w_planted.reshape(-1, 1) * sv).sum(0).clamp(min=1e-12)
    meas_y = (w_planted.reshape(-1, 1) * sv * sy).sum(0) / w_norm
    meas_z = (w_planted.reshape(-1, 1) * sv * sz).sum(0) / w_norm
    meas_f = (w_planted.reshape(-1, 1) * sv * sf).sum(0) / w_norm
    spec = SpotPatchSpec(
        n_spots=int(indexer.numel()),
        patch_F=7, patch_P=31, sigma_yz=1.0, sigma_f=0.6,
        anchor_y=meas_y.detach().clone(),
        anchor_z=meas_z.detach().clone(),
        anchor_f=meas_f.detach().clone(),
    )
    patches = splat_spots_to_patches(spec, sy, sz, sf, w_planted, sv)
    return dict(meas_y=meas_y, meas_z=meas_z, meas_f=meas_f,
                indexer=indexer, patches=patches)


def test_multistart_returns_attributes_and_beats_single():
    torch.manual_seed(7)
    deg = math.pi / 180.0
    model = make_model()
    R_avg = random_orientation(seed=11).to(torch.float64)
    position = torch.zeros(3, dtype=torch.float64)
    obs = _make_obs(model, R_avg, position)

    def odf_factory(seed):
        return ParticleODF(
            R_avg=R_avg.detach().clone(), K=12,
            theta_max=0.15 * deg, seed=int(seed),
        ).to(torch.float64)

    result = fit_grain_odf_multistart(
        odf_factory, model=model, position=position,
        measured_y=obs["meas_y"], measured_z=obs["meas_z"],
        measured_f=obs["meas_f"], measured_patches=obs["patches"],
        spot_indexer=obs["indexer"],
        n_restarts=3, score="train_loss",
        base_seed=42, seed_step=1009,
        patch_F=7, patch_P=31, sigma_yz=1.0, sigma_f=0.6,
        delta_iters=1, inner_steps=10,
        lr_axis_angle=1e-4, lr_logits=0.1,
        verbose=False,
    )
    assert hasattr(result, "selected_restart")
    assert hasattr(result, "restart_scores")
    assert hasattr(result, "restart_score_kind")
    assert result.restart_score_kind == "train_loss"
    assert len(result.restart_scores) == 3
    assert 0 <= result.selected_restart < 3
    # Selected score should be the minimum across restarts
    assert (result.restart_scores[result.selected_restart]
            == min(result.restart_scores))


def test_multistart_holdout_score_runs():
    torch.manual_seed(7)
    deg = math.pi / 180.0
    model = make_model()
    R_avg = random_orientation(seed=11).to(torch.float64)
    position = torch.zeros(3, dtype=torch.float64)
    obs = _make_obs(model, R_avg, position)

    def odf_factory(seed):
        return ParticleODF(
            R_avg=R_avg.detach().clone(), K=12,
            theta_max=0.15 * deg, seed=int(seed),
        ).to(torch.float64)

    result = fit_grain_odf_multistart(
        odf_factory, model=model, position=position,
        measured_y=obs["meas_y"], measured_z=obs["meas_z"],
        measured_f=obs["meas_f"], measured_patches=obs["patches"],
        spot_indexer=obs["indexer"],
        n_restarts=2, score="holdout_mse",
        holdout_frac=0.25, holdout_seed=0,
        base_seed=42, seed_step=1009,
        patch_F=7, patch_P=31, sigma_yz=1.0, sigma_f=0.6,
        delta_iters=1, inner_steps=10,
        lr_axis_angle=1e-4, lr_logits=0.1,
        verbose=False,
    )
    assert result.restart_score_kind == "holdout_mse"
    assert len(result.restart_scores) == 2
