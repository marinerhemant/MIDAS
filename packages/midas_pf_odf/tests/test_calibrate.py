"""Tests for the raw-frame geometry mini-calibration (P1-6).

Synthetic recovery: plant a grain, perturb the TRUE geometry (beam
centre + Lsd), synthesize measured patches whose blobs sit at the true
anchors (as cropped around the unperturbed model's anchors — exactly the
cache layout), then verify the Gauss-Newton calibration recovers the
perturbation and collapses the residual RMS.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from midas_pf_odf.calibrate import (
    calibrate_raw_frame_geometry, measure_patch_offsets,
)
from midas_pf_odf.io import PFGrainDataset, _forward_anchors
from midas_pf_odf.simulate import plant_single_grain
from tests.conftest import (
    build_model, make_fcc_hkls, small_scan_config, standard_pf_geometry,
)

DT = torch.float64


def _make_ds_and_factory(device="cpu"):
    G_cart, thetas, hkls_int = make_fcc_hkls()
    scan = small_scan_config(sample_size_um=12.0, n_scans=7, beam_size_um=4.0)
    plant = plant_single_grain(
        grid_shape=(3, 3), voxel_size_um=4.0,
        lattice=(3.61, 3.61, 3.61, 90.0, 90.0, 90.0),
        eps_gradient_voigt=0, eps_gradient_amp=0.0, eps_gradient_dir="y",
    )

    def factory(geom_mod=None, omega_start=None):
        geom = standard_pf_geometry()
        if geom_mod:
            for k, v in geom_mod.items():
                setattr(geom, k, float(v))
        if omega_start is not None:
            geom.omega_start = float(omega_start)
        from midas_diffract.forward import HEDMForwardModel
        model = HEDMForwardModel(
            hkls=G_cart, thetas=thetas, geometry=geom,
            hkls_int=hkls_int, scan_config=scan, device=device,
        ).to(DT)
        return model

    model0 = factory()
    G = plant.n_voxels
    ds = PFGrainDataset(
        grain_id=1,
        voxel_idx=np.arange(G),
        voxel_pos=plant.voxel_pos.to(DT),
        R_init=plant.R_voxel.to(DT),
        eps_init=torch.zeros(G, 6, dtype=DT),
        lattice_init=plant.lattice.to(DT),
        model=model0,
        grid_shape=(3, 3),
        grid_ij=np.stack(np.meshgrid(np.arange(3), np.arange(3),
                                     indexing="ij"), -1).reshape(-1, 2),
    )
    return ds, factory


def _synth_patches(ds, factory, true_mod, P=21, F=5, sigma=1.2):
    """Measured patches: one Gaussian blob per (spot, scan) at the TRUE
    anchor, cropped around the START model's anchor (cache layout)."""
    ay0, az0, af0, _v, obs, S = _forward_anchors(ds, DT, "cpu")
    ds_true = copy.copy(ds)
    ds_true.model = factory(geom_mod=true_mod)
    ayT, azT, afT, _v2, obsT, _ = _forward_anchors(ds_true, DT, "cpu")
    Sigma = int(ds.model.scan_config.beam_positions.numel())

    dy = (ayT - ay0).cpu().numpy()
    dz = (azT - az0).cpu().numpy()
    obs_np = (obs & obsT).cpu().numpy()

    c = P // 2
    yy, zz = np.meshgrid(np.arange(P), np.arange(P), indexing="ij")
    meas = np.zeros((S, Sigma, F, P, P))
    for s in range(S):
        if not obs_np[s] or abs(dy[s]) > c - 2 or abs(dz[s]) > c - 2:
            continue
        blob = np.exp(-(((yy - (c + dy[s])) ** 2 +
                         (zz - (c + dz[s])) ** 2) / (2 * sigma ** 2)))
        meas[s, :, F // 2] = 100.0 * blob
    return torch.from_numpy(meas)


def test_measure_patch_offsets_centroids():
    S, Sigma, F, P = 4, 3, 5, 21
    meas = torch.zeros(S, Sigma, F, P, P, dtype=DT)
    # Spot 0: blob at +2 px in y; spot 1: −3 px in z; spot 2: empty.
    meas[0, :, 2, 12, 10] = 50.0
    meas[1, :, 2, 10, 7] = 50.0
    meas[3, :, 4, 10, 10] = 50.0            # frame offset +2
    dy, dz, df, ok = measure_patch_offsets(meas)
    assert ok[0] and ok[1] and not ok[2] and ok[3]
    assert dy[0] == pytest.approx(2.0, abs=1e-6)
    assert dz[1] == pytest.approx(-3.0, abs=1e-6)
    assert df[3] == pytest.approx(2.0, abs=1e-6)


def test_gn_calibration_recovers_bc_shift():
    """A planted (y_BC +2 px, z_BC −1.5 px) must be recovered and the
    residual RMS collapsed (the SOH signature: 7.06 → 1.4 px)."""
    ds, factory = _make_ds_and_factory()
    geom0 = standard_pf_geometry()
    true_mod = {"y_BC": geom0.y_BC + 2.0, "z_BC": geom0.z_BC - 1.5}
    meas = _synth_patches(ds, factory, true_mod)

    cal = calibrate_raw_frame_geometry(
        ds, meas, model_factory=factory,
        params_to_fit=("y_BC", "z_BC"),
        n_iters=2, verbose=False,
    )
    assert cal.rms_before_px > 1.0
    assert cal.rms_after_px < 0.35 * cal.rms_before_px
    assert cal.calibrated["y_BC"] == pytest.approx(true_mod["y_BC"], abs=0.5)
    assert cal.calibrated["z_BC"] == pytest.approx(true_mod["z_BC"], abs=0.5)
    # delta property reports the recovered shift.
    assert cal.delta["y_BC"] == pytest.approx(2.0, abs=0.5)


def test_calibration_requires_measurements():
    ds, factory = _make_ds_and_factory()
    ay0, az0, af0, _v, _obs, S = _forward_anchors(ds, DT, "cpu")
    empty = torch.zeros(S, 7, 5, 21, 21, dtype=DT)
    with pytest.raises(ValueError, match="measured centroid"):
        calibrate_raw_frame_geometry(
            ds, empty, model_factory=factory, verbose=False)
