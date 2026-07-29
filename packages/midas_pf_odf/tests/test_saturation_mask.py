"""P2-7 tests: hard saturation mask through assembly → scale → loss.

SOH Varex failure mode: rings 1/2 are 96-98% saturated; flat-top blobs
vs narrow Gaussian splats floor the MSE and the strain runs away. The
fix is a per-pixel weight-0 mask above the detector threshold, applied
in BOTH the data MSE and the closed-form per-spot amplitude.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from midas_pf_odf import fit_grain_peakshape
from midas_pf_odf.forward import closed_form_per_spot_scale
from midas_pf_odf.io import saturation_threshold_from_paramstest
from midas_pf_odf.simulate import plant_single_grain, simulate_grain_patches
from tests.conftest import build_model, make_fcc_hkls, small_scan_config

DT = torch.float64


def test_threshold_from_paramstest():
    assert saturation_threshold_from_paramstest(
        {"UpperBoundThreshold": [["64000"]]}) == 64000.0
    assert saturation_threshold_from_paramstest({}) is None
    assert saturation_threshold_from_paramstest(
        {"UpperBoundThreshold": [["0"]]}) is None


def test_scale_ignores_saturated_flat_top():
    """A clipped measurement must not drag the fitted amplitude down."""
    P = 21
    yy, zz = torch.meshgrid(torch.arange(P, dtype=DT),
                            torch.arange(P, dtype=DT), indexing="ij")
    g = torch.exp(-(((yy - 10) ** 2 + (zz - 10) ** 2) / (2 * 1.5 ** 2)))
    pred = g[None, None, None]                        # (1,1,1,P,P), peak 1.0
    true_amp = 100_000.0
    thr = 64_000.0
    meas = (true_amp * g).clamp(max=thr)[None, None, None]

    c_unmasked = closed_form_per_spot_scale(pred, meas)
    mask = (meas < thr).to(DT)
    c_masked = closed_form_per_spot_scale(pred, meas, pixel_weight=mask)
    # Unmasked: the flat top biases the amplitude LOW. Masked: recovered.
    assert c_unmasked[0] < 0.9 * true_amp
    assert c_masked[0] == pytest.approx(true_amp, rel=0.02)


def _planted_case():
    G_cart, thetas, hkls_int = make_fcc_hkls()
    scan = small_scan_config(sample_size_um=12.0, n_scans=5, beam_size_um=4.0)
    model = build_model(scan, hkls_int, G_cart, thetas)
    plant = plant_single_grain(
        grid_shape=(3, 3), voxel_size_um=4.0,
        lattice=(3.61, 3.61, 3.61, 90.0, 90.0, 90.0),
        eps_gradient_voigt=0, eps_gradient_amp=1e-3, eps_gradient_dir="y",
    )
    data = simulate_grain_patches(plant, model, patch_F=5, patch_P=15)
    return model, plant, data


def _fit(data, model, plant, **over):
    kw = dict(
        voxel_pos=plant.voxel_pos,
        R_init=plant.R_voxel,
        eps_init=torch.zeros(plant.n_voxels, 6, dtype=DT),
        lattice_init=plant.lattice,
        optimizer="adam",
        inner_steps=3,
        lr_aa=0.0, lr_eps=0.0, lr_lat=0.0,
    )
    kw.update(over)
    return fit_grain_peakshape(data, model, **kw)


def test_assembled_mask_and_inversion_runs_with_saturation():
    """Clip the synthetic patches at 60% of max; the masked inversion must
    run clean (finite loss) with the mask wired through
    GrainPatchData.saturation_mask."""
    import dataclasses
    model, plant, data = _planted_case()
    peak = float(data.measured_patches.max())
    thr = 0.6 * peak
    clipped = data.measured_patches.clamp(max=thr)
    data_sat = dataclasses.replace(
        data,
        measured_patches=clipped,
        saturation_mask=(clipped < thr),
    )
    assert data_sat.saturation_mask is not None
    n_sat = int((~data_sat.saturation_mask).sum())
    assert n_sat > 0, "test fixture must actually saturate pixels"

    res = _fit(data_sat, model, plant, inner_steps=5,
               lr_eps=1e-5, lr_aa=1e-4)
    assert np.isfinite(res.losses).all()


def test_mask_changes_data_term():
    """The masked data term must differ from unmasked on clipped data
    (i.e. the mask is actually consumed, not silently dropped)."""
    import dataclasses
    model, plant, data = _planted_case()
    peak = float(data.measured_patches.max())
    thr = 0.5 * peak
    clipped = data.measured_patches.clamp(max=thr)
    d_unmasked = dataclasses.replace(data, measured_patches=clipped)
    d_masked = dataclasses.replace(
        data, measured_patches=clipped, saturation_mask=(clipped < thr))
    r_u = _fit(d_unmasked, model, plant)
    r_m = _fit(d_masked, model, plant)
    assert r_u.losses[0] != pytest.approx(r_m.losses[0], rel=1e-6)
