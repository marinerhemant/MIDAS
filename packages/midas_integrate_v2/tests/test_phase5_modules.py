"""Smoke tests for Phase 5 modules (Items 38, 39, 40, 41, 35)."""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from midas_integrate.params import IntegrationParams

from midas_integrate_v2 import spec_from_v1_params
from midas_integrate_v2.grazing import pixel_to_qy_qz, remap_to_qy_qz_grid
from midas_integrate_v2.inelastic import regroup_eta_R_E_to_Q_E
from midas_integrate_v2.io.aps_dm import APSDMClient, ExperimentMetadata
from midas_integrate_v2.io.multimodal import (
    AuxiliaryStream, align_to_xrd_frames,
)
from midas_integrate_v2.streaming import NumpyArraySource
from midas_integrate_v2.streaming.triggers import (
    TriggerMetadata, TriggerTaggedFrameSource, differential_with_variance,
)


def _spec(NY=64, NZ=64):
    p = IntegrationParams(
        NrPixelsY=NY, NrPixelsZ=NZ,
        pxY=200.0, pxZ=200.0, Lsd=1_000_000.0,
        BC_y=NY / 2.0, BC_z=NZ / 2.0, RhoD=float(NY),
        RMin=1.0, RMax=20.0, RBinSize=0.5,
        EtaMin=-180.0, EtaMax=180.0, EtaBinSize=10.0,
        Wavelength=0.18,
    )
    return spec_from_v1_params(p, requires_grad=False)


def test_gisaxs_pixel_to_qy_qz():
    spec = _spec()
    Y = torch.tensor([10.0, 20.0, 30.0], dtype=torch.float64)
    Z = torch.tensor([40.0, 30.0, 20.0], dtype=torch.float64)
    qy, qz = pixel_to_qy_qz(Y, Z, spec=spec, incidence_angle_deg=0.5)
    assert qy.shape == (3,)
    assert qz.shape == (3,)
    assert torch.isfinite(qy).all() and torch.isfinite(qz).all()


def test_gisaxs_remap_smooth_image():
    spec = _spec()
    img = np.ones((spec.NrPixelsZ, spec.NrPixelsY))
    qy_grid = np.linspace(-0.001, 0.001, 16)
    qz_grid = np.linspace(0.0, 0.002, 16)
    out = remap_to_qy_qz_grid(img, spec, incidence_angle_deg=0.5,
                                 qy_grid=qy_grid, qz_grid=qz_grid)
    assert out.shape == (16, 16)
    assert np.isfinite(out).any()


def test_inelastic_regroup_q_e():
    n_eta, n_r, n_e = 4, 16, 8
    cube = torch.ones(n_eta, n_r, n_e, dtype=torch.float64)
    eta = torch.linspace(-180.0, 180.0, n_eta)
    Q_axis = torch.linspace(0.5, 5.0, n_r)
    E_axis = torch.linspace(900.0, 950.0, n_e)
    Q_grid = torch.linspace(0.6, 4.5, 24)
    out = regroup_eta_R_E_to_Q_E(cube, eta, Q_axis, E_axis, Q_grid=Q_grid)
    assert out.shape == (24, n_e)
    np.testing.assert_allclose(out.numpy(), 1.0, rtol=1e-12)


def test_multimodal_align_linear():
    frame_t = np.array([0.0, 1.0, 2.0, 3.0])
    aux = AuxiliaryStream(
        name="raman", timestamps=np.array([0.0, 2.0, 4.0]),
        values=np.array([10.0, 20.0, 30.0]),
    )
    out = align_to_xrd_frames(frame_t, [aux], interpolation="linear")
    np.testing.assert_allclose(out["raman"], [10.0, 15.0, 20.0, 25.0])


def test_multimodal_align_nearest_and_previous():
    frame_t = np.array([0.0, 1.5, 3.0])
    aux = AuxiliaryStream(
        name="dsc", timestamps=np.array([0.0, 1.0, 2.0, 3.0]),
        values=np.array([0.0, 10.0, 20.0, 30.0]),
    )
    out_n = align_to_xrd_frames(frame_t, [aux], interpolation="nearest")
    # numpy argmin returns the first index on ties; for frame_t=1.5 the
    # distances [1.5, 0.5, 0.5, 1.5] argmin → idx 1 → value 10.
    np.testing.assert_array_equal(out_n["dsc"], [0.0, 10.0, 30.0])
    out_p = align_to_xrd_frames(frame_t, [aux], interpolation="previous")
    np.testing.assert_array_equal(out_p["dsc"], [0.0, 10.0, 30.0])


def test_trigger_tagged_source_wraps_iter():
    base = NumpyArraySource(np.ones((3, 8, 8)))
    triggers = [
        TriggerMetadata(frame_id=f"f_{k}", pump_state="pumped" if k else "off")
        for k in range(3)
    ]
    src = TriggerTaggedFrameSource(base, triggers)
    assert src.n_frames == 3
    assert src.frame_shape == (8, 8)
    seen = list(src)
    assert len(seen) == 3
    assert src.trigger_for(2).pump_state == "pumped"


def test_differential_with_variance_propagates():
    p = torch.tensor([[1.0, 2.0]], dtype=torch.float64)
    sp = torch.tensor([[0.1, 0.2]], dtype=torch.float64)
    u = torch.tensor([[0.5, 1.0]], dtype=torch.float64)
    su = torch.tensor([[0.2, 0.3]], dtype=torch.float64)
    diff, sigma = differential_with_variance(p, sp, u, su)
    np.testing.assert_allclose(diff.numpy(), [[0.5, 1.0]], rtol=1e-12)
    expected_sigma = np.sqrt(np.array([0.1**2 + 0.2**2, 0.2**2 + 0.3**2]))
    np.testing.assert_allclose(sigma.numpy(), [expected_sigma], rtol=1e-12)


def test_apsdm_client_uses_mock_session():
    fake_resp = MagicMock()
    fake_resp.json.return_value = {
        "id": "exp42", "title": "Test", "files": ["a.h5", "b.h5"],
    }
    fake_resp.raise_for_status = MagicMock()
    fake_session = MagicMock()
    fake_session.get.return_value = fake_resp
    client = APSDMClient(dm_url="https://dm-test.aps", session=fake_session)
    md = client.get_experiment("exp42")
    assert isinstance(md, ExperimentMetadata)
    assert md.title == "Test"
    assert md.files == ["a.h5", "b.h5"]
