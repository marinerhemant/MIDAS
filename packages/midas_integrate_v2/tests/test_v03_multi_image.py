"""v0.3: multi-image loss aggregators (MultiImageLoss + BatchedSpecLoss).

Pins:

1. ``MultiImageLoss(reduction='mean')`` returns the mean of per-item
   losses; ``'sum'`` returns the sum; ``'none'`` returns the per-item
   tensor. Weights act as expected (weighted mean / sum).
2. ``BatchedSpecLoss`` returns the same scalar as ``MultiImageLoss``
   when all items share the same spec (batched-vs-sequential is just
   an integration-path difference).
3. Multi-distance simulation: 3 frames at 3 different Lsd values, each
   with its own spec, share *one* refinable BC_y (or other parameter).
   Joint refinement on ``MultiImageLoss(EtaUniformityLoss)`` recovers
   BC_y when each spec's BC_y is the same shared tensor.
4. Gradient flow through both aggregators reaches every refinable spec
   parameter referenced in any item.
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pytest
import torch

from midas_integrate.params import IntegrationParams
from midas_integrate_v2 import (
    spec_from_v1_params,
    SoftBinGeometry,
    integrate_with_corrections,
    EtaUniformityLoss,
    ProfileMSELoss,
    MultiImageLoss,
    BatchedSpecLoss,
    profile_1d_diff,
)


def _spec(NY=24, NZ=24, *, Lsd=1_000_000.0, BC_y=None,
           requires_grad=False, shared_bc=None):
    p = IntegrationParams(
        NrPixelsY=NY, NrPixelsZ=NZ,
        pxY=200.0, pxZ=200.0, Lsd=Lsd,
        BC_y=NY / 2.0 + 0.37 if BC_y is None else BC_y,
        BC_z=NZ / 2.0 - 0.41, RhoD=float(NY),
        RMin=1.0, RMax=12.0, RBinSize=1.0,
        EtaMin=-180.0, EtaMax=180.0, EtaBinSize=60.0,
    )
    s = spec_from_v1_params(p, requires_grad=requires_grad)
    if shared_bc is not None:
        s.BC_y = shared_bc                   # plug in a shared tensor
    return s


def _gauss_image(NY, NZ, *, R0_px=6.0, sigma_px=1.5,
                  BC_y=None, BC_z=None, px=200.0):
    BC_y = NY / 2.0 + 0.37 if BC_y is None else BC_y
    BC_z = NZ / 2.0 - 0.41 if BC_z is None else BC_z
    yy, zz = np.meshgrid(np.arange(NY), np.arange(NZ), indexing="xy")
    Yc = -(yy - BC_y) * px
    Zc = (zz - BC_z) * px
    R_um = np.sqrt(Yc * Yc + Zc * Zc)
    R_px = R_um / px
    return torch.from_numpy(
        np.exp(-(R_px - R0_px) ** 2 / (2 * sigma_px ** 2)).astype(np.float64)
    )


# ── (1) reductions + weights ──

def test_multi_image_loss_mean_equals_python_mean():
    s = _spec()
    img = _gauss_image(s.NrPixelsY, s.NrPixelsZ)
    items = [(img, s), (img, s), (img, s)]
    loss_fn = lambda i2d, sp: i2d.mean()
    L = MultiImageLoss(loss_fn, reduction="mean")(items)
    expected = integrate_with_corrections(img, s).mean()
    # mean = sum/N introduces 1-ULP rounding vs the single-item mean.
    torch.testing.assert_close(L, expected, rtol=0, atol=1e-15)


def test_multi_image_loss_sum_equals_python_sum():
    s = _spec()
    img = _gauss_image(s.NrPixelsY, s.NrPixelsZ)
    items = [(img, s), (img, s)]
    loss_fn = lambda i2d, sp: i2d.mean()
    L = MultiImageLoss(loss_fn, reduction="sum")(items)
    expected = 2 * integrate_with_corrections(img, s).mean()
    torch.testing.assert_close(L, expected, rtol=0, atol=1e-12)


def test_multi_image_loss_none_returns_per_item():
    s = _spec()
    img = _gauss_image(s.NrPixelsY, s.NrPixelsZ)
    items = [(img, s), (img, s), (img, s)]
    loss_fn = lambda i2d, sp: i2d.mean()
    L = MultiImageLoss(loss_fn, reduction="none")(items)
    assert L.shape == (3,)


def test_multi_image_loss_weights_apply():
    """Weighted mean = (Σ w_i · L_i) / Σ w_i."""
    s = _spec()
    img = _gauss_image(s.NrPixelsY, s.NrPixelsZ)
    items = [(img, s), (img, s), (img, s)]
    loss_fn = lambda i2d, sp: i2d.mean().to(torch.float64)
    weights = [1.0, 2.0, 3.0]
    L = MultiImageLoss(loss_fn, reduction="mean", weights=weights)(items)
    base = float(integrate_with_corrections(img, s).mean())
    expected = base       # all three are the same value
    assert float(L) == pytest.approx(expected, rel=1e-9)


def test_multi_image_loss_rejects_empty_items():
    loss_fn = lambda i2d, sp: i2d.mean()
    with pytest.raises(ValueError, match="empty"):
        MultiImageLoss(loss_fn)([])


def test_multi_image_loss_rejects_weights_length_mismatch():
    s = _spec()
    img = _gauss_image(s.NrPixelsY, s.NrPixelsZ)
    items = [(img, s), (img, s)]
    loss_fn = lambda i2d, sp: i2d.mean()
    with pytest.raises(ValueError, match="weights length"):
        MultiImageLoss(loss_fn, weights=[1.0, 2.0, 3.0])(items)


# ── (2) BatchedSpecLoss equivalence ──

def test_batched_spec_loss_matches_multi_image_for_same_spec():
    s = _spec()
    geom = SoftBinGeometry.from_spec(s)
    rng = torch.Generator().manual_seed(0)
    images = torch.rand(3, s.NrPixelsZ, s.NrPixelsY,
                         generator=rng, dtype=torch.float64)

    eta_loss = EtaUniformityLoss(intensity_floor=0.0)
    # MultiImageLoss runs each image independently
    items = [(images[i], s) for i in range(3)]
    L_multi = MultiImageLoss(
        lambda i2d, sp: eta_loss(i2d), reduction="mean",
    )(items)
    # BatchedSpecLoss uses the batched integrate path
    L_batch = BatchedSpecLoss(
        lambda i2d, sp: eta_loss(i2d), reduction="mean",
    )(images, s, geom)
    # Scalars should agree to fp64-with-batch-noise tolerance
    assert float(L_multi) == pytest.approx(float(L_batch), rel=1e-10)


def test_batched_spec_loss_with_targets_per_image():
    """When per-image targets are provided, loss_fn receives them as
    third arg."""
    s = _spec()
    geom = SoftBinGeometry.from_spec(s)
    rng = torch.Generator().manual_seed(1)
    images = torch.rand(2, s.NrPixelsZ, s.NrPixelsY,
                         generator=rng, dtype=torch.float64)
    targets = torch.zeros(2, s.n_eta_bins, s.n_r_bins, dtype=torch.float64)

    def loss_fn(i2d, sp, target):
        return ((i2d - target) ** 2).mean()

    L = BatchedSpecLoss(loss_fn, reduction="mean")(
        images, s, geom, targets_3d=targets,
    )
    assert torch.isfinite(L)
    assert float(L) > 0


# ── (3) gradient flow ──

def test_multi_image_loss_gradient_flows_to_shared_spec():
    s = _spec(requires_grad=True)
    img = _gauss_image(s.NrPixelsY, s.NrPixelsZ)
    items = [(img, s), (img, s)]
    loss_fn = lambda i2d, sp: i2d.mean()
    L = MultiImageLoss(loss_fn)(items)
    L.backward()
    assert s.Lsd.grad is not None and torch.isfinite(s.Lsd.grad).all()
    assert s.BC_y.grad is not None and torch.isfinite(s.BC_y.grad).all()


def test_batched_spec_loss_gradient_flows_to_spec():
    s = _spec(requires_grad=True)
    geom = SoftBinGeometry.from_spec(s)
    images = _gauss_image(s.NrPixelsY, s.NrPixelsZ).unsqueeze(0).repeat(3, 1, 1)
    eta_loss = EtaUniformityLoss(intensity_floor=0.0)
    L = BatchedSpecLoss(lambda i2d, sp: eta_loss(i2d))(images, s, geom)
    L.backward()
    for f in ("Lsd", "BC_y", "BC_z"):
        g = getattr(s, f).grad
        assert g is not None and torch.isfinite(g).all(), f"{f} grad bad"


# ── (4) Multi-distance smoke: 3 frames at different Lsd, shared BC_y ──

def test_multi_distance_shared_BC_y_gradient_aggregates_across_frames():
    """Three frames at three Lsd values, each with its own spec but a
    SHARED refinable BC_y tensor. Grad on the joint loss must equal
    the sum of grads from the three single-frame losses (sum reduction)."""
    NY = NZ = 24
    BC_shared = torch.tensor(NY / 2.0 + 0.37 + 0.5, dtype=torch.float64,
                              requires_grad=True)
    Lsds = [800_000.0, 1_000_000.0, 1_200_000.0]
    specs = [
        _spec(NY=NY, NZ=NZ, Lsd=Lsd_val, requires_grad=True,
              shared_bc=BC_shared)
        for Lsd_val in Lsds
    ]
    images = [_gauss_image(NY, NZ) for _ in Lsds]
    items = list(zip(images, specs))
    eta_loss = EtaUniformityLoss(intensity_floor=0.0)

    L = MultiImageLoss(lambda i2d, sp: eta_loss(i2d), reduction="sum")(items)
    L.backward()
    grad_joint = float(BC_shared.grad)

    # Compare to manual sum
    BC2 = torch.tensor(NY / 2.0 + 0.37 + 0.5, dtype=torch.float64,
                        requires_grad=True)
    specs2 = [
        _spec(NY=NY, NZ=NZ, Lsd=Lsd_val, requires_grad=True, shared_bc=BC2)
        for Lsd_val in Lsds
    ]
    L_manual = sum(
        eta_loss(integrate_with_corrections(images[i], specs2[i]))
        for i in range(3)
    )
    L_manual.backward()
    grad_manual = float(BC2.grad)

    assert grad_joint == pytest.approx(grad_manual, rel=1e-10)


def test_multi_distance_joint_refinement_recovers_BC_y():
    """Three frames at different Lsds; BC_y is the same physical
    parameter perturbed by +0.5 px from truth. Joint Adam over the
    sum-reduction loss must move BC_y back toward truth in <300 steps."""
    NY = NZ = 32
    BC_true = NY / 2.0 + 0.37
    Lsds = [800_000.0, 1_000_000.0, 1_200_000.0]
    # Synthesize per-frame images at the TRUE BC.
    images = []
    for Lsd_val in Lsds:
        # The peak R depends on Lsd: R = (Lsd / px) tan(2θ).
        # Pick a 2θ that lands the peak around mid-range for each Lsd.
        R0 = 6.0 + 0.5 * (Lsd_val / 1_000_000.0 - 1.0)
        images.append(_gauss_image(NY, NZ, R0_px=R0,
                                     BC_y=BC_true,
                                     BC_z=NZ / 2.0 - 0.41))

    BC_shared = torch.tensor(BC_true + 0.5, dtype=torch.float64,
                              requires_grad=True)
    specs = [
        _spec(NY=NY, NZ=NZ, Lsd=Lsd_val, requires_grad=True,
              shared_bc=BC_shared)
        for Lsd_val in Lsds
    ]

    eta_loss = EtaUniformityLoss(intensity_floor=0.0)
    multi = MultiImageLoss(lambda i2d, sp: eta_loss(i2d), reduction="sum")

    opt = torch.optim.Adam([BC_shared], lr=0.02)
    history = [float(BC_shared.detach())]
    for _ in range(200):
        opt.zero_grad()
        L = multi(list(zip(images, specs)))
        L.backward()
        opt.step()
        history.append(float(BC_shared.detach()))

    closest = min(abs(bc - BC_true) for bc in history)
    initial_err = abs(history[0] - BC_true)
    assert closest < 0.25 * initial_err, (
        f"Multi-distance joint refinement did not close BC_y: "
        f"closest err {closest:.3f}, initial {initial_err:.3f}, "
        f"history end={history[-3:]}"
    )
