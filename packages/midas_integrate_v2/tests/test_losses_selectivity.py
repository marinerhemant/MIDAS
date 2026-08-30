"""The five previously-untested loss classes: null, power, and SELECTIVITY.

``EtaSliceLoss``, ``WedgeLoss`` and ``RingMaskedLoss`` all exist to score a
SUBSET of the cake. The invariant that makes them worth having is that the
subset is real: perturbing the region they select must move the loss, and
perturbing anything else must not move it at all. A selector that quietly scores
the whole cake would still look plausible on every other test.

``MultiImageLoss`` and ``BatchedSpecLoss`` are aggregation paths. The one that
matters for ``BatchedSpecLoss`` is parity with the per-image loop — a batched
kernel that disagrees with the loop is the same class of defect as a streaming
path that disagrees with the batch path.

(``ProfileMSELoss``, ``ProfileWeightedMSELoss``, ``EtaUniformityLoss``,
``PeakPositionLoss`` and ``GaussianPriorLoss`` are covered by
``test_followup_losses.py``.)
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest
import torch

from midas_integrate.params import IntegrationParams
from midas_integrate_v2.compat.from_v1 import spec_from_v1_params
from midas_integrate_v2.losses.quasi_2d import (
    EtaSliceLoss, WedgeLoss, RingMaskedLoss)
from midas_integrate_v2.losses.multi import MultiImageLoss, BatchedSpecLoss
from midas_integrate_v2.losses.geometry import EtaUniformityLoss

N = 192


def _spec(n_pix=256, eta_bin=10.0, r_max=110.0):
    p = IntegrationParams()
    p.NrPixelsY = p.NrPixelsZ = n_pix
    p.Lsd = 300_000.0
    p.BC_y = p.BC_z = n_pix / 2.0
    p.pxY = p.pxZ = 200.0          # NOT p.px
    p.RMin, p.RMax, p.RBinSize = 20.0, r_max, 2.0
    p.EtaMin, p.EtaMax, p.EtaBinSize = -180.0, 180.0, eta_bin
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return spec_from_v1_params(p)


@pytest.fixture(scope="module")
def cake():
    spec = _spec()
    g = np.random.default_rng(0)
    ref = torch.tensor(100.0 + 10.0 * g.standard_normal(
        (spec.n_eta_bins, spec.n_r_bins)), dtype=torch.float64)
    return spec, ref


# ----------------------------------------------------------- EtaSliceLoss

def test_eta_slice_null_and_selectivity(cake):
    spec, ref = cake
    sel = [0, 1, 2, 3]
    L = EtaSliceLoss(eta_indices=sel)
    assert float(L(ref.clone(), spec, ref)) == 0.0

    inside = ref.clone(); inside[sel[0], :] += 50.0
    outside = ref.clone(); outside[spec.n_eta_bins // 2, :] += 50.0
    assert float(L(inside, spec, ref)) > 0.0
    assert float(L(outside, spec, ref)) == 0.0, (
        "a change outside the selected eta bins moved the loss — the slice is "
        "not actually selecting")


def test_eta_slice_is_monotone_in_the_perturbation(cake):
    spec, ref = cake
    L = EtaSliceLoss(eta_indices=[0, 1, 2, 3])
    small = ref.clone(); small[0, :] += 50.0
    big = ref.clone(); big[0, :] += 100.0
    assert float(L(big, spec, ref)) > float(L(small, spec, ref))


# -------------------------------------------------------------- WedgeLoss

def test_wedge_selects_by_eta_range(cake):
    spec, ref = cake
    W = WedgeLoss(eta_min_deg=-30.0, eta_max_deg=30.0)
    ref1d = ref.mean(dim=0)
    n_eta = spec.n_eta_bins
    eta_ax = np.linspace(-180, 180, n_eta, endpoint=False) + 180.0 / n_eta
    inside = np.where((eta_ax > -30) & (eta_ax < 30))[0]
    outside = np.where((eta_ax > 90) & (eta_ax < 150))[0]
    assert inside.size and outside.size

    L0 = float(W(ref, spec, ref1d))
    a = ref.clone(); a[inside, :] += 50.0
    b = ref.clone(); b[outside, :] += 50.0
    assert abs(float(W(a, spec, ref1d)) - L0) > 1e-9
    assert abs(float(W(b, spec, ref1d)) - L0) < 1e-9, (
        "a change outside the wedge moved the loss")


# --------------------------------------------------------- RingMaskedLoss

def test_ring_mask_selects_only_masked_cells(cake):
    spec, ref = cake
    R = RingMaskedLoss()
    m = torch.zeros((spec.n_eta_bins, spec.n_r_bins), dtype=torch.bool)
    m[:, :5] = True
    assert float(R(ref.clone(), spec, ref, m)) == 0.0

    inm = ref.clone(); inm[:, 0] += 50.0
    outm = ref.clone(); outm[:, spec.n_r_bins - 1] += 50.0
    assert float(R(inm, spec, ref, m)) > 0.0
    assert float(R(outm, spec, ref, m)) == 0.0, (
        "a change outside the mask moved the loss")


# ------------------------------------------------------- aggregation paths

def _ring_img(n_pix=N, seed=0, amp=3000.0):
    g = np.random.default_rng(seed)
    Zi, Yi = np.mgrid[0:n_pix, 0:n_pix].astype(float)
    R = np.hypot(Yi - n_pix / 2.0, Zi - n_pix / 2.0)
    img = np.full_like(R, 20.0)
    for R0 in (35.0, 55.0, 72.0):
        img += amp * np.exp(-((R - R0) ** 2) / (2 * 1.6 ** 2))
    return torch.tensor(g.poisson(np.clip(img, 0, None)).astype(float))


@pytest.fixture(scope="module")
def images():
    spec = _spec(n_pix=N, eta_bin=20.0, r_max=80.0)
    return spec, _ring_img(seed=0), _ring_img(seed=1)


def _uni():
    L = EtaUniformityLoss(intensity_floor=1e-6)
    return lambda i2, sp: L(i2)


def test_multi_image_reductions(images):
    spec, a, b = images
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mean = MultiImageLoss(loss_fn=_uni(), reduction="mean")
        summ = MultiImageLoss(loss_fn=_uni(), reduction="sum")
        none = MultiImageLoss(loss_fn=_uni(), reduction="none")
        one = float(mean([(a, spec)]))
        dup = float(mean([(a, spec), (a, spec)]))
        s1 = float(summ([(a, spec)]))
        s2 = float(summ([(a, spec), (a, spec)]))
        per = none([(a, spec), (b, spec)])
        both = float(mean([(a, spec), (b, spec)]))
    assert dup == pytest.approx(one, rel=1e-12), "mean must not grow with duplicates"
    assert s2 == pytest.approx(2 * s1, rel=1e-12), "sum must double"
    assert tuple(per.shape) == (2,)
    assert float(per.mean()) == pytest.approx(both, rel=1e-12)


def test_a_zero_weight_drops_the_item(images):
    spec, a, b = images
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        none = MultiImageLoss(loss_fn=_uni(), reduction="none")
        per = none([(a, spec), (b, spec)])
        w = MultiImageLoss(loss_fn=_uni(), reduction="mean", weights=[1.0, 0.0])
        got = float(w([(a, spec), (b, spec)]))
    assert got == pytest.approx(float(per[0]), rel=1e-9)


def test_batched_agrees_with_the_per_image_loop(images):
    """The invariant a batched kernel exists to satisfy."""
    from midas_integrate_v2.binning.soft import SoftBinGeometry

    spec, a, b = images
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        looped = MultiImageLoss(loss_fn=_uni(), reduction="none")(
            [(a, spec), (b, spec)])
        geom = SoftBinGeometry.from_spec(spec)
        batched = BatchedSpecLoss(loss_fn=_uni(), reduction="none")(
            torch.stack([a, b]), spec, geom)
    assert tuple(batched.shape) == (2,)
    # float64 summation order differs between the two paths; 1e-12 relative is
    # the honest tolerance (measured 3e-13).
    assert torch.allclose(batched, looped, rtol=1e-12, atol=0.0), (
        f"batched {batched.tolist()} vs looped {looped.tolist()}")
