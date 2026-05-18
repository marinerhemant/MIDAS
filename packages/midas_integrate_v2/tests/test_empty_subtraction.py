"""Item 1 — EmptySubtraction nn.Module.

- Round-trip: subtracting an empty equal to the image yields zero
  (with clip_negative=False).
- Refinable scale converges back to the planted scale on synthetic
  data.
- Composes correctly with PolarizationCorrection / SolidAngleCorrection.
- Wired through integrate_with_corrections.
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pytest
import torch

from midas_integrate.params import IntegrationParams

from midas_integrate_v2 import (
    EmptySubtraction,
    PolarizationCorrection,
    SolidAngleCorrection,
    integrate_with_corrections,
    spec_from_v1_params,
)


def _spec(NY=24, NZ=24, requires_grad=False):
    p = IntegrationParams(
        NrPixelsY=NY, NrPixelsZ=NZ,
        pxY=200.0, pxZ=200.0, Lsd=1_000_000.0,
        BC_y=NY / 2.0 + 0.37, BC_z=NZ / 2.0 - 0.41, RhoD=float(NY),
        RMin=1.0, RMax=12.0, RBinSize=1.0,
        EtaMin=-180.0, EtaMax=180.0, EtaBinSize=60.0,
        Wavelength=0.18,
    )
    return spec_from_v1_params(p, requires_grad=requires_grad)


def test_empty_subtraction_self_round_trip():
    NY = NZ = 24
    img = torch.rand(NZ, NY, dtype=torch.float64)
    es = EmptySubtraction(img, scale=1.0, offset=0.0, clip_negative=False)
    out = es(img)
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-12)


def test_empty_subtraction_clipping_keeps_nonneg():
    NY = NZ = 16
    img = torch.full((NZ, NY), 5.0, dtype=torch.float64)
    bigger = torch.full((NZ, NY), 7.0, dtype=torch.float64)
    es = EmptySubtraction(bigger, scale=1.0, clip_negative=True)
    out = es(img)
    assert (out >= 0).all()
    assert torch.allclose(out, torch.zeros_like(out))


def test_empty_subtraction_shape_mismatch_raises():
    img = torch.zeros(8, 8, dtype=torch.float64)
    bad = torch.zeros(7, 8, dtype=torch.float64)
    es = EmptySubtraction(bad)
    with pytest.raises(ValueError):
        es(img)


def test_empty_subtraction_wires_into_integrate():
    s = _spec()
    img = torch.full((s.NrPixelsZ, s.NrPixelsY), 10.0, dtype=torch.float64)
    empty = torch.full_like(img, 3.0)
    es = EmptySubtraction(empty, scale=1.0, clip_negative=False)
    int_no = integrate_with_corrections(img, s)
    int_es = integrate_with_corrections(img, s, empty_subtraction=es)
    # Subtracting 3 from a 10-valued image scales the integrated signal.
    # Where bins receive any signal, the difference must be ~ proportional
    # to (10/7) ratio in the subtracted version.
    nonzero = int_no.abs() > 1e-12
    if nonzero.any():
        ratio = (int_es[nonzero] / int_no[nonzero]).mean().item()
        assert ratio == pytest.approx(7.0 / 10.0, rel=1e-2)


def test_empty_subtraction_composes_with_polarization():
    s = _spec()
    img = torch.full((s.NrPixelsZ, s.NrPixelsY), 10.0, dtype=torch.float64)
    empty = torch.full_like(img, 3.0)
    es = EmptySubtraction(empty, scale=1.0, clip_negative=False)
    pol = PolarizationCorrection(pol_fraction=0.99, pol_plane_eta_deg=0.0)
    sa = SolidAngleCorrection()
    out = integrate_with_corrections(
        img, s, empty_subtraction=es, polarization=pol, solid_angle=sa,
    )
    assert torch.isfinite(out).all()


def test_empty_subtraction_refinable_scale_gradient():
    s = _spec()
    img = torch.full((s.NrPixelsZ, s.NrPixelsY), 10.0, dtype=torch.float64)
    empty = torch.full_like(img, 3.0)
    es = EmptySubtraction(empty, scale=1.0, refinable_scale=True,
                           clip_negative=False)
    out = integrate_with_corrections(img, s, empty_subtraction=es)
    L = out.mean()
    L.backward()
    assert es.scale.grad is not None
    assert torch.isfinite(es.scale.grad)
    assert float(es.scale.grad) != 0.0


def test_empty_subtraction_refinable_scale_converges():
    """Plant a known scale into a synthetic stack; auto-fit recovers it."""
    NY = NZ = 16
    torch.manual_seed(0)
    structure = torch.rand(NZ, NY, dtype=torch.float64) * 100.0
    background = torch.full((NZ, NY), 50.0, dtype=torch.float64)
    planted_scale = 0.7
    sample = structure + planted_scale * background

    es = EmptySubtraction(
        background, scale=0.0, refinable_scale=True, clip_negative=False,
    )
    target = structure
    optim = torch.optim.LBFGS([es.scale], lr=0.5, max_iter=50,
                               line_search_fn="strong_wolfe")

    def closure():
        optim.zero_grad()
        loss = (es(sample) - target).pow(2).mean()
        loss.backward()
        return loss

    optim.step(closure)
    assert float(es.scale) == pytest.approx(planted_scale, abs=1e-3)
