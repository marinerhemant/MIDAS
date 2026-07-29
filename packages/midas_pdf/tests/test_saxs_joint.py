"""SAXS Day 2/3 tests: SAXSModel, joint refinement, sphere characteristic function.

Invariants:
  1. SAXSModel(sphere) matches sphere_form_factor_squared * scale.
  2. Lognormal polydispersity integrates to 1.
  3. sphere_characteristic_function γ(0, D) = 1 exactly.
  4. sphere_characteristic_function γ(D, D) = 0 exactly.
  5. Joint fit recovers synthetic (a, U_iso, D) within tolerance.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from midas_pdf.saxs import (
    SAXSModel, lognormal_quadrature_nodes,
    sphere_characteristic_function, joint_refine,
    sphere_form_factor_squared,
)


# ---------------------------------------------------------------------------
# Polydispersity quadrature
# ---------------------------------------------------------------------------

def test_lognormal_quadrature_weights_sum_to_one():
    D, w = lognormal_quadrature_nodes(50.0, 0.1, n_nodes=21)
    assert abs(float(w.sum()) - 1.0) < 1e-6
    assert torch.all(D > 0)


def test_lognormal_quadrature_median_at_center():
    """The median D should sit near the center of the node set."""
    D, w = lognormal_quadrature_nodes(50.0, 0.05, n_nodes=21)
    # Center of the discrete distribution should be close to D_median
    D_centre = float(D[len(D) // 2])
    assert abs(D_centre - 50.0) / 50.0 < 0.1


# ---------------------------------------------------------------------------
# SAXSModel
# ---------------------------------------------------------------------------

def test_saxs_model_sphere_no_polydispersity_matches_form_factor():
    q = torch.linspace(0.01, 0.3, 30, dtype=torch.float64)
    m = SAXSModel(shape="sphere", polydispersity=None)
    I = m.I(q, D_median=50.0, scale=1.0, background=0.0)
    F2 = sphere_form_factor_squared(q, 50.0)
    assert torch.allclose(I, F2, rtol=1e-9)


def test_saxs_model_scale_and_background_pass_through():
    q = torch.linspace(0.01, 0.3, 30, dtype=torch.float64)
    m = SAXSModel(shape="sphere", polydispersity=None)
    I_a = m.I(q, D_median=50.0, scale=1.0, background=0.0)
    I_b = m.I(q, D_median=50.0, scale=2.0, background=5.0)
    assert torch.allclose(I_b, 2.0 * I_a + 5.0, rtol=1e-9)


def test_saxs_model_polydispersity_smears_first_minimum():
    """Polydispersity should reduce the depth of the first form-factor
    minimum (constructive averaging fills it in)."""
    q = torch.linspace(0.01, 0.3, 200, dtype=torch.float64)
    m_mono = SAXSModel(shape="sphere", polydispersity=None)
    m_poly = SAXSModel(shape="sphere", polydispersity=0.15, n_poly_nodes=21)
    I_mono = m_mono.I(q, D_median=50.0)
    I_poly = m_poly.I(q, D_median=50.0)
    assert float(I_poly.min()) > float(I_mono.min())


def test_saxs_model_hard_sphere_S_reduces_intensity_at_low_Q():
    q = torch.linspace(0.005, 0.3, 100, dtype=torch.float64)
    m_bare = SAXSModel(shape="sphere", polydispersity=None)
    m_HS = SAXSModel(shape="sphere", polydispersity=None,
                      S_Q_model="hard_sphere_PY")
    I_bare = m_bare.I(q, D_median=50.0)
    I_HS = m_HS.I(q, D_median=50.0, volume_fraction=0.3)
    # At Q → 0, hard-sphere S(0) < 1 → intensity is suppressed
    assert float(I_HS[0]) < float(I_bare[0])


def test_saxs_model_rejects_unknown_shape():
    with pytest.raises(ValueError):
        SAXSModel(shape="widget").I(
            torch.tensor([0.1], dtype=torch.float64), D_median=50.0)


# ---------------------------------------------------------------------------
# sphere_characteristic_function
# ---------------------------------------------------------------------------

def test_gamma_at_zero_is_one():
    r = torch.tensor([0.0], dtype=torch.float64)
    gamma = sphere_characteristic_function(r, 100.0)
    assert abs(float(gamma[0]) - 1.0) < 1e-12


def test_gamma_at_D_is_zero():
    r = torch.tensor([100.0], dtype=torch.float64)
    gamma = sphere_characteristic_function(r, 100.0)
    assert abs(float(gamma[0])) < 1e-12


def test_gamma_beyond_D_is_zero():
    r = torch.linspace(101.0, 200.0, 20, dtype=torch.float64)
    gamma = sphere_characteristic_function(r, 100.0)
    assert torch.all(gamma == 0.0)


def test_gamma_is_monotonic_decreasing_inside():
    r = torch.linspace(0.0, 99.0, 100, dtype=torch.float64)
    gamma = sphere_characteristic_function(r, 100.0)
    assert torch.all(torch.diff(gamma) <= 1e-9)


def test_gamma_differentiable_in_D():
    r = torch.linspace(1.0, 90.0, 20, dtype=torch.float64)
    D = torch.tensor(100.0, dtype=torch.float64, requires_grad=True)
    gamma = sphere_characteristic_function(r, D)
    gamma.sum().backward()
    assert D.grad is not None and torch.isfinite(D.grad)


# ---------------------------------------------------------------------------
# joint_refine — the headline check
# ---------------------------------------------------------------------------

def _synth_joint_data():
    from midas_hkls import Crystal, Atom, Lattice, SpaceGroup
    from midas_pdf.structure import build_pair_list, pdffit_gr
    a_true, D_true, u_true = 3.524, 100.0, 0.006
    ni = Crystal(lattice=Lattice(a_true, a_true, a_true, 90, 90, 90),
                  space_group=SpaceGroup.from_number(225),
                  atoms=[Atom(element="Ni", fract=(0, 0, 0))], name="Ni").to_torch()
    r = torch.linspace(0.05, 10.0, 150, dtype=torch.float64)
    pairs = build_pair_list(ni, r_max=12.0)
    G_true = pdffit_gr(ni, r, pairs, scale=1.0, u_iso=u_true) \
             * sphere_characteristic_function(r, D_true)
    saxs = SAXSModel(shape="sphere", polydispersity=0.05, n_poly_nodes=11)
    q_saxs = torch.linspace(0.005, 0.3, 60, dtype=torch.float64)
    I_true = saxs.I(q_saxs, D_median=D_true / 2, scale=1.0, background=0.0)
    rng = torch.Generator().manual_seed(0)
    G_obs = G_true + 0.03 * torch.randn(G_true.shape, generator=rng, dtype=torch.float64)
    I_obs = I_true * (1 + 0.05 * torch.randn(I_true.shape, generator=rng,
                                                dtype=torch.float64))
    return {
        "ni": ni, "r": r, "pairs": pairs,
        "G_obs": G_obs, "sigma_G": torch.full_like(G_obs, 0.03),
        "q_saxs": q_saxs, "I_obs": I_obs, "sigma_I": 0.05 * I_true.abs(),
        "saxs_model": saxs, "a_true": a_true, "D_true": D_true,
        "u_true": u_true,
    }


def test_joint_refine_recovers_lattice():
    d = _synth_joint_data()
    res = joint_refine(
        crystal_tensor=d["ni"], r_pdf=d["r"], G_obs=d["G_obs"], pairs=d["pairs"],
        sigma_G=d["sigma_G"], q_saxs=d["q_saxs"], I_saxs=d["I_obs"],
        sigma_I=d["sigma_I"], saxs_model=d["saxs_model"],
        init_a=3.52, init_u_iso=0.005, init_diameter_A=80.0,
        n_steps=50, lr=0.5,
    )
    assert abs(res.fitted["a"] - d["a_true"]) < 0.005


def test_joint_refine_recovers_diameter():
    d = _synth_joint_data()
    res = joint_refine(
        crystal_tensor=d["ni"], r_pdf=d["r"], G_obs=d["G_obs"], pairs=d["pairs"],
        sigma_G=d["sigma_G"], q_saxs=d["q_saxs"], I_saxs=d["I_obs"],
        sigma_I=d["sigma_I"], saxs_model=d["saxs_model"],
        init_a=3.52, init_u_iso=0.005, init_diameter_A=80.0,
        n_steps=100, lr=0.5,
    )
    # Default weights (10, 1) should recover D within a few % on this noise level
    assert abs(res.fitted["diameter_A"] - d["D_true"]) < 5.0


def test_joint_refine_result_has_fit_and_uncertainty():
    d = _synth_joint_data()
    res = joint_refine(
        crystal_tensor=d["ni"], r_pdf=d["r"], G_obs=d["G_obs"], pairs=d["pairs"],
        sigma_G=d["sigma_G"], q_saxs=d["q_saxs"], I_saxs=d["I_obs"],
        sigma_I=d["sigma_I"], saxs_model=d["saxs_model"],
        init_a=3.52, init_u_iso=0.005, init_diameter_A=90.0,
        n_steps=30, lr=0.5,
    )
    for k in ("a", "u_iso", "scale_pdf", "diameter_A",
               "scale_saxs", "background_saxs"):
        assert k in res.fitted
        assert k in res.uncertainty
    assert res.G_calc is not None
    assert res.I_saxs_calc is not None
    assert res.chi2_pdf > 0
    assert res.chi2_saxs > 0
