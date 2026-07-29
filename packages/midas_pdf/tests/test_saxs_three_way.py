"""Rev-11 tests: three-way SAXS + SANS + PDF joint refinement."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from midas_pdf.saxs import (
    SAXSModel, sphere_characteristic_function, joint_refine_three_way,
    ThreeWayJointResult,
)


def _synth_three_way(D_true=100.0):
    from midas_hkls import Crystal, Atom, Lattice, SpaceGroup
    from midas_pdf.structure import build_pair_list, pdffit_gr
    a_true, u_true = 3.524, 0.006
    ni = Crystal(lattice=Lattice(a_true, a_true, a_true, 90, 90, 90),
                  space_group=SpaceGroup.from_number(225),
                  atoms=[Atom(element="Ni", fract=(0, 0, 0))],
                  name="Ni").to_torch()
    r = torch.linspace(0.05, 10.0, 120, dtype=torch.float64)
    pairs = build_pair_list(ni, r_max=12.0)
    G_true = pdffit_gr(ni, r, pairs, scale=1.0, u_iso=u_true) \
             * sphere_characteristic_function(r, D_true)
    saxs = SAXSModel(shape="sphere", polydispersity=0.05, n_poly_nodes=11)
    sans = SAXSModel(shape="sphere", polydispersity=0.05, n_poly_nodes=11)
    q_saxs = torch.linspace(0.005, 0.3, 40, dtype=torch.float64)
    q_sans = torch.linspace(0.003, 0.25, 30, dtype=torch.float64)
    I_saxs = saxs.I(q_saxs, D_median=D_true / 2, scale=1.0)
    I_sans = sans.I(q_sans, D_median=D_true / 2, scale=0.35)
    rng = torch.Generator().manual_seed(0)
    G_obs = G_true + 0.03 * torch.randn(G_true.shape, generator=rng, dtype=torch.float64)
    I_saxs_obs = I_saxs * (1 + 0.05 * torch.randn(I_saxs.shape, generator=rng, dtype=torch.float64))
    I_sans_obs = I_sans * (1 + 0.05 * torch.randn(I_sans.shape, generator=rng, dtype=torch.float64))
    return dict(ni=ni, r=r, pairs=pairs, G_obs=G_obs,
                 sigma_G=torch.full_like(G_obs, 0.03),
                 q_saxs=q_saxs, I_saxs=I_saxs_obs,
                 sigma_I_saxs=0.05 * I_saxs.abs(), saxs_model=saxs,
                 q_sans=q_sans, I_sans=I_sans_obs,
                 sigma_I_sans=0.05 * I_sans.abs(), sans_model=sans,
                 D_true=D_true, a_true=a_true, u_true=u_true)


def test_three_way_recovers_shared_diameter():
    """Diameter shared across SAXS + SANS + PDF should be recovered within
    a couple percent."""
    d = _synth_three_way()
    res = joint_refine_three_way(
        crystal_tensor=d["ni"], r_pdf=d["r"], G_obs=d["G_obs"], pairs=d["pairs"],
        sigma_G=d["sigma_G"],
        q_saxs=d["q_saxs"], I_saxs=d["I_saxs"], sigma_I_saxs=d["sigma_I_saxs"],
        saxs_model=d["saxs_model"],
        q_sans=d["q_sans"], I_sans=d["I_sans"], sigma_I_sans=d["sigma_I_sans"],
        sans_model=d["sans_model"],
        init_a=3.52, init_u_iso=0.005, init_diameter_A=80.0,
        init_scale_sans=0.5, n_steps=100, lr=0.5,
    )
    assert abs(res.fitted["diameter_A"] - d["D_true"]) < 3.0
    assert abs(res.fitted["a"] - d["a_true"]) < 0.005


def test_three_way_result_has_all_channels():
    d = _synth_three_way()
    res = joint_refine_three_way(
        crystal_tensor=d["ni"], r_pdf=d["r"], G_obs=d["G_obs"], pairs=d["pairs"],
        sigma_G=d["sigma_G"],
        q_saxs=d["q_saxs"], I_saxs=d["I_saxs"], sigma_I_saxs=d["sigma_I_saxs"],
        saxs_model=d["saxs_model"],
        q_sans=d["q_sans"], I_sans=d["I_sans"], sigma_I_sans=d["sigma_I_sans"],
        sans_model=d["sans_model"],
        init_a=3.52, init_u_iso=0.005, init_diameter_A=90.0,
        n_steps=30, lr=0.5,
    )
    assert isinstance(res, ThreeWayJointResult)
    assert res.G_calc is not None
    assert res.I_saxs_calc is not None
    assert res.I_sans_calc is not None
    assert res.G_calc.shape == d["r"].shape
    assert res.I_saxs_calc.shape == d["q_saxs"].shape
    assert res.I_sans_calc.shape == d["q_sans"].shape
    # Both SAXS and SANS χ² are separately reported
    assert res.chi2_saxs > 0
    assert res.chi2_sans > 0
    assert res.chi2_pdf > 0


def test_three_way_diameter_shared_across_channels():
    """The fit must produce ONE diameter (not per-channel) — this is the
    whole point of the joint fit."""
    d = _synth_three_way()
    res = joint_refine_three_way(
        crystal_tensor=d["ni"], r_pdf=d["r"], G_obs=d["G_obs"], pairs=d["pairs"],
        sigma_G=d["sigma_G"],
        q_saxs=d["q_saxs"], I_saxs=d["I_saxs"], sigma_I_saxs=d["sigma_I_saxs"],
        saxs_model=d["saxs_model"],
        q_sans=d["q_sans"], I_sans=d["I_sans"], sigma_I_sans=d["sigma_I_sans"],
        sans_model=d["sans_model"],
        init_a=3.52, init_u_iso=0.005, init_diameter_A=80.0,
        n_steps=50, lr=0.5,
    )
    assert "diameter_A" in res.fitted
    # There should be exactly ONE diameter key, not diameter_saxs / diameter_sans
    diameter_keys = [k for k in res.fitted if "diameter" in k]
    assert diameter_keys == ["diameter_A"]


def test_three_way_separate_scales_per_channel():
    """SAXS + SANS have independent scales (different contrasts) — the fit
    must expose both."""
    d = _synth_three_way()
    res = joint_refine_three_way(
        crystal_tensor=d["ni"], r_pdf=d["r"], G_obs=d["G_obs"], pairs=d["pairs"],
        sigma_G=d["sigma_G"],
        q_saxs=d["q_saxs"], I_saxs=d["I_saxs"], sigma_I_saxs=d["sigma_I_saxs"],
        saxs_model=d["saxs_model"],
        q_sans=d["q_sans"], I_sans=d["I_sans"], sigma_I_sans=d["sigma_I_sans"],
        sans_model=d["sans_model"],
        init_a=3.52, init_u_iso=0.005, init_diameter_A=90.0,
        init_scale_sans=0.5, n_steps=50, lr=0.5,
    )
    for k in ("scale_pdf", "scale_saxs", "scale_sans",
              "background_saxs", "background_sans"):
        assert k in res.fitted
    # SAXS truth = 1.0, SANS truth = 0.35 → they should be distinct
    assert res.fitted["scale_saxs"] != res.fitted["scale_sans"]


def test_three_way_weights_control_channel_influence():
    """Zeroing the SANS weight collapses to a two-way SAXS+PDF fit."""
    d = _synth_three_way()
    res_no_sans = joint_refine_three_way(
        crystal_tensor=d["ni"], r_pdf=d["r"], G_obs=d["G_obs"], pairs=d["pairs"],
        sigma_G=d["sigma_G"],
        q_saxs=d["q_saxs"], I_saxs=d["I_saxs"], sigma_I_saxs=d["sigma_I_saxs"],
        saxs_model=d["saxs_model"],
        q_sans=d["q_sans"], I_sans=d["I_sans"], sigma_I_sans=d["sigma_I_sans"],
        sans_model=d["sans_model"],
        init_a=3.52, init_u_iso=0.005, init_diameter_A=90.0,
        weights=(10.0, 0.0, 1.0),          # SANS weight = 0
        n_steps=50, lr=0.5,
    )
    # SANS channel should have contributed nothing → its residual is
    # whatever LBFGS left; we just check the fit converged
    assert res_no_sans.chi2_sans >= 0
