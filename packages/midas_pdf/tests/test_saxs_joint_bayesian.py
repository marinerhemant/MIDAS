"""Rev-8 tests: Bayesian joint SAXS+PDF posterior via Pyro SVI/NUTS."""
from __future__ import annotations

import pytest
import torch

pyro = pytest.importorskip("pyro")


def _synth_joint():
    from midas_hkls import Crystal, Atom, Lattice, SpaceGroup
    from midas_pdf.structure import build_pair_list, pdffit_gr
    from midas_pdf.saxs import SAXSModel, sphere_characteristic_function
    a_true, D_true, u_true = 3.524, 100.0, 0.006
    ni = Crystal(lattice=Lattice(a_true, a_true, a_true, 90, 90, 90),
                  space_group=SpaceGroup.from_number(225),
                  atoms=[Atom(element="Ni", fract=(0, 0, 0))], name="Ni").to_torch()
    r = torch.linspace(0.05, 10.0, 120, dtype=torch.float64)
    pairs = build_pair_list(ni, r_max=12.0)
    G_true = pdffit_gr(ni, r, pairs, scale=1.0, u_iso=u_true) \
             * sphere_characteristic_function(r, D_true)
    saxs = SAXSModel(shape="sphere", polydispersity=0.05, n_poly_nodes=11)
    q = torch.linspace(0.005, 0.3, 40, dtype=torch.float64)
    I_true = saxs.I(q, D_median=D_true / 2, scale=1.0, background=0.0)
    rng = torch.Generator().manual_seed(0)
    G_obs = G_true + 0.03 * torch.randn(G_true.shape, generator=rng,
                                          dtype=torch.float64)
    I_obs = I_true * (1 + 0.05 * torch.randn(I_true.shape, generator=rng,
                                               dtype=torch.float64))
    return dict(ni=ni, r=r, pairs=pairs, G_obs=G_obs,
                 sigma_G=torch.full_like(G_obs, 0.03),
                 q_saxs=q, I_saxs=I_obs, sigma_I=0.05 * I_true.abs(),
                 saxs_model=saxs, D_true=D_true, a_true=a_true, u_true=u_true)


def test_joint_svi_recovers_diameter():
    from midas_pdf.saxs import joint_refine, joint_refine_svi
    d = _synth_joint()
    map_res = joint_refine(
        crystal_tensor=d["ni"], r_pdf=d["r"], G_obs=d["G_obs"], pairs=d["pairs"],
        sigma_G=d["sigma_G"], q_saxs=d["q_saxs"], I_saxs=d["I_saxs"],
        sigma_I=d["sigma_I"], saxs_model=d["saxs_model"],
        init_a=3.52, init_u_iso=0.005, init_diameter_A=90.0,
        n_steps=40, lr=0.5,
    )
    res = joint_refine_svi(
        crystal_tensor=d["ni"], r_pdf=d["r"], G_obs=d["G_obs"], pairs=d["pairs"],
        sigma_G=d["sigma_G"], q_saxs=d["q_saxs"], I_saxs=d["I_saxs"],
        sigma_I=d["sigma_I"], saxs_model=d["saxs_model"],
        map_init=map_res.fitted, n_steps=400, n_posterior_samples=100,
    )
    D_mean = res.summary()["diameter_A"]["mean"]
    assert abs(D_mean - d["D_true"]) < 3.0


def test_joint_svi_result_shapes():
    from midas_pdf.saxs import joint_refine, joint_refine_svi
    d = _synth_joint()
    map_res = joint_refine(
        crystal_tensor=d["ni"], r_pdf=d["r"], G_obs=d["G_obs"], pairs=d["pairs"],
        sigma_G=d["sigma_G"], q_saxs=d["q_saxs"], I_saxs=d["I_saxs"],
        sigma_I=d["sigma_I"], saxs_model=d["saxs_model"],
        init_a=3.52, init_u_iso=0.005, init_diameter_A=90.0,
        n_steps=30, lr=0.5,
    )
    res = joint_refine_svi(
        crystal_tensor=d["ni"], r_pdf=d["r"], G_obs=d["G_obs"], pairs=d["pairs"],
        sigma_G=d["sigma_G"], q_saxs=d["q_saxs"], I_saxs=d["I_saxs"],
        sigma_I=d["sigma_I"], saxs_model=d["saxs_model"],
        map_init=map_res.fitted, n_steps=100, n_posterior_samples=32,
    )
    assert res.G_samples.shape == (32, d["r"].shape[0])
    assert res.I_saxs_samples.shape == (32, d["q_saxs"].shape[0])
    for name in ("a", "u_iso", "diameter_A", "scale_pdf",
                 "scale_saxs", "background_saxs"):
        assert name in res.posterior_samples


def test_joint_svi_correlation_low_on_well_identified_problem():
    """With a good SAXS + PDF dataset, (D, U_iso) should be nearly
    uncorrelated in the posterior — the identifiability the joint fit
    is designed for."""
    from midas_pdf.saxs import joint_refine, joint_refine_svi
    d = _synth_joint()
    map_res = joint_refine(
        crystal_tensor=d["ni"], r_pdf=d["r"], G_obs=d["G_obs"], pairs=d["pairs"],
        sigma_G=d["sigma_G"], q_saxs=d["q_saxs"], I_saxs=d["I_saxs"],
        sigma_I=d["sigma_I"], saxs_model=d["saxs_model"],
        init_a=3.52, init_u_iso=0.005, init_diameter_A=90.0,
        n_steps=30, lr=0.5,
    )
    res = joint_refine_svi(
        crystal_tensor=d["ni"], r_pdf=d["r"], G_obs=d["G_obs"], pairs=d["pairs"],
        sigma_G=d["sigma_G"], q_saxs=d["q_saxs"], I_saxs=d["I_saxs"],
        sigma_I=d["sigma_I"], saxs_model=d["saxs_model"],
        map_init=map_res.fitted, n_steps=400, n_posterior_samples=200,
    )
    corr = res.correlation("diameter_A", "u_iso")
    assert abs(corr) < 0.3, f"D vs U_iso correlation {corr:+.3f} too big — decoupling failed"


def test_joint_nuts_smoke_runs():
    """NUTS on the joint SAXS+PDF model must complete without crashing —
    not required to converge in unit-test timescales."""
    from midas_pdf.saxs import joint_refine, joint_refine_nuts
    d = _synth_joint()
    map_res = joint_refine(
        crystal_tensor=d["ni"], r_pdf=d["r"], G_obs=d["G_obs"], pairs=d["pairs"],
        sigma_G=d["sigma_G"], q_saxs=d["q_saxs"], I_saxs=d["I_saxs"],
        sigma_I=d["sigma_I"], saxs_model=d["saxs_model"],
        init_a=3.52, init_u_iso=0.005, init_diameter_A=90.0,
        n_steps=20, lr=0.5,
    )
    res = joint_refine_nuts(
        crystal_tensor=d["ni"], r_pdf=d["r"], G_obs=d["G_obs"], pairs=d["pairs"],
        sigma_G=d["sigma_G"], q_saxs=d["q_saxs"], I_saxs=d["I_saxs"],
        sigma_I=d["sigma_I"], saxs_model=d["saxs_model"],
        map_init=map_res.fitted, n_warmup=10, n_samples=10,
    )
    for name in ("a", "u_iso", "diameter_A"):
        assert res.posterior_samples[name].shape[0] == 10
