"""Rev-12 tests: Bayesian three-way SAXS + SANS + PDF posterior (SVI)."""
from __future__ import annotations

import pytest
import torch

pyro = pytest.importorskip("pyro")


def _synth_three_way(D_true=100.0):
    from midas_hkls import Crystal, Atom, Lattice, SpaceGroup
    from midas_pdf.structure import build_pair_list, pdffit_gr
    from midas_pdf.saxs import SAXSModel, sphere_characteristic_function
    a_true, u_true = 3.524, 0.006
    ni = Crystal(lattice=Lattice(a_true, a_true, a_true, 90, 90, 90),
                  space_group=SpaceGroup.from_number(225),
                  atoms=[Atom(element="Ni", fract=(0, 0, 0))],
                  name="Ni").to_torch()
    r = torch.linspace(0.05, 10.0, 100, dtype=torch.float64)
    pairs = build_pair_list(ni, r_max=12.0)
    G_true = pdffit_gr(ni, r, pairs, scale=1.0, u_iso=u_true) \
             * sphere_characteristic_function(r, D_true)
    saxs = SAXSModel(shape="sphere", polydispersity=0.05, n_poly_nodes=11)
    sans = SAXSModel(shape="sphere", polydispersity=0.05, n_poly_nodes=11)
    q_saxs = torch.linspace(0.005, 0.3, 30, dtype=torch.float64)
    q_sans = torch.linspace(0.003, 0.25, 25, dtype=torch.float64)
    I_saxs = saxs.I(q_saxs, D_median=D_true / 2, scale=1.0)
    I_sans = sans.I(q_sans, D_median=D_true / 2, scale=0.35)
    rng = torch.Generator().manual_seed(0)
    G_obs = G_true + 0.03 * torch.randn(G_true.shape, generator=rng, dtype=torch.float64)
    I_saxs_obs = I_saxs * (1 + 0.05 * torch.randn(I_saxs.shape, generator=rng,
                                                    dtype=torch.float64))
    I_sans_obs = I_sans * (1 + 0.05 * torch.randn(I_sans.shape, generator=rng,
                                                    dtype=torch.float64))
    return dict(ni=ni, r=r, pairs=pairs, G_obs=G_obs,
                 sigma_G=torch.full_like(G_obs, 0.03),
                 q_saxs=q_saxs, I_saxs=I_saxs_obs,
                 sigma_I_saxs=0.05 * I_saxs.abs(), saxs_model=saxs,
                 q_sans=q_sans, I_sans=I_sans_obs,
                 sigma_I_sans=0.05 * I_sans.abs(), sans_model=sans,
                 D_true=D_true, a_true=a_true, u_true=u_true)


def test_three_way_svi_recovers_diameter():
    from midas_pdf.saxs import joint_refine_three_way, joint_three_way_refine_svi
    d = _synth_three_way()
    map_res = joint_refine_three_way(
        crystal_tensor=d["ni"], r_pdf=d["r"], G_obs=d["G_obs"], pairs=d["pairs"],
        sigma_G=d["sigma_G"], q_saxs=d["q_saxs"], I_saxs=d["I_saxs"],
        sigma_I_saxs=d["sigma_I_saxs"], saxs_model=d["saxs_model"],
        q_sans=d["q_sans"], I_sans=d["I_sans"], sigma_I_sans=d["sigma_I_sans"],
        sans_model=d["sans_model"],
        init_a=3.52, init_u_iso=0.005, init_diameter_A=80.0,
        init_scale_sans=0.5, n_steps=40, lr=0.5,
    )
    svi = joint_three_way_refine_svi(
        crystal_tensor=d["ni"], r_pdf=d["r"], G_obs=d["G_obs"], pairs=d["pairs"],
        sigma_G=d["sigma_G"], q_saxs=d["q_saxs"], I_saxs=d["I_saxs"],
        sigma_I_saxs=d["sigma_I_saxs"], saxs_model=d["saxs_model"],
        q_sans=d["q_sans"], I_sans=d["I_sans"], sigma_I_sans=d["sigma_I_sans"],
        sans_model=d["sans_model"], map_init=map_res.fitted,
        n_steps=300, n_posterior_samples=50,
    )
    assert abs(svi.summary()["diameter_A"]["mean"] - d["D_true"]) < 3.0


def test_three_way_svi_result_has_all_channel_samples():
    from midas_pdf.saxs import joint_refine_three_way, joint_three_way_refine_svi
    d = _synth_three_way()
    map_res = joint_refine_three_way(
        crystal_tensor=d["ni"], r_pdf=d["r"], G_obs=d["G_obs"], pairs=d["pairs"],
        sigma_G=d["sigma_G"], q_saxs=d["q_saxs"], I_saxs=d["I_saxs"],
        sigma_I_saxs=d["sigma_I_saxs"], saxs_model=d["saxs_model"],
        q_sans=d["q_sans"], I_sans=d["I_sans"], sigma_I_sans=d["sigma_I_sans"],
        sans_model=d["sans_model"],
        init_a=3.52, init_u_iso=0.005, init_diameter_A=90.0, n_steps=20, lr=0.5,
    )
    svi = joint_three_way_refine_svi(
        crystal_tensor=d["ni"], r_pdf=d["r"], G_obs=d["G_obs"], pairs=d["pairs"],
        sigma_G=d["sigma_G"], q_saxs=d["q_saxs"], I_saxs=d["I_saxs"],
        sigma_I_saxs=d["sigma_I_saxs"], saxs_model=d["saxs_model"],
        q_sans=d["q_sans"], I_sans=d["I_sans"], sigma_I_sans=d["sigma_I_sans"],
        sans_model=d["sans_model"], map_init=map_res.fitted,
        n_steps=100, n_posterior_samples=16,
    )
    assert svi.G_samples.shape == (16, d["r"].shape[0])
    assert svi.I_saxs_samples.shape == (16, d["q_saxs"].shape[0])
    assert svi.I_sans_samples.shape == (16, d["q_sans"].shape[0])
    for name in ("a", "u_iso", "diameter_A", "scale_saxs",
                 "scale_sans", "background_saxs", "background_sans"):
        assert name in svi.posterior_samples


def test_three_way_nuts_smoke_runs():
    """NUTS on the three-way joint must complete without crashing.
    Convergence not required — reliable NUTS on nonlinear structure
    models needs per-problem tuning beyond a unit-test scope."""
    from midas_pdf.saxs import (
        joint_refine_three_way, joint_three_way_refine_nuts,
    )
    d = _synth_three_way()
    map_res = joint_refine_three_way(
        crystal_tensor=d["ni"], r_pdf=d["r"], G_obs=d["G_obs"], pairs=d["pairs"],
        sigma_G=d["sigma_G"], q_saxs=d["q_saxs"], I_saxs=d["I_saxs"],
        sigma_I_saxs=d["sigma_I_saxs"], saxs_model=d["saxs_model"],
        q_sans=d["q_sans"], I_sans=d["I_sans"], sigma_I_sans=d["sigma_I_sans"],
        sans_model=d["sans_model"],
        init_a=3.52, init_u_iso=0.005, init_diameter_A=90.0,
        n_steps=20, lr=0.5,
    )
    res = joint_three_way_refine_nuts(
        crystal_tensor=d["ni"], r_pdf=d["r"], G_obs=d["G_obs"], pairs=d["pairs"],
        sigma_G=d["sigma_G"], q_saxs=d["q_saxs"], I_saxs=d["I_saxs"],
        sigma_I_saxs=d["sigma_I_saxs"], saxs_model=d["saxs_model"],
        q_sans=d["q_sans"], I_sans=d["I_sans"], sigma_I_sans=d["sigma_I_sans"],
        sans_model=d["sans_model"], map_init=map_res.fitted,
        n_warmup=10, n_samples=10,
    )
    for name in ("a", "u_iso", "diameter_A", "scale_saxs", "scale_sans"):
        assert name in res.posterior_samples
        assert res.posterior_samples[name].shape[0] == 10
    assert res.G_samples.shape[0] == 10
    assert res.I_saxs_samples.shape[0] == 10
    assert res.I_sans_samples.shape[0] == 10
    assert res.method == "NUTS"


def test_three_way_svi_decouples_diameter_from_u_iso():
    """The whole point of three-way over two-way is tighter decoupling."""
    from midas_pdf.saxs import joint_refine_three_way, joint_three_way_refine_svi
    d = _synth_three_way()
    map_res = joint_refine_three_way(
        crystal_tensor=d["ni"], r_pdf=d["r"], G_obs=d["G_obs"], pairs=d["pairs"],
        sigma_G=d["sigma_G"], q_saxs=d["q_saxs"], I_saxs=d["I_saxs"],
        sigma_I_saxs=d["sigma_I_saxs"], saxs_model=d["saxs_model"],
        q_sans=d["q_sans"], I_sans=d["I_sans"], sigma_I_sans=d["sigma_I_sans"],
        sans_model=d["sans_model"],
        init_a=3.52, init_u_iso=0.005, init_diameter_A=90.0, n_steps=30, lr=0.5,
    )
    svi = joint_three_way_refine_svi(
        crystal_tensor=d["ni"], r_pdf=d["r"], G_obs=d["G_obs"], pairs=d["pairs"],
        sigma_G=d["sigma_G"], q_saxs=d["q_saxs"], I_saxs=d["I_saxs"],
        sigma_I_saxs=d["sigma_I_saxs"], saxs_model=d["saxs_model"],
        q_sans=d["q_sans"], I_sans=d["I_sans"], sigma_I_sans=d["sigma_I_sans"],
        sans_model=d["sans_model"], map_init=map_res.fitted,
        n_steps=300, n_posterior_samples=100,
    )
    corr = svi.correlation("diameter_A", "u_iso")
    assert abs(corr) < 0.3, f"D vs U_iso correlation {corr:+.3f} too large"
