"""Regression tests for the Pyro-based Bayesian refinement (SVI + NUTS).

Requires ``pyro-ppl`` (the ``[bayes]`` extra). We test SVI convergence
against MAP + Laplace on a Gaussian-likelihood synthetic problem where
the true posterior IS Gaussian, so SVI should agree with Laplace within
statistical noise.

NUTS is smoke-tested only: it runs without crashing. Reliable NUTS
sampling on nonlinear structure models requires per-problem tuning
(step size, mass matrix, multiple chains) that's beyond a unit-test
scope.
"""
from __future__ import annotations

import pytest
import torch

pyro = pytest.importorskip("pyro")


def _synth_ni():
    from midas_hkls import Crystal, Atom, Lattice, SpaceGroup
    from midas_pdf.structure import build_pair_list, pdffit_gr
    a_true, u_true = 3.524, 0.006
    ni = Crystal(
        lattice=Lattice(a_true, a_true, a_true, 90, 90, 90),
        space_group=SpaceGroup.from_number(225),
        atoms=[Atom(element="Ni", fract=(0, 0, 0))], name="Ni",
    ).to_torch()
    r = torch.linspace(0.05, 8.0, 200, dtype=torch.float64)
    pairs = build_pair_list(ni, r_max=9.0)
    with torch.no_grad():
        G_true = pdffit_gr(ni, r, pairs, scale=1.0, u_iso=u_true)
    rng = torch.Generator().manual_seed(0)
    noise = 0.03
    G_obs = G_true + noise * torch.randn(G_true.shape, generator=rng, dtype=torch.float64)
    sig = torch.full_like(G_obs, noise)
    return ni, r, G_obs, pairs, sig, a_true, u_true


def test_svi_recovers_map_mean_within_std():
    """SVI posterior mean must sit within a few σ of the MAP for a nearly-
    Gaussian likelihood."""
    from midas_pdf.structure import refine_structure
    from midas_pdf.bayesian_refine import bayesian_refine_svi
    ni, r, G_obs, pairs, sig, a_true, u_true = _synth_ni()
    lap = refine_structure(ni, r, G_obs, pairs, sigma_obs=sig,
                            init_a=3.52, init_u_iso=0.005,
                            n_posterior_samples=50)
    svi = bayesian_refine_svi(
        ni, r, G_obs, pairs, sigma_obs=sig,
        map_init=lap.fitted, n_steps=500, n_posterior_samples=200)
    s = svi.summary()
    assert abs(s["a"]["mean"] - lap.fitted["a"]) < 0.005, s["a"]
    assert abs(s["u_iso"]["mean"] - lap.fitted["u_iso"]) < 0.002, s["u_iso"]


def test_svi_posterior_std_positive():
    from midas_pdf.structure import refine_structure
    from midas_pdf.bayesian_refine import bayesian_refine_svi
    ni, r, G_obs, pairs, sig, *_ = _synth_ni()
    lap = refine_structure(ni, r, G_obs, pairs, sigma_obs=sig,
                            init_a=3.52, init_u_iso=0.005)
    svi = bayesian_refine_svi(
        ni, r, G_obs, pairs, sigma_obs=sig, map_init=lap.fitted,
        n_steps=300, n_posterior_samples=100)
    s = svi.summary()
    # every parameter's posterior must have non-zero spread
    for name in ("a", "u_iso", "scale"):
        assert s[name]["std"] > 0, (name, s[name])
    # G(r) posterior band must also be non-degenerate
    assert float(svi.G_std.mean()) > 0


def test_svi_g_samples_shape():
    from midas_pdf.structure import refine_structure
    from midas_pdf.bayesian_refine import bayesian_refine_svi
    ni, r, G_obs, pairs, sig, *_ = _synth_ni()
    lap = refine_structure(ni, r, G_obs, pairs, sigma_obs=sig,
                            init_a=3.52, init_u_iso=0.005)
    svi = bayesian_refine_svi(
        ni, r, G_obs, pairs, sigma_obs=sig, map_init=lap.fitted,
        n_steps=200, n_posterior_samples=64)
    assert svi.G_samples.shape == (64, r.shape[0])
    assert svi.G_mean.shape == r.shape
    assert svi.G_std.shape == r.shape


def test_nuts_smoke_runs_without_crash():
    """NUTS must complete without exceptions. Not tested for convergence:
    that would require per-problem tuning outside the scope of unit tests."""
    from midas_pdf.structure import refine_structure
    from midas_pdf.bayesian_refine import bayesian_refine_nuts
    ni, r, G_obs, pairs, sig, *_ = _synth_ni()
    lap = refine_structure(ni, r, G_obs, pairs, sigma_obs=sig,
                            init_a=3.52, init_u_iso=0.005)
    # 30 warmup + 30 samples: fast smoke; won't converge but must not crash
    nuts = bayesian_refine_nuts(
        ni, r, G_obs, pairs, sigma_obs=sig, map_init=lap.fitted,
        n_warmup=30, n_samples=30)
    for name in ("a", "u_iso", "scale"):
        assert name in nuts.posterior_samples
        assert nuts.posterior_samples[name].shape[0] == 30
    assert nuts.G_samples.shape[0] == 30


def test_result_summary_keys():
    from midas_pdf.structure import refine_structure
    from midas_pdf.bayesian_refine import bayesian_refine_svi
    ni, r, G_obs, pairs, sig, *_ = _synth_ni()
    lap = refine_structure(ni, r, G_obs, pairs, sigma_obs=sig,
                            init_a=3.52, init_u_iso=0.005)
    svi = bayesian_refine_svi(
        ni, r, G_obs, pairs, sigma_obs=sig, map_init=lap.fitted,
        n_steps=100, n_posterior_samples=32)
    s = svi.summary()
    for name in ("a", "u_iso", "scale"):
        assert set(s[name].keys()) >= {"mean", "std", "q05", "q95"}
