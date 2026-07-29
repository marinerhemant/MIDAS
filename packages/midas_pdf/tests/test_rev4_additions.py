"""Regression tests for Rev-4 additions to midas-pdf:

* Anomalous scattering (f' + i f'') via Composition.form_factor_averages.
* Laplace posterior sampling via refine_structure(n_posterior_samples=...).
* Container fluorescence combined check.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from midas_pdf.composition import Composition
from midas_pdf.fluorescence import fluorescence_report_sample_and_container
from midas_pdf.placzek import mean_atomic_mass_u


# ---------------------------------------------------------------------------
# 1. Anomalous scattering
# ---------------------------------------------------------------------------

def test_anomalous_disabled_matches_standard():
    q = torch.linspace(1.0, 20.0, 20, dtype=torch.float64)
    c = Composition({"Ni": 1})
    f_std, f2_std = c.form_factor_averages(q)
    # anomalous=False → same as no argument
    f_a, f2_a = c.form_factor_averages(q, wavelength_A=0.1839, anomalous=False)
    assert torch.allclose(f_std, f_a)
    assert torch.allclose(f2_std, f2_a)


def test_anomalous_perturbs_form_factor():
    """Anomalous should change f² by O(0.1-few%) for Ni at 63 keV."""
    q = torch.linspace(1.0, 20.0, 8, dtype=torch.float64)
    c = Composition({"Ni": 1})
    _, f2_std = c.form_factor_averages(q)
    _, f2_an  = c.form_factor_averages(q, wavelength_A=0.1839, anomalous=True)
    frac = (f2_an - f2_std) / f2_std
    # For Ni at 63 keV, anomalous is a small positive correction that grows
    # with Q (f' constant, f0 decreases → relative bump increases).
    assert torch.all(torch.abs(frac) < 0.05)     # never more than 5%
    assert torch.all(frac > -1e-3)               # positive for Ni at this E
    assert float(frac[-1]) > float(frac[0])      # grows with Q


def test_anomalous_ce_negative():
    """Ce at 63 keV has f' < 0 (below L3 edge dispersion) → f² decreases."""
    q = torch.linspace(1.0, 15.0, 5, dtype=torch.float64)
    c = Composition({"Ce": 1, "O": 2})
    _, f2_std = c.form_factor_averages(q)
    _, f2_an  = c.form_factor_averages(q, wavelength_A=0.1839, anomalous=True)
    # CeO₂ f² should decrease by ~ 0.3-0.5% (Ce dominates, f' negative)
    frac = (f2_an - f2_std) / f2_std
    assert torch.all(frac < 0)
    assert torch.all(torch.abs(frac) < 0.02)


def test_anomalous_requires_wavelength():
    """With anomalous=True but wavelength_A=None, must silently fall back."""
    q = torch.linspace(1.0, 10.0, 5, dtype=torch.float64)
    c = Composition({"Ni": 1})
    f_std, f2_std = c.form_factor_averages(q)
    # anomalous=True but no wavelength → same as standard (fallback path)
    f_a, f2_a = c.form_factor_averages(q, anomalous=True)
    assert torch.allclose(f_std, f_a)
    assert torch.allclose(f2_std, f2_a)


# ---------------------------------------------------------------------------
# 2. Laplace posterior sampling in refine_structure
# ---------------------------------------------------------------------------

def _synthesize_ni_G(a_true=3.524, u_true=0.006, n_r=800, r_max=8.0, noise=0.02):
    """Fake a Ni G(r) for the small-box refiner test."""
    from midas_pdf.structure import build_pair_list, pdffit_gr
    from midas_hkls import Crystal, Atom, Lattice, SpaceGroup
    ni = Crystal(
        lattice=Lattice(a_true, a_true, a_true, 90, 90, 90),
        space_group=SpaceGroup.from_number(225),
        atoms=[Atom(element="Ni", fract=(0, 0, 0))],
        name="Ni",
    ).to_torch()
    r = torch.linspace(0.05, r_max, n_r, dtype=torch.float64)
    pairs = build_pair_list(ni, r_max=r_max + 1.0)
    with torch.no_grad():
        G_true = pdffit_gr(ni, r, pairs, scale=1.0, u_iso=u_true)
    rng = torch.Generator().manual_seed(0)
    G_noisy = G_true + noise * torch.randn(G_true.shape, generator=rng, dtype=torch.float64)
    sigma = torch.full_like(G_noisy, noise)
    return ni, r, G_noisy, pairs, sigma, a_true


def test_refine_laplace_posterior_samples():
    from midas_pdf.structure import refine_structure
    ni, r, G, pairs, sig, a_true = _synthesize_ni_G()
    res = refine_structure(
        ni, r, G, pairs, sigma_obs=sig,
        init_a=a_true - 0.005, init_u_iso=0.004, init_scale=1.0,
        n_posterior_samples=50,
    )
    assert abs(res.fitted["a"] - a_true) < 0.005
    assert res.posterior is not None and "theta_samples" in res.posterior
    assert res.posterior["theta_samples"].shape == (50, 3)
    assert res.posterior["G_samples"].shape[0] == 50
    a_samples = res.posterior["a_samples"]
    # posterior std on a should be close to the reported Hessian σ
    hess_sig = res.uncertainty["a"]
    post_sig = float(a_samples.std())
    if hess_sig > 0 and np.isfinite(hess_sig):
        # within a factor of a few (small sample count → noisy estimate)
        assert 0.2 < post_sig / hess_sig < 5.0, (post_sig, hess_sig)


def test_refine_no_posterior_when_samples_zero():
    from midas_pdf.structure import refine_structure
    ni, r, G, pairs, sig, a_true = _synthesize_ni_G()
    res = refine_structure(
        ni, r, G, pairs, sigma_obs=sig,
        init_a=a_true - 0.005,
        n_posterior_samples=0,        # default → no posterior
    )
    assert res.posterior is None
    # covariance should still be attached
    assert res.cov is not None


# ---------------------------------------------------------------------------
# 3. Container fluorescence
# ---------------------------------------------------------------------------

def _K_lines(lines):
    """Keep only K-shell lines with a real yield (filter out soft L edges)."""
    return [d for d in lines
            if d["shell"] == "K" and d.get("yield") is not None]


def test_fluorescence_sample_and_kapton_container_clean_at_67_keV():
    """Ni + Kapton container at 67.4 keV: only Ni K fires; Kapton clean at K."""
    sample = Composition({"Ni": 1})
    kapton = Composition({"C": 22, "H": 10, "N": 2, "O": 5})
    rep = fluorescence_report_sample_and_container(
        sample, kapton, incident_energy_keV=67.42, min_yield=0.05)
    ni_lines = _K_lines(rep["sample_lines"])
    kap_lines = _K_lines(rep["container_lines"])
    assert ni_lines and abs(ni_lines[0]["line_keV"] - 7.48) < 0.1
    assert kap_lines == []               # Kapton has no K-shell fluorescence here


def test_fluorescence_flags_steel_container():
    """Steel-can container (Fe/Cr/Ni) at 67.4 keV: multiple K lines flagged."""
    sample = Composition({"C": 1})           # graphite, no K fluorescence
    steel = Composition({"Fe": 0.7, "Cr": 0.18, "Ni": 0.12})
    rep = fluorescence_report_sample_and_container(
        sample, steel, incident_energy_keV=67.42, min_yield=0.05)
    assert _K_lines(rep["sample_lines"]) == []
    container_elements = {d["element"] for d in _K_lines(rep["container_lines"])}
    assert container_elements >= {"Fe", "Cr", "Ni"}


def test_fluorescence_no_container_matches_original():
    """With no container, container_lines is empty regardless."""
    sample = Composition({"Cu": 1})
    rep = fluorescence_report_sample_and_container(
        sample, container_composition=None, incident_energy_keV=67.42)
    assert rep["container_lines"] == []
    assert any(d["element"] == "Cu" for d in _K_lines(rep["sample_lines"]))


# ---------------------------------------------------------------------------
# 4. Atomic mass utility (used by cylinder MS and future modules)
# ---------------------------------------------------------------------------

def test_mean_atomic_mass_ni():
    assert abs(mean_atomic_mass_u(Composition({"Ni": 1})) - 58.693) < 0.01


def test_mean_atomic_mass_kapton():
    # C22H10N2O5, mole-fraction-weighted mean atomic mass
    M = mean_atomic_mass_u(Composition({"C": 22, "H": 10, "N": 2, "O": 5}))
    expected = (22*12.011 + 10*1.008 + 2*14.007 + 5*15.999) / (22+10+2+5)
    assert abs(M - expected) < 1e-3


def test_mean_atomic_mass_rejects_unknown():
    with pytest.raises(ValueError):
        mean_atomic_mass_u(Composition({"Uue": 1}))  # ununennium not tabulated
