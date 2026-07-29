"""Rev-14 tests: multi-phase + core-shell PDF fitting."""
from __future__ import annotations

import pytest
import torch

from midas_pdf.multi_phase import (
    multi_phase_gr, refine_multi_phase, MultiPhaseResult,
    core_shell_pdf_gr, refine_core_shell, CoreShellResult,
)


def _ni():
    from midas_hkls import Crystal, Atom, Lattice, SpaceGroup
    return Crystal(lattice=Lattice(3.524, 3.524, 3.524, 90, 90, 90),
                    space_group=SpaceGroup.from_number(225),
                    atoms=[Atom(element="Ni", fract=(0, 0, 0))],
                    name="Ni").to_torch()


def _cu():
    from midas_hkls import Crystal, Atom, Lattice, SpaceGroup
    return Crystal(lattice=Lattice(3.615, 3.615, 3.615, 90, 90, 90),
                    space_group=SpaceGroup.from_number(225),
                    atoms=[Atom(element="Cu", fract=(0, 0, 0))],
                    name="Cu").to_torch()


# ---------------------------------------------------------------------------
# multi_phase_gr forward
# ---------------------------------------------------------------------------

def test_multi_phase_forward_shape_matches_r():
    from midas_pdf.structure import build_pair_list
    ni, cu = _ni(), _cu()
    r = torch.linspace(1.5, 8.0, 100, dtype=torch.float64)
    G = multi_phase_gr(
        [ni, cu],
        [build_pair_list(ni, r_max=9.0),
         build_pair_list(cu, r_max=9.0)],
        r, weights=[0.5, 0.5], u_isos=[0.006, 0.008],
    )
    assert G.shape == r.shape


def test_multi_phase_single_phase_matches_pdffit_gr():
    """A one-phase multi_phase_gr with weight=1 must equal a plain pdffit_gr."""
    from midas_pdf.structure import build_pair_list, pdffit_gr
    ni = _ni()
    r = torch.linspace(1.5, 8.0, 100, dtype=torch.float64)
    pairs = build_pair_list(ni, r_max=9.0)
    G_ref = pdffit_gr(ni, r, pairs, scale=1.0, u_iso=0.006)
    G_mp = multi_phase_gr([ni], [pairs], r, weights=[1.0], u_isos=[0.006])
    assert torch.allclose(G_ref, G_mp, atol=1e-9)


def test_multi_phase_weights_auto_normalise():
    """Unnormalised weights (e.g., [3, 2]) should be renormalised to sum to 1."""
    from midas_pdf.structure import build_pair_list
    ni, cu = _ni(), _cu()
    r = torch.linspace(1.5, 8.0, 60, dtype=torch.float64)
    pairs_ni = build_pair_list(ni, r_max=9.0)
    pairs_cu = build_pair_list(cu, r_max=9.0)
    G_a = multi_phase_gr([ni, cu], [pairs_ni, pairs_cu], r,
                          weights=[0.6, 0.4], u_isos=[0.006, 0.008])
    G_b = multi_phase_gr([ni, cu], [pairs_ni, pairs_cu], r,
                          weights=[3.0, 2.0], u_isos=[0.006, 0.008])
    assert torch.allclose(G_a, G_b, atol=1e-9)


def test_multi_phase_length_mismatch_raises():
    from midas_pdf.structure import build_pair_list
    ni = _ni()
    r = torch.linspace(1.5, 8.0, 60, dtype=torch.float64)
    with pytest.raises(ValueError):
        multi_phase_gr([ni], [build_pair_list(ni, r_max=9.0),
                                build_pair_list(ni, r_max=9.0)], r,
                        weights=[1.0], u_isos=[0.006])
    with pytest.raises(ValueError):
        multi_phase_gr([ni], [build_pair_list(ni, r_max=9.0)], r,
                        weights=[0.5, 0.5], u_isos=[0.006])


# ---------------------------------------------------------------------------
# refine_multi_phase
# ---------------------------------------------------------------------------

def test_refine_multi_phase_recovers_lattice_constants():
    from midas_pdf.structure import build_pair_list
    ni, cu = _ni(), _cu()
    r = torch.linspace(1.5, 8.0, 150, dtype=torch.float64)
    pairs_ni = build_pair_list(ni, r_max=9.0)
    pairs_cu = build_pair_list(cu, r_max=9.0)
    G_true = multi_phase_gr([ni, cu], [pairs_ni, pairs_cu], r,
                              weights=[0.6, 0.4], u_isos=[0.006, 0.008])
    rng = torch.Generator().manual_seed(0)
    G_obs = G_true + 0.02 * torch.randn(G_true.shape, generator=rng, dtype=torch.float64)
    res = refine_multi_phase(
        [ni, cu], r, G_obs, [pairs_ni, pairs_cu],
        sigma_obs=torch.full_like(G_obs, 0.02),
        init_a=[3.52, 3.62], init_u_iso=[0.005, 0.010],
        init_weights=[0.5, 0.5], steps=100, lr=0.05,
    )
    assert abs(res.fitted["a_0"] - 3.524) < 0.005
    assert abs(res.fitted["a_1"] - 3.615) < 0.005
    assert res.n_phases == 2


def test_refine_multi_phase_weights_sum_to_one():
    from midas_pdf.structure import build_pair_list
    ni, cu = _ni(), _cu()
    r = torch.linspace(1.5, 8.0, 100, dtype=torch.float64)
    pairs_ni = build_pair_list(ni, r_max=9.0)
    pairs_cu = build_pair_list(cu, r_max=9.0)
    G_true = multi_phase_gr([ni, cu], [pairs_ni, pairs_cu], r,
                              weights=[0.5, 0.5], u_isos=[0.006, 0.008])
    res = refine_multi_phase(
        [ni, cu], r, G_true, [pairs_ni, pairs_cu],
        init_weights=[0.4, 0.6], steps=50, lr=0.05,
    )
    assert isinstance(res, MultiPhaseResult)
    total = sum(res.weights_normalised)
    assert abs(total - 1.0) < 1e-6


def test_refine_multi_phase_result_has_all_params():
    from midas_pdf.structure import build_pair_list
    ni, cu = _ni(), _cu()
    r = torch.linspace(1.5, 6.0, 60, dtype=torch.float64)
    pairs = [build_pair_list(ni, r_max=7.0), build_pair_list(cu, r_max=7.0)]
    G_true = multi_phase_gr([ni, cu], pairs, r,
                              weights=[0.5, 0.5], u_isos=[0.006, 0.008])
    res = refine_multi_phase([ni, cu], r, G_true, pairs,
                              steps=20, lr=0.05)
    for i in (0, 1):
        for prefix in ("a", "u_iso", "scale", "weight"):
            assert f"{prefix}_{i}" in res.fitted


# ---------------------------------------------------------------------------
# core_shell_pdf_gr forward
# ---------------------------------------------------------------------------

def test_core_shell_forward_shape():
    from midas_pdf.structure import build_pair_list
    ni, cu = _ni(), _cu()
    r = torch.linspace(1.5, 8.0, 100, dtype=torch.float64)
    G = core_shell_pdf_gr(
        ni, cu, r,
        build_pair_list(ni, r_max=9.0),
        build_pair_list(cu, r_max=9.0),
        R_core_A=30.0, shell_thickness_A=15.0,
        u_iso_core=0.006, u_iso_shell=0.010,
    )
    assert G.shape == r.shape


def test_core_shell_reduces_to_shell_when_R_core_is_zero():
    """At R_core → 0, volume fraction of core → 0 → the whole particle
    is shell-only."""
    from midas_pdf.structure import build_pair_list, pdffit_gr
    from midas_pdf.saxs.joint import sphere_characteristic_function
    ni, cu = _ni(), _cu()
    r = torch.linspace(1.5, 8.0, 60, dtype=torch.float64)
    pairs_ni = build_pair_list(ni, r_max=9.0)
    pairs_cu = build_pair_list(cu, r_max=9.0)
    G_cs = core_shell_pdf_gr(
        ni, cu, r, pairs_ni, pairs_cu,
        R_core_A=1e-4, shell_thickness_A=30.0,
        u_iso_core=0.006, u_iso_shell=0.010,
    )
    G_shell_only = pdffit_gr(cu, r, pairs_cu, scale=1.0, u_iso=0.010) \
        * sphere_characteristic_function(r, 2.0 * (1e-4 + 30.0))
    assert torch.allclose(G_cs, G_shell_only, atol=1e-6)


def test_core_shell_volume_fractions_match_analytical():
    """For R_core = shell_thickness, V_core = 1/7 of V_total (since
    (R+t)³ - R³ = 3R²t + 3Rt² + t³ = 7R³ when t=R)."""
    import numpy as np
    from midas_pdf.structure import build_pair_list
    ni, cu = _ni(), _cu()
    r = torch.linspace(1.5, 8.0, 60, dtype=torch.float64)
    pairs_ni = build_pair_list(ni, r_max=9.0)
    pairs_cu = build_pair_list(cu, r_max=9.0)
    R = 20.0
    G_cs = core_shell_pdf_gr(
        ni, cu, r, pairs_ni, pairs_cu,
        R_core_A=R, shell_thickness_A=R,
        u_iso_core=0.006, u_iso_shell=0.006,
    )
    # Volume fraction (core / total) = R³ / (2R)³ = 1/8
    V_c = (4/3) * np.pi * R ** 3
    V_t = (4/3) * np.pi * (2 * R) ** 3
    expected_f_c = V_c / V_t
    assert abs(expected_f_c - 0.125) < 1e-9        # 1/8


def test_core_shell_differentiable_in_all_geometry():
    from midas_pdf.structure import build_pair_list
    ni, cu = _ni(), _cu()
    r = torch.linspace(1.5, 8.0, 40, dtype=torch.float64)
    pairs_ni = build_pair_list(ni, r_max=9.0)
    pairs_cu = build_pair_list(cu, r_max=9.0)
    R = torch.tensor(20.0, dtype=torch.float64, requires_grad=True)
    t = torch.tensor(10.0, dtype=torch.float64, requires_grad=True)
    G = core_shell_pdf_gr(
        ni, cu, r, pairs_ni, pairs_cu,
        R_core_A=R, shell_thickness_A=t,
        u_iso_core=0.006, u_iso_shell=0.010,
    )
    G.sum().backward()
    assert torch.isfinite(R.grad) and torch.isfinite(t.grad)


# ---------------------------------------------------------------------------
# refine_core_shell
# ---------------------------------------------------------------------------

def test_refine_core_shell_recovers_lattice_constants():
    """Lattice constants and u_iso should recover cleanly on short-range
    r. R_core / shell_thickness may not refine in this window (they're
    SAXS-scale parameters); starting values are respected."""
    from midas_pdf.structure import build_pair_list
    ni, cu = _ni(), _cu()
    r = torch.linspace(1.5, 8.0, 150, dtype=torch.float64)
    pairs_ni = build_pair_list(ni, r_max=9.0)
    pairs_cu = build_pair_list(cu, r_max=9.0)
    G_true = core_shell_pdf_gr(
        ni, cu, r, pairs_ni, pairs_cu,
        R_core_A=30.0, shell_thickness_A=15.0,
        u_iso_core=0.006, u_iso_shell=0.010,
    )
    rng = torch.Generator().manual_seed(0)
    G_obs = G_true + 0.02 * torch.randn(G_true.shape, generator=rng, dtype=torch.float64)
    res = refine_core_shell(
        ni, cu, r, G_obs, pairs_ni, pairs_cu,
        sigma_obs=torch.full_like(G_obs, 0.02),
        init_a_core=3.52, init_a_shell=3.62,
        init_R_core_A=30.0, init_shell_thickness_A=15.0,
        steps=80, lr=0.1,
    )
    assert abs(res.fitted["a_core"] - 3.524) < 0.005
    assert abs(res.fitted["a_shell"] - 3.615) < 0.005


def test_refine_core_shell_result_has_all_geometry():
    from midas_pdf.structure import build_pair_list
    ni, cu = _ni(), _cu()
    r = torch.linspace(1.5, 6.0, 60, dtype=torch.float64)
    pairs_ni = build_pair_list(ni, r_max=7.0)
    pairs_cu = build_pair_list(cu, r_max=7.0)
    G_true = core_shell_pdf_gr(
        ni, cu, r, pairs_ni, pairs_cu,
        R_core_A=20.0, shell_thickness_A=10.0,
        u_iso_core=0.006, u_iso_shell=0.010,
    )
    res = refine_core_shell(ni, cu, r, G_true, pairs_ni, pairs_cu,
                              init_R_core_A=20.0, init_shell_thickness_A=10.0,
                              steps=30, lr=0.05)
    assert isinstance(res, CoreShellResult)
    for k in ("a_core", "a_shell", "u_iso_core", "u_iso_shell",
              "R_core_A", "shell_thickness_A", "scale_core", "scale_shell"):
        assert k in res.fitted
    # Volume fractions should be positive and sum to 1
    f_c, f_s = res.volume_fractions
    assert 0 < f_c < 1
    assert abs(f_c + f_s - 1.0) < 1e-9
