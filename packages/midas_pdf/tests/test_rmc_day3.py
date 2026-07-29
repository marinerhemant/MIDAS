"""Day-3 RMC tests: coordination, partial g(r), ergodicity, bias."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from midas_pdf.rmc import (
    Supercell, coordination_number, partial_g_r, ergodicity_diagnostics,
    CoordinationBias, DisplaceMove, rmc_refine, supercell_G_r,
)


def _fcc_ni(a=3.524):
    from midas_hkls import Crystal, Atom, Lattice, SpaceGroup
    return Crystal(
        lattice=Lattice(a, a, a, 90, 90, 90),
        space_group=SpaceGroup.from_number(225),
        atoms=[Atom(element="Ni", fract=(0, 0, 0))], name="Ni",
    ).to_torch()


# ---------------------------------------------------------------------------
# Coordination number
# ---------------------------------------------------------------------------

def test_coordination_fcc_first_shell_is_12():
    """First-shell coordination in FCC is exactly 12."""
    ni = _fcc_ni()
    sc = Supercell.from_crystal(ni, size=(4, 4, 4))
    info = coordination_number(sc, r_shell=(2.0, 2.7))
    assert info["Z_mean"] == pytest.approx(12.0, abs=0.5)
    assert info["Z_std"] == pytest.approx(0.0, abs=0.5)


def test_coordination_fcc_second_shell_is_6():
    """Second shell in FCC is 6 (a = 3.524)."""
    ni = _fcc_ni()
    sc = Supercell.from_crystal(ni, size=(5, 5, 5))
    info = coordination_number(sc, r_shell=(3.0, 3.7))
    assert info["Z_mean"] == pytest.approx(6.0, abs=0.5)


def test_coordination_empty_shell_returns_zero():
    ni = _fcc_ni()
    sc = Supercell.from_crystal(ni, size=(3, 3, 3))
    info = coordination_number(sc, r_shell=(1.0, 1.5))    # too small for FCC
    assert info["Z_mean"] == 0.0
    assert info["Z_std"] == 0.0


# ---------------------------------------------------------------------------
# Partial g(r)
# ---------------------------------------------------------------------------

def test_partial_g_r_monoatomic_returns_single_entry():
    ni = _fcc_ni()
    sc = Supercell.from_crystal(ni, size=(4, 4, 4))
    r = torch.linspace(1.0, 5.0, 100, dtype=torch.float64)
    g = partial_g_r(sc, r, bin_width=float(r[1] - r[0]))
    # monoatomic (species set has one element) → one entry (X, X)
    assert len(g) == 1


def test_partial_g_r_first_shell_peak_position():
    """Peak of g_{NiNi}(r) must be near a/√2 for FCC Ni."""
    ni = _fcc_ni()
    sc = Supercell.from_crystal(ni, size=(4, 4, 4))
    r = torch.linspace(1.0, 5.0, 200, dtype=torch.float64)
    g = partial_g_r(sc, r, bin_width=float(r[1] - r[0]))
    (_, _), g_vals = next(iter(g.items())), None
    for _, v in g.items():
        g_vals = v
    peak_r = float(r[int(g_vals.argmax())])
    assert abs(peak_r - 3.524 / np.sqrt(2)) < 0.05


def test_partial_g_r_multispecies_has_ab_entries():
    """A binary supercell should have {(A,A), (A,B), (B,B)} keys."""
    positions = torch.randn(20, 3, dtype=torch.float64) * 5.0
    species = ["Ni"] * 10 + ["Fe"] * 10
    sc = Supercell(species=species, positions=positions,
                    cell=torch.eye(3) * 20.0)
    r = torch.linspace(0.5, 8.0, 60, dtype=torch.float64)
    g = partial_g_r(sc, r, bin_width=float(r[1] - r[0]))
    keys = set(g.keys())
    assert ("Fe", "Fe") in keys
    assert ("Fe", "Ni") in keys or ("Ni", "Fe") in keys
    assert ("Ni", "Ni") in keys


# ---------------------------------------------------------------------------
# Ergodicity diagnostics
# ---------------------------------------------------------------------------

def test_ergodicity_diagnostics_reports_all_keys():
    ni = _fcc_ni()
    sc = Supercell.from_crystal(ni, size=(3, 3, 3))
    r = torch.linspace(1.5, 4.5, 50, dtype=torch.float64)
    G_target = supercell_G_r(sc, r, u_iso=0.005, r_max=5.0)
    sc.positions = sc.positions + 0.05 * torch.randn(sc.positions.shape,
                                                       dtype=torch.float64)
    res = rmc_refine(sc, r, G_target, n_moves=100, u_iso=0.005, seed=0)
    diag = ergodicity_diagnostics(res)
    assert set(diag.keys()) >= {"acceptance_ratio", "n_moves",
                                 "autocorr_time", "effective_sample_size"}
    assert 0.0 <= diag["acceptance_ratio"] <= 1.0
    assert diag["n_moves"] == res.n_moves


def test_ergodicity_diagnostics_handles_short_trace():
    """Very short traces should not raise; they get NaN autocorr."""
    ni = _fcc_ni()
    sc = Supercell.from_crystal(ni, size=(3, 3, 3))
    r = torch.linspace(1.5, 4.5, 50, dtype=torch.float64)
    G_target = supercell_G_r(sc, r, u_iso=0.005, r_max=5.0)
    res = rmc_refine(sc, r, G_target, n_moves=2, u_iso=0.005, seed=0)
    diag = ergodicity_diagnostics(res)
    # Very short chains fine, ESS just gets small
    assert 0.0 <= diag["acceptance_ratio"] <= 1.0


# ---------------------------------------------------------------------------
# CoordinationBias
# ---------------------------------------------------------------------------

def test_coordination_bias_zero_when_matching():
    """FCC Ni's first shell = 12; bias penalty at target=12 should be 0."""
    ni = _fcc_ni()
    sc = Supercell.from_crystal(ni, size=(4, 4, 4))
    bias = CoordinationBias(r_shell=(2.0, 2.7), Z_target=12.0, weight=100.0)
    pen = bias.penalty(sc)
    assert pen == pytest.approx(0.0, abs=1e-6)


def test_coordination_bias_positive_when_offtarget():
    """If we target Z=8 but the FCC first shell has 12, penalty should be
    non-zero and grow with weight."""
    ni = _fcc_ni()
    sc = Supercell.from_crystal(ni, size=(4, 4, 4))
    b1 = CoordinationBias(r_shell=(2.0, 2.7), Z_target=8.0, weight=10.0)
    b2 = CoordinationBias(r_shell=(2.0, 2.7), Z_target=8.0, weight=100.0)
    p1 = b1.penalty(sc)
    p2 = b2.penalty(sc)
    assert p2 > p1 > 0.0
    assert p2 == pytest.approx(10.0 * p1, rel=1e-6)   # linear in weight


def test_coordination_bias_species_filter():
    """Bias with a species filter counts only that species-pair type."""
    positions = torch.randn(20, 3, dtype=torch.float64) * 5.0
    species = ["Ni"] * 10 + ["Fe"] * 10
    sc = Supercell(species=species, positions=positions,
                    cell=torch.eye(3) * 20.0)
    bias_ni = CoordinationBias(
        r_shell=(0.5, 8.0), Z_target=0.0, weight=1.0, species_i="Ni")
    pen = bias_ni.penalty(sc)
    assert pen >= 0.0
