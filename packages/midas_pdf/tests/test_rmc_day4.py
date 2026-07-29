"""Day-4 RMC tests: parallel-chain ensembles."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from midas_pdf.rmc import (
    Supercell, supercell_G_r, rmc_refine_ensemble,
    ensemble_partial_g_r, ensemble_coordination, ensemble_G_r,
    DisplaceMove, RMCEnsembleResult,
)


def _fcc_ni(a=3.524):
    from midas_hkls import Crystal, Atom, Lattice, SpaceGroup
    return Crystal(
        lattice=Lattice(a, a, a, 90, 90, 90),
        space_group=SpaceGroup.from_number(225),
        atoms=[Atom(element="Ni", fract=(0, 0, 0))], name="Ni",
    ).to_torch()


# ---------------------------------------------------------------------------
# Ensemble driver
# ---------------------------------------------------------------------------

def test_ensemble_returns_expected_n_chains():
    ni = _fcc_ni()
    sc = Supercell.from_crystal(ni, size=(3, 3, 3))
    r = torch.linspace(1.5, 5.0, 60, dtype=torch.float64)
    G_target = supercell_G_r(sc, r, u_iso=0.005, r_max=6.0)
    ens = rmc_refine_ensemble(sc, r, G_target,
                                n_chains=3, n_moves=30, u_iso=0.005,
                                seed=0)
    assert isinstance(ens, RMCEnsembleResult)
    assert ens.n_chains == 3
    assert len(ens.acceptance_ratios) == 3


def test_ensemble_chains_are_independent_states():
    """Each chain should end at a distinct final configuration (positions
    differ across chains under different RNG seeds)."""
    ni = _fcc_ni()
    sc = Supercell.from_crystal(ni, size=(3, 3, 3))
    r = torch.linspace(1.5, 5.0, 60, dtype=torch.float64)
    G_target = supercell_G_r(sc, r, u_iso=0.005, r_max=6.0)
    ens = rmc_refine_ensemble(sc, r, G_target,
                                n_chains=3, n_moves=60, initial_jitter_A=0.05,
                                u_iso=0.005, seed=1, min_distance_A=1.5)
    pos_a = ens.chains[0].supercell.positions
    pos_b = ens.chains[1].supercell.positions
    pos_c = ens.chains[2].supercell.positions
    assert not torch.allclose(pos_a, pos_b)
    assert not torch.allclose(pos_a, pos_c)


def test_ensemble_template_is_not_mutated():
    """The template supercell passed in must be untouched after the run."""
    ni = _fcc_ni()
    sc = Supercell.from_crystal(ni, size=(3, 3, 3))
    r = torch.linspace(1.5, 5.0, 60, dtype=torch.float64)
    G_target = supercell_G_r(sc, r, u_iso=0.005, r_max=6.0)
    orig_positions = sc.positions.clone()
    orig_species = list(sc.species)
    rmc_refine_ensemble(sc, r, G_target,
                          n_chains=2, n_moves=30, initial_jitter_A=0.1,
                          u_iso=0.005, seed=2, min_distance_A=1.5)
    assert torch.allclose(sc.positions, orig_positions)
    assert sc.species == orig_species


# ---------------------------------------------------------------------------
# Ensemble analytics
# ---------------------------------------------------------------------------

def test_ensemble_coordination_reports_mean_and_std():
    ni = _fcc_ni()
    sc = Supercell.from_crystal(ni, size=(3, 3, 3))
    r = torch.linspace(1.5, 5.0, 60, dtype=torch.float64)
    G_target = supercell_G_r(sc, r, u_iso=0.005, r_max=6.0)
    ens = rmc_refine_ensemble(sc, r, G_target,
                                n_chains=3, n_moves=50, initial_jitter_A=0.05,
                                u_iso=0.005, seed=3, min_distance_A=1.5)
    Z = ensemble_coordination(ens, r_shell=(2.0, 2.7))
    assert "Z_mean_ensemble" in Z
    assert "Z_std_ensemble" in Z
    assert Z["n_chains"] == 3
    # For jittered FCC, Z should still be close to 12 (some spread from noise)
    assert 8.0 <= Z["Z_mean_ensemble"] <= 12.5


def test_ensemble_partial_g_r_has_mean_and_std():
    ni = _fcc_ni()
    sc = Supercell.from_crystal(ni, size=(3, 3, 3))
    r = torch.linspace(1.0, 5.0, 60, dtype=torch.float64)
    G_target = supercell_G_r(sc, r, u_iso=0.005, r_max=6.0)
    ens = rmc_refine_ensemble(sc, r, G_target,
                                n_chains=3, n_moves=30, initial_jitter_A=0.05,
                                u_iso=0.005, seed=4)
    g = ensemble_partial_g_r(ens, r)
    for key, entry in g.items():
        assert entry["mean"].shape == r.shape
        assert entry["std"].shape == r.shape
        assert torch.all(entry["std"] >= 0)


def test_ensemble_G_r_returns_mean_std_samples():
    ni = _fcc_ni()
    sc = Supercell.from_crystal(ni, size=(3, 3, 3))
    r = torch.linspace(1.5, 5.0, 60, dtype=torch.float64)
    G_target = supercell_G_r(sc, r, u_iso=0.005, r_max=6.0)
    ens = rmc_refine_ensemble(sc, r, G_target,
                                n_chains=4, n_moves=30, initial_jitter_A=0.05,
                                u_iso=0.005, seed=5)
    G = ensemble_G_r(ens, r, u_iso=0.005, r_max_pairs=6.0)
    assert G["mean"].shape == r.shape
    assert G["std"].shape == r.shape
    assert G["samples"].shape == (4, r.shape[0])
    assert torch.all(G["std"] >= 0)


def test_ensemble_reduces_chi2_uniformly():
    """Every chain must have final_chi2 <= initial_chi2."""
    ni = _fcc_ni()
    sc = Supercell.from_crystal(ni, size=(3, 3, 3))
    r = torch.linspace(1.5, 5.0, 60, dtype=torch.float64)
    G_target = supercell_G_r(sc, r, u_iso=0.005, r_max=6.0)
    ens = rmc_refine_ensemble(sc, r, G_target,
                                n_chains=4, n_moves=100, initial_jitter_A=0.08,
                                u_iso=0.005, seed=6, min_distance_A=1.5)
    initial = ens.initial_chi2()
    final = ens.final_chi2()
    assert np.all(final <= initial + 1e-6), (initial, final)
