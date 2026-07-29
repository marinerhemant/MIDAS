"""Day-2 RMC tests: moves, Metropolis, and the driver.

Invariants:

  1. A DisplaceMove leaves the supercell inside its box (PBC-wrapped).
  2. chi2_supercell of an unperturbed FCC supercell against its own G(r)
     is (nearly) zero.
  3. Metropolis with T=0 accepts only strict improvements (Δχ² < 0).
  4. The rmc_refine driver reduces χ² starting from a jittered
     configuration.
  5. Bad moves (below min_distance_A) are rejected before the χ² eval.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from midas_pdf.rmc import (
    Supercell, DisplaceMove, SwapMove, chi2_supercell,
    metropolis_step, rmc_refine, supercell_G_r,
)


def _fcc_ni(a=3.524):
    from midas_hkls import Crystal, Atom, Lattice, SpaceGroup
    return Crystal(
        lattice=Lattice(a, a, a, 90, 90, 90),
        space_group=SpaceGroup.from_number(225),
        atoms=[Atom(element="Ni", fract=(0, 0, 0))], name="Ni",
    ).to_torch()


# ---------------------------------------------------------------------------
# DisplaceMove basics
# ---------------------------------------------------------------------------

def test_displace_move_returns_expected_shapes():
    ni = _fcc_ni()
    sc = Supercell.from_crystal(ni, size=(3, 3, 3))
    move = DisplaceMove(sigma_A=0.05)
    idx, old, new = move.propose(sc)
    assert 0 <= idx < sc.n_atoms
    assert old.shape == (3,)
    assert new.shape == (3,)


def test_displace_move_new_position_is_pbc_wrapped():
    """The proposed new position must lie inside the supercell after wrap."""
    ni = _fcc_ni()
    sc = Supercell.from_crystal(ni, size=(3, 3, 3))
    move = DisplaceMove(sigma_A=10.0)                # huge kicks → forces wrap
    for _ in range(20):
        _, _, new = move.propose(sc)
        cell_inv = torch.linalg.inv(sc.cell)
        frac = new @ cell_inv
        assert torch.all(frac >= -1e-12), frac
        assert torch.all(frac < 1 + 1e-12), frac


# ---------------------------------------------------------------------------
# SwapMove basics
# ---------------------------------------------------------------------------

def test_swap_move_requires_two_species():
    ni = _fcc_ni()
    sc = Supercell.from_crystal(ni, size=(2, 2, 2))
    move = SwapMove()
    with pytest.raises(RuntimeError):
        move.propose(sc)


def test_swap_move_picks_distinct_species():
    """Build a synthetic 2-species supercell and confirm SwapMove finds a
    valid distinct-species pair."""
    positions = torch.randn(20, 3, dtype=torch.float64)
    species = ["Ni"] * 10 + ["Fe"] * 10
    sc = Supercell(species=species, positions=positions, cell=torch.eye(3) * 10.0)
    move = SwapMove()
    for _ in range(10):
        i, j, sp_i, sp_j = move.propose(sc)
        assert i != j
        assert sp_i != sp_j


# ---------------------------------------------------------------------------
# chi2_supercell
# ---------------------------------------------------------------------------

def test_chi2_of_unperturbed_supercell_is_small():
    """A crystalline FCC supercell fitted against its own G(r) must give
    near-zero χ²."""
    ni = _fcc_ni()
    sc = Supercell.from_crystal(ni, size=(4, 4, 4))
    L = float(sc.cell[0, 0])
    r = torch.linspace(1.5, L / 2 - 0.5, 200, dtype=torch.float64)
    G_target = supercell_G_r(sc, r, u_iso=0.005, r_max=L / 2 + 1)
    chi2, G_calc = chi2_supercell(sc, r, G_target, u_iso=0.005,
                                     r_max_pairs=L / 2 + 1)
    assert chi2 < 1e-6
    assert torch.allclose(G_calc, G_target, atol=1e-6)


# ---------------------------------------------------------------------------
# Metropolis single step
# ---------------------------------------------------------------------------

def test_metropolis_at_zero_temperature_rejects_worsening_move():
    """T → 0 in the Metropolis kernel should never accept Δχ² > 0."""
    ni = _fcc_ni()
    sc = Supercell.from_crystal(ni, size=(3, 3, 3))
    L = float(sc.cell[0, 0])
    r = torch.linspace(1.5, L / 2 - 0.5, 150, dtype=torch.float64)
    G_target = supercell_G_r(sc, r, u_iso=0.005, r_max=L / 2 + 1)
    # Start from a jittered configuration → any move will typically worsen χ²
    jittered = sc.positions + 0.05 * torch.randn(sc.positions.shape,
                                                    dtype=torch.float64)
    cell_inv = torch.linalg.inv(sc.cell)
    frac = jittered @ cell_inv
    frac = frac - torch.floor(frac)
    sc.positions = frac @ sc.cell

    move = DisplaceMove(sigma_A=0.2)          # large kicks → likely worsen
    rng = torch.Generator().manual_seed(0)
    n_accept = 0
    for _ in range(20):
        res = metropolis_step(
            sc, move, r, G_target, u_iso=0.005,
            r_max_pairs=L / 2 + 1,
            temperature=1e-8, rng=rng,           # effectively T=0
        )
        # Should accept only if Δχ² < 0 (strict improvement)
        if res["accepted"]:
            n_accept += 1
            assert res["delta_chi2"] < 0
    # It's OK if none of the moves improve; we just require no false accepts


def test_metropolis_updates_supercell_positions_on_accept():
    ni = _fcc_ni()
    sc = Supercell.from_crystal(ni, size=(3, 3, 3))
    L = float(sc.cell[0, 0])
    r = torch.linspace(1.5, L / 2 - 0.5, 100, dtype=torch.float64)
    G_target = supercell_G_r(sc, r, u_iso=0.005, r_max=L / 2 + 1)
    orig_positions = sc.positions.clone()
    move = DisplaceMove(sigma_A=0.1)
    # High T → nearly always accepts
    rng = torch.Generator().manual_seed(0)
    for _ in range(20):
        metropolis_step(sc, move, r, G_target, u_iso=0.005,
                          r_max_pairs=L / 2 + 1, temperature=1e6, rng=rng)
    # Positions should have changed
    assert not torch.allclose(sc.positions, orig_positions)


# ---------------------------------------------------------------------------
# Hard-sphere veto
# ---------------------------------------------------------------------------

def test_min_distance_veto_rejects_close_moves():
    """Setting an artificially huge min_distance (larger than any inter-atom
    distance) should reject every move."""
    ni = _fcc_ni()
    sc = Supercell.from_crystal(ni, size=(3, 3, 3))
    L = float(sc.cell[0, 0])
    r = torch.linspace(1.5, L / 2 - 0.5, 100, dtype=torch.float64)
    G_target = supercell_G_r(sc, r, u_iso=0.005, r_max=L / 2 + 1)
    rng = torch.Generator().manual_seed(0)
    n_accept = 0
    move = DisplaceMove(sigma_A=0.1)
    for _ in range(30):
        res = metropolis_step(
            sc, move, r, G_target, u_iso=0.005,
            r_max_pairs=L / 2 + 1,
            temperature=1e10,               # always accept if allowed
            min_distance_A=100.0,             # impossibly large
            rng=rng,
        )
        if res["accepted"]:
            n_accept += 1
    assert n_accept == 0


# ---------------------------------------------------------------------------
# Driver: rmc_refine reduces χ²
# ---------------------------------------------------------------------------

def test_rmc_refine_reduces_chi2():
    """Starting from a jittered FCC supercell, rmc_refine must reduce χ²
    below its initial value in a short chain."""
    ni = _fcc_ni()
    sc = Supercell.from_crystal(ni, size=(3, 3, 3))
    L = float(sc.cell[0, 0])
    r = torch.linspace(1.5, L / 2 - 0.5, 150, dtype=torch.float64)
    # Target = crystal G(r); start from small jitter
    G_target = supercell_G_r(sc, r, u_iso=0.005, r_max=L / 2 + 1)
    jitter = 0.1 * torch.randn(sc.positions.shape, dtype=torch.float64,
                                generator=torch.Generator().manual_seed(1))
    sc.positions = sc.positions + jitter
    cell_inv = torch.linalg.inv(sc.cell)
    frac = sc.positions @ cell_inv
    sc.positions = (frac - torch.floor(frac)) @ sc.cell

    res = rmc_refine(
        sc, r, G_target,
        moves=[DisplaceMove(sigma_A=0.02)],
        n_moves=200, u_iso=0.005,
        r_max_pairs=L / 2 + 1,
        temperature=1.0, min_distance_A=1.5, seed=42,
    )
    assert res.n_moves > 0
    assert res.final_chi2 < res.initial_chi2
    assert 0.0 <= res.acceptance_ratio <= 1.0
    assert len(res.chi2_trace) == res.n_moves + 1


def test_rmc_refine_result_shape_and_bookkeeping():
    ni = _fcc_ni()
    sc = Supercell.from_crystal(ni, size=(3, 3, 3))
    r = torch.linspace(1.5, 5.0, 100, dtype=torch.float64)
    G_target = supercell_G_r(sc, r, u_iso=0.005, r_max=6.0)
    res = rmc_refine(
        sc, r, G_target,
        n_moves=50, u_iso=0.005, seed=0,
    )
    assert res.n_moves == 50
    assert res.n_accepted <= res.n_moves
    assert len(res.chi2_trace) == 51                # initial + n_moves
    assert len(res.accept_trace) == res.n_moves
