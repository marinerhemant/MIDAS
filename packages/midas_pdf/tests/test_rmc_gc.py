"""Rev-7 tests: grand-canonical RMC (InsertMove / RemoveMove / variable-N Metropolis)."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from midas_pdf.rmc import (
    Supercell, supercell_G_r, rmc_refine,
    DisplaceMove, InsertMove, RemoveMove,
    grand_canonical_metropolis_step,
)


def _fcc_ni(a=3.524):
    from midas_hkls import Crystal, Atom, Lattice, SpaceGroup
    return Crystal(
        lattice=Lattice(a, a, a, 90, 90, 90),
        space_group=SpaceGroup.from_number(225),
        atoms=[Atom(element="Ni", fract=(0, 0, 0))], name="Ni",
    ).to_torch()


# ---------------------------------------------------------------------------
# Move proposals
# ---------------------------------------------------------------------------

def test_insert_move_position_inside_box():
    ni = _fcc_ni()
    sc = Supercell.from_crystal(ni, size=(3, 3, 3))
    move = InsertMove(species="Ni")
    for _ in range(20):
        species, pos = move.propose(sc)
        assert species == "Ni"
        cell_inv = torch.linalg.inv(sc.cell)
        frac = pos @ cell_inv
        assert torch.all(frac >= -1e-12)
        assert torch.all(frac < 1 + 1e-12)


def test_remove_move_returns_valid_index():
    ni = _fcc_ni()
    sc = Supercell.from_crystal(ni, size=(3, 3, 3))
    move = RemoveMove()
    for _ in range(20):
        i = move.propose(sc)
        assert 0 <= i < sc.n_atoms


def test_remove_move_species_filter():
    """Only atoms of the requested species should be candidates."""
    sc = Supercell(species=["Ni"] * 4 + ["Fe"] * 4,
                    positions=torch.randn(8, 3, dtype=torch.float64) * 5.0,
                    cell=torch.eye(3) * 20.0)
    move_ni = RemoveMove(species="Ni")
    for _ in range(30):
        i = move_ni.propose(sc)
        assert sc.species[i] == "Ni"


def test_remove_move_missing_species_raises():
    sc = Supercell(species=["Ni", "Ni"],
                    positions=torch.randn(2, 3, dtype=torch.float64) * 5.0,
                    cell=torch.eye(3) * 20.0)
    with pytest.raises(RuntimeError):
        RemoveMove(species="Zn").propose(sc)


# ---------------------------------------------------------------------------
# Grand-canonical Metropolis step
# ---------------------------------------------------------------------------

def test_gc_step_insert_increments_N_when_accepted():
    """With very generous chemical potential, insertions should be accepted
    → N grows by exactly 1 per accepted move."""
    ni = _fcc_ni()
    sc = Supercell.from_crystal(ni, size=(3, 3, 3))
    L = float(sc.cell[0, 0])
    r = torch.linspace(1.5, L / 2 - 0.5, 80, dtype=torch.float64)
    G_target = supercell_G_r(sc, r, u_iso=0.005, r_max=L / 2 + 1)
    N_before = sc.n_atoms
    move = InsertMove(species="Ni")
    rng = torch.Generator().manual_seed(0)
    res = grand_canonical_metropolis_step(
        sc, move, r, G_target, u_iso=0.005, r_max_pairs=L / 2 + 1,
        temperature=1e6, chemical_potential=1e6,        # always accept
        min_distance_A=None, rng=rng,
    )
    if res["accepted"]:
        assert res["delta_N"] == +1
        assert sc.n_atoms == N_before + 1
    else:
        assert sc.n_atoms == N_before


def test_gc_step_remove_decrements_N_when_accepted():
    ni = _fcc_ni()
    sc = Supercell.from_crystal(ni, size=(3, 3, 3))
    L = float(sc.cell[0, 0])
    r = torch.linspace(1.5, L / 2 - 0.5, 80, dtype=torch.float64)
    G_target = supercell_G_r(sc, r, u_iso=0.005, r_max=L / 2 + 1)
    N_before = sc.n_atoms
    move = RemoveMove()
    rng = torch.Generator().manual_seed(1)
    res = grand_canonical_metropolis_step(
        sc, move, r, G_target, u_iso=0.005, r_max_pairs=L / 2 + 1,
        temperature=1e6, chemical_potential=-1e6,      # always accept remove
        min_distance_A=None, rng=rng,
    )
    if res["accepted"]:
        assert res["delta_N"] == -1
        assert sc.n_atoms == N_before - 1


def test_gc_step_insert_veto_on_hard_sphere():
    """With an insanely large min_distance_A, no insertion should succeed."""
    ni = _fcc_ni()
    sc = Supercell.from_crystal(ni, size=(3, 3, 3))
    L = float(sc.cell[0, 0])
    r = torch.linspace(1.5, L / 2 - 0.5, 60, dtype=torch.float64)
    G_target = supercell_G_r(sc, r, u_iso=0.005, r_max=L / 2 + 1)
    N_before = sc.n_atoms
    move = InsertMove(species="Ni")
    rng = torch.Generator().manual_seed(2)
    n_accept = 0
    for _ in range(30):
        res = grand_canonical_metropolis_step(
            sc, move, r, G_target, u_iso=0.005, r_max_pairs=L / 2 + 1,
            temperature=1e6, chemical_potential=1e6,
            min_distance_A=100.0, rng=rng,             # too large to satisfy
        )
        if res["accepted"]:
            n_accept += 1
    assert n_accept == 0
    assert sc.n_atoms == N_before


# ---------------------------------------------------------------------------
# End-to-end vacancy recovery: the Rev-6 failure that Rev-7 fixes
# ---------------------------------------------------------------------------

def test_gc_rmc_recovers_target_atom_count():
    """Start with a perfect FCC supercell; target G(r) is from a copy with
    5% vacancies. Grand-canonical RMC (μ = 0) must drive N toward the
    defective target."""
    ni = _fcc_ni()
    perfect = Supercell.from_crystal(ni, size=(4, 4, 4))
    L = float(perfect.cell[0, 0])
    r_grid = torch.linspace(1.5, L / 2 - 0.5, 150, dtype=torch.float64)

    # Defective reference: remove exactly 13 atoms (~5% of 256)
    rng_np = np.random.default_rng(0)
    keep = np.ones(perfect.n_atoms, dtype=bool)
    keep[rng_np.choice(perfect.n_atoms, 13, replace=False)] = False
    defective = Supercell(
        species=[s for s, k in zip(perfect.species, keep) if k],
        positions=perfect.positions[torch.tensor(keep)],
        cell=perfect.cell.clone(),
    )
    N_target = defective.n_atoms
    G_target = supercell_G_r(defective, r_grid, u_iso=0.005, r_max=L / 2 + 1)

    # GC-RMC from the perfect crystal, μ = 0
    sc = Supercell(
        species=list(perfect.species),
        positions=perfect.positions.clone(),
        cell=perfect.cell.clone(),
    )
    res = rmc_refine(
        sc, r_grid, G_target,
        moves=[DisplaceMove(sigma_A=0.02),
               RemoveMove(species="X"),
               InsertMove(species="X")],
        n_moves=3000, u_iso=0.005, r_max_pairs=L / 2 + 1,
        temperature=0.5, chemical_potential=0.0,
        min_distance_A=1.5, seed=42,
    )
    # Should have significantly reduced N (moved toward the vacancy target)
    assert sc.n_atoms < perfect.n_atoms
    # Should be within a few atoms of the target (small drift is OK for stochastic)
    assert abs(sc.n_atoms - N_target) <= 5, (sc.n_atoms, N_target)
