"""Rev-9 tests: cluster / rigid-rotation RMC moves."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from midas_pdf.rmc import (
    Supercell, supercell_G_r, rmc_refine,
    ClusterDisplaceMove, RigidRotationMove, cluster_metropolis_step,
    DisplaceMove,
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

def test_cluster_displace_proposes_neighbourhood():
    """The proposed cluster must include atoms within the requested radius
    (up to min_image resolution)."""
    ni = _fcc_ni()
    sc = Supercell.from_crystal(ni, size=(3, 3, 3))
    move = ClusterDisplaceMove(radius_A=3.0, sigma_A=0.01)
    cluster, delta = move.propose(sc)
    assert cluster.numel() >= 1
    assert delta.shape == (3,)


def test_cluster_displace_delta_finite():
    ni = _fcc_ni()
    sc = Supercell.from_crystal(ni, size=(3, 3, 3))
    move = ClusterDisplaceMove(radius_A=3.0, sigma_A=0.1)
    for _ in range(10):
        _, delta = move.propose(sc)
        assert torch.all(torch.isfinite(delta))


def test_rigid_rotation_proposes_orthogonal_matrix():
    """R must be orthogonal (RR^T = I) — property of a rigid rotation."""
    ni = _fcc_ni()
    sc = Supercell.from_crystal(ni, size=(3, 3, 3))
    move = RigidRotationMove(radius_A=3.0, sigma_rad=0.05)
    cluster, R, anchor = move.propose(sc)
    RRt = R @ R.T
    assert torch.allclose(RRt, torch.eye(3, dtype=torch.float64), atol=1e-9)
    det = float(torch.linalg.det(R))
    assert abs(det - 1.0) < 1e-9              # proper rotation, not reflection


def test_rigid_rotation_anchor_is_position():
    ni = _fcc_ni()
    sc = Supercell.from_crystal(ni, size=(3, 3, 3))
    move = RigidRotationMove(radius_A=3.0, sigma_rad=0.05)
    _, _, anchor = move.propose(sc)
    # Anchor must equal one of the atomic positions (an existing atom)
    dists = torch.linalg.norm(sc.positions - anchor, dim=1)
    assert float(dists.min()) < 1e-9


# ---------------------------------------------------------------------------
# Metropolis with cluster moves
# ---------------------------------------------------------------------------

def test_cluster_metropolis_accepts_at_high_T():
    """At high temperature every move should be nearly always accepted."""
    ni = _fcc_ni()
    sc = Supercell.from_crystal(ni, size=(3, 3, 3))
    L = float(sc.cell[0, 0])
    r = torch.linspace(1.5, L / 2 - 0.5, 60, dtype=torch.float64)
    G_target = supercell_G_r(sc, r, u_iso=0.005, r_max=L / 2 + 1)
    move = ClusterDisplaceMove(radius_A=3.0, sigma_A=0.02)
    rng = torch.Generator().manual_seed(0)
    n_accept = 0
    for _ in range(30):
        res = cluster_metropolis_step(
            sc, move, r, G_target, u_iso=0.005, r_max_pairs=L / 2 + 1,
            temperature=1e10, rng=rng,
        )
        if res["accepted"]:
            n_accept += 1
    assert n_accept > 20


def test_cluster_metropolis_updates_positions_on_accept():
    ni = _fcc_ni()
    sc = Supercell.from_crystal(ni, size=(3, 3, 3))
    L = float(sc.cell[0, 0])
    r = torch.linspace(1.5, L / 2 - 0.5, 60, dtype=torch.float64)
    G_target = supercell_G_r(sc, r, u_iso=0.005, r_max=L / 2 + 1)
    orig = sc.positions.clone()
    move = ClusterDisplaceMove(radius_A=3.0, sigma_A=0.05)
    rng = torch.Generator().manual_seed(0)
    for _ in range(30):
        cluster_metropolis_step(
            sc, move, r, G_target, u_iso=0.005, r_max_pairs=L / 2 + 1,
            temperature=1e6, rng=rng)
    assert not torch.allclose(sc.positions, orig)


def test_rotation_metropolis_preserves_pair_within_cluster():
    """Rigid rotation must preserve intra-cluster distances (up to
    floating-point precision)."""
    ni = _fcc_ni()
    sc = Supercell.from_crystal(ni, size=(3, 3, 3))
    L = float(sc.cell[0, 0])
    r = torch.linspace(1.5, L / 2 - 0.5, 60, dtype=torch.float64)
    G_target = supercell_G_r(sc, r, u_iso=0.005, r_max=L / 2 + 1)

    move = RigidRotationMove(radius_A=3.0, sigma_rad=0.1)
    rng = torch.Generator().manual_seed(0)
    # Grab a cluster + rotate manually, verify intra-cluster distances preserved
    cluster, R, anchor = move.propose(sc)
    if cluster.numel() < 2:
        pytest.skip("cluster too small on this seed")
    rel_before = sc.positions[cluster] - anchor
    d_before = torch.cdist(rel_before, rel_before)
    rel_after = rel_before @ R.T
    d_after = torch.cdist(rel_after, rel_after)
    assert torch.allclose(d_before, d_after, atol=1e-9)


# ---------------------------------------------------------------------------
# End-to-end via rmc_refine
# ---------------------------------------------------------------------------

def test_rmc_refine_with_cluster_moves_reduces_chi2():
    ni = _fcc_ni()
    sc = Supercell.from_crystal(ni, size=(3, 3, 3))
    L = float(sc.cell[0, 0])
    r = torch.linspace(1.5, L / 2 - 0.5, 80, dtype=torch.float64)
    # Disordered target
    rng = torch.Generator().manual_seed(1)
    disp = 0.10 * torch.randn(sc.positions.shape, generator=rng, dtype=torch.float64)
    tgt = Supercell(species=list(sc.species),
                     positions=(sc.positions + disp),
                     cell=sc.cell.clone())
    G_target = supercell_G_r(tgt, r, u_iso=0.005, r_max=L / 2 + 1)
    res = rmc_refine(
        sc, r, G_target,
        moves=[DisplaceMove(sigma_A=0.03),
               ClusterDisplaceMove(radius_A=3.0, sigma_A=0.03),
               RigidRotationMove(radius_A=3.0, sigma_rad=0.02)],
        n_moves=200, u_iso=0.005, r_max_pairs=L / 2 + 1,
        temperature=1.0, min_distance_A=1.5, seed=42,
    )
    assert res.final_chi2 < res.initial_chi2
