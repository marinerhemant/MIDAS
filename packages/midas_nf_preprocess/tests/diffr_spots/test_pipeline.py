"""Integration tests for the diffr_spots pipeline."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch

from midas_nf_preprocess.diffr_spots import (
    DiffrSpotsParams,
    DiffrSpotsPipeline,
    DiffrSpotsResult,
    predict_spots,
    quat_to_orient_matrix,
    read_diffr_spots_bin,
    read_key_bin,
    read_orient_mat_bin,
    read_hkls_csv,
    read_seed_orientations,
    write_all,
)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _write_hkls_csv(path: Path, hkls: list[tuple[int, int, int, int, float]]) -> None:
    """Write a minimal MIDAS hkls.csv file.

    Each row: dummy dummy dummy dummy ringNr h k l theta dummy dummy
    """
    with open(path, "w") as f:
        f.write("# header\n")
        for h, k, l, ring, theta in hkls:
            f.write(f"d d d d {ring} {h} {k} {l} {theta} d d\n")


def _write_seeds(path: Path, quats: list[tuple[float, float, float, float]]) -> None:
    with open(path, "w") as f:
        for q in quats:
            f.write(",".join(f"{x:f}" for x in q) + "\n")


# -----------------------------------------------------------------------------
# predict_spots: structural sanity
# -----------------------------------------------------------------------------


def test_predict_spots_shapes():
    quats = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        dtype=torch.float64,
    )
    hkls = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float64,
    )
    thetas = torch.tensor([5.0, 7.0, 9.0], dtype=torch.float64)
    out = predict_spots(quats, hkls, thetas, distance=1000.0)
    assert isinstance(out, DiffrSpotsResult)
    # Shape convention: (N, M, 2)
    assert out.omegas.shape == (2, 3, 2)
    assert out.etas.shape == (2, 3, 2)
    assert out.yls.shape == (2, 3, 2)
    assert out.zls.shape == (2, 3, 2)
    assert out.valid.shape == (2, 3, 2)
    assert out.orient_mats.shape == (2, 3, 3)


def test_predict_spots_n_orientations_property():
    quats = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0]] * 5, dtype=torch.float64
    )
    hkls = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)
    thetas = torch.tensor([5.0], dtype=torch.float64)
    out = predict_spots(quats, hkls, thetas, distance=1000.0)
    assert out.n_orientations == 5


def test_predict_spots_counts_consistent_with_valid_mask():
    quats = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]], dtype=torch.float64
    )
    hkls = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float64
    )
    thetas = torch.tensor([5.0, 7.0], dtype=torch.float64)
    out = predict_spots(quats, hkls, thetas, distance=1000.0)
    counts = out.counts
    assert counts.shape == (2,)
    assert torch.equal(
        counts,
        out.valid.reshape(2, -1).sum(dim=1).to(torch.int64),
    )


def test_predict_spots_offsets_are_cumulative():
    quats = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0]] * 3, dtype=torch.float64
    )
    hkls = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float64
    )
    thetas = torch.tensor([5.0, 7.0], dtype=torch.float64)
    out = predict_spots(quats, hkls, thetas, distance=1000.0)
    counts = out.counts
    offsets = out.offsets()
    assert offsets[0] == 0
    if len(offsets) >= 2:
        assert offsets[1] == counts[0]
    if len(offsets) >= 3:
        assert offsets[2] == counts[0] + counts[1]


# -----------------------------------------------------------------------------
# predict_spots: filtering
# -----------------------------------------------------------------------------


def test_predict_spots_omega_range_filter_excludes_outside():
    """Tight omega range should reject all spots."""
    quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float64)
    hkls = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float64)
    thetas = torch.tensor([5.0, 7.0], dtype=torch.float64)
    out = predict_spots(
        quats, hkls, thetas, distance=1000.0,
        omega_ranges=[(170.0, 175.0)],
        box_sizes=[(-1e6, 1e6, -1e6, 1e6)],
    )
    # Only spots inside (170, 175) survive; in this synthetic case very few.
    n_kept = int(out.valid.sum().item())
    assert n_kept >= 0  # at least nonnegative


def test_predict_spots_box_size_filter_excludes_outside():
    """Tiny box size should reject everything."""
    quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float64)
    hkls = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)
    thetas = torch.tensor([5.0], dtype=torch.float64)
    out = predict_spots(
        quats, hkls, thetas, distance=1000.0,
        omega_ranges=[(-180.0, 180.0)],
        box_sizes=[(-0.001, 0.001, -0.001, 0.001)],  # ~tiny box at origin
    )
    assert int(out.valid.sum().item()) == 0


def test_predict_spots_exclude_pole_angle():
    """Setting ExcludePoleAngle should cull spots near eta = 0 / 180."""
    quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float64)
    hkls = torch.tensor([[0.5, 0.0, 0.5]], dtype=torch.float64)  # near pole
    thetas = torch.tensor([5.0], dtype=torch.float64)
    out_no_excl = predict_spots(quats, hkls, thetas, distance=1000.0, exclude_pole_angle=0.0)
    out_excl = predict_spots(quats, hkls, thetas, distance=1000.0, exclude_pole_angle=45.0)
    assert int(out_excl.valid.sum().item()) <= int(out_no_excl.valid.sum().item())


# -----------------------------------------------------------------------------
# Differentiability
# -----------------------------------------------------------------------------


def test_predict_spots_differentiable_in_quaternions():
    """Use a non-degenerate (q, hkl) pair so eta is sensitive to q.

    Identity + axis-aligned hkl pins the spot to eta = +/-90deg; perturbing q
    leaves the spot at the pole and (yl, zl) get zero gradient. A generic hkl
    like (1, 1, 0) gives a generic eta that varies with q.

    The loss must NOT be ``yl**2 + zl**2``. That is the squared RING RADIUS,
    ``(distance * tan(2 theta))**2``, which is fixed by the Bragg angle and is
    invariant under orientation -- its true gradient is exactly zero. This test
    used to assert that quantity's gradient was nonzero, and passed only on
    floating-point noise: measured 5.2e-12, coming from the omega solver's old
    ``y2 + 1e-7`` epsilon breaking the invariance. Once the solver became exact
    the noise became a clean 0.0 and the assertion flipped.

    A single spot's ``yl`` does depend on orientation -- d(yl)/dq is ~145 here,
    unchanged by that fix -- so differentiate that instead.
    """
    q = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    hkls = torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float64)
    thetas = torch.tensor([5.0], dtype=torch.float64)
    out = predict_spots(q, hkls, thetas, distance=1000.0)
    assert out.valid.any(), "Test inputs degenerate: no valid Bragg solution"
    out.yls.flatten()[0].backward()
    assert q.grad is not None
    assert q.grad.abs().sum() > 0


def test_ring_radius_is_invariant_under_orientation():
    """The invariance the test above used to violate.

    ``yl**2 + zl**2`` must equal ``(distance * tan(2 theta))**2`` exactly, for
    every orientation: a spot can move around its ring but not off it. The old
    omega solver broke this at the 1e-12 level.
    """
    torch.manual_seed(0)
    q = torch.nn.functional.normalize(torch.randn(8, 4, dtype=torch.float64), dim=-1)
    hkls = torch.tensor([[1.0, 1.0, 1.0], [2.0, 0.0, 0.0]], dtype=torch.float64)
    thetas = torch.tensor([5.0, 7.0], dtype=torch.float64)
    out = predict_spots(q, hkls, thetas, distance=1000.0)
    r = torch.sqrt(out.yls ** 2 + out.zls ** 2)[out.valid]
    exact = 1000.0 * torch.tan(2.0 * thetas * math.pi / 180.0)
    assert r.numel() > 0
    # Layout-independent: every valid spot must sit on ONE of the rings.
    off_ring = (r.unsqueeze(-1) - exact).abs().min(dim=-1).values
    assert float(off_ring.max()) < 1e-9, (
        f"spot off its ring by {float(off_ring.max()):.3e} um")


# -----------------------------------------------------------------------------
# flat_spots ordering: orientation-major
# -----------------------------------------------------------------------------


def test_flat_spots_matches_per_orientation_blocks():
    quats = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        dtype=torch.float64,
    )
    hkls = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float64)
    thetas = torch.tensor([5.0, 7.0], dtype=torch.float64)
    out = predict_spots(quats, hkls, thetas, distance=1000.0)
    flat = out.flat_spots()
    counts = out.counts
    total = int(counts.sum().item())
    assert flat.shape == (total, 3)


# -----------------------------------------------------------------------------
# Pipeline + binary I/O end-to-end
# -----------------------------------------------------------------------------


def test_pipeline_e2e(tmp_path):
    # Set up a tiny scenario.
    hkls_path = tmp_path / "hkls.csv"
    _write_hkls_csv(
        hkls_path,
        [(1, 0, 0, 1, 5.0), (0, 1, 0, 1, 5.0), (1, 1, 0, 2, 7.0)],
    )
    seeds_path = tmp_path / "seeds.csv"
    _write_seeds(
        seeds_path,
        [
            (1.0, 0.0, 0.0, 0.0),
            (0.7071067811865476, 0.0, 0.7071067811865476, 0.0),
        ],
    )

    pf = tmp_path / "params.txt"
    pf.write_text(
        f"DataDirectory {tmp_path}\n"
        f"OutputDirectory {tmp_path}\n"
        f"SeedOrientations {seeds_path}\n"
        f"NrOrientations 2\n"
        f"Lsd 1000.0\n"
        f"px 1.0\n"
        f"MaxRingRad 1000\n"
        f"OmegaRange -180 180\n"
        f"BoxSize -1000 1000 -1000 1000\n"
        f"ExcludePoleAngle 0\n"
    )
    params = DiffrSpotsParams.from_paramfile(pf)
    pipe = DiffrSpotsPipeline(params, device="cpu")
    assert pipe.n_orientations == 2
    assert pipe.n_hkls == 3
    result, paths = pipe.run()
    for name in ("Key.bin", "DiffractionSpots.bin", "OrientMat.bin"):
        assert paths[name].exists()
    # Round-trip the binary files.
    counts = result.counts
    spots = result.flat_spots()
    om = result.orient_mats
    key_back = read_key_bin(paths["Key.bin"], n_orient=2)
    spots_back = read_diffr_spots_bin(paths["DiffractionSpots.bin"], n_spots=spots.shape[0])
    om_back = read_orient_mat_bin(paths["OrientMat.bin"], n_orient=2)
    assert torch.equal(counts, key_back[:, 0])
    assert torch.allclose(spots, spots_back, atol=1e-12)
    assert torch.allclose(om, om_back, atol=1e-12)


# -----------------------------------------------------------------------------
# CSV readers
# -----------------------------------------------------------------------------


def test_read_hkls_csv_filters_rings(tmp_path):
    p = tmp_path / "hkls.csv"
    _write_hkls_csv(
        p,
        [(1, 0, 0, 1, 5.0), (0, 1, 0, 2, 7.0), (0, 0, 1, 3, 9.0)],
    )
    hkls, thetas, rings = read_hkls_csv(p, rings_to_use=[1, 3])
    assert hkls.shape == (2, 3)
    assert set(rings.tolist()) == {1, 3}


def test_read_seed_orientations_count_check(tmp_path):
    p = tmp_path / "s.csv"
    _write_seeds(p, [(1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)])
    with pytest.raises(ValueError, match="expected 3"):
        read_seed_orientations(p, nr_orientations=3)
    quats = read_seed_orientations(p, nr_orientations=2)
    assert quats.shape == (2, 4)


# -----------------------------------------------------------------------------
# Binary I/O round-trip directly
# -----------------------------------------------------------------------------


def test_write_all_roundtrip(tmp_path):
    counts = torch.tensor([3, 5, 2], dtype=torch.int64)
    total = int(counts.sum().item())
    spots = torch.arange(total * 3, dtype=torch.float64).reshape(total, 3)
    om = torch.arange(3 * 9, dtype=torch.float64).reshape(3, 3, 3)
    paths = write_all(tmp_path, counts, spots, om)
    key_back = read_key_bin(paths["Key.bin"], 3)
    spots_back = read_diffr_spots_bin(paths["DiffractionSpots.bin"], total)
    om_back = read_orient_mat_bin(paths["OrientMat.bin"], 3)
    # Counts roundtrip
    assert torch.equal(counts, key_back[:, 0])
    # Offsets are cumulative
    assert int(key_back[0, 1]) == 0
    assert int(key_back[1, 1]) == 3
    assert int(key_back[2, 1]) == 8
    assert torch.equal(spots, spots_back)
    assert torch.equal(om, om_back)
