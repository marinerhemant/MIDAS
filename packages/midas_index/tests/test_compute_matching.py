"""Tests for compute.matching."""

import numpy as np
import pytest
import torch

from midas_index.compute.matching import (
    build_eta_margins,
    build_ome_margins,
    compare_spots,
)


def _make_obs(rows):
    """rows: list of (y, z, omega, ring_radius, spot_id, ring_nr, eta, ttheta, rad_diff)"""
    return torch.tensor(rows, dtype=torch.float64)


def _make_theor(rows):
    """rows: list of 14-element TheorSpots."""
    return torch.tensor(rows, dtype=torch.float64).unsqueeze(0)  # (1, T, 14)


def test_compare_spots_perfect_match():
    # 1 evaluation tuple with 1 theoretical spot, 1 matching observed spot.
    obs = _make_obs([
        # y, z, omega, ring_rad, spot_id, ring_nr, eta, ttheta, rad_diff
        (10.0, 5.0, 1.0, 30000.0, 17, 1, 12.0, 1.5, 0.0),
    ])
    # ndata: 1 bin, count=1, offset=0
    # data: [0]
    n_eta_bins, n_ome_bins = 3600, 3600
    eta_bin_size, ome_bin_size = 0.1, 0.1

    # Compute the bin index for the observed spot — theor must hash to same bin.
    obs_ring_nr = 1
    obs_eta = 12.0
    obs_ome = 1.0
    pos = (
        (obs_ring_nr - 1) * n_eta_bins * n_ome_bins
        + int(np.floor((180 + obs_eta) / eta_bin_size)) * n_ome_bins
        + int(np.floor((180 + obs_ome) / ome_bin_size))
    )
    ndata = torch.zeros(2 * (pos + 1), dtype=torch.int32)
    ndata[2 * pos] = 1
    ndata[2 * pos + 1] = 0
    bin_data = torch.tensor([0], dtype=torch.int32)

    theor = _make_theor([
        # 0 1 2 3 4 5 6:omega 7 8 9:ringnr 10:y 11:z 12:eta 13:rad_diff
        [0, 0, 0, 0, 0, 0, 1.0, 0, 0, 1, 10.0, 5.0, 12.0, 0.0],
    ])
    valid = torch.ones((1, 1), dtype=torch.bool)
    eta_margins = build_eta_margins(
        ring_radii={1: 30000.0}, margin_eta=2.0, stepsize_orient_deg=0.5,
        device=torch.device("cpu"), dtype=torch.float64,
    )
    ome_margins = build_ome_margins(
        margin_ome=0.5, stepsize_orient_deg=0.5,
        device=torch.device("cpu"), dtype=torch.float64,
    )

    res = compare_spots(
        theor=theor, valid=valid, obs=obs,
        bin_data=bin_data, bin_ndata=ndata,
        ref_rad=torch.tensor([30000.0], dtype=torch.float64),
        margin_rad=10.0, margin_radial=10.0,
        eta_margins=eta_margins, ome_margins=ome_margins,
        eta_bin_size=eta_bin_size, ome_bin_size=ome_bin_size,
        n_eta_bins=n_eta_bins, n_ome_bins=n_ome_bins,
        rings_to_reject=torch.tensor([], dtype=torch.int64),
    )
    assert int(res.n_matches.item()) == 1
    assert int(res.matched_obs_id[0, 0].item()) == 17
    assert res.frac_matches[0].item() == 1.0


def test_compare_spots_no_match_when_far():
    # Theoretical spot far from observed.
    obs = _make_obs([
        (10.0, 5.0, 1.0, 30000.0, 17, 1, 12.0, 1.5, 0.0),
    ])
    # Theor at eta=180 (different bin)
    theor = _make_theor([
        [0, 0, 0, 0, 0, 0, 1.0, 0, 0, 1, 10.0, 5.0, 180.0, 0.0],
    ])
    valid = torch.ones((1, 1), dtype=torch.bool)
    n_eta_bins, n_ome_bins = 3600, 3600
    eta_bin_size, ome_bin_size = 0.1, 0.1
    eta_margins = build_eta_margins(
        ring_radii={1: 30000.0}, margin_eta=2.0, stepsize_orient_deg=0.5,
        device=torch.device("cpu"), dtype=torch.float64,
    )
    ome_margins = build_ome_margins(
        margin_ome=0.5, stepsize_orient_deg=0.5,
        device=torch.device("cpu"), dtype=torch.float64,
    )
    # ndata: empty bins
    ndata = torch.zeros(2 * (n_eta_bins * n_ome_bins), dtype=torch.int32)
    bin_data = torch.zeros(0, dtype=torch.int32)

    res = compare_spots(
        theor=theor, valid=valid, obs=obs,
        bin_data=bin_data, bin_ndata=ndata,
        ref_rad=torch.tensor([30000.0], dtype=torch.float64),
        margin_rad=10.0, margin_radial=10.0,
        eta_margins=eta_margins, ome_margins=ome_margins,
        eta_bin_size=eta_bin_size, ome_bin_size=ome_bin_size,
        n_eta_bins=n_eta_bins, n_ome_bins=n_ome_bins,
        rings_to_reject=torch.tensor([], dtype=torch.int64),
    )
    assert int(res.n_matches.item()) == 0


def test_compare_spots_tie_break_picks_smallest_delta_omega():
    # Two observed spots in the same bin; theor must match the one with
    # smaller |delta omega|.
    obs = _make_obs([
        (10.0, 5.0, 0.6, 30000.0, 100, 1, 12.0, 1.5, 0.0),  # Δω=0.4
        (10.0, 5.0, 1.1, 30000.0, 200, 1, 12.0, 1.5, 0.0),  # Δω=0.1  ← winner
        (10.0, 5.0, 1.5, 30000.0, 300, 1, 12.0, 1.5, 0.0),  # Δω=0.5
    ])
    theor = _make_theor([
        [0, 0, 0, 0, 0, 0, 1.0, 0, 0, 1, 10.0, 5.0, 12.0, 0.0],
    ])
    valid = torch.ones((1, 1), dtype=torch.bool)

    n_eta_bins, n_ome_bins = 3600, 3600
    eta_bin_size, ome_bin_size = 0.1, 0.1
    pos = (
        0 * n_eta_bins * n_ome_bins
        + int(np.floor((180 + 12.0) / eta_bin_size)) * n_ome_bins
        + int(np.floor((180 + 1.0) / ome_bin_size))
    )
    # Make both eta bins resolve to same bin: ome bin width is 0.1, and the
    # observed omegas (0.6, 1.1, 1.5) fall in different ome-bins. Force them
    # into the same bin by using a wider OmeBinSize for this test.
    ome_bin_size = 5.0
    n_ome_bins = 72
    pos = (
        0 * n_eta_bins * n_ome_bins
        + int(np.floor((180 + 12.0) / eta_bin_size)) * n_ome_bins
        + int(np.floor((180 + 1.0) / ome_bin_size))
    )
    ndata = torch.zeros(2 * (pos + 10), dtype=torch.int32)
    ndata[2 * pos] = 3
    ndata[2 * pos + 1] = 0
    bin_data = torch.tensor([0, 1, 2], dtype=torch.int32)

    eta_margins = build_eta_margins(
        ring_radii={1: 30000.0}, margin_eta=10.0, stepsize_orient_deg=0.5,
        device=torch.device("cpu"), dtype=torch.float64,
    )
    ome_margins = build_ome_margins(
        margin_ome=2.0, stepsize_orient_deg=0.5,
        device=torch.device("cpu"), dtype=torch.float64,
    )

    res = compare_spots(
        theor=theor, valid=valid, obs=obs,
        bin_data=bin_data, bin_ndata=ndata,
        ref_rad=torch.tensor([30000.0], dtype=torch.float64),
        margin_rad=10.0, margin_radial=10.0,
        eta_margins=eta_margins, ome_margins=ome_margins,
        eta_bin_size=eta_bin_size, ome_bin_size=ome_bin_size,
        n_eta_bins=n_eta_bins, n_ome_bins=n_ome_bins,
        rings_to_reject=torch.tensor([], dtype=torch.int64),
    )
    assert int(res.matched_obs_id[0, 0].item()) == 200


def test_build_eta_margins_shapes_and_zeros_outside_rings():
    eta = build_eta_margins(
        ring_radii={1: 30000.0, 3: 50000.0},
        margin_eta=5.0, stepsize_orient_deg=1.0,
        device=torch.device("cpu"), dtype=torch.float64,
    )
    assert eta.shape == (500,)
    assert eta[2].item() == 0.0       # ring 2 has no radius
    assert eta[1].item() > 0.0
    assert eta[3].item() > 0.0


def test_build_ome_margins_lut():
    ome = build_ome_margins(
        margin_ome=0.5, stepsize_orient_deg=0.5,
        device=torch.device("cpu"), dtype=torch.float64,
    )
    assert ome.shape == (181,)
    # endpoints repeat the i==1 value
    assert ome[0].item() == ome[1].item()
    assert ome[180].item() == ome[1].item()


def test_compare_spots_avg_ia_zero_when_perfect_match():
    """A theor and obs spot at exactly the same lab-frame location must yield IA=0."""
    # 1 evaluation tuple, 1 theor spot, 1 obs spot, identical y/z/omega → IA=0.
    obs = _make_obs([
        (10.0, 5.0, 1.0, 30000.0, 17, 1, 12.0, 1.5, 0.0),
    ])
    theor = _make_theor([
        # Theor cols 10/11 = 10/5 (same as obs). col 6 = 1.0 (same omega).
        [0, 0, 0, 0, 10.0, 5.0, 1.0, 0, 0, 1, 10.0, 5.0, 12.0, 0.0],
    ])
    valid = torch.ones((1, 1), dtype=torch.bool)
    n_eta_bins, n_ome_bins = 3600, 3600
    eta_bin_size, ome_bin_size = 0.1, 0.1
    pos = (
        0 * n_eta_bins * n_ome_bins
        + int(np.floor((180 + 12.0) / eta_bin_size)) * n_ome_bins
        + int(np.floor((180 + 1.0) / ome_bin_size))
    )
    ndata = torch.zeros(2 * (pos + 1), dtype=torch.int32)
    ndata[2 * pos] = 1
    ndata[2 * pos + 1] = 0
    bin_data = torch.tensor([0], dtype=torch.int32)

    eta_margins = build_eta_margins(
        ring_radii={1: 30000.0}, margin_eta=2.0, stepsize_orient_deg=0.5,
        device=torch.device("cpu"), dtype=torch.float64,
    )
    ome_margins = build_ome_margins(
        margin_ome=0.5, stepsize_orient_deg=0.5,
        device=torch.device("cpu"), dtype=torch.float64,
    )

    res = compare_spots(
        theor=theor, valid=valid, obs=obs,
        bin_data=bin_data, bin_ndata=ndata,
        ref_rad=torch.tensor([30000.0], dtype=torch.float64),
        margin_rad=10.0, margin_radial=10.0,
        eta_margins=eta_margins, ome_margins=ome_margins,
        eta_bin_size=eta_bin_size, ome_bin_size=ome_bin_size,
        n_eta_bins=n_eta_bins, n_ome_bins=n_ome_bins,
        rings_to_reject=torch.tensor([], dtype=torch.int64),
        distance=1_000_000.0,
        pos=torch.zeros((1, 3), dtype=torch.float64),
    )
    assert res.avg_ia[0].item() == pytest.approx(0.0, abs=1e-6)


def test_compare_spots_jagged_matches_dense_byte_for_byte():
    """The jagged strategy chunks N but must produce identical numerics to dense."""
    rng = np.random.default_rng(0)
    # Build a non-trivial obs table on ring 1
    n_obs = 12
    obs = np.zeros((n_obs, 9), dtype=np.float64)
    obs[:, 0] = rng.uniform(-50, 50, n_obs)               # y
    obs[:, 1] = rng.uniform(-50, 50, n_obs)               # z
    obs[:, 2] = rng.uniform(-30, 30, n_obs)               # omega
    obs[:, 3] = 30000.0                                    # ring radius
    obs[:, 4] = np.arange(1, n_obs + 1)                   # spot id
    obs[:, 5] = 1                                          # ring nr
    obs[:, 6] = rng.uniform(-30, 30, n_obs)               # eta
    obs[:, 8] = 0.0                                        # rad_diff

    # Build a bin index (with margin spread) so the matching has work to do.
    from midas_index.io import build_bin_index
    bin_data_np, ndata_np = build_bin_index(
        obs, eta_bin_size=1.0, ome_bin_size=1.0, n_rings=1,
        margin_eta=10.0, margin_ome=2.0, stepsize_orient=0.5,
        ring_radii={1: 30000.0},
    )
    bin_data = torch.as_tensor(bin_data_np, dtype=torch.int32)
    bin_ndata = torch.as_tensor(ndata_np, dtype=torch.int32)
    obs_t = torch.as_tensor(obs, dtype=torch.float64)

    # Build N=20 "theor" tuples that should each find a few matches.
    N, T = 20, 4
    theor = torch.zeros((N, T, 14), dtype=torch.float64)
    for n in range(N):
        for t in range(T):
            theor[n, t, 6] = obs[t % n_obs, 2] + 0.05 * (n - 10)   # omega near obs
            theor[n, t, 9] = 1                                       # ring 1
            theor[n, t, 10] = obs[t % n_obs, 0]                      # yl_disp
            theor[n, t, 11] = obs[t % n_obs, 1]                      # zl_disp
            theor[n, t, 12] = obs[t % n_obs, 6] + 0.05 * (n - 10)   # eta_post
            theor[n, t, 13] = 0.0                                    # rad_diff
    valid = torch.ones((N, T), dtype=torch.bool)

    eta_margins = build_eta_margins(
        ring_radii={1: 30000.0}, margin_eta=10.0, stepsize_orient_deg=0.5,
        device=torch.device("cpu"), dtype=torch.float64,
    )
    ome_margins = build_ome_margins(
        margin_ome=2.0, stepsize_orient_deg=0.5,
        device=torch.device("cpu"), dtype=torch.float64,
    )

    common_kwargs = dict(
        theor=theor, valid=valid, obs=obs_t,
        bin_data=bin_data, bin_ndata=bin_ndata,
        ref_rad=torch.full((N,), 30000.0, dtype=torch.float64),
        margin_rad=10.0, margin_radial=10.0,
        eta_margins=eta_margins, ome_margins=ome_margins,
        eta_bin_size=1.0, ome_bin_size=1.0,
        n_eta_bins=360, n_ome_bins=360,
        rings_to_reject=torch.tensor([], dtype=torch.int64),
        distance=1_000_000.0,
        pos=torch.zeros((N, 3), dtype=torch.float64),
    )

    dense = compare_spots(**common_kwargs, strategy="dense")
    jagged = compare_spots(**common_kwargs, strategy="jagged", chunk_size=5)

    # Byte-for-byte parity across all output fields
    np.testing.assert_array_equal(dense.n_matches.numpy(), jagged.n_matches.numpy())
    np.testing.assert_array_equal(
        dense.matched_obs_id.numpy(), jagged.matched_obs_id.numpy(),
    )
    np.testing.assert_array_equal(
        dense.matched_obs_row.numpy(), jagged.matched_obs_row.numpy(),
    )
    np.testing.assert_array_equal(dense.matched.numpy(), jagged.matched.numpy())
    np.testing.assert_allclose(
        dense.frac_matches.numpy(), jagged.frac_matches.numpy(), atol=1e-12,
    )
    np.testing.assert_allclose(
        dense.avg_ia.numpy(), jagged.avg_ia.numpy(), atol=1e-12,
    )


def test_compare_spots_strategy_invalid_raises():
    with pytest.raises(ValueError, match="strategy must be"):
        compare_spots(
            theor=torch.zeros((1, 1, 14), dtype=torch.float64),
            valid=torch.zeros((1, 1), dtype=torch.bool),
            obs=torch.zeros((1, 9), dtype=torch.float64),
            bin_data=torch.zeros(0, dtype=torch.int32),
            bin_ndata=torch.zeros(2, dtype=torch.int32),
            ref_rad=torch.zeros(1, dtype=torch.float64),
            margin_rad=1.0, margin_radial=1.0,
            eta_margins=torch.zeros(500, dtype=torch.float64),
            ome_margins=torch.zeros(181, dtype=torch.float64),
            eta_bin_size=1.0, ome_bin_size=1.0,
            n_eta_bins=360, n_ome_bins=360,
            rings_to_reject=torch.tensor([], dtype=torch.int64),
            strategy="bogus",
        )


def test_compare_spots_avg_ia_zero_when_no_matches():
    """No matches -> avg_ia is 0 (no spots contribute)."""
    obs = _make_obs([
        (0.0, 0.0, 0.0, 0.0, 17, 1, 0.0, 0.0, 0.0),
    ])
    theor = _make_theor([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 99.0, 0],   # eta=99 → won't match
    ])
    valid = torch.ones((1, 1), dtype=torch.bool)
    eta_margins = build_eta_margins(
        ring_radii={1: 30000.0}, margin_eta=2.0, stepsize_orient_deg=0.5,
        device=torch.device("cpu"), dtype=torch.float64,
    )
    ome_margins = build_ome_margins(
        margin_ome=0.5, stepsize_orient_deg=0.5,
        device=torch.device("cpu"), dtype=torch.float64,
    )
    n_eta_bins, n_ome_bins = 3600, 3600
    ndata = torch.zeros(2 * n_eta_bins * n_ome_bins, dtype=torch.int32)
    bin_data = torch.zeros(0, dtype=torch.int32)
    res = compare_spots(
        theor=theor, valid=valid, obs=obs,
        bin_data=bin_data, bin_ndata=ndata,
        ref_rad=torch.tensor([30000.0], dtype=torch.float64),
        margin_rad=10.0, margin_radial=10.0,
        eta_margins=eta_margins, ome_margins=ome_margins,
        eta_bin_size=0.1, ome_bin_size=0.1,
        n_eta_bins=n_eta_bins, n_ome_bins=n_ome_bins,
        rings_to_reject=torch.tensor([], dtype=torch.int64),
        distance=1_000_000.0,
        pos=torch.zeros((1, 3), dtype=torch.float64),
    )
    assert int(res.n_matches.item()) == 0
    assert res.avg_ia[0].item() == 0.0


# ---------------------------------------------------------------------------
# Auto-strategy picker — predicts dense-path peak vs free memory.
# ---------------------------------------------------------------------------

def test_pick_compare_strategy_small_stays_dense():
    """A modest (N=100, T=20, M=10) shape fits in 1 GB → stay dense."""
    from midas_index.compute.matching import pick_compare_strategy
    theor = torch.zeros((100, 20, 14), dtype=torch.float64)
    strategy, chunk_size = pick_compare_strategy(
        theor, max_n_cap=10, free_bytes=1024**3,  # 1 GB
    )
    assert strategy == "dense"


def test_pick_compare_strategy_dense_overflow_chunks():
    """A huge (N, T, M) explicitly oversizes the budget → jagged with a
    chunk_size that bounds chunk_peak inside the budget."""
    from midas_index.compute.matching import (
        pick_compare_strategy, _per_cell_bytes, _JAGGED_CHUNK_MAX,
    )
    N, T, M = 200_000, 200, 50
    theor = torch.zeros((N, T, 14), dtype=torch.float64)
    free = 8 * 1024**3   # 8 GB
    strategy, chunk_size = pick_compare_strategy(
        theor, max_n_cap=M, free_bytes=free, safety=0.5,
    )
    assert strategy == "jagged"
    budget = int(free * 0.5)
    per_n_bytes = T * M * _per_cell_bytes(torch.float64)
    # chunk_size × per_n_bytes must fit the budget.
    assert chunk_size * per_n_bytes <= budget
    # And must not exceed the hard cap or actual N.
    assert chunk_size <= _JAGGED_CHUNK_MAX
    assert chunk_size <= N
    assert chunk_size >= 64  # the floor


def test_pick_compare_strategy_no_max_n_cap_defaults_dense():
    """Without max_n_cap (or 0) the picker can't predict — stay dense."""
    from midas_index.compute.matching import pick_compare_strategy
    theor = torch.zeros((10, 10, 14), dtype=torch.float64)
    assert pick_compare_strategy(theor, max_n_cap=None, free_bytes=1)[0] == "dense"
    assert pick_compare_strategy(theor, max_n_cap=0, free_bytes=1)[0] == "dense"


def test_pick_compare_strategy_cpu_device_defaults_dense():
    """No mem_get_info on CPU → return dense (caller had to be OK with that
    before the picker existed)."""
    from midas_index.compute.matching import pick_compare_strategy
    theor = torch.zeros((10, 10, 14), dtype=torch.float64,
                        device=torch.device("cpu"))
    # free_bytes=None and CPU device → dense path.
    assert pick_compare_strategy(theor, max_n_cap=10)[0] == "dense"


def test_pick_compare_strategy_fp32_uses_smaller_per_cell():
    """fp32 packs more cells into the same memory than fp64."""
    from midas_index.compute.matching import (
        pick_compare_strategy, _PEAK_BYTES_PER_CELL_FP32,
        _PEAK_BYTES_PER_CELL_FP64,
    )
    assert _PEAK_BYTES_PER_CELL_FP32 < _PEAK_BYTES_PER_CELL_FP64
    # A shape that just barely overflows fp32 will overflow fp64 even more.
    N, T, M = 5_000, 200, 50
    free = 1 * 1024**3
    theor64 = torch.zeros((N, T, M), dtype=torch.float64)
    theor32 = torch.zeros((N, T, M), dtype=torch.float32)
    s32, cs32 = pick_compare_strategy(theor32, max_n_cap=M,
                                       free_bytes=free, safety=0.5)
    s64, cs64 = pick_compare_strategy(theor64, max_n_cap=M,
                                       free_bytes=free, safety=0.5)
    if s32 == "jagged" and s64 == "jagged":
        assert cs32 >= cs64   # fp32 gets a larger chunk for the same memory
