"""Tests for soft beam attribution in ``midas_index.compute.matching``
(P6 of the V-map plan).

Covers:
- Back-compat: ``soft_beam_weight_fn=None`` leaves ``weighted_*`` fields
  ``None`` and integer counts unchanged from the existing scan filter.
- Hard-window soft fn (``hard_window_fn``) returns the same support as
  the legacy ``scan_pos_tol_um`` filter — boundary cases inclusive.
- Soft top-hat (``soft_top_hat_fn``) inside the beam → weight = 1;
  on the linear ramp → weight ∈ (0, 1); past the ramp → 0.
- Soft Gaussian (``soft_gaussian_fn``) at d = FWHM/2 → weight ≈ 0.5
  (the half-max definition).
- Friedel-symmetric soft mode: weight at a mirror-pair candidate is the
  maximum of the forward and antisymmetric weights.
- Jagged path threads ``soft_beam_weight_fn`` through and concatenates
  ``weighted_*`` fields correctly.
- Soft-fn helpers are autograd-differentiable in the distance argument
  (``torch.autograd.gradcheck`` on ``soft_gaussian_fn``).
- Multi-device: helpers run on CPU + MPS.
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import math
import numpy as np
import pytest
import torch

from midas_index.compute.matching import (
    build_eta_margins, build_ome_margins, compare_spots,
)
from midas_index.compute.soft_attribution import (
    hard_window_fn, soft_gaussian_fn, soft_top_hat_fn,
)


def _devices() -> list[str]:
    devs = ["cpu"]
    if torch.cuda.is_available():
        devs.append("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        devs.append("mps")
    return devs


# ----------------------------------------------------------- shared fixtures


def _spot10(y, z, omega, ring_rad, spot_id, ring_nr, eta, ttheta, rad_diff, scan_nr):
    return (y, z, omega, ring_rad, spot_id, ring_nr, eta, ttheta, rad_diff, scan_nr)


def _theor_row(omega, ring_nr, y, z, eta, rad_diff):
    return [0, 0, 0, 0, 0, 0, omega, 0, 0, ring_nr, y, z, eta, rad_diff]


def _build_bin(obs_eta, obs_ome, obs_ring_nr, eta_bin_size, ome_bin_size,
               n_eta_bins, n_ome_bins, n_rows):
    pos = (
        (obs_ring_nr - 1) * n_eta_bins * n_ome_bins
        + int(np.floor((180 + obs_eta) / eta_bin_size)) * n_ome_bins
        + int(np.floor((180 + obs_ome) / ome_bin_size))
    )
    ndata = torch.zeros(2 * (pos + 10), dtype=torch.int32)
    ndata[2 * pos] = n_rows
    ndata[2 * pos + 1] = 0
    bin_data = torch.arange(n_rows, dtype=torch.int32)
    return ndata, bin_data, pos


def _matching_kwargs():
    return dict(
        eta_margins=build_eta_margins(
            ring_radii={1: 30000.0}, margin_eta=20.0, stepsize_orient_deg=0.5,
            device=torch.device("cpu"), dtype=torch.float64,
        ),
        ome_margins=build_ome_margins(
            margin_ome=10.0, stepsize_orient_deg=0.5,
            device=torch.device("cpu"), dtype=torch.float64,
        ),
        eta_bin_size=0.1, ome_bin_size=5.0,
        n_eta_bins=3600, n_ome_bins=72,
        rings_to_reject=torch.tensor([], dtype=torch.int64),
        margin_rad=50.0, margin_radial=50.0,
    )


def _scan_setup_single_voxel(spot_scan_idxs, scan_positions_um, voxel_xy_um, omega=0.0):
    """Build a single-voxel, single-theor, multi-obs scan-aware setup.

    Each obs has the same (y, z, eta, ttheta) — distinct only by scan_nr.
    """
    obs10 = torch.tensor([
        _spot10(10.0, 5.0, omega, 30000.0, 100 + i, 1, 12.0, 1.5, 0.0, scan_nr=si)
        for i, si in enumerate(spot_scan_idxs)
    ], dtype=torch.float64)
    theor = torch.tensor([_theor_row(omega, 1, 10.0, 5.0, 12.0, 0.0)],
                         dtype=torch.float64).unsqueeze(0)
    valid = torch.ones((1, 1), dtype=torch.bool)
    kw = _matching_kwargs()
    ndata, bin_data, _ = _build_bin(12.0, omega, 1, kw["eta_bin_size"],
                                    kw["ome_bin_size"], kw["n_eta_bins"],
                                    kw["n_ome_bins"], n_rows=len(spot_scan_idxs))
    return dict(
        obs=obs10, theor=theor, valid=valid,
        bin_data=bin_data, bin_ndata=ndata,
        ref_rad=torch.tensor([30000.0], dtype=torch.float64),
        scan_positions=torch.as_tensor(scan_positions_um, dtype=torch.float64),
        voxel_xy=torch.as_tensor(voxel_xy_um, dtype=torch.float64),
        **kw,
    )


# ----------------------------------------------------------- back-compat


def test_no_soft_fn_leaves_weighted_fields_none():
    """Without ``soft_beam_weight_fn``, weighted_* fields stay None."""
    setup = _scan_setup_single_voxel([0], [0.0], [[0.0, 0.0]])
    res = compare_spots(**setup, scan_pos_tol_um=2.0)
    assert res.weighted_n_matches is None
    assert res.weighted_n_matches_frac is None
    assert res.weighted_frac_matches is None
    assert int(res.n_matches.item()) == 1


def test_no_soft_fn_with_zero_tol_still_no_weighted_fields():
    """Even with the scan filter inactive (tol=0), the weighted fields are None
    when no soft fn is provided."""
    setup = _scan_setup_single_voxel([0], [0.0], [[0.0, 0.0]])
    # tol=0 short-circuits the scan-aware block entirely
    setup.pop("scan_positions"); setup.pop("voxel_xy")
    res = compare_spots(**setup)
    assert res.weighted_n_matches is None


# ----------------------------------------------------------- soft equivalence


def test_hard_window_fn_reproduces_binary_filter():
    """A ``hard_window_fn(tol)`` soft fn must give the same hard support as
    the existing ``scan_pos_tol_um=tol`` filter (i.e., binary mask)."""
    setup = _scan_setup_single_voxel(
        spot_scan_idxs=[0, 1],            # 2 candidates
        scan_positions_um=[0.0, 20.0],    # tol=2 should keep only the first
        voxel_xy_um=[[0.0, 0.0]],
    )
    # Binary mode
    res_hard = compare_spots(**setup, scan_pos_tol_um=2.0)
    # Soft mode with hard window of the same tol (filter is "<", same as legacy)
    res_soft = compare_spots(**setup, scan_pos_tol_um=2.0,
                              soft_beam_weight_fn=hard_window_fn(2.0))
    assert int(res_hard.n_matches.item()) == int(res_soft.n_matches.item()) == 1
    # weighted count == 1 (the one matching candidate has weight 1)
    assert math.isclose(float(res_soft.weighted_n_matches[0]), 1.0)
    # weighted frac == 1.0 (1 of 1 valid theor matched, weighted by 1)
    assert math.isclose(float(res_soft.weighted_frac_matches[0]), 1.0)


def test_soft_top_hat_partial_weight():
    """Spot at distance 6 µm with TopHat width=10 + fall_off=4 → weight = (5+4-6)/4 = 0.75."""
    # MIDAS omega is in *degrees*; ω=90 → s_proj = 10·sin(90°) + 0·cos(90°) = 10.
    # Scan position 4 µm: |10 - 4| = 6 µm.
    setup = _scan_setup_single_voxel(
        spot_scan_idxs=[0],
        scan_positions_um=[4.0],
        voxel_xy_um=[[10.0, 0.0]],
        omega=90.0,
    )
    # Use tol large enough to pass the hard pre-prune (we'll override with soft)
    res = compare_spots(
        **setup, scan_pos_tol_um=20.0,
        soft_beam_weight_fn=soft_top_hat_fn(10.0, fall_off_um=4.0),
    )
    assert int(res.n_matches.item()) == 1
    # Expected weight: distance 6 is past half-width (5) by 1, ramp end is 5+4=9
    # weight = clamp((9 - 6) / 4, 0, 1) = 0.75
    assert math.isclose(float(res.weighted_n_matches[0]), 0.75, rel_tol=1e-9)


def test_soft_gaussian_half_max_at_fwhm_over_two():
    """Gaussian fwhm=10 at d=5 → exp(-25 / (2 σ²)) with σ = 5/sqrt(2 ln 2) ≈ 4.247
    → exp(-25 / (2 * 18.03)) = exp(-0.693) = 0.5."""
    # ω=90° → s_proj = 10.  Scan = 5 → distance = 5.
    setup = _scan_setup_single_voxel(
        spot_scan_idxs=[0], scan_positions_um=[5.0], voxel_xy_um=[[10.0, 0.0]],
        omega=90.0,
    )
    res = compare_spots(
        **setup, scan_pos_tol_um=20.0,
        soft_beam_weight_fn=soft_gaussian_fn(10.0),
    )
    assert int(res.n_matches.item()) == 1
    assert math.isclose(float(res.weighted_n_matches[0]), 0.5, abs_tol=1e-9)


def test_soft_top_hat_outside_beam_zero_match():
    """Spot well outside top-hat support (no fall-off) → no match counted."""
    setup = _scan_setup_single_voxel(
        spot_scan_idxs=[0], scan_positions_um=[50.0], voxel_xy_um=[[0.0, 0.0]],
    )
    res = compare_spots(
        **setup, scan_pos_tol_um=100.0,
        soft_beam_weight_fn=soft_top_hat_fn(10.0),
    )
    assert int(res.n_matches.item()) == 0
    assert float(res.weighted_n_matches[0]) == 0.0


# ----------------------------------------------------------- Friedel


def test_friedel_soft_takes_max_of_pair_weights():
    """At ω=0, voxel (0, 5) → s_proj = 5.  Scan at -5 µm:
    forward distance = |5 - (-5)| = 10 → Gaussian weight tiny
    antisym distance = |-5 - (-5)| = 0 → weight = 1
    With friedel ON, the weight is max → 1."""
    setup = _scan_setup_single_voxel(
        spot_scan_idxs=[0], scan_positions_um=[-5.0], voxel_xy_um=[[0.0, 5.0]],
        omega=0.0,
    )
    res_no_friedel = compare_spots(
        **setup, scan_pos_tol_um=20.0,
        soft_beam_weight_fn=soft_gaussian_fn(2.0),
        friedel_symmetric_scan_filter=False,
    )
    res_friedel = compare_spots(
        **setup, scan_pos_tol_um=20.0,
        soft_beam_weight_fn=soft_gaussian_fn(2.0),
        friedel_symmetric_scan_filter=True,
    )
    # Without Friedel: weight = Gaussian(d=10, fwhm=2) ≈ 0
    w_off = float(res_no_friedel.weighted_n_matches[0])
    # With Friedel: weight = max(Gaussian(10), Gaussian(0)) = 1
    w_on = float(res_friedel.weighted_n_matches[0])
    assert w_off < 1e-6
    assert math.isclose(w_on, 1.0, abs_tol=1e-12)


# ----------------------------------------------------------- jagged path


def test_jagged_threads_soft_fn_through():
    """The jagged path must propagate the soft fn and concatenate the
    weighted_* fields across chunks."""
    # Build a small setup duplicated to N=3 tuples (one chunk forces split).
    n = 3
    obs10 = torch.tensor([
        _spot10(10.0, 5.0, 0.0, 30000.0, 17, 1, 12.0, 1.5, 0.0, scan_nr=0),
    ], dtype=torch.float64)
    theor_row = _theor_row(0.0, 1, 10.0, 5.0, 12.0, 0.0)
    theor = torch.tensor([theor_row] * n, dtype=torch.float64).unsqueeze(1)  # (n, 1, 14)
    valid = torch.ones((n, 1), dtype=torch.bool)
    kw = _matching_kwargs()
    ndata, bin_data, _ = _build_bin(12.0, 0.0, 1, kw["eta_bin_size"],
                                    kw["ome_bin_size"], kw["n_eta_bins"],
                                    kw["n_ome_bins"], n_rows=1)
    res = compare_spots(
        theor=theor, valid=valid, obs=obs10,
        bin_data=bin_data, bin_ndata=ndata,
        ref_rad=torch.full((n,), 30000.0, dtype=torch.float64),
        scan_positions=torch.tensor([0.0], dtype=torch.float64),
        voxel_xy=torch.zeros((n, 2), dtype=torch.float64),
        scan_pos_tol_um=2.0,
        soft_beam_weight_fn=hard_window_fn(2.0),
        strategy="jagged", chunk_size=2,       # force chunking (3 > 2)
        **kw,
    )
    assert res.weighted_n_matches is not None
    assert res.weighted_n_matches.shape == (n,)
    assert torch.allclose(res.weighted_n_matches, torch.ones(n, dtype=torch.float64))


def test_jagged_keeps_weighted_none_when_no_fn():
    n = 3
    obs10 = torch.tensor([
        _spot10(10.0, 5.0, 0.0, 30000.0, 17, 1, 12.0, 1.5, 0.0, scan_nr=0),
    ], dtype=torch.float64)
    theor_row = _theor_row(0.0, 1, 10.0, 5.0, 12.0, 0.0)
    theor = torch.tensor([theor_row] * n, dtype=torch.float64).unsqueeze(1)
    valid = torch.ones((n, 1), dtype=torch.bool)
    kw = _matching_kwargs()
    ndata, bin_data, _ = _build_bin(12.0, 0.0, 1, kw["eta_bin_size"],
                                    kw["ome_bin_size"], kw["n_eta_bins"],
                                    kw["n_ome_bins"], n_rows=1)
    res = compare_spots(
        theor=theor, valid=valid, obs=obs10,
        bin_data=bin_data, bin_ndata=ndata,
        ref_rad=torch.full((n,), 30000.0, dtype=torch.float64),
        scan_positions=torch.tensor([0.0], dtype=torch.float64),
        voxel_xy=torch.zeros((n, 2), dtype=torch.float64),
        scan_pos_tol_um=2.0,
        strategy="jagged", chunk_size=2,
        **kw,
    )
    assert res.weighted_n_matches is None


# ----------------------------------------------------------- helper fns


def test_soft_top_hat_no_fall_off_is_binary():
    fn = soft_top_hat_fn(10.0)
    d = torch.tensor([0.0, 4.999, 5.0, 5.001])
    # Half-width = 5; "d < 5" strict
    w = fn(d)
    assert w.tolist() == [1.0, 1.0, 0.0, 0.0]


def test_soft_gaussian_truncation():
    fn = soft_gaussian_fn(10.0, truncate_at=3.0)
    d = torch.tensor([0.0, 2.0, 3.001, 5.0])
    w = fn(d)
    # d=0,2 within truncation; d=3.001,5 outside → 0
    assert w[0] > 0 and w[1] > 0
    assert w[2] == 0.0 and w[3] == 0.0


def test_soft_gaussian_gradcheck():
    """The Gaussian fn is smooth in d → gradcheck passes."""
    fn = soft_gaussian_fn(10.0)
    def g(d):
        return fn(d).sum()
    d = torch.tensor([1.0, 3.0, 5.0], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(g, (d,), eps=1e-6, atol=1e-7)


def test_soft_top_hat_ramp_gradcheck():
    """Inside the linear ramp, the soft top-hat is locally smooth."""
    fn = soft_top_hat_fn(10.0, fall_off_um=4.0)
    def g(d):
        return fn(d).sum()
    # Stay strictly inside the ramp [5, 9]
    d = torch.tensor([5.5, 6.0, 7.5, 8.5], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(g, (d,), eps=1e-6, atol=1e-7)


# ----------------------------------------------------------- device


@pytest.mark.parametrize("device", _devices())
def test_soft_helpers_runs_on_device(device):
    dtype = torch.float64 if device != "mps" else torch.float32
    d = torch.tensor([0.0, 2.0, 5.0, 10.0], dtype=dtype, device=device)
    for fn in [hard_window_fn(5.0), soft_top_hat_fn(10.0),
               soft_top_hat_fn(10.0, fall_off_um=2.0),
               soft_gaussian_fn(10.0), soft_gaussian_fn(10.0, truncate_at=8.0)]:
        w = fn(d)
        assert w.device.type == torch.device(device).type
        assert w.shape == d.shape
        assert (w >= 0).all() and (w <= 1).all()
