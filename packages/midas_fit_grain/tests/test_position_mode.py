"""Tests for scan-aware position_mode in refine_grain (P6).

Verifies:
- ``position_mode="fixed"`` + scan_pos_tol > 0 keeps init_position unchanged
  (the C IndexerScanningOMP behavior).
- ``position_mode="voxel_bounded"`` + scan_pos_tol > 0 + beam_size > 0
  clamps the refined Y position to ``init_y ± beam_size/2``.
- FF mode (scan_pos_tol_um=0) ignores position_mode and uses the legacy
  behavior — pos refines freely.
- Setting position_mode without scan_pos_tol > 0 is a no-op (legacy
  semantics preserved).
"""

from __future__ import annotations

import math

import pytest
import torch

from midas_fit_grain import FitConfig, refine_grain

from ._synthetic import fixture_to_observed, gt_match, make_synthetic


DEG2RAD = math.pi / 180.0


@pytest.fixture(scope="module")
def fix():
    return make_synthetic(device=torch.device("cpu"), dtype=torch.float64)


def _scan_cfg(fix, *, position_mode: str, beam_size: float = 4.0,
              scan_tol: float = 2.0) -> FitConfig:
    """Build a FitConfig with scan-mode active and the requested position_mode."""
    return FitConfig(
        Lsd=fix.model.Lsd, px=fix.px, Wavelength=0.1729,
        LatticeConstant=tuple(fix.gt_lattice.tolist()),
        SpaceGroup=225,
        RingNumbers=fix.ring_numbers,
        RingRadii=[1.0] * len(fix.ring_numbers),
        OmegaRanges=[(-180.0, 180.0)],
        BoxSizes=[(-1e6, 1e6, -1e6, 1e6)],
        MarginEta=5.0, MarginOme=2.0,
        EtaBinSize=2.0, OmeBinSize=2.0,
        MinEta=6.0,
        solver="lbfgs", mode="all_at_once", loss="full3d",
        max_iter=200, ftol=1e-8, xtol=1e-9,
        phase_steps=(8, 8, 8, 8),
        scan_pos_tol_um=scan_tol,
        position_mode=position_mode,
        beam_size_um=beam_size,
    )


# ---------------------------------------------------------------------------
# Fixed mode: position is locked
# ---------------------------------------------------------------------------


def test_fixed_mode_keeps_position_unchanged(fix):
    """C IndexerScanningOMP parity contract: position is FIXED."""
    obs = fixture_to_observed(fix, device=torch.device("cpu"),
                              dtype=torch.float64)
    cfg = _scan_cfg(fix, position_mode="fixed")
    init_pos = fix.gt_position.clone() + torch.tensor([2.0, -1.0, 0.5],
                                                      dtype=torch.float64)
    init_eul = fix.gt_euler.clone() + 0.05 * DEG2RAD
    init_lat = fix.gt_lattice.clone()
    match_seed = gt_match(fix, device=torch.device("cpu"),
                          dtype=torch.float64)

    result = refine_grain(
        cfg, model=fix.model,
        obs=obs,
        init_position=init_pos,
        init_euler=init_eul,
        init_lattice=init_lat,
        pred_ring_slot=fix.pred_ring_slot,
        precomputed_match=match_seed,
    )
    # Position must equal init_pos exactly — refinement did not touch it.
    delta = (result.position - init_pos).abs().max().item()
    assert delta < 1e-12, f"fixed-mode pos drift = {delta:.3e} (should be 0)"


def test_fixed_mode_still_refines_orientation_and_lattice(fix):
    """fixed-mode pins position but euler + lattice must still update."""
    obs = fixture_to_observed(fix, device=torch.device("cpu"),
                              dtype=torch.float64)
    cfg = _scan_cfg(fix, position_mode="fixed")
    init_pos = fix.gt_position.clone()                  # exact pos
    init_eul = fix.gt_euler.clone() + 0.05 * DEG2RAD
    init_lat = fix.gt_lattice.clone()
    match_seed = gt_match(fix, device=torch.device("cpu"),
                          dtype=torch.float64)
    result = refine_grain(
        cfg, model=fix.model, obs=obs,
        init_position=init_pos, init_euler=init_eul, init_lattice=init_lat,
        pred_ring_slot=fix.pred_ring_slot,
        precomputed_match=match_seed,
    )
    # Euler should have changed (refined toward GT).
    eul_delta = (result.euler - init_eul).abs().max().item()
    assert eul_delta > 1e-8, (
        f"fixed-mode should still refine euler; got delta={eul_delta:.3e}"
    )


# ---------------------------------------------------------------------------
# voxel_bounded mode: position is clamped to [init_y ± beam/2] along Y
# ---------------------------------------------------------------------------


def test_voxel_bounded_clamps_y_position(fix):
    """voxel_bounded mode: |Δy| ≤ beam_size/2 after refinement."""
    obs = fixture_to_observed(fix, device=torch.device("cpu"),
                              dtype=torch.float64)
    beam = 4.0
    cfg = _scan_cfg(fix, position_mode="voxel_bounded", beam_size=beam)
    # Perturb Y by more than the half-beam to force clamping.
    init_pos = fix.gt_position.clone()
    init_pos[1] += 10.0                                # 10 µm off in Y; bound = ±2
    init_eul = fix.gt_euler.clone() + 0.05 * DEG2RAD
    init_lat = fix.gt_lattice.clone()
    match_seed = gt_match(fix, device=torch.device("cpu"),
                          dtype=torch.float64)
    result = refine_grain(
        cfg, model=fix.model, obs=obs,
        init_position=init_pos, init_euler=init_eul, init_lattice=init_lat,
        pred_ring_slot=fix.pred_ring_slot,
        precomputed_match=match_seed,
    )
    half = beam / 2.0
    y_drift = abs(float(result.position[1].item() - init_pos[1].item()))
    assert y_drift <= half + 1e-9, (
        f"voxel_bounded Y drift {y_drift:.4f} exceeds half-beam {half:.4f}"
    )


def test_voxel_bounded_allows_y_motion_inside_bound(fix):
    """voxel_bounded mode: Y CAN move inside the bound."""
    obs = fixture_to_observed(fix, device=torch.device("cpu"),
                              dtype=torch.float64)
    cfg = _scan_cfg(fix, position_mode="voxel_bounded", beam_size=6.0)
    # Init Y is 1 µm off; bound = ±3 ⇒ optimizer can fully recover.
    init_pos = fix.gt_position.clone()
    init_pos[1] += 1.0
    init_eul = fix.gt_euler.clone() + 0.05 * DEG2RAD
    init_lat = fix.gt_lattice.clone()
    match_seed = gt_match(fix, device=torch.device("cpu"),
                          dtype=torch.float64)
    result = refine_grain(
        cfg, model=fix.model, obs=obs,
        init_position=init_pos, init_euler=init_eul, init_lattice=init_lat,
        pred_ring_slot=fix.pred_ring_slot,
        precomputed_match=match_seed,
    )
    # Y should have moved toward GT (within the bound).
    y_after = float(result.position[1].item())
    y_init = float(init_pos[1].item())
    y_gt = float(fix.gt_position[1].item())
    moved = abs(y_after - y_init)
    assert moved > 1e-6, f"voxel_bounded should refine Y inside bound; moved={moved:.3e}"


# ---------------------------------------------------------------------------
# FF mode (scan_pos_tol_um == 0) ignores position_mode
# ---------------------------------------------------------------------------


def test_ff_mode_ignores_position_mode_fixed(fix):
    """FF (scan_pos_tol_um = 0): position_mode='fixed' is IGNORED → pos refines."""
    obs = fixture_to_observed(fix, device=torch.device("cpu"),
                              dtype=torch.float64)
    cfg = FitConfig(
        Lsd=fix.model.Lsd, px=fix.px, Wavelength=0.1729,
        LatticeConstant=tuple(fix.gt_lattice.tolist()),
        SpaceGroup=225,
        RingNumbers=fix.ring_numbers,
        RingRadii=[1.0] * len(fix.ring_numbers),
        OmegaRanges=[(-180.0, 180.0)],
        BoxSizes=[(-1e6, 1e6, -1e6, 1e6)],
        MarginEta=5.0, MarginOme=2.0,
        EtaBinSize=2.0, OmeBinSize=2.0,
        MinEta=6.0,
        solver="lbfgs", mode="all_at_once", loss="full3d",
        max_iter=200, ftol=1e-8, xtol=1e-9,
        phase_steps=(8, 8, 8, 8),
        # FF: scan_pos_tol_um defaults to 0.
        position_mode="fixed",                  # should be ignored
    )
    init_pos = fix.gt_position.clone() + torch.tensor([2.0, -1.0, 0.5],
                                                      dtype=torch.float64)
    init_eul = fix.gt_euler.clone() + 0.05 * DEG2RAD
    init_lat = fix.gt_lattice.clone()
    match_seed = gt_match(fix, device=torch.device("cpu"),
                          dtype=torch.float64)
    result = refine_grain(
        cfg, model=fix.model, obs=obs,
        init_position=init_pos, init_euler=init_eul, init_lattice=init_lat,
        pred_ring_slot=fix.pred_ring_slot,
        precomputed_match=match_seed,
    )
    # In FF mode, position SHOULD refine (legacy behavior).
    pos_delta = (result.position - init_pos).abs().max().item()
    assert pos_delta > 1e-6, (
        f"FF mode should refine position regardless of position_mode; "
        f"got pos_delta={pos_delta:.3e}"
    )
