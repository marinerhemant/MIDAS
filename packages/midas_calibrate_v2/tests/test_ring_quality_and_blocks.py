"""Per-ring quality filter and the distortion-block selector."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from midas_calibrate.params import CalibrationParams

from midas_calibrate_v2.forward.distortion import (
    DISTORTION_BLOCKS, resolve_distortion_block, P_COEF_NAMES,
)
from midas_calibrate_v2.pipelines.auto import _distortion_refine_flags
from midas_calibrate_v2.pipelines._common import (
    FittedDataset, ring_quality, filter_fits_by_ring_quality,
)


# ------------------------------------------------------ per-ring quality

def _fits(spec):
    """spec: list of (n_eta, snr) per ring."""
    Y, Z, rid, snr = [], [], [], []
    for r, (n, s) in enumerate(spec):
        Y += list(np.linspace(10, 20, n))
        Z += list(np.linspace(10, 20, n))
        rid += [r] * n
        snr += [s] * n
    t = lambda v: torch.as_tensor(v, dtype=torch.float64)
    n_tot = len(rid)
    return FittedDataset(
        Y_pix=t(Y), Z_pix=t(Z),
        ring_idx=torch.as_tensor(rid, dtype=torch.long),
        snr=torch.ones(n_tot, dtype=torch.float64),
        ring_two_theta_deg=torch.full((n_tot,), 3.0, dtype=torch.float64),
        rho_d=t(50000.0), weights=torch.ones(n_tot, dtype=torch.float64),
        snr_baseline=t(snr))


def test_ring_quality_measures_each_ring():
    q = ring_quality(_fits([(30, 9.0), (4, 8.0), (25, 1.2)]))
    assert [r.ring_idx for r in q] == [0, 1, 2]
    assert [r.n_eta for r in q] == [30, 4, 25]
    assert q[0].snr_median == pytest.approx(9.0)
    assert q[2].snr_median == pytest.approx(1.2)


def test_disabled_by_default_returns_input_unchanged():
    f = _fits([(30, 9.0), (4, 1.0)])
    out, rep = filter_fits_by_ring_quality(f)
    assert out is f
    assert all(r.kept for r in rep)


def test_drops_ring_with_too_few_eta_bins():
    f = _fits([(30, 9.0), (4, 9.0)])
    out, rep = filter_fits_by_ring_quality(f, min_eta_bins=10, min_rings_kept=0)
    assert out.Y_pix.numel() == 30
    assert set(out.ring_idx.tolist()) == {0}
    dropped = [r for r in rep if not r.kept]
    assert len(dropped) == 1 and "η bins" in dropped[0].reason


def test_drops_ring_below_snr():
    f = _fits([(30, 9.0), (30, 1.2)])
    out, rep = filter_fits_by_ring_quality(f, min_ring_snr=5.0, min_rings_kept=0)
    assert set(out.ring_idx.tolist()) == {0}
    assert "SNR" in [r for r in rep if not r.kept][0].reason


def test_both_criteria_reported_together():
    f = _fits([(30, 9.0), (3, 1.0)])
    _, rep = filter_fits_by_ring_quality(f, min_eta_bins=10, min_ring_snr=5.0,
                                          min_rings_kept=0)
    bad = [r for r in rep if not r.kept][0]
    assert "η bins" in bad.reason and "SNR" in bad.reason


def test_every_aligned_column_is_filtered_together():
    f = _fits([(30, 9.0), (4, 9.0)])
    f.ring_d_spacing_A = torch.ones_like(f.Y_pix)
    f.phase_idx = torch.zeros_like(f.ring_idx)
    out, _ = filter_fits_by_ring_quality(f, min_eta_bins=10, min_rings_kept=0)
    n = out.Y_pix.numel()
    for name in ("Z_pix", "ring_idx", "snr", "ring_two_theta_deg", "weights",
                 "ring_d_spacing_A", "phase_idx", "snr_baseline"):
        v = getattr(out, name)
        assert v is not None and v.numel() == n, name


def test_never_empties_the_dataset():
    """The filter runs on EVERY E-step, including scoring passes at a wandered
    geometry. Aborting there would turn a bad iterate into a dead run, so it
    keeps the best-ranked rings and warns instead."""
    f = _fits([(4, 1.0), (3, 1.0), (9, 2.0), (2, 1.0), (7, 1.5)])
    with pytest.warns(RuntimeWarning, match="would have left"):
        out, rep = filter_fits_by_ring_quality(
            f, min_eta_bins=10, min_ring_snr=5.0, min_rings_kept=4)
    kept = sorted(set(out.ring_idx.tolist()))
    assert len(kept) == 4
    # the rescued rings are the best-covered ones, not an arbitrary slice
    assert 2 in kept and 4 in kept          # n_eta 9 and 7
    assert 3 not in kept                    # n_eta 2, the worst
    assert any("[kept: floor]" in r.reason for r in rep)


def test_floor_does_not_fire_when_enough_rings_survive():
    f = _fits([(30, 9.0), (30, 9.0), (30, 9.0), (30, 9.0), (30, 9.0), (2, 1.0)])
    out, rep = filter_fits_by_ring_quality(f, min_eta_bins=10, min_ring_snr=5.0,
                                            min_rings_kept=4)
    assert sorted(set(out.ring_idx.tolist())) == [0, 1, 2, 3, 4]
    assert not any("floor" in r.reason for r in rep)


def test_params_parse_the_two_thresholds(tmp_path):
    p = tmp_path / "ps.txt"
    p.write_text("Wavelength 0.15\nMinEtaBinsPerRing 10\nMinRingSNR 5.0\n")
    v1 = CalibrationParams.from_file(p)
    assert v1.MinEtaBinsPerRing == 10
    assert v1.MinRingSNR == pytest.approx(5.0)


# ------------------------------------------------------ distortion blocks

def test_blocks_are_cumulative_and_radial_first():
    prev = set()
    for name in ("none", "radial", "radial+2fold", "radial+1fold",
                 "radial+3fold", "radial+4fold", "full"):
        cur = set(DISTORTION_BLOCKS[name])
        assert prev <= cur, f"{name} is not a superset of the previous block"
        prev = cur
    assert set(DISTORTION_BLOCKS["full"]) == set(P_COEF_NAMES)
    assert DISTORTION_BLOCKS["radial"] == ("iso_R2", "iso_R4", "iso_R6")


def test_bool_selectors_keep_old_behaviour():
    assert resolve_distortion_block(True) == DISTORTION_BLOCKS["full"]
    assert resolve_distortion_block(False) == ()
    assert resolve_distortion_block(None) == ()


def test_explicit_name_list_accepted():
    got = resolve_distortion_block(["iso_R2", "a3", "phi3"])
    assert got == ("iso_R2", "a3", "phi3")


def test_amplitude_without_its_phase_is_rejected():
    with pytest.raises(ValueError, match="must be refined together"):
        resolve_distortion_block(["a2"])
    with pytest.raises(ValueError, match="must be refined together"):
        resolve_distortion_block(["phi2"])


def test_unknown_block_and_coefficient_rejected():
    with pytest.raises(ValueError, match="unknown distortion block"):
        resolve_distortion_block("iso")
    with pytest.raises(ValueError, match="unknown distortion coefficient"):
        resolve_distortion_block(["not_a_coeff"])


def test_flags_map_to_the_right_v1_slots():
    """The v1 p-index layout is a permutation, not positional — iso_R2/R4/R6
    are p2/p5/p4, which is exactly the kind of thing to get wrong silently."""
    flags = _distortion_refine_flags("radial")
    on = {k for k, v in flags.items() if v}
    assert on == {"p2", "p4", "p5"}
    assert _distortion_refine_flags(False) == {f"p{i}": False for i in range(15)}
    assert all(_distortion_refine_flags(True).values())


def test_radial_block_leaves_every_harmonic_off():
    flags = _distortion_refine_flags("radial")
    from midas_distortion import V1_TO_V2_DISTORTION as M
    for i in range(15):
        if M[i] not in ("iso_R2", "iso_R4", "iso_R6"):
            assert not flags[f"p{i}"], f"{M[i]} should be frozen"
