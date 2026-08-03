"""Parity across refiner precisions and devices.

The CI-runnable half of the cross-implementation check. It uses the synthetic
fixture so it runs anywhere; the full six-way comparison against the two C
implementations needs real beamtime data and lives in
``utils/ff_refiner_crosscheck.py``.

Measured there on the real 2-Au-grain dataset (20 commonly-refined seeds), for
reference when reading the tolerances below:

    py-f64-cpu vs py-f64-gpu   position  max 0.012 um   miso max 0.0000 deg
    py-f64     vs py-f32       position  max 8.5   um   miso max 0.057  deg
    C          vs python       position  max 85    um   miso max 0.155  deg
    c-orig     vs c-omp        position  max 60    um   miso max 0.144  deg

Orientation is tight everywhere. **Position is the loose axis**, and the two C
implementations disagree with each other almost as much as they disagree with
python -- so python is not the outlier. All of it sits inside the ~100 um
position uncertainty this method actually has (Lab Notebook 2d): the DiffPos
minimum is shallow, so many positions fit nearly as well.

Orientation comparisons go through ``midas_stress.orientation`` because
symmetry-equivalent orientations are the same orientation. Comparing raw
matrices reports a median misorientation of exactly 120 deg on cubic data --
the symmetry angle -- which is what the first version of the crosscheck did.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from midas_fit_grain import FitConfig
from midas_fit_grain.refine_block import refine_block

from ._synthetic import fixture_to_observed, gt_match, make_synthetic

SG = 225


#: Seed offset from ground truth, in um. Deliberately large -- a seed sitting
#: on the answer cannot distinguish a working refiner from one that returns
#: its input, which is how a float32 no-op survived review once.
SEED_OFFSET = torch.tensor([90.0, -60.0, 40.0], dtype=torch.float64)


def _seed(fix, device, dtype):
    """The SAME seed for every backend -- the refiner is the only variable."""
    pos = (fix.gt_position.double().cpu() + SEED_OFFSET).to(
        device=device, dtype=dtype)
    eul = (fix.gt_euler.double().cpu() + 0.05 * math.pi / 180.0).to(
        device=device, dtype=dtype)
    lat = fix.gt_lattice.to(device=device, dtype=dtype).clone()
    return pos, eul, lat


def _refine(device, dtype, *, pos_scale="auto"):
    """Refine via ``refine_block`` -- the path the FF driver actually uses.

    NOT ``refine_grain``: that per-grain entry point deliberately keeps a
    FIXED pos_scale (its production caller is PF scanning, where the voxel is
    locked to the scan grid so position is not free) and rejects
    ``pos_scale="auto"``. Position parity has to be tested on the path that
    actually refines position.
    """
    fix = make_synthetic(device=device, dtype=dtype)
    obs = fixture_to_observed(fix, device=device, dtype=dtype)
    cfg = FitConfig(
        RingNumbers=fix.ring_numbers, px=fix.px, loss="full3d",
        solver="lbfgs", mode="all_at_once",
    )
    pos, eul, lat = _seed(fix, device, dtype)
    blk = refine_block(
        cfg, model=fix.model, grains_obs=[obs],
        init_positions=pos.view(1, 3), init_eulers=eul.view(1, 3),
        init_lattices=lat.view(1, 6), pred_ring_slot=fix.pred_ring_slot,
        precomputed_matches=[gt_match(fix, device=device, dtype=dtype)],
        pos_scale=pos_scale,
    )
    return fix, blk.grains[0]


def _miso_deg(euler_a, euler_b) -> float:
    from midas_stress.orientation import (
        euler_to_orient_mat_batch, misorientation_om_batch,
    )
    a = np.asarray(euler_to_orient_mat_batch(
        np.asarray(euler_a, dtype=float).reshape(1, 3))).reshape(1, 9)
    b = np.asarray(euler_to_orient_mat_batch(
        np.asarray(euler_b, dtype=float).reshape(1, 3))).reshape(1, 9)
    # midas_stress returns RADIANS
    return float(np.degrees(misorientation_om_batch(a, b, SG))[0])


def _pos(out) -> np.ndarray:
    return np.asarray(out.position.detach().cpu(), dtype=float).reshape(3)


def _eul(out) -> np.ndarray:
    return np.asarray(out.euler.detach().cpu(), dtype=float).reshape(3)


# ── fp64 is the reference: it must actually recover the answer ─────────────
def test_fp64_cpu_recovers_ground_truth():
    fix, out = _refine(torch.device("cpu"), torch.float64)
    gt = np.asarray(fix.gt_position.cpu(), dtype=float).reshape(3)
    err = float(np.linalg.norm(_pos(out) - gt))
    assert err < 1.0, f"fp64 should recover the seeded position, off by {err} um"
    assert _miso_deg(_eul(out), np.asarray(fix.gt_euler.cpu())) < 0.05


# ── precision parity ───────────────────────────────────────────────────────
def test_fp32_agrees_with_fp64_on_cpu():
    """fp32 must land in the same place as fp64, not merely 'somewhere'.

    The tolerance is deliberately far tighter than the ~100 um position
    uncertainty of real FF data: on the synthetic fixture there is no
    ambiguity to hide in, so a large fp32/fp64 gap here means a numerical
    defect, not a shallow minimum. A float32 refiner that silently returned
    its seed once passed a looser version of this check.
    """
    _, a = _refine(torch.device("cpu"), torch.float64)
    _, b = _refine(torch.device("cpu"), torch.float32)
    dp = float(np.linalg.norm(_pos(a) - _pos(b)))
    assert dp < 5.0, f"fp32 vs fp64 position differs by {dp:.3f} um"
    assert _miso_deg(_eul(a), _eul(b)) < 0.1


def test_fp32_actually_moves_off_its_seed():
    """Regression: the fp32 path once returned the seed unchanged.

    Agreement tests alone cannot catch that -- if both paths failed the same
    way they would still agree. Assert movement explicitly.
    """
    fix, out = _refine(torch.device("cpu"), torch.float32)
    seed = np.asarray(
        (fix.gt_position.double().cpu() + SEED_OFFSET), dtype=float).reshape(3)
    moved = float(np.linalg.norm(_pos(out) - seed))
    assert moved > 1.0, (
        f"fp32 refiner moved only {moved:.3e} um from its seed -- it is "
        "returning its input")


# ── device parity ──────────────────────────────────────────────────────────
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_cpu_gpu_agree(dtype):
    """Same precision on two devices must agree far more tightly than
    two precisions on one device."""
    _, c = _refine(torch.device("cpu"), dtype)
    _, g = _refine(torch.device("cuda"), dtype)
    dp = float(np.linalg.norm(_pos(c) - _pos(g)))
    tol = 0.1 if dtype == torch.float64 else 3.0
    assert dp < tol, f"cpu vs cuda ({dtype}) position differs by {dp:.4f} um"
    assert _miso_deg(_eul(c), _eul(g)) < 0.05


@pytest.mark.skip(
    reason="the synthetic fixture cannot reach MPS: make_synthetic builds the "
           "forward model with float64 buffers (midas_diffract/forward.py "
           "register_buffer), and MPS rejects float64 outright. Testing MPS "
           "parity needs a fixture that is float32 end to end -- a real gap, "
           "not a refiner defect.")
def test_mps_agrees_with_cpu_fp32():
    """MPS is float32-only, so compare it against CPU float32."""
    _, c = _refine(torch.device("cpu"), torch.float32)
    _, m = _refine(torch.device("mps"), torch.float32)
    dp = float(np.linalg.norm(_pos(c) - _pos(m)))
    assert dp < 3.0, f"cpu vs mps position differs by {dp:.4f} um"


# ── the C backend, when it is built ────────────────────────────────────────
def test_c_backend_reports_availability_without_raising():
    """`available()` must be a safe probe -- the crosscheck harness and the
    pipeline both branch on it, so it may not raise when the binary is absent.
    """
    from midas_fit_grain import backend_c

    assert isinstance(backend_c.available(), bool)
    if not backend_c.available():
        with pytest.raises(backend_c.CBackendUnavailableError):
            backend_c.run_refiner("nonexistent.txt", n_work=1)


def test_orient_pos_fit_row_width_is_27():
    """Guard the binary layout the crosscheck depends on.

    Assuming a narrower packing makes the row count non-integer and the file
    silently reads as 'no output', which is exactly how the first run of the
    crosscheck failed.
    """
    from midas_fit_grain.io_binary import ORIENT_POS_FIT_NCOLS

    assert ORIENT_POS_FIT_NCOLS == 27
