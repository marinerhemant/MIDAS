"""The fused Triton kernel must apply the same OmegaRange/BoxSize gate as eager.

``midas_nf_fitorientation.triton_kernels.fused_hard_frac`` fuses the whole
NF forward + obs lookup into one launch. It carries its own inlined copy of
the paired ``OmegaRange`` / ``BoxSize`` acceptance gate that
``midas_diffract.forward.HEDMForwardModel.omega_box_mask`` implements on the
eager path -- so the two can drift. These tests hold both to the same
contract.

Reference C: the ``KeepSpot`` loop of ``CalcDiffrSpots_Furnace``
(``NF_HEDM/src/CalcDiffractionSpots.c:225-243``)::

    RealType RingRadius = distance * tan(2 * deg2rad * Thetas[indexhkl]);
    CalcSpotPosition(RingRadius, etas[i], &yl, &zl);
      # yl = -(sin(eta) * RingRadius);  zl = cos(eta) * RingRadius;
    for (OmegaRangeNo = 0; OmegaRangeNo < NOmegaRanges; OmegaRangeNo++) {
      KeepSpot = 0;
      if ((Omega > OmegaRange[i][0]) && (Omega < OmegaRange[i][1]) &&
          (yl > BoxSizes[i][0]) && (yl < BoxSizes[i][1]) &&
          (zl > BoxSizes[i][2]) && (zl < BoxSizes[i][3])) {
        KeepSpot = 1; break;
      }
    }
    if (KeepSpot == 1) { spots[spotnr*3 + ...] = ...; spotnr++; }

``distance`` is ``Lsd[0]`` (the caller passes it explicitly,
``NF_HEDM/src/SharedFuncsFit.c:830``), so the gate compares the NOMINAL ring
position at the FIRST distance in MICROMETRES -- not per-distance, not the
final projected pixel, no beam-centre offset -- BEFORE grain displacement and
detector tilts. All six comparisons are strict. Range ``i`` pairs with box
``i`` by index and any pair accepting is enough. A rejected spot never enters
``TheorSpots``, so it leaves BOTH the numerator and the denominator of
``CalcFracOverlap`` -- it is not counted as a miss.

The eager-path counterpart of this file is
``packages/midas_diffract/tests/test_omega_box_filter.py``; the assertions
here deliberately mirror it.

Run with (CUDA + Triton required, otherwise the whole module skips)::

    cd packages/midas_nf_fitorientation
    PYTHONPATH=../midas_diffract python -m pytest tests/test_triton_box_gate.py -v
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from midas_diffract.forward import HEDMForwardModel, HEDMGeometry

from midas_nf_fitorientation.obs_volume import ObsVolume
from midas_nf_fitorientation.soft_overlap import forward_batched_grains

try:
    from midas_nf_fitorientation.triton_kernels import HAS_TRITON, fused_hard_frac
except Exception:                                        # pragma: no cover
    HAS_TRITON = False
    fused_hard_frac = None

_CUDA = torch.cuda.is_available()

pytestmark = pytest.mark.skipif(
    not (_CUDA and HAS_TRITON),
    reason=(
        "fused_hard_frac needs a CUDA device and an importable Triton "
        f"(cuda={_CUDA}, triton={HAS_TRITON})"
    ),
)


# ---------------------------------------------------------------------------
#  Synthetic case
# ---------------------------------------------------------------------------
#
# Coarse two-distance NF geometry: 128 x 128 pixels of 2 mm, 36 frames of
# 10 deg. The packed obs volume is 2*36*128*128 bits = 144 KiB, so we can
# build it outright instead of reading SpotsInfo.bin. Gold (a = 4.08 A) at
# 0.172979 A puts all five rings on the detector at both distances.
#
# The Euler set was picked so that every predicted spot sits at least
# 0.046 px from a pixel boundary and 0.0036 frames from a frame boundary --
# ~2000x the fp32 rounding of the kernel, so the fp32/fp64 comparison below
# can never straddle an integer floor.

LSD = [1_000_000.0, 1_050_000.0]
YBC = [64.0, 64.0]
ZBC = [64.0, 64.0]
PX = 2000.0
OMEGA_START = -180.0
OMEGA_STEP = 10.0
N_FRAMES = 36
N_PIX = 128
MIN_ETA = 6.0
WAVELENGTH = 0.172979
LATTICE_A = 4.08
D = len(LSD)

HKLS_INT = torch.tensor(
    [[1, 0, 0], [1, 1, 0], [1, 1, 1], [2, 0, 0], [2, 1, 1]], dtype=torch.float64
)

EULERS = torch.tensor([
    [5.542, 1.079, 3.750],
    [4.745, 2.883, 4.283],
    [1.662, 2.804, 6.201],
    [0.749, 1.061, 3.756],
], dtype=torch.float64)

# Non-zero voxel offsets exercise the displacement path; the gate must be
# blind to them (it is evaluated before DisplacementSpots in the C).
POSITIONS = torch.tensor([
    [0.0, 0.0, 0.0],
    [300.0, -200.0, 0.0],
    [-450.0, 150.0, 0.0],
    [100.0, 400.0, 0.0],
], dtype=torch.float64)

WIDE_OMEGA = (-360.0, 360.0)
OPEN_BOX = (-1e9, 1e9, -1e9, 1e9)
SHUT_BOX = (1e8, 1e9, 1e8, 1e9)          # no ring can reach 100 mm x 100 mm
RAD2DEG = 180.0 / math.pi

TOL = 1e-6          # kernel is fp32, eager is fp64: close, not bit-identical


def _device() -> torch.device:
    return torch.device("cuda")


# ---------------------------------------------------------------------------
#  Model / kernel plumbing
# ---------------------------------------------------------------------------

def _model(box_rows=None, device=None) -> HEDMForwardModel:
    """NF-mode forward model.

    ``box_rows`` is a list of ``(o_lo, o_hi, y_lo, y_hi, z_lo, z_hi)`` rows --
    exactly the ``(NBOX, 6)`` layout ``fused_hard_frac(ome_box=...)`` takes --
    split back into the ``omega_ranges`` / ``box_sizes`` pair the eager
    geometry wants. ``None`` leaves the gate off on both sides.
    """
    device = device or _device()
    om_r = [(r[0], r[1]) for r in box_rows] if box_rows else None
    bx = [(r[2], r[3], r[4], r[5]) for r in box_rows] if box_rows else None
    geom = HEDMGeometry(
        Lsd=list(LSD), y_BC=list(YBC), z_BC=list(ZBC), px=PX,
        omega_start=OMEGA_START, omega_step=OMEGA_STEP,
        n_frames=N_FRAMES, n_pixels_y=N_PIX, n_pixels_z=N_PIX,
        min_eta=MIN_ETA, wavelength=WAVELENGTH,
        flip_y=False,               # NF convention
        multi_mode="layered",       # AllDistsFound
        omega_ranges=om_r, box_sizes=bx,
    )
    B0 = torch.eye(3, dtype=torch.float64) / LATTICE_A
    hkls_cart = HKLS_INT @ B0.T
    thetas = torch.asin(torch.linalg.norm(hkls_cart, dim=-1) * WAVELENGTH / 2.0)
    return HEDMForwardModel(
        hkls=hkls_cart, thetas=thetas, geometry=geom,
        hkls_int=HKLS_INT, device=device,
    )


def _forward(model, eulers, positions):
    """(frame_nr, valid, y_pixel, z_pixel) in the (B, K, M) fit layout."""
    dev = model.hkls.device
    return forward_batched_grains(
        model,
        eulers.to(device=dev, dtype=torch.float64),
        positions.to(device=dev, dtype=torch.float64),
    )


def _eager_frac(model, obs: ObsVolume, eulers, positions) -> torch.Tensor:
    """Per-grain hard FracOverlap from the eager path, shape (B,).

    NOTE: ``SpotDescriptors.omega`` is in RADIANS while ``omega_start`` /
    ``omega_step`` are in DEGREES -- never recompute the frame index from
    omega here, use the ready-made ``frame_nr``. ``valid`` goes in as a
    FLOAT mask (it is the denominator weight, not a boolean index).
    """
    frame_nr, valid, y_pixel, z_pixel = _forward(model, eulers, positions)
    assert valid.dtype.is_floating_point
    return obs.hard_fraction(frame_nr, y_pixel, z_pixel, valid).double().cpu()


def _triton_frac(model, obs_packed, eulers, positions, box_rows) -> torch.Tensor:
    """Per-grain hard FracOverlap from the fused kernel, shape (B,).

    Mirrors the production launch site,
    ``fit_orientation.py::_batched_neg_hard_frac_factory``.
    """
    dev = model.hkls.device
    ome_box = None
    if box_rows is not None:
        ome_box = torch.tensor(
            np.asarray(box_rows, dtype=np.float32), dtype=torch.float32,
            device=dev,
        )
    out = fused_hard_frac(
        eulers.to(device=dev, dtype=torch.float32).contiguous(),
        positions.to(device=dev, dtype=torch.float32).contiguous(),
        model.hkls.contiguous().to(torch.float32),
        model.thetas.contiguous().to(torch.float32),
        torch.tensor(LSD, device=dev, dtype=torch.float32),
        torch.tensor(YBC, device=dev, dtype=torch.float32),
        torch.tensor(ZBC, device=dev, dtype=torch.float32),
        torch.zeros(D, 9, device=dev, dtype=torch.float32),
        obs_packed,
        px=PX,
        wedge_rad=0.0,
        omega_start_deg=OMEGA_START,
        omega_step_deg=OMEGA_STEP,
        min_eta_rad=MIN_ETA * math.pi / 180.0,
        n_frames=N_FRAMES,
        n_y=N_PIX,
        n_z=N_PIX,
        has_tilts=False,
        has_wedge=False,
        ome_box=ome_box,
    )
    return out.double().cpu()


# ---------------------------------------------------------------------------
#  Nominal-ring bookkeeping (the quantity the gate actually compares)
# ---------------------------------------------------------------------------

def _ring_table(model, eulers):
    """(omega_deg, yl, zl, valid) exactly as C ``CalcSpotPosition`` computes.

    Shape ``(2B, M)``; rows ``[0:B)`` are the K=0 omega branch and rows
    ``[B:2B)`` the K=1 branch. Pass a model with the gate OFF -- otherwise
    ``valid`` already has the gate folded in.
    """
    om, eta, tt, valid = model.calc_bragg_geometry(
        model.euler2mat(eulers.to(device=model.hkls.device, dtype=torch.float64))
    )
    ring = torch.as_tensor(LSD[0], dtype=om.dtype, device=om.device) * torch.tan(tt)
    return om * RAD2DEG, -torch.sin(eta) * ring, torch.cos(eta) * ring, valid


def _windows(om_deg, yl, zl, sel, half_om=1.0, half_um=1000.0):
    """One ``(o_lo, o_hi, y_lo, y_hi, z_lo, z_hi)`` row per selected spot.

    The half-widths are ~5x smaller than the closest approach of any two
    predicted spots in this case (Chebyshev separation 19873 in
    ``(1000*omega_deg, yl_um, zl_um)``) and ~4 orders of magnitude larger
    than the fp32/fp64 disagreement, so each row admits exactly the spot it
    was built from -- which the tests then confirm against the eager path.
    """
    rows = []
    for idx in torch.nonzero(sel, as_tuple=False):
        i = tuple(int(v) for v in idx)
        rows.append((
            float(om_deg[i]) - half_om, float(om_deg[i]) + half_om,
            float(yl[i]) - half_um, float(yl[i]) + half_um,
            float(zl[i]) - half_um, float(zl[i]) + half_um,
        ))
    return rows


def _z_cut(zl_valid, frac=0.25, min_gap=100.0) -> float:
    """A z floor placed midway between two adjacent sorted ``zl`` values.

    Never a data point and never near one, so the eager (fp64) and Triton
    (fp32) evaluations of ``zl > z_cut`` can never disagree -- this test file
    is about the gate's semantics, not about fp32 tie-breaking.
    """
    s, _ = torch.sort(zl_valid.reshape(-1))
    k = min(max(int(round(frac * s.numel())) - 1, 0), s.numel() - 2)
    gap = float(s[k + 1] - s[k])
    assert gap > min_gap, f"adjacent zl gap {gap} um is too small to cut safely"
    return float(0.5 * (s[k] + s[k + 1]))


def _every_nth_valid(valid, n, offset=0):
    """Deterministic boolean mask over every ``n``-th valid spot."""
    flat = valid.reshape(-1) > 0
    idx = torch.nonzero(flat, as_tuple=True)[0][offset::n]
    out = torch.zeros_like(flat)
    out[idx] = True
    return out.reshape(valid.shape)


# ---------------------------------------------------------------------------
#  Obs volumes
# ---------------------------------------------------------------------------

def _obs_from_dense(arr: np.ndarray, device) -> ObsVolume:
    return ObsVolume.from_dense_array(arr, device=device, packed=True)


def _obs_all_ones(device) -> ObsVolume:
    arr = np.ones((D, N_FRAMES, N_PIX, N_PIX), dtype=np.uint8)
    return _obs_from_dense(arr, device)


def _obs_ones_with_holes(model, eulers, positions, hole_mask, device) -> ObsVolume:
    """All-ones obs with the d=0 bit cleared for every spot in ``hole_mask``.

    Clearing at one distance is enough: ``hard_fraction`` ANDs across
    distances. Every punched spot becomes a guaranteed miss while staying in
    the denominator, so the resulting fraction is strictly inside (0, 1).
    """
    frame_nr, valid, y_pixel, z_pixel = _forward(model, eulers, positions)
    arr = np.ones((D, N_FRAMES, N_PIX, N_PIX), dtype=np.uint8)
    f = frame_nr.long()[hole_mask].cpu().numpy()
    y = y_pixel[0].long()[hole_mask].cpu().numpy()
    z = z_pixel[0].long()[hole_mask].cpu().numpy()
    arr[0, f, y, z] = 0
    return _obs_from_dense(arr, device)


def _obs_firing_only_at(model, eulers, positions, mask, device) -> ObsVolume:
    """Obs that fires ONLY at the pixels of the spots selected by ``mask``."""
    frame_nr, valid, y_pixel, z_pixel = _forward(model, eulers, positions)
    arr = np.zeros((D, N_FRAMES, N_PIX, N_PIX), dtype=np.uint8)
    f = frame_nr.long()[mask].cpu().numpy()
    for d in range(D):
        y = y_pixel[d].long()[mask].cpu().numpy()
        z = z_pixel[d].long()[mask].cpu().numpy()
        arr[d, f, y, z] = 1
    return _obs_from_dense(arr, device)


def _distinct_pixels(model, eulers, positions, mask) -> bool:
    """True when no two selected spots share a (frame, y, z) at any distance."""
    frame_nr, valid, y_pixel, z_pixel = _forward(model, eulers, positions)
    n = int(mask.sum())
    for d in range(D):
        seen = {
            (int(a), int(b), int(c))
            for a, b, c in zip(frame_nr.long()[mask],
                               y_pixel[d].long()[mask],
                               z_pixel[d].long()[mask])
        }
        if len(seen) != n:
            return False
    return True


# ===========================================================================
#  1. Triton vs eager parity with the gate ON  (the headline test)
# ===========================================================================

class TestTritonEagerParity:
    def test_gate_on_matches_eager(self):
        dev = _device()
        m_off = _model(device=dev)
        om_deg, yl, zl, valid = _ring_table(m_off, EULERS)
        base = valid.bool()
        assert int(base.sum()) > 20, "test case must have plenty of spots"

        # A z floor that removes a real slice of the spot list.
        z_cut = _z_cut(zl[base], 0.25)
        rows = [(*WIDE_OMEGA, -1e9, 1e9, z_cut, 1e9)]
        n_kept = int((base & (zl > z_cut)).sum())
        assert 0 < n_kept < int(base.sum()), "box must exclude something"

        m_on = _model(rows, device=dev)
        # Punch holes among the spots the gate KEEPS, so the fractions sit
        # strictly inside (0, 1) -- an all-hit or all-miss case would pass
        # even with a broken denominator.
        _fn, val_on, _yp, _zp = _forward(m_on, EULERS, POSITIONS)
        holes = _every_nth_valid(val_on, 3)
        assert 0 < int(holes.sum()) < int(val_on.sum())
        obs = _obs_ones_with_holes(m_off, EULERS, POSITIONS, holes, dev)

        eager = _eager_frac(m_on, obs, EULERS, POSITIONS)
        trit = _triton_frac(m_on, obs.packed, EULERS, POSITIONS, rows)

        assert torch.all(eager > 0.0) and torch.all(eager < 1.0), (
            f"parity case is degenerate: {eager.tolist()}"
        )
        torch.testing.assert_close(trit, eager, atol=TOL, rtol=0.0)

    def test_multi_row_gate_matches_eager(self):
        """Same parity check with a two-row table and a y window as well."""
        dev = _device()
        m_off = _model(device=dev)
        om_deg, yl, zl, valid = _ring_table(m_off, EULERS)
        base = valid.bool()
        rows = [
            (-360.0, 0.0, -1e9, 0.0, -1e9, 1e9),
            (0.0, 360.0, 0.0, 1e9, -1e9, 1e9),
        ]
        m_on = _model(rows, device=dev)
        _fn, val_on, _yp, _zp = _forward(m_on, EULERS, POSITIONS)
        assert 0 < int(val_on.sum()) < int(base.sum())

        holes = _every_nth_valid(val_on, 2)
        obs = _obs_ones_with_holes(m_off, EULERS, POSITIONS, holes, dev)

        eager = _eager_frac(m_on, obs, EULERS, POSITIONS)
        trit = _triton_frac(m_on, obs.packed, EULERS, POSITIONS, rows)
        assert torch.any((eager > 0.0) & (eager < 1.0)), (
            f"parity case is degenerate: {eager.tolist()}"
        )
        torch.testing.assert_close(trit, eager, atol=TOL, rtol=0.0)


# ===========================================================================
#  2. Gate OFF control -- ome_box=None must be the pre-gate kernel
# ===========================================================================

class TestGateOffControl:
    def test_none_reproduces_ungated_eager(self):
        dev = _device()
        m_off = _model(device=dev)
        _fn, val_off, _yp, _zp = _forward(m_off, EULERS, POSITIONS)
        holes = _every_nth_valid(val_off, 3)
        obs = _obs_ones_with_holes(m_off, EULERS, POSITIONS, holes, dev)

        eager = _eager_frac(m_off, obs, EULERS, POSITIONS)
        trit = _triton_frac(m_off, obs.packed, EULERS, POSITIONS, None)
        assert torch.all(eager > 0.0) and torch.all(eager < 1.0)
        torch.testing.assert_close(trit, eager, atol=TOL, rtol=0.0)

    def test_wide_open_box_equals_gate_off(self):
        """A box wider than any ring must not change a single spot."""
        dev = _device()
        m_off = _model(device=dev)
        rows = [(*WIDE_OMEGA, *OPEN_BOX)]
        m_open = _model(rows, device=dev)
        _fn, val_off, _yp, _zp = _forward(m_off, EULERS, POSITIONS)
        obs = _obs_ones_with_holes(
            m_off, EULERS, POSITIONS, _every_nth_valid(val_off, 3), dev)
        a = _triton_frac(m_off, obs.packed, EULERS, POSITIONS, None)
        b = _triton_frac(m_open, obs.packed, EULERS, POSITIONS, rows)
        # NBOX=0 vs NBOX=1-with-everything-inside are two different compiled
        # kernels; the fractions must still agree bit-for-bit.
        torch.testing.assert_close(a, b, atol=0.0, rtol=0.0)
        assert torch.all(a > 0.0) and torch.all(a < 1.0)   # not vacuous

    def test_gate_on_differs_from_gate_off(self):
        """The control has teeth only if the gate actually moves the number."""
        dev = _device()
        m_off = _model(device=dev)
        om_deg, yl, zl, valid = _ring_table(m_off, EULERS)
        base = valid.bool()
        z_cut = _z_cut(zl[base], 0.5)
        rows = [(*WIDE_OMEGA, -1e9, 1e9, z_cut, 1e9)]
        m_on = _model(rows, device=dev)

        _fn, val_off, _yp, _zp = _forward(m_off, EULERS, POSITIONS)
        # Only the out-of-box spots miss -> the gate lifts the fraction.
        keep = val_off.bool() & (
            zl.reshape(2, EULERS.shape[0], -1).permute(1, 0, 2) > z_cut
        )
        obs = _obs_firing_only_at(m_off, EULERS, POSITIONS, keep, dev)

        off = _triton_frac(m_off, obs.packed, EULERS, POSITIONS, None)
        on = _triton_frac(m_on, obs.packed, EULERS, POSITIONS, rows)
        assert torch.any(off < on - 1e-3), (
            f"gate ON must differ from gate OFF; got {off.tolist()} vs {on.tolist()}"
        )


# ===========================================================================
#  3. Denominator semantics -- a rejected spot is NOT a miss
# ===========================================================================

class TestDenominatorSemantics:
    def test_out_of_box_spot_leaves_the_denominator(self):
        """The real-data signature of the bug: 0.949153 -> 1.000000.

        The observation fires ONLY at the in-box spots, so every out-of-box
        spot is a genuine miss as far as the bitmap is concerned. Applying
        the gate must raise the fraction to exactly 1.0 (the spot left the
        denominator), not leave it below 1.0 (the spot counted as a miss).
        """
        dev = _device()
        m_off = _model(device=dev)
        om_deg, yl, zl, valid = _ring_table(m_off, EULERS)
        B = EULERS.shape[0]
        zl_bkm = zl.reshape(2, B, -1).permute(1, 0, 2)

        _fn, val_off, _yp, _zp = _forward(m_off, EULERS, POSITIONS)
        base = val_off.bool()
        z_cut = _z_cut(zl[valid.bool()], 0.25)
        keep = base & (zl_bkm > z_cut)
        n_base = base.sum(dim=(1, 2)).cpu()
        n_keep = keep.sum(dim=(1, 2)).cpu()
        assert torch.any(n_keep < n_base) and torch.all(n_keep > 0)
        assert _distinct_pixels(m_off, EULERS, POSITIONS, base)

        obs = _obs_firing_only_at(m_off, EULERS, POSITIONS, keep, dev)
        rows = [(*WIDE_OMEGA, -1e9, 1e9, z_cut, 1e9)]
        m_on = _model(rows, device=dev)

        # Sanity: the eager gate keeps exactly the spots we selected.
        _fn, val_on, _yp, _zp = _forward(m_on, EULERS, POSITIONS)
        assert torch.equal(val_on.bool(), keep)

        on = _triton_frac(m_on, obs.packed, EULERS, POSITIONS, rows)
        off = _triton_frac(m_off, obs.packed, EULERS, POSITIONS, None)
        expect_off = (n_keep.double() / n_base.double())

        torch.testing.assert_close(
            on, torch.ones_like(on), atol=TOL, rtol=0.0,
        )
        torch.testing.assert_close(off, expect_off, atol=TOL, rtol=0.0)
        assert torch.any(off < 1.0 - 1e-6), "control must actually be < 1"
        # ... and the eager path says the same thing.
        torch.testing.assert_close(
            _eager_frac(m_on, obs, EULERS, POSITIONS), on, atol=TOL, rtol=0.0,
        )

    def test_in_box_spot_can_still_miss(self):
        """The gate must not blanket-exclude: in-box spots stay in the
        denominator and can still be counted as misses."""
        dev = _device()
        m_off = _model(device=dev)
        om_deg, yl, zl, valid = _ring_table(m_off, EULERS)
        B = EULERS.shape[0]
        zl_bkm = zl.reshape(2, B, -1).permute(1, 0, 2)
        _fn, val_off, _yp, _zp = _forward(m_off, EULERS, POSITIONS)
        base = val_off.bool()
        z_cut = _z_cut(zl[valid.bool()], 0.25)
        keep = base & (zl_bkm > z_cut)
        assert _distinct_pixels(m_off, EULERS, POSITIONS, base)

        # Blank one in-box spot per grain.
        holes = torch.zeros_like(base)
        for b in range(B):
            idx = torch.nonzero(keep[b], as_tuple=False)[0]
            holes[b, int(idx[0]), int(idx[1])] = True
        obs = _obs_ones_with_holes(m_off, EULERS, POSITIONS, holes, dev)

        rows = [(*WIDE_OMEGA, -1e9, 1e9, z_cut, 1e9)]
        m_on = _model(rows, device=dev)
        n_keep = keep.sum(dim=(1, 2)).cpu().double()
        on = _triton_frac(m_on, obs.packed, EULERS, POSITIONS, rows)
        torch.testing.assert_close(
            on, (n_keep - 1.0) / n_keep, atol=TOL, rtol=0.0,
        )
        assert torch.all(on < 1.0)


# ===========================================================================
#  4. Micrometres at Lsd[0], not pixels
# ===========================================================================

class TestUnitsAreMicrometres:
    def test_pm2048_box_kills_every_spot(self):
        """A +/-2048 box read as PIXELS would keep every spot on this 128 px
        detector; read as MICROMETRES (the C convention) it keeps none."""
        dev = _device()
        m_off = _model(device=dev)
        om_deg, yl, zl, valid = _ring_table(m_off, EULERS)
        base = valid.bool()
        # Under a pixel reading every spot passes...
        _fn, _v, y_pix, z_pix = _forward(m_off, EULERS, POSITIONS)
        assert float(y_pix.abs().max()) < 2048.0
        assert float(z_pix.abs().max()) < 2048.0
        # ...but under the correct um reading no ring comes near the window.
        assert float(yl[base].abs().min()) > 2048.0

        rows = [(*WIDE_OMEGA, -2048.0, 2048.0, -2048.0, 2048.0)]
        obs = _obs_all_ones(dev)
        m_on = _model(rows, device=dev)

        gate_off = _triton_frac(m_off, obs.packed, EULERS, POSITIONS, None)
        gate_on = _triton_frac(m_on, obs.packed, EULERS, POSITIONS, rows)
        # All-ones obs: 1.0 means "denominator non-empty", 0.0 means "empty".
        assert torch.all(gate_off == 1.0)
        assert torch.all(gate_on == 0.0)
        torch.testing.assert_close(
            _eager_frac(m_on, obs, EULERS, POSITIONS), gate_on,
            atol=TOL, rtol=0.0,
        )

    def test_gate_ignores_the_beam_centre(self):
        """The C compares the raw ring position, with no BC offset. A box
        centred on the ring in um therefore keeps the spot regardless of
        where the beam centre puts it in pixels."""
        dev = _device()
        m_off = _model(device=dev)
        om_deg, yl, zl, valid = _ring_table(m_off, EULERS[:1])
        sel = valid.bool()
        rows = _windows(om_deg, yl, zl, sel, half_om=1.0, half_um=200.0)
        m_on = _model(rows, device=dev)
        _fn, val_on, _yp, _zp = _forward(m_on, EULERS[:1], POSITIONS[:1])
        # Every spot survives its own window; adding the beam centre
        # (+64 px = +128000 um) would put none of them inside.
        assert int(val_on.sum()) == int(sel.sum())
        assert _distinct_pixels(m_off, EULERS[:1], POSITIONS[:1], val_on.bool())

        # Punch one hole so the fraction reports the denominator size rather
        # than just "non-empty".
        holes = torch.zeros_like(val_on, dtype=torch.bool)
        idx = torch.nonzero(val_on.bool()[0], as_tuple=False)[0]
        holes[0, int(idx[0]), int(idx[1])] = True
        obs = _obs_ones_with_holes(m_off, EULERS[:1], POSITIONS[:1], holes, dev)
        n = float(val_on.sum())
        got = float(_triton_frac(
            m_on, obs.packed, EULERS[:1], POSITIONS[:1], rows)[0])
        assert got == pytest.approx((n - 1.0) / n, abs=TOL), (
            f"expected {(n - 1.0) / n} from {int(n)} accepted spots, got {got}"
        )

    def test_gate_is_evaluated_before_grain_displacement(self):
        """C order: CalcDiffrSpots_Furnace gates, THEN DisplacementSpots
        moves the spot. A 3 mm voxel offset must not change the accept set,
        even against a +/-200 um window."""
        dev = _device()
        m_off = _model(device=dev)
        om_deg, yl, zl, valid = _ring_table(m_off, EULERS[:1])
        sel = valid.bool()
        rows = _windows(om_deg, yl, zl, sel, half_om=1.0, half_um=200.0)
        m_on = _model(rows, device=dev)

        far = torch.tensor([[2000.0, 3000.0, 0.0]], dtype=torch.float64)
        _fn, val_far, _yp, _zp = _forward(m_on, EULERS[:1], far)
        assert int(val_far.sum()) == int(sel.sum()), (
            "eager gate is already position-dependent -- fix that first"
        )
        assert _distinct_pixels(m_off, EULERS[:1], far, val_far.bool())

        # Punch one hole so the fraction reports the denominator size.
        holes = torch.zeros_like(val_far, dtype=torch.bool)
        idx = torch.nonzero(val_far.bool()[0], as_tuple=False)[0]
        holes[0, int(idx[0]), int(idx[1])] = True
        obs = _obs_ones_with_holes(m_off, EULERS[:1], far, holes, dev)

        n = float(val_far.sum())
        got = float(_triton_frac(m_on, obs.packed, EULERS[:1], far, rows)[0])
        assert got == pytest.approx((n - 1.0) / n, abs=TOL), (
            f"expected {(n - 1.0) / n} from {int(n)} accepted spots, got {got}"
        )


# ===========================================================================
#  5. Strict (exclusive) bounds on every edge
# ===========================================================================

def _bisect_edge(accepts, lo: float, hi: float) -> float:
    """Smallest float32 in ``[lo, hi]`` that ``accepts`` rejects.

    ``accepts`` must be monotone -- True below the kernel's own edge value,
    False at and above it. Returns the exact fp32 value at the transition,
    which is the kernel's OWN fp32 arithmetic result. That makes the
    strictness assertions below independent of how libdevice rounds ``tan``
    and of the fp32/fp64 gap.
    """
    lo32, hi32 = np.float32(lo), np.float32(hi)
    assert accepts(float(lo32)), "bisection lower bound must accept"
    assert not accepts(float(hi32)), "bisection upper bound must reject"
    while True:
        mid = np.float32((np.float64(lo32) + np.float64(hi32)) / 2.0)
        if mid <= lo32 or mid >= hi32:
            break
        if accepts(float(mid)):
            lo32 = mid
        else:
            hi32 = mid
    return float(hi32)


class TestStrictBounds:
    """All six comparisons in the C are ``>`` / ``<``, never ``>=`` / ``<=``.

    Strategy: locate one isolated spot, then use the kernel itself as an
    oracle to find the exact fp32 edge value E where the ``z_lo`` half-plane
    flips from accept to reject. Under STRICT semantics E is the kernel's own
    ``zl``, so putting E on the OPPOSITE side of the window (``z_hi = E``)
    must also reject. Under ``>=`` semantics E would be one ULP above ``zl``
    and ``z_hi = E`` would accept -- so this catches a non-strict comparison
    on either edge without needing to predict ``zl`` bit-for-bit.
    """

    @staticmethod
    def _isolated_spot(dev):
        m_off = _model(device=dev)
        om_deg, yl, zl, valid = _ring_table(m_off, EULERS[:1])
        sel = valid.bool()
        # Deterministic pick: the valid spot with the largest zl.
        flat = torch.nonzero(sel, as_tuple=False)
        i = max((tuple(int(v) for v in r) for r in flat), key=lambda t: float(zl[t]))
        o_i, y_i, z_i = float(om_deg[i]), float(yl[i]), float(zl[i])
        window = (o_i - 1.0, o_i + 1.0, y_i - 1000.0, y_i + 1000.0)

        # The window must admit exactly this one spot (checked on eager).
        rows = [(*window, z_i - 1000.0, z_i + 1000.0)]
        m_on = _model(rows, device=dev)
        _fn, val_on, _yp, _zp = _forward(m_on, EULERS[:1], POSITIONS[:1])
        assert int(val_on.sum()) == 1, "isolation window is not isolating"
        return m_off, window, o_i, y_i, z_i

    @staticmethod
    def _accept_fn(m_off, obs_ones, dev):
        def accepts(row):
            f = _triton_frac(
                m_off, obs_ones.packed, EULERS[:1], POSITIONS[:1], [row],
            )
            return float(f[0]) == 1.0
        return accepts

    def test_z_edges_are_exclusive(self):
        dev = _device()
        m_off, window, o_i, y_i, z_i = self._isolated_spot(dev)
        obs = _obs_all_ones(dev)
        accepts = self._accept_fn(m_off, obs, dev)

        edge = _bisect_edge(
            lambda zlo: accepts((*window, zlo, z_i + 1000.0)),
            z_i - 1000.0, z_i + 1000.0,
        )
        # The oracle found the kernel's own zl (fp32), not something else.
        assert abs(edge - z_i) < 1.0, f"edge {edge} vs fp64 zl {z_i}"

        # zmin exactly on the spot -> rejected (this is the bisection result).
        assert not accepts((*window, edge, z_i + 1000.0))
        # zmax exactly on the spot -> rejected too. THIS is the strictness
        # assertion: with `<=` (or with `>=` on zmin shifting `edge` up by one
        # ULP) the spot would come back.
        assert not accepts((*window, z_i - 1000.0, edge))
        # One ULP of slack on either side brings it back -> not vacuous.
        assert accepts((*window, np.nextafter(np.float32(edge), np.float32(-np.inf)).item(),
                        z_i + 1000.0))
        assert accepts((*window, z_i - 1000.0,
                        np.nextafter(np.float32(edge), np.float32(np.inf)).item()))

    def test_omega_edges_are_exclusive(self):
        dev = _device()
        m_off, window, o_i, y_i, z_i = self._isolated_spot(dev)
        obs = _obs_all_ones(dev)
        accepts = self._accept_fn(m_off, obs, dev)
        zbox = (y_i - 1000.0, y_i + 1000.0, z_i - 1000.0, z_i + 1000.0)

        edge = _bisect_edge(
            lambda olo: accepts((olo, o_i + 1.0, *zbox)),
            o_i - 1.0, o_i + 1.0,
        )
        assert abs(edge - o_i) < 1e-2, f"edge {edge} vs fp64 omega {o_i}"

        assert not accepts((edge, o_i + 1.0, *zbox))
        assert not accepts((o_i - 1.0, edge, *zbox))
        assert accepts((np.nextafter(np.float32(edge), np.float32(-np.inf)).item(),
                        o_i + 1.0, *zbox))
        assert accepts((o_i - 1.0,
                        np.nextafter(np.float32(edge), np.float32(np.inf)).item(),
                        *zbox))

    def test_degenerate_window_accepts_nothing(self):
        """``z_lo == z_hi`` is an EMPTY open interval under strict bounds."""
        dev = _device()
        m_off, window, o_i, y_i, z_i = self._isolated_spot(dev)
        obs = _obs_all_ones(dev)
        accepts = self._accept_fn(m_off, obs, dev)
        assert not accepts((*window, z_i, z_i))
        assert not accepts((o_i, o_i, y_i - 1000.0, y_i + 1000.0,
                            z_i - 1000.0, z_i + 1000.0))


# ===========================================================================
#  6. Paired by index, any pair accepts
# ===========================================================================

class TestPairing:
    """Range i pairs with box i. A spot needs BOTH halves of the SAME pair.

    A global OR over all omega ranges crossed with all boxes -- the obvious
    way to get this wrong -- would accept every spot in these fixtures, so
    the swap below separates the two implementations cleanly.
    """

    @staticmethod
    def _split(dev):
        m_off = _model(device=dev)
        om_deg, yl, zl, valid = _ring_table(m_off, EULERS)
        B = EULERS.shape[0]
        om_bkm = om_deg.reshape(2, B, -1).permute(1, 0, 2)
        _fn, val_off, _yp, _zp = _forward(m_off, EULERS, POSITIONS)
        base = val_off.bool()
        neg = base & (om_bkm < 0)
        pos = base & (om_bkm > 0)
        assert int(neg.sum()) > 0 and int(pos.sum()) > 0
        assert torch.equal(neg | pos, base)
        return m_off, base, neg, pos

    def test_any_pair_accepts_and_pairing_is_positional(self):
        dev = _device()
        m_off, base, neg, pos = self._split(dev)
        # Obs fires only where omega < 0.
        obs = _obs_firing_only_at(m_off, EULERS, POSITIONS, neg, dev)
        assert _distinct_pixels(m_off, EULERS, POSITIONS, base)

        # Pair 0 = (omega < 0, open box); pair 1 = (omega > 0, shut box).
        rows_a = [(-360.0, 0.0, *OPEN_BOX), (0.0, 360.0, *SHUT_BOX)]
        rows_b = [(-360.0, 0.0, *SHUT_BOX), (0.0, 360.0, *OPEN_BOX)]
        m_a, m_b = _model(rows_a, device=dev), _model(rows_b, device=dev)

        # The eager gate is the reference for WHICH spots survive.
        _fn, val_a, _yp, _zp = _forward(m_a, EULERS, POSITIONS)
        _fn, val_b, _yp, _zp = _forward(m_b, EULERS, POSITIONS)
        assert torch.equal(val_a.bool(), neg)
        assert torch.equal(val_b.bool(), pos)

        frac_a = _triton_frac(m_a, obs.packed, EULERS, POSITIONS, rows_a)
        frac_b = _triton_frac(m_b, obs.packed, EULERS, POSITIONS, rows_b)
        # Pairing a: only the firing spots are in the denominator -> 1.0.
        torch.testing.assert_close(
            frac_a, torch.ones_like(frac_a), atol=TOL, rtol=0.0)
        # Swapping the boxes swaps which half survives -> 0.0, and it is a
        # real 0/n, not an empty 0/0 (checked with an all-ones obs below).
        torch.testing.assert_close(
            frac_b, torch.zeros_like(frac_b), atol=TOL, rtol=0.0)
        ones = _obs_all_ones(dev)
        assert torch.all(
            _triton_frac(m_b, ones.packed, EULERS, POSITIONS, rows_b) == 1.0
        )
        # Both agree with eager.
        torch.testing.assert_close(
            _eager_frac(m_a, obs, EULERS, POSITIONS), frac_a, atol=TOL, rtol=0.0)
        torch.testing.assert_close(
            _eager_frac(m_b, obs, EULERS, POSITIONS), frac_b, atol=TOL, rtol=0.0)

        # A global OR of all omegas with all boxes would keep everything and
        # land strictly between the two -- assert we are not doing that.
        rows_or = [(-360.0, 360.0, *OPEN_BOX)]
        frac_or = _triton_frac(
            _model(rows_or, device=dev), obs.packed, EULERS, POSITIONS, rows_or)
        assert torch.all(frac_or > 0.0) and torch.all(frac_or < 1.0), (
            f"global-OR control is degenerate: {frac_or.tolist()}"
        )

    def test_both_pairs_open_is_the_union(self):
        dev = _device()
        m_off, base, neg, pos = self._split(dev)
        rows = [(-360.0, 0.0, *OPEN_BOX), (0.0, 360.0, *OPEN_BOX)]
        m_on = _model(rows, device=dev)
        _fn, val_on, _yp, _zp = _forward(m_on, EULERS, POSITIONS)
        assert torch.equal(val_on.bool(), base)
        obs = _obs_all_ones(dev)
        torch.testing.assert_close(
            _triton_frac(m_on, obs.packed, EULERS, POSITIONS, rows),
            _triton_frac(m_off, obs.packed, EULERS, POSITIONS, None),
            atol=0.0, rtol=0.0,
        )


# ===========================================================================
#  7. Both omega solutions -- the kernel inlines K=0 and K=1 separately
# ===========================================================================

class TestBothOmegaBranches:
    """``fused_hard_frac_kernel`` has no K loop: the K=0 and K=1 bodies are
    two copy-pasted blocks, each with its own copy of the gate. A gate edited
    into only one of them would still pass a test that mixes the branches, so
    these two cases isolate them.
    """

    @staticmethod
    def _branch_masks(dev):
        m_off = _model(device=dev)
        om_deg, yl, zl, valid = _ring_table(m_off, EULERS[:1])
        sel = valid.bool()                       # (2, M): row 0 = K0, row 1 = K1
        k0 = sel.clone(); k0[1] = False
        k1 = sel.clone(); k1[0] = False
        assert int(k0.sum()) > 0 and int(k1.sum()) > 0
        rows_k0 = _windows(om_deg, yl, zl, k0)
        rows_k1 = _windows(om_deg, yl, zl, k1)

        # (B, K, M) view of the same masks for the obs builder.
        k0_bkm = k0.reshape(2, 1, -1).permute(1, 0, 2)
        k1_bkm = k1.reshape(2, 1, -1).permute(1, 0, 2)
        return m_off, rows_k0, rows_k1, k0_bkm, k1_bkm, sel

    def test_gate_is_live_in_both_branches(self):
        dev = _device()
        m_off, rows_k0, rows_k1, k0, k1, sel = self._branch_masks(dev)
        eul, pos = EULERS[:1], POSITIONS[:1]

        # Each window set must admit exactly its own branch (eager reference).
        m_k0, m_k1 = _model(rows_k0, device=dev), _model(rows_k1, device=dev)
        _fn, v0, _yp, _zp = _forward(m_k0, eul, pos)
        _fn, v1, _yp, _zp = _forward(m_k1, eul, pos)
        assert torch.equal(v0.bool(), k0), "K=0 windows leaked into K=1"
        assert torch.equal(v1.bool(), k1), "K=1 windows leaked into K=0"

        _fn, val_off, _yp, _zp = _forward(m_off, eul, pos)
        assert _distinct_pixels(m_off, eul, pos, val_off.bool())
        # Obs fires ONLY at the K=0 spots.
        obs = _obs_firing_only_at(m_off, eul, pos, k0, dev)

        # K=0 windows: denominator is exactly the firing spots -> 1.0.
        # If the K=1 branch had no gate, its spots would stay in the
        # denominator and miss, dragging this below 1.
        f0 = _triton_frac(m_k0, obs.packed, eul, pos, rows_k0)
        assert float(f0[0]) == pytest.approx(1.0, abs=TOL), (
            "K=1 branch is not applying the gate"
        )

        # K=1 windows: denominator is exactly the non-firing spots -> 0.0.
        # If the K=0 branch had no gate, its spots would stay in and HIT,
        # pushing this above 0.
        f1 = _triton_frac(m_k1, obs.packed, eul, pos, rows_k1)
        assert float(f1[0]) == pytest.approx(0.0, abs=TOL), (
            "K=0 branch is not applying the gate"
        )
        # ...and 0.0 here is a real 0/n, not an empty 0/0.
        ones = _obs_all_ones(dev)
        assert float(_triton_frac(m_k1, ones.packed, eul, pos, rows_k1)[0]) == 1.0

        # Both match eager.
        torch.testing.assert_close(
            _eager_frac(m_k0, obs, eul, pos), f0, atol=TOL, rtol=0.0)
        torch.testing.assert_close(
            _eager_frac(m_k1, obs, eul, pos), f1, atol=TOL, rtol=0.0)

    def test_mixed_branch_selection_matches_eager(self):
        """A table that takes some spots from each branch."""
        dev = _device()
        m_off, rows_k0, rows_k1, k0, k1, sel = self._branch_masks(dev)
        eul, pos = EULERS[:1], POSITIONS[:1]
        rows = rows_k0[: max(1, len(rows_k0) // 2)] + rows_k1[: max(1, len(rows_k1) // 2)]
        m_on = _model(rows, device=dev)
        _fn, val_on, _yp, _zp = _forward(m_on, eul, pos)
        assert int(val_on[:, 0].sum()) > 0 and int(val_on[:, 1].sum()) > 0, (
            "mixed case must draw from both branches"
        )
        obs = _obs_firing_only_at(m_off, eul, pos, k0, dev)
        torch.testing.assert_close(
            _triton_frac(m_on, obs.packed, eul, pos, rows),
            _eager_frac(m_on, obs, eul, pos),
            atol=TOL, rtol=0.0,
        )


# ===========================================================================
#  8. Arity validation
# ===========================================================================

class TestValidation:
    @pytest.mark.parametrize("shape", [(1, 4), (2, 5), (3, 7), (1, 2)])
    def test_wrong_second_dimension_raises(self, shape):
        dev = _device()
        m = _model(device=dev)
        obs = _obs_all_ones(dev)
        bad = torch.zeros(*shape, dtype=torch.float32, device=dev)
        with pytest.raises(ValueError, match=r"ome_box must be \(NBOX, 6\)"):
            _call_with_raw_box(m, obs.packed, bad)

    def test_wrong_rank_raises(self):
        dev = _device()
        m = _model(device=dev)
        obs = _obs_all_ones(dev)
        for bad in (
            torch.zeros(6, dtype=torch.float32, device=dev),
            torch.zeros(1, 1, 6, dtype=torch.float32, device=dev),
        ):
            with pytest.raises(ValueError, match=r"ome_box must be \(NBOX, 6\)"):
                _call_with_raw_box(m, obs.packed, bad)

    def test_correct_shape_is_accepted(self):
        dev = _device()
        m = _model(device=dev)
        obs = _obs_all_ones(dev)
        good = torch.tensor(
            [[*WIDE_OMEGA, *OPEN_BOX]], dtype=torch.float32, device=dev)
        out = _call_with_raw_box(m, obs.packed, good)
        assert out.shape == (EULERS.shape[0],)


def _call_with_raw_box(model, obs_packed, ome_box):
    """``fused_hard_frac`` with ``ome_box`` passed through untouched."""
    dev = model.hkls.device
    return fused_hard_frac(
        EULERS.to(device=dev, dtype=torch.float32).contiguous(),
        POSITIONS.to(device=dev, dtype=torch.float32).contiguous(),
        model.hkls.contiguous().to(torch.float32),
        model.thetas.contiguous().to(torch.float32),
        torch.tensor(LSD, device=dev, dtype=torch.float32),
        torch.tensor(YBC, device=dev, dtype=torch.float32),
        torch.tensor(ZBC, device=dev, dtype=torch.float32),
        torch.zeros(D, 9, device=dev, dtype=torch.float32),
        obs_packed,
        px=PX, wedge_rad=0.0,
        omega_start_deg=OMEGA_START, omega_step_deg=OMEGA_STEP,
        min_eta_rad=MIN_ETA * math.pi / 180.0,
        n_frames=N_FRAMES, n_y=N_PIX, n_z=N_PIX,
        has_tilts=False, has_wedge=False,
        ome_box=ome_box,
    )
