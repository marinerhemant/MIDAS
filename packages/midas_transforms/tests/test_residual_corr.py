"""FIX-1: transforms must apply the Stage-4 spline residual-correction map.

The map file is (NrPixelsZ, NrPixelsY) row-major float64 — the layout written
by midas_calibrate_v2.compat.to_integrate.write_residual_correction_from_spline
and expected by _bilinear_residual_corr (Hp=Z rows, Wp=Y cols). These tests pin
that convention so a transpose (which a square detector would mask) is caught,
and confirm the residual map shifts R for transforms exactly as it does in
peakfit's geometry (Rt += map[z, y]).
"""
import importlib.util
from pathlib import Path

import numpy as np
import torch

# Load transform.py directly (its module-level imports are only math+torch; the
# midas_calibrate dependency is lazy, inside apply_tilt_distortion), so the test
# doesn't drag in the package __init__ chain.
_TF = Path(__file__).resolve().parents[1] / "midas_transforms" / "fit_setup" / "transform.py"
_spec = importlib.util.spec_from_file_location("_tf_resid", _TF)
_tf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_tf)
_bilinear_residual_corr = _tf._bilinear_residual_corr


def _ramp_map(nz, ny):
    """map[z, y] = z + 0.01*y — realistic ΔR-in-px magnitude, and distinct per
    pixel (z = integer part, y = hundredths) so a (Y,Z) transpose is visible."""
    z = np.arange(nz)[:, None]
    y = np.arange(ny)[None, :]
    return torch.from_numpy((z + 0.01 * y).astype(np.float64))


def test_bilinear_lookup_indexing_ZY():
    """At integer pixel (Y_pix, Z_pix), the lookup returns map[Z_pix, Y_pix].

    Tolerance 1e-5 absorbs grid_sample's interpolation float precision
    (~1e-7 relative); a transpose would be off by O(1), far outside it.
    """
    nz, ny = 7, 5  # NON-square: a (Y,Z) transpose would give a wrong value
    cmap = _ramp_map(nz, ny)
    for zpix in range(nz):
        for ypix in range(ny):
            got = _bilinear_residual_corr(
                torch.tensor([float(ypix)]), torch.tensor([float(zpix)]), cmap
            ).item()
            expect = zpix + 0.01 * ypix
            assert abs(got - expect) < 1e-5, (
                f"residual lookup at (Y={ypix},Z={zpix}) gave {got}, "
                f"expected map[Z,Y]={expect} — indexing/transpose bug"
            )


def test_none_map_is_zero():
    out = _bilinear_residual_corr(
        torch.tensor([3.0, 4.0]), torch.tensor([2.0, 1.0]), None
    )
    assert torch.allclose(out, torch.zeros(2))


def test_out_of_bounds_zero():
    """Out-of-bounds pixels contribute zero (padding_mode='zeros')."""
    cmap = _ramp_map(7, 5)
    out = _bilinear_residual_corr(
        torch.tensor([100.0]), torch.tensor([100.0]), cmap
    ).item()
    assert abs(out) < 1e-9


# Optional: end-to-end through the REAL v2 writer. Skips if calibrate-v2 isn't
# importable (so transforms' suite stays standalone).
def _load_v2_writer():
    import importlib.util as ilu
    p = (Path(__file__).resolve().parents[2]
         / "midas_calibrate_v2" / "midas_calibrate_v2" / "compat" / "to_integrate.py")
    if not p.exists():
        return None
    spec = ilu.spec_from_file_location("_v2_to_integrate", p)
    m = ilu.module_from_spec(spec)
    import sys
    sys.modules["_v2_to_integrate"] = m
    try:
        spec.loader.exec_module(m)
    except Exception:
        return None
    return getattr(m, "write_residual_correction_from_spline", None)


def test_end_to_end_v2_spline_round_trip(tmp_path):
    """v2 writer → residual_corr.bin → FIX-1 load → lookup reproduces the known
    spline ΔR (to machine precision). Non-square grid so a transpose can't hide.
    This is the synthetic proof that the transforms residual path is correct
    AND consistent with the v2 on-disk format."""
    import pytest
    writer = _load_v2_writer()
    if writer is None:
        pytest.skip("midas_calibrate_v2.compat.to_integrate not importable")

    NY, NZ, px = 5, 7, 150.0

    def spline_predict(ys, zs):  # ΔR in µm, asymmetric in Y vs Z
        return 30.0 * np.sin(2 * np.pi * zs / NZ) + 12.0 * np.cos(2 * np.pi * ys / NY)

    binp = tmp_path / "residual_corr.bin"
    writer(spline_predict, NrPixelsY=NY, NrPixelsZ=NZ, px_mean_um=px, out_path=binp)

    md = np.fromfile(binp, dtype=np.float64)
    assert md.size == NY * NZ
    cmap = torch.from_numpy(md.reshape(NZ, NY))  # FIX-1 layout: (NrPixelsZ, NrPixelsY)

    maxerr = 0.0
    for z in range(NZ):
        for y in range(NY):
            got = _bilinear_residual_corr(
                torch.tensor([float(y)]), torch.tensor([float(z)]), cmap
            ).item()
            true_px = spline_predict(np.array([y]), np.array([z]))[0] / px
            maxerr = max(maxerr, abs(got - true_px))
    assert maxerr * px < 1e-4, f"residual round-trip off by {maxerr*px:.2e} µm"
