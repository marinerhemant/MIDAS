"""The residual correction map must reach the pixel geometry.

``IntegrationSpec`` declared ``ResidualCorrectionMap`` and the compat
converters shuttled it to and from v1, but nothing in the v2 binning path ever
read it. v1 applies it (``midas_integrate.detector_mapper``), so an
integration through v2 silently discarded a correction the calibration had
already measured -- and the discrepancy is a sub-pixel radial shift, which is
invisible in intensity and directly wrong in peak position.

These tests pin the wiring, the orientation, and the two failure modes that
must be loud rather than silent.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from midas_integrate_v2 import IntegrationSpec
from midas_integrate_v2.forward.pixels import eval_pixel_REta, _load_residual_map


NY = NZ = 48


def _spec(**kw) -> IntegrationSpec:
    # RhoD defaults to 0.0 and the distortion polynomial divides by it, so a
    # spec built without one yields NaN for every radius -- with or without a
    # residual map. Set it from the detector, as a real caller would.
    rho_d = float(np.hypot(NY / 2, NZ / 2) * 200.0)
    base = dict(NrPixelsY=NY, NrPixelsZ=NZ, pxY=200.0, pxZ=200.0, RhoD=rho_d,
                Lsd=torch.tensor(1.0e6, dtype=torch.float64),
                BC_y=torch.tensor(NY / 2.0, dtype=torch.float64),
                BC_z=torch.tensor(NZ / 2.0, dtype=torch.float64),
                RMin=1.0, RMax=20.0, RBinSize=1.0,
                EtaMin=-180.0, EtaMax=180.0, EtaBinSize=10.0)
    base.update(kw)
    return IntegrationSpec(**base)


def test_no_map_configured_changes_nothing():
    """The common case must be byte-for-byte what it was before."""
    R, _ = eval_pixel_REta(_spec())
    assert torch.isfinite(R).all()
    assert _load_residual_map(_spec()) is None


def test_a_constant_map_shifts_every_radius_by_that_amount(tmp_path):
    """ΔR is in pixels of radius and adds to R. A constant map is the
    cleanest possible check that it is applied, and applied with the right
    sign."""
    R0, _ = eval_pixel_REta(_spec())

    p = tmp_path / "resid.bin"
    np.full(NY * NZ, 0.25, dtype=np.float64).tofile(p)
    R1, _ = eval_pixel_REta(_spec(ResidualCorrectionMap=str(p)))

    d = (R1 - R0).numpy()
    assert np.allclose(d, 0.25, atol=1e-9), (
        f"expected every radius shifted by +0.25 px, got "
        f"{d.min():.4f}..{d.max():.4f}")


def test_map_orientation_is_z_outer_y_inner(tmp_path):
    """v1 C stores the map row-major as ``map[z * Ny + y]``, i.e. [Nz, Ny].

    Reshaping the other way transposes the correction, which no downstream
    check would catch -- the radii merely become wrong. Use a map that is a
    function of z alone, so a transpose is unambiguous.
    """
    m = np.zeros((NZ, NY), dtype=np.float64)
    m[NZ // 2:, :] = 1.0                       # bottom half only
    p = tmp_path / "half.bin"
    m.reshape(-1).tofile(p)

    R0, _ = eval_pixel_REta(_spec())
    R1, _ = eval_pixel_REta(_spec(ResidualCorrectionMap=str(p)))
    d = (R1 - R0).numpy()                       # shape (NZ, NY)

    # Rows below the split must be shifted, columns must not be.
    assert d[NZ // 2 + 2, :].mean() > 0.9, "the shifted half is not where it should be"
    assert d[2, :].mean() < 0.1, "the unshifted half is not where it should be"
    assert abs(d[:, 2].mean() - d[:, NY - 3].mean()) < 0.05, (
        "the correction varies along y; the map was transposed")


def test_missing_file_raises_rather_than_silently_skipping(tmp_path):
    """A configured-but-unreadable map means the caller believes a correction
    is applied that is not. That must not be a warning."""
    with pytest.raises(FileNotFoundError, match="cannot be read"):
        eval_pixel_REta(_spec(ResidualCorrectionMap=str(tmp_path / "nope.bin")))


def test_wrong_sized_map_raises(tmp_path):
    """A map from a different detector is a real mistake and is detectable."""
    p = tmp_path / "wrong.bin"
    np.zeros(NY * NZ + 7, dtype=np.float64).tofile(p)
    with pytest.raises(ValueError, match="different detector"):
        eval_pixel_REta(_spec(ResidualCorrectionMap=str(p)))


def test_map_is_cached_not_reread(tmp_path):
    """66 MB at a real detector size; re-reading per geometry build is the
    difference between free and not."""
    p = tmp_path / "resid.bin"
    np.full(NY * NZ, 0.1, dtype=np.float64).tofile(p)
    s = _spec(ResidualCorrectionMap=str(p))
    a = _load_residual_map(s)
    b = _load_residual_map(_spec(ResidualCorrectionMap=str(p)))
    assert a is b, "the same map on disk should be loaded once"
