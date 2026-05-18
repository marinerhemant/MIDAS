"""integrate_v2 hex-lattice (PIXIRAD) support.

Coverage:
- IntegrationSpec.lattice='hex_offset_y' + Apothem flows through
  pixel_to_REta_from_spec and emits hex-centroid (R, η).
- effective_pxYZ() derives pxY=2a, pxZ=a√3.
- PolygonBinGeometry rejects hex specs with a clear error.
- SubpixelBinGeometry works on hex specs (centroid mode, pixel_shape='rect').
- SubpixelBinGeometry with pixel_shape='hexagon' produces the same K² sample
  count and confines samples to the hex unit cell.
- spec_from_v1_paramstest reads PixelLattice / Apothem / LatticeOrientation.
"""
from __future__ import annotations

import math
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from pathlib import Path

import numpy as np
import pytest
import torch


def _hex_spec(NY=24, NZ=24, apothem=30.0):
    from midas_integrate_v2 import spec_from_v1_params
    from midas_integrate.params import IntegrationParams
    p = IntegrationParams(
        NrPixelsY=NY, NrPixelsZ=NZ,
        pxY=2.0 * apothem, pxZ=apothem * math.sqrt(3.0),
        Lsd=1_000_000.0,
        BC_y=NY / 2.0, BC_z=NZ / 2.0, RhoD=float(NY),
        RMin=1.0, RMax=12.0, RBinSize=1.0,
        EtaMin=-180.0, EtaMax=180.0, EtaBinSize=60.0,
    )
    s = spec_from_v1_params(p, requires_grad=False)
    s.lattice = "hex_offset_y"
    s.Apothem = torch.tensor(apothem, dtype=torch.float64)
    s.LatticeOrientation = torch.tensor(0.0, dtype=torch.float64)
    return s


def test_effective_pxYZ_hex():
    """effective_pxYZ() derives pxY=2a, pxZ=a√3 for hex spec."""
    s = _hex_spec(apothem=30.0)
    pY, pZ = s.effective_pxYZ()
    assert math.isclose(pY, 60.0, rel_tol=1e-12)
    assert math.isclose(pZ, 30.0 * math.sqrt(3.0), rel_tol=1e-12)


def test_effective_pxYZ_cartesian_passthrough():
    """Cartesian spec returns stored pxY/pxZ unchanged."""
    from midas_integrate_v2 import spec_from_v1_params
    from midas_integrate.params import IntegrationParams
    p = IntegrationParams(
        NrPixelsY=24, NrPixelsZ=24,
        pxY=200.0, pxZ=180.0, Lsd=1_000_000.0,
        BC_y=12.0, BC_z=12.0, RhoD=24.0,
        RMin=1.0, RMax=12.0, RBinSize=1.0,
        EtaMin=-180.0, EtaMax=180.0, EtaBinSize=60.0,
    )
    s = spec_from_v1_params(p, requires_grad=False)
    assert s.lattice == "cartesian"
    pY, pZ = s.effective_pxYZ()
    assert pY == 200.0 and pZ == 180.0


def test_pixel_to_REta_from_spec_hex_centroids():
    """eval_pixel_REta on a hex spec produces hex-row-offset centroids
    that differ from the cartesian counterpart."""
    from midas_integrate_v2 import eval_pixel_REta
    from midas_integrate_v2.compat import spec_from_v1_params
    from midas_integrate.params import IntegrationParams

    apothem = 30.0
    cart = IntegrationParams(
        NrPixelsY=8, NrPixelsZ=8,
        pxY=2 * apothem, pxZ=apothem * math.sqrt(3.0),
        Lsd=500_000.0,
        BC_y=4.0, BC_z=4.0, RhoD=8.0,
        RMin=1.0, RMax=4.0, RBinSize=1.0,
        EtaMin=-180.0, EtaMax=180.0, EtaBinSize=60.0,
    )
    s_cart = spec_from_v1_params(cart, requires_grad=False)
    s_hex = spec_from_v1_params(cart, requires_grad=False)
    s_hex.lattice = "hex_offset_y"
    s_hex.Apothem = torch.tensor(apothem, dtype=torch.float64)
    s_hex.LatticeOrientation = torch.tensor(0.0, dtype=torch.float64)

    R_cart, eta_cart = eval_pixel_REta(s_cart)
    R_hex, eta_hex = eval_pixel_REta(s_hex)
    # Row 0 (even) and row 1 (odd) cartesian eta are the same column-wise,
    # but hex has the +a Y-shift in row 1 → different.
    diff = (R_cart - R_hex).abs()
    assert float(diff.max()) > 1e-9


def test_polygon_rejects_hex_spec():
    """PolygonBinGeometry.from_spec must refuse hex specs cleanly."""
    from midas_integrate_v2.binning.polygon import PolygonBinGeometry

    s = _hex_spec()
    with pytest.raises(NotImplementedError, match="hex"):
        PolygonBinGeometry.from_spec(s)


def test_subpixel_works_on_hex_spec_centroid_mode():
    """SubpixelBinGeometry.from_spec works on a hex spec with default
    pixel_shape='rect' (centroid-only mode, same K² samples)."""
    from midas_integrate_v2.binning.subpixel import SubpixelBinGeometry

    s = _hex_spec(NY=16, NZ=16)
    geom = SubpixelBinGeometry.from_spec(s, K=2)
    assert geom.flat_bin.shape[0] == 4   # K² samples
    assert geom.n_pixels_y == 16 and geom.n_pixels_z == 16


def test_subpixel_pixel_shape_hexagon_opt_in():
    """pixel_shape='hexagon' produces the same K² samples but confines
    them to the hex unit cell (vertices at (±0.5, ±1/3) and (0, ±2/3))."""
    from midas_integrate_v2.binning.subpixel import (
        SubpixelBinGeometry, _hex_cell_subpixel_offsets,
    )

    K = 4
    offs = _hex_cell_subpixel_offsets(K, dtype=torch.float64, device=torch.device("cpu"))
    assert len(offs) == K * K
    for (dy, dz) in offs:
        y = float(dy); z = float(dz)
        assert abs(y) <= 0.5 + 1e-12, f"sample Y outside hex bbox: {y}"
        assert abs(z) <= 2.0 / 3.0 + 1e-12, f"sample Z outside hex bbox: {z}"
        # Inside-hex test: |z| ≤ 2/3 - (2/3)|y|
        assert abs(z) <= 2.0 / 3.0 - (2.0 / 3.0) * abs(y) + 1e-12, \
            f"sample ({y}, {z}) outside hex cell"

    # full from_spec call with hexagon shape
    s = _hex_spec(NY=12, NZ=12)
    geom = SubpixelBinGeometry.from_spec(s, K=K, pixel_shape="hexagon")
    assert geom.flat_bin.shape[0] == K * K


def test_subpixel_hexagon_requires_hex_spec():
    """pixel_shape='hexagon' on a cartesian spec must raise."""
    from midas_integrate_v2 import spec_from_v1_params
    from midas_integrate_v2.binning.subpixel import SubpixelBinGeometry
    from midas_integrate.params import IntegrationParams

    p = IntegrationParams(
        NrPixelsY=12, NrPixelsZ=12, pxY=200.0, pxZ=200.0, Lsd=1_000_000.0,
        BC_y=6.0, BC_z=6.0, RhoD=12.0,
        RMin=1.0, RMax=6.0, RBinSize=1.0,
        EtaMin=-180.0, EtaMax=180.0, EtaBinSize=60.0,
    )
    s = spec_from_v1_params(p)   # cartesian
    with pytest.raises(ValueError, match="hex_offset_y"):
        SubpixelBinGeometry.from_spec(s, K=2, pixel_shape="hexagon")


def test_spec_from_paramstest_reads_hex_keys(tmp_path):
    """spec_from_v1_paramstest reads PixelLattice / Apothem /
    LatticeOrientation if present, defaults to cartesian otherwise."""
    from midas_integrate_v2.compat import spec_from_v1_paramstest

    text = """
NrPixelsY 512
NrPixelsZ 476
pxY 60.0
pxZ 51.96
Lsd 500000.0
BC 256.0 238.0
Wavelength 0.5
RhoD 250.0
RMin 1.0
RMax 200.0
RBinSize 1.0
EtaMin -180.0
EtaMax 180.0
EtaBinSize 5.0
PixelLattice hex_offset_y
Apothem 30.0
LatticeOrientation 0.5
"""
    p = tmp_path / "paramstest.txt"
    p.write_text(text)
    spec = spec_from_v1_paramstest(p)
    assert spec.lattice == "hex_offset_y"
    assert math.isclose(float(spec.Apothem), 30.0, rel_tol=1e-12)
    assert math.isclose(float(spec.LatticeOrientation), 0.5, rel_tol=1e-12)
    # effective pxY/pxZ derived from apothem
    pY, pZ = spec.effective_pxYZ()
    assert math.isclose(pY, 60.0, rel_tol=1e-12)
    assert math.isclose(pZ, 30.0 * math.sqrt(3.0), rel_tol=1e-12)


def test_spec_validate_hex_requires_apothem():
    """Hex spec with Apothem=0 must fail validate()."""
    s = _hex_spec()
    s.Apothem = torch.tensor(0.0, dtype=torch.float64)
    with pytest.raises(ValueError, match="Apothem"):
        s.validate()


def test_bootstrap_estimate_initial_spec_hex():
    """estimate_initial_spec with lattice='hex_offset_y' returns a hex
    spec; pxY/pxZ are derived from apothem; Apothem field is set."""
    from midas_integrate_v2.bootstrap import estimate_initial_spec

    NY, NZ = 64, 60
    img = np.zeros((NZ, NY), dtype=np.float64)
    # Plant a bright ring at R ≈ 20 px so estimate_BC has signal
    yy, zz = np.meshgrid(np.arange(NY), np.arange(NZ), indexing="xy")
    R = np.sqrt((yy - NY / 2.0) ** 2 + (zz - NZ / 2.0) ** 2)
    img[(R > 18) & (R < 22)] = 100.0

    spec = estimate_initial_spec(
        img, NrPixelsY=NY, NrPixelsZ=NZ,
        pxY_um=999.0,            # should be overridden
        Lsd_um=500_000.0, Wavelength_A=0.5,
        lattice="hex_offset_y", apothem_um=30.0,
    )
    assert spec.lattice == "hex_offset_y"
    assert math.isclose(float(spec.Apothem), 30.0, rel_tol=1e-12)
    pY, pZ = spec.effective_pxYZ()
    assert math.isclose(pY, 60.0, rel_tol=1e-12)
    assert math.isclose(pZ, 30.0 * math.sqrt(3.0), rel_tol=1e-12)
