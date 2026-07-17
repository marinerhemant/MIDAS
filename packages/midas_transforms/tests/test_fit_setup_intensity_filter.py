"""N8: ``MinIntegratedIntensity`` fit_setup spot filter.

Default 0 = off (no FF behaviour change); when set, spots below the
threshold are rejected exactly like the MinEta/BoxSize filters (row
zeroed except SpotID) and the key is recorded in paramstest.txt so
reruns see it — replacing the Ni-run's unrecorded hand-editing of layer
CSVs (awk $15>=200).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from midas_transforms.fit_setup import fit_setup
from midas_transforms.params import ZarrParams, read_paramstest


def _zarr_params(**kw) -> ZarrParams:
    zp = ZarrParams()
    zp.Lsd = 1_000_000.0
    zp.Wavelength = 0.18
    zp.PixelSize = 200.0
    zp.YCen = 0.0
    zp.ZCen = 0.0
    zp.RingThresh = [(2, 80.0)]
    zp.OverallRingToIndex = 2
    zp.MinEta = 6.0
    zp.MaxRingRad = 300_000.0
    zp.BeamSize = 2000.0
    zp.EndNr = 10
    zp.LatticeConstant = (3.6, 3.6, 3.6, 90.0, 90.0, 90.0)
    for k, v in kw.items():
        setattr(zp, k, v)
    return zp


def _hkls_csv(path: Path) -> None:
    txt = "h k l D-spacing RingNr g1 g2 g3 Theta(deg) 2Theta(deg) Radius\n"
    txt += "2 0 0 1.8 2 0 0 0 2.87 5.73 140000.0\n"
    path.write_text(txt)


def _radius_array(intensities) -> np.ndarray:
    """One ring-2 spot per intensity, at distinct omegas/eta positions."""
    n = len(intensities)
    arr = np.zeros((n, 26), dtype=np.float64)
    for i, ii in enumerate(intensities):
        eta = 30.0 + 10.0 * i
        r_px = 700.0
        arr[i, 0] = i + 1                       # SpotID
        arr[i, 1] = ii                          # IntegratedIntensity
        arr[i, 2] = -50.0 + 10.0 * i            # Omega
        arr[i, 3] = -r_px * math.sin(math.radians(eta))   # YCen px
        arr[i, 4] = r_px * math.cos(math.radians(eta))    # ZCen px
        arr[i, 13] = 2                          # RingNr
        arr[i, 15] = 5.0                        # GrainRadius
        arr[i, 21] = ii                         # RawSum
        arr[i, 24] = 1000 + i                   # OrigSpotID
    return arr


@pytest.mark.parametrize("min_int,expected_kept", [(0.0, 3), (200.0, 2)])
def test_intensity_filter(tmp_path: Path, min_int: float, expected_kept: int):
    zp = _zarr_params(MinIntegratedIntensity=min_int)
    _hkls_csv(tmp_path / "hkls.csv")
    res = fit_setup(
        result_folder=tmp_path,
        zarr_params=zp,
        radius_array=_radius_array([50.0, 500.0, 5000.0]),
        hkls_path=tmp_path / "hkls.csv",
        write=True,
        device="cpu", dtype="float64",
    )
    inputall = res.spots_inputall.detach().cpu().numpy()
    # Rejected rows stay in the table with everything but SpotID zeroed
    # (C convention) — count rows with a live YLab/ZLab.
    kept = int((np.abs(inputall[:, [0, 1]]).sum(axis=1) > 0).sum())
    assert kept == expected_kept

    # The applied threshold is recorded in paramstest.txt iff set.
    pt = read_paramstest(tmp_path / "paramstest.txt")
    assert pt.MinIntegratedIntensity == pytest.approx(min_int)
    text = (tmp_path / "paramstest.txt").read_text()
    if min_int:
        assert "MinIntegratedIntensity" in text
    else:
        assert "MinIntegratedIntensity" not in text
