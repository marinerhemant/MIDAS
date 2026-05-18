"""Real-data smoke #2: Varex Aero CeO₂ from FF_HEDM/Example/Calibration.

Same example folder as the Pilatus smoke (test_followup_real_data_ceo2.py)
but the Varex frame has no pre-calibrated parameter file — the
``runAllCalibrations.sh`` script feeds it through ``AutoCalibrateZarr``
from scratch. Here we don't run a calibration loop (8.3 M pixels is too
slow for a unit-test refinement); we just verify v2 can:

1. Build an :class:`IntegrationSpec` from a rough initial estimate
   (pixel size + Lsd from the filename, BC at detector centre, zero
   distortion) and integrate the Varex frame end-to-end via the
   pure-torch :func:`integrate_soft` path.

2. The resulting 1D profile shows a recognisable CeO₂ first ring at the
   physically-predicted radius (within a generous tolerance to allow
   for the rough BC initial guess).

Marked ``@pytest.mark.slow`` — single integration on 2880×2880 = 8.3 M
pixels takes a few seconds.
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from pathlib import Path

import numpy as np
import pytest
import torch

_REPO = Path(__file__).resolve().parents[3]
_DATA = _REPO / "FF_HEDM" / "Example" / "Calibration"
_IMAGE = _DATA / "Ceria_63keV_900mm_100x100_0p5s_aero_0_001137.tif"

if not _IMAGE.exists():
    pytest.skip("FF_HEDM/Example/Calibration Varex Aero frame not found",
                allow_module_level=True)

tifffile = pytest.importorskip("tifffile")

from midas_integrate.params import IntegrationParams

from midas_integrate_v2 import (
    spec_from_v1_params,
    SoftBinGeometry,
    integrate_soft,
    profile_1d_diff,
)

pytestmark = pytest.mark.slow


def _load_varex_image() -> torch.Tensor:
    img = tifffile.imread(_IMAGE).astype(np.float64)
    # ImTransOpt 2 = invert Z (rows) — same convention as the Pilatus
    # entry in runAllCalibrations.sh.
    img = img[::-1, :].copy()
    return torch.from_numpy(img)


def _rough_varex_params():
    """Initial-estimate geometry from the runAllCalibrations.sh entry:
    px=150 µm, λ=0.196793 Å, Lsd=900 mm, BC at detector centre, zero
    distortion. No tilts. RhoD = half the detector diagonal."""
    NY = NZ = 2880
    p = IntegrationParams(
        NrPixelsY=NY, NrPixelsZ=NZ,
        pxY=150.0, pxZ=150.0,
        Lsd=900_000.0,                           # 900 mm in µm
        BC_y=NY / 2.0, BC_z=NZ / 2.0,
        RhoD=float(NY),                          # generous; only used by distortion=0
        # Wide R range that covers the strongest CeO₂ rings.
        RMin=50.0, RMax=1500.0, RBinSize=2.0,
        EtaMin=-180.0, EtaMax=180.0, EtaBinSize=10.0,
        Wavelength=0.196793,
    )
    return p


def test_v2_integrates_varex_aero_one_shot():
    """Smoke: build IntegrationSpec from a rough estimate, run
    SoftBinGeometry + integrate_soft on the Varex frame, get a 1D
    profile with a CeO₂ first-ring peak in the predicted band."""
    p = _rough_varex_params()
    img = _load_varex_image()
    assert img.shape == (p.NrPixelsZ, p.NrPixelsY), (
        f"unexpected image shape {tuple(img.shape)}"
    )

    spec = spec_from_v1_params(p, requires_grad=False)
    geom = SoftBinGeometry.from_spec(spec)
    int2d = integrate_soft(img, geom)
    prof = profile_1d_diff(int2d, spec).detach().numpy()

    n_r = spec.n_r_bins
    r_axis = spec.RMin + spec.RBinSize * (np.arange(n_r) + 0.5)

    # CeO₂ first-ring (111): d = 3.124 Å.
    # 2θ = 2 arcsin(λ / 2d) with λ = 0.196793 Å:
    #   2θ ≈ 3.61°
    # R = (Lsd / px) tan(2θ) = (900_000 / 150) tan(3.61°) ≈ 379 px.
    # Allow a wide search band because BC is rough.
    band = (r_axis > 320) & (r_axis < 450)
    band_idx = np.where(band)[0]
    assert band_idx.size > 0, "search band [320, 450] is empty"
    peak_idx = band_idx[np.argmax(prof[band])]
    peak_R = r_axis[peak_idx]

    # Generous tolerance — rough BC is likely off by tens of px.
    assert 350 <= peak_R <= 410, (
        f"Varex first CeO₂ ring at R={peak_R:.1f} px, expected "
        "around 379 px (rough estimate; adjust BC for a tighter band)"
    )
