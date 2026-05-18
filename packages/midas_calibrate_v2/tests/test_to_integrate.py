"""Tests for the v2 → midas-integrate handoff (compat/to_integrate.py).

Three behaviours pinned:

1. ``to_integrate_params`` accepts a ``PVCalibrationResult``-shape (only
   ``.unpacked``) and returns an ``IntegrationParams`` whose distortion
   slots match the canonical v2 → v1 remap.

2. ``to_integrate_params`` accepts a ``FourStageResult``-shape
   (``.stage2.unpacked`` + ``.stage3_spline_fn``) and, when
   ``output_dir`` is given, evaluates the spline onto a binary grid and
   wires ``IntegrationParams.ResidualCorrectionMap`` to the resulting
   path. Loading the file back through
   :func:`midas_integrate.residual_corr.load_residual_correction_map`
   reproduces the spline values within fp64 tolerance.

3. δr_k present in ``unpacked`` plus ring info plus ``output_dir``
   produces the JSON sidecar at the expected path; without ring info
   the warning from ``params_from_v2_unpacked`` fires (already pinned
   in midas_integrate).
"""
from __future__ import annotations

import json
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("midas_integrate")

from midas_calibrate_v2.compat.to_integrate import to_integrate_params
from midas_integrate.params import IntegrationParams


# ---- minimal stand-ins for the v2 result types ----

@dataclass
class _PVLike:
    unpacked: dict


@dataclass
class _StageLike:
    unpacked: dict


@dataclass
class _FourStageLike:
    stage2: _StageLike
    stage3_spline_fn: Optional[object] = None


def _v2_unpacked(**overrides):
    base = {
        "Lsd": torch.tensor(895_900.0, dtype=torch.float64),
        "BC_y": torch.tensor(1447.0, dtype=torch.float64),
        "BC_z": torch.tensor(1469.0, dtype=torch.float64),
        "tx": torch.tensor(0.0, dtype=torch.float64),
        "ty": torch.tensor(-0.31, dtype=torch.float64),
        "tz": torch.tensor(0.39, dtype=torch.float64),
        "Parallax": torch.tensor(12.5, dtype=torch.float64),
        "iso_R2": torch.tensor(3.4e-4, dtype=torch.float64),
        "iso_R4": torch.tensor(1.2e-3, dtype=torch.float64),
        "iso_R6": torch.tensor(-9.1e-4, dtype=torch.float64),
        "a1": torch.tensor(4.3e-4, dtype=torch.float64),
        "phi1": torch.tensor(110.1, dtype=torch.float64),
    }
    base.update({k: torch.tensor(v, dtype=torch.float64)
                 for k, v in overrides.items()})
    return base


def _template():
    return IntegrationParams(
        NrPixelsY=2880, NrPixelsZ=2880,
        pxY=150.0, pxZ=150.0,
        RhoD=309_094.286,
    )


# ---- 1. PVCalibrationResult shape ----

def test_to_integrate_params_pv_result_shape():
    pv = _PVLike(unpacked=_v2_unpacked())
    ip = to_integrate_params(pv, template=_template(), warn_on_dropped=False)

    # Distortion remap landed in the right v1 slots:
    assert ip.p2 == pytest.approx(3.4e-4)   # iso_R2
    assert ip.p5 == pytest.approx(1.2e-3)   # iso_R4
    assert ip.p4 == pytest.approx(-9.1e-4)  # iso_R6
    assert ip.p7 == pytest.approx(4.3e-4)   # a1
    assert ip.p8 == pytest.approx(110.1)    # phi1
    # Geometry passed through:
    assert ip.Lsd == pytest.approx(895_900.0)
    assert ip.BC_y == pytest.approx(1447.0)
    assert ip.Parallax == pytest.approx(12.5)
    # No spline → ResidualCorrectionMap stays empty:
    assert ip.ResidualCorrectionMap == ""


# ---- 2. FourStageResult shape with Stage 4 spline ----

def test_to_integrate_params_four_stage_writes_residual_map(tmp_path):
    NrY = NrZ = 64
    # Synthetic spline: ΔR(Y, Z) = 0.5 + 1e-3 * (Y - NrY/2) (in µm).
    def spline_predict(Y, Z):
        return 0.5 + 1e-3 * (Y - NrY / 2.0)

    fs = _FourStageLike(
        stage2=_StageLike(unpacked=_v2_unpacked()),
        stage3_spline_fn=spline_predict,
    )
    template = IntegrationParams(
        NrPixelsY=NrY, NrPixelsZ=NrZ, pxY=150.0, pxZ=150.0,
        RhoD=float(NrY),
    )
    ip = to_integrate_params(fs, template=template, output_dir=tmp_path,
                              warn_on_dropped=False)

    # File written at the default name and wired into the params:
    assert ip.ResidualCorrectionMap == str(tmp_path / "residual_corr.bin")
    assert (tmp_path / "residual_corr.bin").exists()

    # Round-trip: read the binary back and verify it matches the spline
    # (in pixels, with px_mean = 150 µm).
    grid = np.fromfile(tmp_path / "residual_corr.bin",
                       dtype=np.float64).reshape(NrZ, NrY)
    expected = (0.5 + 1e-3 * (np.arange(NrY) - NrY / 2.0)) / 150.0
    # Each row is identical (spline doesn't depend on Z); compare the
    # first row.
    np.testing.assert_allclose(grid[0, :], expected, rtol=0, atol=1e-12)
    # All rows identical:
    np.testing.assert_allclose(grid - grid[0:1, :], 0.0, atol=1e-15)

    # Distortion remap still works through the FourStage path:
    assert ip.p2 == pytest.approx(3.4e-4)


# ---- 3. δr_k JSON sidecar ----

def test_to_integrate_params_writes_delta_r_k_sidecar(tmp_path):
    pv = _PVLike(unpacked=_v2_unpacked(
        delta_r_k=np.array([0.01, -0.005, 0.0, 0.002, -0.007],
                            dtype=np.float64),
    ))
    pv.unpacked["delta_r_k"] = torch.tensor(
        [0.01, -0.005, 0.0, 0.002, -0.007], dtype=torch.float64,
    )
    ring_d = np.array([3.124, 2.708, 1.913, 1.633, 1.562])
    ring_tt = np.array([3.61, 4.16, 5.89, 6.91, 7.22])

    ip = to_integrate_params(
        pv, template=_template(), output_dir=tmp_path,
        ring_d_spacing_A=ring_d, ring_two_theta_deg=ring_tt,
        warn_on_dropped=False,
    )
    sidecar = tmp_path / "delta_r_k.json"
    assert sidecar.exists()
    payload = json.loads(sidecar.read_text())
    assert payload["n_rings"] == 5
    assert len(payload["rings"]) == 5
    assert payload["rings"][0]["delta_r_px"] == pytest.approx(0.01)
    assert payload["rings"][2]["d_spacing_A"] == pytest.approx(1.913)


# ---- 4. result-type guard ----

def test_to_integrate_params_rejects_unknown_result_type():
    class Bogus:
        pass

    with pytest.raises(TypeError, match="has neither .unpacked nor .stage2"):
        to_integrate_params(Bogus(), template=_template())
