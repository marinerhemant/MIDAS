"""Phase 0 tests: IntegrationSpec + v1 ↔ v2 round-trip."""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import pytest
import torch

from midas_integrate.params import IntegrationParams

from midas_integrate_v2 import (
    IntegrationSpec,
    spec_from_v1_params,
    v1_params_from_spec,
    DISTORTION_NAMES,
)


def _populated_v1():
    p = IntegrationParams()
    p.NrPixelsY = 2880; p.NrPixelsZ = 2880
    p.pxY = 150.0; p.pxZ = 150.0
    p.RhoD = 309_094.286
    p.Lsd = 895_900.0
    p.BC_y = 1447.0; p.BC_z = 1469.0
    p.tx = 0.0; p.ty = -0.31; p.tz = 0.39
    p.Parallax = 12.5
    p.Wavelength = 0.172979
    # All 15 distortion coeffs unique so the remap is testable
    for k in range(15):
        setattr(p, f"p{k}", 1e-3 + 1e-4 * k)
    p.RMin = 10.0; p.RMax = 2000.0; p.RBinSize = 1.0
    p.EtaMin = -180.0; p.EtaMax = 180.0; p.EtaBinSize = 5.0
    p.TransOpt = [2, 1]; p.NrTransOpt = 2
    p.MaskFile = "/tmp/mask.bin"
    p.PanelGapsY = [11, 22]; p.PanelGapsZ = [33]
    return p


def test_spec_default_construction():
    s = IntegrationSpec()
    assert isinstance(s.Lsd, torch.Tensor)
    assert s.Lsd.dtype == torch.float64
    for name in DISTORTION_NAMES:
        assert isinstance(getattr(s, name), torch.Tensor)
        assert float(getattr(s, name)) == 0.0


def test_round_trip_v1_v2_v1_identity_on_all_fields():
    """Every field must round-trip v1 → v2 → v1 identically. Catches
    silent drift in the distortion remap or any field added on one side
    but not the other."""
    p_in = _populated_v1()
    s = spec_from_v1_params(p_in)
    p_out = v1_params_from_spec(s)

    scalar_fields = (
        "NrPixelsY", "NrPixelsZ", "pxY", "pxZ", "RhoD",
        "Lsd", "BC_y", "BC_z", "tx", "ty", "tz",
        "Parallax", "Wavelength",
        *(f"p{k}" for k in range(15)),
        "RMin", "RMax", "RBinSize",
        "EtaMin", "EtaMax", "EtaBinSize",
        "QMin", "QMax", "QBinSize",
        "Normalize", "SumImages", "Write2D", "DoBinSort",
        "SubPixelLevel", "SubPixelCardinalWidth",
        "SolidAngleCorrection", "PolarizationCorrection",
        "PolarizationFraction", "PolarizationPlaneEtaDeg",
        "GradientCorrection", "NrTransOpt",
        "GapIntensity", "BadPxIntensity",
        "DistortionFile", "NPanelsY", "NPanelsZ",
        "PanelSizeY", "PanelSizeZ",
        "PanelShiftsFile", "ResidualCorrectionMap",
        "DataDirectory", "SkipFrame", "FlatFile", "MaskFile",
    )
    for f in scalar_fields:
        assert getattr(p_in, f) == pytest.approx(getattr(p_out, f)), (
            f"v1→v2→v1 round-trip lost {f}: in={getattr(p_in, f)}, "
            f"out={getattr(p_out, f)}"
        )

    list_fields = ("TransOpt", "PanelGapsY", "PanelGapsZ")
    for f in list_fields:
        assert list(getattr(p_in, f)) == list(getattr(p_out, f)), (
            f"v1→v2→v1 round-trip lost list field {f}"
        )


def test_distortion_remap_v1_to_v2_canonical_names():
    """Verify the exact v1 p-index → v2 name mapping (must match
    midas_integrate.compat.from_v2._V2_TO_V1_DISTORTION)."""
    p = IntegrationParams()
    # Each slot gets a unique value
    p.p0 = 100.0; p.p1 = 101.0; p.p2 = 102.0; p.p3 = 103.0; p.p4 = 104.0
    p.p5 = 105.0; p.p6 = 106.0; p.p7 = 107.0; p.p8 = 108.0; p.p9 = 109.0
    p.p10 = 110.0; p.p11 = 111.0; p.p12 = 112.0; p.p13 = 113.0; p.p14 = 114.0
    p.NrPixelsY = 1; p.NrPixelsZ = 1
    p.pxY = 1.0; p.pxZ = 1.0; p.Lsd = 1.0
    s = spec_from_v1_params(p)

    # iso_R2→p2, iso_R4→p5, iso_R6→p4
    assert float(s.iso_R2) == 102.0
    assert float(s.iso_R4) == 105.0
    assert float(s.iso_R6) == 104.0
    # a1→p7, phi1→p8
    assert float(s.a1) == 107.0
    assert float(s.phi1) == 108.0
    # a2→p0, phi2→p6
    assert float(s.a2) == 100.0
    assert float(s.phi2) == 106.0
    # a3→p9, phi3→p10
    assert float(s.a3) == 109.0
    assert float(s.phi3) == 110.0
    # a4→p1, phi4→p3
    assert float(s.a4) == 101.0
    assert float(s.phi4) == 103.0
    # a5→p11, phi5→p12
    assert float(s.a5) == 111.0
    assert float(s.phi5) == 112.0
    # a6→p13, phi6→p14
    assert float(s.a6) == 113.0
    assert float(s.phi6) == 114.0


def test_requires_grad_propagates_to_refinable_tensors():
    p = _populated_v1()
    s = spec_from_v1_params(p, requires_grad=True)
    for name in ("Lsd", "BC_y", "BC_z", "tx", "ty", "tz",
                 "Parallax", "Wavelength", *DISTORTION_NAMES):
        t = getattr(s, name)
        assert t.requires_grad, f"{name} should require grad"
        assert t.grad is None  # not yet backward'd


def test_round_trip_remap_consistent_with_v1_compat_module():
    """Pin that v2's remap matches the one already in
    ``midas_integrate.compat.from_v2`` — single source of truth across
    the three packages."""
    from midas_integrate.compat.from_v2 import _V2_TO_V1_DISTORTION as A
    from midas_integrate_v2.compat.to_v1 import _V2_TO_V1_DISTORTION as B
    assert A == B, (
        "Distortion remap divergence between midas_integrate.compat and "
        f"midas_integrate_v2.compat: A={A}, B={B}"
    )


def test_refinable_tensors_dict_includes_all_geometry_and_distortion():
    s = spec_from_v1_params(_populated_v1(), requires_grad=True)
    refinable = s.refinable_tensors()
    expected = {"Lsd", "BC_y", "BC_z", "tx", "ty", "tz",
                "Parallax", "Wavelength", *DISTORTION_NAMES}
    assert set(refinable.keys()) == expected
    for v in refinable.values():
        assert isinstance(v, torch.Tensor) and v.requires_grad
