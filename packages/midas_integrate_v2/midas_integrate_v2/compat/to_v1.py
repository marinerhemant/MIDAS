"""Build a v1 :class:`IntegrationParams` from a v2 :class:`IntegrationSpec`.

Inverse of :func:`spec_from_v1_params`. Tensor fields are detached and
converted to Python scalars; the v2 distortion names are remapped to the
v1 ``p0``..``p14`` slots so downstream tools that only know about v1 (the
``midas-integrate`` CLI, third-party Rietveld engines, etc.) keep
working.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Optional

import torch

from midas_integrate.params import IntegrationParams

from ..spec import IntegrationSpec
from .from_v1 import _V1_TO_V2_DISTORTION


# v2 → v1 (inverse of _V1_TO_V2_DISTORTION).
_V2_TO_V1_DISTORTION = {v: k for k, v in _V1_TO_V2_DISTORTION.items()}


def _scalar(t) -> float:
    if isinstance(t, torch.Tensor):
        return float(t.detach().cpu().item())
    return float(t)


def v1_params_from_spec(s: IntegrationSpec) -> IntegrationParams:
    """Convert v2 :class:`IntegrationSpec` to v1 :class:`IntegrationParams`."""
    p = IntegrationParams()

    p.NrPixelsY = int(s.NrPixelsY); p.NrPixelsZ = int(s.NrPixelsZ)
    p.pxY = float(s.pxY); p.pxZ = float(s.pxZ)
    p.RhoD = float(s.RhoD)

    p.Lsd = _scalar(s.Lsd)
    p.BC_y = _scalar(s.BC_y); p.BC_z = _scalar(s.BC_z)
    p.tx = _scalar(s.tx); p.ty = _scalar(s.ty); p.tz = _scalar(s.tz)
    p.Parallax = _scalar(s.Parallax)
    p.Wavelength = _scalar(s.Wavelength)

    for v2_name, v1_key in _V2_TO_V1_DISTORTION.items():
        setattr(p, v1_key, _scalar(getattr(s, v2_name)))

    p.RMin = float(s.RMin); p.RMax = float(s.RMax)
    p.RBinSize = float(s.RBinSize)
    p.EtaMin = float(s.EtaMin); p.EtaMax = float(s.EtaMax)
    p.EtaBinSize = float(s.EtaBinSize)
    p.QMin = float(s.QMin); p.QMax = float(s.QMax)
    p.QBinSize = float(s.QBinSize)

    p.Normalize = int(s.Normalize); p.SumImages = int(s.SumImages)
    p.Write2D = int(s.Write2D); p.DoBinSort = int(s.DoBinSort)
    p.SubPixelLevel = int(s.SubPixelLevel)
    p.SubPixelCardinalWidth = float(s.SubPixelCardinalWidth)
    p.SolidAngleCorrection = int(s.SolidAngleCorrection)
    p.PolarizationCorrection = int(s.PolarizationCorrection)
    p.PolarizationFraction = float(s.PolarizationFraction)
    p.PolarizationPlaneEtaDeg = float(s.PolarizationPlaneEtaDeg)
    p.GradientCorrection = int(s.GradientCorrection)
    p.NrTransOpt = int(s.NrTransOpt)
    p.TransOpt = list(s.TransOpt)
    p.GapIntensity = int(s.GapIntensity); p.BadPxIntensity = int(s.BadPxIntensity)

    p.DistortionFile = str(s.DistortionFile)
    p.NPanelsY = int(s.NPanelsY); p.NPanelsZ = int(s.NPanelsZ)
    p.PanelSizeY = int(s.PanelSizeY); p.PanelSizeZ = int(s.PanelSizeZ)
    p.PanelGapsY = list(s.PanelGapsY); p.PanelGapsZ = list(s.PanelGapsZ)
    p.PanelShiftsFile = str(s.PanelShiftsFile)
    p.ResidualCorrectionMap = str(s.ResidualCorrectionMap)

    p.DataDirectory = str(s.DataDirectory)
    p.SkipFrame = int(s.SkipFrame)
    p.FlatFile = str(s.FlatFile); p.MaskFile = str(s.MaskFile)

    return p


__all__ = ["v1_params_from_spec", "_V2_TO_V1_DISTORTION"]
