"""A bad-pixel mask must follow the spec's device (hard and subpixel binning).

Found on a real 2880 x 2880 frame: ``midas-integrate-v2 --device cuda --mask m.tif --mode subpixel``
died with "Expected all tensors to be on the same device, cuda:0 and cpu" because the mask tensor
was built on the CPU.  CPU-only machines cannot reproduce it, so this skips without CUDA.
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pytest
import torch

from midas_integrate.params import IntegrationParams
from midas_integrate_v2 import spec_from_v1_params
from midas_integrate_v2.binning.hard import HardBinGeometry
from midas_integrate_v2.binning.subpixel import SubpixelBinGeometry


def _spec_cuda():
    p = IntegrationParams(
        NrPixelsY=32, NrPixelsZ=32, pxY=200.0, pxZ=200.0, Lsd=1_000_000.0,
        BC_y=16.3, BC_z=15.6, RhoD=32.0, RMin=1.0, RMax=14.0, RBinSize=1.0,
        EtaMin=-180.0, EtaMax=180.0, EtaBinSize=30.0,
    )
    return spec_from_v1_params(p, device=torch.device("cuda"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("cls,kw", [(SubpixelBinGeometry, {"K": 2}), (HardBinGeometry, {})])
def test_mask_follows_spec_device(cls, kw):
    spec = _spec_cuda()
    mask = np.zeros((32, 32), np.uint8)
    mask[5, :] = 1
    geom = cls.from_spec(spec, mask=mask, **kw)   # used to raise a cpu/cuda device mismatch
    assert geom.valid.device.type == "cuda"
