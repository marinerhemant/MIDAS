"""The v1 mapper is a single-pitch model (pxY for both axes, pxZ ignored): non-square pixels must warn, square must not."""
import warnings

import pytest

from midas_integrate.detector_mapper import build_map
from midas_integrate.params import IntegrationParams


def _p(pxz):
    return IntegrationParams(NrPixelsY=24, NrPixelsZ=24, pxY=172.0, pxZ=pxz, Lsd=1_000_000.0, BC_y=12.3, BC_z=11.7,
                             RhoD=24.0 * 172.0, RMin=1.0, RMax=10.0, RBinSize=1.0, EtaMin=-180.0, EtaMax=180.0,
                             EtaBinSize=60.0, SubPixelLevel=1)


def test_non_square_pixels_warn():
    with pytest.warns(UserWarning, match="single-pitch"):
        build_map(_p(200.0), auto_load=False, verbose=False, use_numba=False)


def test_square_and_unset_pxZ_do_not_warn():
    for pxz in (172.0, 0.0):
        with warnings.catch_warnings():
            warnings.filterwarnings("error", message=".*single-pitch.*")
            build_map(_p(pxz), auto_load=False, verbose=False, use_numba=False)
