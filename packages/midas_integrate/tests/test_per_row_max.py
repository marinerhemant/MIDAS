"""per_row_max_entries must not silently drop map entries.

Regression tests for the trap found on the 20-ID Pilatus (2026-08-18): the old
fixed default ``max(NrPixelsY*4, 4000)`` was too small at ``RBinSize 0.5`` and
dropped 42,471 entries plus 1,482 whole bins with no warning unless
``verbose=True``. Truncation removes ``frac`` and ``areaWeight`` together, so a
normalised profile still looks plausible while absolute flux and bin occupancy
are wrong — which is exactly why it has to be loud.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from midas_integrate.detector_mapper import (
    DEFAULT_MAP_BUFFER_BYTES,
    MapTruncationWarning,
    build_map,
    estimate_per_row_max,
)
from midas_integrate.params import IntegrationParams


def _pilatus_like(**over) -> IntegrationParams:
    """The 20-ID geometry that exposed the bug (48-panel Pilatus, dR=0.5)."""
    p = IntegrationParams(
        NrPixelsY=1475, NrPixelsZ=1679, pxY=172.0, pxZ=172.0,
        Lsd=646789.128263, BC_y=738.249292, BC_z=835.678,
        RMin=100.25, RMax=1000.25, RBinSize=0.5,
        EtaMin=-179.5, EtaMax=180.5, EtaBinSize=1.0,
        Wavelength=0.1958242893, RhoD=218641.942637,
    )
    for k, v in over.items():
        setattr(p, k, v)
    return p


# ── the estimator ────────────────────────────────────────────────────────
def test_estimate_beats_the_old_fixed_default_on_fine_r_bins():
    """The geometry that broke: old default 5900 was far too small."""
    p = _pilatus_like()
    old_default = max(p.NrPixelsY * 4, 4000)
    assert old_default == 5900                       # what it used to be
    assert estimate_per_row_max(p) > old_default


def test_estimate_scales_inversely_with_r_bin_size():
    coarse = estimate_per_row_max(_pilatus_like(RBinSize=2.0))
    fine = estimate_per_row_max(_pilatus_like(RBinSize=0.25))
    assert fine > coarse


def test_estimate_scales_with_subpixel_level_squared():
    one = estimate_per_row_max(_pilatus_like(SubPixelLevel=1))
    four = estimate_per_row_max(_pilatus_like(SubPixelLevel=4),
                                max_bytes=1 << 60)   # no clamp
    assert four >= 8 * one       # ~16x before headroom/floor effects


def test_estimate_respects_the_memory_budget():
    p = _pilatus_like(SubPixelLevel=4)
    est = estimate_per_row_max(p, max_bytes=64 * 1024 ** 2)
    assert est * p.NrPixelsZ * 6 * 8 <= 64 * 1024 ** 2


def test_estimate_has_a_floor_for_tiny_geometries():
    p = IntegrationParams(NrPixelsY=32, NrPixelsZ=32, pxY=200.0, pxZ=200.0,
                          Lsd=100000.0, BC_y=16, BC_z=16,
                          RMin=2.0, RMax=14.0, RBinSize=2.0,
                          EtaMin=-180.0, EtaMax=180.0, EtaBinSize=10.0)
    assert estimate_per_row_max(p) >= 4000


def test_default_budget_is_documented_and_sane():
    assert DEFAULT_MAP_BUFFER_BYTES == 4 * 1024 ** 3


# ── the warning ──────────────────────────────────────────────────────────
def _small_params() -> IntegrationParams:
    """Small enough to build quickly, fine enough to overflow a tiny buffer."""
    return IntegrationParams(
        NrPixelsY=64, NrPixelsZ=64, pxY=200.0, pxZ=200.0,
        Lsd=200000.0, BC_y=32.0, BC_z=32.0,
        RMin=2.0, RMax=26.0, RBinSize=0.5,
        EtaMin=-180.0, EtaMax=180.0, EtaBinSize=2.0,
        Wavelength=0.2, RhoD=100000.0,
    )


def test_truncation_warns_even_when_not_verbose():
    """The old code hid this behind verbose=True. It must always fire."""
    p = _small_params()
    with pytest.warns(MapTruncationWarning, match="map truncated"):
        build_map(p, per_row_max_entries=4, verbose=False)


def test_no_warning_when_the_buffer_is_adequate():
    p = _small_params()
    with warnings.catch_warnings():
        warnings.simplefilter("error", MapTruncationWarning)
        build_map(p, per_row_max_entries=200_000, verbose=False)


def test_truncation_actually_loses_entries():
    """Guards the claim in the warning text: this is data loss, not a nag."""
    p = _small_params()
    full = build_map(p, per_row_max_entries=200_000, verbose=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MapTruncationWarning)
        cut = build_map(p, per_row_max_entries=4, verbose=False)
    assert cut.pxList.shape[0] < full.pxList.shape[0]
    assert int((cut.counts > 0).sum()) < int((full.counts > 0).sum())


# ── the C-parity default ─────────────────────────────────────────────────
def test_subpixel_cardinal_width_default_matches_the_c_code():
    """DetectorMapper.c:109 uses 10.0; a different default here silently
    produced a different map from the same parameter file."""
    assert IntegrationParams().SubPixelCardinalWidth == 10.0
