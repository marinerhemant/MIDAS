"""Phase 1: bit-identical map + 1D profile parity vs midas-integrate v1.

If v2's wrapper drifts from v1 (e.g. wrong field carried, wrong device,
wrong dtype propagation) the smoke profile diverges immediately. The
bar is set tight on purpose — the wrapper does not change the math.
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pytest
import torch

from midas_integrate.params import IntegrationParams
from midas_integrate.detector_mapper import build_map as v1_build_map
from midas_integrate.bin_io import PixelMap
from midas_integrate.kernels import (
    build_csr as v1_build_csr,
    integrate as v1_integrate,
    profile_1d as v1_profile_1d,
)

from midas_integrate_v2 import (
    IntegrationSpec,
    spec_from_v1_params,
    build_geometry,
    integrate as v2_integrate,
    profile_1d as v2_profile_1d,
    build_map as v2_build_map,
)
from midas_integrate_v2.binning.build import _spec_param_hash


def _v1():
    p = IntegrationParams(
        NrPixelsY=64, NrPixelsZ=64,
        pxY=200.0, pxZ=200.0, Lsd=1_000_000.0,
        BC_y=32.0, BC_z=32.0, RhoD=64.0,
        RMin=2.0, RMax=30.0, RBinSize=2.0,
        EtaMin=-180.0, EtaMax=180.0, EtaBinSize=60.0,
        SubPixelLevel=1,
    )
    p.p2 = 3.4e-4
    p.p7 = 4.3e-4; p.p8 = 110.1
    return p


def test_v2_build_map_matches_v1_pixel_for_pixel():
    """Same geometry → same Map structure, no field drift through the
    v1 ↔ v2 ↔ v1 round-trip."""
    p = _v1()
    s = spec_from_v1_params(p)

    r1 = v1_build_map(p, auto_load=False, verbose=False, use_numba=False)
    r2 = v2_build_map(s, auto_load=False, verbose=False, use_numba=False).result

    assert r1.pxList.shape == r2.pxList.shape
    assert r1.counts.shape == r2.counts.shape
    np.testing.assert_array_equal(r1.counts, r2.counts)
    np.testing.assert_array_equal(r1.offsets, r2.offsets)
    for f in ("y", "z", "frac", "areaWeight"):
        np.testing.assert_array_equal(r1.pxList[f], r2.pxList[f])


def test_v2_param_hash_matches_v1():
    """v2's spec hash must match v1's compute_param_hash bit-for-bit so
    Map.bin caches are interchangeable across the two packages."""
    from midas_integrate.bin_io import compute_param_hash
    p = _v1()
    s = spec_from_v1_params(p)
    h_v1 = compute_param_hash(
        Lsd=p.Lsd, Ycen=p.BC_y, Zcen=p.BC_z, pxY=p.pxY, pxZ=p.pxZ,
        tx=p.tx, ty=p.ty, tz=p.tz,
        p0=p.p0, p1=p.p1, p2=p.p2, p3=p.p3, p4=p.p4,
        p5=p.p5, p6=p.p6, p7=p.p7, p8=p.p8, p9=p.p9,
        p10=p.p10, p11=p.p11, p12=p.p12, p13=p.p13, p14=p.p14,
        Parallax=p.Parallax, RhoD=p.RhoD,
        RBinSize=p.RBinSize, EtaBinSize=p.EtaBinSize,
        RMin=p.RMin, RMax=p.RMax, EtaMin=p.EtaMin, EtaMax=p.EtaMax,
        NrPixelsY=p.NrPixelsY, NrPixelsZ=p.NrPixelsZ,
        TransOpt=tuple(p.TransOpt),
        qMode=int(p.q_mode_active),
        Wavelength=p.Wavelength,
    )
    h_v2 = _spec_param_hash(s)
    assert h_v1 == h_v2


def test_v2_integrated_profile_bit_identical_to_v1():
    """Same image + same map → same integrated profile.  No tolerances:
    v2's forward calls v1's CSR kernel directly, so any drift is a
    wrapper bug, not numerics."""
    p = _v1()
    s = spec_from_v1_params(p)

    # Build map both ways.
    r1 = v1_build_map(p, auto_load=False, verbose=False, use_numba=False)
    pm = PixelMap(
        pxList=r1.pxList, counts=r1.counts, offsets=r1.offsets,
        map_header=None, nmap_header=None,
    )
    g1 = v1_build_csr(pm, n_r=p.n_r_bins, n_eta=p.n_eta_bins,
                      n_pixels_y=p.NrPixelsY, n_pixels_z=p.NrPixelsZ,
                      device="cpu", dtype=torch.float64,
                      bc_y=p.BC_y, bc_z=p.BC_z)
    # Build via v2 path. use_numba=False sidesteps a numba/torch
    # OpenMP thread-state conflict that triggers a segfault when torch
    # was imported earlier in the same Python session.
    g2 = build_geometry(s, device="cpu", dtype=torch.float64,
                          use_numba=False)

    rng = np.random.default_rng(0xC011AB)
    img_np = rng.uniform(1.0, 100.0, size=(p.NrPixelsZ, p.NrPixelsY)).astype(np.float64)
    img = torch.from_numpy(img_np)

    int2d_v1 = v1_integrate(img, g1, mode="floor", normalize=True)
    int2d_v2 = v2_integrate(img, g2, mode="floor", normalize=True)
    torch.testing.assert_close(int2d_v1, int2d_v2, rtol=0, atol=0)

    prof_v1 = v1_profile_1d(int2d_v1, g1, mode="area_weighted")
    prof_v2 = v2_profile_1d(int2d_v2, g2, mode="area_weighted")
    torch.testing.assert_close(prof_v1, prof_v2, rtol=0, atol=0)


def test_map_cache_skips_rebuild_when_unchanged():
    from midas_integrate_v2 import MapCache

    p = _v1()
    s = spec_from_v1_params(p)
    cache = MapCache(max_entries=2)

    bm_a = cache.get(s, auto_load=False, verbose=False, use_numba=False)
    bm_b = cache.get(s, auto_load=False, verbose=False, use_numba=False)
    assert bm_a is bm_b, "cache should return the same BuiltMap on a hit"

    # Perturb a refinable field — must trigger a rebuild.
    s2 = spec_from_v1_params(p)
    s2.Lsd = torch.tensor(float(s2.Lsd) + 1.0, dtype=torch.float64)
    bm_c = cache.get(s2, auto_load=False, verbose=False, use_numba=False)
    assert bm_c is not bm_a
    assert bm_c.param_hash != bm_a.param_hash


def test_map_cache_lru_eviction():
    from midas_integrate_v2 import MapCache

    p = _v1()
    s1 = spec_from_v1_params(p)
    s2 = spec_from_v1_params(p)
    s2.Lsd = torch.tensor(float(s2.Lsd) + 1.0, dtype=torch.float64)
    s3 = spec_from_v1_params(p)
    s3.Lsd = torch.tensor(float(s3.Lsd) + 2.0, dtype=torch.float64)

    cache = MapCache(max_entries=1)
    cache.get(s1, auto_load=False, verbose=False, use_numba=False)
    cache.get(s2, auto_load=False, verbose=False, use_numba=False)
    # s1 should have been evicted.
    assert _spec_param_hash(s2) in cache._store
    assert _spec_param_hash(s1) not in cache._store
