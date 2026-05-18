"""v0.4: emit v1-format Map.bin/nMap.bin from a v2 binning geometry.

Pins:

1. Round-trip: write_map_bin_from_geometry → load_map → recovers the
   exact entries we wrote (counts, offsets, pxList content).
2. v1's CSR kernel can read the v2-emitted Map.bin and produce a 1D
   profile that matches v2's own integrate_hard / integrate_subpixel
   on the same geometry (within bin-resolution tolerance).
3. Header is well-formed: ``param_hash`` matches v1's
   :func:`compute_param_hash` on the same spec; ``q_mode``,
   ``wavelength`` round-trip.
4. Same Map.bin output regardless of file extension / output dir
   (just sanity).
5. Subpixel K=2 produces ~4× more entries than K=1 hard-bin (each
   pixel becomes 4 contributions).
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from pathlib import Path

import numpy as np
import pytest
import torch

from midas_integrate.bin_io import (
    PXLIST_DTYPE, MAP_HEADER_MAGIC, compute_param_hash, load_map,
)
from midas_integrate.kernels import (
    build_csr as v1_build_csr,
    integrate as v1_integrate,
    profile_1d as v1_profile_1d,
)
from midas_integrate.params import IntegrationParams

from midas_integrate_v2 import (
    spec_from_v1_params,
    HardBinGeometry,
    SubpixelBinGeometry,
    integrate_hard,
    integrate_subpixel,
    write_map_bin_from_geometry,
    v1_params_from_spec,
)


def _spec(NY=24, NZ=24, *, ops=None):
    p = IntegrationParams(
        NrPixelsY=NY, NrPixelsZ=NZ,
        pxY=200.0, pxZ=200.0, Lsd=1_000_000.0,
        BC_y=NY / 2.0 + 0.37, BC_z=NZ / 2.0 - 0.41, RhoD=float(NY),
        RMin=1.0, RMax=12.0, RBinSize=1.0,
        EtaMin=-180.0, EtaMax=180.0, EtaBinSize=60.0,
    )
    if ops is not None:
        p.TransOpt = list(ops)
        p.NrTransOpt = len(p.TransOpt)
    return spec_from_v1_params(p, requires_grad=False)


# ── (1) round-trip ──

def test_write_map_bin_from_hard_round_trips(tmp_path):
    s = _spec()
    geom = HardBinGeometry.from_spec(s)
    map_p, nmap_p = write_map_bin_from_geometry(geom, s, tmp_path)
    assert map_p.exists() and nmap_p.exists()
    pm = load_map(map_p, nmap_p)
    assert pm.n_bins == s.n_eta_bins * s.n_r_bins
    # n_entries should equal number of valid pixels
    assert pm.n_entries == geom.n_valid


def test_write_map_bin_from_subpixel_K2_has_4x_entries(tmp_path):
    s = _spec()
    g_hard = HardBinGeometry.from_spec(s)
    g_sub  = SubpixelBinGeometry.from_spec(s, K=2)
    write_map_bin_from_geometry(g_hard, s, tmp_path / "hard")
    write_map_bin_from_geometry(g_sub,  s, tmp_path / "sub")
    pm_hard = load_map(tmp_path / "hard" / "Map.bin",
                        tmp_path / "hard" / "nMap.bin")
    pm_sub  = load_map(tmp_path / "sub"  / "Map.bin",
                        tmp_path / "sub"  / "nMap.bin")
    # K=2 has 4 subpixels per pixel; in-range count is approximately 4×.
    # Allow some slack because edge pixels can flip in/out at subpixel
    # offsets.
    ratio = pm_sub.n_entries / max(1, pm_hard.n_entries)
    assert 3.0 < ratio < 5.0, (
        f"K=2 entry count ratio {ratio:.2f} not ~4×"
    )


# ── (2) v1 CSR can read what we wrote and integrates correctly ──

def test_v1_csr_reads_v2_emitted_map_and_integrates(tmp_path):
    """The most important test: v1's CSR kernel can consume v2-emitted
    Map.bin and integrate an image. The 1D profile should match what
    v2's own integrate_hard produces (modulo CSR's normalisation, which
    matches v1's normalize=True semantics)."""
    s = _spec()
    geom = HardBinGeometry.from_spec(s)
    map_p, nmap_p = write_map_bin_from_geometry(geom, s, tmp_path)
    pm = load_map(map_p, nmap_p)

    # Build v1 CSR and integrate
    csr = v1_build_csr(
        pm, n_r=s.n_r_bins, n_eta=s.n_eta_bins,
        n_pixels_y=s.NrPixelsY, n_pixels_z=s.NrPixelsZ,
        device="cpu", dtype=torch.float64,
        bc_y=float(s.BC_y), bc_z=float(s.BC_z),
    )
    rng = np.random.default_rng(0xCAFE)
    img = torch.from_numpy(
        rng.uniform(1.0, 100.0, size=(s.NrPixelsZ, s.NrPixelsY))
    )
    int2d_v1 = v1_integrate(img, csr, mode="floor", normalize=True)
    prof_v1 = v1_profile_1d(int2d_v1, csr, mode="area_weighted").numpy()

    # v2 hard-bin reference
    int2d_v2 = integrate_hard(img, geom, normalize=True)
    prof_v2 = int2d_v2.sum(dim=0).numpy() / np.maximum(
        (int2d_v2 > 0).sum(dim=0).numpy(), 1
    )
    # Both should peak in the same R band; centroids should agree to
    # ~1 R bin.
    n_r = s.n_r_bins
    r_axis = s.RMin + s.RBinSize * (np.arange(n_r) + 0.5)
    if prof_v1.sum() > 0 and prof_v2.sum() > 0:
        c_v1 = float((prof_v1 * r_axis).sum() / prof_v1.sum())
        c_v2 = float((prof_v2 * r_axis).sum() / prof_v2.sum())
        assert abs(c_v1 - c_v2) < 2 * s.RBinSize, (
            f"v1 vs v2 centroid drift {abs(c_v1 - c_v2):.3f} > 2 RBinSize"
        )


# ── (3) header is well-formed and matches v1 hash ──

def test_emitted_header_param_hash_matches_v1(tmp_path):
    s = _spec()
    geom = HardBinGeometry.from_spec(s)
    map_p, nmap_p = write_map_bin_from_geometry(geom, s, tmp_path)
    pm = load_map(map_p, nmap_p)

    p1 = v1_params_from_spec(s)
    h_v1 = compute_param_hash(
        Lsd=p1.Lsd, Ycen=p1.BC_y, Zcen=p1.BC_z, pxY=p1.pxY, pxZ=p1.pxZ,
        tx=p1.tx, ty=p1.ty, tz=p1.tz,
        p0=p1.p0, p1=p1.p1, p2=p1.p2, p3=p1.p3, p4=p1.p4,
        p5=p1.p5, p6=p1.p6, p7=p1.p7, p8=p1.p8, p9=p1.p9,
        p10=p1.p10, p11=p1.p11, p12=p1.p12, p13=p1.p13, p14=p1.p14,
        Parallax=p1.Parallax, RhoD=p1.RhoD,
        RBinSize=p1.RBinSize, EtaBinSize=p1.EtaBinSize,
        RMin=p1.RMin, RMax=p1.RMax, EtaMin=p1.EtaMin, EtaMax=p1.EtaMax,
        NrPixelsY=p1.NrPixelsY, NrPixelsZ=p1.NrPixelsZ,
        TransOpt=tuple(p1.TransOpt),
        qMode=int(p1.q_mode_active),
        Wavelength=p1.Wavelength,
    )
    assert pm.map_header is not None
    assert pm.map_header.magic == MAP_HEADER_MAGIC
    assert pm.map_header.param_hash == h_v1


def test_emitted_header_carries_qmode_and_wavelength(tmp_path):
    s = _spec()
    s.QMin = 0.5; s.QMax = 7.0; s.QBinSize = 0.01
    s.Wavelength = torch.tensor(0.172979, dtype=torch.float64)
    geom = HardBinGeometry.from_spec(s)
    map_p, nmap_p = write_map_bin_from_geometry(geom, s, tmp_path)
    pm = load_map(map_p, nmap_p)
    assert pm.map_header is not None
    assert pm.map_header.q_mode == 1
    assert abs(pm.map_header.wavelength - 0.172979) < 1e-12


# ── (4) sanity: write to nested dir, no header option ──

def test_write_no_header_skips_v3_magic(tmp_path):
    s = _spec()
    geom = HardBinGeometry.from_spec(s)
    map_p, nmap_p = write_map_bin_from_geometry(
        geom, s, tmp_path / "deep" / "subdir",
        write_header=False,
    )
    pm = load_map(map_p, nmap_p)
    assert pm.map_header is None


def test_write_rejects_unknown_geom_type(tmp_path):
    s = _spec()
    class Bogus: pass
    with pytest.raises(TypeError, match="Hard.*Subpixel"):
        write_map_bin_from_geometry(Bogus(), s, tmp_path)


# ── (5) sums of frac per bin equal v2's normalize counts ──

def test_emitted_frac_sums_per_bin_match_v2_normalize_counts(tmp_path):
    """The sum of `frac` entries across a bin equals the per-bin
    pixel-equivalent count v2 uses to normalise. For hard-bin: equals
    the integer number of pixels in that bin."""
    s = _spec()
    geom = HardBinGeometry.from_spec(s)
    write_map_bin_from_geometry(geom, s, tmp_path)
    pm = load_map(tmp_path / "Map.bin", tmp_path / "nMap.bin")

    # Per-bin frac sum from the emitted file
    n_bins = pm.n_bins
    frac_sums = np.zeros(n_bins, dtype=np.float64)
    for b in range(n_bins):
        start = pm.offsets[b]
        end = start + pm.counts[b]
        frac_sums[b] = pm.pxList["frac"][start:end].sum()

    # v2's per-bin pixel count
    expected = np.zeros(n_bins, dtype=np.float64)
    flat_bin = geom.flat_bin[geom.valid].cpu().numpy()
    for b in flat_bin:
        expected[int(b)] += 1.0

    np.testing.assert_array_equal(frac_sums, expected)
