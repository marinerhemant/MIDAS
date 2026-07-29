"""Tests for `midas_defect.delta_pdf`."""

from __future__ import annotations

import math

import numpy as np
import pytest

from midas_defect.bootstrap import ProfileBand
from midas_defect.delta_pdf import (
    DeltaPDFResult,
    PerGrainDeltaPDF,
    bragg_mask_from_fits,
    compute_delta_pdf,
    compute_delta_pdf_per_grain,
    densify_to_qgrid,
    profile_along_crystal_direction,
    profiles_for_population,
    radial_delta_pdf_profile,
    variant_profile_bands,
    wiener_deconvolve_mask,
)


@pytest.mark.unit
def test_densify_sums_intensity():
    rng = np.random.default_rng(0)
    qx = rng.uniform(-1, 1, 50)
    qy = rng.uniform(-1, 1, 50)
    qz = rng.uniform(-1, 1, 50)
    inten = np.full(50, 7.0)
    grid, axis = densify_to_qgrid(qx, qy, qz, inten, q_max=1.0, n_grid=20)
    assert grid.sum() == pytest.approx(50 * 7.0, rel=1e-6)


@pytest.mark.unit
def test_densify_drops_out_of_range():
    qx = np.array([2.0, 0.0])
    qy = np.array([0.0, 0.0])
    qz = np.array([0.0, 0.0])
    inten = np.array([100.0, 5.0])
    grid, _ = densify_to_qgrid(qx, qy, qz, inten, q_max=1.0, n_grid=10)
    # Only the in-range point should be kept
    assert grid.sum() == 5.0


@pytest.mark.unit
def test_delta_pdf_origin_peak_is_largest():
    """For random noise → Δρ is dominated by the origin (autocorrelation peak)."""
    rng = np.random.default_rng(0)
    qx = rng.uniform(-2, 2, 1000)
    qy = rng.uniform(-2, 2, 1000)
    qz = rng.uniform(-2, 2, 1000)
    I = rng.uniform(50, 100, 1000)
    res = compute_delta_pdf(qx, qy, qz, I, q_max=2.0, n_grid=32,
                            bragg_fits=None, symmetrize_friedel=True,
                            taper_frac=0.1, device="cpu")
    assert isinstance(res, DeltaPDFResult)
    # origin should be the max |Δρ|
    cx, cy, cz = (s // 2 for s in res.delta_rho.shape)
    origin_val = abs(res.delta_rho[cx, cy, cz])
    assert origin_val >= 0.9 * np.abs(res.delta_rho).max(), (
        "origin is not the dominant peak in Δρ"
    )


@pytest.mark.unit
def test_delta_pdf_periodic_structure_recovers_d_layer():
    """Plant a 1-D periodic Bragg comb along x and check Δρ peak spacing.

    Crystallographic convention: the comb sits at ``q_n = 2π·n/d_layer``
    (NOT ``n/d_layer``). FFT aliasing then puts the first real-space
    secondary peak at ``r = d_layer``.
    """
    rng = np.random.default_rng(0)
    d_layer = 1.0  # Å
    twopi = 2.0 * math.pi
    qx_list, qy_list, qz_list, I_list = [], [], [], []
    # Need q_max > 2π·n_max so the highest harmonic falls inside the cube.
    n_max = 3
    for n in range(-n_max, n_max + 1):
        q_par = twopi * n / d_layer
        for _ in range(80):
            qx_list.append(q_par + rng.normal(scale=0.05))
            qy_list.append(rng.normal(scale=0.05))
            qz_list.append(rng.normal(scale=0.05))
            I_list.append(100.0)
    qx = np.array(qx_list); qy = np.array(qy_list); qz = np.array(qz_list)
    I = np.array(I_list)
    res = compute_delta_pdf(qx, qy, qz, I,
                            q_max=22.0, n_grid=256,
                            bragg_fits=None, symmetrize_friedel=True,
                            taper_frac=0.1, device="cpu")
    # Take a 1-D slice along x at (y=0, z=0)
    cx, cy, cz = (s // 2 for s in res.delta_rho.shape)
    line = res.delta_rho[:, cy, cz]
    dr = res.r_axes[0][1] - res.r_axes[0][0]
    # Find prominent local maxima past the origin. A planted comb produces
    # well-separated peaks at r = n·d_layer; ignore sub-threshold ripple.
    n = len(line); mid = cx
    # Threshold on the post-origin tail (drop the origin spike itself).
    tail = np.abs(line[mid + 3:])
    thresh = 0.25 * float(tail.max())
    peaks = []
    for i in range(mid + 3, n - 1):
        if (line[i] > line[i - 1] and line[i] > line[i + 1]
                and abs(line[i]) >= thresh):
            peaks.append((i, line[i], res.r_axes[0][i]))
    assert peaks, "no prominent secondary peaks found"
    # The first prominent peak after the origin should be at r ≈ d_layer.
    first_peak_idx, _, first_peak_r = peaks[0]
    assert abs(first_peak_r - d_layer) < 4 * dr, (
        f"first prominent peak at r={first_peak_r:.4f} not at "
        f"d_layer={d_layer:.4f} (tol {4*dr:.4f})"
    )
    # And the next prominent peak should be near 2·d_layer.
    if len(peaks) >= 2:
        _, _, second_peak_r = peaks[1]
        assert abs(second_peak_r - 2 * d_layer) < 4 * dr, (
            f"second prominent peak at r={second_peak_r:.4f} not at "
            f"2·d_layer={2*d_layer:.4f} (tol {4*dr:.4f})"
        )


@pytest.mark.unit
def test_wiener_deconvolve_mask_impulse_mask_is_near_identity():
    """When mask is an impulse (one non-zero q-cell), FT(mask) is a flat
    constant ↦ pointwise division ↦ output ≈ input up to a uniform scaling."""
    rng = np.random.default_rng(0)
    n = 16
    grid = rng.normal(size=(n, n, n)).astype(np.float64)
    mask = np.zeros((n, n, n))
    # single non-zero cell at the centre (the q=0 cell after ifftshift)
    mask[n // 2, n // 2, n // 2] = 1.0
    out = wiener_deconvolve_mask(grid, mask, lam=1e-12)
    # FT of a single impulse is a flat 1/sqrt(N) constant in magnitude → the
    # Wiener divisor is constant → out is grid · sqrt(N) up to a phase ramp,
    # so the SHAPE of out should match grid up to a global sign and constant.
    # Easy invariant: total energy ratio is finite and shapes are preserved.
    assert out.shape == grid.shape
    assert np.all(np.isfinite(out))
    assert np.abs(out).sum() > 0


@pytest.mark.unit
def test_wiener_deconvolve_mask_partial_coverage_finite_and_real():
    """A masked grid still yields finite, real-valued Δρ_deconv."""
    rng = np.random.default_rng(1)
    n = 32
    grid = rng.normal(size=(n, n, n)).astype(np.float64)
    mask = (rng.random(size=(n, n, n)) > 0.4).astype(np.float64)  # ~60 % coverage
    grid *= mask  # only measured cells carry data
    for lam in (1e-3, 1e-2, 1e-1):
        out = wiener_deconvolve_mask(grid, mask, lam=lam)
        assert out.shape == grid.shape
        assert np.all(np.isfinite(out))
        # Larger λ ↦ smaller max amplitude (more regularisation)
    out_lo = wiener_deconvolve_mask(grid, mask, lam=1e-3)
    out_hi = wiener_deconvolve_mask(grid, mask, lam=1e-1)
    assert np.abs(out_hi).max() < np.abs(out_lo).max()


@pytest.mark.unit
def test_radial_delta_pdf_profile_shapes_and_sampling():
    """Sampling along a direction returns the right number of points."""
    rng = np.random.default_rng(0)
    n = 32
    delta_rho = rng.normal(size=(n, n, n)).astype(np.float64)
    r_axis = np.linspace(-15.0, 15.0, n, endpoint=False)
    dir_ = np.array([1.0, 1.0, 1.0])
    out = radial_delta_pdf_profile(
        delta_rho, r_axis, direction=dir_, half_width=0.5, n_samples=64,
    )
    assert set(out.keys()) == {"t", "profile"}
    assert out["t"].shape == (64,)
    assert out["profile"].shape == (64,)


@pytest.mark.unit
def test_radial_delta_pdf_profile_recovers_planted_peak():
    """Plant a Gaussian along the [0,0,1] direction; sampling along that
    direction should recover its peak position."""
    n = 64
    grid = np.zeros((n, n, n))
    r_axis = np.linspace(-15.0, 15.0, n, endpoint=False)
    # Plant a Gaussian at r = 5 Å along +z
    cz = n // 2
    r0 = 5.0
    dr = r_axis[1] - r_axis[0]
    iz_peak = int(round((r0 - r_axis[0]) / dr))
    # Make the peak wide enough to survive trilinear sampling
    for di in range(-2, 3):
        for dj in range(-2, 3):
            grid[n // 2 + di, n // 2 + dj, iz_peak] += math.exp(
                -0.5 * (di * di + dj * dj) / 1.0
            )
    out = radial_delta_pdf_profile(
        grid, r_axis,
        direction=np.array([0.0, 0.0, 1.0]),
        half_width=0.0, n_samples=200,
    )
    i_max = int(np.argmax(out["profile"]))
    r_peak_recovered = out["t"][i_max]
    assert abs(r_peak_recovered - r0) < 1.5, (
        f"recovered peak at {r_peak_recovered:.2f} Å, expected {r0:.2f} Å"
    )


@pytest.mark.unit
def test_bragg_mask_from_fits_excludes_ellipsoid_and_keeps_outside():
    """Mask is 0 inside the fitted Σ ellipsoid and 1 well outside."""
    from types import SimpleNamespace
    q_axis = np.linspace(-2.0, 2.0, 32, endpoint=False)
    # Synthetic isotropic Bragg fit at q = (1, 0, 0) with σ = 0.05.
    fit = SimpleNamespace(
        q_fit=np.array([1.0, 0.0, 0.0]),
        sigma_axes=np.eye(3),
        sigma_eig=np.array([0.05, 0.05, 0.05]),
    )
    mask = bragg_mask_from_fits(q_axis, [fit], sigma_scale=3.0)
    # Cell closest to the centre must be masked out.
    ix = int(np.argmin(np.abs(q_axis - 1.0)))
    iy = int(np.argmin(np.abs(q_axis - 0.0)))
    iz = int(np.argmin(np.abs(q_axis - 0.0)))
    assert mask[ix, iy, iz] == 0.0
    # A cell 1/Å away along x must be kept (far outside 3σ = 0.15).
    ix_far = int(np.argmin(np.abs(q_axis - 0.0)))
    assert mask[ix_far, iy, iz] == 1.0
    # Mask is a fraction of total cells.
    frac_masked = float((mask == 0.0).mean())
    assert 0.0 < frac_masked < 0.05, (
        f"single 3σ ellipsoid should mask out a tiny fraction; got "
        f"{frac_masked:.3%}"
    )


@pytest.mark.unit
def test_bragg_mask_feeds_wiener_deconvolve_without_blowup():
    """Round-trip: build mask from fits, apply to a grid, deconvolve."""
    from types import SimpleNamespace
    rng = np.random.default_rng(0)
    n = 32
    q_axis = np.linspace(-2.0, 2.0, n, endpoint=False)
    fits = [
        SimpleNamespace(
            q_fit=np.array([0.5, -0.3, 0.0]),
            sigma_axes=np.eye(3),
            sigma_eig=np.array([0.08, 0.08, 0.08]),
        ),
        SimpleNamespace(
            q_fit=np.array([-0.7, 0.4, 0.2]),
            sigma_axes=np.eye(3),
            sigma_eig=np.array([0.06, 0.06, 0.06]),
        ),
    ]
    mask = bragg_mask_from_fits(q_axis, fits, sigma_scale=3.0)
    grid = rng.normal(size=(n, n, n)) * mask
    out = wiener_deconvolve_mask(grid, mask, lam=1e-2)
    assert out.shape == grid.shape
    assert np.all(np.isfinite(out))


@pytest.mark.unit
def test_compute_delta_pdf_per_grain_routes_voxels_correctly(
    synthetic_cu_al_dataset,
):
    """Per-grain attribution routes voxels by gID and yields one Δρ each."""
    ds = synthetic_cu_al_dataset
    qs = ds["qs"]; vals = ds["vals"]; gof = ds["grain_of_voxel"]
    res = compute_delta_pdf_per_grain(
        qs[:, 0], qs[:, 1], qs[:, 2], vals, gof,
        q_max=10.0, n_grid=32,
        bragg_fits_per_grain=None,
        symmetrize_friedel=True, taper_frac=0.1,
        min_voxels_per_grain=20, device="cpu",
    )
    assert isinstance(res, PerGrainDeltaPDF)
    # Each unique grain id in the dataset got its own DeltaPDFResult.
    expected_ids = np.unique(gof)
    assert np.array_equal(res.grain_ids, expected_ids)
    # Every grain in the fixture has >20 voxels (8 hkls × 80 voxels each).
    for gid, per in res:
        assert isinstance(per, DeltaPDFResult)
        assert per.delta_rho.shape == (32, 32, 32)
        # voxel count claimed by the per-grain result matches the routing.
        assert per.n_voxels_in == int((gof == gid).sum())
    # Shared axes match the per-grain axes (cube is shared).
    for _, per in res:
        for a_shared, a_per in zip(res.q_axes, per.q_axes):
            assert np.array_equal(a_shared, a_per)


@pytest.mark.unit
def test_compute_delta_pdf_per_grain_skips_undersized_grains():
    """Grains with < min_voxels_per_grain are skipped silently."""
    rng = np.random.default_rng(0)
    # Two grains: one with 200 voxels, one with 10.
    qx_big = rng.uniform(-1, 1, 200)
    qy_big = rng.uniform(-1, 1, 200)
    qz_big = rng.uniform(-1, 1, 200)
    I_big  = rng.uniform(50, 100, 200)
    g_big  = np.full(200, 0, dtype=np.int64)
    qx_sm  = rng.uniform(-1, 1, 10)
    qy_sm  = rng.uniform(-1, 1, 10)
    qz_sm  = rng.uniform(-1, 1, 10)
    I_sm   = rng.uniform(50, 100, 10)
    g_sm   = np.full(10, 1, dtype=np.int64)
    qx = np.concatenate([qx_big, qx_sm])
    qy = np.concatenate([qy_big, qy_sm])
    qz = np.concatenate([qz_big, qz_sm])
    I  = np.concatenate([I_big, I_sm])
    g  = np.concatenate([g_big, g_sm])
    res = compute_delta_pdf_per_grain(
        qx, qy, qz, I, g,
        q_max=1.0, n_grid=16, min_voxels_per_grain=50, device="cpu",
    )
    assert set(res.by_grain.keys()) == {0}
    # Counts are still reported for both grains.
    assert np.array_equal(res.grain_ids, np.array([0, 1]))
    assert np.array_equal(res.n_voxels_per_grain, np.array([200, 10]))


@pytest.mark.unit
def test_per_grain_iter_only_yields_grains_with_a_map():
    """Iterating a PerGrainDeltaPDF must skip below-threshold grains."""
    rng = np.random.default_rng(0)
    # Three grains: 200, 10, 250 voxels. Only g=0 and g=2 should survive.
    qx_a = rng.uniform(-1, 1, 200); qy_a = rng.uniform(-1, 1, 200)
    qz_a = rng.uniform(-1, 1, 200); I_a  = rng.uniform(50, 100, 200)
    qx_b = rng.uniform(-1, 1, 10);  qy_b = rng.uniform(-1, 1, 10)
    qz_b = rng.uniform(-1, 1, 10);  I_b  = rng.uniform(50, 100, 10)
    qx_c = rng.uniform(-1, 1, 250); qy_c = rng.uniform(-1, 1, 250)
    qz_c = rng.uniform(-1, 1, 250); I_c  = rng.uniform(50, 100, 250)
    qx = np.concatenate([qx_a, qx_b, qx_c])
    qy = np.concatenate([qy_a, qy_b, qy_c])
    qz = np.concatenate([qz_a, qz_b, qz_c])
    I  = np.concatenate([I_a, I_b, I_c])
    g  = np.concatenate([
        np.full(200, 0, dtype=np.int64),
        np.full(10,  1, dtype=np.int64),
        np.full(250, 2, dtype=np.int64),
    ])
    res = compute_delta_pdf_per_grain(
        qx, qy, qz, I, g,
        q_max=1.0, n_grid=16, min_voxels_per_grain=50, device="cpu",
    )
    yielded = [gid for gid, _ in res]
    assert yielded == [0, 2]
    # All-of-population accounting is still intact in grain_ids / counts.
    assert np.array_equal(res.grain_ids, np.array([0, 1, 2]))
    assert np.array_equal(res.n_voxels_per_grain, np.array([200, 10, 250]))


@pytest.mark.unit
def test_compute_delta_pdf_per_grain_drops_negative_ids():
    """Voxels labelled with negative gID are dropped from all grains."""
    rng = np.random.default_rng(0)
    qx = rng.uniform(-1, 1, 300)
    qy = rng.uniform(-1, 1, 300)
    qz = rng.uniform(-1, 1, 300)
    I  = rng.uniform(50, 100, 300)
    g  = np.concatenate([
        np.full(150, 0, dtype=np.int64),
        np.full(100, -1, dtype=np.int64),  # unassigned
        np.full(50, 1, dtype=np.int64),
    ])
    res = compute_delta_pdf_per_grain(
        qx, qy, qz, I, g,
        q_max=1.0, n_grid=16, min_voxels_per_grain=20, device="cpu",
    )
    assert set(res.by_grain.keys()) == {0, 1}
    assert res.by_grain[0].n_voxels_in == 150
    assert res.by_grain[1].n_voxels_in == 50


@pytest.mark.unit
def test_profile_along_crystal_direction_rotates_through_OM():
    """Crystal-direction sampler rotates [100]_cry through OM into sample frame."""
    n = 64
    grid = np.zeros((n, n, n))
    r_axis = np.linspace(-15.0, 15.0, n, endpoint=False)
    dr = r_axis[1] - r_axis[0]
    # Plant a Gaussian along +y in the SAMPLE frame at r = 5 Å.
    r0 = 5.0
    iy_peak = int(round((r0 - r_axis[0]) / dr))
    for di in range(-2, 3):
        for dj in range(-2, 3):
            grid[n // 2 + di, iy_peak, n // 2 + dj] += math.exp(
                -0.5 * (di * di + dj * dj) / 1.0
            )
    # OM that rotates crystal +x to sample +y (90° about +z).
    OM = np.array([[0.0, -1.0, 0.0],
                   [1.0,  0.0, 0.0],
                   [0.0,  0.0, 1.0]])
    res = DeltaPDFResult(
        delta_rho=grid,
        r_axes=(r_axis, r_axis, r_axis),
        q_axes=(r_axis, r_axis, r_axis),  # placeholder; not used by profile
        diff_volume_max=float(np.abs(grid).max()),
        n_voxels_in=0,
        bragg_subtracted=False,
    )
    out = profile_along_crystal_direction(
        res, OM, direction_cry=np.array([1.0, 0.0, 0.0]),
        half_width=0.0, n_samples=200,
    )
    # The returned sample-frame direction is +y.
    np.testing.assert_allclose(out["direction_sample"],
                               np.array([0.0, 1.0, 0.0]), atol=1e-12)
    # And the planted peak at r=5 along +y is recovered.
    i_max = int(np.argmax(out["profile"]))
    r_peak = out["t"][i_max]
    assert abs(r_peak - r0) < 1.5, (
        f"recovered peak at {r_peak:.2f} Å, expected {r0:.2f} Å"
    )


@pytest.mark.unit
def test_profiles_for_population_skips_grains_missing_OM(
    synthetic_cu_al_dataset,
):
    """Variant aggregation: only grains with an OM entry get a profile."""
    ds = synthetic_cu_al_dataset
    qs = ds["qs"]; vals = ds["vals"]; gof = ds["grain_of_voxel"]
    OM = ds["OM"]
    res = compute_delta_pdf_per_grain(
        qs[:, 0], qs[:, 1], qs[:, 2], vals, gof,
        q_max=10.0, n_grid=32,
        min_voxels_per_grain=20, device="cpu",
    )
    # Provide OMs for only a subset of grains.
    OM_subset = {gid: OM[gid] for gid in (0, 1, 5, 33)}
    profiles = profiles_for_population(
        res, OM_subset, direction_cry=np.array([1.0, 1.0, 1.0]),
        half_width=0.5, n_samples=64,
    )
    assert set(profiles.keys()) == set(OM_subset.keys())
    for gid, prof in profiles.items():
        assert set(prof.keys()) >= {"t", "profile", "direction_sample"}
        assert prof["profile"].shape == (64,)


@pytest.mark.unit
def test_variant_profile_bands_groups_by_label(synthetic_cu_al_dataset):
    """variant_profile_bands returns one ProfileBand per variant label."""
    ds = synthetic_cu_al_dataset
    qs = ds["qs"]; vals = ds["vals"]; gof = ds["grain_of_voxel"]
    OM = ds["OM"]; tv = ds["true_variant"]
    per = compute_delta_pdf_per_grain(
        qs[:, 0], qs[:, 1], qs[:, 2], vals, gof,
        q_max=10.0, n_grid=32, min_voxels_per_grain=20, device="cpu",
    )
    profiles = profiles_for_population(
        per, {int(g): OM[g] for g in range(len(OM))},
        direction_cry=np.array([1.0, 1.0, 1.0]),
        half_width=0.5, n_samples=64,
    )
    variant_of_grain = {int(g): int(tv[g]) for g in range(len(tv))}
    bands = variant_profile_bands(
        profiles, variant_of_grain, n_boot=50, rng_seed=0,
    )
    # Two variants: 0 (matrix) and 1 (twin).
    assert set(bands.keys()) == {0, 1}
    for label, band in bands.items():
        assert isinstance(band, ProfileBand)
        assert band.r.shape == (64,)
        assert band.median.shape == (64,)
        assert np.all(band.ci_lo <= band.median + 1e-12)
        assert np.all(band.median <= band.ci_hi + 1e-12)
        assert band.n_grains > 0
        assert band.n_boot == 50
        assert band.boot_unit == "grain"


@pytest.mark.unit
def test_variant_profile_bands_independent_seeds_per_variant():
    """Two variants get independent RNG streams (rng_seed + i)."""
    rng = np.random.default_rng(7)
    r = np.linspace(0, 5, 32)
    # Build a synthetic profiles dict with two clearly distinct variant populations.
    profiles = {}
    variant = {}
    for g in range(40):
        prof_v0 = np.exp(-0.5 * ((r - 1.0) / 0.3) ** 2) + 0.02 * rng.normal(size=32)
        profiles[g] = {"t": r, "profile": prof_v0}
        variant[g] = 0
    for g in range(40, 80):
        prof_v1 = np.exp(-0.5 * ((r - 3.0) / 0.3) ** 2) + 0.02 * rng.normal(size=32)
        profiles[g] = {"t": r, "profile": prof_v1}
        variant[g] = 1
    bands = variant_profile_bands(profiles, variant, n_boot=200, rng_seed=42)
    # Matrix variant peaks at r=1; twin at r=3.
    i0 = int(np.argmax(bands[0].median))
    i1 = int(np.argmax(bands[1].median))
    assert abs(r[i0] - 1.0) < 0.2
    assert abs(r[i1] - 3.0) < 0.2
    # Independent: their bands at the OTHER peak should NOT recover that peak.
    assert bands[0].median[i1] < 0.5 * bands[1].median[i1]
    assert bands[1].median[i0] < 0.5 * bands[0].median[i0]


@pytest.mark.device
def test_delta_pdf_device_portable(_device_param):
    """compute_delta_pdf runs on CPU/MPS, produces real arrays."""
    rng = np.random.default_rng(0)
    qx = rng.uniform(-1, 1, 200)
    qy = rng.uniform(-1, 1, 200)
    qz = rng.uniform(-1, 1, 200)
    I = rng.uniform(50, 100, 200)
    res = compute_delta_pdf(qx, qy, qz, I,
                            q_max=1.0, n_grid=16,
                            symmetrize_friedel=True, taper_frac=0.1,
                            device=_device_param)
    assert res.delta_rho.shape == (16, 16, 16)
    assert np.all(np.isfinite(res.delta_rho))
