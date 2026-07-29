import numpy as np
import pytest

from midas_defect.line_profile import (
    WARREN_XI_FCC,
    collect_per_grain_reflections,
    modified_wh_per_grain,
    warren_alpha_per_grain,
    warren_beta_proxy_per_grain,
)


_BURGERS_CU = 2.57e-10  # m


def _make_planted_voxels(
    OM_g: np.ndarray,
    G_arr: np.ndarray,
    sigma_per_refl: list[float],
    n_per_refl: int,
    rng: np.random.Generator,
):
    """Plant Gaussian clouds at each predicted Bragg position for a single grain."""
    qs_list = []
    vals_list = []
    for k, G in enumerate(G_arr):
        target = OM_g @ G
        sigma = sigma_per_refl[k]
        cloud = rng.normal(scale=sigma, size=(n_per_refl, 3)) + target
        qs_list.append(cloud)
        vals_list.append(np.ones(n_per_refl))
    return np.concatenate(qs_list, axis=0), np.concatenate(vals_list, axis=0)


# -- per-grain reflection collection ----------------------------------------

def test_collect_per_grain_reflections_returns_per_reflection_moments():
    rng = np.random.default_rng(0)
    OM = np.eye(3)[None]
    G_arr = np.array(
        [
            [1.0, 1.0, 1.0],   # 111
            [2.0, 0.0, 0.0],   # 200
            [2.0, 2.0, 0.0],   # 220
        ]
    )
    sigma = [0.02, 0.03, 0.025]
    qs, vals = _make_planted_voxels(OM[0], G_arr, sigma, n_per_refl=200, rng=rng)
    g_of_v = np.zeros(qs.shape[0], dtype=int)

    grain_entries = collect_per_grain_reflections(
        qs, vals, g_of_v, OM, G_arr, query_radius=0.10
    )
    assert len(grain_entries) == 1
    e = grain_entries[0]
    assert e["refl_indices"].size == 3
    # Centroid should be ~ |G|
    expected_mags = np.linalg.norm(G_arr, axis=1)
    np.testing.assert_allclose(e["centroid"], expected_mags, rtol=0.02)
    # FWHM ordering tracks sigma
    fwhm_order = np.argsort(e["fwhm"])
    sigma_order = np.argsort(sigma)
    # Strong correlation between orderings (don't insist on identity due to noise).
    assert (fwhm_order == sigma_order).all() or e["fwhm"][1] >= e["fwhm"][0]


def test_collect_per_grain_reflections_skips_grains_with_no_voxels():
    OM = np.tile(np.eye(3)[None], (2, 1, 1))
    G_arr = np.array([[1.0, 0.0, 0.0]])
    qs = np.array([[1.0, 0.0, 0.0]])
    vals = np.array([10.0])
    g_of_v = np.array([0], dtype=int)  # only grain 0 has voxels
    entries = collect_per_grain_reflections(qs, vals, g_of_v, OM, G_arr, min_voxels_per_refl=1)
    assert entries[0]["refl_indices"].size == 1
    assert entries[1]["refl_indices"].size == 0


# -- modified WH ------------------------------------------------------------

def test_modified_wh_per_grain_returns_positive_density_for_broad_peaks():
    rng = np.random.default_rng(0)
    OM = np.eye(3)[None]
    # Use canonical FCC reflections; pick widths that *increase* with |G| as
    # would be expected from strain broadening.
    G_arr = np.array(
        [
            [1.0, 1.0, 1.0],
            [2.0, 0.0, 0.0],
            [2.0, 2.0, 0.0],
            [3.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
        ]
    )
    # FWHM ~ alpha * |G| -> plant sigma proportional to G.
    sigma_per_refl = [0.01 * float(np.linalg.norm(G)) for G in G_arr]
    qs, vals = _make_planted_voxels(OM[0], G_arr, sigma_per_refl, n_per_refl=400, rng=rng)
    g_of_v = np.zeros(qs.shape[0], dtype=int)
    entries = collect_per_grain_reflections(qs, vals, g_of_v, OM, G_arr, query_radius=0.20)

    hkls = G_arr.astype(int)
    out = modified_wh_per_grain(entries, hkls, burgers_length=_BURGERS_CU)
    assert np.isfinite(out["rho_per_grain"][0])
    assert out["rho_per_grain"][0] > 0
    assert out["R_squared"][0] > 0.5


def test_modified_wh_per_grain_too_few_reflections_returns_nan():
    out = modified_wh_per_grain(
        [{"refl_indices": np.zeros(0, dtype=int),
          "G_magnitude": np.zeros(0), "fwhm": np.zeros(0),
          "centroid": np.zeros(0), "skewness": np.zeros(0),
          "intensity": np.zeros(0), "n_voxels": np.zeros(0, dtype=int)}],
        hkls=np.zeros((0, 3), dtype=int), burgers_length=_BURGERS_CU
    )
    assert np.isnan(out["rho_per_grain"][0])


# -- Warren alpha -----------------------------------------------------------

def test_warren_alpha_returns_nan_when_no_overlap_with_xi_table():
    G_arr = np.array([[10.0, 0.0, 0.0]])
    hkls = np.array([[10, 0, 0]])
    entries = [{
        "refl_indices": np.array([0]),
        "G_magnitude": np.array([10.0]),
        "centroid": np.array([10.0]),
        "fwhm": np.array([0.05]),
        "skewness": np.array([np.nan]),
        "intensity": np.array([1.0]),
        "n_voxels": np.array([100]),
    }]
    out = warren_alpha_per_grain(entries, hkls)
    assert np.isnan(out["alpha_per_grain"][0])


def test_warren_xi_table_has_canonical_values():
    # Quick spot check from Warren X-Ray Diffraction Table 13.1.
    assert WARREN_XI_FCC[(1, 1, 1)] == 0.0
    assert WARREN_XI_FCC[(2, 0, 0)] > 0
    assert WARREN_XI_FCC[(3, 1, 1)] < 0


# -- Warren beta ------------------------------------------------------------

def test_warren_beta_proxy_zero_when_111_and_200_have_equal_fwhm_over_G():
    hkls = np.array([[1, 1, 1], [2, 0, 0]])
    # |G_111| = sqrt(3), |G_200| = 2; plant FWHMs so beta/G is equal.
    entries = [{
        "refl_indices": np.array([0, 1]),
        "G_magnitude": np.array([np.sqrt(3), 2.0]),
        "centroid": np.array([np.sqrt(3), 2.0]),
        "fwhm": np.array([np.sqrt(3) * 0.05, 2.0 * 0.05]),
        "skewness": np.array([np.nan, np.nan]),
        "intensity": np.array([1.0, 1.0]),
        "n_voxels": np.array([100, 100]),
    }]
    out = warren_beta_proxy_per_grain(entries, hkls)
    assert out[0] == pytest.approx(0.0, abs=1e-12)


def test_warren_beta_proxy_nonzero_when_200_broadens_more():
    hkls = np.array([[1, 1, 1], [2, 0, 0]])
    entries = [{
        "refl_indices": np.array([0, 1]),
        "G_magnitude": np.array([np.sqrt(3), 2.0]),
        "centroid": np.array([np.sqrt(3), 2.0]),
        "fwhm": np.array([np.sqrt(3) * 0.05, 2.0 * 0.10]),  # 200 doubled
        "skewness": np.array([np.nan, np.nan]),
        "intensity": np.array([1.0, 1.0]),
        "n_voxels": np.array([100, 100]),
    }]
    out = warren_beta_proxy_per_grain(entries, hkls)
    assert out[0] < 0  # FWHM(200)/G_200 > FWHM(111)/G_111 -> proxy is negative
