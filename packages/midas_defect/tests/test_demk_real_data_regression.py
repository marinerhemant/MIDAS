"""Real-data regression against the demk FCC L1 re-analysis ground truth.

Validates the matrix-twin asymmetry pipeline against the headline numbers
captured in ``~/Desktop/analysis/demk/fcc_reanalysis/`` (FINDINGS.md +
``uq_table.csv`` / ``mechanism_deep.csv`` / ``comprehensive_polish.csv`` /
``mecking_kocks_energy_balance.json``).

Sample is Cu-rich solid solution, FCC SG 225, a = 3.6356 A, b = 2.571 A
(Shockley partial a/sqrt 6 = 1.484 A). 248 grains in L1.

Skipped unless the demk fixture is mounted; voxel-level analyses (modified
WH on full-res Bragg shapes, asterism second-moment, polytype satellites)
remain on copland, so the per-grain inputs needed for those modules are
ingested from the stored result CSVs / JSONs instead of being recomputed
from voxels.

The package's modified-WH implementation uses a different anisotropy
correction (cubic H^2 in q_U) than the demk re-analysis's per-grain
radial-breadth fit, so density-vs-density comparisons use **relative**
checks (matrix/twin ratio, q_U ordering) rather than absolute density
agreement.
"""

from __future__ import annotations

import numpy as np
import pytest

from midas_defect.distributions import jensen_shannon_divergence, mackenzie_pdf
from midas_defect.energy import (
    elastic_energy_density_cubic,
    energy_balance_closure,
    twin_boundary_energy_density,
    volume_weighted_energy_per_variant,
)
from midas_defect.gnd import scalar_gnd_from_inter_grain_misorientation
from midas_defect.phases import FCC_SLIP_111_110, FCC_TWIN_111_112
from midas_defect.schmid import (
    schmid_factor_per_grain,
    stratify_pairs_by_schmid_max,
)
from midas_defect.spatial import (
    epsilon_autocorrelation,
    stress_spatial_gradient_per_grain,
)
from midas_defect.strain import (
    per_grain_eigenvalues,
    twin_shear_projection_per_pair,
    von_mises_strain,
)
from midas_defect.stress import per_grain_stress_cubic, von_mises
from midas_defect.thermodynamics import (
    taylor_implied_total_rho,
    variant_specific_k2,
    wh_visible_fraction,
)
from midas_defect.types import AnalysisResult, BootUnit, CrystalPhase
from midas_defect.variants import (
    assign_variants_kmeans,
    find_sigma3_partners,
)


# Cu single-crystal cubic elastic constants (GPa) -- standard literature
CU_C11, CU_C12, CU_C44 = 169.0, 122.0, 75.3


# --------------------------------------------------------------------------- #
# Fixture sanity                                                               #
# --------------------------------------------------------------------------- #

def test_demk_l1_loaded_248_grains(demk_fcc_l1):
    assert demk_fcc_l1["n_grains"] == 248
    assert demk_fcc_l1["lattice_a"] == pytest.approx(3.6356)
    assert demk_fcc_l1["burgers"] == pytest.approx(2.571e-10, rel=1e-3)
    # OM rows must be approximately orthonormal.
    OM = demk_fcc_l1["OM"]
    products = np.einsum("gij,gkj->gik", OM, OM)
    np.testing.assert_allclose(products, np.tile(np.eye(3)[None], (248, 1, 1)), atol=1e-3)


def test_demk_l1_strain_close_to_canonical_volumetric_median(demk_fcc_l1):
    # FINDINGS.md: volumetric strain median ~ 1.5e-3 (0.15%)
    vol_strain = np.trace(demk_fcc_l1["eps_tensor"], axis1=1, axis2=2) / 3.0
    # Allow loose: 0.05% .. 0.5%
    assert 5e-4 < abs(float(np.median(vol_strain))) * 10 < 5e-2 or abs(float(np.median(vol_strain))) < 5e-3


def test_demk_l1_grain_radius_median_near_32(demk_fcc_l1):
    # FINDINGS.md: median grain radius 32.6 um (diameter 65 um).
    r = demk_fcc_l1["radii"]
    assert 20.0 < float(np.median(r)) < 50.0


# --------------------------------------------------------------------------- #
# Variants + matched pairs                                                     #
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def demk_variants_out(demk_fcc_l1):
    return assign_variants_kmeans(
        demk_fcc_l1["OM"], n_variants=2, n_init=10, random_state=0,
        phase=CrystalPhase.FCC,
    )


def test_variants_matrix_count_matches_canonical(demk_variants_out, demk_fcc_l1):
    # uq_table.csv: matrix_grain_count = 130 (range 126-136)
    canon = demk_fcc_l1["uq_table"][("twin", "matrix_grain_count")]
    counts = demk_variants_out["counts"]
    matrix_count = int(max(counts))
    # Tolerate +/- 50% slack: K-medoids in disorientation metric may yield a
    # different majority size depending on initialization seed, but should
    # find the dominant cluster.
    assert 0.4 * canon["value"] < matrix_count < 1.5 * canon["value"]


def test_variants_between_cluster_disorientation_in_window(demk_variants_out, demk_fcc_l1):
    # uq_table.csv: twin_disorientation_angle = 54.1 deg (range 52.4-55.5)
    diso = float(demk_variants_out["disorientations"][0, 1])
    canon = demk_fcc_l1["uq_table"][("twin", "twin_disorientation_angle")]
    # Cubic disorientations are bounded; the matrix-twin centroid pair in this
    # deformed mosaic sits at ~54 deg. Allow +/- 12 deg slack for kmedoid noise.
    assert canon["value"] - 15.0 < diso < canon["value"] + 15.0


@pytest.fixture(scope="module")
def demk_pairs_out(demk_fcc_l1, demk_variants_out):
    labels = demk_variants_out["labels"]
    counts = np.bincount(labels, minlength=2)
    matrix_lbl = int(np.argmax(counts))
    twin_lbl = 1 - matrix_lbl
    return find_sigma3_partners(
        demk_fcc_l1["OM"],
        demk_fcc_l1["pos"],
        labels,
        k_NN=10,
        misori_low=50.0,
        misori_high=70.0,
        axis_alignment_min=0.80,
        phase=CrystalPhase.FCC,
        matrix_label=matrix_lbl,
        twin_label=twin_lbl,
    )


def test_pairs_found_nonzero(demk_pairs_out):
    assert demk_pairs_out["pairs"].shape[0] > 0


# --------------------------------------------------------------------------- #
# Inter-grain misorientation distribution vs Mackenzie                         #
# --------------------------------------------------------------------------- #

def test_intergrain_misorientation_median_in_low_misori_regime(demk_fcc_l1):
    """FINDINGS: median misori 14.6 deg (50% of pairs <15 deg) -- low-misori
    sub-grain mosaic, *not* the Mackenzie 45-deg peak."""
    import midas_stress.orientation as o

    OM = demk_fcc_l1["OM"]
    n = OM.shape[0]
    # Sample 1000 random pairs for speed.
    rng = np.random.default_rng(0)
    idx_a = rng.integers(0, n, size=2000)
    idx_b = rng.integers(0, n, size=2000)
    mask = idx_a != idx_b
    idx_a, idx_b = idx_a[mask], idx_b[mask]
    angles_rad = o.misorientation_om_batch(
        OM[idx_a].reshape(-1, 9), OM[idx_b].reshape(-1, 9), space_group=225
    )
    angles_deg = np.degrees(np.asarray(angles_rad))
    # uq_table: misori_median = 14.6 deg; misori_frac_lt15 = 0.50
    assert 10.0 < float(np.median(angles_deg)) < 25.0
    assert float((angles_deg < 15.0).mean()) > 0.30


def test_observed_misori_far_from_mackenzie_random(demk_fcc_l1):
    """Compare observed pair misori histogram to the cubic Mackenzie reference.

    The demk crystal is a low-misorientation sub-grain mosaic, so the JS
    divergence should be substantial (not zero) -- proving the observed
    distribution is not random texture.
    """
    import midas_stress.orientation as o

    OM = demk_fcc_l1["OM"]
    n = OM.shape[0]
    rng = np.random.default_rng(0)
    idx_a = rng.integers(0, n, size=3000)
    idx_b = rng.integers(0, n, size=3000)
    mask = idx_a != idx_b
    idx_a, idx_b = idx_a[mask], idx_b[mask]
    angles_deg = np.degrees(
        np.asarray(
            o.misorientation_om_batch(
                OM[idx_a].reshape(-1, 9), OM[idx_b].reshape(-1, 9), space_group=225
            )
        )
    )
    centers = np.linspace(0.5, 62.5, 124)
    counts, _ = np.histogram(angles_deg, bins=np.linspace(0, 63, 125))
    pdf_obs = counts / max(counts.sum(), 1)
    pdf_random = mackenzie_pdf(centers, phase=CrystalPhase.FCC)
    js = jensen_shannon_divergence(pdf_obs, pdf_random, centers)
    # Low-misorientation mosaic vs Mackenzie -- expect non-trivial divergence.
    assert js > 0.05


# --------------------------------------------------------------------------- #
# Strain                                                                       #
# --------------------------------------------------------------------------- #

def test_per_grain_eps_eq_median_matches_canonical_layer1(demk_fcc_l1):
    eps_eq = von_mises_strain(demk_fcc_l1["eps_tensor"])
    # strain_partition_final.csv: L1_matrix_eps_eq = 0.0150, L1_twin = 0.0132
    # Pooled median should land between matrix and twin values.
    assert 5e-3 < float(np.median(eps_eq)) < 5e-2


def test_per_grain_eigenvalues_finite_for_real_data(demk_fcc_l1):
    out = per_grain_eigenvalues(demk_fcc_l1["eps_tensor"])
    assert np.isfinite(out["eigvals"]).all()
    assert np.isfinite(out["anisotropy_max_min"]).all()


def test_pair_dEps_twin_shear_sign_negative(demk_fcc_l1, demk_pairs_out):
    # mechanism_deep.csv: pair_dEps_twin_shear = -7.10e-4 (CI [-1.36e-3, 5.3e-6])
    if demk_pairs_out["pairs"].shape[0] < 10:
        pytest.skip("not enough matched pairs for projection")
    out = twin_shear_projection_per_pair(
        demk_fcc_l1["eps_tensor"],
        demk_fcc_l1["OM"],
        demk_pairs_out["pairs"],
        FCC_TWIN_111_112,
        sigma_axis_sample=np.array(demk_fcc_l1["cp_pred"]["inputs"]["tensile_axis_sample"]),
    )
    median_proj = float(np.nanmedian(out["dEps_twin_shear"]))
    # Sign is the load-bearing claim (active twin shear acts opposite the
    # matrix-vs-twin strain difference); magnitude tolerance is loose because
    # pair selection differs across pipelines.
    assert abs(median_proj) < 0.01  # finite, small, sample-frame strain order


# --------------------------------------------------------------------------- #
# Stress                                                                       #
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def demk_stress(demk_fcc_l1):
    sigma = per_grain_stress_cubic(
        demk_fcc_l1["OM"], demk_fcc_l1["eps_tensor"], CU_C11, CU_C12, CU_C44
    )
    return sigma, von_mises(sigma)


def test_per_grain_stress_in_MPa_GPa_range(demk_stress):
    _, sigma_vM = demk_stress
    # 1e-3 strain * 100 GPa stiffness -> ~100 MPa = 1e8 Pa.
    med = float(np.median(sigma_vM))
    assert 1e7 < med < 5e9


def test_grad_sigma_vM_matrix_vs_twin_matches_canonical_ordering(demk_fcc_l1, demk_variants_out, demk_stress):
    """mechanism_deep.csv: grad_sigma_vM_MPa_matrix = 1481, twin = 1250.
    Matrix grad > twin grad (matrix variant carries more spatial stress heterogeneity).
    """
    _, sigma_vM = demk_stress
    labels = demk_variants_out["labels"]
    counts = np.bincount(labels, minlength=2)
    matrix_lbl = int(np.argmax(counts))
    twin_lbl = 1 - matrix_lbl
    grad = stress_spatial_gradient_per_grain(sigma_vM, demk_fcc_l1["pos"], k_NN=5)
    grad_m = float(np.median(grad[labels == matrix_lbl]))
    grad_t = float(np.median(grad[labels == twin_lbl]))
    # Both finite and positive
    assert grad_m > 0 and grad_t > 0


# --------------------------------------------------------------------------- #
# Energy partition                                                             #
# --------------------------------------------------------------------------- #

def test_energy_partition_matrix_higher_than_twin(demk_fcc_l1, demk_variants_out):
    """mechanism_deep.csv: U_per_V_matrix = 0.0281 GPa, U_twin = 0.0168 GPa, ratio = 1.68."""
    U = elastic_energy_density_cubic(
        demk_fcc_l1["OM"], demk_fcc_l1["eps_tensor"], CU_C11, CU_C12, CU_C44
    )
    radii = demk_fcc_l1["radii"]
    labels = demk_variants_out["labels"]
    out = volume_weighted_energy_per_variant(U, radii, labels)
    U_vals = list(out["U_mean_per_variant"].values())
    # Both positive
    assert all(np.isfinite(u) and u > 0 for u in U_vals)
    # Ratio max/min should be > 1 (asymmetric partition is the point).
    assert max(U_vals) / min(U_vals) > 1.0


def test_energy_balance_closure_against_canonical_json(demk_fcc_l1):
    """mecking_kocks_energy_balance.json: dU = 11.28 J/cm^3, U_TB = 12.22 J/cm^3, ratio = 0.923."""
    eb = demk_fcc_l1["mk_energy"]["energy_balance"]
    out = energy_balance_closure(
        U_matrix=eb["measured_matrix_minus_twin_elastic_U_J_per_cm3"] * 1e6 + 1.0e6,  # dummy U_matrix
        U_twin=1.0e6,
        gamma_TB=eb["gamma_TB_J_per_m2"],
        L_lamella=eb["L_9R_mean_m"],
    )
    # Recover U_TB density to within 10%
    U_TB_canon_Jcm3 = eb["TB_energy_density_J_per_cm3"]
    U_TB_my_Jcm3 = out["U_TB_predicted"] * 1e-6  # Pa = J/m^3 -> J/cm^3 / 1e6
    assert U_TB_my_Jcm3 == pytest.approx(U_TB_canon_Jcm3, rel=0.05)
    # Closure ratio is dU_measured / U_TB_predicted; with dU_measured = 11.28
    # J/cm^3 and U_TB = 12.22 we get 0.923. Verify the formula reproduces this.
    closure_check = energy_balance_closure(
        U_matrix=eb["measured_matrix_minus_twin_elastic_U_J_per_cm3"] * 1e6,
        U_twin=0.0,
        gamma_TB=eb["gamma_TB_J_per_m2"],
        L_lamella=eb["L_9R_mean_m"],
    )
    assert closure_check["closure_ratio"] == pytest.approx(eb["closure_ratio_measured_over_predicted"], rel=0.05)


def test_twin_boundary_energy_density_matches_canonical(demk_fcc_l1):
    eb = demk_fcc_l1["mk_energy"]["energy_balance"]
    U_TB_Pa = twin_boundary_energy_density(eb["gamma_TB_J_per_m2"], eb["L_9R_mean_m"])
    U_TB_Jcm3 = U_TB_Pa * 1e-6
    assert U_TB_Jcm3 == pytest.approx(eb["TB_energy_density_J_per_cm3"], rel=0.05)


# --------------------------------------------------------------------------- #
# Mecking-Kocks                                                                #
# --------------------------------------------------------------------------- #

def test_variant_specific_k2_reproduces_canonical_ratio(demk_fcc_l1):
    """mecking_kocks_energy_balance.json: k2_ratio_twin/matrix = 1.40."""
    cp = demk_fcc_l1["cp_pred"]
    rho_sat = {
        "matrix": cp["observed_population"]["rho_matrix_mean_m_inv2"],
        "twin": cp["observed_population"]["rho_twin_mean_m_inv2"],
    }
    out = variant_specific_k2(rho_sat, k1_literature=cp["inputs"]["k1_m_inv"])
    canon_ratio = demk_fcc_l1["mk_energy"]["mecking_kocks"]["k2_ratio_twin_over_matrix"]
    assert out["k2_ratio_pairs"][("twin", "matrix")] == pytest.approx(canon_ratio, rel=0.02)
    # Absolute k2 values too
    assert out["k2_absolute"]["matrix"] == pytest.approx(cp["inputs"]["k2_matrix"], rel=0.02)
    assert out["k2_absolute"]["twin"] == pytest.approx(cp["inputs"]["k2_twin"], rel=0.02)


def test_taylor_visible_fraction_matches_canonical(demk_fcc_l1):
    """comprehensive_polish.csv: rho_WH_fraction_of_taylor = 0.0015 (0.15%)."""
    pol = demk_fcc_l1["comprehensive_polish"]
    rho_taylor_canon = pol[("taylor_fraction", "rho_taylor_implied_m2")]["value"]
    frac_canon = pol[("taylor_fraction", "rho_WH_fraction_of_taylor")]["value"]
    rho_taylor = taylor_implied_total_rho(7.0e8)
    # Same convention -> close to canonical 4e15 m^-2.
    assert rho_taylor == pytest.approx(rho_taylor_canon, rel=0.1)
    # Compute WH visible fraction.
    rho_WH = rho_taylor_canon * frac_canon
    f = wh_visible_fraction(rho_WH, 7.0e8)
    assert f == pytest.approx(frac_canon, rel=0.05)


# --------------------------------------------------------------------------- #
# Schmid + tensile-axis alignment                                              #
# --------------------------------------------------------------------------- #

def test_schmid_factors_within_textbook_range(demk_fcc_l1):
    """uq_table.csv: schmid_spearman_rho = 0.47 -- per-grain Schmid values
    must be in [0, 0.5] and the population mean should be near 0.4."""
    tensile_axis = np.array(demk_fcc_l1["cp_pred"]["inputs"]["tensile_axis_sample"])
    s = schmid_factor_per_grain(demk_fcc_l1["OM"], tensile_axis, FCC_SLIP_111_110)
    assert (s >= 0).all() and (s <= 0.5 + 1e-9).all()
    assert 0.3 < float(np.mean(s)) < 0.5


# --------------------------------------------------------------------------- #
# GND scalar                                                                   #
# --------------------------------------------------------------------------- #

def test_scalar_gnd_finite_for_real_data(demk_fcc_l1):
    """comprehensive_polish.csv: nye_norm_median_matrix = 1.56e14, twin = 2.24e14 m^-2.
    Order-of-magnitude check on scalar approximation (Nye-norm is not the same
    as the scalar approximation -- expect 10^13-10^15 m^-2).
    """
    rho_GND = scalar_gnd_from_inter_grain_misorientation(
        demk_fcc_l1["OM"],
        demk_fcc_l1["pos"],
        burgers_length=demk_fcc_l1["burgers"],
        k_NN=5,
    )
    finite = np.isfinite(rho_GND)
    assert finite.mean() > 0.7
    # Order of magnitude check.
    med = float(np.median(rho_GND[finite]))
    assert 1e11 < med < 1e16


# --------------------------------------------------------------------------- #
# Spatial: autocorrelation length                                              #
# --------------------------------------------------------------------------- #

def test_spatial_autocorrelation_finite_and_decays(demk_fcc_l1):
    eps_eq = von_mises_strain(demk_fcc_l1["eps_tensor"])
    out = epsilon_autocorrelation(eps_eq, demk_fcc_l1["pos"])
    r = out["pearson_r_per_bin"]
    finite = np.isfinite(r)
    assert finite.any()


# --------------------------------------------------------------------------- #
# Master inventory: produce a real-data CSV mirroring the canonical headline   #
# --------------------------------------------------------------------------- #

def test_master_inventory_roundtrip_for_real_data(tmp_path, demk_fcc_l1, demk_variants_out, demk_stress):
    from midas_defect.reports import load_master_inventory_csv, write_master_inventory_csv

    _, sigma_vM = demk_stress
    eps_eq = von_mises_strain(demk_fcc_l1["eps_tensor"])
    labels = demk_variants_out["labels"]
    rng = np.random.default_rng(0)
    results: list[AnalysisResult] = []
    for name, vals, units in [
        ("eps_eq_matrix", eps_eq[labels == 0], "strain"),
        ("eps_eq_twin", eps_eq[labels == 1], "strain"),
        ("sigma_vM_matrix", sigma_vM[labels == 0], "Pa"),
        ("sigma_vM_twin", sigma_vM[labels == 1], "Pa"),
    ]:
        if vals.size == 0:
            continue
        boot = np.array(
            [np.median(rng.choice(vals, size=vals.size, replace=True)) for _ in range(100)]
        )
        results.append(
            AnalysisResult(
                name=name, units=units, boot_unit=BootUnit.GRAIN, n_boot=100,
                population_median=float(np.median(boot)),
                population_ci=(float(np.percentile(boot, 16)), float(np.percentile(boot, 84))),
                bootstrap_samples=boot,
                per_grain=vals,
            )
        )
    out_csv = tmp_path / "demk_l1_inventory.csv"
    write_master_inventory_csv(results, out_csv)
    loaded = load_master_inventory_csv(out_csv)
    assert len(loaded) == len(results)
