"""End-to-end synthetic Cu-Al-like regression.

Drives every Phase 0-3 module against a single shared fixture
``synthetic_cu_al_dataset`` (see conftest). The intent is a tight CI gate
against module regressions, not a paper-grade numerical reproduction --
property-style assertions only (monotone, sign, plausible range).

The pipeline mirrors the structure of the matrix-twin asymmetry paper:
    1. Variants     : K-medoids -> matrix vs twin labels
    2. Pairs        : matched Σ3 partners
    3. Schmid       : per-grain Schmid factor, tercile stratification
    4. Strain       : per-pair twin-shear projection
    5. Stress       : cubic per-grain stress + invariants
    6. Energy       : per-grain elastic energy, volume-weighted, closure
    7. Line profile : per-grain modified-WH dislocation density
    8. Debye-Waller : per-grain Wilson B
    9. Asterism     : second moment + edge fraction
   10. GND          : Nye-tensor rho_GND + SSD split
   11. Distributions: Mackenzie, Friedel
   12. Spatial      : autocorrelation, stress gradient
   13. Thermodynamics: MK + Taylor
   14. Reports     : master inventory + figures
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from midas_defect.asterism import (
    asterism_anisotropy_per_grain,
    edge_fraction_per_grain,
    per_grain_asterism_tensor,
)
from midas_defect.debye_waller import per_grain_B_factor
from midas_defect.distributions import (
    friedel_pair_asymmetry,
    jensen_shannon_divergence,
    kl_divergence,
    mackenzie_pdf,
)
from midas_defect.energy import (
    elastic_energy_density_cubic,
    energy_balance_closure,
    volume_weighted_energy_per_variant,
)
from midas_defect.gnd import (
    per_grain_nye_tensor,
    scalar_gnd_from_inter_grain_misorientation,
    ssd_gnd_decomposition,
)
from midas_defect.line_profile import (
    collect_per_grain_reflections,
    modified_wh_per_grain,
)
from midas_defect.phases import FCC_SLIP_111_110, FCC_TWIN_111_112
from midas_defect.reports import (
    cover_figure,
    length_scale_hierarchy,
    load_master_inventory_csv,
    matrix_twin_summary_figure,
    schmid_mechanism_figure,
    write_master_inventory_csv,
)
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
    mk_evolve,
    taylor_implied_total_rho,
    variant_specific_k2,
    wh_visible_fraction,
)
from midas_defect.types import AnalysisResult, BootUnit, CrystalPhase
from midas_defect.variants import (
    assign_variants_kmeans,
    find_sigma3_partners,
)


# Cu single-crystal cubic elastic constants (GPa)
CU_C11, CU_C12, CU_C44 = 169.0, 122.0, 75.3


# --------------------------------------------------------------------------- #
# 1. Variants                                                                  #
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def variants_out(synthetic_cu_al_dataset):
    out = assign_variants_kmeans(
        synthetic_cu_al_dataset["OM"], n_variants=2, n_init=10, random_state=0,
        phase=CrystalPhase.FCC,
    )
    return out


def test_variants_recovers_two_clusters(variants_out, synthetic_cu_al_dataset):
    labels = variants_out["labels"]
    truth = synthetic_cu_al_dataset["true_variant"]
    acc = max((labels == truth).mean(), (labels == 1 - truth).mean())
    assert acc > 0.85
    # K=2 clusters: between-cluster disorientation should be near 60 deg (Sigma3).
    assert 50.0 < float(variants_out["disorientations"][0, 1]) < 65.0


# --------------------------------------------------------------------------- #
# 2. Matched pairs                                                             #
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def pairs_out(synthetic_cu_al_dataset, variants_out):
    # Use the recovered labels but with the convention that "matrix" is the
    # majority label and "twin" is the minority.
    labels = variants_out["labels"]
    counts = np.bincount(labels, minlength=2)
    matrix_lbl = int(np.argmax(counts))
    twin_lbl = 1 - matrix_lbl
    return find_sigma3_partners(
        synthetic_cu_al_dataset["OM"],
        synthetic_cu_al_dataset["pos"],
        labels,
        k_NN=5,
        misori_low=50.0,
        misori_high=70.0,
        axis_alignment_min=0.85,
        phase=CrystalPhase.FCC,
        matrix_label=matrix_lbl,
        twin_label=twin_lbl,
    )


def test_matched_pairs_found(pairs_out):
    n_pairs = pairs_out["pairs"].shape[0]
    assert n_pairs > 0
    # Misorientations should cluster near 60 deg.
    assert 55.0 < np.median(pairs_out["pair_misori"]) < 65.0


# --------------------------------------------------------------------------- #
# 3. Schmid + stratification                                                   #
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def schmid_out(synthetic_cu_al_dataset):
    s, active = schmid_factor_per_grain(
        synthetic_cu_al_dataset["OM"],
        synthetic_cu_al_dataset["loading_axis"],
        FCC_SLIP_111_110,
        return_active_system=True,
    )
    return s, active


def test_schmid_factors_in_textbook_range(schmid_out):
    s, _ = schmid_out
    assert (s >= 0).all() and (s <= 0.5 + 1e-9).all()
    # Population mean Schmid for random texture should be ~ 0.4.
    assert 0.3 < float(np.mean(s)) < 0.5


def test_schmid_stratify_returns_three_tiers(schmid_out, pairs_out):
    s, _ = schmid_out
    if pairs_out["pairs"].shape[0] < 5:
        pytest.skip("not enough matched pairs to stratify")
    out = stratify_pairs_by_schmid_max(pairs_out["pairs"], s)
    assert len(out["tier_pair_indices"]) == 3
    # Tier edges monotonically increasing
    assert (out["tier_edges"][:-1] <= out["tier_edges"][1:]).all()


# --------------------------------------------------------------------------- #
# 4. Strain                                                                    #
# --------------------------------------------------------------------------- #

def test_per_pair_twin_shear_projection_finite(synthetic_cu_al_dataset, pairs_out):
    if pairs_out["pairs"].shape[0] < 3:
        pytest.skip("not enough pairs")
    out = twin_shear_projection_per_pair(
        synthetic_cu_al_dataset["eps_sample"],
        synthetic_cu_al_dataset["OM"],
        pairs_out["pairs"],
        FCC_TWIN_111_112,
        synthetic_cu_al_dataset["loading_axis"],
    )
    assert np.isfinite(out["dEps_twin_shear"]).all()
    assert np.isfinite(out["dEps_orthogonal"]).all()


def test_per_grain_eigenvalues_lode_in_range(synthetic_cu_al_dataset):
    out = per_grain_eigenvalues(synthetic_cu_al_dataset["eps_sample"])
    lode = out["lode_parameter"]
    finite = np.isfinite(lode)
    assert finite.any()
    # Lode mu in [-1, +1]
    assert ((lode[finite] >= -1.0 - 1e-9) & (lode[finite] <= 1.0 + 1e-9)).all()


def test_population_vm_strain_around_planted_scale(synthetic_cu_al_dataset):
    eps_eq = von_mises_strain(synthetic_cu_al_dataset["eps_sample"])
    # Planted strain scale ~ 1e-3; equivalent strain of same order.
    assert 1e-4 < float(np.median(eps_eq)) < 1e-2


# --------------------------------------------------------------------------- #
# 5. Stress                                                                    #
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def stress_out(synthetic_cu_al_dataset):
    sigma = per_grain_stress_cubic(
        synthetic_cu_al_dataset["OM"],
        synthetic_cu_al_dataset["eps_sample"],
        CU_C11, CU_C12, CU_C44,
    )
    sigma_vM = von_mises(sigma)
    return sigma, sigma_vM


def test_per_grain_stress_in_MPa_GPa_range(stress_out):
    sigma, sigma_vM = stress_out
    # 1e-3 strain * 100 GPa elastic stiffness ~ 100 MPa stress order.
    assert 1e7 < float(np.median(sigma_vM)) < 5e9


# --------------------------------------------------------------------------- #
# 6. Energy                                                                    #
# --------------------------------------------------------------------------- #

def test_energy_partition_matrix_higher_than_twin(synthetic_cu_al_dataset, variants_out):
    U = elastic_energy_density_cubic(
        synthetic_cu_al_dataset["OM"],
        synthetic_cu_al_dataset["eps_sample"],
        CU_C11, CU_C12, CU_C44,
    )
    radii = synthetic_cu_al_dataset["radii"]
    truth = synthetic_cu_al_dataset["true_variant"]
    out = volume_weighted_energy_per_variant(U, radii, truth)
    # Plant ratio: matrix strain scale 1.2e-3 vs twin 0.8e-3 -> U_matrix > U_twin.
    assert out["U_mean_per_variant"][0] > out["U_mean_per_variant"][1]
    # Energy-balance closure: just check the call returns finite ratio.
    closure = energy_balance_closure(
        out["U_mean_per_variant"][0],
        out["U_mean_per_variant"][1],
        gamma_TB=0.04,    # Cu TB energy J/m^2
        L_lamella=1.0e-7,  # 100 nm
    )
    assert np.isfinite(closure["closure_ratio"])


# --------------------------------------------------------------------------- #
# 7. Line profile                                                              #
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def per_grain_refl_entries(synthetic_cu_al_dataset):
    return collect_per_grain_reflections(
        synthetic_cu_al_dataset["qs"],
        synthetic_cu_al_dataset["vals"],
        synthetic_cu_al_dataset["grain_of_voxel"],
        synthetic_cu_al_dataset["OM"],
        synthetic_cu_al_dataset["G_arr"],
        query_radius=0.20,
        min_voxels_per_refl=8,
    )


def test_per_grain_reflections_collected_for_all_grains(per_grain_refl_entries, synthetic_cu_al_dataset):
    assert len(per_grain_refl_entries) == synthetic_cu_al_dataset["OM"].shape[0]
    # Most grains should have at least 4 reflections.
    n_refl = np.array([e["refl_indices"].size for e in per_grain_refl_entries])
    assert (n_refl >= 4).mean() > 0.8


def test_modified_wh_recovers_per_grain_density(per_grain_refl_entries, synthetic_cu_al_dataset):
    out = modified_wh_per_grain(
        per_grain_refl_entries,
        synthetic_cu_al_dataset["hkls"],
        burgers_length=synthetic_cu_al_dataset["burgers"],
    )
    rho = out["rho_per_grain"]
    finite = np.isfinite(rho)
    assert finite.mean() > 0.8
    # Plant: twin grains have ~2x the per-G FWHM scale -> higher rho than matrix.
    truth = synthetic_cu_al_dataset["true_variant"]
    rho_m = rho[(truth == 0) & finite]
    rho_t = rho[(truth == 1) & finite]
    assert np.nanmedian(rho_t) > np.nanmedian(rho_m)


# --------------------------------------------------------------------------- #
# 8. Debye-Waller                                                              #
# --------------------------------------------------------------------------- #

def test_per_grain_B_finite_and_positive(per_grain_refl_entries, synthetic_cu_al_dataset):
    out = per_grain_B_factor(
        per_grain_refl_entries,
        synthetic_cu_al_dataset["hkls"],
        structure_factor_squared=lambda hkl: 1.0,  # equal F^2 placeholder
    )
    B = out["B_per_grain"]
    # Some grains may fit; require *some* finite values; don't constrain sign
    # because synthetic plant lacks a real DW falloff.
    assert np.isfinite(B).any()


# --------------------------------------------------------------------------- #
# 9. Asterism                                                                  #
# --------------------------------------------------------------------------- #

def test_per_grain_asterism_tensor_and_anisotropy_finite(synthetic_cu_al_dataset):
    qs = synthetic_cu_al_dataset["qs"]
    vals = synthetic_cu_al_dataset["vals"]
    g_of_v = synthetic_cu_al_dataset["grain_of_voxel"]
    OM = synthetic_cu_al_dataset["OM"]
    G_arr = synthetic_cu_al_dataset["G_arr"]

    # Build P_all_nearest = predicted Bragg per voxel (the nearest sample-frame
    # target across all hkls for that voxel's grain).
    n_voxels = qs.shape[0]
    Pn = np.zeros_like(qs)
    for v in range(n_voxels):
        g = g_of_v[v]
        targets = (OM[g] @ G_arr.T).T
        d = np.linalg.norm(targets - qs[v], axis=1)
        Pn[v] = targets[int(np.argmin(d))]

    mask = np.ones(n_voxels, dtype=bool)
    M = per_grain_asterism_tensor(
        qs, vals, g_of_v, Pn, mask, n_grains=OM.shape[0], min_voxels_per_grain=20
    )
    finite_M = np.array([np.isfinite(M[i]).all() for i in range(M.shape[0])])
    assert finite_M.mean() > 0.5

    f = edge_fraction_per_grain(M, qs, vals, g_of_v, mask, n_grains=OM.shape[0])
    finite_f = np.isfinite(f)
    assert finite_f.any()
    assert ((f[finite_f] >= 0) & (f[finite_f] <= 1)).all()

    aniso = asterism_anisotropy_per_grain(M)
    finite_a = np.isfinite(aniso["anisotropy_max_min"])
    assert finite_a.any()
    # Anisotropy must be >= 1 by definition (largest / smallest).
    assert (aniso["anisotropy_max_min"][finite_a] >= 1.0 - 1e-9).all()


# --------------------------------------------------------------------------- #
# 10. GND                                                                      #
# --------------------------------------------------------------------------- #

def test_scalar_gnd_finite_and_positive(synthetic_cu_al_dataset):
    rho_GND = scalar_gnd_from_inter_grain_misorientation(
        synthetic_cu_al_dataset["OM"],
        synthetic_cu_al_dataset["pos"],
        burgers_length=synthetic_cu_al_dataset["burgers"],
        k_NN=4,
    )
    finite = np.isfinite(rho_GND)
    assert finite.mean() > 0.5
    assert (rho_GND[finite] >= 0).all()


def test_nye_tensor_and_ssd_decomposition(synthetic_cu_al_dataset, per_grain_refl_entries):
    nye = per_grain_nye_tensor(
        synthetic_cu_al_dataset["OM"],
        synthetic_cu_al_dataset["pos"],
        burgers_length=synthetic_cu_al_dataset["burgers"],
        k_NN=8,
    )
    rho_GND = nye["rho_GND_per_grain"]
    out_wh = modified_wh_per_grain(
        per_grain_refl_entries,
        synthetic_cu_al_dataset["hkls"],
        burgers_length=synthetic_cu_al_dataset["burgers"],
    )
    rho_total = out_wh["rho_per_grain"]
    split = ssd_gnd_decomposition(rho_total, rho_GND)
    finite = np.isfinite(split["rho_SSD_per_grain"])
    assert finite.any()
    assert (split["rho_SSD_per_grain"][finite] >= 0).all()


# --------------------------------------------------------------------------- #
# 11. Distributions                                                            #
# --------------------------------------------------------------------------- #

def test_mackenzie_kl_against_uniform_far_from_zero():
    centers = np.linspace(0.5, 62.5, 124)
    observed = mackenzie_pdf(centers, phase=CrystalPhase.FCC)
    reference = np.ones_like(observed)
    d_kl = kl_divergence(observed, reference, centers)
    d_js = jensen_shannon_divergence(observed, reference, centers)
    assert d_kl > 0
    assert 0 <= d_js <= np.log(2.0)


def test_friedel_asymmetry_on_synthetic_population():
    intensities = {(0, (1, 1, 1)): 100.0, (0, (-1, -1, -1)): 80.0,
                   (1, (2, 0, 0)): 50.0, (1, (-2, 0, 0)): 55.0}
    out = friedel_pair_asymmetry(intensities)
    assert out["asymmetry_per_pair"].size == 2
    assert 0 < out["mean_asymmetry"] < 1


# --------------------------------------------------------------------------- #
# 12. Spatial                                                                  #
# --------------------------------------------------------------------------- #

def test_spatial_autocorrelation_and_stress_gradient(synthetic_cu_al_dataset, stress_out):
    _, sigma_vM = stress_out
    eps_eq = von_mises_strain(synthetic_cu_al_dataset["eps_sample"])
    out = epsilon_autocorrelation(eps_eq, synthetic_cu_al_dataset["pos"])
    finite = np.isfinite(out["pearson_r_per_bin"])
    assert finite.any()
    grad = stress_spatial_gradient_per_grain(sigma_vM, synthetic_cu_al_dataset["pos"], k_NN=4)
    assert np.isfinite(grad).all()
    assert (grad >= 0).all()


# --------------------------------------------------------------------------- #
# 13. Thermodynamics                                                           #
# --------------------------------------------------------------------------- #

def test_thermodynamics_chain(per_grain_refl_entries, synthetic_cu_al_dataset):
    out_wh = modified_wh_per_grain(
        per_grain_refl_entries,
        synthetic_cu_al_dataset["hkls"],
        burgers_length=synthetic_cu_al_dataset["burgers"],
    )
    rho = out_wh["rho_per_grain"]
    # Saturated rho per variant (use 84th-pctile as a stand-in for sat).
    truth = synthetic_cu_al_dataset["true_variant"]
    finite = np.isfinite(rho)
    rho_sat = {
        "matrix": float(np.nanpercentile(rho[(truth == 0) & finite], 84)),
        "twin": float(np.nanpercentile(rho[(truth == 1) & finite], 84)),
    }
    mk = variant_specific_k2(rho_sat)
    assert mk["k2_ratio_pairs"][("matrix", "twin")] > 0
    # MK forward integration -> finite sequence
    eps_traj = np.linspace(0.0, 1.0, 20)
    rho_t = mk_evolve(eps_traj, k1=1e8, k2=5.0, rho_init=1e10)
    assert np.isfinite(rho_t).all()
    # Taylor inversion + visible fraction
    rho_taylor = taylor_implied_total_rho(7.0e8)
    visible = wh_visible_fraction(float(np.nanmedian(rho[finite])), 7.0e8)
    assert 0 < visible < 1
    assert rho_taylor > 0


# --------------------------------------------------------------------------- #
# 14. Reports: master inventory + figures                                      #
# --------------------------------------------------------------------------- #

def test_master_inventory_roundtrip_for_synthetic_pipeline(tmp_path, synthetic_cu_al_dataset, stress_out):
    _, sigma_vM = stress_out
    eps_eq = von_mises_strain(synthetic_cu_al_dataset["eps_sample"])
    rng = np.random.default_rng(0)
    results: list[AnalysisResult] = []
    for name, vals, units in [
        ("eps_eq_matrix", eps_eq[synthetic_cu_al_dataset["true_variant"] == 0], "strain"),
        ("eps_eq_twin", eps_eq[synthetic_cu_al_dataset["true_variant"] == 1], "strain"),
        ("sigma_vM_matrix", sigma_vM[synthetic_cu_al_dataset["true_variant"] == 0], "Pa"),
        ("sigma_vM_twin", sigma_vM[synthetic_cu_al_dataset["true_variant"] == 1], "Pa"),
    ]:
        boot = np.array([np.median(rng.choice(vals, size=vals.size, replace=True)) for _ in range(64)])
        results.append(
            AnalysisResult(
                name=name, units=units, boot_unit=BootUnit.GRAIN,
                n_boot=64,
                population_median=float(np.median(boot)),
                population_ci=(float(np.percentile(boot, 16)), float(np.percentile(boot, 84))),
                bootstrap_samples=boot,
                per_grain=vals,
            )
        )
    out_csv = tmp_path / "inventory.csv"
    write_master_inventory_csv(results, out_csv)
    loaded = load_master_inventory_csv(out_csv)
    assert len(loaded) == len(results)
    for r_in, r_out in zip(results, loaded):
        assert r_in.name == r_out.name
        np.testing.assert_allclose(r_in.bootstrap_samples, r_out.bootstrap_samples)


def test_full_figure_set_renders(tmp_path, synthetic_cu_al_dataset, stress_out, pairs_out, schmid_out):
    _, sigma_vM = stress_out
    eps_eq = von_mises_strain(synthetic_cu_al_dataset["eps_sample"])
    rng = np.random.default_rng(0)
    truth = synthetic_cu_al_dataset["true_variant"]
    results: list[AnalysisResult] = []
    for name, vals, units in [
        ("eps_eq_matrix", eps_eq[truth == 0], "strain"),
        ("eps_eq_twin", eps_eq[truth == 1], "strain"),
        ("sigma_vM_matrix", sigma_vM[truth == 0], "Pa"),
        ("sigma_vM_twin", sigma_vM[truth == 1], "Pa"),
    ]:
        boot = np.array([np.median(rng.choice(vals, size=vals.size, replace=True)) for _ in range(64)])
        results.append(
            AnalysisResult(
                name=name, units=units, boot_unit=BootUnit.GRAIN, n_boot=64,
                population_median=float(np.median(boot)),
                population_ci=(float(np.percentile(boot, 16)), float(np.percentile(boot, 84))),
                bootstrap_samples=boot,
                per_grain=vals,
            )
        )

    p_summary = tmp_path / "summary.png"
    matrix_twin_summary_figure(results, p_summary)
    assert p_summary.exists()

    p_cover = tmp_path / "cover.png"
    cover_figure(results, p_cover, headline_metrics=[r.name for r in results])
    assert p_cover.exists()

    if pairs_out["pairs"].shape[0] >= 5:
        s, _ = schmid_out
        s_pair = np.maximum(s[pairs_out["pairs"][:, 0]], s[pairs_out["pairs"][:, 1]])
        dEps = np.zeros_like(s_pair)
        tier_edges = np.percentile(s_pair, [0, 33, 67, 100])
        p_mech = tmp_path / "mech.png"
        schmid_mechanism_figure(s_pair, dEps, tier_edges, p_mech)
        assert p_mech.exists()

    p_len = tmp_path / "hierarchy.png"
    length_scale_hierarchy(
        {"coh D": (200.0, 150.0, 280.0), "lamella": (30.0, 25.0, 40.0)}, p_len
    )
    assert p_len.exists()
